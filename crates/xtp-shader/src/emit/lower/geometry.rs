//! Merged ES + GS programs: the console's export shader (ES) as a D3D12
//! vertex shader, its geometry shader (GS) as a D3D12 geometry shader, and
//! the two LDS rings between and after them as signatures. Backlog G1 and
//! G2, task W10; the design, the title's programs and the D3D12 choice are
//! in `docs/analysis/es-gs.md`.
//!
//! # The console's form
//!
//! GFX10 runs both halves as one hardware stage (the legacy GS, its rings
//! on chip), but each half is a program of its own in the pack: the ES in
//! the vertex shader's container, the GS in the geometry shader's.
//!
//! * The ES runs one lane per vertex and writes the vertex's outputs to the
//!   ESGS ring in LDS, one item per vertex at `lane × item` after its
//!   wave's base (`v_mbcnt` of every lane, the wave index of s3). Item dword
//!   `4 r + c` is the output in row `r`, column `c` of the ES's output
//!   signature. It then waits at `s_barrier` and jumps to the GS half
//!   (`s_setpc_b64`, an [`Exit::Indirect`](crate::ir::Exit)).
//! * The GS runs one lane per primitive. The hardware passes it the ESGS
//!   ring offsets of its input vertices, in dwords, in 16-bit halves of
//!   VGPRs ([`VERTEX_OFFSETS`]); dword `d` of input vertex `k` is at
//!   `4 × offset_k + 4 d`.
//! * The GS writes each vertex it makes to the GSVS ring, also in LDS:
//!   vertex `n` at `lane × item + n × vertex + 4 (4 r + c)` after a base the
//!   program computes, `r` and `c` the row and column of the GS's output
//!   signature. `s_sendmsg` GS EMIT says a vertex is complete, GS CUT ends
//!   the strip, GS_DONE ends the program; the console's copy shader (made
//!   by its driver, not in the pack) then reads the ring.
//!
//! # The host's form
//!
//! The input assembler feeds a D3D12 vertex shader per vertex and a
//! geometry shader per primitive: no ring at all. [`es_plan`] and
//! [`gs_plan`] resolve every ring access of a program to its slot before the
//! lowering, from the address arithmetic, which the lowering then leaves out
//! ([`EsPlan::address_only`], [`GsPlan::address_only`]): the GS's ring base
//! is read from a table of the console's driver, which the host does not
//! have.
//!
//! * An ES write of item dword `d` stores the vertex output at row `d / 4`,
//!   column `d % 4` ([`write_item`]).
//! * A GS read of dword `d` of input vertex `k` is `loadInput` of that row
//!   and column of vertex `k` ([`read_vertex`]).
//! * A GS write to dword `d` of vertex `n` is kept until vertex `n` is
//!   emitted, which stores the vertex's outputs then calls `emitStream`
//!   ([`Vertices`]); a cut is `cutStream`. GS_DONE alone is nothing; it may
//!   also carry a last emit or cut, which is lowered as one.
//!
//! The vertex an emit completes is the number of emits before it, which
//! must be the same on every path; and every emit and cut must run under
//! the same lane mask, since the console counts a lane's vertices by the
//! emits it is in EXEC for. No write may come after its vertex's emit, and
//! on the paths to an emit each dword of its vertex may be written once at
//! most, by an instruction on every one of them (so a vertex can be
//! written and emitted on either side of a branch): the copy shader reads
//! the ring once the program is done, D3D12 the outputs at each emit, and
//! these rules make the two the same. A write in a loop must also run for
//! every lane in its block: a lane its EXEC leaves out on a later pass
//! keeps the dword an earlier pass wrote, where the host keeps the last
//! pass's ([`LOOPED`]). Programs that break them are refused, as are emits
//! on other streams than 0: none of the title's programs writes one, so
//! the ring layout of streams 1 to 3 is not known.
//!
//! Only the ring dwords a signature element holds are converted: a write
//! of a whole row (`ds_write2_b64`) into a short last row also moves
//! dwords nothing reads, which may hold values with no host form.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use super::{Lower, Stop, Val, Why, unsupported};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};
use crate::emit::bitcode::{Attr, Cast, Term, V};
use crate::emit::container::SigElement;
use crate::emit::shader::Shader;
use crate::gfx10::{Encoding, Fields, Instruction, NULL, Operand, Valu};
use crate::io::Io;
use crate::ir::{EXEC, Exit, Inst, M0, Program, Reg, Value, idoms, reverse_postorder};
use crate::native::{GeometryIo, HwStage, NativeProgram, StageIo, VsOutput};
use crate::resource::{Site, inline};

/// A ring access the rings do not explain: an address the plans cannot
/// resolve to a slot, or an LDS instruction that is no plain dword read or
/// write.
pub(crate) const RING: Why = "an LDS access the merged ES + GS rings do not explain";
/// An ES ring item larger than a D3D12 vertex's 32 output rows.
pub(crate) const ITEM: Why = "an ES ring item past D3D12's 32 output rows";
/// A GSVS ring item that is not one stream's vertices.
pub(crate) const STREAMS: Why = "a GSVS ring item of several streams";
pub(crate) const STREAM: Why = "a geometry shader emit or cut on a stream other than 0";
pub(crate) const MESSAGE: Why = "a message a geometry shader does not send";
pub(crate) const PATH: Why = "a geometry shader emit whose vertex depends on the path";
pub(crate) const AFTER_DONE: Why = "a geometry shader message after GS_DONE";
pub(crate) const TOO_MANY: Why = "a geometry shader emitting more vertices than its maximum";
/// Emits and cuts under different lane masks: a lane outside one emit's
/// mask counts its vertices from there on one fewer than the plan, which
/// numbers them per path.
pub(crate) const MASKED: Why =
    "a geometry shader emit or cut under another lane mask than its others";
pub(crate) const LATE: Why = "a geometry shader output written after its vertex is emitted";
pub(crate) const TWICE: Why =
    "a geometry shader output written twice on the paths to its vertex's emit";
pub(crate) const SOME_PATHS: Why =
    "a geometry shader output written on some paths to its vertex's emit only";
/// A GSVS write in a loop under an EXEC narrower than its block's lanes
/// (Codex review 3983f844): a lane left out on a later pass keeps in the
/// ring what an earlier pass wrote, the host the last pass's value.
pub(crate) const LOOPED: Why =
    "a geometry shader output written in a loop under a narrower lane mask";
pub(crate) const NOT_INPUT: Why =
    "a geometry shader reading a ring dword its input signature does not hold";
pub(crate) const VERTICES: Why = "a geometry shader beyond D3D12's 1024 output vertices";
pub(crate) const SCALARS: Why =
    "a geometry shader whose vertices exceed D3D12's 1024 output scalars";
pub(crate) const ROWS: Why = "a geometry shader signature past D3D12's 32 rows";
pub(crate) const INSTANCES: Why = "a geometry shader beyond D3D12's 32 instances";
pub(crate) const BOOLEAN: Why = "a boolean geometry shader input or output";
/// A merged half that does not start by setting EXEC to its lanes from s3
/// as the title's do ([`drop_lane_prologue`]).
pub(crate) const PROLOGUE: Why =
    "a merged ES + GS program whose EXEC set-up from s3 the translation does not know";
pub(crate) const PSV: Why = "a geometry shader whose PSV0 the translation does not read";
pub(crate) const PRIMITIVE: Why =
    "a geometry shader input primitive other than points, lines or triangles";
pub(crate) const TOPOLOGY: Why =
    "a geometry shader output topology other than points, line strips or triangle strips";
pub(crate) const MOST_VERTICES: Why =
    "a geometry shader whose stage data and PSV0 disagree on its most vertices";
pub(crate) const COUNT: Why =
    "a geometry shader whose stage data instance or stream count is not 1";

/// D3D12's geometry-shader limits: registers per input vertex and per output
/// vertex (`D3D12_GS_INPUT_REGISTER_COUNT`, `D3D12_GS_OUTPUT_REGISTER_COUNT`),
/// vertices across instances
/// (`D3D12_GS_MAX_OUTPUT_VERTEX_COUNT_ACROSS_INSTANCES`), 32-bit scalars one
/// invocation outputs (its most vertices times each vertex's scalars) and
/// instances (`D3D12_GS_MAX_INSTANCE_COUNT`).
pub(crate) const GS_REGISTERS: u32 = 32;
pub(crate) const GS_MAX_VERTICES: u32 = 1024;
pub(crate) const GS_MAX_SCALARS: u32 = 1024;
pub(crate) const GS_MAX_INSTANCES: u32 = 32;

/// Where the hardware passes a GS lane the ESGS ring offsets of its input
/// vertices, in dwords: the VGPR, then the first bit of its 16-bit half.
/// Vertices 0 to 2 are as the title's programs read them; 4 and 5 (the
/// adjacent vertices, in v4) are the GFX9 merged layout, not seen in use.
pub(crate) const VERTEX_OFFSETS: [(u8, u8); 6] =
    [(0, 0), (0, 16), (1, 0), (1, 16), (4, 0), (4, 16)];

/// `s_sendmsg` message ids (bits 0-3) of the GS messages.
const MSG_GS: u16 = 2;
const MSG_GS_DONE: u16 = 3;

/// What a ring address is made of: the program's lane, or a dword the
/// hardware sets or a scalar load reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Base {
    /// The lane in the wave: `v_mbcnt_lo` then `v_mbcnt_hi` of every lane.
    Lane,
    /// `v_mbcnt_lo_u32_b32` of every lane in a wave64 program: the lanes
    /// 0-31 below this one (0 to 32).
    LaneLo,
    /// `v_mbcnt_hi_u32_b32` of every lane: the lanes 32-63 below this one.
    LaneHi,
    /// A register on entry.
    Input(Value),
    /// A dword a scalar load reads: the same in every lane.
    Loaded(Value),
}

/// Bits `lo .. lo + width` of a base, as a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Field {
    base: Base,
    lo: u8,
    width: u8,
}

/// The ones in the low `width` bits.
fn ones(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1 << width) - 1
    }
}

/// A dword as `constant + Σ coefficient × field`, exact modulo 2^32.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Form {
    constant: u32,
    terms: BTreeMap<Field, u32>,
}

impl Form {
    fn constant(constant: u32) -> Form {
        Form {
            constant,
            terms: BTreeMap::new(),
        }
    }

    fn field(base: Base, width: u8) -> Form {
        let field = Field { base, lo: 0, width };
        Form {
            constant: 0,
            terms: BTreeMap::from([(field, 1)]),
        }
    }

    fn as_constant(&self) -> Option<u32> {
        self.terms.is_empty().then_some(self.constant)
    }

    fn plus(mut self, other: &Form) -> Form {
        self.constant = self.constant.wrapping_add(other.constant);
        for (&f, &c) in &other.terms {
            let e = self.terms.entry(f).or_insert(0);
            *e = e.wrapping_add(c);
        }
        self.terms.retain(|_, c| *c != 0);
        // The two halves of `v_mbcnt` add up to the lane.
        let lo = Field {
            base: Base::LaneLo,
            lo: 0,
            width: 6,
        };
        let hi = Field {
            base: Base::LaneHi,
            lo: 0,
            width: 5,
        };
        if let (Some(&a), Some(&b)) = (self.terms.get(&lo), self.terms.get(&hi))
            && a == b
        {
            self.terms.remove(&lo);
            self.terms.remove(&hi);
            let lane = Field {
                base: Base::Lane,
                lo: 0,
                width: 6,
            };
            let e = self.terms.entry(lane).or_insert(0);
            *e = e.wrapping_add(a);
            self.terms.retain(|_, c| *c != 0);
        }
        self
    }

    fn scale(mut self, k: u32) -> Form {
        self.constant = self.constant.wrapping_mul(k);
        for c in self.terms.values_mut() {
            *c = c.wrapping_mul(k);
        }
        self.terms.retain(|_, c| *c != 0);
        self
    }

    fn minus(self, other: &Form) -> Form {
        self.plus(&other.clone().scale(u32::MAX))
    }

    fn lshl(self, k: u32) -> Form {
        self.scale(1 << (k & 31))
    }

    /// The form as an OR of shifted fields and the constant, when it is
    /// one: every coefficient a power of two, no two parts sharing a bit.
    /// Fields come cut to the bits a dword keeps.
    fn parts(&self) -> Option<Vec<(Field, u32)>> {
        let mut used = self.constant;
        let mut out = Vec::with_capacity(self.terms.len());
        for (&f, &c) in &self.terms {
            if !c.is_power_of_two() {
                return None;
            }
            let shift = c.trailing_zeros();
            let width = u32::from(f.width).min(32 - shift);
            let span = ones(width) << shift;
            if used & span != 0 {
                return None;
            }
            used |= span;
            out.push((
                Field {
                    width: width as u8,
                    ..f
                },
                shift,
            ));
        }
        Some(out)
    }

    fn from_parts(constant: u32, parts: Vec<(Field, u32)>) -> Form {
        Form {
            constant,
            terms: parts.into_iter().map(|(f, s)| (f, 1 << s)).collect(),
        }
    }

    /// `self & mask`, when each part keeps one run of its bits.
    fn and(&self, mask: u32) -> Option<Form> {
        let mut kept = Vec::new();
        for (f, shift) in self.parts()? {
            let bits = ones(u32::from(f.width)) << shift & mask;
            if bits == 0 {
                continue;
            }
            let (a, b) = (bits.trailing_zeros(), 32 - bits.leading_zeros());
            if bits != ones(b - a) << a {
                return None;
            }
            let field = Field {
                base: f.base,
                lo: f.lo + (a - shift) as u8,
                width: (b - a) as u8,
            };
            kept.push((field, a));
        }
        Some(Form::from_parts(self.constant & mask, kept))
    }

    /// `self >> k` (logical, `k` below 32).
    fn lshr(&self, k: u32) -> Option<Form> {
        let mut kept = Vec::new();
        for (f, shift) in self.parts()? {
            let end = shift + u32::from(f.width);
            if end <= k {
                continue;
            }
            if shift >= k {
                kept.push((f, shift - k));
            } else {
                let field = Field {
                    base: f.base,
                    lo: f.lo + (k - shift) as u8,
                    width: (end - k) as u8,
                };
                kept.push((field, 0));
            }
        }
        Some(Form::from_parts(self.constant >> k, kept))
    }

    /// Bits `lo .. lo + width` of the dword, as a number.
    fn bits(&self, lo: u32, width: u32) -> Option<Form> {
        self.lshr(lo)?.and(ones(width))
    }
}

fn mul(a: &Form, b: &Form) -> Option<Form> {
    match (a.as_constant(), b.as_constant()) {
        (_, Some(k)) => Some(a.clone().scale(k)),
        (Some(k), _) => Some(b.clone().scale(k)),
        _ => None,
    }
}

fn and(a: &Form, b: &Form) -> Option<Form> {
    match (a.as_constant(), b.as_constant()) {
        (_, Some(k)) => a.and(k),
        (Some(k), _) => b.and(k),
        _ => None,
    }
}

/// What an LDS address is, by the plans' rules.
enum Where {
    /// Input vertex `vertex`'s item, from `offset` bytes on.
    Vertex { vertex: u8, offset: u32 },
    /// This lane's item of `stride` bytes after the same base for every
    /// lane (`rest`, then `offset` bytes).
    Lane {
        stride: u32,
        rest: Form,
        offset: u32,
    },
}

/// The forms of a program's values, as far as ring addresses need them.
struct Forms<'p> {
    p: &'p Program,
    defs: HashMap<Value, Site>,
    inputs: HashMap<Value, Reg>,
    memo: HashMap<Value, Option<Form>>,
}

impl<'p> Forms<'p> {
    fn new(p: &'p Program) -> Self {
        let mut defs = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs.insert(v, (b, i));
                }
            }
        }
        Forms {
            p,
            defs,
            inputs: p.inputs.iter().map(|&(r, v)| (v, r)).collect(),
            memo: HashMap::new(),
        }
    }

    /// The form of `v`; `None` for a value no rule covers (a phi, a vector
    /// load, a float operation...).
    fn value(&mut self, v: Value) -> Option<Form> {
        if let Some(f) = self.memo.get(&v) {
            return f.clone();
        }
        let f = if self.inputs.contains_key(&v) {
            Some(Form::field(Base::Input(v), 32))
        } else if let Some(&site) = self.defs.get(&v) {
            self.defined(site, v)
        } else {
            None
        };
        self.memo.insert(v, f.clone());
        f
    }

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Option<Form> {
        match *op {
            Operand::Const(code) => inline(code).map(Form::constant),
            Operand::Literal(x) => Some(Form::constant(x)),
            Operand::Scalar { code: NULL, .. } => Some(Form::constant(0)),
            Operand::Scalar { code, dwords: 1 } => self.value(inst.read(Reg::S(code))?),
            Operand::Vector { reg, dwords: 1 } => self.value(inst.read(Reg::V(reg))?),
            _ => None,
        }
    }

    fn defined(&mut self, (b, i): Site, v: Value) -> Option<Form> {
        let p = self.p;
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        match &inst.fields {
            Fields::Smem { .. }
                if name.starts_with("s_load") || name.starts_with("s_buffer_load") =>
            {
                Some(Form::field(Base::Loaded(v), 32))
            }
            Fields::Sop {
                dst: Some(Operand::Scalar { code, dwords: 1 }),
                src,
                imm,
            } if inst.write(Reg::S(*code)) == Some(v) => self.scalar(inst, src, *imm),
            Fields::Valu(valu) => match valu.dst {
                Some(Operand::Vector { reg, dwords: 1 }) if inst.write(Reg::V(reg)) == Some(v) => {
                    self.vector(inst, valu)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn scalar(&mut self, inst: &Inst, src: &[Operand], imm: Option<u16>) -> Option<Form> {
        let name = inst.name();
        if name == "s_movk_i32" {
            return Some(Form::constant(imm? as i16 as i32 as u32));
        }
        let a = self.operand(inst, src.first()?)?;
        if name == "s_mov_b32" {
            return Some(a);
        }
        let b = self.operand(inst, src.get(1)?)?;
        match name {
            "s_add_u32" | "s_add_i32" => Some(a.plus(&b)),
            "s_sub_u32" | "s_sub_i32" => Some(a.minus(&b)),
            "s_mul_i32" => mul(&a, &b),
            "s_and_b32" => and(&a, &b),
            "s_lshl_b32" => Some(a.lshl(b.as_constant()?)),
            "s_lshr_b32" => a.lshr(b.as_constant()? & 31),
            "s_bfe_u32" => {
                let k = b.as_constant()?;
                a.bits(k & 31, k >> 16 & 0x7F)
            }
            _ => None,
        }
    }

    fn vector(&mut self, inst: &Inst, valu: &Valu) -> Option<Form> {
        if valu.neg | valu.abs | valu.neg_hi | valu.opsel != 0
            || valu.clamp
            || valu.omod != 0
            || valu.dpp.is_some()
        {
            return None;
        }
        if valu.sdwa.is_some_and(|s| s.dst_sel != 6 || s.sext != 0) {
            return None;
        }
        let mut src = Vec::with_capacity(valu.src.len());
        for (k, op) in valu.src.iter().enumerate() {
            let f = self.operand(inst, op)?;
            let f = match valu.sdwa {
                Some(s) if k < 2 => match s.src_sel[k] {
                    sel @ 0..=3 => f.bits(8 * u32::from(sel), 8)?,
                    sel @ 4..=5 => f.bits(16 * u32::from(sel - 4), 16)?,
                    _ => f,
                },
                _ => f,
            };
            src.push(f);
        }
        let k = |f: &Form| f.as_constant();
        let get = |n: usize| src.get(n);
        let wave64 = self.p.wave64;
        match inst.name() {
            "v_mov_b32" => get(0).cloned(),
            "v_add_nc_u32" => Some(get(0)?.clone().plus(get(1)?)),
            "v_sub_nc_u32" => Some(get(0)?.clone().minus(get(1)?)),
            "v_subrev_nc_u32" => Some(get(1)?.clone().minus(get(0)?)),
            "v_add3_u32" => Some(get(0)?.clone().plus(get(1)?).plus(get(2)?)),
            "v_lshlrev_b32" => Some(get(1)?.clone().lshl(k(get(0)?)?)),
            "v_lshrrev_b32" => get(1)?.lshr(k(get(0)?)? & 31),
            "v_lshl_add_u32" => Some(get(0)?.clone().lshl(k(get(1)?)?).plus(get(2)?)),
            "v_and_b32" => and(get(0)?, get(1)?),
            "v_mul_lo_u32" => mul(get(0)?, get(1)?),
            "v_mul_u32_u24" => mul(&get(0)?.and(0xFF_FFFF)?, &get(1)?.and(0xFF_FFFF)?),
            "v_mad_u32_u24" => {
                let product = mul(&get(0)?.and(0xFF_FFFF)?, &get(1)?.and(0xFF_FFFF)?)?;
                Some(product.plus(get(2)?))
            }
            "v_bfe_u32" => get(0)?.bits(k(get(1)?)? & 31, k(get(2)?)? & 31),
            "v_mbcnt_lo_u32_b32" if k(get(0)?) == Some(u32::MAX) => {
                let lanes = if wave64 {
                    Form::field(Base::LaneLo, 6)
                } else {
                    Form::field(Base::Lane, 5)
                };
                Some(get(1)?.clone().plus(&lanes))
            }
            "v_mbcnt_hi_u32_b32" if k(get(0)?) == Some(u32::MAX) => {
                let base = get(1)?.clone();
                Some(if wave64 {
                    base.plus(&Form::field(Base::LaneHi, 5))
                } else {
                    base
                })
            }
            _ => None,
        }
    }

    /// Whether `base` is the same in every lane: a scalar register on
    /// entry, or what a scalar load reads.
    fn uniform(&self, base: Base) -> bool {
        match base {
            Base::Input(v) => matches!(self.inputs.get(&v), Some(Reg::S(_))),
            Base::Loaded(_) => true,
            Base::Lane | Base::LaneLo | Base::LaneHi => false,
        }
    }

    /// What the address `f` is: an input vertex's item if it holds one of
    /// `vertices` (four times the dword offset), else this lane's item.
    fn classify(&self, f: &Form, vertices: &[Option<Field>]) -> Option<Where> {
        let vertex = f
            .terms
            .keys()
            .find_map(|t| vertices.iter().position(|v| *v == Some(*t)));
        if let Some(k) = vertex {
            let field = vertices[k]?;
            return (f.terms.len() == 1 && f.terms.get(&field) == Some(&4)).then_some(
                Where::Vertex {
                    vertex: k as u8,
                    offset: f.constant,
                },
            );
        }
        let lane = Field {
            base: Base::Lane,
            lo: 0,
            width: 6,
        };
        let lane = if self.p.wave64 {
            lane
        } else {
            Field { width: 5, ..lane }
        };
        let stride = *f.terms.get(&lane)?;
        let mut rest = f.clone();
        rest.constant = 0;
        rest.terms.remove(&lane);
        if !rest.terms.keys().all(|t| self.uniform(t.base)) {
            return None;
        }
        Some(Where::Lane {
            stride,
            rest,
            offset: f.constant,
        })
    }
}

/// Each dword a plain `ds_*` read or write moves: whether it writes, then
/// the register dword and its byte offset from the address. `None` for
/// other LDS instructions (atomics, bytes and halves, exchanges).
fn ds_dwords(inst: &Inst) -> Option<(bool, Vec<(Reg, u32)>)> {
    let Fields::Ds {
        dst,
        data,
        offset0,
        offset1,
        gds: false,
        ..
    } = &inst.fields
    else {
        return None;
    };
    let name = inst.name();
    let (write, pair_scale) = match name {
        "ds_read_b32" | "ds_read_b64" | "ds_read_b96" | "ds_read_b128" => (false, None),
        "ds_read2_b32" | "ds_read2_b64" => (false, Some(1)),
        "ds_read2st64_b32" | "ds_read2st64_b64" => (false, Some(64)),
        "ds_write_b32" | "ds_write_b64" | "ds_write_b96" | "ds_write_b128" => (true, None),
        "ds_write2_b32" | "ds_write2_b64" => (true, Some(1)),
        "ds_write2st64_b32" | "ds_write2st64_b64" => (true, Some(64)),
        _ => return None,
    };
    let regs = |op: Option<&Operand>| match op {
        Some(&Operand::Vector { reg, dwords }) => Some((reg, dwords)),
        _ => None,
    };
    let element = if name.ends_with("_b64") { 8 } else { 4 };
    let single = u32::from(*offset0) | u32::from(*offset1) << 8;
    let offsets = [*offset0, *offset1].map(|o| u32::from(o) * element * pair_scale.unwrap_or(0));
    let mut out = Vec::new();
    let mut push = |reg: u8, n: u8, at: u32| {
        for i in 0..n {
            out.push((Reg::V(reg + i), at + 4 * u32::from(i)));
        }
    };
    match (write, pair_scale) {
        (true, None) => {
            let (reg, n) = regs(data.first())?;
            push(reg, n, single);
        }
        (true, Some(_)) => {
            for (k, at) in offsets.into_iter().enumerate() {
                let (reg, n) = regs(data.get(k))?;
                push(reg, n, at);
            }
        }
        (false, None) => {
            let (reg, n) = regs(dst.as_ref())?;
            push(reg, n, single);
        }
        (false, Some(_)) => {
            let (reg, n) = regs(dst.as_ref())?;
            let half = n / 2;
            for (k, at) in offsets.into_iter().enumerate() {
                push(reg + half * k as u8, half, at);
            }
        }
    }
    Some((write, out))
}

/// The value a `ds_*` instruction reads as its address.
fn ds_address(inst: &Inst) -> Option<Value> {
    match inst.fields {
        Fields::Ds {
            addr: Some(Operand::Vector { reg, dwords: 1 }),
            ..
        } => inst.read(Reg::V(reg)),
        _ => None,
    }
}

/// A ring dword: of an ES vertex's item (`vertex` 0, its own; for a GS
/// read the input vertex) or of a vertex the GS makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Slot {
    pub vertex: u32,
    pub dword: u32,
}

/// One ring access: each register dword it reads or writes, and its slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RingAccess {
    pub site: Site,
    pub dwords: Vec<(Reg, Slot)>,
}

/// An export shader's ring writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EsPlan {
    /// One vertex's ring item, in bytes (each lane's stride in the ring).
    pub item_bytes: u32,
    pub writes: Vec<RingAccess>,
    /// The instructions whose every result only computes ring addresses:
    /// the lowering leaves them out.
    pub address_only: BTreeSet<Site>,
}

/// Resolves the ESGS ring writes of the export shader `p`: each dword it
/// writes and the item dword it goes to.
pub(crate) fn es_plan(p: &Program) -> Result<EsPlan, Why> {
    let mut forms = Forms::new(p);
    let mut writes = Vec::new();
    let mut layout: Option<(u32, Form)> = None;
    let mut uses = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if !matches!(inst.fields, Fields::Ds { .. }) {
                continue;
            }
            let (true, dwords) = ds_dwords(inst).ok_or(RING)? else {
                return Err(RING);
            };
            let address = ds_address(inst).ok_or(RING)?;
            let form = forms.value(address).ok_or(RING)?;
            let Some(Where::Lane {
                stride,
                rest,
                offset,
            }) = forms.classify(&form, &[])
            else {
                return Err(RING);
            };
            match &layout {
                None => layout = Some((stride, rest)),
                Some((s, r)) if *s == stride && *r == rest => {}
                Some(_) => return Err(RING),
            }
            let mut out = Vec::with_capacity(dwords.len());
            for (reg, at) in dwords {
                let byte = offset.wrapping_add(at);
                if !byte.is_multiple_of(4) || byte >= stride {
                    return Err(RING);
                }
                if byte / 4 >= 4 * GS_REGISTERS {
                    return Err(ITEM);
                }
                out.push((
                    reg,
                    Slot {
                        vertex: 0,
                        dword: byte / 4,
                    },
                ));
            }
            uses.push(((b, i), address));
            writes.push(RingAccess {
                site: (b, i),
                dwords: out,
            });
        }
    }
    let item_bytes = layout.map_or(0, |(s, _)| s);
    if !item_bytes.is_multiple_of(4) {
        return Err(RING);
    }
    Ok(EsPlan {
        item_bytes,
        writes,
        address_only: address_only(p, &uses),
    })
}

/// The input primitive of a geometry shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Primitive {
    Point,
    Line,
    Triangle,
}

impl Primitive {
    /// DXIL's code (`D3D10_SB_PRIMITIVE`).
    pub(crate) fn dxil(self) -> u32 {
        match self {
            Primitive::Point => 1,
            Primitive::Line => 2,
            Primitive::Triangle => 3,
        }
    }

    pub(crate) fn vertices(self) -> usize {
        match self {
            Primitive::Point => 1,
            Primitive::Line => 2,
            Primitive::Triangle => 3,
        }
    }
}

/// What a geometry shader's stream 0 draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Topology {
    Points,
    LineStrip,
    TriangleStrip,
}

impl Topology {
    /// DXIL's code (`D3D_PRIMITIVE_TOPOLOGY`).
    pub(crate) fn dxil(self) -> u32 {
        match self {
            Topology::Points => 1,
            Topology::LineStrip => 3,
            Topology::TriangleStrip => 5,
        }
    }
}

/// A geometry shader's shape: what its container's `PSV0` says (input
/// primitive, output topology, most vertices) and its GSVS ring layout
/// from the console's stage data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GsShape {
    pub input: Primitive,
    pub topology: Topology,
    pub max_vertices: u32,
    /// One vertex of stream 0 in the GSVS ring, and one lane's whole item,
    /// in dwords.
    pub vertex_dwords: u32,
    pub item_dwords: u32,
    pub instances: u32,
}

impl GsShape {
    /// The shape of a console geometry shader: input primitive, output
    /// topology and most vertices from its container's `PSV0` (`psv`, the
    /// part's data), the GSVS ring layout from its stage data (`io`), which
    /// must agree on the most vertices. Stream 0 only, one instance: the
    /// stage data's count is 1 in every program of the title, which does not
    /// say whether it counts instances or streams.
    pub(crate) fn of(psv: &[u8], io: &GeometryIo) -> Result<GsShape, Why> {
        let at = |o: usize| {
            psv.get(o..o + 4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let size = at(0).ok_or(PSV)? as usize;
        // GSInfo, then runtime info 1: the stage (2) and the most vertices.
        if size < 28 || psv.len() < 4 + size || psv[4 + 24] != 2 {
            return Err(PSV);
        }
        let input = match at(4).ok_or(PSV)? {
            1 => Primitive::Point,
            2 => Primitive::Line,
            3 => Primitive::Triangle,
            _ => return Err(PRIMITIVE),
        };
        let topology = match at(8).ok_or(PSV)? {
            1 => Topology::Points,
            3 => Topology::LineStrip,
            5 => Topology::TriangleStrip,
            _ => return Err(TOPOLOGY),
        };
        if at(12).ok_or(PSV)? != 1 {
            return Err(STREAMS);
        }
        let max_vertices = u32::from(u16::from_le_bytes([psv[4 + 26], psv[4 + 27]]));
        if max_vertices != io.max_vertices {
            return Err(MOST_VERTICES);
        }
        if io.count != 1 {
            return Err(COUNT);
        }
        Ok(GsShape {
            input,
            topology,
            max_vertices,
            vertex_dwords: io.vertex_dwords,
            item_dwords: io.item_dwords,
            instances: 1,
        })
    }

    /// D3D12's limits for this shape with these signatures (`inputs` one
    /// input vertex's, `outputs` stream 0's).
    pub(crate) fn check(&self, inputs: &[SigElement], outputs: &[SigElement]) -> Result<(), Why> {
        if self.max_vertices == 0 || self.max_vertices > GS_MAX_VERTICES {
            return Err(VERTICES);
        }
        if self.instances == 0 || self.instances > GS_MAX_INSTANCES {
            return Err(INSTANCES);
        }
        if self.max_vertices * self.instances > GS_MAX_VERTICES {
            return Err(VERTICES);
        }
        for e in inputs.iter().chain(outputs) {
            if e.row.is_some_and(|r| r >= GS_REGISTERS) {
                return Err(ROWS);
            }
            if e.comp == 1 {
                return Err(BOOLEAN);
            }
        }
        let scalars: u32 = outputs.iter().map(|e| u32::from(e.cols)).sum();
        if self.max_vertices * scalars > GS_MAX_SCALARS {
            return Err(SCALARS);
        }
        Ok(())
    }

    /// DXIL's geometry state (entry property tag 1): input primitive, most
    /// vertices, stream mask, output topology, instances.
    pub(crate) fn state(&self) -> [u32; 5] {
        [
            self.input.dxil(),
            self.max_vertices,
            1,
            self.topology.dxil(),
            self.instances,
        ]
    }

    /// Writes the geometry runtime information into `psv` (a `PSV0` part):
    /// input primitive, output topology, stream mask, whether a position is
    /// output, and the most vertices.
    pub(crate) fn runtime_info(&self, psv: &mut [u8], position: bool) {
        let info = 4;
        psv[info..info + 4].copy_from_slice(&self.input.dxil().to_le_bytes());
        psv[info + 4..info + 8].copy_from_slice(&self.topology.dxil().to_le_bytes());
        psv[info + 8..info + 12].copy_from_slice(&1u32.to_le_bytes());
        psv[info + 12] = u8::from(position);
        psv[info + 26..info + 28].copy_from_slice(&(self.max_vertices as u16).to_le_bytes());
    }
}

/// A geometry shader's message, with the vertex an emit completes.
///
/// GS_DONE carries an operation as GS does (its bits 4-5, the stream in
/// 8-9): with one, the wave's last message is that emit or cut, then the
/// end (`done`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Message {
    /// GS EMIT of vertex `vertex` on stream 0; with `cut`, GS EMIT_CUT;
    /// with `done`, sent as GS_DONE's operation.
    Emit { vertex: u32, cut: bool, done: bool },
    /// GS CUT: the strip ends; with `done`, sent as GS_DONE's operation.
    Cut { done: bool },
    /// GS_DONE with no operation.
    Done,
}

impl Message {
    /// Whether the program sends nothing after this message.
    pub(crate) fn done(self) -> bool {
        match self {
            Message::Emit { done, .. } | Message::Cut { done } => done,
            Message::Done => true,
        }
    }
}

/// The message `inst` sends, before vertices are numbered (`vertex` 0).
fn message(inst: &Inst) -> Option<Result<Message, Why>> {
    let name = inst.name();
    if name != "s_sendmsg" && name != "s_sendmsghalt" {
        return None;
    }
    let Fields::Sopp { imm } = inst.fields else {
        return Some(Err(MESSAGE));
    };
    if name == "s_sendmsghalt" {
        return Some(Err(MESSAGE));
    }
    let (id, op, stream) = (imm & 0xF, imm >> 4 & 3, imm >> 8 & 3);
    if id != MSG_GS && id != MSG_GS_DONE {
        return Some(Err(MESSAGE));
    }
    let done = id == MSG_GS_DONE;
    Some(match op {
        // GS with no operation, or GS_DONE alone naming a stream: no
        // compiler sends them (LLVM's sendmsg rules).
        0 if done && stream == 0 => Ok(Message::Done),
        0 => Err(MESSAGE),
        _ if stream != 0 => Err(STREAM),
        1 => Ok(Message::Cut { done }),
        _ => Ok(Message::Emit {
            vertex: 0,
            cut: op == 3,
            done,
        }),
    })
}

/// What a register holds when an instruction runs ([`live_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Live {
    Value(Value),
    /// The entry's, which the program never reads (no value names it).
    Entry,
    /// The paths that reach the instruction bring different values, which
    /// no phi joins (the IR places phis only where a register is live, and
    /// a message reads M0 alone), or no path does.
    Unknown,
}

/// The value `reg` holds when the instruction at `site` runs: on each path
/// back from it, the last instruction to read or write `reg`, else the
/// block's phi of `reg`, else the entry's input. Every path must bring the
/// same: a join without a phi of `reg` (it is dead there) may bring two.
fn live_at(p: &Program, reg: Reg, site: Site) -> Live {
    let mut found = None;
    let mut seen = HashSet::new();
    let mut work = vec![site];
    while let Some((b, i)) = work.pop() {
        let block = &p.blocks[b];
        let here = block.insts[..i]
            .iter()
            .rev()
            .find_map(|x| x.write(reg).or_else(|| x.read(reg)))
            .or_else(|| {
                block
                    .phis
                    .iter()
                    .find(|ph| ph.reg == reg)
                    .map(|ph| ph.value)
            });
        let reached = match here {
            Some(v) => Some(Live::Value(v)),
            None if b == 0 => Some(
                p.inputs
                    .iter()
                    .find(|r| r.0 == reg)
                    .map_or(Live::Entry, |r| Live::Value(r.1)),
            ),
            None => None,
        };
        if let Some(v) = reached {
            match found {
                None => found = Some(v),
                Some(f) if f != v => return Live::Unknown,
                Some(_) => {}
            }
        }
        // The entry's predecessors (a loop back to the first instruction)
        // bring their own value too.
        if here.is_none() {
            for &q in &block.preds {
                if seen.insert(q) {
                    work.push((q, p.blocks[q].insts.len()));
                }
            }
        }
    }
    found.unwrap_or(Live::Unknown)
}

/// The lane mask (EXEC) at `site`: its low dword, and in a wave64 program
/// its high one; `None` when a path brings another ([`Live::Unknown`]).
fn exec_at(p: &Program, site: Site) -> Option<(Live, Live)> {
    let lo = live_at(p, Reg::S(EXEC), site);
    let hi = match p.wave64 {
        true => live_at(p, Reg::S(EXEC + 1), site),
        false => Live::Entry,
    };
    (lo != Live::Unknown && hi != Live::Unknown).then_some((lo, hi))
}

/// Each half of a merged stage first sets EXEC to its own lanes from the
/// counts in s3 (the ES's in bits 0-7, the GS's in 8-15), as the title's
/// programs do it:
///
/// ```text
/// s_bfe_i32 sN, s3, 0x80000 | first bit
/// s_bfm_b64 exec, sN, 0
/// s_cmp_eq_u32 sN, 64
/// s_cselect_b64 exec, -1, exec
/// ```
///
/// On the host every vertex shader and geometry shader invocation is one
/// of those lanes, so EXEC stays as it enters (every lane): this removes
/// the four instructions and gives their EXEC's readers the entry's. Other
/// stages are left as they are; a merged half without this set-up, or
/// whose count or compare something else reads, is refused. Runs before
/// anything else looks at the program (its instructions' sites change).
pub(crate) fn drop_lane_prologue(p: &mut Program, stage: HwStage) -> Result<(), Why> {
    let first = match stage {
        HwStage::Es => 0,
        HwStage::Gs => 8,
        _ => return Ok(()),
    };
    if !p.wave64 || p.blocks.is_empty() {
        return Err(PROLOGUE);
    }
    let s3 = p.inputs.iter().find(|i| i.0 == Reg::S(3)).map(|i| i.1);
    let insts = &p.blocks[0].insts;
    let scalar = |op: Option<Operand>| match op {
        Some(Operand::Scalar { code, .. }) => Some(code),
        _ => None,
    };
    let constant = |op: Option<Operand>| match op {
        Some(Operand::Const(c)) => inline(c),
        Some(Operand::Literal(x)) => Some(x),
        _ => None,
    };
    let src = |i: &Inst, k: usize| match &i.fields {
        Fields::Sop { src, .. } => src.get(k).copied(),
        _ => None,
    };
    let dst = |i: &Inst| match &i.fields {
        Fields::Sop { dst, .. } => scalar(*dst),
        _ => None,
    };
    let at = (0..insts.len().saturating_sub(3)).find(|&k| {
        let [bfe, bfm, cmp, sel] = [&insts[k], &insts[k + 1], &insts[k + 2], &insts[k + 3]];
        let Some(n) = dst(bfe) else {
            return false;
        };
        let count = bfe.write(Reg::S(n));
        matches!(bfe.name(), "s_bfe_i32" | "s_bfe_u32")
            && scalar(src(bfe, 0)) == Some(3)
            && s3.is_some()
            && bfe.read(Reg::S(3)) == s3
            && constant(src(bfe, 1)) == Some(8 << 16 | first)
            && bfm.name() == "s_bfm_b64"
            && dst(bfm) == Some(EXEC)
            && bfm.read(Reg::S(n)) == count
            && constant(src(bfm, 1)) == Some(0)
            && cmp.name() == "s_cmp_eq_u32"
            && scalar(src(cmp, 0)) == Some(n)
            && cmp.read(Reg::S(n)) == count
            && constant(src(cmp, 1)) == Some(64)
            && sel.name() == "s_cselect_b64"
            && dst(sel) == Some(EXEC)
            && constant(src(sel, 0)) == Some(u32::MAX)
            && scalar(src(sel, 1)) == Some(EXEC)
            && sel.read(Reg::Scc) == cmp.write(Reg::Scc)
            && sel.read(Reg::S(EXEC)) == bfm.write(Reg::S(EXEC))
            && sel.read(Reg::S(EXEC + 1)) == bfm.write(Reg::S(EXEC + 1))
    });
    let at = at.ok_or(PROLOGUE)?;
    let removed: Vec<Inst> = p.blocks[0].insts[at..at + 4].to_vec();
    let sel = &removed[3];
    // What the first three write, which only the four may read: the count,
    // SCC (the extract's, which the compare overwrites, and the compare's),
    // the narrowed EXEC. The selected EXEC's readers get the entry's.
    let internal: HashSet<Value> = removed[..3]
        .iter()
        .flat_map(|i| i.writes.iter().map(|w| w.1))
        .collect();
    let (Some(lo), Some(hi)) = (sel.write(Reg::S(EXEC)), sel.write(Reg::S(EXEC + 1))) else {
        return Err(PROLOGUE);
    };
    let read_elsewhere = p.blocks.iter().enumerate().any(|(b, block)| {
        block
            .phis
            .iter()
            .any(|phi| phi.args.iter().any(|a| internal.contains(a)))
            || block.insts.iter().enumerate().any(|(i, x)| {
                (b != 0 || !(at..at + 4).contains(&i))
                    && x.reads.iter().any(|r| internal.contains(&r.1))
            })
    });
    if read_elsewhere {
        return Err(PROLOGUE);
    }
    let mut next = fresh_value(p);
    let mut entry = |p: &mut Program, reg: Reg| match p.inputs.iter().find(|i| i.0 == reg) {
        Some(i) => i.1,
        None => {
            let v = Value(next);
            next += 1;
            p.inputs.push((reg, v));
            v
        }
    };
    let map = [
        (lo, entry(p, Reg::S(EXEC))),
        (hi, entry(p, Reg::S(EXEC + 1))),
    ];
    let to = |v: Value| map.iter().find(|m| m.0 == v).map_or(v, |m| m.1);
    p.blocks[0].insts.drain(at..at + 4);
    for block in &mut p.blocks {
        for phi in &mut block.phis {
            for a in &mut phi.args {
                *a = to(*a);
            }
        }
        for x in &mut block.insts {
            for r in &mut x.reads {
                r.1 = to(r.1);
            }
        }
    }
    Ok(())
}

/// A value number no value of `p` has.
fn fresh_value(p: &Program) -> u32 {
    let mut top = p.inputs.iter().map(|i| i.1.0).max().unwrap_or(0);
    for block in &p.blocks {
        for phi in &block.phis {
            top = phi
                .args
                .iter()
                .map(|a| a.0)
                .fold(top.max(phi.value.0), u32::max);
        }
        for x in &block.insts {
            top = x
                .reads
                .iter()
                .chain(&x.writes)
                .map(|r| r.1.0)
                .fold(top, u32::max);
        }
    }
    top + 1
}

/// A geometry shader's ring accesses and messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GsPlan {
    /// ESGS ring reads: each register dword and the input vertex and item
    /// dword it reads.
    pub reads: Vec<RingAccess>,
    /// GSVS ring writes: each register dword and the vertex and vertex
    /// dword it goes to.
    pub writes: Vec<RingAccess>,
    pub messages: Vec<(Site, Message)>,
    /// For each emit, the dwords of the vertex it completes and where each
    /// was written (dwords no path to the emit writes are left out: the
    /// console's ring holds nothing defined there either).
    pub emits: BTreeMap<Site, Vec<Written>>,
    /// The GSVS writes in a loop (their block on a cycle): each may run
    /// more than once before its vertex's emit, and the lowering keeps its
    /// last pass's values, which are the ring's only where every lane in
    /// the block writes ([`LOOPED`]).
    pub looped: BTreeSet<Site>,
    /// The most vertices a path to the end emits.
    pub emitted: u32,
    /// As [`EsPlan::address_only`], with the GS wave id the messages read
    /// in M0.
    pub address_only: BTreeSet<Site>,
}

/// A dword of an emitted vertex: the register dword the GSVS write at
/// `site` wrote it from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Written {
    pub dword: u32,
    pub site: Site,
    pub reg: Reg,
}

/// Each block's immediate dominator and the blocks each one reaches, to
/// tell whether one instruction runs on every path to another, or on one.
struct Paths {
    idom: Vec<usize>,
    reach: Vec<HashSet<usize>>,
}

impl Paths {
    fn new(p: &Program, order: &[usize], index: &[usize]) -> Paths {
        let reach = (0..p.blocks.len())
            .map(|b| {
                let mut seen = HashSet::new();
                let mut work = p.blocks[b].exit.successors();
                while let Some(s) = work.pop() {
                    if seen.insert(s) {
                        work.extend(p.blocks[s].exit.successors());
                    }
                }
                seen
            })
            .collect();
        Paths {
            idom: idoms(&p.blocks, order, index),
            reach,
        }
    }

    /// Whether every path to `b` runs `a` first.
    fn dominates(&self, a: Site, b: Site) -> bool {
        if a.0 == b.0 {
            return a.1 < b.1;
        }
        let mut d = b.0;
        loop {
            let up = self.idom[d];
            if up == usize::MAX || up == d {
                return false;
            }
            if up == a.0 {
                return true;
            }
            d = up;
        }
    }

    /// Whether some path runs `a`, then `b`.
    fn reaches(&self, a: Site, b: Site) -> bool {
        (a.0 == b.0 && a.1 < b.1) || self.reach[a.0].contains(&b.0)
    }
}

/// Resolves the ring accesses of the geometry shader `p` of shape `shape`
/// and numbers the vertices its emits complete; refuses what D3D12's
/// emits cannot give as the console's copy shader would (module notes).
pub(crate) fn gs_plan(p: &Program, shape: &GsShape) -> Result<GsPlan, Why> {
    if shape.vertex_dwords == 0 || shape.item_dwords != shape.max_vertices * shape.vertex_dwords {
        return Err(STREAMS);
    }
    let mut forms = Forms::new(p);
    let vertices: Vec<Option<Field>> = VERTEX_OFFSETS[..shape.input.vertices()]
        .iter()
        .map(|&(reg, lo)| {
            p.inputs.iter().find(|r| r.0 == Reg::V(reg)).map(|r| Field {
                base: Base::Input(r.1),
                lo,
                width: 16,
            })
        })
        .collect();
    let (item, vertex_bytes) = (shape.item_dwords * 4, shape.vertex_dwords * 4);
    let (mut reads, mut writes, mut sent, mut uses) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut base: Option<Form> = None;
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            if let Some(m) = message(inst) {
                sent.push((site, m?));
                if let Some(m0) = inst.read(Reg::S(M0)) {
                    uses.push((site, m0));
                }
                continue;
            }
            if !matches!(inst.fields, Fields::Ds { .. }) {
                continue;
            }
            let (write, dwords) = ds_dwords(inst).ok_or(RING)?;
            let address = ds_address(inst).ok_or(RING)?;
            let form = forms.value(address).ok_or(RING)?;
            let mut out = Vec::with_capacity(dwords.len());
            match (write, forms.classify(&form, &vertices).ok_or(RING)?) {
                (false, Where::Vertex { vertex, offset }) => {
                    for (reg, at) in dwords {
                        let byte = offset.wrapping_add(at);
                        if !byte.is_multiple_of(4) || byte / 4 >= 4 * GS_REGISTERS {
                            return Err(RING);
                        }
                        let vertex = u32::from(vertex);
                        out.push((
                            reg,
                            Slot {
                                vertex,
                                dword: byte / 4,
                            },
                        ));
                    }
                    reads.push(RingAccess { site, dwords: out });
                }
                (
                    true,
                    Where::Lane {
                        stride,
                        rest,
                        offset,
                    },
                ) if stride == item => {
                    match &base {
                        None => base = Some(rest),
                        Some(r) if *r == rest => {}
                        Some(_) => return Err(RING),
                    }
                    for (reg, at) in dwords {
                        let byte = offset.wrapping_add(at);
                        if !byte.is_multiple_of(4) || byte >= item {
                            return Err(RING);
                        }
                        let slot = Slot {
                            vertex: byte / vertex_bytes,
                            dword: byte % vertex_bytes / 4,
                        };
                        out.push((reg, slot));
                    }
                    writes.push(RingAccess { site, dwords: out });
                }
                _ => return Err(RING),
            }
            uses.push((site, address));
        }
    }

    // Number the emits: the emits before each instruction, the same on
    // every path; nothing after GS_DONE.
    let order = reverse_postorder(&p.blocks);
    let mut index = vec![usize::MAX; p.blocks.len()];
    for (k, &b) in order.iter().enumerate() {
        index[b] = k;
    }
    let messages_at: HashMap<Site, usize> =
        sent.iter().enumerate().map(|(k, m)| (m.0, k)).collect();
    let written: HashSet<Site> = writes.iter().map(|w| w.site).collect();
    let mut entry: Vec<Option<(u32, bool)>> = vec![None; p.blocks.len()];
    let mut exit: Vec<Option<(u32, bool)>> = vec![None; p.blocks.len()];
    let mut before: HashMap<Site, u32> = HashMap::new();
    for &b in &order {
        let mut state = (b == order[0]).then_some((0, false));
        for &q in &p.blocks[b].preds {
            if index[q] == usize::MAX || index[q] >= index[b] {
                continue;
            }
            let e = exit[q];
            match state {
                None => state = e,
                Some(s) if Some(s) != e => return Err(PATH),
                Some(_) => {}
            }
        }
        let state = state.ok_or(PATH)?;
        entry[b] = Some(state);
        let (mut count, mut done) = state;
        for i in 0..p.blocks[b].insts.len() {
            if written.contains(&(b, i)) {
                before.insert((b, i), count);
            }
            let Some(&k) = messages_at.get(&(b, i)) else {
                continue;
            };
            if done {
                return Err(AFTER_DONE);
            }
            let m = &mut sent[k].1;
            if let Message::Emit { vertex, .. } = m {
                if count >= shape.max_vertices {
                    return Err(TOO_MANY);
                }
                *vertex = count;
                count += 1;
            }
            done = m.done();
        }
        exit[b] = Some((count, done));
    }
    for &b in &order {
        for s in p.blocks[b].exit.successors() {
            if index[s] <= index[b] && exit[b] != entry[s] {
                return Err(PATH);
            }
        }
    }
    // Every emit and cut under one lane mask: the console counts a lane's
    // vertices by the emits it is in EXEC for, the numbering above by the
    // path. GS_DONE alone emits nothing a lane counts.
    let mut mask = None;
    for &(site, m) in &sent {
        if m == Message::Done || index[site.0] == usize::MAX {
            continue;
        }
        let here = exec_at(p, site).ok_or(MASKED)?;
        match mask {
            None => mask = Some(here),
            Some(m) if m != here => return Err(MASKED),
            Some(_) => {}
        }
    }

    // Every write before its vertex's emit; for each emit, each dword of
    // its vertex written at most once on the paths to it, by an instruction
    // on every one of them.
    let paths = Paths::new(p, &order, &index);
    let mut by_slot: BTreeMap<Slot, Vec<(Site, Reg)>> = BTreeMap::new();
    for w in &writes {
        // A write in a block no path reaches writes nothing.
        let Some(&emitted) = before.get(&w.site) else {
            continue;
        };
        for &(reg, slot) in &w.dwords {
            if slot.vertex < emitted {
                return Err(LATE);
            }
            by_slot.entry(slot).or_default().push((w.site, reg));
        }
    }
    let mut emits = BTreeMap::new();
    for &(site, m) in &sent {
        let Message::Emit { vertex, .. } = m else {
            continue;
        };
        if index[site.0] == usize::MAX {
            continue;
        }
        let first = Slot { vertex, dword: 0 };
        let next = Slot {
            vertex: vertex + 1,
            dword: 0,
        };
        let mut dwords = Vec::new();
        for (slot, ws) in by_slot.range(first..next) {
            let on_paths: Vec<_> = ws.iter().filter(|w| paths.reaches(w.0, site)).collect();
            if on_paths.iter().any(|w| !paths.dominates(w.0, site)) {
                return Err(SOME_PATHS);
            }
            let (w, reg) = match on_paths[..] {
                [] => continue,
                [&one] => one,
                _ => return Err(TWICE),
            };
            dwords.push(Written {
                dword: slot.dword,
                site: w,
                reg,
            });
        }
        emits.insert(site, dwords);
    }
    let looped = writes
        .iter()
        .map(|w| w.site)
        .filter(|&(b, _)| paths.reach[b].contains(&b))
        .collect();
    let emitted = order
        .iter()
        .filter(|&&b| p.blocks[b].exit.successors().is_empty())
        .filter_map(|&b| exit[b].map(|e| e.0))
        .max()
        .unwrap_or(0);
    Ok(GsPlan {
        reads,
        writes,
        messages: sent,
        emits,
        looped,
        emitted,
        address_only: address_only(p, &uses),
    })
}

/// The registers an instruction reads as operands (not EXEC, not the old
/// value of a VGPR it writes).
fn sources(fields: &Fields) -> Vec<Reg> {
    let mut out = Vec::new();
    let mut add = |op: &Operand| operand_regs(op, &mut out);
    match fields {
        Fields::Sop { src, .. } => src.iter().for_each(&mut add),
        Fields::Valu(v) => v.src.iter().for_each(&mut add),
        Fields::Ds { addr, data, .. } => addr.iter().chain(data).for_each(&mut add),
        Fields::Smem {
            data,
            base,
            soffset,
            ..
        } => data.iter().chain([base, soffset]).for_each(&mut add),
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            ..
        } => data.iter().chain([addr, rsrc, soffset]).for_each(&mut add),
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            ..
        } => [data, rsrc]
            .into_iter()
            .chain(addr)
            .chain(sampler)
            .for_each(&mut add),
        Fields::Flat {
            addr, data, saddr, ..
        } => addr.iter().chain(data).chain(saddr).for_each(&mut add),
        Fields::Exp { src, .. } => src.iter().flatten().for_each(&mut add),
        Fields::Vintrp { src, attr, .. } => [src, attr].into_iter().for_each(&mut add),
        Fields::Sopp { .. } => {}
    }
    out
}

/// The register dwords operand `op` names, added to `out`.
fn operand_regs(op: &Operand, out: &mut Vec<Reg>) {
    match *op {
        Operand::Scalar { code, dwords } => out.extend((0..dwords).map(|i| Reg::S(code + i))),
        Operand::Vector { reg, dwords } => out.extend((0..dwords).map(|i| Reg::V(reg + i))),
        _ => {}
    }
}

/// The registers a ring access or a message reads besides the value its
/// use names (its address, or M0): an LDS write's data.
fn data_regs(fields: &Fields) -> Vec<Reg> {
    let mut out = Vec::new();
    if let Fields::Ds { data, .. } = fields {
        for op in data {
            operand_regs(op, &mut out);
        }
    }
    out
}

/// Whether `inst` only computes register values from its operands (no
/// memory but scalar loads, no EXEC, no call).
fn pure(inst: &Inst) -> bool {
    let exec = inst
        .writes
        .iter()
        .any(|(r, _)| matches!(r, Reg::S(c) if *c == EXEC || *c == EXEC + 1));
    let name = inst.name();
    !exec
        && match inst.fields {
            Fields::Sop { .. } => !matches!(name, "s_swappc_b64" | "s_setpc_b64" | "s_call_b64"),
            Fields::Valu(_) => true,
            Fields::Smem { .. } => name.starts_with("s_load") || name.starts_with("s_buffer_load"),
            _ => false,
        }
}

/// The instructions whose every result is used only by others of them, or
/// as `uses` lists (an instruction and a value it reads: a ring access's
/// address, a message's M0), found from those values back.
fn address_only(p: &Program, uses: &[(Site, Value)]) -> BTreeSet<Site> {
    let inst = |(b, i): Site| &p.blocks[b].insts[i];
    let mut defs: HashMap<Value, Site> = HashMap::new();
    let mut readers: HashMap<Value, Vec<Site>> = HashMap::new();
    let mut escapes: HashSet<Value> = HashSet::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for phi in &block.phis {
            escapes.extend(&phi.args);
        }
        for (i, x) in block.insts.iter().enumerate() {
            for &(_, v) in &x.writes {
                defs.insert(v, (b, i));
            }
            let srcs = sources(&x.fields);
            for &(reg, v) in &x.reads {
                let exec = matches!(reg, Reg::S(c) if c == EXEC || c == EXEC + 1);
                let old = matches!(reg, Reg::V(_))
                    && x.writes.iter().any(|w| w.0 == reg)
                    && !srcs.contains(&reg);
                if !exec && !old {
                    readers.entry(v).or_default().push((b, i));
                }
            }
        }
    }
    // A use excuses its consumer's read of the value only when the consumer
    // reads it as nothing else: an LDS write may also store its own
    // address, which then must be computed.
    let allowed: HashSet<(Site, Value)> = uses
        .iter()
        .copied()
        .filter(|&(site, v)| {
            let x = inst(site);
            let data = data_regs(&x.fields);
            !x.reads.iter().any(|&(r, w)| w == v && data.contains(&r))
        })
        .collect();
    let consumers: HashSet<Site> = uses.iter().map(|u| u.0).collect();
    let mut slice: BTreeSet<Site> = BTreeSet::new();
    let mut work: Vec<Value> = uses.iter().map(|u| u.1).collect();
    while let Some(v) = work.pop() {
        let Some(&site) = defs.get(&v) else {
            continue;
        };
        if consumers.contains(&site) || !pure(inst(site)) || !slice.insert(site) {
            continue;
        }
        let x = inst(site);
        let srcs = sources(&x.fields);
        work.extend(
            x.reads
                .iter()
                .filter(|(r, _)| srcs.contains(r))
                .map(|&(_, v)| v),
        );
    }
    loop {
        let before = slice.len();
        let kept: BTreeSet<Site> = slice
            .iter()
            .copied()
            .filter(|&site| {
                inst(site).writes.iter().all(|&(_, w)| {
                    !escapes.contains(&w)
                        && readers.get(&w).is_none_or(|rs| {
                            rs.iter()
                                .all(|r| slice.contains(r) || allowed.contains(&(*r, w)))
                        })
                })
            })
            .collect();
        slice = kept;
        if slice.len() == before {
            return slice;
        }
    }
}

/// Where dword `dword` of a ring vertex (row `dword / 4`, column
/// `dword % 4`) is in `elements`: the element's index and the column within
/// it.
pub(crate) fn element_at(elements: &[SigElement], dword: u32) -> Option<(u32, u8)> {
    let (row, col) = (dword / 4, (dword % 4) as u8);
    elements.iter().enumerate().find_map(|(k, e)| {
        (e.row == Some(row) && (e.col..e.col + e.cols).contains(&col))
            .then(|| (k as u32, col - e.col))
    })
}

/// Stores `value` (a dword) to column `col` of output element `id`.
fn store(s: &mut Shader, block: usize, e: &SigElement, id: u32, col: u8, value: V) {
    let (value, suffix) = if e.comp == 9 {
        let f32t = s.t.f32;
        (s.m.cast(block, Cast::Bitcast, value, f32t), "f32")
    } else {
        (value, "i32")
    };
    let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), value];
    let void = s.t.void;
    s.op(
        block,
        &format!("storeOutput.{suffix}"),
        Attr::NoUnwind,
        void,
        5,
        &args,
    );
}

/// An ES ring write of item dword `dword`, as the vertex shader's output in
/// `outputs` (the ES's output signature) at that row and column; a dword no
/// element holds is no output.
pub(crate) fn write_item(
    s: &mut Shader,
    block: usize,
    outputs: &[SigElement],
    dword: u32,
    value: V,
) {
    if let Some((id, col)) = element_at(outputs, dword) {
        store(s, block, &outputs[id as usize], id, col, value);
    }
}

/// A GS read of a ring dword of an input vertex: `loadInput` of that row and
/// column of `inputs` (the GS's input signature) for that vertex, as a
/// dword.
pub(crate) fn read_vertex(
    s: &mut Shader,
    block: usize,
    inputs: &[SigElement],
    slot: Slot,
) -> Result<V, Why> {
    let (id, col) = element_at(inputs, slot.dword).ok_or(NOT_INPUT)?;
    let float = inputs[id as usize].comp == 9;
    let (ty, suffix) = if float {
        (s.t.f32, "f32")
    } else {
        (s.t.i32, "i32")
    };
    let args = [s.m.i32(id), s.m.i32(0), s.m.i8(col), s.m.i32(slot.vertex)];
    let v = s.op(
        block,
        &format!("loadInput.{suffix}"),
        Attr::ReadNone,
        ty,
        4,
        &args,
    );
    Ok(if float {
        let i32t = s.t.i32;
        s.m.cast(block, Cast::Bitcast, v, i32t)
    } else {
        v
    })
}

/// The GSVS ring of one GS invocation: the dwords its ring writes wrote,
/// by write, until the vertices they belong to are emitted.
#[derive(Debug, Default)]
pub(crate) struct Vertices {
    values: HashMap<(Site, Reg), V>,
}

impl Vertices {
    /// Keeps `value`, which the GSVS write at `site` writes from `reg`.
    pub(crate) fn write(&mut self, site: Site, reg: Reg, value: V) {
        self.values.insert((site, reg), value);
    }

    /// Emits on stream 0 the vertex whose dwords `dwords` lists (an emit's
    /// [`GsPlan::emits`]): those `outputs` (stream 0's output signature)
    /// holds, then `emitStream`. Each write of such a dword must have been
    /// kept first, as its dominating the emit lets the lowering do; the
    /// other dwords are no output, and the lowering keeps none of them.
    pub(crate) fn emit(
        &self,
        s: &mut Shader,
        block: usize,
        outputs: &[SigElement],
        dwords: &[Written],
    ) -> Result<(), Why> {
        for w in dwords {
            let Some((id, col)) = element_at(outputs, w.dword) else {
                continue;
            };
            let &value = self.values.get(&(w.site, w.reg)).ok_or(SOME_PATHS)?;
            store(s, block, &outputs[id as usize], id, col, value);
        }
        let (stream, void) = (s.m.i8(0), s.t.void);
        s.op(block, "emitStream", Attr::NoUnwind, void, 97, &[stream]);
        Ok(())
    }

    /// Ends the strip on stream 0.
    pub(crate) fn cut(s: &mut Shader, block: usize) {
        let (stream, void) = (s.m.i8(0), s.t.void);
        s.op(block, "cutStream", Attr::NoUnwind, void, 98, &[stream]);
    }
}

// The translation's use of the plans (`docs/analysis/es-gs.md`, section 6):
// the entry point prepares a merged half and its resources, the lowering
// calls the hooks below at the plans' sites.

/// A merged half whose stage data the translation does not read
/// (`native.rs`): no ring layout, no outputs.
pub(crate) const STAGE_DATA: Why =
    "a merged ES + GS program whose stage data the translation does not read";
/// A ring row holding two 16-bit values per dword: the ring dword is no
/// signature element's 32-bit value.
pub(crate) const HALF: Why = "a 16-bit row in a merged ES + GS ring";

fn refused(what: Why) -> ShaderError {
    ShaderError::Unsupported { offset: 0, what }
}

/// A merged half's program as the rest of the translation takes it: each
/// half's EXEC set-up from s3 left out ([`drop_lane_prologue`]); other
/// stages' as they are.
pub(crate) fn prepared(mut p: Program, stage: HwStage) -> Result<Program, ShaderError> {
    drop_lane_prologue(&mut p, stage).map_err(refused)?;
    Ok(p)
}

/// The shape of the geometry shader of `c`, whose console program is
/// `native` ([`GsShape::of`]).
fn shape_of(c: &Container, native: &NativeProgram) -> Result<GsShape, ShaderError> {
    let StageIo::Gs(io) = &native.io else {
        return Err(refused(STAGE_DATA));
    };
    let psv = c.part(FourCC::PSV0).ok_or(refused(PSV))?.data;
    GsShape::of(psv, io).map_err(refused)
}

/// The program the resource tracker reads: in a merged half, the scalar
/// loads that only set up ring addresses ([`EsPlan::address_only`],
/// [`GsPlan::address_only`]) are an `s_nop` with the same results. They read
/// a table of the console's driver (the GS's ring base), which no root
/// parameter holds; the lowering leaves them out. A plan's refusal is
/// reported here, at the resources' step: the plan says which loads those
/// are.
// shortcut: until `resource::track` takes the sites to leave out (es-gs.md,
// section 6, item 5), it reads this copy; sites and values are the same.
pub(crate) fn tracked<'p>(
    c: &Container,
    p: &'p Program,
    native: &NativeProgram,
) -> Result<Cow<'p, Program>, ShaderError> {
    let address_only = match native.stage {
        HwStage::Es => es_plan(p).map_err(refused)?.address_only,
        HwStage::Gs => {
            gs_plan(p, &shape_of(c, native)?)
                .map_err(refused)?
                .address_only
        }
        _ => return Ok(Cow::Borrowed(p)),
    };
    let loads: Vec<Site> = address_only
        .into_iter()
        .filter(|&(b, i)| matches!(p.blocks[b].insts[i].fields, Fields::Smem { .. }))
        .collect();
    if loads.is_empty() {
        return Ok(Cow::Borrowed(p));
    }
    let mut q = p.clone();
    for (b, i) in loads {
        let inst = &mut q.blocks[b].insts[i];
        inst.machine = Instruction {
            offset: inst.machine.offset,
            encoding: Encoding::Sopp,
            opcode: 0,
            words: vec![0xBF80_0000],
        };
        inst.fields = Fields::Sopp { imm: 0 };
        inst.reads.clear();
    }
    Ok(Cow::Owned(q))
}

/// `io` with the geometry shader's shape ([`Io::geometry`]) when `native`
/// is a GS: its container's `PSV0` gives it, which the lowering does not
/// read.
pub(crate) fn with_shape<'i>(
    c: &Container,
    native: &NativeProgram,
    io: &'i Io,
) -> Result<Cow<'i, Io>, ShaderError> {
    if native.stage != HwStage::Gs || io.geometry.is_some() {
        return Ok(Cow::Borrowed(io));
    }
    Ok(Cow::Owned(Io {
        geometry: Some(shape_of(c, native)?),
        ..io.clone()
    }))
}

/// DXC's interpolation of a vertex output by its kind and component type,
/// as `emit_for` declares a vertex shader's: a geometry shader's inputs
/// are declared as the vertex shader's outputs they are.
fn vertex_interp(e: &SigElement) -> u8 {
    match (e.kind, e.comp) {
        (3, _) => 4,
        (13, _) => 1,
        (28, _) => 2,
        (0 | 6 | 7, 9) => 2,
        (0, _) => 1,
        _ => 0,
    }
}

/// What the lowering of a merged half needs at its plan's sites.
pub(super) struct Merged {
    stage: HwStage,
    /// GS ring reads, ES or GS ring writes, by site.
    reads: HashMap<Site, Vec<(Reg, Slot)>>,
    writes: HashMap<Site, Vec<(Reg, Slot)>>,
    messages: HashMap<Site, Message>,
    emits: BTreeMap<Site, Vec<Written>>,
    /// GS ring writes in a loop ([`GsPlan::looped`]).
    looped: BTreeSet<Site>,
    address_only: BTreeSet<Site>,
    vertices: Vertices,
    /// The shader's signatures, as declared.
    inputs: Vec<SigElement>,
    outputs: Vec<SigElement>,
}

impl Lower<'_> {
    /// Before the blocks: an ES or GS program's plan, and for a GS its
    /// geometry state and input declarations.
    pub(super) fn merged_start(&mut self) -> Result<(), Stop> {
        let x = self.x;
        let stage = x.native.stage;
        let rows = |outputs: &[VsOutput]| outputs.iter().any(|o| o.fp16 || o.pack2);
        let half = |els: &[SigElement]| els.iter().any(|e| e.half);
        let fail = |what| unsupported(0, what);
        let mut m = Merged {
            stage,
            reads: HashMap::new(),
            writes: HashMap::new(),
            messages: HashMap::new(),
            emits: BTreeMap::new(),
            looped: BTreeSet::new(),
            address_only: BTreeSet::new(),
            vertices: Vertices::default(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        };
        match (&x.native.io, stage) {
            (StageIo::Es(es), HwStage::Es) => {
                if rows(&es.outputs) || half(&self.s.outputs) {
                    return Err(fail(HALF));
                }
                let plan = es_plan(x.p).map_err(fail)?;
                m.writes = plan
                    .writes
                    .into_iter()
                    .map(|w| (w.site, w.dwords))
                    .collect();
                m.address_only = plan.address_only;
            }
            (StageIo::Gs(gs), HwStage::Gs) => {
                if rows(&gs.outputs) || half(&self.s.inputs) || half(&self.s.outputs) {
                    return Err(fail(HALF));
                }
                let shape = x.io.geometry.ok_or_else(|| fail(PSV))?;
                for e in &mut self.s.inputs {
                    e.interp = vertex_interp(e);
                }
                shape.check(&self.s.inputs, &self.s.outputs).map_err(fail)?;
                let plan = gs_plan(x.p, &shape).map_err(fail)?;
                m.reads = plan.reads.into_iter().map(|r| (r.site, r.dwords)).collect();
                m.writes = plan
                    .writes
                    .into_iter()
                    .map(|w| (w.site, w.dwords))
                    .collect();
                m.messages = plan.messages.into_iter().collect();
                m.emits = plan.emits;
                m.looped = plan.looped;
                m.address_only = plan.address_only;
                self.s.geometry = Some(shape);
            }
            (_, HwStage::Es | HwStage::Gs) => return Err(fail(STAGE_DATA)),
            _ => return Ok(()),
        }
        m.inputs = self.s.inputs.clone();
        m.outputs = self.s.outputs.clone();
        self.merged = Some(Box::new(m));
        Ok(())
    }

    /// Instruction `(b, i)` of a merged half where the plan says what it is
    /// (`None` elsewhere): ring set-up is left out, ring reads load inputs,
    /// ES ring writes store outputs, GS ring writes are kept until their
    /// vertex's emit, messages emit and cut. The halves' barriers and the
    /// ES's jump to the GS half have no host form: the rings they order are
    /// gone.
    pub(super) fn merged_inst(&mut self, b: usize, i: usize) -> Option<Result<(), Stop>> {
        let mut m = self.merged.take()?;
        let r = self.merged_site(&mut m, b, i);
        self.merged = Some(m);
        r
    }

    fn merged_site(&mut self, m: &mut Merged, b: usize, i: usize) -> Option<Result<(), Stop>> {
        let inst: &Inst = &self.x.p.blocks[b].insts[i];
        let site = (b, i);
        let offset = inst.machine.offset;
        if m.address_only.contains(&site) || matches!(inst.name(), "s_barrier" | "s_setpc_b64") {
            return Some(Ok(()));
        }
        if let Some(dwords) = m.reads.get(&site) {
            return Some(self.merged_read(inst, b, i, &m.inputs, dwords));
        }
        if let Some(dwords) = m.writes.get(&site) {
            let r = match m.stage {
                HwStage::Es => self.item_write(inst, b, i, &m.outputs, dwords),
                _ => {
                    let looped = m.looped.contains(&site);
                    self.vertex_write(inst, b, i, &m.outputs, dwords, looped)
                        .map(|kept| {
                            for (reg, value) in kept {
                                m.vertices.write(site, reg, value);
                            }
                        })
                }
            };
            return Some(r.map_err(|why| unsupported(offset, why)));
        }
        let &message = m.messages.get(&site)?;
        Some(
            self.message(m, b, i, message)
                .map_err(|why| unsupported(offset, why)),
        )
    }

    /// A GS ring read: each dword it loads is that input vertex's
    /// signature element, under the instruction's lane guard. A dword no
    /// element holds reads as missing ([`NOT_INPUT`]).
    fn merged_read(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        inputs: &[SigElement],
        dwords: &[(Reg, Slot)],
    ) -> Result<(), Stop> {
        let guard = self
            .guard(b, i)
            .map_err(|why| unsupported(inst.machine.offset, why))?;
        for &(reg, slot) in dwords {
            match read_vertex(&mut self.s, self.cur, inputs, slot) {
                Ok(v) => self.write_placed(inst, reg, v, guard),
                Err(why) => self.set(inst.write(reg), Val::Missing(why)),
            }
        }
        Ok(())
    }

    /// A GS ring write: the values of the dwords stream 0's outputs hold,
    /// which the lowering keeps until their vertex's emit. The others (a
    /// short last row's fill) are no output, whatever they hold.
    ///
    /// The value kept is what the write's lanes store, and an emit's lanes
    /// read it whatever EXEC the write had: a lane outside it leaves the
    /// ring dword as it was. Where the write runs once, that is nothing
    /// the program wrote (no other write of the dword reaches the emit:
    /// [`TWICE`]), any value; in a loop it is an earlier pass's, which the
    /// host does not keep: refused there unless every lane in the block
    /// writes ([`LOOPED`]).
    fn vertex_write(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        outputs: &[SigElement],
        dwords: &[(Reg, Slot)],
        looped: bool,
    ) -> Result<Vec<(Reg, V)>, Why> {
        let held: Vec<Reg> = dwords
            .iter()
            .filter(|&&(_, slot)| element_at(outputs, slot.dword).is_some())
            .map(|&(reg, _)| reg)
            .collect();
        if looped && !held.is_empty() && self.guard(b, i)?.is_some() {
            return Err(LOOPED);
        }
        held.into_iter()
            .map(|reg| {
                let v = inst.read(reg).ok_or("an unread operand")?;
                Ok((reg, self.i32_of(v)?))
            })
            .collect()
    }

    /// An ES ring write: each item dword is the vertex shader's output at
    /// its row and column, stored by the lanes the instruction runs in. A
    /// dword no output element holds (a short last row's fill) is no
    /// output, whatever it holds.
    fn item_write(
        &mut self,
        inst: &Inst,
        b: usize,
        i: usize,
        outputs: &[SigElement],
        dwords: &[(Reg, Slot)],
    ) -> Result<(), Why> {
        let values = dwords
            .iter()
            .filter(|&&(_, slot)| element_at(outputs, slot.dword).is_some())
            .map(|&(reg, slot)| {
                let v = inst.read(reg).ok_or("an unread operand")?;
                Ok((slot.dword, self.i32_of(v)?))
            })
            .collect::<Result<Vec<_>, Why>>()?;
        let guard = self.guard(b, i)?;
        let at = self.begin_guarded(guard);
        for (dword, value) in values {
            write_item(&mut self.s, self.cur, outputs, dword, value);
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// A GS message: an emit stores its vertex and emits it, a cut ends the
    /// strip, under the lanes' EXEC where some lane outside it gets there.
    fn message(&mut self, m: &Merged, b: usize, i: usize, message: Message) -> Result<(), Why> {
        if message == Message::Done {
            return Ok(());
        }
        let guard = self.scalar_guard(b, i)?;
        let at = self.begin_guarded(guard);
        match message {
            Message::Emit { cut, .. } => {
                let dwords = m.emits.get(&(b, i)).ok_or(SOME_PATHS)?;
                m.vertices.emit(&mut self.s, self.cur, &m.outputs, dwords)?;
                if cut {
                    Vertices::cut(&mut self.s, self.cur);
                }
            }
            Message::Cut { .. } => Vertices::cut(&mut self.s, self.cur),
            Message::Done => {}
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Whether block `b` ends a merged half's program: the ES's jump to the
    /// GS half ends the vertex shader (it returns).
    pub(super) fn merged_end(&mut self, b: usize) -> bool {
        let ends = self.merged.as_ref().is_some_and(|m| m.stage == HwStage::Es)
            && self.x.p.blocks[b].exit == Exit::Indirect;
        if ends {
            self.s.m.terminate(self.cur, Term::Ret);
        }
        ends
    }
}
