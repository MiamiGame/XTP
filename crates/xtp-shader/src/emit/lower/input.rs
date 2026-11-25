//! Program inputs: user data, system values, vertex and pixel inputs,
//! interpolation (`v_interp_*`). Backlog P1 (packed 16-bit pixel inputs),
//! P2 (exact interpolation), S3 (hardware SGPR inputs).

use std::collections::{HashMap, HashSet};

use super::{Lower, Stop, Val, Why, packed_half, packed_partners};
use crate::emit::UserData;
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::container::SigElement;
use crate::emit::shader::flags;
use crate::gfx10::{Fields, Operand};
use crate::io::{Barycentric, Input, Interp};
use crate::ir::{Inst, Reg, Value};
use crate::native::{PsInput, RootType, StageIo};
use crate::resource::Site;
use crate::structure::{Atom, Bool, LaneExit};

/// The second rows of pixel inputs packing two 16-bit rows per dword, by
/// first row: the vertex outputs' rule ([`super::export::half_outputs`]),
/// the next 16-bit row of the signature that no input entry names, taken
/// in row order ([`packed_partners`]).
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    packed_partners(
        inputs,
        |r| entries.iter().any(|i| i.row == r),
        entries.iter().filter(|i| i.fp16 && i.pack2).map(|i| i.row),
    )
}

/// The second row of packed pixel input `entry` ([`half_input_partners`]).
/// The metadata names it for some inputs (`row2` ≠ `row`), never for vertex
/// outputs, so a row it names that the rule does not give is refused: the
/// vertex program packs another row there.
fn half_input_partner(
    io: &StageIo,
    inputs: &[SigElement],
    entry: &PsInput,
) -> Result<Option<u8>, Why> {
    let partner = half_input_partners(io, inputs).get(&entry.row).copied();
    if entry.row2 != entry.row && partner != Some(entry.row2) {
        return Err("a packed input's second row the vertex outputs' rule does not pair");
    }
    Ok(partner)
}

/// Where half `h` of a pixel input packing rows `a` and `b` comes from
/// ([`packed_half`]), or `None` for the padding of a second row the metadata
/// names (`explicit_second_row`). With a partner found by
/// [`half_input_partners`] a half no element holds stays an error: the
/// guess must not turn real data into zeros.
fn packed_input_half(
    elements: &[SigElement],
    a: u8,
    b: Option<u8>,
    h: u8,
    explicit_second_row: bool,
) -> Result<Option<(u8, u8)>, Why> {
    let half = packed_half(elements, a, b, h);
    if half.is_none() && !explicit_second_row {
        Err("a packed input half no element holds")
    } else {
        Ok(half)
    }
}

/// Terme auquel s'ajoute le produit du poids par la différence des sommets.
#[derive(Debug, Clone, Copy)]
enum InterpBase {
    P0,
    Accumulator(V),
    Zero,
}

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// Somme de la base et des termes pondérés I (P1 − P0) et J (P2 − P0).
    Weights {
        base: InterpBase,
        i: Option<V>,
        j: Option<V>,
    },
}

impl<'a> Lower<'a> {
    /// Un SGPR temporaire peut contenir un masque ou un descripteur sur une
    /// branche et des données sur l'autre. Suivre la première jusqu'à la
    /// prochaine écriture pour vérifier qu'aucune lecture ne voit ses bits.
    pub(super) fn unused_pixel_sgpr(&self, reg: Reg, pred: usize, join: usize) -> bool {
        if !self.s.pixel || self.x.io.initialized_sgprs.is_none() || !matches!(reg, Reg::S(0..=105))
        {
            return false;
        }
        let mut work = vec![(pred, join)];
        let mut seen = HashSet::new();
        'edge: while let Some((pred, b)) = work.pop() {
            if !seen.insert((pred, b)) {
                continue;
            }
            let block = &self.x.p.blocks[b];
            for i in &block.insts {
                if i.read(reg).is_some() {
                    return false;
                }
                if i.write(reg).is_some() {
                    continue 'edge;
                }
            }
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = self.x.st.exits[b]
                && let Some(t) = self.pixel_edge_condition(b, pred, when)
            {
                work.push((b, if t { taken } else { not_taken }));
            } else {
                work.extend(block.exit.successors().into_iter().map(|to| (b, to)));
            }
        }
        true
    }

    /// A flag VGPR set on each predecessor is often tested at the join
    /// before reading the associated data. Evaluate a plain equality test
    /// with this edge's incoming constants; either successor stays possible
    /// if the test (or its negation) is not known.
    fn pixel_edge_condition(&self, b: usize, pred: usize, when: Bool) -> Option<bool> {
        let (Atom::Lane(v), no, yes) = self.x.st.bools.node(when)? else {
            return None;
        };
        let block = &self.x.p.blocks[b];
        let cmp = block
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        if cmp.name() != "v_cmp_eq_u32"
            || cmp.machine.encoding != crate::gfx10::Encoding::Vopc
            || cmp.machine.words.len() != 1
        {
            return None;
        }
        let Fields::Valu(alu) = &cmp.fields else {
            return None;
        };
        let word = |op| match op {
            Operand::Const(c) => crate::resource::inline(c),
            Operand::Vector { reg, .. } => {
                let mut v = cmp.read(Reg::V(reg))?;
                if let Some(phi) = block.phis.iter().find(|p| p.value == v) {
                    v = phi.args[block.preds.iter().position(|&p| p == pred)?];
                }
                let Val::I32(x) = self.vals.get(&v)? else {
                    return None;
                };
                self.s.m.int_value(*x).map(|k| k as u32)
            }
            _ => None,
        };
        match if word(alu.src[0])? == word(alu.src[1])? {
            yes
        } else {
            no
        } {
            Bool::TRUE => Some(true),
            Bool::FALSE => Some(false),
            _ => None,
        }
    }

    /// Values of the program's input registers.
    pub(super) fn inputs(&mut self) -> Result<(), Stop> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`lazy_barycentric`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::FloatMode => Val::I32(self.s.m.i32(self.x.native.float_mode.into())),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    pub(super) fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) if self.flat_group.is_some() => {
                return Val::I32(self.flat_thread_id(c));
            }
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return Val::I32(self.group_id(c)),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let g = self.group_id(0);
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    pub(super) fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    // shortcut: triangles only; on lines and points the console's i and j
    // are not checked against the host's weights.
    pub(super) fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one). DXIL allows two, one perspective and one
        // linear: a second perspective (or linear) mode is refused, since
        // its weights differ from the first's where the modes' sample points
        // do (centroid weights stay inside the triangle, center ones need
        // not), so reusing the declared element would be a silent wrong
        // value.
        let bary = |e: &SigElement| e.kind == 28;
        let clash = self.s.inputs.iter().any(|e| {
            bary(e) && e.usage != 0 && e.interp != mode && linear(e.interp) == linear(mode)
        });
        if clash {
            return Val::Missing(if linear(mode) {
                "barycentrics of two linear modes (SV_Barycentrics allows one)"
            } else {
                "barycentrics of two perspective modes (SV_Barycentrics allows one)"
            });
        }
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    /// A component of a compute thread or group id, read in the entry block.
    pub(super) fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    /// Component `c` of the console thread id in a group the host declares
    /// as one row ([`Lower::flat_group`]): from the flat id, x fastest, as
    /// [`Self::thread_id`] makes it with console lanes (whose flat id is
    /// the relabelled one). The host's id in its row is the flat id.
    fn flat_thread_id(&mut self, c: u8) -> V {
        let [x, y, _] = self.flat_group.expect("a group declared as one row");
        let f = if self.console_lanes {
            self.console_flat()
        } else {
            match self.compute_id("threadIdInGroup.i32", 95, 0) {
                Val::I32(f) => f,
                _ => unreachable!(),
            }
        };
        let e = self.s.entry;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        }
    }

    pub(super) fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    pub(super) fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    pub(super) fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    /// A user-data dword: a root parameter's value.
    pub(super) fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        // A spilled signature starts at table byte zero. Bind only the
        // reads the program needs, without reserving its absent user SGPRs.
        let logical = u32::from(rp.user_data) + dword;
        let index = if self.x.res.spill_loads.is_empty() {
            logical
        } else {
            self.user_reads
                .get(&logical)
                .map_or(self.user_reads.len() as u32, |u| u.index)
        };
        self.user_reads.insert(
            logical,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    /// Ported from fxceless's VIctory fork #646 (head 170c24269) by mynzjerko,
    /// après VICTORY_THREAD_DIMS de Yorivel (2521646b8).
    /// Les threads de queue participent aux barrières. Les effets et les
    /// votes utilisent ce prédicat, y compris dans les groupes fractionnés.
    pub(super) fn dispatch_threads(&mut self) {
        let StageIo::Cs(cs) = self.x.native.io else {
            return;
        };
        let e = self.s.entry;
        for (axis, dim) in cs
            .threads
            .into_iter()
            .enumerate()
            .filter(|&(_, dim)| dim > 1)
        {
            let limit = self.host_word(crate::emit::DISPATCH_THREADS, 0, axis as u32);
            let group = self.group_id(axis as u8);
            let Val::I32(thread) = self.input(Input::ThreadIdInGroup(axis as u8)) else {
                unreachable!()
            };
            let zero = self.s.m.i32(0);
            let size = self.s.m.i32(dim);
            let whole = self.s.m.bin(e, Bin::UDiv, limit, size);
            let tail = self.s.m.bin(e, Bin::URem, limit, size);
            let full_group = self.s.m.cmp(e, pred::IULT, group, whole);
            let last_group = self.s.m.cmp(e, pred::EQ, group, whole);
            let in_tail = self.s.m.cmp(e, pred::IULT, thread, tail);
            let in_tail = self.s.m.bin(e, Bin::And, last_group, in_tail);
            let enabled = self.s.m.bin(e, Bin::Or, full_group, in_tail);
            let unlimited = self.s.m.cmp(e, pred::EQ, limit, zero);
            let enabled = self.s.m.bin(e, Bin::Or, unlimited, enabled);
            self.partial_active = Some(self.with_partial(e, enabled));
        }
    }

    pub(super) fn with_partial(&mut self, at: usize, bit: V) -> V {
        self.partial_active
            .map_or(bit, |active| self.s.m.bin(at, Bin::And, active, bit))
    }

    /// Les dwords fournis par l'hôte suivent les arguments du titre.
    pub(super) fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        if let Some((_, v)) = self
            .host_words
            .iter()
            .find(|(u, _)| (u.root, u.slot, u.dword) == (root, slot, dword))
        {
            return *v;
        }
        let index = self.host_words.len() as u32;
        if index.is_multiple_of(4) {
            self.host_rows.push(self.s.m.pending_i32());
        }
        let row = self.host_rows[index as usize / 4];
        let h = *self
            .user_data
            .get_or_insert_with(|| self.s.user_data_handle());
        let e = self.s.entry;
        let values = self.s.op(
            e,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        let v = self.s.m.extract(e, values, index % 4);
        self.host_words.push((
            UserData {
                index,
                slot,
                root,
                dword,
            },
            v,
        ));
        v
    }

    pub(super) fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let clamp = interp_clamp(inst)?;
        if let Interp::Interpolated { first, .. } = interp {
            // The host evaluates the pair from this second half: what a
            // first half's modifiers would do to its partial result is lost.
            for &(b, i) in first {
                if interp_clamp(&self.x.p.blocks[b].insts[i])? {
                    return Err("a clamped first half of an interpolation the host evaluates");
                }
            }
        }
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => (v.dst.ok_or("an interpolation")?, v.src.get(1).copied()),
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        let (attribute, eval) = match *interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        base: InterpBase::P0,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (
                    attribute,
                    Eval::Weights {
                        base: InterpBase::Zero,
                        i,
                        j,
                    },
                )
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        base: InterpBase::P0,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                let base = InterpBase::Accumulator(float(self, old)?);
                let j = Some(float(self, src)?);
                (attribute, Eval::Weights { base, i: None, j })
            }
        };
        // (pack2, the second row, whether the metadata names it).
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| -> Result<_, Why> {
                    let b = if i.pack2 {
                        half_input_partner(&self.x.native.io, &self.s.inputs, i)?
                    } else {
                        None
                    };
                    Ok((i.pack2, b, i.pack2 && i.row2 != i.row))
                })
                .transpose()?,
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            // Loaded as is: the bits the host interpolated.
            None => self.eval_channel(attribute.row, attribute.channel, eval)?,
            Some((pack2, b, explicit_second_row)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 && f16 {
                        packed_input_half(&l.s.inputs, attribute.row, b, h, explicit_second_row)?
                    } else if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => {
                            let x = l.eval_channel(row, ch, eval)?;
                            Ok(Some(l.as_float(x)))
                        }
                        // shortcut: a 16-bit read of the padding after a
                        // second row the metadata names reads zero. The
                        // host declares no element there; the console's
                        // half holds whatever the producing stage exported
                        // in it.
                        None if f16 && explicit_second_row => Ok(Some(l.s.m.f32(0))),
                        None => Ok(None),
                    }
                };
                let to_f16 = |l: &mut Self, f: V| {
                    let i32t = l.s.t.i32;
                    l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                // The clamp of a VOP3 form, to [0, 1] in the result's format:
                // saturating the f32 first is the same for an f16 result.
                let clamped = |l: &mut Self, f: V| if clamp { l.unary(7, f) } else { f };
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values (a first half
                    // evaluated on its own).
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let f = clamped(self, f);
                    self.as_int(f)
                } else if f16 {
                    let f = half(self, h)?.ok_or("a packed input half no element holds")?;
                    let f = clamped(self, f);
                    let lo = to_f16(self, f);
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { base, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = match base {
            InterpBase::P0 => Some(v0),
            InterpBase::Accumulator(v) => Some(v),
            InterpBase::Zero => None,
        };
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    pub(super) fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = match inst.read(Reg::V(reg)).map(|o| self.i32_of(o)) {
            Some(Ok(o)) => o,
            _ => self.s.m.i32(0),
        };
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }
}

/// Whether an interpolation clamps its result: the one modifier of the
/// VOP3 forms (`v_interp_p1ll_f16`, `v_interp_p2_f16`) the lowering
/// applies. Operand modifiers, an output modifier and op_sel (which
/// selects the destination's high half among others) are refused; the
/// VINTRP forms have none.
fn interp_clamp(inst: &Inst) -> Result<bool, Why> {
    let Fields::Valu(v) = &inst.fields else {
        return Ok(false);
    };
    if v.abs != 0 || v.neg != 0 {
        return Err("interpolation operand modifiers");
    }
    if v.omod != 0 {
        return Err("an interpolation with an output modifier");
    }
    if v.opsel != 0 {
        return Err("an interpolation with op_sel");
    }
    Ok(v.clamp)
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

/// Whether DXIL interpolation mode `mode` is a linear (noperspective) one.
fn linear(mode: u8) -> bool {
    matches!(mode, 4 | 5 | 7)
}

#[cfg(test)]
mod tests {
    use super::{half_input_partners, packed_input_half};
    use crate::emit::container::SigElement;
    use crate::native::{PsInput, StageIo, Usage};

    /// A 16-bit TEXCOORD element of `cols` channels at `row`.
    fn element(row: u32, cols: u8) -> SigElement {
        SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: true,
        }
    }

    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            input_addr: 0,
            rsrc2: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(|r| element(r, 4)).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        // The row 5 names is the one the rule gives.
        assert_eq!(p.get(&5), Some(&6));
    }

    /// Pixel inputs pair their packed rows by the vertex outputs' rule
    /// (`half_outputs`), whose metadata names no second row: where the
    /// pixel metadata names one the rule does not give, the input is
    /// refused rather than read from a row the vertex program did not
    /// pack there (review of VIctory #592).
    #[test]
    fn packed_rows_pair_alike_in_vertex_outputs_and_pixel_inputs() {
        use super::half_input_partner;
        use crate::emit::lower::export::half_outputs;
        use crate::native::VsOutput;
        // 16-bit rows 0-4; rows 0 and 2 pack.
        let sig: Vec<_> = (0..5).map(|r| element(r, 4)).collect();
        let output = |row| VsOutput {
            usage: Usage::Generic,
            row,
            param: row,
            fp16: true,
            pack2: true,
        };
        let vs = StageIo::Vs {
            inputs: vec![],
            outputs: vec![output(0), output(2)],
        };
        let out = half_outputs(&vs, &sig);
        assert_eq!(out.get(&0), Some(&(true, Some(1))));
        assert_eq!(out.get(&2), Some(&(true, Some(3))));
        let input = |row, row2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2: true,
            row2,
        };
        let ps = |inputs| StageIo::Ps {
            inputs,
            input_addr: 0,
            rsrc2: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        // Second rows left out, or named as the rule pairs them: the
        // outputs' pairs.
        for row2 in [[0, 2], [1, 2], [0, 3], [1, 3]] {
            let entries = vec![input(0, row2[0]), input(2, row2[1])];
            let io = ps(entries.clone());
            for (e, want) in entries.iter().zip([1, 3]) {
                assert_eq!(half_input_partner(&io, &sig, e), Ok(Some(want)));
            }
        }
        // Row 2 naming row 1, which the vertex program packs with row 0:
        // refused. Row 0 still pairs with row 1, as the outputs do (the
        // rule that left the named rows out paired it with row 3).
        let entries = vec![input(0, 0), input(2, 1)];
        let io = ps(entries.clone());
        assert_eq!(half_input_partner(&io, &sig, &entries[0]), Ok(Some(1)));
        let e = half_input_partner(&io, &sig, &entries[1]).unwrap_err();
        assert!(e.contains("second row"), "{e}");
    }

    #[test]
    fn explicit_packed_input_tail_is_padding() {
        // Four halves in row 6 followed by only two in its explicit partner
        // row 7. The next two reads address the packing's zero padding.
        let sig = vec![element(6, 4), element(7, 2)];
        assert_eq!(
            packed_input_half(&sig, 6, Some(7), 5, true),
            Ok(Some((7, 1)))
        );
        assert_eq!(packed_input_half(&sig, 6, Some(7), 6, true), Ok(None));
        assert_eq!(packed_input_half(&sig, 6, Some(7), 7, true), Ok(None));
        // With an inferred partner the same gap is not assumed to be zero.
        assert!(packed_input_half(&sig, 6, Some(7), 6, false).is_err());
        assert!(packed_input_half(&sig, 6, None, 4, false).is_err());
    }

    /// A pixel program interpolating the high half of channel z or w of a
    /// 16-bit input packing rows 0 (four halves) and 1 (two), whose
    /// metadata names the second row (`row2`) or leaves it out: the
    /// translation, validated where `dxil.dll` is found, or the error.
    fn packed_read(high_channel: u8, row2: u8) -> Result<(), String> {
        interp16(high_channel, row2, 0, [0, 0], false).map(|_| ())
    }

    /// [`packed_read`]'s program with `p1` OR-ed into the first half's
    /// first word and `p2` into the second half's words, and j computed
    /// (`v_mul_f32 v3, 0.5, v1`) when `computed`: the translation's module
    /// summary, or the error.
    fn interp16(
        high_channel: u8,
        row2: u8,
        p1: u32,
        p2: [u32; 2],
        computed: bool,
    ) -> Result<String, String> {
        use crate::dxbc::{self, Container, FourCC};
        use crate::emit::{container, emit, validator::Validator};
        use crate::native::{HwStage, NativeProgram};
        use crate::{io, ir, resource, structure};
        // src0 of the VOP3 interpolations: attr0, the channel, high half.
        let attr = u32::from(high_channel) << 6 | 1 << 8;
        let j = if computed { 259 } else { 257 };
        let mut words = Vec::new();
        if computed {
            words.push(0x1006_02F0); // v_mul_f32 v3, 0.5, v1
        }
        words.extend([
            0xD742_0002 | p1, // v_interp_p1ll_f16 v2, v0, attr0.<c>.high
            attr | 256 << 9,
            0xD75A_0002 | p2[0], // v_interp_p2_f16 v2, attr0.<c>.high, v<j>, v2
            attr | j << 9 | 258 << 18 | p2[1],
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ]);
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            float_mode: 0,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0xF,
                    attr: 0,
                    flat: false,
                    fp16: true,
                    pack2: true,
                    row2,
                }],
                // PERSP_CENTER: i and j in v0 and v1.
                input_addr: 1 << 1,
                rsrc2: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        // The console's signature: 16-bit floats (component type 6).
        let mut isg1 = container::signature(&[element(0, 4), element(1, 2)], true);
        for k in 0..2 {
            isg1[8 + 32 * k + 16..][..4].copy_from_slice(&6u32.to_le_bytes());
        }
        let target = SigElement {
            name: "SV_Target".into(),
            kind: 16,
            row: Some(0),
            usage: 15,
            half: false,
            ..element(0, 4)
        };
        let osg1 = container::signature(&[target], false);
        let bytes = dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).map_err(|e| e.to_string())?;
        let summary = crate::emit::last_module_summary();
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        Ok(summary)
    }

    /// VOP3 interpolations: a clamp on the second half saturates the result,
    /// interpolated by the host or from computed barycentrics; one on a
    /// first half the host's interpolation replaces is refused, as are an
    /// output modifier and op_sel (the destination's high half), which the
    /// lowering does not apply (review of VIctory #453).
    #[test]
    fn vop3_interpolation_modifiers_are_applied_or_refused() {
        const CLAMP: u32 = 1 << 15;
        const OMOD_MUL2: u32 = 1 << 27;
        const DST_HIGH: u32 = 8 << 11;
        for computed in [false, true] {
            let plain = interp16(2, 0, 0, [0, 0], computed).unwrap();
            assert!(!plain.contains("unary"), "{plain}");
            let clamped = interp16(2, 0, 0, [CLAMP, 0], computed).unwrap();
            assert!(clamped.contains("unary"), "{clamped}");
            // A first half evaluated on its own clamps its own result.
            let first = interp16(2, 0, CLAMP, [0, 0], computed);
            if computed {
                assert!(first.unwrap().contains("unary"));
            } else {
                let e = first.unwrap_err();
                assert!(e.contains("a clamped first half"), "{e}");
            }
            for (p2, why) in [
                ([0, OMOD_MUL2], "an output modifier"),
                ([DST_HIGH, 0], "op_sel"),
            ] {
                let e = interp16(2, 0, 0, p2, computed).unwrap_err();
                assert!(e.contains(why), "{e}");
            }
        }
    }

    #[test]
    fn a_packed_pixel_input_reads_the_second_row_the_metadata_leaves_out() {
        // Half 5 is channel 1 of row 1, which no input entry names.
        packed_read(2, 0).unwrap();
        // Half 7 is past row 1's two halves: with a guessed second row it
        // is no padding the metadata describes.
        let e = packed_read(3, 0).unwrap_err();
        assert!(e.contains("a packed input half no element holds"), "{e}");
        // With the second row named, it is padding.
        packed_read(2, 1).unwrap();
        packed_read(3, 1).unwrap();
    }
}
