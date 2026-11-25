//! Merged ES + GS programs (`lower/geometry.rs`, `docs/analysis/es-gs.md`).
//! Hand-assembled GFX10 programs shaped as the title's: an export shader
//! writing its ring item at its lane after its wave's base, a geometry
//! shader reading its input vertices at the offsets of v0 and v1 and
//! writing its vertices at its lane after a base it loads, then GS EMIT,
//! CUT and GS_DONE. What the plans resolve and refuse, on straight and
//! branched programs; then geometry shaders made through the lowering's
//! hooks (a small walker stands in for the lowering, which W10's second
//! phase wires in) drawn on the test adapter (WARP on this PC) with their
//! stream 0 streamed out, and every dword of every vertex read back.

use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Interface, PCSTR};

use super::bitcode::{Attr, Term, V};
use super::container::SigElement;
use super::host::Host;
use super::lower::geometry::{
    self, EsPlan, GsPlan, GsShape, Message, Primitive, RingAccess, Slot, Topology, Vertices,
};
use super::shader::Shader;
use super::{HostShader, container};
use crate::dxbc::{self, Container, FourCC};
use crate::gfx10::{Fields, Operand};
use crate::ir::{self, Exit, Inst, Program, Reg};
use crate::native::{ExportIo, GeometryIo, HwStage, NativeProgram, StageIo, Usage, VsOutput};
use crate::resource::Site;

// GFX10 encodings (AMD's RDNA ISA reference).
const LITERAL: u32 = 0xFF;
const SDWA: u32 = 0xF9;
const MINUS_ONE: u32 = 0xC1;
const M0: u8 = 124;
const S_MOV_B32: u32 = 0x03;
const S_SETPC_B64: u32 = 0x20;
const S_ADD_U32: u32 = 0x00;
const S_SUB_U32: u32 = 0x01;
const S_LSHL_B32: u32 = 0x1E;
const S_MUL_I32: u32 = 0x26;
const S_BFE_U32: u32 = 0x27;
const S_CMP_EQ_U32: u32 = 0x06;
const S_BFE_I32: u32 = 0x28;
const S_BFM_B64: u32 = 0x25;
const S_CSELECT_B32: u32 = 0x0A;
const S_CSELECT_B64: u32 = 0x0B;
const S_ENDPGM: u32 = 0xBF81_0000;
const S_BRANCH: u32 = 0x02;
const S_CBRANCH_SCC0: u32 = 0x04;
const S_CBRANCH_SCC1: u32 = 0x05;
const S_BARRIER: u32 = 0xBF8A_0000;
const V_MOV_B32: u32 = 0x01;
const V_LSHRREV_B32: u32 = 0x16;
const V_LSHLREV_B32: u32 = 0x1A;
const V_AND_B32: u32 = 0x1B;
const V_MAD_U32_U24: u32 = 0x143;
const V_MBCNT_LO: u32 = 0x365;
const V_MBCNT_HI: u32 = 0x366;
const DS_WRITE_B32: u32 = 0x0D;
const DS_WRITE2_B32: u32 = 0x0E;
const DS_WRITE2_B64: u32 = 0x4E;
const DS_READ_B32: u32 = 0x36;
const DS_READ2_B32: u32 = 0x37;
const DS_READ_B64: u32 = 0x76;
/// `s_sendmsg` GS EMIT, GS CUT and GS EMIT_CUT on stream 0, and GS_DONE.
const EMIT: u32 = 0xBF90_0022;
const CUT: u32 = 0xBF90_0012;
const EMIT_CUT: u32 = 0xBF90_0032;
const DONE: u32 = 0xBF90_0003;
/// GS_DONE carrying a last EMIT, CUT or EMIT_CUT (its operation in bits
/// 4-5, as GS's).
const DONE_EMIT: u32 = DONE | 2 << 4;
const DONE_CUT: u32 = DONE | 1 << 4;
const DONE_EMIT_CUT: u32 = DONE | 3 << 4;
const S_AND_SAVEEXEC_B64: u32 = 0x24;
const S_OR_B64: u32 = 0x11;
const EXEC: u8 = 126;
/// VCC's operand code (its low dword; a 64-bit operand takes the pair).
const VCC: u32 = 106;

/// A VGPR as a 9-bit source operand.
fn vgpr(n: u8) -> u32 {
    256 + u32::from(n)
}

/// An inline integer constant (0 to 64).
fn int(n: u32) -> u32 {
    128 + n
}

fn sop1(op: u32, sdst: u8, src: u32) -> u32 {
    0xBE80_0000 | u32::from(sdst) << 16 | op << 8 | src
}

fn sop2(op: u32, sdst: u8, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | u32::from(sdst) << 16 | src1 << 8 | src0
}

fn sopc(op: u32, src0: u32, src1: u32) -> u32 {
    0xBF00_0000 | op << 16 | src1 << 8 | src0
}

fn sopp(op: u32, imm: u16) -> u32 {
    0xBF80_0000 | op << 16 | u32::from(imm)
}

/// `s_load_dword sdst, s[sbase:sbase+1], offset`.
fn s_load_dword(sdst: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | u32::from(sdst) << 6 | u32::from(sbase >> 1),
        0xFA00_0000 | offset,
    ]
}

fn vop1(op: u32, vdst: u8, src0: u32) -> u32 {
    0x7E00_0000 | u32::from(vdst) << 17 | op << 9 | src0
}

fn vop2(op: u32, vdst: u8, src0: u32, vsrc1: u8) -> u32 {
    op << 25 | u32::from(vdst) << 17 | u32::from(vsrc1) << 9 | src0
}

fn vop3(op: u32, vdst: u8, src0: u32, src1: u32, src2: u32) -> [u32; 2] {
    [
        0xD400_0000 | op << 16 | u32::from(vdst),
        src2 << 18 | src1 << 9 | src0,
    ]
}

fn ds(op: u32, vdst: u8, addr: u8, data: [u8; 2], offset0: u8, offset1: u8) -> [u32; 2] {
    [
        0xD800_0000 | op << 18 | u32::from(offset1) << 8 | u32::from(offset0),
        u32::from(vdst) << 24
            | u32::from(data[1]) << 16
            | u32::from(data[0]) << 8
            | u32::from(addr),
    ]
}

/// A one-address `ds_*` with a 16-bit byte offset.
fn ds_at(op: u32, vdst: u8, addr: u8, data: u8, offset: u16) -> [u32; 2] {
    ds(op, vdst, addr, [data, 0], offset as u8, (offset >> 8) as u8)
}

/// Code with forward branches to labels.
#[derive(Default)]
struct Code {
    words: Vec<u32>,
    branches: Vec<(usize, &'static str)>,
    labels: HashMap<&'static str, usize>,
}

impl Code {
    fn add(&mut self, words: impl IntoIterator<Item = u32>) {
        self.words.extend(words);
    }

    fn branch(&mut self, op: u32, to: &'static str) {
        self.branches.push((self.words.len(), to));
        self.words.push(sopp(op, 0));
    }

    fn label(&mut self, name: &'static str) {
        self.labels.insert(name, self.words.len());
    }

    /// The words, each branch to its label (before or after it).
    fn resolved(mut self) -> Vec<u32> {
        for &(at, to) in &self.branches {
            let skip = self.labels[to] as i32 - at as i32 - 1;
            self.words[at] |= u32::from(skip as i16 as u16);
        }
        self.words
    }

    fn program(self) -> Program {
        program_of(&self.resolved())
    }
}

/// The wave64 program of the code `words`.
fn program_of(words: &[u32]) -> Program {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&bytes, true, 64).expect("a program the IR builds")
}

/// The GS vertices' dwords in these tests: four rows (position, a uint,
/// four floats, a uint).
const VERTEX_DWORDS: u32 = 16;

/// A geometry shader's address set-up and input reads, as the title's
/// programs do them: input vertex k's item at four times its 16-bit offset
/// (v10 from v0's low half, masked; v11 from v0's high half, shifted; v12
/// from v1's low half through an SDWA select), its vertices at
/// `lane × item_bytes` after a base loaded from user data 0-1 times 256
/// plus s2, M0 the wave id of s3. Vertex k's item dwords 0-8 land in
/// v[20 + 10k ..].
fn gs_prelude(code: &mut Code, item_bytes: u32) {
    code.add([
        sop1(S_MOV_B32, 30, int(2)),
        vop2(V_LSHLREV_B32, 10, int(2), 0),
        vop2(V_AND_B32, 10, LITERAL, 10),
        0x3_FFFC,
        vop2(V_LSHRREV_B32, 11, int(16), 0),
        vop2(V_LSHLREV_B32, 11, int(2), 11),
        // v_lshlrev_b32 v12, s30, v1 src1_sel:WORD_0
        vop2(V_LSHLREV_B32, 12, SDWA, 1),
        30 | 6 << 8 | 6 << 16 | 1 << 23 | 4 << 24,
    ]);
    code.add(s_load_dword(20, 8, 0x40));
    code.add([sop2(S_LSHL_B32, 20, 20, int(8)), sop2(S_ADD_U32, 20, 20, 2)]);
    code.add(vop3(V_MBCNT_LO, 13, MINUS_ONE, int(0), 0));
    code.add(vop3(V_MBCNT_HI, 13, MINUS_ONE, vgpr(13), 0));
    code.add(vop3(V_MAD_U32_U24, 13, LITERAL, vgpr(13), 20));
    code.add([item_bytes]);
    code.add([
        sop2(S_BFE_U32, 21, 3, LITERAL),
        0x8_0010,
        sop1(S_MOV_B32, M0, 21),
    ]);
    for k in 0..3 {
        let (a, b) = (10 + k, 20 + 10 * k);
        code.add(ds(DS_READ2_B32, b, a, [0, 0], 0, 1));
        code.add(ds(DS_READ2_B32, b + 2, a, [0, 0], 2, 3));
        code.add(ds_at(DS_READ_B64, b + 4, a, 0, 16));
        code.add(ds(DS_READ2_B32, b + 6, a, [0, 0], 6, 7));
        code.add(ds_at(DS_READ_B32, b + 8, a, 0, 32));
    }
}

/// The instructions [`gs_prelude`] has before its reads: all address
/// set-up.
const PRELUDE_SETUP: usize = 14;

/// Writes vertex `n` from input vertex `k`: its position, `100 + n`, its
/// four floats and its uint.
fn gs_vertex(code: &mut Code, n: u32, k: u8) {
    gs_vertex_with(code, n, k, false);
}

/// [`gs_vertex`]; with `fill`, `100 + n` goes in with the rest of its
/// short row (`ds_write2_b64` of v[60:61] and v[2:3]): dwords 5 to 7, which
/// no output element holds, then get v61 (never set) and the GS half's
/// system VGPRs 2 and 3, which have no host value.
fn gs_vertex_with(code: &mut Code, n: u32, k: u8, fill: bool) {
    let (b, d) = (20 + 10 * k, (VERTEX_DWORDS * n) as u8);
    code.add([vop1(V_MOV_B32, 60, LITERAL), 100 + n]);
    code.add(ds(DS_WRITE2_B32, 0, 13, [b, b + 1], d, d + 1));
    code.add(ds(DS_WRITE2_B32, 0, 13, [b + 2, b + 3], d + 2, d + 3));
    if fill {
        code.add(ds(DS_WRITE2_B64, 0, 13, [60, 2], d / 2 + 2, d / 2 + 3));
    } else {
        code.add(ds_at(DS_WRITE_B32, 0, 13, 60, 4 * (u16::from(d) + 4)));
    }
    code.add(ds(
        DS_WRITE2_B64,
        0,
        13,
        [b + 4, b + 6],
        d / 2 + 4,
        d / 2 + 5,
    ));
    code.add(ds_at(DS_WRITE_B32, 0, 13, b + 8, 4 * (u16::from(d) + 12)));
}

#[derive(Debug, Clone, Copy)]
enum Step {
    /// Writes the next vertex from this input vertex, then GS EMIT.
    Emit(u8),
    /// The same with GS EMIT_CUT.
    EmitCut(u8),
    /// The same with GS_DONE carrying the emit.
    EmitDone(u8),
    Cut,
    Done,
}

fn gs_program(steps: &[Step], max_vertices: u32) -> Program {
    let mut code = Code::default();
    gs_code(&mut code, steps, max_vertices);
    code.program()
}

/// [`gs_program`]'s code, after what `code` holds.
fn gs_code(code: &mut Code, steps: &[Step], max_vertices: u32) {
    gs_code_with(code, steps, max_vertices, false);
}

/// [`gs_code`], each vertex written as [`gs_vertex_with`] `fill` writes it.
fn gs_code_with(code: &mut Code, steps: &[Step], max_vertices: u32, fill: bool) {
    gs_prelude(code, max_vertices * VERTEX_DWORDS * 4);
    let mut n = 0;
    for &step in steps {
        match step {
            Step::Emit(k) | Step::EmitCut(k) | Step::EmitDone(k) => {
                gs_vertex_with(code, n, k, fill);
                code.add([match step {
                    Step::Emit(_) => EMIT,
                    Step::EmitCut(_) => EMIT_CUT,
                    _ => DONE_EMIT,
                }]);
                n += 1;
            }
            Step::Cut => code.add([CUT]),
            Step::Done => code.add([DONE]),
        }
    }
    code.add([S_ENDPGM]);
}

fn shape(max_vertices: u32) -> GsShape {
    GsShape {
        input: Primitive::Triangle,
        topology: Topology::TriangleStrip,
        max_vertices,
        vertex_dwords: VERTEX_DWORDS,
        item_dwords: max_vertices * VERTEX_DWORDS,
        instances: 1,
    }
}

fn plan(steps: &[Step], max_vertices: u32) -> Result<GsPlan, &'static str> {
    geometry::gs_plan(&gs_program(steps, max_vertices), &shape(max_vertices))
}

fn el(name: &str, index: u32, kind: u8, comp: u8, interp: u8, row: u32, cols: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp,
        interp,
        row: Some(row),
        col: 0,
        cols,
        usage: (1 << cols) - 1,
        half: false,
    }
}

/// The ES ring item, as the ES's output signature and the GS's input one:
/// a position, four floats, a uint.
fn item() -> Vec<SigElement> {
    vec![
        el("SV_Position", 0, 3, 9, 4, 0, 4),
        el("TEXCOORD", 0, 0, 9, 2, 1, 4),
        el("TEXCOORD", 1, 0, 5, 1, 2, 1),
    ]
}

/// The GS's stream 0, as [`gs_vertex`] writes it.
fn outputs() -> Vec<SigElement> {
    vec![
        el("SV_Position", 0, 3, 9, 4, 0, 4),
        el("TEXCOORD", 0, 0, 5, 1, 1, 1),
        el("TEXCOORD", 1, 0, 9, 2, 2, 4),
        el("TEXCOORD", 2, 0, 5, 1, 3, 1),
    ]
}

fn slot(vertex: u32, dword: u32) -> Slot {
    Slot { vertex, dword }
}

fn emit(vertex: u32, cut: bool, done: bool) -> Message {
    Message::Emit { vertex, cut, done }
}

fn flat(accesses: &[RingAccess]) -> Vec<(Reg, Slot)> {
    accesses.iter().flat_map(|a| a.dwords.clone()).collect()
}

/// The ring dwords [`gs_vertex`] writes for vertex `n` from input vertex
/// `k`.
fn written(n: u32, k: u8) -> Vec<(Reg, Slot)> {
    let b = 20 + 10 * k;
    let at = |r: u8, d: u32| (Reg::V(r), slot(n, d));
    vec![
        at(b, 0),
        at(b + 1, 1),
        at(b + 2, 2),
        at(b + 3, 3),
        at(60, 4),
        at(b + 4, 8),
        at(b + 5, 9),
        at(b + 6, 10),
        at(b + 7, 11),
        at(b + 8, 12),
    ]
}

/// Every ring access of the title's form resolves: the input vertices from
/// their halves of v0 and v1 (masked, shifted, or selected), the vertices
/// made from the lane and the loaded base; emits are numbered in order; the
/// address set-up, the base's load among it, is what the lowering leaves
/// out.
#[test]
fn geometry_plan_resolves_the_rings_and_numbers_the_emits() {
    let steps = [
        Step::Emit(2),
        Step::Emit(0),
        Step::Emit(1),
        Step::Cut,
        Step::Done,
    ];
    let plan = plan(&steps, 3).expect("a plan");
    let mut reads = Vec::new();
    for k in 0..3u32 {
        let b = 20 + 10 * k as u8;
        reads.extend((0..9).map(|d| (Reg::V(b + d as u8), slot(k, d))));
    }
    assert_eq!(flat(&plan.reads), reads);
    let mut writes = written(0, 2);
    writes.extend(written(1, 0));
    writes.extend(written(2, 1));
    assert_eq!(flat(&plan.writes), writes);
    let messages: Vec<Message> = plan.messages.iter().map(|m| m.1).collect();
    assert_eq!(
        messages,
        [
            emit(0, false, false),
            emit(1, false, false),
            emit(2, false, false),
            Message::Cut { done: false },
            Message::Done,
        ]
    );
    assert_eq!(plan.emitted, 3);
    // Each emit lists its vertex's ten dwords, from the writes before it.
    for (n, (site, _)) in plan.messages.iter().take(3).enumerate() {
        let dwords = &plan.emits[site];
        let w: Vec<Slot> = plan.writes[5 * n..5 * n + 5]
            .iter()
            .flat_map(|a| a.dwords.iter().map(|d| d.1))
            .collect();
        let listed: Vec<Slot> = dwords.iter().map(|d| slot(n as u32, d.dword)).collect();
        let mut sorted = w.clone();
        sorted.sort();
        assert_eq!(listed, sorted, "vertex {n}");
        assert!(dwords.iter().all(|d| d.site < *site && d.site.0 == site.0));
    }
    let setup: BTreeSet<Site> = (0..PRELUDE_SETUP).map(|i| (0, i)).collect();
    assert_eq!(plan.address_only, setup);
}

/// An export shader of the title's form: its item at `lane × 48` after
/// the wave's base (`wave × 64 × 48`), position and four floats as
/// constants, then its vertex id (v5); the barrier, then the indirect jump
/// to the GS half.
fn es_code() -> Code {
    let mut code = Code::default();
    code.add([sop2(S_BFE_U32, 10, 3, LITERAL), 0x4_0018]);
    code.add([sop2(S_MUL_I32, 10, 10, LITERAL), 64 * 48]);
    code.add(vop3(V_MBCNT_LO, 9, MINUS_ONE, int(0), 0));
    code.add(vop3(V_MBCNT_HI, 9, MINUS_ONE, vgpr(9), 0));
    code.add(vop3(V_MAD_U32_U24, 15, vgpr(9), int(48), 10));
    code
}

/// The ES's constant dwords 0-7: a position, then four floats.
const ES_CONSTANTS: [f32; 8] = [0.5, 0.25, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0];

fn es_tail(code: &mut Code) {
    es_tail_with(code, false);
}

/// [`es_tail`]; with `fill`, the vertex id goes in with the rest of its
/// short last row (`ds_write2_b64` of v[5:6] and v[0:1]): item dwords 9 to
/// 11, which no output element holds, then get v6 (never set) and the GS
/// half's system VGPRs 0 and 1, which have no host value in the ES.
fn es_tail_with(code: &mut Code, fill: bool) {
    for (k, c) in ES_CONSTANTS.iter().enumerate().skip(4) {
        code.add([vop1(V_MOV_B32, 16 + k as u8, LITERAL), c.to_bits()]);
    }
    code.add(ds(DS_WRITE2_B32, 0, 15, [20, 21], 4, 5));
    code.add(ds(DS_WRITE2_B32, 0, 15, [22, 23], 6, 7));
    let id = crate::io::ES_VERTEX_ID;
    if fill {
        code.add(ds(DS_WRITE2_B64, 0, 15, [id, 0], 4, 5));
    } else {
        code.add(ds_at(DS_WRITE_B32, 0, 15, id, 32));
    }
    code.add([S_BARRIER, sop1(S_SETPC_B64, 0, 6)]);
}

fn es_position(code: &mut Code, position: [f32; 4]) {
    for (k, c) in position.iter().enumerate() {
        code.add([vop1(V_MOV_B32, 16 + k as u8, LITERAL), c.to_bits()]);
    }
    code.add(ds(DS_WRITE2_B64, 0, 15, [16, 18], 0, 1));
}

fn es_program() -> Program {
    let mut code = es_code();
    es_position(&mut code, ES_CONSTANTS[..4].try_into().unwrap());
    es_tail(&mut code);
    code.program()
}

fn es_item(write: &[(u8, u32)]) -> Vec<(Reg, Slot)> {
    write
        .iter()
        .map(|&(r, d)| (Reg::V(r), slot(0, d)))
        .collect()
}

/// The ES's item resolves from the lane and the wave's base: its stride,
/// each dword's place, and the address set-up the lowering leaves out.
#[test]
fn geometry_es_plan_resolves_the_item() {
    let p = es_program();
    let plan: EsPlan = geometry::es_plan(&p).expect("a plan");
    assert_eq!(plan.item_bytes, 48);
    assert_eq!(
        flat(&plan.writes),
        es_item(&[
            (16, 0),
            (17, 1),
            (18, 2),
            (19, 3),
            (20, 4),
            (21, 5),
            (22, 6),
            (23, 7),
            (5, 8),
        ])
    );
    let setup: BTreeSet<Site> = (0..5).map(|i| (0, i)).collect();
    assert_eq!(plan.address_only, setup);
}

/// A branched export shader (written for this test, after a review note:
/// an indirect jump must end its block): ring writes on both sides of a
/// branch and after the join, the barrier, then `s_setpc_b64`. The jump
/// ends the last block, with no successor and nothing after it (the GS
/// half is a program of its own here, not code that follows), and the plan
/// takes every write on every path.
#[test]
fn geometry_branched_es_ends_at_its_indirect_jump() {
    let mut code = es_code();
    code.add([sopc(S_CMP_EQ_U32, 12, int(0))]);
    code.branch(S_CBRANCH_SCC1, "else");
    es_position(&mut code, [1.0, 2.0, 3.0, 1.0]);
    code.branch(S_BRANCH, "join");
    code.label("else");
    es_position(&mut code, [4.0, 5.0, 6.0, 1.0]);
    code.label("join");
    es_tail(&mut code);
    let p = code.program();
    let jumps: Vec<usize> = (0..p.blocks.len())
        .filter(|&b| p.blocks[b].exit == Exit::Indirect)
        .collect();
    assert_eq!(jumps.len(), 1, "one indirect jump");
    let last = jumps[0];
    assert!(p.blocks[last].exit.successors().is_empty());
    assert_eq!(p.blocks[last].preds.len(), 2, "both sides join before it");
    assert_eq!(
        p.blocks[last].insts.last().map(Inst::name),
        Some("s_setpc_b64")
    );
    assert!(
        p.blocks.iter().all(|b| b.exit != Exit::End),
        "no end but the jump"
    );
    let plan = geometry::es_plan(&p).expect("a plan");
    assert_eq!(plan.item_bytes, 48);
    let sides: Vec<usize> = plan.writes.iter().map(|w| w.site.0).collect();
    assert_eq!(sides.len(), 5);
    assert!(sides[0] != sides[1] && sides[2..].iter().all(|&b| b == last));
    for w in &plan.writes[..2] {
        assert_eq!(
            w.dwords.iter().map(|d| d.1.dword).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }
    // The compare feeds the branch: not address set-up.
    assert_eq!(
        plan.address_only,
        (0..5).map(|i| (0, i)).collect::<BTreeSet<Site>>()
    );
}

/// A branched geometry shader (written for this test): vertex 0 written
/// and emitted on either side of a branch, then two more after the join.
/// Both emits make vertex 0, each from its own side's writes. Emitting on
/// one side only, or writing on both sides and emitting after the join,
/// is refused.
#[test]
fn geometry_branched_gs_emits_on_both_sides() {
    let branched = |then: Option<u8>, other: Option<u8>, emit_in_branches: bool| {
        let mut code = Code::default();
        gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
        code.add([sopc(S_CMP_EQ_U32, 12, int(0))]);
        code.branch(S_CBRANCH_SCC1, "else");
        if let Some(k) = then {
            gs_vertex(&mut code, 0, k);
            if emit_in_branches {
                code.add([EMIT]);
            }
        }
        code.branch(S_BRANCH, "join");
        code.label("else");
        if let Some(k) = other {
            gs_vertex(&mut code, 0, k);
            if emit_in_branches {
                code.add([EMIT]);
            }
        }
        code.label("join");
        if !emit_in_branches {
            code.add([EMIT]);
        }
        gs_vertex(&mut code, 1, 2);
        code.add([EMIT]);
        gs_vertex(&mut code, 2, 0);
        code.add([EMIT, CUT, DONE, S_ENDPGM]);
        geometry::gs_plan(&code.program(), &shape(3))
    };
    let plan = branched(Some(0), Some(1), true).expect("a plan");
    let emits: Vec<(Site, u32)> = plan
        .messages
        .iter()
        .filter_map(|&(s, m)| match m {
            Message::Emit { vertex, .. } => Some((s, vertex)),
            _ => None,
        })
        .collect();
    assert_eq!(emits.iter().map(|e| e.1).collect::<Vec<_>>(), [0, 0, 1, 2]);
    assert_ne!(emits[0].0.0, emits[1].0.0, "one emit on each side");
    for &(site, _) in &emits[..2] {
        let dwords = &plan.emits[&site];
        assert_eq!(dwords.len(), 10);
        assert!(
            dwords.iter().all(|d| d.site.0 == site.0),
            "its own side's writes"
        );
    }
    assert_eq!(plan.emitted, 3);
    assert_eq!(branched(Some(0), None, true), Err(geometry::PATH));
    assert_eq!(branched(Some(0), Some(1), false), Err(geometry::SOME_PATHS));
}

/// What the lowering cannot give as the console's copy shader would.
#[test]
fn geometry_plans_refuse_what_emits_cannot_give() {
    let three = [Step::Emit(0), Step::Emit(1), Step::Emit(2)];
    let with = |extra: &[u32], max: u32| {
        let mut code = Code::default();
        gs_prelude(&mut code, max * VERTEX_DWORDS * 4);
        for (n, k) in [0, 1, 2].into_iter().enumerate() {
            gs_vertex(&mut code, n as u32, k);
            code.add([EMIT]);
        }
        code.add(extra.iter().copied());
        code.add([S_ENDPGM]);
        geometry::gs_plan(&code.program(), &shape(max))
    };
    assert!(plan(&three, 3).is_ok());
    // A write to vertex 0 after its emit.
    let late = ds_at(DS_WRITE_B32, 0, 13, 60, 16);
    assert_eq!(with(&late, 3), Err(geometry::LATE));
    // An emit past the most vertices.
    assert_eq!(with(&[EMIT], 3), Err(geometry::TOO_MANY));
    // Emits and cuts on stream 1, a message after GS_DONE, and messages a
    // geometry shader does not send (GS NOP, the NGG output request).
    assert_eq!(with(&[EMIT | 1 << 8], 4), Err(geometry::STREAM));
    assert_eq!(with(&[CUT | 1 << 8], 4), Err(geometry::STREAM));
    assert_eq!(with(&[DONE, CUT], 3), Err(geometry::AFTER_DONE));
    assert_eq!(with(&[0xBF90_0002], 3), Err(geometry::MESSAGE));
    assert_eq!(with(&[0xBF90_0009], 3), Err(geometry::MESSAGE));
    // Vertex 0's uint written twice before its emit.
    let mut code = Code::default();
    gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
    code.add(ds_at(DS_WRITE_B32, 0, 13, 20, 16));
    gs_vertex(&mut code, 0, 0);
    code.add([EMIT, S_ENDPGM]);
    assert_eq!(
        geometry::gs_plan(&code.program(), &shape(3)),
        Err(geometry::TWICE)
    );
    // An LDS address of no ring (v2, the primitive id), a read of the GSVS
    // ring, and a ring item of two streams.
    let other = ds_at(DS_WRITE_B32, 0, 2, 60, 0);
    assert_eq!(with(&other, 3), Err(geometry::RING));
    let read_back = ds_at(DS_READ_B32, 61, 13, 0, 0);
    assert_eq!(with(&read_back, 3), Err(geometry::RING));
    let two_streams = GsShape {
        item_dwords: 2 * 3 * VERTEX_DWORDS,
        ..shape(3)
    };
    assert_eq!(
        geometry::gs_plan(&gs_program(&three, 3), &two_streams),
        Err(geometry::STREAMS)
    );
    // An ES that reads LDS, or writes outside its item.
    let mut code = es_code();
    code.add(ds_at(DS_READ_B32, 30, 15, 0, 0));
    code.add([S_BARRIER, sop1(S_SETPC_B64, 0, 6)]);
    assert_eq!(geometry::es_plan(&code.program()), Err(geometry::RING));
    let mut code = es_code();
    code.add(ds_at(DS_WRITE_B32, 0, 15, 5, 48));
    code.add([S_BARRIER, sop1(S_SETPC_B64, 0, 6)]);
    assert_eq!(geometry::es_plan(&code.program()), Err(geometry::RING));
}

/// A ring write may store its own address as data (Codex review of W10's
/// first phase): the address's set-up then computes a value the write
/// exports, and is no longer left out. In an ES, the item's address as
/// item dword 8; in a GS, the vertex's address as vertex 0's dword 4 (the
/// loaded base then feeds an output, and only the input vertices' set-up
/// and M0's stay out).
#[test]
fn geometry_plan_keeps_an_address_stored_as_data() {
    let mut code = es_code();
    es_position(&mut code, ES_CONSTANTS[..4].try_into().unwrap());
    code.add(ds_at(DS_WRITE_B32, 0, 15, 15, 32));
    code.add([S_BARRIER, sop1(S_SETPC_B64, 0, 6)]);
    let plan = geometry::es_plan(&code.program()).expect("a plan");
    assert_eq!(plan.item_bytes, 48);
    assert_eq!(
        plan.writes.last().map(|w| w.dwords.clone()),
        Some(es_item(&[(15, 8)]))
    );
    assert_eq!(plan.address_only, BTreeSet::new(), "the address is data");

    let mut code = Code::default();
    gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
    code.add(ds_at(DS_WRITE_B32, 0, 13, 13, 16));
    for (n, k) in [0, 1, 2].into_iter().enumerate() {
        let d = (VERTEX_DWORDS * n as u32) as u8;
        let b = 20 + 10 * k;
        code.add(ds(DS_WRITE2_B32, 0, 13, [b, b + 1], d, d + 1));
        code.add(ds(DS_WRITE2_B32, 0, 13, [b + 2, b + 3], d + 2, d + 3));
        code.add([EMIT]);
    }
    code.add([DONE, S_ENDPGM]);
    let plan = geometry::gs_plan(&code.program(), &shape(3)).expect("a plan");
    let first = &plan.emits[&plan.messages[0].0];
    assert!(
        first
            .iter()
            .any(|w| w.dword == 4 && w.reg == Reg::V(13) && w.site.1 == PRELUDE_SETUP + 15),
        "vertex 0's dword 4 is the address: {first:?}"
    );
    // Kept: the base's load, its shift and offset (6-8), the lane and the
    // address (9-11). Left out: the input vertices' offsets and M0.
    let setup: BTreeSet<Site> = [0, 1, 2, 3, 4, 5, 12, 13].map(|i| (0, i)).into();
    assert_eq!(plan.address_only, setup);
}

/// GS_DONE carries an operation as GS does (Codex review of W10's first
/// phase): a last emit, cut or emit-and-cut, which the plan numbers and
/// checks as the others, with nothing after it.
#[test]
fn geometry_gs_done_carries_its_last_emit() {
    let plan = plan(&[Step::Emit(0), Step::Emit(1), Step::EmitDone(2)], 3).expect("a plan");
    let messages: Vec<Message> = plan.messages.iter().map(|m| m.1).collect();
    assert_eq!(
        messages,
        [
            emit(0, false, false),
            emit(1, false, false),
            emit(2, false, true)
        ]
    );
    assert_eq!(plan.emitted, 3);
    let last = &plan.emits[&plan.messages[2].0];
    assert_eq!(
        last.len(),
        10,
        "the last emit takes its vertex's ten dwords"
    );
    let with = |extra: &[u32], max: u32| {
        let mut code = Code::default();
        gs_prelude(&mut code, max * VERTEX_DWORDS * 4);
        for (n, k) in [0, 1, 2].into_iter().enumerate() {
            gs_vertex(&mut code, n as u32, k);
            code.add([EMIT]);
        }
        code.add(extra.iter().copied());
        code.add([S_ENDPGM]);
        geometry::gs_plan(&code.program(), &shape(max)).map(|p| (p.messages, p.emitted))
    };
    let last = |extra: &[u32], max: u32| with(extra, max).map(|(m, e)| (m[m.len() - 1].1, e));
    assert_eq!(last(&[DONE_CUT], 3), Ok((Message::Cut { done: true }, 3)));
    // GS_DONE's emit past the most vertices, after another GS_DONE, or
    // followed by a message; on stream 1; GS_DONE alone naming a stream.
    assert_eq!(last(&[DONE_EMIT], 3), Err(geometry::TOO_MANY));
    let mut code = Code::default();
    gs_prelude(&mut code, 4 * VERTEX_DWORDS * 4);
    for n in 0..4 {
        gs_vertex(&mut code, n, 0);
        code.add([if n == 3 { DONE_EMIT_CUT } else { EMIT }]);
    }
    code.add([S_ENDPGM]);
    let plan = geometry::gs_plan(&code.program(), &shape(4)).expect("a plan");
    assert_eq!(plan.messages[3].1, emit(3, true, true));
    assert_eq!(plan.emitted, 4);
    assert_eq!(with(&[DONE, DONE_CUT], 3), Err(geometry::AFTER_DONE));
    assert_eq!(with(&[DONE_CUT, CUT], 3), Err(geometry::AFTER_DONE));
    assert_eq!(with(&[DONE_CUT | 1 << 8], 3), Err(geometry::STREAM));
    assert_eq!(with(&[DONE | 1 << 8], 3), Err(geometry::MESSAGE));
}

/// Emits and cuts under different lane masks are refused (Codex review of
/// W10's first phase): a lane outside one emit's EXEC counts its later
/// vertices one fewer than the path does. Under one mask, even a narrower
/// one than the entry's, they are numbered; GS_DONE alone may come after
/// the mask is restored.
#[test]
fn geometry_emits_share_one_lane_mask() {
    let saveexec = sop1(S_AND_SAVEEXEC_B64, 40, 42);
    let restore = sop2(S_OR_B64, EXEC, u32::from(EXEC), 40);
    let program = |masked: &[usize], cut_masked: bool| {
        let mut code = Code::default();
        gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
        let mut inside = false;
        let mut set = |code: &mut Code, want: bool| {
            if want != inside {
                code.add([if want { saveexec } else { restore }]);
                inside = want;
            }
        };
        for (n, k) in [0u8, 1, 2].into_iter().enumerate() {
            set(&mut code, masked.contains(&n));
            gs_vertex(&mut code, n as u32, k);
            code.add([EMIT]);
        }
        set(&mut code, cut_masked);
        code.add([CUT]);
        set(&mut code, false);
        code.add([DONE, S_ENDPGM]);
        geometry::gs_plan(&code.program(), &shape(3))
    };
    let all = program(&[0, 1, 2], true).expect("one mask");
    let vertices: Vec<u32> = all
        .messages
        .iter()
        .filter_map(|m| match m.1 {
            Message::Emit { vertex, .. } => Some(vertex),
            _ => None,
        })
        .collect();
    assert_eq!(vertices, [0, 1, 2]);
    assert!(program(&[], false).is_ok());
    assert_eq!(program(&[1], false), Err(geometry::MASKED));
    assert_eq!(program(&[0], false), Err(geometry::MASKED));
    assert_eq!(program(&[0, 1, 2], false), Err(geometry::MASKED));
    assert_eq!(program(&[], true), Err(geometry::MASKED));
}

/// An emit after a join where one side narrowed EXEC and nothing after the
/// join reads EXEC (a message reads M0 alone), so no phi joins the two
/// masks: the emit's mask depends on the path, and it is refused whichever
/// side narrows it (Codex review of W10's review fixes). When both sides
/// keep the entry's mask, the emit is numbered.
#[test]
fn geometry_emit_after_a_join_without_an_exec_phi() {
    let saveexec = sop1(S_AND_SAVEEXEC_B64, 40, 42);
    let other = sop1(S_MOV_B32, 50, int(1));
    // Vertices 0-2 written and 0-1 emitted under the entry's mask, then a
    // branch on user data, `taken` on one side and `fallen` on the other,
    // and vertex 2's emit after the join.
    let program = |taken: u32, fallen: u32| {
        let mut code = Code::default();
        gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
        for (n, k) in [0u8, 1, 2].into_iter().enumerate() {
            gs_vertex(&mut code, n as u32, k);
        }
        code.add([EMIT, EMIT, sopc(S_CMP_EQ_U32, 9, int(0))]);
        code.branch(S_CBRANCH_SCC1, "taken");
        code.add([fallen]);
        code.branch(S_BRANCH, "join");
        code.label("taken");
        code.add([taken]);
        code.label("join");
        code.add([EMIT, DONE, S_ENDPGM]);
        let p = code.program();
        let join = p.blocks.iter().position(|b| b.preds.len() == 2);
        let join = join.expect("a join");
        assert!(
            p.blocks[join].phis.iter().all(|ph| ph.reg != Reg::S(EXEC)),
            "no phi of EXEC at the join"
        );
        geometry::gs_plan(&p, &shape(3))
    };
    assert_eq!(program(saveexec, other), Err(geometry::MASKED));
    assert_eq!(program(other, saveexec), Err(geometry::MASKED));
    let plan = program(other, other).expect("one mask");
    assert_eq!(plan.emitted, 3);
}

/// D3D12's geometry-shader limits, and the codes DXIL gives the shapes.
#[test]
fn geometry_shapes_within_d3d12_limits() {
    let (inputs, outs) = (item(), outputs());
    assert_eq!(shape(3).check(&inputs, &outs), Ok(()));
    // 10 scalars per vertex: 102 vertices fit in 1024, 103 do not.
    assert_eq!(shape(102).check(&inputs, &outs), Ok(()));
    assert_eq!(shape(103).check(&inputs, &outs), Err(geometry::SCALARS));
    assert_eq!(shape(0).check(&inputs, &outs), Err(geometry::VERTICES));
    let instanced = |instances| GsShape {
        instances,
        ..shape(3)
    };
    assert_eq!(instanced(32).check(&inputs, &outs), Ok(()));
    assert_eq!(
        instanced(33).check(&inputs, &outs),
        Err(geometry::INSTANCES)
    );
    let mut far = outputs();
    far[3].row = Some(32);
    assert_eq!(shape(3).check(&inputs, &far), Err(geometry::ROWS));
    let mut boolean = outputs();
    boolean[1].comp = 1;
    assert_eq!(shape(3).check(&inputs, &boolean), Err(geometry::BOOLEAN));
    assert_eq!(shape(3).state(), [3, 3, 1, 5, 1]);
    let points = GsShape {
        input: Primitive::Point,
        topology: Topology::Points,
        ..shape(1)
    };
    let lines = GsShape {
        input: Primitive::Line,
        topology: Topology::LineStrip,
        ..shape(2)
    };
    assert_eq!(points.state(), [1, 1, 1, 1, 1]);
    assert_eq!(lines.state(), [2, 2, 1, 3, 1]);
    assert_eq!(
        [Primitive::Point, Primitive::Line, Primitive::Triangle].map(Primitive::vertices),
        [1, 2, 3]
    );
    // A point's GS has one input vertex: v0's high half is no vertex.
    let p = gs_program(&[Step::Emit(0)], 1);
    assert_eq!(geometry::gs_plan(&p, &points), Err(geometry::RING));
    assert_eq!(geometry::element_at(&outs, 4), Some((1, 0)));
    assert_eq!(geometry::element_at(&outs, 5), None);
    assert_eq!(geometry::element_at(&outs, 10), Some((2, 2)));
}

/// A `PSV0` part's data as a geometry shader's: GSInfo (input primitive,
/// output topology, stream mask, a position output), then runtime
/// information 1 with the stage (`stage`, 2 for geometry) and the most
/// vertices; no resources.
fn gs_psv(input: u32, topology: u32, streams: u32, stage: u8, max_vertices: u16) -> Vec<u8> {
    let mut info = [0u8; 48];
    info[0..4].copy_from_slice(&input.to_le_bytes());
    info[4..8].copy_from_slice(&topology.to_le_bytes());
    info[8..12].copy_from_slice(&streams.to_le_bytes());
    info[12] = 1;
    info[24] = stage;
    info[26..28].copy_from_slice(&max_vertices.to_le_bytes());
    let mut psv = 48u32.to_le_bytes().to_vec();
    psv.extend(info);
    psv.extend(0u32.to_le_bytes());
    psv
}

/// The stage data of the GS of [`shape`]`(max_vertices)`.
fn gs_io(max_vertices: u32, count: u32) -> GeometryIo {
    GeometryIo {
        outputs: Vec::new(),
        item_dwords: max_vertices * VERTEX_DWORDS,
        count,
        max_vertices,
        vertex_dwords: VERTEX_DWORDS,
    }
}

/// A console geometry shader's shape: the primitive, topology and most
/// vertices of its `PSV0`, the ring layout of its stage data, which must
/// agree; the shapes the translation does not know are refused. A
/// geometry shader the translation makes (`Shader::finish` with its
/// shape) reads back as the same shape.
#[test]
fn geometry_shape_from_psv0_and_stage_data() {
    let io = gs_io(3, 1);
    assert_eq!(GsShape::of(&gs_psv(3, 5, 1, 2, 3), &io), Ok(shape(3)));
    let points = GsShape::of(&gs_psv(1, 1, 1, 2, 1), &gs_io(1, 1)).expect("points");
    assert_eq!(
        (points.input, points.topology),
        (Primitive::Point, Topology::Points)
    );
    let lines = GsShape::of(&gs_psv(2, 3, 1, 2, 2), &gs_io(2, 1)).expect("lines");
    assert_eq!(
        (lines.input, lines.topology),
        (Primitive::Line, Topology::LineStrip)
    );
    // Triangles with adjacency; a triangle list; streams 0 and 1.
    assert_eq!(
        GsShape::of(&gs_psv(7, 5, 1, 2, 3), &io),
        Err(geometry::PRIMITIVE)
    );
    assert_eq!(
        GsShape::of(&gs_psv(3, 4, 1, 2, 3), &io),
        Err(geometry::TOPOLOGY)
    );
    assert_eq!(
        GsShape::of(&gs_psv(3, 5, 3, 2, 3), &io),
        Err(geometry::STREAMS)
    );
    assert_eq!(
        GsShape::of(&gs_psv(3, 5, 1, 2, 4), &io),
        Err(geometry::MOST_VERTICES)
    );
    assert_eq!(
        GsShape::of(&gs_psv(3, 5, 1, 2, 3), &gs_io(3, 2)),
        Err(geometry::COUNT)
    );
    // Another stage's runtime information; one too short; none.
    assert_eq!(GsShape::of(&gs_psv(3, 5, 1, 1, 3), &io), Err(geometry::PSV));
    let mut short = gs_psv(3, 5, 1, 2, 3);
    short[..4].copy_from_slice(&24u32.to_le_bytes());
    assert_eq!(GsShape::of(&short, &io), Err(geometry::PSV));
    assert_eq!(GsShape::of(&short[..30], &io), Err(geometry::PSV));
    assert_eq!(GsShape::of(&[], &io), Err(geometry::PSV));

    let steps = [Step::Emit(0), Step::Emit(1), Step::Emit(2), Step::Done];
    let p = gs_program(&steps, 3);
    let plan = geometry::gs_plan(&p, &shape(3)).expect("a plan");
    let made = lower_gs(&p, &plan, &shape(3));
    let c = crate::Container::parse(&made).expect("a container");
    let psv = c.part(crate::FourCC::PSV0).expect("a PSV0 part").data;
    assert_eq!(GsShape::of(psv, &io), Ok(shape(3)));
    // Program kind 2 (geometry), shader model 6.6.
    let dxil = c.part(crate::FourCC::DXIL).expect("a DXIL part").data;
    assert_eq!(&dxil[..4], &(2u32 << 16 | 0x66).to_le_bytes());
}

/// The lane set-up a merged half starts with, as the title's do it: EXEC
/// from its lane count in s3 (`first` its first bit: 0 for the ES, 8 for
/// the GS), every lane when the count is 64.
fn lane_prologue(code: &mut Code, first: u32) {
    code.add([sop2(S_BFE_I32, 4, 3, LITERAL), 8 << 16 | first]);
    code.add([
        sop2(S_BFM_B64, EXEC, 4, int(0)),
        sopc(S_CMP_EQ_U32, 4, int(64)),
        sop2(S_CSELECT_B64, EXEC, MINUS_ONE, u32::from(EXEC)),
    ]);
}

/// Each half's EXEC set-up from s3 goes, and what read the EXEC it made
/// reads the entry's instead (every vertex shader or geometry shader
/// invocation on the host is one of the half's lanes). Another stage is
/// left as it is; another set-up, or one whose count or compare something
/// else reads, is refused.
#[test]
fn geometry_lane_prologue_leaves_every_lane_on() {
    let program = |first: u32, tail: &[u32]| {
        let mut code = Code::default();
        lane_prologue(&mut code, first);
        code.add([vop1(V_MOV_B32, 1, vgpr(0))]);
        code.add(tail.iter().copied());
        code.add([S_ENDPGM]);
        code.program()
    };
    let exec = |p: &Program, k: u8| {
        let entry = p.inputs.iter().find(|i| i.0 == Reg::S(ir::EXEC + k));
        entry.map(|i| i.1).expect("EXEC on entry")
    };
    for (stage, first) in [(HwStage::Es, 0), (HwStage::Gs, 8)] {
        let mut p = program(first, &[]);
        let before = p.blocks[0].insts.len();
        geometry::drop_lane_prologue(&mut p, stage).expect("the title's set-up");
        assert_eq!(p.blocks[0].insts.len(), before - 4, "{stage:?}");
        let mov = &p.blocks[0].insts[0];
        assert_eq!(mov.name(), "v_mov_b32");
        assert_eq!(mov.read(Reg::S(ir::EXEC)), Some(exec(&p, 0)));
        assert_eq!(mov.read(Reg::S(ir::EXEC + 1)), Some(exec(&p, 1)));
    }
    let same = program(0, &[]);
    let mut p = same.clone();
    geometry::drop_lane_prologue(&mut p, HwStage::Vs).expect("nothing to do");
    assert_eq!(p, same);
    let refused = |first: u32, tail: &[u32], stage: HwStage| {
        geometry::drop_lane_prologue(&mut program(first, tail), stage)
    };
    // The other half's count.
    assert_eq!(refused(8, &[], HwStage::Es), Err(geometry::PROLOGUE));
    assert_eq!(refused(0, &[], HwStage::Gs), Err(geometry::PROLOGUE));
    // The count read again; the compare's SCC read again.
    let count = sop1(S_MOV_B32, 20, 4);
    assert_eq!(refused(0, &[count], HwStage::Es), Err(geometry::PROLOGUE));
    let scc = sop2(S_CSELECT_B32, 20, int(1), int(0));
    assert_eq!(refused(0, &[scc], HwStage::Es), Err(geometry::PROLOGUE));
    // No set-up at all.
    let mut bare = gs_program(&[Step::Done], 3);
    assert_eq!(
        geometry::drop_lane_prologue(&mut bare, HwStage::Gs),
        Err(geometry::PROLOGUE)
    );
}

/// `v_mov_b32` of a constant or a VGPR: the only data instructions the
/// walkers below lower.
fn moved(s: &mut Shader, inst: &Inst, regs: &mut HashMap<Reg, V>) -> bool {
    let Fields::Valu(v) = &inst.fields else {
        return false;
    };
    let Some(Operand::Vector { reg, .. }) = v.dst else {
        return false;
    };
    if inst.name() != "v_mov_b32" {
        return false;
    }
    let value = match v.src[0] {
        Operand::Literal(x) => s.m.i32(x),
        Operand::Const(c) => {
            s.m.i32(crate::resource::inline(c).expect("an inline constant"))
        }
        Operand::Vector { reg: r, .. } => regs[&Reg::V(r)],
        _ => return false,
    };
    regs.insert(Reg::V(reg), value);
    true
}

/// Validates `container` with the system's `dxil.dll` when there is one
/// (a skip is said on standard error).
fn validate(what: &str, container: &[u8]) {
    use super::validator::Validator;
    match Validator::load(Validator::default_path().as_deref()) {
        Ok(mut v) => {
            if let Err(e) = v.validate(&mut container.to_vec()) {
                panic!("{what}: validation: {e}");
            }
        }
        Err(e) => eprintln!("SKIPPED validation of {what}: {e}"),
    }
}

/// The geometry shader of the console GS `p` and its plan, lowered through
/// the hooks of `lower/geometry.rs` as the lowering will call them: the
/// address set-up left out, ring reads as inputs, ring writes kept, emits
/// and cuts as stream operations.
fn lower_gs(p: &Program, plan: &GsPlan, shape: &GsShape) -> Vec<u8> {
    let (inputs, outs) = (item(), outputs());
    shape.check(&inputs, &outs).expect("within D3D12's limits");
    let mut s = Shader::new(false);
    let block = s.entry;
    let reads: HashMap<Site, &RingAccess> = plan.reads.iter().map(|r| (r.site, r)).collect();
    let writes: HashMap<Site, &RingAccess> = plan.writes.iter().map(|w| (w.site, w)).collect();
    let messages: HashMap<Site, Message> = plan.messages.iter().copied().collect();
    let mut regs: HashMap<Reg, V> = HashMap::new();
    let mut vertices = Vertices::default();
    for (b, blk) in p.blocks.iter().enumerate() {
        for (i, inst) in blk.insts.iter().enumerate() {
            let site = (b, i);
            if plan.address_only.contains(&site) {
                continue;
            }
            if let Some(r) = reads.get(&site) {
                for &(reg, slot) in &r.dwords {
                    let v = geometry::read_vertex(&mut s, block, &inputs, slot).expect("an input");
                    regs.insert(reg, v);
                }
                continue;
            }
            if let Some(w) = writes.get(&site) {
                for &(reg, _) in &w.dwords {
                    vertices.write(site, reg, regs[&reg]);
                }
                continue;
            }
            match messages.get(&site) {
                Some(&Message::Emit { cut, .. }) => {
                    vertices
                        .emit(&mut s, block, &outs, &plan.emits[&site])
                        .expect("its writes kept");
                    if cut {
                        Vertices::cut(&mut s, block);
                    }
                    continue;
                }
                Some(Message::Cut { .. }) => {
                    Vertices::cut(&mut s, block);
                    continue;
                }
                Some(Message::Done) => continue,
                None => {}
            }
            assert!(
                moved(&mut s, inst, &mut regs) || inst.name() == "s_endpgm",
                "the walker does not lower {}",
                inst.name()
            );
        }
    }
    s.m.terminate(block, Term::Ret);
    s.inputs = inputs;
    s.outputs = outs;
    s.geometry = Some(*shape);
    s.finish()
}

/// The vertex shader of the console ES `p` and its plan: the address
/// set-up left out, ring writes as outputs, the barrier and the jump to
/// the GS half as the end.
fn lower_es(p: &Program, plan: &EsPlan) -> Vec<u8> {
    let mut s = Shader::new(false);
    let block = s.entry;
    s.inputs.push(SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    });
    let i32t = s.t.i32;
    let undef = s.m.undef(i32t);
    let args = [s.m.i32(0), s.m.i32(0), s.m.i8(0), undef];
    let id = s.op(block, "loadInput.i32", Attr::ReadNone, i32t, 4, &args);
    let mut regs = HashMap::from([(Reg::V(crate::io::ES_VERTEX_ID), id)]);
    let writes: HashMap<Site, &RingAccess> = plan.writes.iter().map(|w| (w.site, w)).collect();
    let outputs = item();
    for (b, blk) in p.blocks.iter().enumerate() {
        for (i, inst) in blk.insts.iter().enumerate() {
            if plan.address_only.contains(&(b, i)) {
                continue;
            }
            if let Some(w) = writes.get(&(b, i)) {
                for &(reg, slot) in &w.dwords {
                    geometry::write_item(&mut s, block, &outputs, slot.dword, regs[&reg]);
                }
                continue;
            }
            assert!(
                moved(&mut s, inst, &mut regs)
                    || matches!(inst.name(), "s_barrier" | "s_setpc_b64"),
                "the walker does not lower {}",
                inst.name()
            );
        }
    }
    s.outputs = outputs;
    s.m.terminate(block, Term::Ret);
    s.finish()
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Room for streamed-out vertices, in bytes.
const CAPACITY: u64 = 1 << 16;

/// Draws `vertices` vertices as a triangle list through `vs` and `gs`
/// with no rasterizer, the geometry shader's stream 0 (`outs`) streamed
/// out; the dwords written, in order.
fn stream_out(
    host: &Host,
    vs: &[u8],
    gs: &[u8],
    outs: &[SigElement],
    vertices: u32,
) -> Result<Vec<u32>, String> {
    let dev = &host.device;
    let names: Vec<CString> = outs
        .iter()
        .map(|e| CString::new(e.name.clone()).expect("a name"))
        .collect();
    let entries: Vec<D3D12_SO_DECLARATION_ENTRY> = outs
        .iter()
        .zip(&names)
        .map(|(e, n)| D3D12_SO_DECLARATION_ENTRY {
            Stream: 0,
            SemanticName: PCSTR(n.as_ptr().cast()),
            SemanticIndex: e.index,
            StartComponent: e.col,
            ComponentCount: e.cols,
            OutputSlot: 0,
        })
        .collect();
    let strides = [outs.iter().map(|e| 4 * u32::from(e.cols)).sum::<u32>()];
    // SAFETY: D3D12 calls with descriptions that outlive them; resources
    // live until the fence says the GPU is done with them.
    unsafe {
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT,
                    ..Default::default()
                },
            },
        };
        let (mut blob, mut error) = (None, None);
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)).map_err(err)?;
        let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).map_err(err)?;
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(vs),
            GS: code(gs),
            StreamOutput: D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: entries.as_ptr(),
                NumEntries: entries.len() as u32,
                pBufferStrides: strides.as_ptr(),
                NumStrides: 1,
                RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM,
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&pso_desc).map_err(err)?;
        let make = |kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
                .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let target = make(
            D3D12_HEAP_TYPE_DEFAULT,
            CAPACITY,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let filled = make(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST)?;
        let zeros = make(
            D3D12_HEAP_TYPE_UPLOAD,
            256,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let readback = make(
            D3D12_HEAP_TYPE_READBACK,
            CAPACITY + 256,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let mut p = std::ptr::null_mut();
        zeros.Map(0, None, Some(&mut p)).map_err(err)?;
        std::ptr::write_bytes(p.cast::<u8>(), 0, 256);
        zeros.Unmap(0, None);
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        list.CopyBufferRegion(&filled, 0, &zeros, 0, 256);
        list.ResourceBarrier(&[
            transition(
                &target,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_STREAM_OUT,
            ),
            transition(
                &filled,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_STREAM_OUT,
            ),
        ]);
        list.SetGraphicsRootSignature(&root);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.SOSetTargets(
            0,
            Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW {
                BufferLocation: target.GetGPUVirtualAddress(),
                SizeInBytes: CAPACITY,
                BufferFilledSizeLocation: filled.GetGPUVirtualAddress(),
            }]),
        );
        list.DrawInstanced(vertices, 1, 0, 0);
        list.ResourceBarrier(&[
            transition(
                &target,
                D3D12_RESOURCE_STATE_STREAM_OUT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
            transition(
                &filled,
                D3D12_RESOURCE_STATE_STREAM_OUT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
        ]);
        list.CopyBufferRegion(&readback, 0, &target, 0, CAPACITY);
        list.CopyBufferRegion(&readback, CAPACITY, &filled, 0, 256);
        list.Close().map_err(err)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            std::thread::yield_now();
        }
        if let Err(e) = dev.GetDeviceRemovedReason() {
            return Err(format!("device removed: {}", err(e)));
        }
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).map_err(err)?;
        let words = std::slice::from_raw_parts(p.cast::<u32>(), (CAPACITY as usize + 256) / 4);
        let filled_bytes = words[CAPACITY as usize / 4] as usize;
        let out = words[..filled_bytes.min(CAPACITY as usize) / 4].to_vec();
        readback.Unmap(0, None);
        Ok(out)
    }
}

/// The probe vertex shader's dword `c` of item element `e` for vertex `n`.
fn probe_word(n: usize, e: usize, c: usize) -> u32 {
    match e {
        0 => (1.0 + n as f32 + 0.125 * c as f32).to_bits(),
        1 => (10.0 + n as f32 + 0.25 * c as f32).to_bits(),
        _ => 1000 + n as u32,
    }
}

/// The ES-made vertex shader's: its constants, then the vertex id.
fn es_word(n: usize, e: usize, c: usize) -> u32 {
    match e {
        0 | 1 => ES_CONSTANTS[4 * e + c].to_bits(),
        _ => n as u32,
    }
}

/// The streamed-out record of vertex `n` that a GS invocation for
/// primitive `prim` makes from its input vertex `k`, whose dwords `word`
/// gives.
fn record(word: fn(usize, usize, usize) -> u32, prim: usize, n: u32, k: usize) -> Vec<u32> {
    let id = 3 * prim + k;
    let mut r: Vec<u32> = (0..4).map(|c| word(id, 0, c)).collect();
    r.push(100 + n);
    r.extend((0..4).map(|c| word(id, 1, c)));
    r.push(word(id, 2, 0));
    r
}

/// The test adapter's device, with the lock that lets one GPU test of this
/// crate use it at a time (two at once on WARP can end the test process
/// with an access violation: `valu_tests::ONE_AT_A_TIME`); `None` on a
/// machine without one.
fn device() -> Option<(Host, std::sync::MutexGuard<'static, ()>)> {
    let one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    match Host::new(false) {
        Ok(host) => Some((host, one)),
        Err(e) => {
            eprintln!("no SM 6.6 device ({e}); skipped");
            None
        }
    }
}

/// The probe vertex shader over [`item`] (positions, floats and uints per
/// vertex id), for `vertices` vertices.
fn probe_vs(vertices: usize) -> Vec<u8> {
    super::rect_gs::probe::with_outputs(item(), vertices, probe_word)
}

/// The geometry shader made from `steps` with `max_vertices`, drawn over
/// `primitives` triangles of the probe; what it streams out, or `None` on
/// a machine without a device.
fn run_gs(steps: &[Step], max_vertices: u32, primitives: u32) -> Option<Vec<u32>> {
    let p = gs_program(steps, max_vertices);
    let shape = shape(max_vertices);
    let plan = geometry::gs_plan(&p, &shape).expect("a plan");
    let gs = lower_gs(&p, &plan, &shape);
    validate("geometry shader", &gs);
    let vs = probe_vs(3 * primitives as usize);
    validate("vertex shader", &vs);
    let (host, _one) = device()?;
    Some(stream_out(&host, &vs, &gs, &outputs(), 3 * primitives).expect("a draw"))
}

/// The records of one stream-out, split.
fn records(words: &[u32]) -> Vec<Vec<u32>> {
    assert_eq!(words.len() % 10, 0, "whole records");
    words.chunks(10).map(<[u32]>::to_vec).collect()
}

/// Three emits, a cut, GS_DONE: one triangle per invocation, in the order
/// emitted, every dword from the input vertex each emit names, the uint
/// from the emit's own number; two primitives, in order.
#[test]
fn geometry_shader_streams_out_every_emitted_vertex() {
    let steps = [
        Step::Emit(2),
        Step::Emit(0),
        Step::Emit(1),
        Step::Cut,
        Step::Done,
    ];
    let Some(words) = run_gs(&steps, 3, 2) else {
        return;
    };
    let mut want = Vec::new();
    for prim in 0..2 {
        for (n, k) in [2, 0, 1].into_iter().enumerate() {
            want.push(record(probe_word, prim, n as u32, k));
        }
    }
    assert_eq!(records(&words), want);
}

/// A cut after two vertices: that strip makes no triangle, the three
/// vertices after it one.
#[test]
fn geometry_shader_cut_ends_the_strip() {
    let steps = [
        Step::Emit(0),
        Step::Emit(1),
        Step::Cut,
        Step::Emit(2),
        Step::Emit(1),
        Step::EmitCut(0),
        Step::Done,
    ];
    let Some(words) = run_gs(&steps, 5, 1) else {
        return;
    };
    let want: Vec<Vec<u32>> = [(2, 2), (3, 1), (4, 0)]
        .into_iter()
        .map(|(n, k)| record(probe_word, 0, n, k))
        .collect();
    assert_eq!(records(&words), want);
}

/// Four vertices in one strip (`maxvertexcount` 4, the last emit an
/// EMIT_CUT): two triangles, the second the strip's next three vertices
/// turned the same way as the first (D3D12 may start it at any of its
/// vertices).
#[test]
fn geometry_shader_vertex_counts() {
    let steps = [
        Step::Emit(0),
        Step::Emit(1),
        Step::Emit(2),
        Step::EmitCut(0),
        Step::Done,
    ];
    let plan = plan(&steps, 4).expect("a plan");
    assert_eq!(plan.emitted, 4);
    assert_eq!(shape(4).state()[1], 4, "maxvertexcount");
    let Some(words) = run_gs(&steps, 4, 1) else {
        return;
    };
    let v = |n: u32, k: usize| record(probe_word, 0, n, k);
    let got = records(&words);
    assert_eq!(got.len(), 6, "two triangles");
    assert_eq!(got[..3], [v(0, 0), v(1, 1), v(2, 2)]);
    let second = [v(1, 1), v(3, 0), v(2, 2)];
    let turned = (0..3).any(|r| (0..3).all(|i| got[3 + i] == second[(i + r) % 3]));
    assert!(turned, "second triangle {:?}", &got[3..]);
}

/// GS_DONE's own emit is drawn as the others are: two emits, then GS_DONE
/// carrying the third, one triangle per invocation (before the Codex
/// review's fix it was dropped, and nothing was drawn).
#[test]
fn geometry_shader_streams_out_the_emit_gs_done_carries() {
    let steps = [Step::Emit(1), Step::Emit(0), Step::EmitDone(2)];
    let Some(words) = run_gs(&steps, 3, 2) else {
        return;
    };
    let mut want = Vec::new();
    for prim in 0..2 {
        for (n, k) in [1, 0, 2].into_iter().enumerate() {
            want.push(record(probe_word, prim, n as u32, k));
        }
    }
    assert_eq!(records(&words), want);
}

/// The pair, both halves made from console programs: the ES's ring item
/// as the vertex shader's outputs (its constants and its vertex id), the
/// GS's reads of it as inputs.
#[test]
fn geometry_es_and_gs_pair_streams_out() {
    let es = es_program();
    let es_plan = geometry::es_plan(&es).expect("an ES plan");
    let vs = lower_es(&es, &es_plan);
    validate("vertex shader from the ES", &vs);
    let steps = [
        Step::Emit(1),
        Step::Emit(2),
        Step::Emit(0),
        Step::Cut,
        Step::Done,
    ];
    let p = gs_program(&steps, 3);
    let plan = geometry::gs_plan(&p, &shape(3)).expect("a plan");
    let gs = lower_gs(&p, &plan, &shape(3));
    validate("geometry shader", &gs);
    let Some((host, _one)) = device() else {
        return;
    };
    let words = stream_out(&host, &vs, &gs, &outputs(), 6).expect("a draw");
    let mut want = Vec::new();
    for prim in 0..2 {
        for (n, k) in [1, 2, 0].into_iter().enumerate() {
            want.push(record(es_word, prim, n as u32, k));
        }
    }
    assert_eq!(records(&words), want);
}

/// The vertex shader container of an ES of [`es_program`]'s form, its
/// lane set-up first, as the translation takes it: the ES beside its
/// output signature (the ring item's rows).
fn es_container() -> (Vec<u8>, NativeProgram) {
    es_container_with(false)
}

/// [`es_container`], its tail as [`es_tail_with`] `fill` writes it.
fn es_container_with(fill: bool) -> (Vec<u8>, NativeProgram) {
    let mut code = Code::default();
    lane_prologue(&mut code, 0);
    code.add(es_code().words);
    es_position(&mut code, ES_CONSTANTS[..4].try_into().unwrap());
    es_tail_with(&mut code, fill);
    let out = |usage, row| VsOutput {
        usage,
        row,
        param: 0,
        fp16: false,
        pack2: false,
    };
    let native = merged_native(
        HwStage::Es,
        &code.words,
        StageIo::Es(ExportIo {
            inputs: Vec::new(),
            outputs: vec![
                out(Usage::Other(0), 0),
                out(Usage::Generic, 1),
                out(Usage::Generic, 2),
            ],
            item_dwords: 12,
        }),
    );
    let osg1 = container::signature(&item(), false);
    (dxbc::build(&[(FourCC::OSG1, &osg1)]), native)
}

/// The geometry shader container of [`gs_program`]`(steps, 3)`, its lane
/// set-up first: its input signature (the ES's item), its stream 0, and a
/// `PSV0` of its shape (`psv` false: none).
fn gs_container(steps: &[Step], psv: bool) -> (Vec<u8>, NativeProgram) {
    let mut code = Code::default();
    lane_prologue(&mut code, 8);
    gs_code(&mut code, steps, 3);
    gs_container_of(&code.resolved(), psv)
}

/// [`gs_container`] of the code `words` (lane set-up included), a GS of
/// [`shape`]`(3)`.
fn gs_container_of(words: &[u32], psv: bool) -> (Vec<u8>, NativeProgram) {
    let mut gs_io = gs_io(3, 1);
    gs_io.outputs = (0..4)
        .map(|row| VsOutput {
            usage: if row == 0 {
                Usage::Other(0)
            } else {
                Usage::Generic
            },
            row,
            param: 0,
            fp16: false,
            pack2: false,
        })
        .collect();
    let native = merged_native(HwStage::Gs, words, StageIo::Gs(gs_io));
    let (isg1, osg1) = (
        container::signature(&item(), true),
        container::signature(&outputs(), false),
    );
    let shape = gs_psv(3, 5, 1, 2, 3);
    let mut parts = vec![(FourCC::ISG1, &isg1[..]), (FourCC::OSG1, &osg1[..])];
    if psv {
        parts.push((FourCC::PSV0, &shape[..]));
    }
    (dxbc::build(&parts), native)
}

/// A wave64 console program of a merged half with no root parameters: the
/// GS's ring base comes from user data no root parameter holds (the
/// console driver's table).
fn merged_native(stage: HwStage, words: &[u32], io: StageIo) -> NativeProgram {
    NativeProgram {
        stage,
        wave64: true,
        float_mode: 0,
        vgprs: 64,
        layout: Vec::new(),
        used: Vec::new(),
        io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// What the translation makes of `container` and its program, panicking
/// with the step and reason when it refuses it.
fn translated(what: &str, (container, native): (Vec<u8>, NativeProgram)) -> HostShader {
    let c = Container::parse(&container).expect("a container");
    let host = super::translate_program(&c, &native)
        .unwrap_or_else(|(step, e)| panic!("{what}: {}: {e}", step.name()));
    validate(what, &host.dxil);
    host
}

/// The pair through the translation itself (W10's second phase): each
/// half's EXEC set-up from s3 left out, the GS's ring base, loaded from the
/// console driver's table, left out of its resources, the ES translated as
/// a vertex shader and the GS as a geometry shader (`gs_6_6`). Drawn, they
/// stream out what the console's copy shader would read from the ring. A
/// GS container without a `PSV0` has no shape, and is refused.
#[test]
fn geometry_pair_translates_and_streams_out() {
    let vs = translated("the ES", es_container());
    assert_eq!(vs.stage, HwStage::Es);
    let dxil = Container::parse(&vs.dxil).expect("a container");
    let program = dxil.part(FourCC::DXIL).expect("a DXIL part").data;
    assert_eq!(&program[..4], &(1u32 << 16 | 0x66).to_le_bytes(), "vs_6_6");
    let steps = [
        Step::Emit(1),
        Step::Emit(2),
        Step::Emit(0),
        Step::Cut,
        Step::Done,
    ];
    let gs = translated("the GS", gs_container(&steps, true));
    assert_eq!(gs.stage, HwStage::Gs);
    assert!(gs.user_data.is_empty(), "the ring base's table is not read");
    let dxil = Container::parse(&gs.dxil).expect("a container");
    let psv = dxil.part(FourCC::PSV0).expect("a PSV0 part").data;
    assert_eq!(GsShape::of(psv, &gs_io(3, 1)), Ok(shape(3)));

    let (container, native) = gs_container(&steps, false);
    let c = Container::parse(&container).expect("a container");
    match super::translate_program(&c, &native) {
        Err((super::Step::Resources, crate::ShaderError::Unsupported { what, .. })) => {
            assert_eq!(what, geometry::PSV);
        }
        other => panic!("{:?}", other.map(|_| ())),
    }

    let Some((host, _one)) = device() else {
        return;
    };
    let words = stream_out(&host, &vs.dxil, &gs.dxil, &outputs(), 6).expect("a draw");
    let mut want = Vec::new();
    for prim in 0..2 {
        for (n, k) in [1, 2, 0].into_iter().enumerate() {
            want.push(record(es_word, prim, n as u32, k));
        }
    }
    assert_eq!(records(&words), want);
}

/// What the translation stops a merged half with, if anything.
fn refusal((container, native): (Vec<u8>, NativeProgram)) -> Option<&'static str> {
    let c = Container::parse(&container).expect("a container");
    match super::translate_program(&c, &native) {
        Ok(_) => None,
        Err((_, crate::ShaderError::Unsupported { what, .. })) => Some(what),
        Err((step, e)) => panic!("{}: {e}", step.name()),
    }
}

/// Draws the pair `vs`, `gs` over two triangles of the ES-made vertex
/// shader and checks that each GS invocation emits its input vertices 0, 1
/// and 2 in order, as the console's copy shader reads them from the ring;
/// nothing on a machine without a device.
fn streams_out_vertices_in_order(vs: &HostShader, gs: &HostShader) {
    let Some((host, _one)) = device() else {
        return;
    };
    let words = stream_out(&host, &vs.dxil, &gs.dxil, &outputs(), 6).expect("a draw");
    let mut want = Vec::new();
    for prim in 0..2 {
        for k in 0..3 {
            want.push(record(es_word, prim, k as u32, k));
        }
    }
    assert_eq!(records(&words), want);
}

/// A row-filling write into a short last row (Codex review 3983f844, P2):
/// the ES writes its vertex id with item dwords 9 to 11 (`ds_write2_b64`),
/// the GS each vertex's `100 + n` with its dwords 5 to 7, and those dwords
/// hold values with no host form (the GS half's system VGPRs). No output
/// element holds them, so nothing reads them: both halves translate (each
/// was refused before, "a pixel system input"), and the pair streams out
/// what the console's copy shader reads.
#[test]
fn geometry_short_rows_filled_with_unread_values_translate() {
    let vs = translated("the ES filling its last row", es_container_with(true));
    let steps = [Step::Emit(0), Step::Emit(1), Step::Emit(2), Step::Done];
    let mut code = Code::default();
    lane_prologue(&mut code, 8);
    gs_code_with(&mut code, &steps, 3, true);
    let gs = translated(
        "the GS filling its vertices' short row",
        gs_container_of(&code.resolved(), true),
    );
    streams_out_vertices_in_order(&vs, &gs);
}

/// `v_cmp_ne_u32 vcc, src0, v<vsrc1>`.
fn v_cmp_ne_u32(src0: u32, vsrc1: u8) -> u32 {
    0x7C00_0000 | 0xC5 << 17 | u32::from(vsrc1) << 9 | src0
}

/// A geometry shader of [`shape`]`(3)` (lane set-up first) whose vertex 0
/// is written `passes` times by the same instructions (a loop on s50 where
/// `passes` is above 1), then emitted after them, then vertices 1 and 2.
/// With `masked`, each pass writes it under an EXEC narrowed without a
/// branch to the lanes whose input vertex 0 has a nonzero dword 0
/// (`s_and_saveexec_b64` of a compare), restored after the write, so that
/// the emits run under the entry's mask.
fn gs_vertex_in_a_loop(masked: bool, passes: u32) -> Vec<u32> {
    let mut code = Code::default();
    lane_prologue(&mut code, 8);
    gs_prelude(&mut code, 3 * VERTEX_DWORDS * 4);
    code.add([sop1(S_MOV_B32, 50, int(passes))]);
    code.label("pass");
    if masked {
        code.add([v_cmp_ne_u32(int(0), 20), sop1(S_AND_SAVEEXEC_B64, 40, VCC)]);
    }
    gs_vertex(&mut code, 0, 0);
    if masked {
        code.add([sop2(S_OR_B64, EXEC, u32::from(EXEC), 40)]);
    }
    if passes > 1 {
        code.add([
            sop2(S_SUB_U32, 50, 50, int(1)),
            sopc(S_CMP_EQ_U32, 50, int(0)),
        ]);
        code.branch(S_CBRANCH_SCC0, "pass");
    }
    code.add([EMIT]);
    gs_vertex(&mut code, 1, 1);
    code.add([EMIT]);
    gs_vertex(&mut code, 2, 2);
    code.add([EMIT, CUT, DONE, S_ENDPGM]);
    code.resolved()
}

/// A GSVS write in a loop under a narrowed EXEC (Codex review 3983f844,
/// P1): a lane the mask leaves out of a later pass keeps in the ring what
/// an earlier pass wrote, where the host keeps the last pass's value, so
/// it is refused. The plan takes the program (one write per dword on the
/// paths to the emit, before it); the lowering finds the guard. In a loop
/// with every lane writing, or under the mask without a loop (a lane left
/// out holds nothing the program wrote), it translates and streams out
/// what the console does (every lane passes the compare here).
#[test]
fn geometry_masked_write_in_a_loop_is_refused() {
    let mut p = program_of(&gs_vertex_in_a_loop(true, 2));
    geometry::drop_lane_prologue(&mut p, HwStage::Gs).expect("the title's set-up");
    let plan = geometry::gs_plan(&p, &shape(3)).expect("a plan");
    assert_eq!(plan.looped.len(), 5, "vertex 0's five writes: {plan:?}");
    assert_eq!(
        refusal(gs_container_of(&gs_vertex_in_a_loop(true, 2), true)),
        Some(geometry::LOOPED)
    );
    let vs = translated("the ES", es_container());
    for (masked, passes) in [(false, 2), (true, 1)] {
        let what = format!("the GS, masked {masked}, {passes} passes");
        let gs = translated(
            &what,
            gs_container_of(&gs_vertex_in_a_loop(masked, passes), true),
        );
        streams_out_vertices_in_order(&vs, &gs);
    }
}
