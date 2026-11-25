//! R1 phase 2 (docs/analysis/r1-scalar-data.md, section 5): a constant
//! buffer's descriptor whose registers the program gives to a number after
//! the descriptor's last access (`lower/smem_data.rs`). Hand-assembled
//! GFX10 compute programs (our own code) in one wave of 32, a table of
//! descriptors at s0: entry 0 (dword 0) a constant buffer, entry 1
//! (dword 8) a raw UAV, entry 2 (dword 16) another constant buffer. Every
//! translation is checked by the validator where `dxil.dll` is found, and
//! the one that translates is run on WARP.
//!
//! The names start with `smem_`, so the task's check (`cargo test -p
//! xtp-shader smem`) runs them.

use super::host::{Dispatch, Host, Root, View};
use super::tests::{compute_of, emit_for};
use super::{HostShader, Target};
use crate::native::{
    ComputeIo, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// Lanes in the group (one wave of 32).
const LANES: u32 = 32;

// Encodings (GFX10 ISA reference, "Microcode Formats").
const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_waitcnt lgkmcnt(0)`.
const WAIT_SCALAR: u32 = 0xBF8C_C07F;
/// Operand code of `null` (no register).
const NULL: u8 = 125;
/// Operand code of the inline constant 0.
const ZERO: u8 = 0x80;
/// SMEM opcodes.
const S_LOAD_DWORDX4: u32 = 2;
const S_BUFFER_LOAD_DWORD: u32 = 8;
/// MUBUF opcode.
const BUFFER_STORE_DWORD: u32 = 0x1C;

/// Operand code of the inline constant `n` (0 to 64).
const fn inline(n: u8) -> u8 {
    0x80 + n
}

/// `s_mov_b32 s<d>, <src>` (an operand code).
fn s_mov(d: u8, src: u8) -> u32 {
    0xBE80_0300 | u32::from(d) << 16 | u32::from(src)
}

/// `s_cmp_eq_u32 <a>, <b>` (operand codes).
fn s_cmp_eq_u32(a: u8, b: u8) -> u32 {
    0xBF06_0000 | u32::from(b) << 8 | u32::from(a)
}

/// `s_cbranch_scc1` over the next `skip` dwords.
fn s_cbranch_scc1(skip: u16) -> u32 {
    0xBF85_0000 | u32::from(skip)
}

/// SMEM `op` into s<data> through s[base:...] at byte `offset` (no
/// register offset).
fn smem(op: u32, data: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | op << 18 | u32::from(data) << 6 | u32::from(base / 2),
        u32::from(NULL) << 25 | offset,
    ]
}

/// `v_mov_b32 v<v>, <src>` (an operand code).
fn v_mov(v: u8, src: u8) -> u32 {
    0x7E00_0200 | u32::from(v) << 17 | u32::from(src)
}

/// `v_lshlrev_b32 v<d>, by, v<v>`.
fn v_lshl(d: u8, by: u8, v: u8) -> u32 {
    0x3400_0000 | u32::from(d) << 17 | u32::from(v) << 9 | u32::from(inline(by))
}

/// `buffer_store_dword v<data>, v<addr>, s[rsrc:rsrc+3], 0 offen`.
fn store(data: u8, addr: u8, rsrc: u8) -> [u32; 2] {
    [
        0xE000_0000 | BUFFER_STORE_DWORD << 18 | 1 << 12,
        u32::from(addr) | u32::from(data) << 8 | u32::from(rsrc / 4) << 16 | u32::from(ZERO) << 24,
    ]
}

/// A used-resource entry of the table: `kind` descriptors of four dwords
/// from dword `at`.
fn entry(kind: ResourceType, at: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }
}

/// How the program uses the registers of the constant buffer at entry 0
/// (s[8:11]) after its accesses.
#[derive(Clone, Copy)]
enum Reuse {
    /// Unless the flag (the buffer's dword 0, in s12) is 0, a scalar load
    /// of its dword 1 overwrites s8; after the join each lane stores s8
    /// where the flag (tested again) is set, 7 elsewhere.
    Number,
    /// The same with s[8:11] reloaded from entry 2 instead (a constant
    /// buffer read by an `s_buffer_load` there): the join takes two
    /// descriptors' dwords.
    Descriptor,
    /// No branch: each lane stores s10, the record count.
    Count,
    /// [`Reuse::Number`] without the second test (Codex review 3983f844):
    /// after the join every lane stores s8, the number where the flag is
    /// set, the buffer's descriptor dword 0 (its address) where it is 0.
    EveryEdge,
}

/// A loop whose lane read (`v_readlane_b32`) and divergent branch make a
/// program run in the console flow (`lower/flow.rs`'s
/// `the_console_flow_keeps_a_descriptor_join_unproven_and_counts_it`): it
/// writes s9, s10 (EXEC kept) and v2, which the programs below set again
/// before they read them.
const CONSOLE_FLOW_LOOP: [u32; 12] = [
    0xBE89_0380, // s_mov_b32 s9, 0
    0x7E04_0280, // v_mov_b32 v2, 0
    0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
    0xBE8A_3C6A, // s_and_saveexec_b32 s10, vcc_lo
    0xBF88_0006, // s_cbranch_execz -> end
    0xD760_0009,
    0x0001_0300, // loop: v_readlane_b32 s9, v0, 1
    0x4A04_0481, // v_add_nc_u32 v2, 1, v2
    0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
    0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
    0xBF89_FFFA, // s_cbranch_execnz -> loop
    0x887E_0A7E, // end: s_or_b32 exec_lo, exec_lo, s10
];

/// The program of [`Reuse`] `reuse`: lane t stores at 4 × t of the UAV.
fn program(reuse: Reuse) -> NativeProgram {
    program_after(&[], reuse)
}

/// [`program`] after the code `first`.
fn program_after(first: &[u32], reuse: Reuse) -> NativeProgram {
    let mut w = first.to_vec();
    w.extend([s_mov(5, ZERO), s_mov(4, 0)]);
    w.extend(smem(S_LOAD_DWORDX4, 8, 4, 0));
    w.extend(smem(S_LOAD_DWORDX4, 20, 4, 0x20));
    w.push(WAIT_SCALAR);
    w.extend(smem(S_BUFFER_LOAD_DWORD, 12, 8, 0));
    w.extend([WAIT_SCALAR, v_mov(1, inline(7))]);
    match reuse {
        Reuse::Number => {
            w.extend([s_cmp_eq_u32(12, ZERO), s_cbranch_scc1(3)]);
            w.extend(smem(S_BUFFER_LOAD_DWORD, 8, 8, 4));
            w.push(WAIT_SCALAR);
            w.extend([s_cmp_eq_u32(12, ZERO), s_cbranch_scc1(1), v_mov(1, 8)]);
        }
        Reuse::Descriptor => {
            w.extend([s_cmp_eq_u32(12, ZERO), s_cbranch_scc1(6)]);
            w.extend(smem(S_LOAD_DWORDX4, 8, 4, 0x40));
            w.push(WAIT_SCALAR);
            w.extend(smem(S_BUFFER_LOAD_DWORD, 13, 8, 0));
            w.push(WAIT_SCALAR);
            w.extend([s_cmp_eq_u32(12, ZERO), s_cbranch_scc1(1), v_mov(1, 8)]);
        }
        Reuse::Count => w.push(v_mov(1, 10)),
        Reuse::EveryEdge => {
            w.extend([s_cmp_eq_u32(12, ZERO), s_cbranch_scc1(3)]);
            w.extend(smem(S_BUFFER_LOAD_DWORD, 8, 8, 4));
            w.extend([WAIT_SCALAR, v_mov(1, 8)]);
        }
    }
    w.push(v_lshl(2, 2, 0));
    w.extend(store(1, 2, 20));
    w.push(S_ENDPGM);
    let mut p = compute_of(&w, false, LANES);
    p.vgprs = 8;
    p.layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    p.used = vec![
        entry(ResourceType::Cbv, 0),
        entry(ResourceType::Uav, 8),
        entry(ResourceType::Cbv, 16),
    ];
    p.io = StageIo::Cs(ComputeIo {
        rsrc2: 1 << 1,
        threads: [LANES, 1, 1],
    });
    p
}

/// `native` translated for any host, validated where `dxil.dll` is.
fn translate(native: &NativeProgram) -> Result<HostShader, crate::ShaderError> {
    let mut cs = emit_for(native, Target::ANY)?;
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil).expect("DXIL validation");
    }
    Ok(cs)
}

/// What stops `native`, if anything.
fn refusal(native: &NativeProgram) -> Option<&'static str> {
    match emit_for(native, Target::ANY) {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(e) => panic!("{e}"),
    }
}

/// The registers of a constant buffer read only by its `s_buffer_load`s and
/// by a join with a number the program reads behind the flag that chose
/// the number (r1smem's group B, `01d73927f08aed16` at 0x1cec): the
/// buffer's dwords are no data, the program translates, and on WARP the
/// number's edge gives the number (77, the buffer's dword 1), the other
/// edge 7.
#[test]
fn smem_a_constant_buffers_registers_reused_for_a_number_translate() {
    let cs = translate(&program(Reuse::Number)).expect("translated");
    number_on_its_edge_only(&cs);
}

/// Runs `cs`, a program of [`Reuse::Number`]'s form, on WARP: the
/// number's edge gives the number (77, the buffer's dword 1), the other
/// edge 7. Nothing where there is no WARP device.
fn number_on_its_edge_only(cs: &HostShader) {
    // One GPU test at a time (see `valu_tests::ONE_AT_A_TIME`).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(true) else {
        eprintln!("no WARP device; skipped");
        return;
    };
    let (root, pso) = host.compute_pipeline(cs).expect("a pipeline");
    for (flag, want) in [(1, 77), (0, 7)] {
        // A whole constant buffer (256 bytes): the flag, then the number.
        let mut cb = vec![0u32; 64];
        cb[..2].copy_from_slice(&[flag, 77]);
        let views = [View::Cbv(&cb), View::RawUav(LANES), View::Cbv(&cb)];
        let user = vec![0; cs.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(user))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("a dispatch");
        assert_eq!(out[0], vec![want; LANES as usize], "flag {flag}");
    }
}

/// A join of two constant buffers' dwords (s8 holds one descriptor or the
/// other) that the program reads as a number takes no number: it stays
/// refused, and so does a constant buffer's record count read directly (no
/// D3D12 query gives a constant buffer's size).
#[test]
fn smem_a_constant_buffers_dwords_read_as_numbers_stay_refused() {
    for reuse in [Reuse::Descriptor, Reuse::Count] {
        assert_eq!(
            refusal(&program(reuse)),
            Some("a constant buffer's descriptor read as data")
        );
    }
}

/// A number among the join's values does not show that only its edge is
/// read (Codex review 3983f844, P1): [`Reuse::EveryEdge`] reads the joined
/// s8 on both edges, so where the flag is 0 the console stores the
/// descriptor's dword 0. Per lane the join's proof finds that read and the
/// program stops; in the console flow (after [`CONSOLE_FLOW_LOOP`]), where
/// a descriptor's dword at a join is taken undefined without the proof,
/// the load keeps its refusal, and the per-lane form, which the program
/// then takes, stops it the same way. Before the fix the console flow
/// translated it, the store taking an undefined value on that edge.
#[test]
fn smem_a_number_join_read_on_the_descriptors_edge_stays_refused() {
    for first in [&[][..], &CONSOLE_FLOW_LOOP[..]] {
        assert_eq!(
            refusal(&program_after(first, Reuse::EveryEdge)),
            Some("a descriptor's address or format read as data"),
            "{} words first",
            first.len()
        );
    }
}

/// [`Reuse::Number`] after [`CONSOLE_FLOW_LOOP`]: the console flow keeps
/// the constant buffer's refusal, so the program takes the per-lane form,
/// whose proof finds the joined s8 read only behind the flag (before the
/// fix it ran in the console flow, the join unproven). On WARP where the
/// shader needs no wave size: its lane read asks for 32-lane waves, for
/// which the farm's WARP makes no pipeline ("The parameter is incorrect"),
/// and the first test above checks this proof's values.
#[test]
fn smem_a_number_join_in_a_console_flow_program_takes_the_per_lane_form() {
    let cs = translate(&program_after(&CONSOLE_FLOW_LOOP, Reuse::Number)).expect("translated");
    let flow = super::last_module_summary();
    assert!(flow.ends_with("; per lane"), "{flow}");
    match cs.wave_size() {
        None => number_on_its_edge_only(&cs),
        Some(n) => eprintln!("WaveSize({n}): not run on WARP"),
    }
}
