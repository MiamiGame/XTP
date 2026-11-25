//! Console wave64 compute programs on hosts with and without 64-lane waves:
//! hand-assembled GFX10 code, dispatched on every hardware adapter, checked
//! lane by lane against what the console's lane layout gives. On a host
//! without 64-lane waves, programs whose lanes read only their own 32-lane
//! half run as two host waves of 32 (see `lower::Lower::new`).

use super::host::{Dispatch, Host, Root, View, adapters};
use super::tests::{compute, emit_for};
use crate::native::{RootParam, RootType, StageIo};

const S_ENDPGM: u32 = 0xBF81_0000;

/// The wave size a DXIL container's `PSV0` requires (minimum = maximum).
fn wave_size(dxil: &[u8]) -> Option<u32> {
    let c = crate::Container::parse(dxil).ok()?;
    let psv = c.part(crate::FourCC::PSV0)?.data;
    let u = |at: usize| crate::dxbc::u32_at(psv, at).ok();
    let (min, max) = (u(4 + 16)?, u(4 + 20)?);
    (min == max).then_some(min)
}

/// A program over `threads` x 1 x 1 lanes per group with a UAV table in
/// user SGPR 0 and the group id in s1: lane t of group g stores the
/// result of `body` (v0 = t in, v2 out) at dword g * threads + t.
fn program(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut words = vec![
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0x9308_FF01,
        threads * 4, // s_mul_i32 s8, s1, threads * 4
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A06_0608, // v_add_nc_u32 v3, s8, v3
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ]);
    let mut p = compute(&words, true);
    p.layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    // One user SGPR, then the group id x.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 1 << 7,
        threads: [threads, 1, 1],
    });
    p
}

/// Swaps the 16-lane rows of each 32 (`ds_swizzle` xor 16), then mirrors
/// each row (DPP): lane t ends with t ^ 31, every read inside its half.
const IN_HALF: &[u32] = &[
    0xD8D4_401F,
    0x0100_0000, // ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
    0x7E04_02FA,
    0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf
];

fn run(host: &Host, native: &crate::NativeProgram, groups: u32) -> Result<Vec<u32>, String> {
    run_for(host, native, groups, host.target())
}

fn run_for(
    host: &Host,
    native: &crate::NativeProgram,
    groups: u32,
    target: super::Target,
) -> Result<Vec<u32>, String> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let cs = emit_for(native, target).map_err(|e| e.to_string())?;
    // The validator, where one is installed: drivers take some modules it
    // refuses.
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil.clone())
            .map_err(|e| format!("validation: {e}"))?;
    }
    let (root, pso) = host.compute_pipeline(&cs)?;
    let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
    let StageIo::Cs(io) = &native.io else {
        unreachable!()
    };
    let n = io.threads.iter().product::<u32>() * groups;
    let views = [View::RawUav(n)];
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &views,
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )?;
    Ok(out.into_iter().next().unwrap_or_default())
}

thread_local! {
    /// Set by [`every_hardware_case_translates_and_validates_without_a_device`]:
    /// the cases below translate and validate, and run on no device.
    static DRY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Every hardware adapter with an SM 6.6 device; none in a dry run.
///
/// The tests that use it are `#[ignore]`d: this PC's tests stay on WARP
/// (`XTP_TEST_ADAPTER=warp`), whose 4-lane waves run none of these
/// programs, and before they passed there without having run anything.
/// They run with `--ignored` where there is a hardware adapter (the Google
/// Cloud L4 VM: 32-lane waves), and fail if they find none;
/// [`every_hardware_case_translates_and_validates_without_a_device`] runs
/// their translations everywhere.
fn hardware() -> Vec<(String, Host)> {
    if DRY.with(std::cell::Cell::get) {
        return Vec::new();
    }
    let names = adapters().unwrap_or_else(|e| panic!("no DXGI: {e}"));
    let hosts: Vec<(String, Host)> = names
        .into_iter()
        .filter_map(|name| match Host::on(&name) {
            Ok(host) => Some((name, host)),
            Err(e) => {
                eprintln!("{name}: no SM 6.6 device ({e}); skipped");
                None
            }
        })
        .collect();
    assert!(
        !hosts.is_empty(),
        "no hardware adapter with an SM 6.6 device (XTP_TEST_ADAPTER keeps tests off hardware?)"
    );
    hosts
}

/// The host kinds every case must translate for without a device: NVIDIA
/// (32-lane waves only) and AMD RDNA (32 or 64).
const HOSTS: [super::Target; 2] = [
    super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    },
    super::Target {
        lanes_min: 32,
        lanes_max: 64,
        stride_table: false,
    },
];

/// `p` translates for both kinds of host, and the validator (where one is
/// installed) takes each module: what the hardware tests check without a
/// device.
fn translates_everywhere(label: &str, p: &crate::NativeProgram) {
    let mut validator =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
            .ok();
    for target in HOSTS {
        let mut cs = emit_for(p, target).unwrap_or_else(|e| panic!("{label}, {target:?}: {e}"));
        if let Some(v) = validator.as_mut() {
            v.validate(&mut cs.dxil)
                .unwrap_or_else(|e| panic!("{label}, {target:?}: validation: {e}"));
        }
    }
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn in_half_lane_reads_match_the_console_on_every_adapter() {
    for threads in [64, 128, 32] {
        translates_everywhere("in-half reads", &program(IN_HALF, threads));
    }
    for (name, host) in hardware() {
        let target = host.target();
        // One console wave, two per group, and a group whose second wave
        // has its lower half only.
        for threads in [64, 128, 32] {
            let p = program(IN_HALF, threads);
            let cs = emit_for(&p, target).unwrap_or_else(|e| panic!("{name}: {e}"));
            let split = !target.runs(64);
            assert_eq!(
                wave_size(&cs.dxil),
                Some(if split { 32 } else { 64 }),
                "{name}: the declared wave size"
            );
            let groups = 3;
            let out = run(&host, &p, groups).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..groups)
                .flat_map(|_| (0..threads).map(|t| t ^ 31))
                .collect();
            assert_eq!(out, want, "{name}: {threads} threads (split {split})");
            eprintln!("{name}: {threads} threads per group exact (split {split})");
        }
    }
}

/// 3-D groups: the console forms waves from the flat thread id (x
/// fastest). Lane f (f = x + X * (y + Y * z)) stores f ^ 31 at dword f.
fn program_3d(threads: [u32; 3]) -> crate::NativeProgram {
    let [x, y, _] = threads;
    let (sx, sxy) = (x.trailing_zeros(), (x * y).trailing_zeros());
    assert!(x.is_power_of_two() && y.is_power_of_two());
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000,               // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F,               // s_waitcnt lgkmcnt(0)
        0x340A_0200 | (128 + sx),  // v_lshlrev_b32 v5, log2 X, v1
        0x340C_0400 | (128 + sxy), // v_lshlrev_b32 v6, log2 XY, v2
        0x4A08_0B00,               // v_add_nc_u32 v4, v0, v5
        0x4A08_0D04,               // v_add_nc_u32 v4, v4, v6
        0xD8D4_401F,
        0x0100_0004, // ds_swizzle_b32 v1, v4 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
        0x7E04_02FA,
        0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror
        0x3406_0882, // v_lshlrev_b32 v3, 2, v4
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        S_ENDPGM,
    ];
    let mut p = compute(&words, true);
    p.layout = program(&[], 64).layout;
    // One user SGPR; thread ids x, y and z in v0..v2.
    p.io = StageIo::Cs(crate::native::ComputeIo {
        rsrc2: 1 << 1 | 2 << 11,
        threads,
    });
    p
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn three_dimensional_groups_keep_the_consoles_lanes() {
    const GROUPS: [[u32; 3]; 6] = [
        [4, 4, 4],
        [8, 8, 2],
        [2, 4, 8],
        [16, 2, 4],
        [8, 8, 4],
        [4, 8, 8],
    ];
    for threads in GROUPS {
        translates_everywhere("3-D groups", &program_3d(threads));
    }
    for (name, host) in hardware() {
        let split = !host.target().runs(64);
        for threads in GROUPS {
            let p = program_3d(threads);
            let n: u32 = threads.iter().product();
            let out = run(&host, &p, 1).unwrap_or_else(|e| panic!("{name}: {e}"));
            let want: Vec<u32> = (0..n).map(|f| f ^ 31).collect();
            assert_eq!(out, want, "{name}: {threads:?} (split {split})");
            eprintln!("{name}: {threads:?} exact (split {split})");
        }
    }
}

/// Runs `body` (v0 = t in, v2 out) on every hardware adapter in groups of
/// `threads` and checks lane t against `want(t)`; on a host without
/// 64-lane waves it must have translated in the console flow.
fn console_flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32) {
    flow_case(label, body, threads, want, true);
}

/// [`console_flow_case`], checking hosts with 64-lane waves (the per-lane
/// form) only when `native` says.
fn flow_case(label: &str, body: &[u32], threads: u32, want: impl Fn(u32) -> u32, native: bool) {
    program_case(label, &program(body, threads), want, native);
}

/// [`flow_case`] of a program built by the caller.
fn program_case(label: &str, p: &crate::NativeProgram, want: impl Fn(u32) -> u32, native: bool) {
    let StageIo::Cs(io) = &p.io else {
        unreachable!()
    };
    let threads = io.threads[0];
    translates_everywhere(label, p);
    for (name, host) in hardware() {
        let split = !host.target().runs(64);
        let groups = 2;
        let out = run(&host, p, groups).unwrap_or_else(|e| panic!("{name}: {label}: {e}"));
        let expect: Vec<u32> = (0..groups).flat_map(|_| (0..threads).map(&want)).collect();
        if !split && !native {
            if out != expect {
                eprintln!("{name}: {label}: the per-lane form differs (known)");
            }
            continue;
        }
        assert_eq!(
            out, expect,
            "{name}: {label}, {threads} threads (split {split})"
        );
        eprintln!("{name}: {label}, {threads} threads exact (split {split})");
    }
}

/// A mask counted whole: 40 lanes of the console wave, 32 in one half and
/// 8 in the other.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn masks_read_whole_count_both_halves() {
    let body = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("s_bcnt1 of lanes below 40", &body, 64, |_| 40);
    // 48 threads: the upper half is partial.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 48, |_| 40);
    // 32 threads: no upper half at all.
    console_flow_case("s_bcnt1 of lanes below 40", &body, 32, |_| 32);
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn masked_double_sum_counts_both_halves() {
    // Chaque lane fournit 0.0 ou 1.0 en double précision. Lire les lanes
    // une par une donne un ordre de sommation indépendant des permutations.
    let mut body = vec![
        0x7d88_00a9, // vcc = (41 > v0)
        0x7e02_0280,
        0x7e04_02ff,
        0x3ff0_0000, // v[1:2] = 1.0
        0xbe8e_25c1, // s[14:15] = EXEC, EXEC = -1
        0x8790_0e6a, // s[16:17] = vcc & s[14:15]
        0xd501_0003,
        0x0042_0280, // v3 = masque ? v1 : 0
        0xd501_0004,
        0x0042_0480, // v4 = masque ? v2 : 0
    ];
    for lane in 0..64 {
        let index = (128 + lane) << 9;
        body.extend([0xd760_000a, index | 259, 0xd760_000b, index | 260]);
        if lane == 0 {
            body.extend([0x7e0a_020a, 0x7e0c_020b]); // v[5:6] = s[10:11]
        } else {
            body.extend([0xd564_0005, (261 << 9) | 10]); // v[5:6] += s[10:11]
        }
    }
    body.extend([
        0xd760_000a,
        0x0001_0105,
        0xd760_000b,
        0x0001_0106, // lecture de la somme
        0xbefe_040e,
        0x7e04_1e0a, // restauration d’EXEC et conversion dans v2
    ]);
    console_flow_case("masked double sum", &body, 64, |_| 41f32.to_bits());
}

/// A lane of the other half: lane 40 is the upper half's lane 8.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lane_reads_cross_halves() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("v_readlane 40", &body, 64, |_| 40);
}

/// A branch only lanes below 20 take (all in the lower half), with a lane
/// read inside: the upper half has no lane there but still meets the lower
/// one at the exchange.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn exchanges_inside_a_branch_one_half_skips() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) v2 = readlane(10)", &body, 64, |t| {
        if t < 20 { 10 } else { 1 }
    });
}

/// The halves exchange through two group-shared dwords next to the
/// program's own LDS. With 32 KiB of LDS, DXIL's limit, there is no room:
/// the program stops with that reason rather than giving a module the
/// validator refuses. 31.5 KiB leave room. (No device needed.)
#[test]
fn the_halves_exchange_must_fit_next_to_the_programs_lds() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let with_lds = |granules: u32| {
        let mut p = program(&body, 64);
        let StageIo::Cs(io) = &mut p.io else {
            unreachable!()
        };
        io.rsrc2 |= granules << 15;
        p
    };
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    assert_eq!(
        emit_for(&with_lds(64), nvidia).err(),
        Some(crate::ShaderError::Unsupported {
            offset: 0,
            what: "a wave64 program whose LDS leaves no room for its halves' exchange",
        })
    );
    assert!(emit_for(&with_lds(63), nvidia).is_ok());
    // A host with 64-lane waves exchanges nothing.
    let rdna = super::Target {
        lanes_min: 32,
        lanes_max: 64,
        stride_table: false,
    };
    assert!(emit_for(&with_lds(64), rdna).is_ok());
}

/// A branch on the SCC of `s_and_saveexec` (some lane is left in the new
/// EXEC), with no lane left: the console skips the block, so the scalar
/// write inside never happens and every lane stores the 3 from before.
/// Taking the SCC from the saved (old) EXEC ran the block with EXEC = 0
/// and stored 7. The lane read at the start puts the program in the
/// console flow on every host.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn a_branch_on_the_scc_of_saveexec_skips_when_no_lane_is_left() {
    let body = [
        0xD760_000C,
        0x0001_5100, // v_readlane_b32 s12, v0, 40
        0xBE89_0383, // s_mov_b32 s9, 3
        0x7D88_0080, // v_cmp_gt_u32 vcc, 0, v0 (no lane)
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF84_0001, // s_cbranch_scc0 +1
        0xBE89_0387, // s_mov_b32 s9, 7
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("skip on the SCC of saveexec", &body, 64, |_| 3);
}

/// A select on a mask with a part held as a number: `vcc` (all lanes) and
/// s[12:13] built dword by dword (lanes 0-15 and 48-63). The swizzle makes
/// it a program whose lanes read only their half, which ran in two halves;
/// `lane_bit` then tested the host lane's bit, so console lanes 32-63
/// took lanes 0-31's (a debug assertion; wrong in release). It now takes
/// the console flow: lane t selects 2 where t < 16 or t >= 48, else 1.
const HELD_SELECT: &[u32] = &[
    0xD8D4_401F,
    0x0100_0000, // ds_swizzle_b32 v1, v0 (xor 0x10, unused)
    0x7D88_00C0, // v_cmp_gt_u32 vcc, 64, v0
    0xBE8C_03FF,
    0x0000_FFFF, // s_mov_b32 s12, 0xffff
    0xBE8D_03FF,
    0xFFFF_0000, // s_mov_b32 s13, 0xffff0000
    0x878A_0C6A, // s_and_b64 s[10:11], vcc, s[12:13]
    0xD501_0002,
    0x0029_0481, // v_cndmask_b32 v2, 1, 2, s[10:11]
];

fn held_select(t: u32) -> u32 {
    if !(16..48).contains(&t) { 2 } else { 1 }
}

#[test]
fn a_select_on_a_mask_held_as_a_number_translates_for_every_host() {
    // No device needed: the lowering itself asserted before.
    for target in [
        super::Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        },
        super::Target {
            lanes_min: 64,
            lanes_max: 64,
            stride_table: false,
        },
    ] {
        let cs = emit_for(&program(HELD_SELECT, 64), target)
            .unwrap_or_else(|e| panic!("{target:?}: {e}"));
        assert_eq!(
            wave_size(&cs.dxil),
            Some(if target.runs(64) { 64 } else { 32 })
        );
    }
    emit_for(&program(HELD_SELECT, 128), HOSTS[0]).unwrap();
    assert!(super::last_module_summary().ends_with("; console flow with paired lanes"));
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn a_select_on_a_mask_held_as_a_number_runs_as_the_console() {
    console_flow_case(
        "select on a mask held as a number",
        HELD_SELECT,
        64,
        held_select,
    );
    console_flow_case("numeric mask in independent waves", HELD_SELECT, 128, |t| {
        if t < 64 { held_select(t) } else { 1 }
    });
}

/// Les groupes complets utilisent les lanes appariées pour les deux sortes de sélecteur.
#[test]
fn several_console_waves_choose_the_exchange_their_lane_reads_need() {
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    assert!(emit_for(&program(&body, 64), nvidia).is_ok());
    emit_for(&program(&body, 128), nvidia).unwrap();
    assert!(super::last_module_summary().ends_with("; console flow with paired lanes"));
    assert!(emit_for(&program(&body, 96), nvidia).is_err());
    let lane31 = [body[0], 0x0001_3f00, body[2]];
    emit_for(&program(&lane31, 96), nvidia).unwrap();
    assert!(super::last_module_summary().ends_with("lane reads through group-shared memory"));
    // A computed lane (s1, the group id).
    let computed = [
        0xD760_0009,
        0x0000_0300, // v_readlane_b32 s9, v0, s1
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    emit_for(&program(&computed, 128), nvidia).unwrap();
    assert!(super::last_module_summary().ends_with("; console flow with paired lanes"));
    assert!(emit_for(&program(&computed, 96), nvidia).is_err());
}

/// Lecture de la lane 40 de chaque wave console dans un groupe complet.
/// Valeur attendue pour chaque thread :
/// - its own id (two and four console waves per group);
/// - what it read from LDS, after an `s_barrier`, at the other wave of its
///   pair (t ^ 64), which that wave wrote before the barrier;
/// - its own id, where the last console wave of the group ends first (no
///   thread of it stores: 0).
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lane_reads_cross_halves_in_groups_of_several_console_waves() {
    let own = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let lane_40 = |t: u32| (t & !63) + 40;
    for threads in [128, 256] {
        program_case("v_readlane 40", &program(&own, threads), lane_40, true);
    }
    let lane31 = [own[0], 0x0001_3f00, own[2]];
    program_case(
        "lane 31 dans 96 threads",
        &program(&lane31, 96),
        |t| (t & !63) + 31,
        true,
    );
    let lds = [
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xBF8A_0000, // s_barrier
        0x3A10_00C0, // v_xor_b32 v8, 64, v0
        0x3410_1082, // v_lshlrev_b32 v8, 2, v8
        0xD8D8_0000,
        0x0500_0008, // ds_read_b32 v5, v8
        0xD760_0009,
        0x0001_5105, // v_readlane_b32 s9, v5, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let mut p = program(&lds, 256);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    // 1 KiB: a dword per thread.
    io.rsrc2 |= 2 << 15;
    program_case(
        "LDS of the other wave, then v_readlane 40",
        &p,
        |t| lane_40(t) ^ 64,
        true,
    );
    let ended = [
        0x7D88_00FF,
        0x0000_00C0, // v_cmp_gt_u32 vcc, 192, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0007, // s_cbranch_execz to the end: the fourth console wave ends
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    program_case(
        "v_readlane 40 with the last wave ended",
        &program(&ended, 256),
        |t| if t < 192 { lane_40(t) } else { 0 },
        true,
    );
}

/// A loop lane t runs t times, with a lane read of the other half in
/// every iteration: the lower half's lanes are done after 31 iterations,
/// the upper half's go on to 63, and the console wave loops while any lane
/// does. After it, s9 holds lane 40's id, as the last iteration read it.
///
/// The per-lane form got this wrong on every host (#225); hosts with
/// 64-lane waves now run it in the console flow too, without exchanges.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn loops_run_while_any_lane_of_the_console_wave_does() {
    let body = [
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFA, // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    console_flow_case("loop t times reading lane 40", &body, 64, |t| t + 40);
}

/// DPP's row mask names the console's rows: with `row_mask:0x3` only
/// rows 0 and 1 (lanes 0 to 31) write, so the upper half keeps its value
/// although its host rows are 0 and 1 too.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn dpp_row_masks_name_console_rows() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7E04_02FA,
        0x3F00_E400, // v_mov_b32_dpp v2, v0 quad_perm:[0,1,2,3] row_mask:0x3 bank_mask:0xf
    ];
    console_flow_case(
        "DPP row_mask 0x3",
        &body,
        64,
        |t| if t < 32 { t } else { 1 },
    );
}

/// `v_readfirstlane` reads the console wave's first lane: lane 0 here, in
/// the upper half too. With DPP in the program it used to be split in
/// halves, where the upper half read its own first lane (32).
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn first_lane_reads_are_the_console_waves() {
    let body = [
        0x7E06_02FA,
        0xFF00_E400, // v_mov_b32_dpp v3, v0 quad_perm:[0,1,2,3]
        0x7E12_0500, // v_readfirstlane_b32 s9, v0
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    console_flow_case("DPP and v_readfirstlane", &body, 64, |_| 0);
    console_flow_case("first active lane of four waves", &body, 256, |t| t & !63);
}

/// `v_writelane` ignores EXEC: inside a branch only lanes below 20 take,
/// it still writes lane 40. The per-lane form never had lane 40 there.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lane_writes_reach_lanes_outside_the_branch() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0004, // s_cbranch_execz +4
        0xBE89_03FF,
        0x0000_004D, // s_mov_b32 s9, 77
        0xD761_0002,
        0x0001_5009, // v_writelane_b32 v2, s9, 40
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("if (lane < 20) writelane(40, 77)", &body, 64, |t| {
        if t == 40 { 77 } else { 1 }
    });
}

/// The loop form compilers emit: lanes that are done are added to an exit
/// mask that starts as 0, EXEC loses them, and the mask restores EXEC
/// after the loop. Lane t runs t + 1 iterations with a read of lane 40 in
/// each; afterwards every lane has s9 = 40 and counts all 64 lanes in the
/// exit mask.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn loops_with_an_exit_mask_grown_from_zero() {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFF9, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE8C_100A, // s_bcnt1_i32_b64 s12, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
        0x4A04_040C, // v_add_nc_u32 v2, s12, v2
    ];
    console_flow_case("exit-mask loop reading lane 40", &body, 64, |_| 40 + 64);
    // The second wave takes 64 more iterations than the first.
    console_flow_case("independent wave loops", &body, 128, |t| {
        (t & !63) + 40 + 64
    });
}

/// [`program`] with 512 bytes of LDS.
fn program_lds(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.rsrc2 |= 1 << 15;
    p
}

/// One console wave sharing LDS without `s_barrier`: the console runs each
/// instruction in all 64 lanes before the next, so a lane reads what a lane
/// of the other half wrote one instruction earlier. Lane t writes t to
/// dword t, then reads dword 63 - t; then writes that plus 50 to dword t
/// and reads dword t ^ 32.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn one_wave_shares_lds_in_instruction_order() {
    let body = [
        0x4C0A_00BF, // v_sub_nc_u32 v5, 63, v0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0700_0006, // ds_read_b32 v7, v6
        0x4A0E_0EB2, // v_add_nc_u32 v7, 50, v7
        0xD834_0000,
        0x0000_0704, // ds_write_b32 v4, v7
        0x3A10_00A0, // v_xor_b32 v8, 32, v0
        0x3410_1082, // v_lshlrev_b32 v8, 2, v8
        0xD8D8_0000,
        0x0200_0008, // ds_read_b32 v2, v8
    ];
    // Dword u holds (63 - u) + 50 after the second write.
    program_case(
        "LDS in instruction order",
        &program_lds(&body, 64),
        |t| (63 - (t ^ 32)) + 50,
        true,
    );
}

/// The same sharing with the halves pulled apart: only the lower half does
/// 48 buffer stores before its LDS write, so without a barrier the upper
/// half's read of the lower half's dword can come first. Lane t writes t to
/// dword t and reads dword 63 - t.
///
/// On an RTX 5070 this test and the one above also pass with the barrier
/// before each LDS access left out: that card keeps the two waves in step
/// for these programs. They check that the form gives the console's result,
/// not that the barrier is what gives it; D3D12 promises no order between a
/// group's waves without one. `lds_written_and_read_in_different_masked_blocks`
/// has a case that does fail without it.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lds_order_holds_when_one_half_is_busy() {
    let mut body = vec![
        0x3412_0082, // v_lshlrev_b32 v9, 2, v0
        0x4A12_1208, // v_add_nc_u32 v9, s8, v9
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
    ];
    for _ in 0..48 {
        // buffer_store_dword v0, v9, s[4:7], 0 offen
        body.extend_from_slice(&[0xE070_1000, 0x8001_0009]);
    }
    body.extend_from_slice(&[
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4C0A_00BF, // v_sub_nc_u32 v5, 63, v0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
        0xD834_0000,
        0x0000_0004, // ds_write_b32 v4, v0
        0xD8D8_0000,
        0x0200_0006, // ds_read_b32 v2, v6
    ]);
    program_case(
        "LDS order with a busy lower half",
        &program_lds(&body, 64),
        |t| 63 - t,
        true,
    );
}

/// The diagnostic summary of the module emitted last: what it declares and
/// the operations it calls, nothing of the console program.
#[test]
fn the_last_modules_summary_names_its_group_wave_size_and_operations() {
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    let body = [
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let mut p = program(&body, 64);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.threads = [8, 8, 1];
    emit_for(&p, nvidia).unwrap();
    let summary = super::last_module_summary();
    eprintln!("{summary}");
    assert!(
        summary.starts_with("threads 64x1x1, wave size 32; ops: "),
        "{summary}"
    );
    for op in ["barrier", "waveReadLaneAt", "flattenedThreadIdInGroup"] {
        assert!(summary.contains(op), "{op} in {summary}");
    }
}

/// Sum lane IDs within rows, add the preceding row, then carry lane 31's
/// inclusive result into the upper half. Lane 32 must start at 528.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn wave64_inclusive_prefix_joins_both_host_waves() {
    let body = [
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        0x4A02_02FA,
        0xFF01_1101, // v_add_nc_u32 v1, v1, v1 row_shr:1
        0x4A02_02FA,
        0xFF01_1201, // row_shr:2
        0x4A02_02FA,
        0xFF01_1401, // row_shr:4
        0x4A02_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E06_02FA,
        0xAF00_E402, // v_mov_b32 v3, v2, row_mask:0xa
        0x4A02_0701, // v_add_nc_u32 v1, v1, v3
        0xD760_0009,
        0x0001_3F01, // v_readlane_b32 s9, v1, 31
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x4A02_0209, // v_add_nc_u32 v1, s9, v1
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0x7E04_0301, // v_mov_b32 v2, v1
    ];
    console_flow_case("whole-wave inclusive sum", &body, 64, |n| n * (n + 1) / 2);
    console_flow_case("prefix in several console waves", &body, 128, |t| {
        let lane = t & 63;
        (lane + 1) * (t & !63) + lane * (lane + 1) / 2
    });
}

/// A scan of the lane ids with `opcode` (a VOP2 operation) as its step and
/// `ident` (an inline constant operand) in the lanes outside the mask, as
/// in the test above, then a read of lane 63: the reduction over the whole
/// console wave, left in s10. It uses v1 to v3 as scratch in every lane.
fn reduction_scan(opcode: u32, ident: u32) -> Vec<u32> {
    reduction_scan_of(opcode, ident, None)
}

/// [`reduction_scan`] with the identity as a literal when `literal` gives
/// one (`ident` is then the literal operand, 0xFF).
fn reduction_scan_of(opcode: u32, ident: u32, literal: Option<u32>) -> Vec<u32> {
    let op = opcode << 25;
    let mut words = vec![
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0000 | ident, // v_cndmask_b32 v1, ident, v0, s[0:1]
    ];
    words.extend(literal);
    words.extend([
        op | 0x0002_02FA,
        0xFF01_1101, // v_<op> v1, v1, v1 row_shr:1
        op | 0x0002_02FA,
        0xFF01_1201, // row_shr:2
        op | 0x0002_02FA,
        0xFF01_1401, // row_shr:4
        op | 0x0002_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301,         // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0200 | ident, // v_mov_b32 v3, ident
    ]);
    words.extend(literal);
    words.extend([
        0x7E06_02FA,
        0xAF00_E402,      // v_mov_b32 v3, v2 row_mask:0xa
        op | 0x0002_0701, // v_<op> v1, v1, v3
        0xD760_0009,
        0x0001_3F01,      // v_readlane_b32 s9, v1, 31
        0xBEFE_0380,      // s_mov_b32 exec_lo, 0
        0xBEFF_03C1,      // s_mov_b32 exec_hi, -1
        op | 0x0002_0209, // v_<op> v1, s9, v1
        0xD760_000A,
        0x0001_7F01, // v_readlane_b32 s10, v1, 63
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
    ]);
    words
}

/// Reductions over the whole console wave need both halves: the sum of the
/// lane ids, their maximum (lane 63, in the upper half), their minimum
/// (lane 0, in the lower half) and their bitwise or.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn whole_wave_reductions_join_both_host_waves() {
    // (operation, VOP2 opcode, identity operand, the console's result)
    for (label, opcode, ident, want) in [
        ("v_add_nc_u32", 37, 0x80, 2016),
        ("v_max_u32", 20, 0x80, 63),
        ("v_min_u32", 19, 0xC1, 0),
        ("v_or_b32", 28, 0x80, 63),
    ] {
        let mut body = reduction_scan(opcode, ident);
        body.push(0x7E04_020A); // v_mov_b32 v2, s10
        console_flow_case(&format!("whole-wave {label}"), &body, 64, |_| want);
    }
}

/// The same sum over a mask that straddles the halves: lanes below 40 (32
/// in the lower half, 8 in the upper). The other lanes keep their value.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn masked_whole_wave_sum_straddles_the_halves() {
    // The result goes through v5: the scan leaves no old value in v2.
    let mut body = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    ];
    body.extend(reduction_scan(37, 0x80));
    body.extend([
        0x7E0A_020A, // v_mov_b32 v5, s10
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E04_0305, // v_mov_b32 v2, v5
    ]);
    // 0 + 1 + ... + 39 = 780.
    console_flow_case("masked whole-wave sum", &body, 64, |t| {
        if t < 40 { 780 } else { 1 }
    });
}

/// A whole-wave section crosses the halves only in a group of exactly one
/// console wave: with a partial upper half, or none, it still stops.
#[test]
fn whole_wave_sections_in_partial_groups_still_stop() {
    let nvidia = super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    let mut body = reduction_scan(37, 0x80);
    body.push(0x7E04_020A); // v_mov_b32 v2, s10
    assert!(emit_for(&program(&body, 64), nvidia).is_ok());
    for threads in [48, 32] {
        assert_eq!(
            emit_for(&program(&body, threads), nvidia),
            Err(crate::ShaderError::Unsupported {
                offset: 0,
                what: "a whole-wave section of a wave64 on a host without 64-lane waves",
            }),
            "{threads} threads"
        );
    }
}

/// One console wave whose LDS write and read sit in different exec-masked
/// blocks (the pattern measured on upstream PR #189): lanes write dword t,
/// then lanes 0..31 read dword t + 32 in a later masked block. The console
/// runs the wave in lockstep, so the read sees the write.
///
/// `delay` has only lanes 32..63 write, after 4096 iterations of a
/// scalar-counted loop in their own masked block, so a lower half that was
/// not held back would read first.
///
/// In the console flow both halves are in step here twice over: a branch on
/// EXEC is decided by both halves through an exchange, which is a group
/// barrier, and each LDS access has a group barrier before it. So these two
/// also pass on an RTX 5070 with the second left out; the case without
/// branches in the test below does not.
fn lds_across_masked_blocks(delay: bool) -> Vec<u32> {
    let mut body = vec![
        0x7E04_0287, // v_mov_b32 v2, 7
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x4A0A_0087, // v_add_nc_u32 v5, 7, v0
    ];
    if delay {
        body.extend_from_slice(&[
            0x7D82_009F, // v_cmp_lt_u32 vcc, 31, v0
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_000A, // s_cbranch_execz join
            0xBE89_0380, // s_mov_b32 s9, 0
            // loop:
            0x160A_0AFF,
            40503,       // v_mul_u32_u24 v5, 40503, v5
            0x4A0A_0A81, // v_add_nc_u32 v5, 1, v5
            0x8109_8109, // s_add_i32 s9, s9, 1
            0xBF0A_FF09,
            4096,        // s_cmp_lt_u32 s9, 4096
            0xBF85_FFF9, // s_cbranch_scc1 loop
            0xD834_0000,
            0x0000_0504, // ds_write_b32 v4, v5
            // join:
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        ]);
    } else {
        body.extend_from_slice(&[
            0xD834_0000,
            0x0000_0504, // ds_write_b32 v4, v5
        ]);
    }
    body.extend_from_slice(&[
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0002, // s_cbranch_execz join
        0xD8D8_0080,
        0x0200_0004, // ds_read_b32 v2, v4 offset:128
        // join:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ]);
    body
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lds_written_and_read_in_different_masked_blocks() {
    program_case(
        "LDS written by every lane, then read by the lower half",
        &program_lds(&lds_across_masked_blocks(false), 64),
        |t| if t < 32 { t + 39 } else { 7 },
        true,
    );
    // What a writer leaves: 4096 times v = v_mul_u32_u24(40503, v) + 1.
    let late = |t: u32| {
        (0..4096).fold(t + 7, |v, _| {
            (v & 0xFF_FFFF).wrapping_mul(40503).wrapping_add(1)
        })
    };
    program_case(
        "LDS written late by the upper half, then read by the lower",
        &program_lds(&lds_across_masked_blocks(true), 64),
        |t| if t < 32 { late(t + 32) } else { 7 },
        true,
    );
    // The same masks with no branch around either access: nothing but the
    // barrier before each LDS access orders the halves. With it left out,
    // the lower half of an RTX 5070 reads 0 here every time (20 of 20 runs):
    // this is the case that shows the barrier is needed.
    let masked = [
        0x7E04_0287, // v_mov_b32 v2, 7
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0x4A0A_0087, // v_add_nc_u32 v5, 7, v0
        0x7D82_009F, // v_cmp_lt_u32 vcc, 31, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xD834_0000,
        0x0000_0504, // ds_write_b32 v4, v5
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7D88_00A0, // v_cmp_gt_u32 vcc, 32, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xD8D8_0080,
        0x0200_0004, // ds_read_b32 v2, v4 offset:128
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    program_case(
        "LDS written by the upper half and read by the lower, no branches",
        &program_lds(&masked, 64),
        |t| if t < 32 { t + 39 } else { 7 },
        true,
    );
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn backward_permute_wraps_within_each_half() {
    let body = [0x3406_0082, 0xDACC_0004, 0x0200_0003];
    console_flow_case("backward permute", &body, 64, |lane| {
        (lane & !31) | ((lane + 1) & 31)
    });
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn backward_permute_respects_both_exec_masks() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    console_flow_case("masked backward permute", &body, 64, |lane| match lane {
        0..19 => lane + 1,
        19 => 0,
        _ => 1,
    });
}

/// `ds_bpermute_b32` where lane t reads v0 of lane t + 1 of its half, in the
/// masks, group sizes and forms the two tests above leave out.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn backward_permute_in_other_masks_groups_and_forms() {
    let next = |t: u32| (t & !31) | ((t + 1) & 31);
    // EXEC over part of each half: lanes 0..21 and 42..63.
    let straddle = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A02_0096, // v_add_nc_u32 v1, 22, v0
        0x3602_02BF, // v_and_b32 v1, 63, v1
        0x7D88_02AC, // v_cmp_gt_u32 vcc, 44, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    let active = |t: u32| !(22..42).contains(&t);
    console_flow_case(
        "backward permute under a mask over both halves",
        &straddle,
        64,
        |t| match (active(t), active(next(t))) {
            (false, _) => 1,
            (true, true) => next(t),
            (true, false) => 0,
        },
    );
    let plain = [
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
    ];
    // Two console waves per group, then a group whose upper half has 16
    // lanes (a source lane that is not there reads 0), then one half only.
    console_flow_case("backward permute, two console waves", &plain, 128, next);
    console_flow_case("backward permute, 48 threads", &plain, 48, |t| {
        if next(t) < 48 { next(t) } else { 0 }
    });
    console_flow_case("backward permute, 32 threads", &plain, 32, next);
    // Beside a lane read across the halves, which takes the console flow on
    // a host without 64-lane waves.
    let with_read = [
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xDACC_0004,
        0x0200_0003, // ds_bpermute_b32 v2, v3, v0 offset:4
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    console_flow_case("backward permute beside a lane read", &with_read, 64, |t| {
        next(t) + 40
    });
}

/// Float reductions over the whole console wave: the lane ids as floats,
/// their sum, their maximum (lane 63, in the upper half) and their minimum
/// (lane 0, in the lower half), with the identities the operations need
/// (0.0, -inf, +inf). Each half's result crosses as its bits.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn float_whole_wave_reductions_join_both_host_waves() {
    // (operation, VOP2 opcode, identity operand, its literal, the result)
    for (label, opcode, ident, literal, want) in [
        ("v_add_f32", 3, 0x80, None, 2016),
        ("v_max_f32", 16, 0xFF, Some(0xFF80_0000), 63),
        ("v_min_f32", 15, 0xFF, Some(0x7F80_0000), 0),
    ] {
        let mut body = vec![0x7E00_0D00]; // v_cvt_f32_u32 v0, v0
        body.extend(reduction_scan_of(opcode, ident, literal));
        body.extend([
            0x7E00_0F00, // v_cvt_u32_f32 v0, v0
            0x7E04_020A, // v_mov_b32 v2, s10
            0x7E04_0F02, // v_cvt_u32_f32 v2, v2
        ]);
        console_flow_case(&format!("whole-wave {label}"), &body, 64, |_| want);
    }
}

/// A float sum over a mask that straddles the halves (lanes below 40), and
/// the inclusive float prefix sum of the lane ids, whose upper half starts
/// from the lower half's total.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn masked_and_prefix_float_sums_cross_the_halves() {
    let mut body = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0x7E00_0D00, // v_cvt_f32_u32 v0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
    ];
    body.extend(reduction_scan(3, 0x80));
    body.extend([
        0x7E0A_020A, // v_mov_b32 v5, s10
        0x7E0A_0F05, // v_cvt_u32_f32 v5, v5
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E00_0F00, // v_cvt_u32_f32 v0, v0
        0x7E04_0305, // v_mov_b32 v2, v5
    ]);
    console_flow_case("masked whole-wave float sum", &body, 64, |t| {
        if t < 40 { 780 } else { 1 }
    });
    let prefix = [
        0x7E00_0D00, // v_cvt_f32_u32 v0, v0
        0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
        0xD501_0001,
        0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        0x0602_02FA,
        0xFF01_1101, // v_add_f32 v1, v1, v1 row_shr:1
        0x0602_02FA,
        0xFF01_1201, // row_shr:2
        0x0602_02FA,
        0xFF01_1401, // row_shr:4
        0x0602_02FA,
        0xFF01_1801, // row_shr:8
        0xD778_0002,
        0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        0x7E06_0280, // v_mov_b32 v3, 0
        0x7E06_02FA,
        0xAF00_E402, // v_mov_b32 v3, v2 row_mask:0xa
        0x0602_0701, // v_add_f32 v1, v1, v3
        0xD760_0009,
        0x0001_3F01, // v_readlane_b32 s9, v1, 31
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x0602_0209, // v_add_f32 v1, s9, v1
        0xBEFE_0400, // s_mov_b64 exec, s[0:1]
        0x7E04_0F01, // v_cvt_u32_f32 v2, v1
        0x7E00_0F00, // v_cvt_u32_f32 v0, v0
    ];
    console_flow_case("whole-wave inclusive float sum", &prefix, 64, |n| {
        n * (n + 1) / 2
    });
}

/// A select whose selector is a number, not a compare result: a constant
/// in each mask word, and in the second program a word with a bit that
/// depends on the group (s8 is 256 in group 1). The lane takes its own
/// bit of the console's mask: lanes 32..63 of a wave64 the high word's,
/// each lane of a wave32 the one word's.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn selects_on_masks_held_as_numbers() {
    let constant = [
        0xBE8A_03FF,
        0x0F0F_0F0F, // s_mov_b32 s10, 0x0f0f0f0f
        0xBE8B_03FF,
        0x3333_3333, // s_mov_b32 s11, 0x33333333
        0x7E02_0287, // v_mov_b32 v1, 7
        0xD501_0002,
        0x002A_0101, // v_cndmask_b32 v2, v1, v0, s[10:11]
    ];
    let built = [
        0xBE8A_03FF,
        0x0F0F_0F0F, // s_mov_b32 s10, 0x0f0f0f0f
        0x880A_080A, // s_or_b32 s10, s10, s8
        0xBE8B_03FF,
        0x3333_3333, // s_mov_b32 s11, 0x33333333
        0x7E02_0287, // v_mov_b32 v1, 7
        0xD501_0002,
        0x002A_0101, // v_cndmask_b32 v2, v1, v0, s[10:11]
    ];
    let cases = [("constant", &constant[..], false), ("built", &built, true)];
    let forms = [(true, 64), (false, 64), (false, 32)];
    for (label, body, _) in cases {
        for (wave64, threads) in forms {
            let mut p = program(body, threads);
            p.wave64 = wave64;
            translates_everywhere(label, &p);
        }
    }
    for (name, host) in hardware() {
        for (label, body, by_group) in cases {
            for (wave64, threads) in forms {
                let mut p = program(body, threads);
                p.wave64 = wave64;
                let groups = 2;
                let out = run(&host, &p, groups)
                    .unwrap_or_else(|e| panic!("{name}: {label} wave64 {wave64}: {e}"));
                let want: Vec<u32> = (0..groups)
                    .flat_map(|g| {
                        // s8 = g * threads * 4: bit 8 in group 1 of 64, bit 7 of 32.
                        let lo = 0x0F0F_0F0F | if by_group { g * threads * 4 } else { 0 };
                        (0..threads).map(move |t| {
                            let (word, lane) = if wave64 && t >= 32 {
                                (0x3333_3333, t - 32)
                            } else {
                                (lo, t % 32)
                            };
                            if word >> lane & 1 != 0 { t } else { 7 }
                        })
                    })
                    .collect();
                assert_eq!(
                    out, want,
                    "{name}: a select on a {label} mask, wave64 {wave64}, {threads} threads"
                );
                eprintln!(
                    "{name}: a select on a {label} mask, wave64 {wave64}, {threads} threads exact"
                );
            }
        }
    }
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn numeric_selects_beside_in_half_reads() {
    let mut body = IN_HALF.to_vec();
    body.extend([
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_03C1, // s_mov_b32 s11, -1
    ]);
    let select = body.len();
    body.extend([
        0xD501_0001,
        0x0029_0F02, // v_cndmask_b32 v1, v2, 7, s[10:11] (unused)
    ]);
    let wave32 = super::Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    let unused = program(&body, 128);
    let shader = emit_for(&unused, wave32).unwrap();
    assert_eq!(wave_size(&shader.dxil), Some(32));
    program_case(
        "unused numeric select beside in-half reads",
        &unused,
        |t| t ^ 31,
        true,
    );

    body[select] = 0xD501_0002; // write v2, the stored result
    let used = program(&body, 64);
    assert!(emit_for(&used, wave32).is_ok());
    program_case(
        "used numeric select beside in-half reads",
        &used,
        |t| if t < 32 { t ^ 31 } else { 7 },
        true,
    );
}

/// A lane read inside a divergent branch in a program that also has a
/// whole-wave section: the lane read takes the console flow on every host,
/// and the section is lowered there as it is elsewhere. Lanes below 20
/// read lane 10; every lane then adds the wave's sum of lane ids.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lane_reads_in_branches_beside_a_whole_wave_section() {
    let mut body = reduction_scan(37, 0x80); // s10 = 2016
    body.extend([
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x4A04_040A, // v_add_nc_u32 v2, s10, v2
    ]);
    console_flow_case(
        "a lane read in a branch beside a whole-wave section",
        &body,
        64,
        |t| 2016 + if t < 20 { 10 } else { 1 },
    );
}

/// EXEC restored dword by dword from a mask: the structure pass holds such
/// a mask as a number, and that number is known only as the mask's own
/// dword. The lowering used to follow that in a circle until the stack
/// ran out; it stops with a reason now.
#[test]
fn a_mask_copied_dword_by_dword_stops_instead_of_recursing() {
    let body = [
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x878C_6A7E, // s_and_b64 s[12:13], exec, vcc
        0xBEFE_030C, // s_mov_b32 exec_lo, s12
        0xBEFF_030D, // s_mov_b32 exec_hi, s13
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    ];
    // In the console flow (a host without 64-lane waves) and per lane.
    for (lanes_min, lanes_max, why) in [
        (32, 32, "an opaque lane mask"),
        (32, 64, "a lane mask holding lanes elsewhere read as data"),
    ] {
        let target = super::Target {
            lanes_min,
            lanes_max,
            stride_table: false,
        };
        let e = emit_for(&program(&body, 64), target)
            .expect_err("stops")
            .to_string();
        assert!(e.contains(why), "{lanes_min}..{lanes_max}: {e}");
    }
}

/// `v_readlane` takes its lane from the low six bits of the operand
/// (five in wave32): lane 70 is lane 6, so every lane stores 6. The host's
/// `WaveReadLaneAt` of lane 70 is undefined.
const READLANE_70: &[u32] = &[
    0xBE8D_03FF,
    70, // s_mov_b32 s13, 70
    0xD760_0009,
    0x0000_1B00, // v_readlane_b32 s9, v0, s13
    0x7E04_0209, // v_mov_b32 v2, s9
];

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn a_lane_read_past_the_wave_reads_its_low_bits() {
    console_flow_case("v_readlane 70", READLANE_70, 64, |_| 6);
}

/// LDS in a group of 48 threads: the console wave's upper half has 16
/// lanes. Lane t writes t to dword t, then reads dword (t + 16) % 48,
/// across the halves both ways, one instruction after the write (no
/// `s_barrier`: the console runs each instruction in every lane first).
const LDS_48: &[u32] = &[
    0x4A0A_0090, // v_add_nc_u32 v5, 16, v0
    0x4E0C_0AB0, // v_subrev_nc_u32 v6, 48, v5
    0x7D88_0AB0, // v_cmp_gt_u32 vcc, 48, v5
    0x020A_0B06, // v_cndmask_b32 v5, v6, v5, vcc
    0x3408_0082, // v_lshlrev_b32 v4, 2, v0
    0x340C_0A82, // v_lshlrev_b32 v6, 2, v5
    0xD834_0000,
    0x0000_0004, // ds_write_b32 v4, v0
    0xD8D8_0000,
    0x0200_0006, // ds_read_b32 v2, v6
];

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn lds_in_a_group_of_48_threads_reads_both_halves() {
    program_case(
        "LDS, 48 threads",
        &program_lds(LDS_48, 48),
        |t| (t + 16) % 48,
        true,
    );
}

/// Without a device: on a 32-lane host that program runs in two waves of
/// 32 in the console flow, a group barrier before each LDS access.
#[test]
fn lds_in_a_group_of_48_threads_takes_the_console_flow_on_32_lane_hosts() {
    let nvidia = HOSTS[0];
    let cs = emit_for(&program_lds(LDS_48, 48), nvidia).unwrap();
    assert_eq!(wave_size(&cs.dxil), Some(32));
    let summary = super::last_module_summary();
    assert!(
        summary.starts_with("threads 48x1x1, wave size 32;"),
        "{summary}"
    );
    assert!(summary.contains("barrier"), "{summary}");
}

/// Every hardware case above translates for both kinds of host and the
/// validator takes it: their dry run, which needs no device and runs on
/// every PC (the hardware part runs with `--ignored`).
#[test]
fn every_hardware_case_translates_and_validates_without_a_device() {
    DRY.with(|d| d.set(true));
    in_half_lane_reads_match_the_console_on_every_adapter();
    three_dimensional_groups_keep_the_consoles_lanes();
    masks_read_whole_count_both_halves();
    masked_double_sum_counts_both_halves();
    lane_reads_cross_halves();
    lane_reads_cross_halves_in_groups_of_several_console_waves();
    exchanges_inside_a_branch_one_half_skips();
    a_branch_on_the_scc_of_saveexec_skips_when_no_lane_is_left();
    a_select_on_a_mask_held_as_a_number_runs_as_the_console();
    loops_run_while_any_lane_of_the_console_wave_does();
    dpp_row_masks_name_console_rows();
    first_lane_reads_are_the_console_waves();
    lane_writes_reach_lanes_outside_the_branch();
    loops_with_an_exit_mask_grown_from_zero();
    one_wave_shares_lds_in_instruction_order();
    lds_order_holds_when_one_half_is_busy();
    wave64_inclusive_prefix_joins_both_host_waves();
    whole_wave_reductions_join_both_host_waves();
    masked_whole_wave_sum_straddles_the_halves();
    lds_written_and_read_in_different_masked_blocks();
    backward_permute_wraps_within_each_half();
    backward_permute_respects_both_exec_masks();
    backward_permute_in_other_masks_groups_and_forms();
    float_whole_wave_reductions_join_both_host_waves();
    masked_and_prefix_float_sums_cross_the_halves();
    selects_on_masks_held_as_numbers();
    numeric_selects_beside_in_half_reads();
    lane_reads_in_branches_beside_a_whole_wave_section();
    a_lane_read_past_the_wave_reads_its_low_bits();
    paired_lanes_share_lds_and_count_their_own_mask_bits();
    paired_lanes_execute_a_half_precision_whole_wave_section();
    paired_reads_ignore_exec_and_lds_writes_keep_their_order();
    paired_exec_dwords_first_lane_and_partial_mbcnt();
    lds_in_a_group_of_48_threads_reads_both_halves();
    a_loop_exit_mask_read_inside_the_loop_counts_the_lanes_that_left();
    DRY.with(|d| d.set(false));
}

/// A mask with no lane read as data, in a block where structurisation
/// does not know EXEC yet (no vector instruction before it): 0, whatever
/// the lanes. It stopped as a mask holding lanes elsewhere. (No device.)
#[test]
fn an_empty_mask_read_as_data_is_zero() {
    let body = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE8A_046A, // s_mov_b64 s[10:11], vcc
        0xBF82_0000, // s_branch +0 (a block of its own)
        0x878C_800A, // s_and_b64 s[12:13], s[10:11], 0
        0xBE89_100C, // s_bcnt1_i32_b64 s9, s[12:13]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    translates_everywhere("an empty mask read as data", &program(&body, 64));
}

/// The exit mask of a loop lanes leave at different iterations, counted
/// inside the loop: it holds lanes that are no longer in the loop, which
/// no ballot there sees. Lane t runs the loop t + 1 times; at the last
/// iteration (lane 63's 64th) all 64 lanes have left, so every lane ends
/// with 64. The per-lane form stopped ("a lane mask holding lanes
/// elsewhere read as data"); the console flow counts them.
const LOOP_EXIT_COUNT: &[u32] = &[
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBE8A_0480, // s_mov_b64 s[10:11], 0
    0x4A02_0281, // 8: v_add_nc_u32 v1, 1, v1
    0x7D88_0101, // v_cmp_gt_u32 vcc, v1, v0
    0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
    0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
    0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
    0xBF89_FFFA, // s_cbranch_execnz -> 8
    0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    0x7E04_0209, // v_mov_b32 v2, s9
];

/// [`LOOP_EXIT_COUNT`] in wave32, for groups of 32: every lane ends with
/// 32. Per lane, each lane counted the lanes of its own last iteration,
/// itself alone (1), on every host: the exit mask's join at the loop header
/// is "no lane" in every lane still in the loop, so the mask looked like
/// one the lanes there hold whole.
const LOOP_EXIT_COUNT_32: &[u32] = &[
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBE8A_0380, // s_mov_b32 s10, 0
    0x4A02_0281, // 8: v_add_nc_u32 v1, 1, v1
    0x7D88_0101, // v_cmp_gt_u32 vcc_lo, v1, v0
    0x880A_0A6A, // s_or_b32 s10, vcc_lo, s10
    0xBE89_0F0A, // s_bcnt1_i32_b32 s9, s10
    0x8A7E_0A7E, // s_andn2_b32 exec_lo, exec_lo, s10
    0xBF89_FFFA, // s_cbranch_execnz -> 8
    0x887E_0A7E, // s_or_b32 exec_lo, exec_lo, s10
    0x7E04_0209, // v_mov_b32 v2, s9
];

/// [`program`] in wave32.
fn program_wave32(body: &[u32], threads: u32) -> crate::NativeProgram {
    let mut p = program(body, threads);
    p.wave64 = false;
    p
}

/// The form `p` is lowered in for each of [`HOSTS`] (what
/// [`super::last_module_summary`] ends with): `flows[n]` for `HOSTS[n]`.
fn assert_flows(label: &str, p: &crate::NativeProgram, flows: [&str; 2]) {
    for (target, flow) in HOSTS.into_iter().zip(flows) {
        emit_for(p, target).unwrap_or_else(|e| panic!("{label}, {target:?}: {e}"));
        let summary = super::last_module_summary();
        assert!(
            summary.ends_with(&format!("; {flow}")),
            "{label}, {target:?}: {flow} wanted: {summary}"
        );
    }
}

/// Without a device: the exit mask read inside the loop takes the console
/// flow on every host, in wave64 (in halves without 64-lane waves) and in
/// wave32. A host with 64-lane waves lowered the wave64 form per lane, and
/// every host the wave32 one.
#[test]
fn a_loop_exit_mask_read_as_data_takes_the_console_flow_on_every_host() {
    let halves = ["console flow in halves", "console flow"];
    assert_flows("wave64", &program(LOOP_EXIT_COUNT, 64), halves);
    let whole = ["console flow", "console flow"];
    assert_flows("wave32", &program_wave32(LOOP_EXIT_COUNT_32, 32), whole);
    // A mask read as data that no join feeds stays per lane where the host
    // runs the console's wave.
    let body = [
        0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
        0xBE89_106A, // s_bcnt1_i32_b64 s9, vcc
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let split = ["console flow in halves", "per lane"];
    assert_flows("vcc counted", &program(&body, 64), split);
}

/// The exit mask's join read itself, before the loop adds this iteration's
/// lanes (the mask starts as EXEC and 0, so it is a lane mask from the
/// start): per lane it is "no lane", the constant the join resolves to,
/// which was read as 0 without asking whether the lanes that left are in
/// it. Lane t runs t + 1 iterations; at the last one, lanes 0 to 62 have
/// left, so every lane ends with 63. It takes the console flow too.
const LOOP_EXIT_JOIN_COUNT: &[u32] = &[
    0x7E02_0280, // v_mov_b32 v1, 0
    0x878A_807E, // s_and_b64 s[10:11], exec, 0
    0xBE89_100A, // 8: s_bcnt1_i32_b64 s9, s[10:11]
    0x4A02_0281, // v_add_nc_u32 v1, 1, v1
    0x7D88_0101, // v_cmp_gt_u32 vcc, v1, v0
    0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
    0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
    0xBF89_FFFA, // s_cbranch_execnz -> 8
    0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    0x7E04_0209, // v_mov_b32 v2, s9
];

#[test]
fn a_loop_exit_join_read_as_data_is_not_taken_for_no_lane() {
    let halves = ["console flow in halves", "console flow"];
    assert_flows("join counted", &program(LOOP_EXIT_JOIN_COUNT, 64), halves);
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn a_loop_exit_mask_read_inside_the_loop_counts_the_lanes_that_left() {
    let halves = ["console flow in halves", "console flow"];
    assert_flows("wave64", &program(LOOP_EXIT_COUNT, 64), halves);
    console_flow_case("loop exit mask counted inside", LOOP_EXIT_COUNT, 64, |_| 64);
    assert_flows("join", &program(LOOP_EXIT_JOIN_COUNT, 64), halves);
    console_flow_case("loop exit join counted", LOOP_EXIT_JOIN_COUNT, 64, |_| 63);
    let whole = ["console flow", "console flow"];
    let p = program_wave32(LOOP_EXIT_COUNT_32, 32);
    assert_flows("wave32", &p, whole);
    program_case("wave32 loop exit mask counted inside", &p, |_| 32, true);
}

/// Lanes from different console waves exchange LDS across s_barrier.
/// The read after the predicated atomic uses the original lane mask and
/// its return value. Each active lane updates a distinct dword.
#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn paired_lanes_share_lds_and_count_their_own_mask_bits() {
    let body = [
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD766_0001,
        0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xD834_0000,
        0x0000_0104, // ds_write_b32 v4, v1
        0xBF8A_0000, // s_barrier
        0x3A0A_00C0, // v_xor_b32 v5, 64, v0
        0x340A_0A82, // v_lshlrev_b32 v5, 2, v5
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D88_02A8, // v_cmp_gt_u32 vcc, 40, v1
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xD880_0000,
        0x0200_0005, // ds_add_rtn_u32 v2, v5, v0
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBF8A_0000, // s_barrier
        0xD8D8_0000,
        0x0600_0004, // ds_read_b32 v6, v4
        0x4A04_0D02, // v_add_nc_u32 v2, v2, v6
    ];
    program_case(
        "masked LDS add and mbcnt",
        &program_lds(&body, 128),
        |t| {
            let lane = t & 63;
            if lane < 40 { 2 * lane + (t ^ 64) } else { lane }
        },
        true,
    );
}

/// Chaque nibble choisit une lane dans la rangée opposée. On conserve
/// le motif lu et le résultat arrondi de deux opérations FP16 successives.
#[test]
#[ignore = "hardware: 32-lane waves (see hardware())"]
fn paired_lanes_execute_a_half_precision_whole_wave_section() {
    let body = [
        0x3608_00a0, // v4 = id & 32
        0x3408_0886, // v4 <<= 6
        0x4a02_00ff,
        0x3800,
        0x4a02_0901, // v1 = 0x3800 + id + ((id & 32) << 6)
        0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
        0xbe8c_03ff,
        0x0123_4567,
        0xbe8d_03ff,
        0xfedc_ba98,
        0xd778_0002,
        (13 << 18) | (12 << 9) | 257,
        0x7e06_0302, // v3 = motif FP16 choisi
        0x7e0c_02ff,
        0x6800,      // v6 = 2048 en FP16
        0x6404_0d02, // v2 = v2 + v6
        0x6604_0d02, // v2 = v2 - v6
        0xD760_0009,
        0x0001_4102, // v_readlane_b32 s9, v2, 32
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
        0x4a04_0702, // v2 += v3
    ];
    paired_case("arrondi FP16 après permutation", &body, |t| {
        let pick = if t & 8 == 0 { 7 - (t & 7) } else { t & 15 };
        let source = ((t & !15) ^ 16) + pick;
        // x est dans [0,5 ; 0,57[ ou [2 ; 2,25[. Avec un pas de 2
        // à 2048, les deux arrondis (au plus proche et vers zéro)
        // donnent (2048 + x) - 2048 = 0 ou 2. Réassocier donnerait x.
        let bits = if source & 32 == 0 { 0 } else { 0x4000 };
        let x = 0x3800 + source + ((source & 32) << 6);
        bits + 0x4000 + x
    });
}

fn paired_case(label: &str, body: &[u32], want: impl Fn(u32) -> u32) {
    let p = program(body, 128);
    let mut cs = emit_for(&p, HOSTS[0]).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(super::last_module_summary().ends_with("console flow with paired lanes"));
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator.validate(&mut cs.dxil).unwrap();
    }
    for (name, host) in hardware() {
        if host.target().runs(32) {
            let out = run_for(&host, &p, 1, HOSTS[0]).unwrap();
            assert_eq!(
                out,
                (0..128).map(&want).collect::<Vec<_>>(),
                "{name}: {label}"
            );
        }
    }
}

/// Le registre de la lane 40 vaut 140 dans la première wave. Il reste
/// lisible par les lanes 0 à 39 dans la branche, même quand EXEC[40] vaut 0.
const MASKED_READ: &[u32] = &[
    0x4a0a_00ff,
    100,         // v5 = 100 + id
    0x3602_00bf, // v1 = id & 63
    0x7e04_0280,
    0x7d88_02a8, // vcc = lane < 40
    0xbe8a_246a,
    0xbf88_0003,
    0xd760_0009,
    0x0001_5105, // s9 = v5[40]
    0x7e04_0209,
    0x88fe_0a7e,
];

/// Deux écritures successives par adresse, séparées seulement par waitcnt.
/// L'échange des adresses entre demi-waves laisse 2 dans chaque case.
fn lds_two_writes(atomic: bool) -> Vec<u32> {
    let mut body = vec![
        0x3408_0082, // v4 = 4 * id
        0x3a0a_00a0, // v5 = id ^ 32
        0x340a_0a82,
        0x7e02_0281,
        0xd834_0000,
        0x0000_0104, // LDS[v4] = 1
        0xbf8c_c07f,
        0x7e02_0282,
    ];
    if atomic {
        body.extend([0xd880_0000, 0x0600_0105]); // v6 = atomic_add(LDS[v5], 2)
    } else {
        body.extend([0xd834_0000, 0x0000_0105]);
    }
    body.extend([0xbf8a_0000, 0xd8d8_0000, 0x0200_0004]);
    if atomic {
        body.push(0x4a04_0d02); // 3 relu + 1 retourné
    }
    body.extend([0xd760_0009, 0x0001_3f02, 0x7e04_0209]); // lane 31
    body
}

#[test]
fn partial_groups_refuse_divergent_exchanges_and_unordered_lds() {
    let even_lanes = [
        0xbe89_0380, // s9 = 0 avant la branche
        0x3602_0081, // v1 = id & 1
        0x7d84_0280, // vcc = v1 == 0
        0xbe8a_246a,
        0xbf88_0002, // reconvergence sans autre barrière
        0xd760_0009,
        0x0001_3f00,
        0x88fe_0a7e,
        0x7e04_0209,
    ];
    for body in [
        even_lanes.as_slice(),
        MASKED_READ,
        &lds_two_writes(false),
        &lds_two_writes(true),
    ] {
        assert!(emit_for(&program_lds(body, 96), HOSTS[0]).is_err());
    }
    // Sans lecture de lane, les 32 Kio restent entièrement disponibles.
    let mut p = program(IN_HALF, 128);
    let StageIo::Cs(io) = &mut p.io else {
        unreachable!()
    };
    io.rsrc2 |= 64 << 15;
    emit_for(&p, HOSTS[0]).unwrap();
    assert!(super::last_module_summary().ends_with("console flow with paired lanes"));
}

#[test]
fn whole_wave_read_of_an_incoming_vgpr_is_not_a_local_value() {
    let p = program_wave32(
        &[
            0xbe8e_3dc1, // s_or_saveexec_b32 s14, -1
            0xd760_000a,
            0x0001_0b00, // s10 = v0[5]
            0xbe8b_030a, // s11 = s10
            0xbefe_030e, // exec_lo = s14
            0x7e02_020b, // v1 = s11
            0x7e04_0301,
        ],
        32,
    );
    let err = emit_for(&p, HOSTS[0]).unwrap_err();
    assert!(err.to_string().contains("whole-wave section"), "{err}");
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn paired_reads_ignore_exec_and_lds_writes_keep_their_order() {
    program_case(
        "lecture de la lane 40 masquée",
        &program(MASKED_READ, 128),
        |t| if t & 63 < 40 { (t & !63) + 140 } else { 0 },
        true,
    );
    for atomic in [false, true] {
        program_case(
            "deux écritures LDS",
            &program_lds(&lds_two_writes(atomic), 128),
            |_| if atomic { 4 } else { 2 },
            true,
        );
    }
    // Les lanes paires publient sous une garde, puis chaque lane lit
    // l'adresse de la lane paire voisine de l'autre demi-wave, sans s_barrier.
    let body = [
        0x3408_0082,
        0x3602_0081,
        0x7d84_0280,
        0xbe8a_246a,
        0xd834_0000,
        0x0000_0004,
        0x88fe_0a7e,
        0x360a_00c2, // v5 = id & -2
        0x3a0a_0aa0,
        0x340a_0a82,
        0xd8d8_0000,
        0x0200_0005,
        0xd760_0009,
        0x0001_0100, // base de la wave (lane 0)
        0x4a04_0409,
    ];
    program_case(
        "LDS entre lanes, sans barrière",
        &program_lds(&body, 128),
        |t| ((t & !1) ^ 32) + (t & !63),
        true,
    );
}

#[test]
#[ignore = "hardware: 32- or 64-lane waves (see hardware())"]
fn paired_exec_dwords_first_lane_and_partial_mbcnt() {
    let body = [
        0xbefe_0380, // exec_lo = 0
        0xbeff_0388, // exec_hi = 8 : première lane active = 35
        0x7e12_0500, // s9 = readfirstlane(v0)
        0xbefe_04c1, // exec = -1
        0xbe8a_03ff,
        0x8000_0005,
        0xbe8b_03ff,
        0x4000_000a,
        0xd765_0002,
        0x0001_000a, // mbcnt_lo(s10, 0)
        0xd766_0002,
        0x0002_040b, // mbcnt_hi(s11, v2)
        0x4a04_0409,
    ];
    paired_case("masques EXEC et mbcnt", &body, |t| {
        let below = (1u64 << (t & 63)) - 1;
        (t & !63) + 35 + (0x4000_000a_8000_0005 & below).count_ones()
    });
}
