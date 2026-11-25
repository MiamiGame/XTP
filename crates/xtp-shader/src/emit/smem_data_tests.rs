//! End-to-end tests of descriptors whose dwords the code also moves or
//! joins as numbers (task meshdata), and of the lane masks joined with
//! numbers (`lower/flow.rs`, `dead_at_join`): synthetic GFX10 programs
//! lowered to DXIL, validated by `dxil.dll` when present, and run on the
//! host's hardware GPU and on WARP.

use super::container::SigElement;
use super::host::{Dispatch, Draw, Host, Root, Stride, View};
use super::{HostShader, Target, lower, validator};
use crate::ShaderError;
use crate::native::{
    ComputeIo, HwStage, MeshIo, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo,
    Usage, UsedResource, VsOutput,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// Lowers a wave32 compute program of 32 threads with the table of views at
/// s0: structured buffers of dwords at entries 0 and 1, a raw UAV at entry
/// 2. `uniform` handles only, every access the whole wave's.
fn build(words: &[u32]) -> HostShader {
    let range = |kind, table_offset_dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        float_mode: 0xc0,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 4),
            range(ResourceType::Uav, 16, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
    })
    .expect("lower");
    assert_eq!(lowered.nonuniform_handles, 0);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        group_base: lowered.group_base,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        stride_table: lowered.stride_table,
        run_time_buffers: lowered.run_time_buffers,
    };
    validate(&shader);
    shader
}

/// `shader` checked by dxil.dll, where it is.
fn validate(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .unwrap_or_else(|e| panic!("DXIL validation failed: {e}"));
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
}

/// The devices these tests run on, by label: the first hardware adapter
/// ("hardware"), unless `XTP_TEST_ADAPTER` keeps hardware out (`warp` on the
/// person's PC), and WARP. Without its device a run is left out, and with
/// `XTP_REQUIRE_GPU` set (the L4's `-GpuTests`) a missing hardware adapter
/// fails the test: there, "hardware" is a real GPU, never WARP standing in.
fn devices() -> Vec<(&'static str, Host)> {
    let strict = std::env::var_os("XTP_REQUIRE_GPU").is_some_and(|v| !v.is_empty() && v != "0");
    let mut out = Vec::new();
    match super::host::test_adapter() {
        Some(t) => eprintln!("hardware: kept out (XTP_TEST_ADAPTER={t})"),
        None => {
            let hardware = super::host::adapters().and_then(|a| match a.first() {
                Some(name) => Host::on(name),
                None => Err("no hardware adapter".to_string()),
            });
            match hardware {
                Ok(host) => out.push(("hardware", host)),
                Err(e) if strict => panic!("hardware: {e} (XTP_REQUIRE_GPU is set)"),
                Err(e) => eprintln!("hardware: {e}"),
            }
        }
    }
    match Host::new(true) {
        Ok(host) => out.push(("warp", host)),
        Err(e) => eprintln!("warp: {e}"),
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device; skipped");
    }
    out
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on each of
/// [`devices`] and returns, per one that ran, the UAV's 32 dwords.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    // One GPU test at a time (see `valu_tests::ONE_AT_A_TIME`).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, host) in devices() {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        // The table starts at heap index 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

/// `resource::tests::kept_per_lane`: the buffer both arms loaded from table
/// entry 0 and kept in VGPRs, read through a uniform handle in one pass of
/// the waterfall loop. Lanes below 24 (both arms) load and store dword
/// `lane` of the first buffer; the others hold numbers there, take no pass
/// and store nothing.
#[test]
fn lanes_read_the_buffer_both_arms_kept_in_vgprs() {
    use crate::resource::tests::{SecondArm, kept_per_lane};
    let shader = build(&kept_per_lane(SecondArm::SameEntry));
    let first: Vec<u32> = (0..32).map(|i| 100 + i).collect();
    let second: Vec<u32> = (0..32).map(|i| 200 + i).collect();
    let views = [
        View::Buffer {
            data: &first,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &second,
            stride: Stride::Structured(4),
        },
        View::RawUav(32),
    ];
    let results = run(&shader, &views);
    for (label, got) in results {
        let want: Vec<u32> = (0..32).map(|l| if l < 24 { 100 + l } else { 0 }).collect();
        assert_eq!(got, want, "{label}");
    }
}

/// What puts a number in s3 in the arm of [`mask_joined_with_a_number`]:
/// `s_add_i32 s3, s<sgpr>, 1`, from the root constant (s8 in a mesh
/// program, s0 in compute).
fn computed(sgpr: u32) -> u32 {
    0x8103_8100 | sgpr
}
const FIVE: u32 = 0xBE83_0385; // s_mov_b32 s3, 5
/// `v_mbcnt_lo_u32_b32 v0, -1, 0`: the lane below 32 (32 above).
const MBCNT_LO: [u32; 2] = [0xD765_0000, 0x0001_00C1];
/// `v_mbcnt_hi_u32_b32 v0, -1, v0`: with [`MBCNT_LO`], the lane.
const MBCNT_HI: [u32; 2] = [0xD766_0000, 0x0002_00C1];

/// The skinned and shadow mesh programs' join (235e56869c1ea883 at 0x1980),
/// as a wave64 program with a root constant: s[2:3] is a lane mask
/// (lanes below 5, the EXEC of an arm); the arm sets s3 to `number` and a
/// flag in v1. After the arm, s3 joins the mask's high dword (the edge
/// that skips the arm) with that number, and only the flagged lanes, which
/// ran the arm, read it, into v9. `end`: what reads v9 at the end.
fn mask_joined_with_a_number(number: u32, end: &[u32]) -> Vec<u32> {
    let mut words = vec![
        MBCNT_LO[0],
        MBCNT_LO[1],
        0x7E12_0280, // v_mov_b32 v9, 0
        0x7E02_0280, // v_mov_b32 v1, 0
        0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
        0xBE82_046A, // s_mov_b64 s[2:3], vcc
        0xBE94_2402, // s_and_saveexec_b64 s[20:21], s[2:3]
        0xBF88_0002, // s_cbranch_execz -> the join
        number,
        0x7E02_0281, // v_mov_b32 v1, 1
        0xBEFE_0414, // the join: s_mov_b64 exec, s[20:21]
        0x7D84_0281, // v_cmp_eq_u32 vcc, 1, v1
        0xBE96_246A, // s_and_saveexec_b64 s[22:23], vcc
        0xBF88_0001, // s_cbranch_execz -> the end
        0x7E12_0203, // v_mov_b32 v9, s3
        0xBEFE_0416, // the end: s_mov_b64 exec, s[22:23]
    ];
    words.extend(end);
    words.push(S_ENDPGM);
    words
}

/// The same join in the form 235e56869c1ea883 compiles it (0x1800 to
/// 0x1994): s[2:3] is all lanes or none (`s_cselect_b64` on the root
/// constant s8 not being 0), the arm runs where every lane of EXEC is in
/// it (`s_andn2_b64` + `s_cbranch_vccnz`), sets s3 to s8 + 1 and the flag
/// v1 to -1 where s3 is not 0 (0 before the arm). After the join only
/// lanes whose flag is -1 read s3, into v9.
const ANDN2_JOIN: [u32; 14] = [
    0x7E12_0280, // v_mov_b32 v9, 0
    0x7E02_0280, // v_mov_b32 v1, 0
    0xBF07_8008, // s_cmp_lg_u32 s8, 0
    0x8582_807E, // s_cselect_b64 s[2:3], exec, 0
    0x8AEA_027E, // s_andn2_b64 vcc, exec, s[2:3]
    0xBF87_0004, // s_cbranch_vccnz -> the join
    0x8103_8108, // s_add_i32 s3, s8, 1
    0xBF07_8003, // s_cmp_lg_u32 s3, 0
    0x8502_80C1, // s_cselect_b32 s2, -1, 0
    0x7E02_0202, // v_mov_b32 v1, s2
    0x7D84_02C1, // the join: v_cmp_eq_u32 vcc, -1, v1
    0x8AEA_6A7E, // s_andn2_b64 vcc, exec, vcc
    0xBF87_0001, // s_cbranch_vccnz -> the end
    0x7E12_0203, // v_mov_b32 v9, s3
];

/// The root constant of [`mask_joined_with_a_number`] (user data 0).
fn constant() -> RootParam {
    RootParam {
        root: RootType::Constants,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }
}

fn empty() -> crate::Container<'static> {
    crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    }
}

/// A wave64 mesh program of `threads` lanes with the root constant at s8.
fn mesh_program(code: &[u32], threads: u32) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        float_mode: 0xc0,
        vgprs: 24,
        layout: vec![constant()],
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            outputs: Vec::new(),
            clip_cntl: 0,
            threads,
            max_vertices: 128,
            max_primitives: 128,
            topology: 2,
            lds_bytes: 0,
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` translated for `target` in a container of `parts`, checked by
/// dxil.dll where it is; why not, if refused.
fn translate(
    parts: &crate::Container,
    native: &NativeProgram,
    target: Target,
) -> Result<HostShader, &'static str> {
    match super::translate_program_for(parts, native, target) {
        Ok(s) => {
            validate(&s);
            Ok(s)
        }
        Err((_, ShaderError::Unsupported { what, .. })) => Err(what),
        Err((_, e)) => panic!("{e}"),
    }
}

/// `words` as a wave64 mesh program of 64 lanes, translated for 32-lane
/// hosts, and checked by dxil.dll where it is; why not, if refused.
fn mesh(words: &[u32]) -> Result<(), &'static str> {
    let mut code = words.to_vec();
    // exp prim, v9, off, off, off done; then s_endpgm again.
    code.splice(code.len() - 1.., [0xF800_0941, 0x0000_0009, S_ENDPGM]);
    let target = Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    translate(&empty(), &mesh_program(&code, 64), target).map(|_| ())
}

/// `words` as a wave64 compute program of 64 lanes with 512 bytes of LDS,
/// lowered for any host; why not, if refused.
fn compute(words: &[u32]) -> Result<(), &'static str> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        float_mode: 0xc0,
        vgprs: 24,
        layout: vec![constant()],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 1 << 1,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    match super::tests::emit_for(&native, Target::ANY) {
        Ok(_) => Ok(()),
        Err(ShaderError::Unsupported { what, .. }) => Err(what),
        Err(e) => panic!("{e}"),
    }
}

fn element(name: &str, kind: u8, row: u32, usage: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage,
        half: false,
    }
}

/// A container with these signature parts.
fn signatures(parts: &[(crate::FourCC, Vec<u8>)]) -> Vec<u8> {
    let parts: Vec<(crate::FourCC, &[u8])> = parts.iter().map(|(f, d)| (*f, &d[..])).collect();
    crate::dxbc::build(&parts)
}

/// `words` as a wave32 pixel program with the root constant at s0, then
/// `exp mrt0, v9, v9, v9, v9 done vm`, translated for any host; why not, if
/// refused.
fn pixel(words: &[u32]) -> Result<(), &'static str> {
    let mut code = words.to_vec();
    code.extend([0xF800_180F, 0x0909_0909, S_ENDPGM]);
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        float_mode: 0xc0,
        vgprs: 24,
        layout: vec![constant()],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0b1111,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 1 << 1,
            rsrc2: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let bytes = signatures(&[
        (
            crate::FourCC::ISG1,
            super::container::signature(&[element("TEXCOORD", 0, 0, 0)], true),
        ),
        (
            crate::FourCC::OSG1,
            super::container::signature(&[element("SV_Target", 16, 0, 15)], false),
        ),
    ]);
    translate(
        &crate::Container::parse(&bytes).unwrap(),
        &native,
        Target::ANY,
    )
    .map(|_| ())
}

const MASK_AS_DATA: &str = "a lane mask read as data";

/// The mask's dword on the edge that skips the arm is no number the
/// program reads: in a mesh program, where no ballot gives a mask as data,
/// the join takes an undefined value there, since the lanes that take that
/// edge have the flag 0 and do not read s3 (`unread_on_edge`). Not in
/// compute (ballots are exact there), nor where the number is a constant
/// (the shape of `cond ? ballot : 5`): refused as before.
#[test]
fn a_mask_joined_with_a_computed_number_is_dead_outside_compute() {
    assert_eq!(mesh(&mask_joined_with_a_number(computed(8), &[])), Ok(()));
    let mut andn2 = MBCNT_LO.to_vec();
    andn2.extend(ANDN2_JOIN);
    andn2.push(S_ENDPGM);
    assert_eq!(mesh(&andn2), Ok(()));
    assert_eq!(
        mesh(&mask_joined_with_a_number(FIVE, &[])),
        Err(MASK_AS_DATA)
    );
    // ds_write_b32 v0, v9 (by dword: v0 * 4 first)
    let store = [0x3400_0082, 0xD834_0000, 0x0000_0900];
    for number in [computed(0), FIVE] {
        assert_eq!(
            compute(&mask_joined_with_a_number(number, &store)),
            Err(MASK_AS_DATA)
        );
    }
}

/// The counter-example of the review of meshdata: a ballot's high dword in
/// s3, s3 overwritten with the root constant + 1 under a condition of its
/// own, then s3 read after the join whatever the lane took. Where the arm
/// is skipped the ballot's dword reaches the read, so the join cannot take
/// an undefined value there: the program stops, as a direct read of the
/// mask does. The same with the condition per lane, and in a wave32 pixel
/// program (`MaskData::Host`), whose mask is s2.
#[test]
fn a_mask_read_after_its_join_with_a_number_stops() {
    let ballot = [
        MBCNT_LO[0],
        MBCNT_LO[1],
        0x7E12_0280, // v_mov_b32 v9, 0
        0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
        0xBE82_046A, // s_mov_b64 s[2:3], vcc
    ];
    let uniform = [
        0xBF06_8008, // s_cmp_eq_u32 s8, 0
        0xBF85_0001, // s_cbranch_scc1 -> the join
        computed(8),
        0x7E12_0203, // the join: v_mov_b32 v9, s3
        S_ENDPGM,
    ];
    let per_lane = [
        0x7D88_0083, // v_cmp_gt_u32 vcc, 3, v0
        0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
        0xBF88_0001, // s_cbranch_execz -> the join
        computed(8),
        0xBEFE_0414, // the join: s_mov_b64 exec, s[20:21]
        0x7E12_0203, // v_mov_b32 v9, s3
        S_ENDPGM,
    ];
    for arm in [&uniform[..], &per_lane[..]] {
        let words = [&ballot[..], arm].concat();
        assert_eq!(mesh(&words), Err(MASK_AS_DATA), "{words:x?}");
    }
    let wave32 = [
        MBCNT_LO[0],
        MBCNT_LO[1],
        0x7E12_0280, // v_mov_b32 v9, 0
        0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0
        0xBE82_036A, // s_mov_b32 s2, vcc_lo
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0001, // s_cbranch_scc1 -> the join
        0x8102_8100, // s_add_i32 s2, s0, 1
        0x7E12_0202, // the join: v_mov_b32 v9, s2
    ];
    assert_eq!(pixel(&wave32), Err(MASK_AS_DATA));
}

/// A pixel program that writes its attribute 0 as the console's flat
/// interpolation gives it (the provoking vertex's).
fn flat_pixel() -> NativeProgram {
    let words = [
        0xC802_0002, // v_interp_mov_f32 v0, p0, attr0.x
        0xC806_0102, // v_interp_mov_f32 v1, p0, attr0.y
        0xC80A_0202, // v_interp_mov_f32 v2, p0, attr0.z
        0xC80E_0302, // v_interp_mov_f32 v3, p0, attr0.w
        0xF800_180F,
        0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
        S_ENDPGM,
    ];
    NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        float_mode: 0,
        vgprs: 4,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 1,
                mask: 0b1111,
                attr: 0,
                flat: true,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 0,
            rsrc2: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Columns of [`drawn`]'s strip: texels (k, row) for k below this.
const COLUMNS: u32 = 31;

/// `body` (lane in v0, what it computes in v9) as a wave64 mesh program of
/// 64 vertices and primitives that exports v9: vertex v at column v / 2
/// (x from -1 to 1 in [`COLUMNS`] steps), y 1 when v is even, -1 when odd,
/// with (v9, v, 0, 0) as floats in its attribute 0 (TEXCOORD, row 1);
/// primitive v the triangle (v, v + 1, v + 2), none past vertex 61. Texel
/// (k, row) of a [`COLUMNS`] x 2 target is then vertex 2k + row's.
fn drawn(body: &[u32]) -> NativeProgram {
    let mut words = [MBCNT_LO, MBCNT_HI].concat();
    words.extend(body);
    words.extend([
        0x2C0A_0081, // v_lshrrev_b32 v5, 1, v0
        0x7E0A_0D05, // v_cvt_f32_u32 v5, v5
        0x100A_0AFF,
        (2.0f32 / COLUMNS as f32).to_bits(), // v_mul_f32 v5, lit(2 / 31), v5
        0x060A_0AF3,                         // v_add_f32 v5, -1.0, v5
        0x360C_0081,                         // v_and_b32 v6, 1, v0
        0x7E0C_0D06,                         // v_cvt_f32_u32 v6, v6
        0x100C_0CF5,                         // v_mul_f32 v6, -2.0, v6
        0x060C_0CF2,                         // v_add_f32 v6, 1.0, v6
        0x7E0E_02F0,                         // v_mov_b32 v7, 0.5
        0x7E10_02F2,                         // v_mov_b32 v8, 1.0
        0xF800_08CF,
        0x0807_0605, // exp pos0, v5, v6, v7, v8 done
        0x7E14_0D09, // v_cvt_f32_u32 v10, v9
        0x7E16_0D00, // v_cvt_f32_u32 v11, v0
        0x7E18_0280, // v_mov_b32 v12, 0
        0x7E1A_0280, // v_mov_b32 v13, 0
        0xF800_020F,
        0x0D0C_0B0A, // exp param0, v10, v11, v12, v13
        0x4A1C_0081, // v_add_nc_u32 v14, 1, v0
        0x4A1E_0082, // v_add_nc_u32 v15, 2, v0
        0xD76F_0010,
        0x0401_150E, // v_lshl_or_b32 v16, v14, 10, v0
        0xD76F_0010,
        0x0441_290F, // v_lshl_or_b32 v16, v15, 20, v16
        0x7D82_00FF,
        0x0000_003D, // v_cmp_lt_u32 vcc, lit(61), v0
        0x7E22_7081, // v_bfrev_b32 v17, 1 (the null primitive)
        0x0220_2310, // v_cndmask_b32 v16, v16, v17, vcc
        0xF800_0941,
        0x0000_0010, // exp prim, v16, off, off, off done
        S_ENDPGM,
    ]);
    let mut p = mesh_program(&words, 64);
    let StageIo::Ms(ms) = &mut p.io else {
        unreachable!()
    };
    ms.max_vertices = 64;
    ms.max_primitives = 64;
    ms.outputs = vec![VsOutput {
        usage: Usage::Generic,
        row: 1,
        param: 0,
        fp16: false,
        pack2: false,
    }];
    p
}

/// `native` (a [`drawn`] program) drawn with [`flat_pixel`] on `host`, for
/// its waves, with the root constant `s8`: the texels' attribute.
fn draw(host: &Host, label: &str, native: &NativeProgram, s8: u32) -> Vec<[f32; 4]> {
    let target = host.target();
    let ms_sig = signatures(&[(
        crate::FourCC::OSG1,
        super::container::signature(
            &[
                element("SV_Position", 3, 0, 15),
                element("TEXCOORD", 0, 1, 15),
            ],
            false,
        ),
    )]);
    let ps_sig = signatures(&[
        (
            crate::FourCC::ISG1,
            super::container::signature(
                &[
                    element("SV_Position", 3, 0, 0),
                    element("TEXCOORD", 0, 1, 15),
                ],
                true,
            ),
        ),
        (
            crate::FourCC::OSG1,
            super::container::signature(&[element("SV_Target", 16, 0, 15)], false),
        ),
    ]);
    let ms_c = crate::Container::parse(&ms_sig).unwrap();
    let ps_c = crate::Container::parse(&ps_sig).unwrap();
    let ms = translate(&ms_c, native, target)
        .unwrap_or_else(|w| panic!("{label}: mesh program refused for {target:?}: {w}"));
    let ps = translate(&ps_c, &flat_pixel(), target)
        .unwrap_or_else(|w| panic!("{label}: pixel program refused: {w}"));
    let (root, pso) = host
        .mesh_pipeline(&ms, &ps)
        .unwrap_or_else(|e| panic!("{label}: mesh_pipeline failed: {e}"));
    let mut ud = vec![0u32; ms.user_data_dwords().max(1) as usize];
    ud[0] = s8;
    host.draw(
        &root,
        &pso,
        &Draw {
            roots: vec![(0, Root::Constants(ud))],
            textures: &[],
            samplers: &[],
            vertices: &[],
            stride: 0,
            size: (COLUMNS, 2),
            mesh_groups: 1,
        },
    )
    .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"))
}

/// The two joins of a mask with a number drawn on each device: what each
/// vertex exports is what it read after the join, the arm's s8 + 1 where
/// its flag says it ran the arm and 0 elsewhere, the mask's edge included
/// (the undefined value there is never read). [`mask_joined_with_a_number`]
/// with s8 = 41: 42 in lanes below 5. [`ANDN2_JOIN`]: 42 in every lane with
/// s8 = 41, and 0 in every lane with s8 = 0, where the wave skips the arm.
#[test]
fn a_mask_joined_with_a_number_exports_the_arms_number() {
    let mut execz = mask_joined_with_a_number(computed(8), &[]);
    // Not the lane number below 32: the drawn body starts after it.
    execz.drain(..2);
    execz.pop();
    // The shape, its body, s8, and what vertex v exports.
    type Case<'a> = (&'a str, &'a [u32], u32, fn(u32) -> u32);
    let cases: [Case; 3] = [
        ("execz", &execz, 41, |v| if v < 5 { 42 } else { 0 }),
        ("andn2", &ANDN2_JOIN, 41, |_| 42),
        ("andn2", &ANDN2_JOIN, 0, |_| 0),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, host) in devices() {
        for &(shape, body, s8, want) in &cases {
            let texels = draw(&host, label, &drawn(body), s8);
            assert_eq!(texels.len(), 2 * COLUMNS as usize, "{label}");
            let mut wrong = Vec::new();
            for (n, t) in texels.iter().enumerate() {
                let (k, row) = (n as u32 % COLUMNS, n as u32 / COLUMNS);
                let v = 2 * k + row;
                let expected = [want(v) as f32, v as f32, 0.0, 0.0];
                if *t != expected {
                    wrong.push(format!("texel ({k}, {row}): {t:?}, not {expected:?}"));
                }
            }
            assert!(
                wrong.is_empty(),
                "{label}, {shape}, s8 = {s8}: {} of {} texels wrong:\n{}",
                wrong.len(),
                texels.len(),
                wrong[..wrong.len().min(16)].join("\n")
            );
        }
    }
}
