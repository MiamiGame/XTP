//! Descriptors read as data (`lower/smem.rs`: `descriptor_data` and
//! `unused_descriptor`): hand-assembled GFX10 compute programs (our own
//! code) whose descriptor loads no access uses as descriptors, and record
//! counts and scalar loads the runtime stride table routes. On the host
//! GPU (WARP in tests on the person's PC) the dwords rebuilt are checked
//! against the views' sizes (skipped without an SM 6.6 device); every
//! translation is checked by the validator where `dxil.dll` is found.
//!
//! - an image the program only measures (width, height, levels), picked
//!   from the heap with no entry: a 2D texture (fork #203's rule);
//! - a buffer the program only tests for its record count: its entry's
//!   view, or the stride table's count;
//! - a sampler's dword joined with data at a branch: undefined on the
//!   sampler's edge where no lane that takes it reads the register, the
//!   data on the other; read directly, or on its edge, it stops;
//! - with the stride table, the count and the scalar loads of a buffer
//!   read by element go through the table's raw view, not the slot's own
//!   (the Codex review of rawfix, P1); an MTBUF access's count is in
//!   elements (P2).

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::tests::{compute_of, emit_for};
use super::{HostShader, StrideEntry, Target};
use crate::native::{
    ComputeIo, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// Lanes in the group (one wave of 32).
const LANES: u32 = 32;

// Encodings (GFX10 ISA reference, "Microcode Formats").
const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_waitcnt lgkmcnt(0)`.
const WAIT_SCALAR: u32 = 0xBF8C_C07F;
/// `s_waitcnt vmcnt(0)`.
const WAIT_VECTOR: u32 = 0xBF8C_3F70;
/// Operand code of `null` (no register).
const NULL: u8 = 125;
/// Operand code of the inline constant 0.
const ZERO: u8 = 0x80;
const LOAD_DWORD: u32 = 0x0C;
const STORE_DWORD: u32 = 0x1C;
const STORE_DWORDX4: u32 = 0x1E;

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

/// SMEM `op` (2 `s_load_dwordx4`, 3 `_dwordx8`, 8 `s_buffer_load_dword`)
/// into s<data> through s[base:...] at byte `offset` plus `soffset` (an
/// SGPR, or [`NULL`]).
fn smem(op: u32, data: u8, base: u8, offset: u32, soffset: u8) -> [u32; 2] {
    [
        0xF400_0000 | op << 18 | u32::from(data) << 6 | u32::from(base / 2),
        u32::from(soffset) << 25 | offset,
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

/// MUBUF `op` of v[data..] at v<addr> (an offset with `offen`, an index
/// with `idxen`) through s[rsrc:rsrc+3].
fn mubuf(op: u32, data: u8, addr: u8, rsrc: u8, offen: bool, idxen: bool) -> [u32; 2] {
    [
        0xE000_0000 | op << 18 | u32::from(idxen) << 13 | u32::from(offen) << 12,
        u32::from(addr) | u32::from(data) << 8 | u32::from(rsrc / 4) << 16 | u32::from(ZERO) << 24,
    ]
}

/// `tbuffer_load_format_x v<data>, v<addr>, s[rsrc:rsrc+3], 0 offen` in
/// buffer format `format` (GFX10's numbering: 20 is 32_UINT).
fn tbuffer_load_x(data: u8, addr: u8, rsrc: u8, format: u32) -> [u32; 2] {
    [
        0xE800_0000 | format << 19 | 1 << 12,
        u32::from(addr) | u32::from(data) << 8 | u32::from(rsrc / 4) << 16 | u32::from(ZERO) << 24,
    ]
}

/// `image_sample_lz v[4:7], v[2:3], s[8:15], s[16:19] dmask:0xf` (2D).
const IMAGE_SAMPLE_LZ: [u32; 2] = [0xF09C_0F08, 0x0082_0402];

/// A table of descriptors at user data `user_data`.
fn table(user_data: u8) -> RootParam {
    RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: 0,
    }
}

/// The resource heap at user data `user_data`.
fn heap(user_data: u8) -> RootParam {
    RootParam {
        root: RootType::ResourceHeap,
        ..table(user_data)
    }
}

/// A used-resource entry of the table at user data `user_data`: `kind`
/// descriptors of `dwords` dwords from dword `at`, of `stride` bytes per
/// element for buffers (0: not known).
fn entry(kind: ResourceType, user_data: u8, at: u32, dwords: u8, stride: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride,
        dwords,
        user_data,
        slot: 0,
    }
}

/// A wave32 compute program of one wave with `user` user SGPRs, the root
/// parameters `layout` and the entries `used`.
fn native(
    words: &[u32],
    user: u32,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> NativeProgram {
    let mut p = compute_of(words, false, LANES);
    p.vgprs = 16;
    p.layout = layout;
    p.used = used;
    p.io = StageIo::Cs(ComputeIo {
        rsrc2: user << 1,
        threads: [LANES, 1, 1],
    });
    p
}

/// `native` translated for `target`, validated where `dxil.dll` is.
fn translate(native: &NativeProgram, target: Target) -> Result<HostShader, crate::ShaderError> {
    let mut cs = emit_for(native, target)?;
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil).expect("DXIL validation");
    }
    Ok(cs)
}

/// What stops `native` for `target`, if anything.
fn refusal(native: &NativeProgram, target: Target) -> Option<&'static str> {
    match emit_for(native, target) {
        Ok(_) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(e) => panic!("{e}"),
    }
}

/// Sets [`super::raw_structured`] for this thread while it lives.
struct RawStructured;

impl RawStructured {
    fn set(on: bool) -> Self {
        super::RAW_STRUCTURED_TEST.set(Some(on));
        RawStructured
    }
}

impl Drop for RawStructured {
    fn drop(&mut self) {
        super::RAW_STRUCTURED_TEST.set(None);
    }
}

/// Runs `cs` with all user data 0 (every table and heap at its heap's
/// start) over `views` and `samplers`, and the runtime stride table
/// `strides` where the translation reads one: each unordered view's
/// dwords afterwards, in heap order. None without an SM 6.6 device.
fn run(
    cs: &HostShader,
    views: &[View],
    samplers: &[D3D12_SAMPLER_DESC],
    strides: &[StrideEntry],
) -> Option<Vec<Vec<u32>>> {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 device ({e}); skipped");
            return None;
        }
    };
    let mut roots = vec![(
        0,
        Root::Constants(vec![0; cs.user_data_dwords().max(1) as usize]),
    )];
    assert_eq!(cs.stride_table, !strides.is_empty(), "the stride table");
    if cs.stride_table {
        let data: Vec<u32> = strides
            .iter()
            .flat_map(|e| {
                let bits = e.bits().expect("an entry the table holds");
                [bits as u32, (bits >> 32) as u32]
            })
            .collect();
        // The table's header first (`emit::STRIDE_HEADER`): every slot mirrored.
        let n = strides.len() as u32;
        let data = [vec![0, n, 0, n], data].concat();
        roots.push((1, Root::View { srv: true, data }));
    }
    let (root, pso) = host.compute_pipeline(cs).expect("a pipeline");
    Some(
        host.dispatch(
            &root,
            &pso,
            &Dispatch {
                roots,
                views,
                samplers,
                groups: [1, 1, 1],
            },
        )
        .expect("a dispatch"),
    )
}

/// The table's entry for slot `slot` read raw in place: `count` elements
/// of `stride` bytes (dwords for a raw view).
fn own(slot: u32, stride: u32, count: u32) -> StrideEntry {
    StrideEntry {
        stride,
        raw_slot: Some(slot),
        bias: 0,
        count,
    }
}

/// The puddle detection's shape (`6b6db9f1a526daeb`): an image picked from
/// the heap (s1) at slot 1, no entry describing it, whose dwords 1 to 3
/// (width, height, levels) the program reads and stores, lane by lane, at
/// 16 × the lane of the UAV at table (s0) entry 0; `depth` reads dword 4
/// too.
fn measured_image(depth: bool) -> NativeProgram {
    let mut w = vec![s_mov(5, ZERO), s_mov(4, 0), s_mov(3, ZERO), s_mov(2, 1)];
    w.extend(smem(2, 8, 4, 0, NULL));
    w.push(s_mov(12, inline(32)));
    w.extend(smem(3, 16, 2, 0, 12));
    w.push(WAIT_SCALAR);
    w.extend([
        v_mov(1, 17),
        v_mov(2, 18),
        v_mov(3, 19),
        v_mov(4, if depth { 20 } else { ZERO }),
        v_lshl(5, 4, 0),
    ]);
    w.extend(mubuf(STORE_DWORDX4, 1, 5, 8, true, false));
    w.push(S_ENDPGM);
    native(
        &w,
        2,
        vec![table(0), heap(1)],
        vec![entry(ResourceType::Uav, 0, 0, 4, 0)],
    )
}

/// The image's size bits come from the host view's dimensions in the GFX10
/// layout, as a 2D texture's: a 7 × 3 texture of two levels gives width - 1
/// (6) over dword 1 bits 30-31 and dword 2 bits 0-13, height - 1 (2) at
/// dword 2 bit 14, the last level (1) at dword 3 bit 16 beside the 2D type
/// (9 at bit 28) and the identity swizzle. Not where the program reads
/// depth or layers too: nothing says the image is 2D then.
#[test]
fn an_image_only_measured_from_the_heap_is_read_as_a_2d_textures_sizes() {
    assert_eq!(
        refusal(&measured_image(true), Target::ANY),
        Some("a descriptor read only as data")
    );
    let cs = translate(&measured_image(false), Target::ANY).expect("translated");
    let texels = vec![[0.5; 4]; 7 * 3 + 3];
    let views = [View::RawUav(4 * LANES), View::Levels(7, 3, 2, &texels)];
    let Some(out) = run(&cs, &views, &[], &[]) else {
        return;
    };
    let want: Vec<u32> = (0..LANES)
        .flat_map(|_| [0x8000_0000, 0x8001, 0x9001_0FAC, 0])
        .collect();
    assert_eq!(out[0], want);
}

/// The shape of `ceb5eb68cd99cb07`, which skips its pass when a buffer is
/// empty: the table (s0) entry at dword 0 is read only for its record
/// count (dword 2), which every lane stores at 4 × the lane of the UAV at
/// entry 8. `used` lists the table's entries.
fn counted_buffer(used: Vec<UsedResource>) -> NativeProgram {
    let mut w = vec![s_mov(5, ZERO), s_mov(4, 0)];
    w.extend(smem(2, 8, 4, 0x20, NULL));
    w.extend(smem(2, 12, 4, 0, NULL));
    w.extend([WAIT_SCALAR, v_mov(1, 14), v_lshl(5, 2, 0)]);
    w.extend(mubuf(STORE_DWORD, 1, 5, 8, true, false));
    w.push(S_ENDPGM);
    native(&w, 1, vec![table(0)], used)
}

/// The out UAV of [`counted_buffer`] and [`table_reads`]: entry 8.
fn out_entry() -> UsedResource {
    entry(ResourceType::Uav, 0, 8, 4, 0)
}

/// The count is the one the entry's view gives: elements of a structured
/// one (5 of 16 bytes), bytes of a raw one (80), elements again through a
/// raw view with `XTP_RAW_STRUCTURED=1`; with the stride table, the
/// table's (whatever the slot's own view is). With no entry nothing says
/// the slot holds a buffer view: it stops, table or not.
#[test]
fn a_buffer_only_counted_takes_its_entrys_view() {
    let data: Vec<u32> = (0..20).collect();
    let no_entry = counted_buffer(vec![out_entry()]);
    for target in [
        Target::ANY,
        Target {
            stride_table: true,
            ..Target::ANY
        },
    ] {
        assert_eq!(
            refusal(&no_entry, target),
            Some("a descriptor read only as data")
        );
    }
    // (stride, raw structured, view, table, count)
    let structured = View::Buffer {
        data: &data,
        stride: Stride::Structured(16),
    };
    let raw = View::Buffer {
        data: &data,
        stride: Stride::Raw,
    };
    let small = View::Buffer {
        data: &data[..8],
        stride: Stride::Raw,
    };
    let cases = [
        (16, false, structured, false, 5),
        (0, false, raw, false, 80),
        (16, true, raw, false, 5),
        (16, false, small, true, 5),
        (0, false, small, true, 80),
    ];
    for (stride, raw_structured, view, through_table, count) in cases {
        let _raw = RawStructured::set(raw_structured);
        let p = counted_buffer(vec![entry(ResourceType::Srv, 0, 0, 4, stride), out_entry()]);
        let target = Target {
            stride_table: through_table,
            ..Target::ANY
        };
        let cs = translate(&p, target).expect("translated");
        // The slot's view is 8 dwords; its table entry says 5 elements of
        // 16 bytes (or 20 dwords of a raw view).
        let strides = if through_table {
            vec![
                own(0, stride, if stride == 0 { 20 } else { 5 }),
                own(1, 0, LANES),
            ]
        } else {
            vec![]
        };
        let views = [view, View::RawUav(LANES)];
        let Some(out) = run(&cs, &views, &[], &strides) else {
            return;
        };
        assert_eq!(
            out[0],
            vec![count; LANES as usize],
            "stride {stride}, raw structured {raw_structured}, table {through_table}"
        );
    }
}

/// A buffer of a stride known only at run time (the entry's is 0) read by
/// element (the lane), by a scalar load at byte 8, and for its record
/// count: lane t stores (count, element t's first dword, the scalar
/// load's dword, 0) at 16 × t of the UAV at entry 8.
fn table_reads() -> NativeProgram {
    let mut w = vec![s_mov(5, ZERO), s_mov(4, 0)];
    w.extend(smem(2, 8, 4, 0x20, NULL));
    w.extend(smem(2, 12, 4, 0, NULL));
    w.push(WAIT_SCALAR);
    w.extend(smem(8, 16, 12, 8, NULL));
    w.push(v_mov(1, 14));
    w.extend(mubuf(LOAD_DWORD, 2, 0, 12, false, true));
    w.extend([
        WAIT_SCALAR,
        v_mov(3, 16),
        v_mov(4, ZERO),
        WAIT_VECTOR,
        v_lshl(5, 4, 0),
    ]);
    w.extend(mubuf(STORE_DWORDX4, 1, 5, 8, true, false));
    w.push(S_ENDPGM);
    native(
        &w,
        1,
        vec![table(0)],
        vec![entry(ResourceType::Srv, 0, 0, 4, 0), out_entry()],
    )
}

/// The Codex review of rawfix, P1: the table names, for the slot, a raw
/// view of the host copy from dword 4 with a bias of 4 bytes (16 elements
/// of 12 bytes from byte 20, as a view the slot could not hold), while the
/// slot itself holds 8 other dwords. The element loads, the record count
/// and the scalar load all read the table's view: 16 records (not the
/// slot's 32 bytes), element t at copy dword 5 + 3t (none past 16), byte 8
/// at copy dword 7 (not the slot's dword 2).
#[test]
fn counts_and_scalar_loads_through_the_stride_table_read_its_view() {
    let target = Target {
        stride_table: true,
        ..Target::ANY
    };
    let cs = translate(&table_reads(), target).expect("translated");
    let slot: Vec<u32> = (0..8).map(|k| 0xDEAD_0000 + k).collect();
    let copy: Vec<u32> = (0..64).map(|k| 1000 + k).collect();
    let views = [
        View::Buffer {
            data: &slot,
            stride: Stride::Raw,
        },
        View::RawUav(4 * LANES),
        View::Buffer {
            data: &copy,
            stride: Stride::Raw,
        },
        View::RawAlias {
            of: 2,
            first: 4,
            dwords: 52,
        },
    ];
    let strides = [
        StrideEntry {
            stride: 12,
            raw_slot: Some(3),
            bias: 4,
            count: 16,
        },
        own(1, 0, 4 * LANES),
        StrideEntry::NONE,
        StrideEntry::NONE,
    ];
    let Some(out) = run(&cs, &views, &[], &strides) else {
        return;
    };
    let want: Vec<u32> = (0..LANES)
        .flat_map(|t| [16, if t < 16 { 1005 + 3 * t } else { 0 }, 1007, 0])
        .collect();
    assert_eq!(out[0], want);
}

/// The Codex review of rawfix, P2: a buffer whose entry says 16 elements
/// of 12 bytes, read by `tbuffer_load_format_x` (32_UINT) at 4 × the lane
/// and for its record count: the count is 16 elements, as for an untyped
/// access, not 192 bytes. Through the slot's structured view the raw read
/// is the mismatch `XTP_RAW_STRUCTURED` removes, so the value is checked
/// with it only.
#[test]
fn an_mtbuf_accesss_record_count_is_in_elements() {
    let mut w = vec![s_mov(5, ZERO), s_mov(4, 0)];
    w.extend(smem(2, 8, 4, 0x20, NULL));
    w.extend(smem(2, 12, 4, 0, NULL));
    w.extend([v_lshl(5, 2, 0), WAIT_SCALAR]);
    w.extend(tbuffer_load_x(2, 5, 12, 20));
    w.extend([
        v_mov(1, 14),
        v_mov(3, ZERO),
        v_mov(4, ZERO),
        WAIT_VECTOR,
        v_lshl(6, 4, 0),
    ]);
    w.extend(mubuf(STORE_DWORDX4, 1, 6, 8, true, false));
    w.push(S_ENDPGM);
    let p = native(
        &w,
        1,
        vec![table(0)],
        vec![entry(ResourceType::Srv, 0, 0, 4, 12), out_entry()],
    );
    let data: Vec<u32> = (0..48).map(|k| 500 + k).collect();
    for raw_structured in [false, true] {
        let _raw = RawStructured::set(raw_structured);
        let cs = translate(&p, Target::ANY).expect("translated");
        let view = View::Buffer {
            data: &data,
            stride: if raw_structured {
                Stride::Raw
            } else {
                Stride::Structured(12)
            },
        };
        let Some(out) = run(&cs, &[view, View::RawUav(4 * LANES)], &[], &[]) else {
            return;
        };
        for t in 0..LANES as usize {
            let got = &out[0][4 * t..4 * t + 2];
            assert_eq!(got[0], 16, "lane {t}, raw structured {raw_structured}");
            if raw_structured {
                assert_eq!(got[1], data[t], "lane {t}");
            }
        }
    }
}

/// Where [`sampler_joined`] reads s17.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    /// With no branch before: the sampler's dword read directly.
    Direct,
    /// After the join, in every lane: on the sampler's edge, its dword.
    Joined,
    /// After the join, behind the flag tested again: only where the scalar
    /// load overwrote the sampler's dword.
    Retested,
}

/// The material programs' shape (`f23ac5187c49056a`): a sampler (s1's
/// table) samples an image (s0's table, entry 0); then, unless a flag the
/// program reads (dword 1 of the raw buffer at entry 16) is 0, a scalar
/// load of the buffer's dword 0 overwrites the sampler's dword 1, and each
/// lane stores what the register holds after the join at 4 × the lane of
/// the UAV at entry 8, as `read` says.
fn sampler_joined(read: Read) -> NativeProgram {
    let mut w = vec![s_mov(5, ZERO), s_mov(4, 0), s_mov(7, ZERO), s_mov(6, 1)];
    w.extend(smem(3, 8, 4, 0, NULL));
    w.extend(smem(2, 16, 6, 0, NULL));
    w.extend(smem(2, 20, 4, 0x20, NULL));
    w.extend(smem(2, 24, 4, 0x40, NULL));
    w.extend([v_mov(2, ZERO), v_mov(3, ZERO), WAIT_SCALAR]);
    w.extend(IMAGE_SAMPLE_LZ);
    if read != Read::Direct {
        w.extend(smem(8, 28, 24, 4, NULL));
        w.extend([WAIT_SCALAR, s_cmp_eq_u32(28, ZERO), s_cbranch_scc1(3)]);
        w.extend(smem(8, 17, 24, 0, NULL));
        w.push(WAIT_SCALAR);
    }
    if read == Read::Retested {
        // Over the store's four dwords.
        w.extend([s_cmp_eq_u32(28, ZERO), s_cbranch_scc1(4)]);
    }
    w.extend([v_mov(1, 17), v_lshl(8, 2, 0)]);
    w.extend(mubuf(STORE_DWORD, 1, 8, 20, true, false));
    w.push(S_ENDPGM);
    native(
        &w,
        2,
        vec![table(0), table(1)],
        vec![
            entry(ResourceType::Srv, 0, 0, 8, 0),
            out_entry(),
            entry(ResourceType::Srv, 0, 16, 4, 0),
            entry(ResourceType::Sampler, 1, 0, 4, 0),
        ],
    )
}

/// No program reads a sampler's dwords as numbers: where one meets data at
/// a join and the lanes that take the sampler's edge never read the
/// register (the flag tested again, `lower/flow.rs`: `unread_on_edge`),
/// it is dead on that edge (undefined there), and holds the data on the
/// other: 77 with the flag set, nothing stored with it clear. Read
/// directly, or after the join on the sampler's edge too (the review of
/// w7skin), the sampler's dword has no value: it stops, as a descriptor's
/// address does.
#[test]
fn a_samplers_dword_joined_with_data_is_dead_on_its_edge() {
    for read in [Read::Direct, Read::Joined] {
        assert_eq!(
            refusal(&sampler_joined(read), Target::ANY),
            Some("a descriptor's address or format read as data")
        );
    }
    let cs = translate(&sampler_joined(Read::Retested), Target::ANY).expect("translated");
    let texel = [[0.25; 4]];
    let point = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    for (flag, want) in [(1, 77), (0, 0)] {
        let data = [77, flag, 0, 0];
        let views = [
            View::Texture(1, 1, &texel),
            View::RawUav(LANES),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
        ];
        let Some(out) = run(&cs, &views, &[point], &[]) else {
            return;
        };
        assert_eq!(out[0], vec![want; LANES as usize], "flag {flag}");
    }
}

/// Two loads of a heap slot (s1) at the same offset (s12, slot 1): the
/// first used by a raw load, the second read for its record count, which
/// takes the first's view. A load of another slot (s13, slot 2) is no
/// twin and nothing says what it holds: it stops.
#[test]
fn a_heap_descriptor_read_as_data_needs_a_twin_load() {
    let program = |soffset: u8| {
        let mut w = vec![
            s_mov(5, ZERO),
            s_mov(4, 0),
            s_mov(3, ZERO),
            s_mov(2, 1),
            s_mov(12, inline(32)),
            s_mov(13, inline(64)),
        ];
        w.extend(smem(2, 8, 4, 0, NULL));
        w.extend(smem(2, 16, 2, 0, 12));
        w.extend(smem(2, 20, 2, 0, soffset));
        w.extend([WAIT_SCALAR, v_lshl(5, 2, 0)]);
        w.extend(mubuf(LOAD_DWORD, 2, 5, 16, true, false));
        w.extend([v_mov(1, 22), v_mov(3, ZERO), v_mov(4, ZERO), WAIT_VECTOR]);
        w.push(v_lshl(6, 4, 0));
        w.extend(mubuf(STORE_DWORDX4, 1, 6, 8, true, false));
        w.push(S_ENDPGM);
        native(
            &w,
            2,
            vec![table(0), heap(1)],
            vec![entry(ResourceType::Uav, 0, 0, 4, 0)],
        )
    };
    assert_eq!(refusal(&program(12), Target::ANY), None);
    assert_eq!(
        refusal(&program(13), Target::ANY),
        Some("a descriptor read only as data")
    );
}
