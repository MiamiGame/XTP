//! Buffers read by element without a stride known at translation, through
//! the host's runtime stride table ([`super::StrideEntry`], `t0` in
//! [`super::STRIDE_SPACE`]): hand-assembled compute programs run on WARP
//! (and the default adapter outside `XTP_TEST_ADAPTER=warp`) with the
//! views and table entries the GPU side makes. Ported from
//! Marvisak/VIctory#704 (r4tedx), with the range checks and the entry of
//! zeros that holds no view added here.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, RawWindow, StrideEntry, Target};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// One thread, user SGPRs `s0..s4`: four root constants, then the table.
const ONE_THREAD: ComputeIo = ComputeIo {
    rsrc2: 5 << 1,
    threads: [1, 1, 1],
};

/// The target with the stride table, else as any host.
const TABLE: Target = Target {
    stride_table: true,
    ..Target::ANY
};

/// A compute program over a table at `s4` (user data `s0..s3` before it):
/// a shader view at table dword 0, an unordered access view at table dword
/// `uav`, neither with a stride known here; lowered for `target`.
fn lowered(
    words: &[u32],
    uav: u32,
    io: ComputeIo,
    target: Target,
) -> Result<super::lower::Lowered, crate::ShaderError> {
    lowered_with(words, uav, io, target, 0)
}

/// [`lowered`] with both views of `stride` bytes known at translation.
fn lowered_with(
    words: &[u32],
    uav: u32,
    io: ComputeIo,
    target: Target,
    stride: u32,
) -> Result<super::lower::Lowered, crate::ShaderError> {
    let used = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 4, 0, 0),
            param(RootType::Table, 1, 4, 1),
        ],
        used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, uav)],
        io: StageIo::Cs(io),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target,
        console_cf: false,
        per_lane_only: false,
    })
}

/// [`lowered`] for the target with the table, validated where `dxil.dll`
/// is, as a host shader.
fn shader(words: &[u32], uav: u32, io: ComputeIo) -> HostShader {
    let mut l = lowered(words, uav, io, TABLE).expect("lowered with the table");
    assert!(l.stride_table, "the program reads the table");
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut l.dxil).expect("DXIL validation");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        group_base: l.group_base,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        stride_table: l.stride_table,
        run_time_buffers: l.run_time_buffers,
    }
}

/// The devices to run on, one GPU test at a time (see
/// `valu_tests::ONE_AT_A_TIME`): WARP, and the default adapter unless
/// `XTP_TEST_ADAPTER` keeps tests off it.
fn hosts() -> (
    std::sync::MutexGuard<'static, ()>,
    Vec<(&'static str, Host)>,
) {
    let one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut hosts = Vec::new();
    for (name, warp) in [("WARP", true), ("the default adapter", false)] {
        match Host::new(warp) {
            Ok(h) => hosts.push((name, h)),
            Err(e) => eprintln!("{name}: no SM 6.6 device ({e}); skipped"),
        }
    }
    // In tests WARP may be the only device: once is enough.
    hosts.truncate(if super::host::test_adapter().is_some() {
        1
    } else {
        2
    });
    (one, hosts)
}

/// What the GPU side makes for a buffer view of `count` elements of
/// `stride` bytes from element `first` of the buffer in heap slot `of`:
/// the raw view of those bytes (for heap slot `raw_slot`) and the slot's
/// table entry.
fn raw_view_of(
    of: usize,
    raw_slot: u32,
    (first, count, stride): (u32, u32, u32),
) -> (View<'static>, StrideEntry) {
    let window = RawWindow::covering(u64::from(first * stride), u64::from(count * stride)).unwrap();
    (
        View::RawAlias {
            of,
            first: window.first as u32,
            dwords: window.dwords,
        },
        StrideEntry {
            stride,
            raw_slot: Some(raw_slot),
            bias: window.bias,
            count: count.min(window.elements(stride)),
        },
    )
}

/// The table's dwords for `entries`, slot 0 on, after its header
/// ([`super::STRIDE_HEADER`]: every slot mirrored).
fn table(entries: &[StrideEntry]) -> Vec<u32> {
    let n = entries.len() as u32;
    let mut words = vec![0, n, 0, n];
    words.extend(entries.iter().flat_map(|e| {
        let bits = e.bits().expect("an entry the table holds");
        [bits as u32, (bits >> 32) as u32]
    }));
    words
}

/// Runs `cs` in `groups` groups with user data `user` (`s0..s3`; the table
/// at heap slot 0 in `s4`), `views` in the heap and the stride table of
/// `entries`: what each unordered view read back holds.
fn run(
    host: &Host,
    cs: &HostShader,
    user: [u32; 4],
    views: &[View],
    entries: &[StrideEntry],
    groups: u32,
) -> Vec<Vec<u32>> {
    let (root, pso) = host.compute_pipeline(cs).expect("a pipeline");
    let mut constants = user.to_vec();
    constants.push(0);
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![
                (0, Root::Constants(constants)),
                (
                    1,
                    Root::View {
                        srv: true,
                        data: table(entries),
                    },
                ),
            ],
            views,
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )
    .expect("a dispatch")
}

/// Loads `[2, 4]` of the view in table slot 0, stores it at `[1, 8]` of the
/// unordered view in slot 1 and adds 10 twice at `[2, 0]` there.
const LOAD_STORE_ATOMICS: &[u32] = &[
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xF408_0302,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    0x7E00_0282, // v_mov_b32 v0, 2
    0x7E02_0284, // v_mov_b32 v1, 4
    0xE030_3000,
    0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
    0x7E00_0281, // v_mov_b32 v0, 1
    0x7E02_0288, // v_mov_b32 v1, 8
    0xE070_3000,
    0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
    0x7E00_0282, // v_mov_b32 v0, 2
    0x7E02_0280, // v_mov_b32 v1, 0
    0x7E06_028A, // v_mov_b32 v3, 10
    0xE0C8_3000,
    0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
    0xE0C8_3000,
    0x8003_0300, // buffer_atomic_add v3, v[0:1], s[12:15], 0 idxen offen
    0xBF81_0000, // s_endpgm
];

/// Loads a dword and two dwords at `[s0, s1]` of the view in table slot 0
/// (strides unknown) and stores them at dwords 0, 1 and 2 of the raw view
/// in slot 1 (no element, so no stride needed).
const LOAD_AT: &[u32] = &[
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xF408_0302,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    0x7E00_0200, // v_mov_b32 v0, s0
    0x7E02_0201, // v_mov_b32 v1, s1
    0xE030_3000,
    0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
    0xE034_3000,
    0x8002_0300, // buffer_load_dwordx2 v[3:4], v[0:1], s[8:11], 0 idxen offen
    0x7E0A_0280, // v_mov_b32 v5, 0
    0xE070_1000,
    0x8003_0205, // buffer_store_dword v2, v5, s[12:15], 0 offen
    0xE074_1004,
    0x8003_0305, // buffer_store_dwordx2 v[3:4], v5, s[12:15], 0 offen offset:4
    0xBF81_0000, // s_endpgm
];

/// A program that reads by element a buffer whose known stride the table
/// cannot carry is refused with the stride it saw (watch.md item 318):
/// 1 to 3 bytes (R8 and R16 structured buffers), or off a multiple of 4;
/// they were counted as unknown strides, which the table translates.
#[test]
fn small_and_odd_strides_are_refused_apart_from_unknown_ones() {
    let why = |stride, target| match lowered_with(LOAD_STORE_ATOMICS, 8, ONE_THREAD, target, stride)
    {
        Err(crate::ShaderError::Unsupported { what, .. }) => what.to_string(),
        other => format!("{:?}", other.map(|l| l.stride_table)),
    };
    for target in [Target::ANY, TABLE] {
        for stride in [1, 2, 3] {
            assert_eq!(why(stride, target), "a structured buffer of stride 1 to 3");
        }
        for stride in [6, 10, 18] {
            assert_eq!(
                why(stride, target),
                "a structured buffer of stride off a multiple of 4"
            );
        }
    }
    assert_eq!(why(0, Target::ANY), "a structured buffer of unknown stride");
    assert_eq!(why(0, TABLE), "Ok(true)", "the table translates it");
}

/// Without the table (the host binds none) a program that reads a buffer
/// by element of unknown stride is refused as before; with it, it reads
/// the table.
#[test]
fn unknown_strides_translate_only_for_a_target_with_the_table() {
    let refused = lowered(LOAD_STORE_ATOMICS, 8, ONE_THREAD, Target::ANY).err();
    assert!(
        matches!(
            refused,
            Some(crate::ShaderError::Unsupported { what, .. })
                if what == "a structured buffer of unknown stride"
        ),
        "{refused:?}"
    );
    let l = lowered(LOAD_STORE_ATOMICS, 8, ONE_THREAD, TABLE).unwrap();
    assert!(l.stride_table);
    // An access by offset alone of a view-heap slot reads the table too
    // on a target with it (the raw view it names), and none without.
    let raw_only = &[
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x7E0A_0280, // v_mov_b32 v5, 0
        0xE070_1000,
        0x8003_0505, // buffer_store_dword v5, v5, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    assert!(
        lowered(raw_only, 8, ONE_THREAD, TABLE)
            .unwrap()
            .stride_table
    );
    assert!(
        !lowered(raw_only, 8, ONE_THREAD, Target::ANY)
            .unwrap()
            .stride_table
    );
}

/// Loads the dword at byte `s1` of the view in table slot 0 by offset
/// alone and stores it at dword 0 of the raw view in slot 1, both through
/// the table.
const LOAD_BY_OFFSET: &[u32] = &[
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xF408_0302,
    0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
    0x7E02_0201, // v_mov_b32 v1, s1
    0xE030_1000,
    0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
    0x7E0A_0280, // v_mov_b32 v5, 0
    0xE070_1000,
    0x8003_0205, // buffer_store_dword v2, v5, s[12:15], 0 offen
    0xBF81_0000, // s_endpgm
];

/// A structured view read by offset alone, as raw (byte addressing over
/// the view): through the raw view of the same bytes the table names, so
/// the handle and the view are of one kind (NVIDIA reads other dwords or
/// zeros raw through a structured view: watch.md items 148, 160); zeros
/// past the view's bytes and through an entry of zeros, where a texture,
/// constant or null slot would otherwise be read through a raw handle.
#[test]
fn accesses_by_offset_go_through_the_raw_view_too() {
    let cs = shader(LOAD_BY_OFFSET, 8, ONE_THREAD);
    let data: Vec<u32> = (100..164).collect();
    // Four elements of 16 bytes from element 1 (byte 16): 64 bytes.
    let (raw, entry) = raw_view_of(0, 2, (1, 4, 16));
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Structured(16),
        },
        View::RawUav(4),
        raw,
    ];
    let out_entry = StrideEntry {
        stride: 0,
        raw_slot: Some(1),
        bias: 0,
        count: 4,
    };
    let (_one, hosts) = hosts();
    for (name, host) in &hosts {
        for (off, want) in [(0, 104), (4, 105), (60, 119), (64, 0), (u32::MAX - 1, 0)] {
            let out = run(host, &cs, [0, off, 0, 0], &views, &[entry, out_entry], 1);
            assert_eq!(out[0][0], want, "byte {off} on {name}");
        }
        let none = run(
            host,
            &cs,
            [0, 4, 0, 0],
            &views,
            &[StrideEntry::NONE, out_entry],
            1,
        );
        assert_eq!(none[0][0], 0, "an entry of zeros on {name}");
    }
}

/// Every kind of view the title can put in the slot reads and writes the
/// bytes the console does: structured of 4, 12, 16 and 20 bytes (also
/// from an element off 16 bytes, so the raw view starts 4 or 12 bytes
/// early), typed (16 bytes an element) and raw (the offset alone: a
/// descriptor of no stride). Through the raw view the entry names, never
/// the slot's own view through a handle of another kind (on which an RTX
/// 5070 reads other dwords than WARP and a Radeon, VIctory#704).
#[test]
fn loads_stores_and_atomics_through_the_table_reach_the_views_bytes() {
    let cs = shader(LOAD_STORE_ATOMICS, 8, ONE_THREAD);
    let data: Vec<u32> = (100..164).collect();
    // The view in the slot, its stride and (first, count) elements; none:
    // a raw view, read in its own slot.
    let kinds = [
        ("structured 4", Stride::Structured(4), 4, Some((0, 64))),
        ("structured 12", Stride::Structured(12), 12, Some((0, 21))),
        ("structured 16", Stride::Structured(16), 16, Some((0, 16))),
        ("structured 20", Stride::Structured(20), 20, Some((0, 12))),
        ("typed RGBA32F", Stride::Typed, 16, Some((0, 16))),
        (
            "structured 12 from element 3",
            Stride::Structured(12),
            12,
            Some((3, 10)),
        ),
        (
            "structured 20 from element 3",
            Stride::Structured(20),
            20,
            Some((3, 6)),
        ),
        ("raw", Stride::Raw, 0, None),
    ];
    let (_one, hosts) = hosts();
    for (name, host) in &hosts {
        for (kind, view, stride, elements) in kinds {
            // Heap: the shader view, the unordered one, their raw views.
            let mut views = vec![
                View::Buffer {
                    data: &data,
                    stride: view,
                },
                View::BufferUav(64, view),
            ];
            let mut entries = Vec::new();
            for slot in 0..2 {
                entries.push(match elements {
                    Some(e) => {
                        let (raw, entry) = raw_view_of(slot, 2 + slot as u32, (e.0, e.1, stride));
                        views.push(raw);
                        entry
                    }
                    None => StrideEntry {
                        stride: 0,
                        raw_slot: Some(slot as u32),
                        bias: 0,
                        count: 64,
                    },
                });
            }
            let out = run(host, &cs, [0; 4], &views, &entries, 1);
            let first = elements.map_or(0, |e| e.0);
            let at = |index: u32, off: u32| (((first + index) * stride + off) / 4) as usize;
            // Inside the element (a raw view has none: its offset alone).
            let inside = |off: u32| stride == 0 || off + 4 <= stride;
            // The load of [2, 4], stored at [1, 8]; 10 added twice at [2, 0].
            // With 4-byte elements both offsets are past the element: the
            // load reads 0 and the store is dropped.
            let mut want = vec![0; 64];
            if inside(8) {
                want[at(1, 8)] = if inside(4) { data[at(2, 4)] } else { 0 };
            }
            want[at(2, 0)] = 20;
            assert_eq!(out[0], want, "{kind} on {name}");
        }
    }
}

/// The console's range checks: an access is made at an element below the
/// view's count and inside the element, else a load reads 0 (the next
/// element, the bytes past the view and those before it in its raw view
/// are not read); an index whose product with the stride wraps is past the
/// count.
#[test]
fn reads_past_the_view_or_its_element_read_zero() {
    let cs = shader(LOAD_AT, 8, ONE_THREAD);
    let data: Vec<u32> = (100..164).collect();
    // Elements 3 to 6 of 12 bytes (bytes 36 to 84): the raw view starts at
    // byte 32, 4 bytes early, so the dword before the view is in it.
    let (raw, entry) = raw_view_of(0, 2, (3, 4, 12));
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Structured(12),
        },
        View::RawUav(4),
        raw,
    ];
    // The raw view the results are stored in, by offset: its own slot.
    let results = StrideEntry {
        stride: 0,
        raw_slot: Some(1),
        bias: 0,
        count: 4,
    };
    let (_one, hosts) = hosts();
    for (name, host) in &hosts {
        // [index, offset] → the dword, then the two dwords.
        let dword = |i: u32, o: u32| data[((3 + i) * 12 + o) as usize / 4];
        for (index, off, want) in [
            (0, 0, [dword(0, 0), dword(0, 0), dword(0, 4)]),
            (3, 4, [dword(3, 4), dword(3, 4), dword(3, 8)]),
            // The last dword of the last element: only the first component fits.
            (3, 8, [dword(3, 8), dword(3, 8), 0]),
            // Past the count: the buffer's next bytes are not read.
            (4, 0, [0, 0, 0]),
            // Past the element: not the next element's first dword.
            (1, 12, [0, 0, 0]),
            // Wrapping: (2^32 / 4) * 12 + 4 is byte 4 mod 2^32, which the
            // raw view holds before the view (its bias).
            (1 << 30, 4, [0, 0, 0]),
            (u32::MAX, 0, [0, 0, 0]),
        ] {
            let out = run(host, &cs, [index, off, 0, 0], &views, &[entry, results], 1);
            assert_eq!(out[0][..3], want, "[{index}, {off}] on {name}");
        }
    }
}

/// A slot whose entry is zeros (a null, texture or constant view, or a
/// slot never made) is never accessed: loads read 0, stores and atomics
/// are dropped. Here slot 0 holds a buffer whose bytes would be read
/// through a raw handle of its slot if zeros meant the slot itself.
#[test]
fn an_entry_of_zeros_holds_no_view() {
    let cs = shader(LOAD_STORE_ATOMICS, 8, ONE_THREAD);
    let data: Vec<u32> = (100..164).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::BufferUav(64, Stride::Raw),
    ];
    let (_one, hosts) = hosts();
    for (name, host) in &hosts {
        let out = run(host, &cs, [0; 4], &views, &[StrideEntry::NONE; 2], 1);
        assert_eq!(out[0], vec![0; 64], "on {name}");
    }
}

/// The heap slot of an unknown-stride read loaded from a buffer when the
/// program runs, per thread group: each group's lanes read the bytes of
/// the view in the slot their group chose (structured of 12 bytes in one,
/// of 16 in the other), since the entry is read at the same computed slot
/// the handle is made of; a lane past the chosen view's elements reads 0.
/// From VIctory#704's `unknown_stride_views_chosen_by_a_loaded_heap_index`.
#[test]
fn views_chosen_by_a_loaded_heap_index() {
    let words: &[u32] = &[
        0xBE94_0304, // s_mov_b32 s20, s4
        0xBE95_0380, // s_mov_b32 s21, 0
        0xF408_040A,
        0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
        0x8F07_8205, // s_lshl_b32 s7, s5, 2
        0xF420_0188,
        0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
        0x8F06_8506, // s_lshl_b32 s6, s6, 5
        0xF408_020A,
        0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
        0xF408_030A,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
        0x7E02_0284, // v_mov_b32 v1, 4
        0xE030_3000,
        0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
        0x8F07_8405, // s_lshl_b32 s7, s5, 4
        0x7E02_0207, // v_mov_b32 v1, s7
        0xE070_3000,
        0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
        0xBF81_0000, // s_endpgm
    ];
    // Four threads (v0: the thread's id), the group's id in s5 after the
    // user SGPRs s0..s4.
    let io = ComputeIo {
        rsrc2: 5 << 1 | 1 << 7,
        threads: [4, 1, 1],
    };
    let cs = shader(words, 8, io);
    // The load's slot is picked when the program runs (the store's is at a
    // constant place of the table): the host must make its view itself.
    assert_eq!(cs.run_time_buffers, super::RUN_TIME_READS);
    let named = lowered(LOAD_STORE_ATOMICS, 8, ONE_THREAD, TABLE).unwrap();
    assert_eq!(named.run_time_buffers, 0, "slots the table names");
    let without = lowered(words, 8, io, Target::ANY).err();
    assert!(without.is_some(), "an unknown stride without the table");
    let (a, b): (Vec<u32>, Vec<u32>) = ((100..164).collect(), (200..264).collect());
    // Slot 2: three elements of 12 bytes; slot 3: four of 16 bytes from
    // element 1; their raw views in slots 4 and 5. Slot 1, written: four
    // elements of 32 bytes, its raw view in slot 6. Slot 0, read by a
    // scalar load, is read through the table too: its two dwords raw in
    // place, as the host's table names a raw view.
    let (raw_a, entry_a) = raw_view_of(2, 4, (0, 3, 12));
    let (raw_b, entry_b) = raw_view_of(3, 5, (1, 4, 16));
    let (raw_out, entry_out) = raw_view_of(1, 6, (0, 4, 32));
    let index = StrideEntry {
        stride: 0,
        raw_slot: Some(0),
        bias: 0,
        count: 2,
    };
    let entries = [index, entry_out, entry_a, entry_b];
    // Lane `l` reads [l, 4]: of the first view dwords 1, 4, 7 and (past its
    // three elements) nothing, of the second 5, 9, 13, 17; group g stores
    // it at [l, 16 g], dword 8 l + 4 g.
    let of_a = [101, 104, 107, 0];
    let of_b = [205, 209, 213, 217];
    let (_one, hosts) = hosts();
    for (name, host) in &hosts {
        for (slots, read) in [([2u32, 3], [of_a, of_b]), ([3, 2], [of_b, of_a])] {
            let mut want = vec![0; 32];
            for (g, values) in read.iter().enumerate() {
                for (l, &v) in values.iter().enumerate() {
                    want[8 * l + 4 * g] = v;
                }
            }
            let views = [
                View::Buffer {
                    data: &slots,
                    stride: Stride::Raw,
                },
                View::BufferUav(32, Stride::Structured(32)),
                View::Buffer {
                    data: &a,
                    stride: Stride::Structured(12),
                },
                View::Buffer {
                    data: &b,
                    stride: Stride::Structured(16),
                },
                raw_a,
                raw_b,
                raw_out,
            ];
            let out = run(host, &cs, [0; 4], &views, &entries, 2);
            assert_eq!(out[0], want, "slots {slots:?} on {name}");
        }
    }
}

/// The entry's bits: what [`StrideEntry::bits`] holds comes back, and what
/// it cannot hold is refused.
#[test]
fn entries_hold_their_fields_in_one_word() {
    assert_eq!(StrideEntry::NONE.bits(), Some(0));
    assert_eq!(StrideEntry::from_bits(0), StrideEntry::NONE);
    for e in [
        StrideEntry {
            stride: 12,
            raw_slot: Some(1_245_183 + (1 << 18)),
            bias: 12,
            count: 21,
        },
        StrideEntry {
            stride: 0,
            raw_slot: Some(0),
            bias: 0,
            count: StrideEntry::MAX_COUNT,
        },
        StrideEntry {
            stride: StrideEntry::MAX_STRIDE,
            raw_slot: Some(StrideEntry::RAW_SLOTS - 1),
            bias: 4,
            count: 1,
        },
    ] {
        assert_eq!(StrideEntry::from_bits(e.bits().unwrap()), e);
    }
    let base = StrideEntry {
        stride: 16,
        raw_slot: Some(3),
        bias: 0,
        count: 4,
    };
    // No element, or no raw view: no view at all.
    assert_eq!(StrideEntry { count: 0, ..base }.bits(), Some(0));
    assert_eq!(
        StrideEntry {
            raw_slot: None,
            ..base
        }
        .bits(),
        Some(0)
    );
    for bad in [
        StrideEntry { stride: 6, ..base },
        StrideEntry {
            stride: 4096,
            ..base
        },
        StrideEntry {
            raw_slot: Some(StrideEntry::RAW_SLOTS),
            ..base
        },
        StrideEntry { bias: 16, ..base },
        StrideEntry { bias: 2, ..base },
    ] {
        assert_eq!(bad.bits(), None, "{bad:?}");
    }
}

/// Raw windows start on 16 bytes, up to 12 before the view, end on the
/// dword of its last byte, and hold none of its bytes past what a host
/// view takes.
#[test]
fn raw_windows_cover_the_views_bytes() {
    let window = |first, dwords, bias| {
        Some(RawWindow {
            first,
            dwords,
            bias,
        })
    };
    assert_eq!(RawWindow::covering(32, 24), window(8, 6, 0));
    assert_eq!(RawWindow::covering(36, 24), window(8, 7, 4));
    assert_eq!(RawWindow::covering(44, 22), window(8, 9, 12));
    // No bytes, or a start off a dword: no raw view reads it.
    assert_eq!(RawWindow::covering(0, 0), None);
    assert_eq!(RawWindow::covering(34, 24), None);
    // Past the host's 2^27 elements a view is cut there.
    let big = RawWindow::covering(4, 1 << 32).unwrap();
    assert_eq!((big.dwords, big.bias), (RawWindow::MAX_DWORDS, 4));
    assert_eq!(big.elements(16), ((1u64 << 29) - 4) as u32 / 16);
    // Elements that lie wholly in the window: 4 of 12 bytes from byte 4.
    assert_eq!(RawWindow::covering(36, 48).unwrap().elements(12), 4);
}

/// With `XTP_RAW_STRUCTURED=1` the host makes structured views raw views of
/// the same bytes (VIctory#1762), so a program that reads by element with a
/// stride known at translation reads raw too: at `index * stride + off`
/// through a raw handle, or through the table's raw view where there is a
/// table. A structured handle over a raw view reads zeros on NVIDIA
/// (watch.md item 160). Here the views are raw, as the host makes them.
#[test]
fn known_strides_read_raw_views_with_xtp_raw_structured() {
    super::RAW_STRUCTURED_TEST.set(Some(true));
    let data: Vec<u32> = (100..164).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::BufferUav(64, Stride::Raw),
    ];
    // Stride 16: the load of [2, 4] is dword 9, stored at [1, 8] (dword
    // 6); 10 added twice at [2, 0] (dword 8).
    let mut want = vec![0; 64];
    (want[6], want[8]) = (data[9], 20);
    let (_one, hosts) = hosts();
    // Without the table: raw handles.
    let mut plain = lowered_with(LOAD_STORE_ATOMICS, 8, ONE_THREAD, Target::ANY, 16).unwrap();
    assert!(!plain.stride_table);
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut plain.dxil).expect("DXIL validation");
    }
    let cs = HostShader {
        stage: HwStage::Cs,
        dxil: plain.dxil,
        group_base: plain.group_base,
        user_data: plain.user_data,
        samplers: plain.samplers,
        static_samplers: Vec::new(),
        root_cbvs: plain.root_cbvs,
        root_srvs: plain.root_srvs,
        root_uavs: plain.root_uavs,
        stride_table: false,
        run_time_buffers: plain.run_time_buffers,
    };
    // With the table: its raw views (each slot's own here), stride 16.
    let table = {
        let mut l = lowered_with(LOAD_STORE_ATOMICS, 8, ONE_THREAD, TABLE, 16).unwrap();
        assert!(l.stride_table, "known strides through the table");
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil).expect("DXIL validation");
        }
        HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            group_base: l.group_base,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            stride_table: true,
            run_time_buffers: l.run_time_buffers,
        }
    };
    let own = |slot| StrideEntry {
        stride: 16,
        raw_slot: Some(slot),
        bias: 0,
        count: 16,
    };
    for (name, host) in &hosts {
        let (root, pso) = host.compute_pipeline(&cs).expect("a pipeline");
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(vec![0; 5]))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("a dispatch");
        assert_eq!(out[0], want, "raw handles on {name}");
        let out = run(host, &table, [0; 4], &views, &[own(0), own(1)], 1);
        assert_eq!(out[0], want, "through the table on {name}");
    }
    super::RAW_STRUCTURED_TEST.set(None);
}
