//! Buffers of a stride known at translation through the views the GPU side
//! makes for them: the title's structured views as they are, or raw views
//! of the same bytes with `XTP_RAW_STRUCTURED=1` ([`super::raw_structured`],
//! Marvisak/VIctory#1762), with or without the runtime stride table. Hand-
//! assembled compute programs run on WARP and the default adapter (outside
//! `XTP_TEST_ADAPTER=warp`): each reads a buffer's record count (descriptor
//! dword 2) as data and accesses the buffer through the same descriptor.
//!
//! The cases follow the measurements of VIctory#1762's thread (an RTX 5080
//! and WARP): AlanQiang0418's counts by access kind (b86d853) and indexed
//! reads at strides 8 to 24 (f050b1d), which an RTX read wrong through a
//! structured handle over a raw view. Written anew for this test host, with
//! the stride table, atomics, a scalar load of an unordered view and reads
//! inside an element added.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, StrideEntry, Target};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// The views a program runs with, as the GPU side makes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `XTP_RAW_STRUCTURED` off: structured views stay structured.
    Structured,
    /// `XTP_RAW_STRUCTURED=1`: structured views become raw views.
    Raw,
    /// `XTP_RAW_STRUCTURED=1` with the runtime stride table, whose entries
    /// name each slot's own (raw) view.
    Table,
}

const MODES: [Mode; 3] = [Mode::Structured, Mode::Raw, Mode::Table];

/// The kind of the title's view of the buffer read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// A raw view (no stride in the used-resource entry).
    Raw,
    /// RGBA32F elements (16 bytes in the entry).
    Typed,
    /// Structured, elements of this many bytes.
    Structured(u32),
}

impl Kind {
    /// The used-resource entry's stride.
    fn stride(self) -> u32 {
        match self {
            Kind::Raw => 0,
            Kind::Typed => 16,
            Kind::Structured(s) => s,
        }
    }

    /// How the host views the buffer in `mode`.
    fn view(self, mode: Mode) -> Stride {
        match (self, mode) {
            (Kind::Raw, _) | (Kind::Structured(_), Mode::Raw | Mode::Table) => Stride::Raw,
            (Kind::Typed, _) => Stride::Typed,
            (Kind::Structured(s), Mode::Structured) => Stride::Structured(s),
        }
    }

    /// What the console's descriptor holds in dword 2 for [`BYTES`] bytes:
    /// elements of a structured or typed view, bytes of a raw one.
    fn records(self) -> u32 {
        match self {
            Kind::Raw => BYTES,
            Kind::Typed | Kind::Structured(_) => BYTES / self.stride(),
        }
    }
}

/// The buffer read: 48 dwords.
const BYTES: u32 = 192;

/// What the buffer holds: dword `i` is the float `100 + i` (floats, so that
/// a typed RGBA32F read returns the same bits).
fn data() -> Vec<u32> {
    (0..BYTES / 4)
        .map(|i| (100.0 + i as f32).to_bits())
        .collect()
}

/// The table at `s0` (user SGPR 0); the buffer's descriptor at table dword 0
/// into `s[8:11]`, the output's at dword 8 into `s[16:19]`; the record count
/// (`s10`) into `v1`. The access follows, then [`EPILOGUE`].
const PROLOGUE: [u32; 8] = [
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE84_0300, // s_mov_b32 s4, s0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
    0xF408_0402,
    0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_020A, // v_mov_b32 v1, s10 (the record count)
];

/// Stores the count (`v1`) and what the access left in `v2` at dwords 0
/// and 1 of the output.
const EPILOGUE: [u32; 6] = [
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_0000,
    0x8004_0100, // buffer_store_dword v1, off, s[16:19], 0
    0xE070_0004,
    0x8004_0200, // buffer_store_dword v2, off, s[16:19], 0 offset:4
    0xBF81_0000, // s_endpgm
];

/// Loads element 1 into `v2`.
const INDEXED_LOAD: &[u32] = &[
    0x7E06_0281, // v_mov_b32 v3, 1
    0xE030_2000,
    0x8002_0203, // buffer_load_dword v2, v3, s[8:11], 0 idxen
];

/// Loads byte 4 of element 1 into `v2`.
const INDEXED_LOAD_AT_4: &[u32] = &[
    0x7E06_0281, // v_mov_b32 v3, 1
    0x7E08_0284, // v_mov_b32 v4, 4
    0xE030_3000,
    0x8002_0203, // buffer_load_dword v2, v[3:4], s[8:11], 0 idxen offen
];

/// Loads byte 12 into `v2`.
const OFFSET_LOAD: &[u32] = &[
    0x7E06_028C, // v_mov_b32 v3, 12
    0xE030_1000,
    0x8002_0203, // buffer_load_dword v2, v3, s[8:11], 0 offen
];

/// Loads byte 12 with a scalar load, into `v2`.
const SCALAR_LOAD: &[u32] = &[
    0xF420_0504,
    0xFA00_000C, // s_buffer_load_dword s20, s[8:11], 0xc
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E04_0214, // v_mov_b32 v2, s20
];

/// Loads channel x of element 1 into `v2`.
const TYPED_LOAD: &[u32] = &[
    0x7E06_0281, // v_mov_b32 v3, 1
    0xE000_2000,
    0x8002_0203, // buffer_load_format_x v2, v3, s[8:11], 0 idxen
];

/// Stores 42 (`v2`) at element 1.
const INDEXED_STORE: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_0281, // v_mov_b32 v3, 1
    0xE070_2000,
    0x8002_0203, // buffer_store_dword v2, v3, s[8:11], 0 idxen
];

/// Stores 42 (`v2`) at byte 12.
const OFFSET_STORE: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_028C, // v_mov_b32 v3, 12
    0xE070_1000,
    0x8002_0203, // buffer_store_dword v2, v3, s[8:11], 0 offen
];

/// Adds 42 (`v2`, which keeps it: no glc) at element 1.
const INDEXED_ATOMIC: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_0281, // v_mov_b32 v3, 1
    0xE0C8_2000,
    0x8002_0203, // buffer_atomic_add v2, v3, s[8:11], 0 idxen
];

/// A used-resource entry of the table at `s0`.
fn entry(kind: ResourceType, table_offset_dwords: u32, stride: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }
}

/// One thread running [`PROLOGUE`], `access`, [`EPILOGUE`] over a buffer of
/// `kind` (an unordered view where `uav`) and a raw output.
fn program(access: &[u32], kind: Kind, uav: bool) -> NativeProgram {
    let words: Vec<u32> = PROLOGUE
        .iter()
        .chain(access)
        .chain(&EPILOGUE)
        .copied()
        .collect();
    let source = if uav {
        ResourceType::Uav
    } else {
        ResourceType::Srv
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            entry(source, 0, kind.stride()),
            entry(ResourceType::Uav, 8, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
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

/// `native` translated for `mode` (with [`super::raw_structured`] as the
/// mode has it), validated where `dxil.dll` is.
fn translate(native: &NativeProgram, mode: Mode) -> Result<HostShader, crate::ShaderError> {
    let _raw = RawStructured::set(mode != Mode::Structured);
    let target = Target {
        stride_table: mode == Mode::Table,
        ..Target::ANY
    };
    let mut cs = super::tests::emit_for(native, target)?;
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut cs.dxil).expect("DXIL validation");
    }
    Ok(cs)
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
    if std::env::var("XTP_REQUIRE_GPU").as_deref() == Ok("1") {
        let required = if super::host::test_adapter().is_some() {
            "WARP"
        } else {
            "the default adapter"
        };
        assert!(
            hosts.iter().any(|(name, _)| *name == required),
            "missing required {required}"
        );
    }
    (one, hosts)
}

/// Runs `cs` over the buffer of `kind` holding [`data`] (an unordered view
/// of zeros where `uav`) in heap slot 0 and a raw output in
/// slot 1, as `mode` views them: what each unordered view read back holds.
fn run(
    host: &Host,
    cs: &HostShader,
    kind: Kind,
    uav: bool,
    mode: Mode,
    output_words: u32,
) -> Vec<Vec<u32>> {
    let data = data();
    let stride = kind.view(mode);
    let views = [
        if uav {
            View::BufferUav(BYTES / 4, stride)
        } else {
            View::Buffer {
                data: &data,
                stride,
            }
        },
        View::RawUav(output_words),
    ];
    let mut roots = vec![(
        0,
        Root::Constants(vec![0; cs.user_data_dwords().max(1) as usize]),
    )];
    if !cs.root_srvs.is_empty() {
        roots.push((
            1,
            Root::View {
                srv: true,
                data: data.clone(),
            },
        ));
    }
    if cs.stride_table {
        // Each slot's own view, raw here (a typed one has no raw view of
        // its own: typed loads do not read the table).
        let own = |slot, stride, count| StrideEntry {
            stride,
            raw_slot: Some(slot),
            bias: 0,
            count,
        };
        let source = match kind {
            Kind::Raw => own(0, 0, BYTES / 4),
            Kind::Typed => StrideEntry::NONE,
            Kind::Structured(s) => own(0, s, BYTES / s),
        };
        let table: Vec<u32> = [source, own(1, 0, output_words)]
            .iter()
            .flat_map(|e| {
                let bits = e.bits().expect("an entry the table holds");
                [bits as u32, (bits >> 32) as u32]
            })
            .collect();
        // The table's header first (`emit::STRIDE_HEADER`): both slots mirrored.
        let table = [vec![0, 2, 0, 2], table].concat();
        roots.push((
            roots.len() as u32,
            Root::View {
                srv: true,
                data: table,
            },
        ));
    }
    let (root, pso) = host.compute_pipeline(cs).expect("a pipeline");
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots,
            views: &views,
            samplers: &[],
            groups: [1, 1, 1],
        },
    )
    .expect("a dispatch")
}

/// A program of [`program`] and what it must give.
struct Case {
    name: String,
    access: &'static [u32],
    kind: Kind,
    uav: bool,
    /// The access reads or writes raw (by byte offset, or a scalar load):
    /// over a structured view (`XTP_RAW_STRUCTURED` off) it is the very
    /// mismatch `XTP_RAW_STRUCTURED` removes (NVIDIA reads other dwords or
    /// zeros, watch.md items 148, 160), so only its count is checked there.
    raw_access: bool,
    /// Dword 1 of the output: the value loaded, or the one stored.
    value: u32,
    /// For unordered views, the buffer after the access: `(k, v)`, `v` at
    /// dword `k` and 0 everywhere else.
    stored: Option<(usize, u32)>,
    /// The views it runs with.
    modes: &'static [Mode],
}

/// Runs every case in each of its modes on every host; the failures.
fn check(cases: &[Case]) -> Vec<String> {
    let (_one, hosts) = hosts();
    let mut failures = Vec::new();
    for case in cases {
        let native = program(case.access, case.kind, case.uav);
        for &mode in case.modes {
            let what = format!("{} ({mode:?} views)", case.name);
            let cs = match translate(&native, mode) {
                Ok(cs) => cs,
                Err(e) => {
                    failures.push(format!("{what}: not translated: {e}"));
                    continue;
                }
            };
            let values = !(case.raw_access
                && mode == Mode::Structured
                && matches!(case.kind, Kind::Structured(_)));
            for (host_name, host) in &hosts {
                let out = run(host, &cs, case.kind, case.uav, mode, 2);
                let got = out.last().expect("the output");
                let count = case.kind.records();
                if got[0] != count {
                    failures.push(format!(
                        "{what} on {host_name}: record count {}, not {count}",
                        got[0]
                    ));
                }
                if values && got[1] != case.value {
                    failures.push(format!(
                        "{what} on {host_name}: value {:#x}, not {:#x}",
                        got[1], case.value
                    ));
                }
                if let Some((k, v)) = case.stored.filter(|_| values) {
                    let mut want = vec![0; (BYTES / 4) as usize];
                    want[k] = v;
                    if out[0] != want {
                        let set: Vec<(usize, u32)> = out[0]
                            .iter()
                            .copied()
                            .enumerate()
                            .filter(|&(_, x)| x != 0)
                            .collect();
                        failures.push(format!(
                            "{what} on {host_name}: the buffer's non-zero dwords {set:?}, not \
                             [({k}, {v})]"
                        ));
                    }
                }
            }
        }
    }
    failures
}

/// A case of `access` over a buffer of `kind` (an unordered view where
/// `uav`), run in every mode.
fn case(
    name: &str,
    access: &'static [u32],
    kind: Kind,
    uav: bool,
    raw_access: bool,
    value: u32,
    stored: Option<(usize, u32)>,
) -> Case {
    Case {
        name: name.to_owned(),
        access,
        kind,
        uav,
        raw_access,
        value,
        stored,
        modes: &MODES,
    }
}

/// A buffer's record count read as data is the console's whatever the host
/// views and however the program accesses the buffer: 16 for 192 bytes of
/// 12-byte structures, by index, by byte offset, by a scalar load, through
/// a shader or an unordered view (stores and atomics); 192 bytes for a raw
/// view, 12 for an RGBA32F one. With `XTP_RAW_STRUCTURED=1` a structured
/// handle's size of the raw view the host makes is other numbers on
/// NVIDIA, a raw handle's size its bytes; by byte offset or a scalar load
/// the count was the bytes whatever the view, and a count read beside a
/// scalar load stopped the translation.
#[test]
fn record_counts_read_as_data_are_the_consoles() {
    let data = data();
    let s12 = Kind::Structured(12);
    let raw = Kind::Raw;
    // Stores and atomics write 42 at dword 3 (byte 12), nothing else.
    let at3 = Some((3, 42));
    #[rustfmt::skip]
    let cases = [
        case("indexed load", INDEXED_LOAD, s12, false, false, data[3], None),
        case("offset load", OFFSET_LOAD, s12, false, true, data[3], None),
        case("scalar load", SCALAR_LOAD, s12, false, true, data[3], None),
        case("raw offset load", OFFSET_LOAD, raw, false, true, data[3], None),
        case("raw scalar load", SCALAR_LOAD, raw, false, true, data[3], None),
        case("typed load", TYPED_LOAD, Kind::Typed, false, false, data[4], None),
        case("indexed store", INDEXED_STORE, s12, true, false, 42, at3),
        case("offset store", OFFSET_STORE, s12, true, true, 42, at3),
        case("indexed atomic", INDEXED_ATOMIC, s12, true, false, 42, at3),
        case("scalar load of an unordered view", SCALAR_LOAD, s12, true, true, 0, None),
        case("raw offset store", OFFSET_STORE, raw, true, true, 42, at3),
    ];
    let failures = check(&cases);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A load by element reads `index * stride` (plus the offset) at every
/// stride, and the count is the elements: an RTX 5080 read dword 1 or 2 of
/// element 1 at strides 8 to 24 through a structured handle over a raw
/// view (VIctory#1762), so with `XTP_RAW_STRUCTURED=1` such loads go
/// through raw handles (or the table's raw views); without it, the
/// structured handle reads the structured view the title made.
#[test]
fn loads_by_element_read_index_times_stride_at_every_stride() {
    let data = data();
    let mut cases = Vec::new();
    for s in [4, 8, 12, 16, 24] {
        let at = (s / 4) as usize;
        let kind = Kind::Structured(s);
        cases.push(case(
            &format!("element 1, stride {s}"),
            INDEXED_LOAD,
            kind,
            false,
            false,
            data[at],
            None,
        ));
        // Byte 4 of the element (inside it from stride 8 on).
        if s >= 8 {
            cases.push(case(
                &format!("byte 4 of element 1, stride {s}"),
                INDEXED_LOAD_AT_4,
                kind,
                false,
                false,
                data[at + 1],
                None,
            ));
        }
    }
    let failures = check(&cases);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Loads byte 12 of element 1 into `v2`: past a 12-byte element.
const LOAD_PAST_THE_ELEMENT: &[u32] = &[
    0x7E06_0281, // v_mov_b32 v3, 1
    0x7E08_028C, // v_mov_b32 v4, 12
    0xE030_3000,
    0x8002_0203, // buffer_load_dword v2, v[3:4], s[8:11], 0 idxen offen
];

/// Stores 42 (`v2`) at byte 12 of element 1.
const STORE_PAST_THE_ELEMENT: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_0281, // v_mov_b32 v3, 1
    0x7E08_028C, // v_mov_b32 v4, 12
    0xE070_3000,
    0x8002_0203, // buffer_store_dword v2, v[3:4], s[8:11], 0 idxen offen
];

/// Adds 42 (`v2`) at byte 12 of element 1.
const ATOMIC_PAST_THE_ELEMENT: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_0281, // v_mov_b32 v3, 1
    0x7E08_028C, // v_mov_b32 v4, 12
    0xE0C8_3000,
    0x8002_0203, // buffer_atomic_add v2, v[3:4], s[8:11], 0 idxen offen
];

/// Loads element 2^30 + 1 into `v2`: at a stride of 4, `index * stride` is
/// 2^32 + 4, byte 4 once it wraps.
const LOAD_WRAPPING: &[u32] = &[
    0x7E06_02FF,
    0x4000_0001, // v_mov_b32 v3, 0x40000001
    0xE030_2000,
    0x8002_0203, // buffer_load_dword v2, v3, s[8:11], 0 idxen
];

/// Stores 42 (`v2`) at element 2^30 + 1.
const STORE_WRAPPING: &[u32] = &[
    0x7E04_02AA, // v_mov_b32 v2, 42
    0x7E06_02FF,
    0x4000_0001, // v_mov_b32 v3, 0x40000001
    0xE070_2000,
    0x8002_0203, // buffer_store_dword v2, v3, s[8:11], 0 idxen
];

/// With `XTP_RAW_STRUCTURED=1` an access by element through a raw handle
/// makes the console's range checks of a structured view, as the stride
/// table's raw views do: at an element's end (byte 12 of a 12-byte
/// element, the next one's first dword) or at an element whose bytes pass
/// 2^32 (`index * stride` wrapping to a byte inside the view), a load reads
/// 0 and a store or an atomic leaves the buffer as it was (Codex's review
/// of the stride table's landing, board task 01dada4d). Not run with
/// structured views, whose checks are the host's own.
#[test]
fn accesses_past_their_element_or_wrapping_reach_nothing() {
    const RAW: &[Mode] = &[Mode::Raw, Mode::Table];
    let (s12, s4) = (Kind::Structured(12), Kind::Structured(4));
    // Nothing written: dword 6 (the next element's first, at stride 12) and
    // dword 1 (where the wrapped store lands) stay 0, as all the others.
    #[rustfmt::skip]
    let cases = [
        case("a load past the element", LOAD_PAST_THE_ELEMENT, s12, false, false, 0, None),
        case("a store past the element", STORE_PAST_THE_ELEMENT, s12, true, false, 42, Some((6, 0))),
        case("an atomic past the element", ATOMIC_PAST_THE_ELEMENT, s12, true, false, 42, Some((6, 0))),
        case("a load at a wrapping index", LOAD_WRAPPING, s4, false, false, 0, None),
        case("a store at a wrapping index", STORE_WRAPPING, s4, true, false, 42, Some((1, 0))),
    ]
    .map(|c| Case { modes: RAW, ..c });
    let failures = check(&cases);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Read every result register and the entire destination UAV. The first
/// one, two or three components fit for x2, x3 and x4 respectively.
#[test]
fn raw_multidword_transfers_keep_each_in_range_component() {
    let (_one, hosts) = hosts();
    let kind = Kind::Structured(12);
    let source = data();
    for (count, opcode, offset) in [(2u32, 13u32, 8u32), (3, 15, 4), (4, 14, 0)] {
        for store in [false, true] {
            let mut access = vec![0x7e0c_0281]; // index 1 in v6
            for k in 0..count {
                // Preload v2 onwards with distinct nonzero values.
                access.push(0x7e04_02aa + (k << 17) + k);
            }
            let op = opcode + if store { 16 } else { 0 };
            access.extend([0xe000_2000 | (op << 18) | offset, 0x8002_0206]);
            access.push(0xbf8c_3f70);
            for k in 0..count {
                // Export the registers independently to output dwords 2.. .
                access.extend([0xe070_0008 + 4 * k, 0x8004_0200 + (k << 8)]);
            }
            let p = program(&access, kind, store);
            for mode in [Mode::Raw, Mode::Table] {
                let cs = translate(&p, mode).unwrap();
                if !store && let Some(ir) = super::smem_tests::disassembly(&cs) {
                    // La table se lit par paires de dwords. Aucun load scalaire
                    // ne doit apparaître dans ces programmes de lecture vectorielle.
                    let loads: Vec<_> = ir
                        .lines()
                        .filter(|line| {
                            line.contains("call") && line.contains("@dx.op.rawBufferLoad")
                        })
                        .collect();
                    let mask = format!("i8 {}, i32 4)", (1 << count) - 1);
                    assert!(
                        loads.iter().any(|line| line.contains(&mask)),
                        "x{count} {mode:?} : {loads:?}"
                    );
                    assert!(
                        loads.iter().all(|line| !line.contains("i8 1, i32 4)")),
                        "x{count} {mode:?} : {loads:?}"
                    );
                    if mode == Mode::Raw {
                        assert_eq!(loads.len(), 1);
                    }
                    eprintln!(
                        "x{count} {mode:?} : {} octets DXIL, {} lectures, {} handles du tas, {} branches conditionnelles",
                        cs.dxil.len(),
                        loads.len(),
                        ir.lines()
                            .filter(|line| line.contains("call")
                                && line.contains("@dx.op.createHandleFromHeap"))
                            .count(),
                        ir.matches("br i1 ").count()
                    );
                }
                for (name, host) in &hosts {
                    let out = run(host, &cs, kind, store, mode, 2 + count);
                    let start = (12 + offset) as usize / 4;
                    let mut want = vec![0; count as usize];
                    let fits = count as usize - 1;
                    if store {
                        let mut memory = vec![0; (BYTES / 4) as usize];
                        for (k, v) in want.iter_mut().enumerate() {
                            *v = 42 + k as u32;
                        }
                        memory[start..start + fits].copy_from_slice(&want[..fits]);
                        assert_eq!(out[0], memory, "store x{count} {mode:?} on {name}");
                    } else {
                        want[..fits].copy_from_slice(&source[start..start + fits]);
                    }
                    assert_eq!(
                        out.last().unwrap()[2..],
                        want,
                        "x{count} {mode:?} on {name}"
                    );
                }
            }
        }
    }
}

#[test]
fn raw_scalar_offset_moves_the_base_for_load_store_and_atomic() {
    let (_one, hosts) = hosts();
    let kind = Kind::Structured(12);
    for (op, uav) in [
        (0xe030_2000, false),
        (0xe070_2000, true),
        (0xe0c8_2000, true),
    ] {
        let p = program(
            &[
                0x7e04_02aa, // v2 = 42
                0x7e06_0280, // v3 = 0 (index)
                0xbe94_038c, // s20 = 12 (SOFFSET)
                op,
                0x1402_0203,
            ],
            kind,
            uav,
        );
        for mode in [Mode::Raw, Mode::Table] {
            let cs = translate(&p, mode).unwrap();
            for (name, host) in &hosts {
                let out = run(host, &cs, kind, uav, mode, 2);
                if uav {
                    let mut memory = vec![0; (BYTES / 4) as usize];
                    memory[3] = 42;
                    assert_eq!(out[0], memory, "{op:x} {mode:?} on {name}");
                } else {
                    assert_eq!(out[0][1], data()[3], "{mode:?} on {name}");
                }
            }
        }
    }
}

/// Root SRV at s[2:3], descriptor at s[8:11], base displaced by 12.
fn rebuilt_program(
    word1: u32,
    records: u32,
    word3: u32,
    threads: u32,
    access: &[u32],
) -> NativeProgram {
    let mut words = vec![
        PROLOGUE[0],
        PROLOGUE[1],
        PROLOGUE[4],
        PROLOGUE[5],
        PROLOGUE[6],
        0x8008_8c02, // s_add_u32 s8, s2, 12
        0x8209_8003, // s_addc_u32 s9, s3, 0
        0x8809_ff09,
        word1, // s_or_b32 s9, s9, contrôles du descripteur
        0xbe8a_03ff,
        records,
        0xbe8b_03ff,
        word3,
    ];
    words.extend_from_slice(access);
    let mut p = program(&[], Kind::Raw, false);
    p.code = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    p.layout.push(RootParam {
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        sgprs: 2,
        base_register: 0,
        user_data: 2,
        slot: 1,
    });
    p.used[0] = UsedResource {
        root: RootType::Descriptor,
        user_data: 2,
        slot: 1,
        ..entry(ResourceType::Srv, 0, 0)
    };
    if let StageIo::Cs(io) = &mut p.io {
        io.rsrc2 = 4 << 1;
        io.threads = [threads, 1, 1];
    }
    p
}

#[test]
fn raw_rebuilt_descriptor_uses_its_base_count_and_oob_select() {
    let (_one, hosts) = hosts();
    // Le stride vient du mot 1 du descripteur reconstruit.
    for (oob, stride, records, index, within, scalar, want) in [
        (0, 12, 2, 0, 0, 0, Some(3)),
        (0, 12, 2, 0, 0, 12, Some(6)),
        (0, 12, 2, 0, 12, 0, None),
        (0, 12, 2, 2, 0, 0, None),
        (0, 6, 2, 0, 4, 0, None), // Deux octets du dword dépassent du record.
        (1, 12, 2, 0, 12, 0, Some(6)),
        (1, 12, 2, 2, 0, 0, None),
        (2, 12, 1, 2, 12, 0, Some(12)),
        (2, 12, 0, 0, 0, 0, None),
        (3, 12, 16, 0, 0, 12, Some(6)),
        (3, 12, 16, 0, 4, 12, None),
        (3, 12, 14, 0, 0, 12, None),
        (3, 12, 14, 0, 12, 0, None),
        (3, 0, 20, 0, 8, 8, Some(7)), // Le dernier octet tient exactement.
        // Vecteurs de docs/buffer_bounds.md, projet de @ubiquitarianism
        // (GPL-3.0, mesures SHADER-116 et SHADER-341 sur gfx1036/gfx1011).
        // Source également citée par rminzao : https://github.com/Marvisak/VIctory/pull/2084
        (3, 0, 18, 0, 8, 8, None),
        (3, 16, 18, 0, 8, 8, None),
        (3, 18, 18, 0, 8, 8, None),
        (3, 16, 18, 4, 0, 0, Some(19)), // Index × stride déplace seulement l'adresse.
        (0, 16, 18, 0, 13, 0, None),
        (0, 16, 18, 0, 12, 0, Some(6)),
    ] {
        // OFFEN fournit l'offset en octets, comme dans la sonde matérielle.
        let (op, addr) = if index == 0 {
            (0xe030_1000, 0x1402_0204) // v4 (offset)
        } else {
            (0xe030_3000, 0x1402_0203) // v[3:4] (index, offset)
        };
        let mut access = vec![
            0x7e02_0280, // v1 = 0
            0x7e06_0280 + index,
            0x7e08_0280 + within,
            0xbe94_0380 + scalar,
            op,
            addr,
        ];
        access.extend(EPILOGUE);
        let p = rebuilt_program(stride << 16, records, 0x0101_4fac | (oob << 28), 1, &access);
        for mode in [Mode::Raw, Mode::Table] {
            let cs = translate(&p, mode).unwrap();
            assert_eq!(cs.root_srvs, [1]);
            for (name, host) in &hosts {
                let out = run(host, &cs, Kind::Raw, false, mode, 2);
                assert_eq!(
                    out[0][1],
                    want.map_or(0, |i| data()[i]),
                    "OOB_SELECT={oob} count={records} index={index} offset={within} scalar={scalar} {mode:?} on {name}"
                );
            }
        }
    }
}

#[test]
fn raw_add_tid_indexes_by_console_lane_with_and_without_idxen() {
    let (_one, hosts) = hosts();
    let source = data();
    for wave in [32, 64] {
        for (opcode, idxen, add_tid) in [
            (0xe030_0000, false, false),
            (0xe030_0000, false, true),
            (0xe030_0000, true, true),
            (0xe800_0000 | (20 << 19), false, true), // MTBUF 32_UINT
            (0xe800_0000 | (20 << 19), true, true),
        ] {
            let access = [
                0x7e06_0282, // v3 = 2 (used only with IDXEN)
                opcode | (u32::from(idxen) << 13),
                0x8002_0203,
                0xbf8c_3f70,
                0x3408_0082, // v4 = group thread x * 4
                0xe070_1000,
                0x8004_0204, // output[v4] = v2, OFFEN
                0xbf81_0000,
            ];
            let word3 = 0x0101_4fac | (u32::from(add_tid) << 23);
            let mut p = rebuilt_program(4 << 16, 40, word3, 96, &access);
            p.wave64 = wave == 64;
            let want: Vec<_> = (0..96)
                .map(|x| {
                    let lane = if add_tid { x % wave } else { 0 };
                    let index = lane + usize::from(idxen) * 2;
                    if index < 40 { source[3 + index] } else { 0 }
                })
                .collect();
            for mode in [Mode::Raw, Mode::Table] {
                let cs = translate(&p, mode).unwrap();
                for (name, host) in &hosts {
                    let out = run(host, &cs, Kind::Raw, false, mode, 96);
                    assert_eq!(
                        out[0], want,
                        "{opcode:x} wave{wave} idxen={idxen} ADD_TID_ENABLE={add_tid} {mode:?} on {name}"
                    );
                }
            }
        }
    }
}

#[test]
fn raw_rebuilt_vector_and_swizzled_format_respect_the_record_end() {
    let (_one, hosts) = hosts();
    let src = data();
    // Index 9 : deuxième groupe de 8, lane 1. SOFFSET décale la base
    // d'un dword après le swizzle. En MTBUF les deux canaux doivent tenir.
    for (swizzle, opcode, want) in [
        (false, 0xe038_2000, vec![src[31], src[32], src[33], 0]),
        (true, 0xe038_2000, vec![src[29], src[37], src[45], 0]),
        (true, 0xe801_2004 | (62 << 19), vec![src[37], src[45]]),
        (true, 0xe801_2008 | (62 << 19), vec![0, 0]),
    ] {
        let mut access = vec![
            0x7e0c_0289, // v6 = 9
            0xbe94_0384, // s20 = 4
            opcode,
            0x1402_0206,
            0xbf8c_3f70,
        ];
        for k in 0..want.len() as u32 {
            access.extend([0xe070_0000 + 4 * k, 0x8004_0200 + (k << 8)]);
        }
        access.push(0xbf81_0000);
        let word1 = (12 << 16) | (u32::from(swizzle) << 31);
        let oob = if swizzle { 3 } else { 0 };
        let p = rebuilt_program(word1, 10, 0x0101_4fac | (oob << 28), 1, &access);
        for mode in [Mode::Raw, Mode::Table] {
            let cs = translate(&p, mode).unwrap();
            for (name, host) in &hosts {
                let out = run(host, &cs, Kind::Raw, false, mode, want.len() as u32);
                assert_eq!(
                    out[0], want,
                    "{opcode:x} swizzle={swizzle} {mode:?} sur {name}"
                );
            }
        }
    }
}
