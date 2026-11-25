//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content, by the translator that
//! made it (the fingerprint of `xtp-shader`'s sources, [`xtp_shader::SOURCES`],
//! and of this file, with the translation settings: [`build`]), so a
//! changed translator never serves an old translation while a build that
//! changed only elsewhere keeps them, and by the wave sizes it was made for
//! (VIctory #223: a translation for a 32-lane host is not a 64-lane one's).
//! Entries go through `xtp-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway: this file is part of the key).
//!
//! A container this build cannot translate for a target is cached under the
//! same key as a translation. The disk cache verifies the payload's length
//! and digest first. The shader decoder then checks the refusal's marker,
//! text length, checksum and UTF-8. The translator's refusals
//! ([`xtp_shader::ShaderError`]: a malformed container, GFX10 code it
//! cannot decode or does not handle yet) depend only on the container and
//! on what `key()` hashes (our executable, its loop limit, the target's
//! wave sizes), so it would fail the same way again. Without that, every
//! run translated the same few hundred refused programs again (seconds each
//! for the large ones) while the title waited for them. Nothing transient
//! is kept: the host driver's compile, where out of memory and a removed
//! device happen, comes after this cache, and an entry that cannot be
//! written is answered from the translation without being kept.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use xtp_pipeline::{Cache, Converter};
use xtp_shader::emit::{HostShader, StaticSampler, Target, UserData};
use xtp_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// The disk the cache may take, oldest entries out first. One translator's
/// answers for the whole pack take 1.1 GiB (25,889 entries, `xtp precook`
/// on the PC, 2026-11-25): more than the former 1 GiB, which evicted what
/// the precook or the session before had made. Room for a few translators
/// (switches on and off) side by side.
const LIMIT: u64 = 8 << 30;

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open_with_limit(dir, LIMIT) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start. Refusals,
/// fresh or kept, are neither: the frame line counts shaders.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Counts `result` in `counter` if it is a shader (not a refusal).
fn count<T, E>(counter: &AtomicU64, result: Result<T, E>) -> Result<T, E> {
    if result.is_ok() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    result
}

/// What identifies the translator this build runs: the fingerprint of
/// `xtp-shader`'s sources ([`xtp_shader::SOURCES`]), this file (the
/// entries' encoding, [`ENCODING`]) and the translation settings it runs
/// with (the loop limit, the loops' budget, the DXIL clean-up passes and
/// the raw structured reads, which any build can change at run time). Not
/// the whole executable: before, the least change anywhere (the HLE, a log
/// line, another worker's branch) made every translation again, and each
/// worktree's sessions started cold.
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let limit = xtp_shader::emit::loop_limit();
        if limit != xtp_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(limit, "XTP_SHADER_LOOP_LIMIT: translated loops stop sooner");
        }
        let budget = xtp_shader::emit::loop_budget();
        if budget != xtp_shader::emit::DEFAULT_LOOP_BUDGET {
            tracing::warn!(
                budget,
                "XTP_SHADER_LOOP_BUDGET: back edges a lane may take in all loops together (0: no budget)"
            );
        }
        // And the clean-up passes run over the translated code.
        let passes = xtp_shader::emit::dxil_opt();
        if passes != 0 {
            tracing::info!(passes, "XTP_DXIL_OPT: clean-up passes run over translated shaders");
        }
        // And structured reads through raw handles (XTP_RAW_STRUCTURED).
        let raw = u32::from(xtp_shader::emit::raw_structured());
        build_of(xtp_shader::SOURCES, ENCODING, [limit, budget, passes, raw])
    })
}

/// This file: how entries are encoded and decoded, and which translation
/// each holds, so a change to it is a new key.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

/// [`build`]'s key from its parts: the translator's sources, the entries'
/// encoding and the settings (loop limit, loop budget, clean-up passes, raw
/// structured reads).
fn build_of(sources: &str, encoding: &[u8], settings: [u32; 4]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for part in [sources.as_bytes(), encoding] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    for setting in settings {
        h.update(&setting.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// The key this run's translations are kept under: this build and the
/// host's target ([`crate::device::host_target`]). `xtp precook` notes it
/// beside a complete pass, to know a pass for this key was made.
#[cfg(windows)]
pub(crate) fn host_key() -> [u8; 32] {
    key(build(), crate::device::host_target())
}

/// The translation of a console shader container for a host that runs
/// `target`'s wave sizes, from the cache when there is one.
pub(crate) fn host_shader(container: &[u8], target: Target) -> Result<HostShader, String> {
    let translate =
        |c: &[u8]| xtp_shader::emit::translate_for(c, target).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => kept(cache, container, target, translate),
        None => translate(container),
    }
}

/// The translation of the export shader (ES) of a console vertex shader
/// container, the vertex shader of a pipeline with a geometry shader
/// (`xtp_shader::emit::translate_es_for`), from the cache when there is
/// one: under a key of its own, the container being the same.
pub(crate) fn host_es_shader(container: &[u8], target: Target) -> Result<HostShader, String> {
    let translate =
        |c: &[u8]| xtp_shader::emit::translate_es_for(c, target).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, &es_key(&key(build(), target)), container, translate),
        None => translate(container),
    }
}

/// An ES translation's key ([`host_es_shader`]) from the container's
/// translation's ([`key`]).
fn es_key(key: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(b"export shader");
    *h.finalize().as_bytes()
}

/// `translate`'s translation of `container` for `target` from `cache`,
/// under this build's key for that target; made and stored on a miss.
/// The target that keys the entry is the one `translate` is for: the
/// caller passes both. An error of `translate` is kept too, as the
/// container's answer for this build and target: it must be one that would
/// come again (a refusal of the translator), never a passing condition.
pub(crate) fn kept(
    cache: &Cache,
    container: &[u8],
    target: Target,
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    cached(cache, &key(build(), target), container, translate)
}

/// The part of an entry's key before the container: the build, and the
/// wave sizes translated for (fixed sizes, so no two pairs run together).
fn key(build: &[u8; 32], target: Target) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&target.key());
    *h.finalize().as_bytes()
}

/// The cache entry's converter: the key ([`key`]) and the container in,
/// the encoded translation (or refusal) out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F> Translate<F> {
    fn new(translate: F) -> Translate<F> {
        Translate {
            translate,
            ran: AtomicBool::new(false),
        }
    }
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        Ok(match (self.translate)(&input[32..]) {
            Ok(shader) => encode(&shader),
            Err(error) => encode_failure(&error),
        })
    }
}

fn cached(
    cache: &Cache,
    key: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&key[..], container].concat();
    let converter = Translate::new(translate);
    let result = cache
        .get_or_convert_with(&converter, &input, |bytes| {
            decode(&bytes)
                .map(Ok)
                .or_else(|| decode_failure(&bytes).map(Err))
        })
        .map_err(|e| format!("{e:#}"))?;
    let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
        (&MISSES, "translated")
    } else {
        (&HITS, "from the cache")
    };
    tracing::debug!(ok = result.is_ok(), "shader {what}");
    count(counter, result)
}

/// For tests: [`wave64_program`] with its lane reads.
#[cfg(test)]
pub(crate) fn wave64_in_half_program() -> (xtp_shader::Container<'static>, xtp_shader::NativeProgram)
{
    wave64_program(true)
}

/// For tests: a console wave64 compute program of 64 threads that stores
/// one dword per thread, dword 64 g + t for thread t of group g, through a
/// raw UAV whose table is user data 0, with an empty container. With
/// `in_half`, its lanes read only inside their 32-lane half (a `ds_swizzle`
/// xor 16, then a DPP row mirror) and thread t stores t ^ 31: translated
/// for a host without 64-lane waves it runs as two 32-lane waves and
/// declares `WaveSize(32)`, else `WaveSize(64)`. Without, thread t stores
/// t and the translation runs in waves of any size. The code is the wave64
/// port's own test program (`emit/split_tests.rs`): ported from
/// Marvisak/VIctory#233 (d1fecaa) by r4tedx.
#[cfg(test)]
pub(crate) fn wave64_program(
    in_half: bool,
) -> (xtp_shader::Container<'static>, xtp_shader::NativeProgram) {
    use xtp_shader::native::{ComputeIo, RootParam, StageIo};
    let threads = 64;
    let reads: &[u32] = if in_half {
        &[
            0xD8D4_401F,
            0x0100_0000, // ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
            0x7E04_02FA,
            0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf
        ]
    } else {
        &[0x7E04_0300] // v_mov_b32 v2, v0
    };
    let words = [
        &[
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0101,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
            0x9308_FF01,
            threads * 4, // s_mul_i32 s8, s1, threads * 4
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ][..],
        reads,
        &[
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0x4A06_0608, // v_add_nc_u32 v3, s8, v3
            0xE070_1000,
            0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
            0xBF81_0000, // s_endpgm
        ],
    ]
    .concat();
    let native = xtp_shader::NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        float_mode: 0,
        vgprs: 8,
        // A UAV table in user SGPR 0; the group id in s1.
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 7,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = xtp_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    (container, native)
}

/// A translation's entry: this, then [`encode`]'s words. 2: with whether it
/// reads the runtime stride table (VIctory #193) and the buffers it reads
/// through slots picked at run time.
/// Version 3 ajoute la base XYZ des dispatches fractionnés.
const MAGIC: &[u8; 4] = b"VSH4";
/// A refusal's entry: this, then the text's length (little endian, 4 bytes),
/// then a checksum of the text (4 bytes), then the error's text (UTF-8). The
/// length tells a cut-short entry from a shorter error, and the checksum
/// tells a changed byte in an entry of the right length: either is damage,
/// and is translated again rather than served with a wrong reason.
const FAILED: &[u8; 4] = b"VSF2";
/// The bytes the payload's length and checksum take, after [`FAILED`].
const LENGTH: usize = 4;
const CHECK: usize = 4;

/// The first four bytes of the text's BLAKE3: enough to call a changed byte
/// damage (this is not against an adversary, only against a torn write).
fn checksum(text: &[u8]) -> [u8; 4] {
    let h = blake3::hash(text);
    h.as_bytes()[..CHECK].try_into().unwrap()
}

fn encode_failure(error: &str) -> Vec<u8> {
    let mut entry = Vec::with_capacity(FAILED.len() + LENGTH + CHECK + error.len());
    entry.extend_from_slice(FAILED);
    entry.extend_from_slice(&(error.len() as u32).to_le_bytes());
    entry.extend_from_slice(&checksum(error.as_bytes()));
    entry.extend_from_slice(error.as_bytes());
    entry
}

fn decode_failure(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(FAILED.as_slice())?;
    let (head, text) = rest.split_at_checked(LENGTH + CHECK)?;
    let length = u32::from_le_bytes(head[..LENGTH].try_into().ok()?) as usize;
    if text.len() != length || checksum(text) != head[LENGTH..] {
        return None;
    }
    String::from_utf8(text.to_vec()).ok()
}
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, u32::from(s.stride_table));
    word(&mut out, u32::from(s.run_time_buffers));
    word(&mut out, s.group_base.unwrap_or(u32::MAX));
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let stride_table = match r.word()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    let run_time_buffers = u8::try_from(r.word()?).ok()?;
    let group_base = match r.word()? {
        u32::MAX => None,
        n if n <= 60 && n.is_multiple_of(4) => Some(n),
        _ => return None,
    };
    r.0.is_empty().then_some(HostShader {
        group_base,
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        stride_table,
        run_time_buffers,
    })
}

/// For tests: held by every test that goes through [`cached`] (here and in
/// `pipeline::host`), so that one reading the process-wide hit and miss
/// counters sees only its own.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static CACHE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    CACHE_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame line counts shaders ("from the cache and translated so
    /// far"): a refusal, fresh or served from its entry, is neither and
    /// moves no counter; a shader is counted translated once, then from
    /// the cache. Ported from Marvisak/VIctory#1094 (commit e090242) by
    /// adamdotpy.
    #[test]
    fn refusals_are_not_counted_as_translated() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [9u8; 32];
        let container = b"yet another container that does not translate".as_slice();
        let fail = |_: &[u8]| Err::<HostShader, _>("a scalar memory read as data".to_owned());
        let error = Err("a scalar memory read as data".to_owned());

        let (h0, m0) = counts();
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error, "from disk");
        let (h1, m1) = counts();
        assert_eq!(
            (h1 - h0, m1 - m0),
            (0, 0),
            "a refusal, fresh or kept, is neither a hit nor a translation"
        );

        // Another build, so the kept refusal is not in the way.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h2, m2) = counts();
        assert_eq!((h2 - h1, m2 - m1), (0, 1), "translated once");
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h3, m3) = counts();
        assert_eq!((h3 - h2, m3 - m2), (1, 0), "then from the cache");
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            stride_table: true,
            run_time_buffers: 3,
            group_base: Some(12),
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn failures_come_back_from_the_cache_too() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a container that does not translate".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data is not supported".to_owned())
        };
        let error = Err("a scalar memory read as data is not supported".to_owned());
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build tries again, and may translate it.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[8; 32], container, ok), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // The disk header is valid, but the refusal's own checksum is stale.
        let input = [&build[..], container].concat();
        let converter = Translate::new(fail);
        let entry = xtp_pipeline::CacheKey::new(&converter, &input);
        let mut damaged = encode_failure("unsupported operand");
        *damaged.last_mut().unwrap() ^= 1;
        let frame = [
            b"XTP\x01".as_slice(),
            &(damaged.len() as u64).to_le_bytes(),
            blake3::hash(&damaged).as_bytes(),
            &damaged,
        ]
        .concat();
        std::fs::write(cache.path_for(&entry), frame).unwrap();
        assert_eq!(
            cache.get_or_convert(&converter, &input).unwrap().as_ref(),
            damaged
        );
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "the rebuilt refusal was cached"
        );
    }

    #[test]
    fn failure_entries_round_trip_and_are_no_shader() {
        let bytes = encode_failure("an address not from a root parameter");
        assert_eq!(
            decode_failure(&bytes).as_deref(),
            Some("an address not from a root parameter")
        );
        assert_eq!(decode(&bytes), None);
        assert_eq!(decode_failure(&encode(&shader())), None);
    }

    #[test]
    fn a_damaged_failure_entry_is_no_entry() {
        let full = encode_failure("a scalar memory read as data");
        assert_eq!(
            decode_failure(&full).as_deref(),
            Some("a scalar memory read as data")
        );
        // Truncated after the marker, inside and right after the header
        // (no text at all), and after a valid UTF-8 prefix: all damaged,
        // not an entry with a shorter error, so all are retried.
        for n in [
            FAILED.len(),
            FAILED.len() + 4,
            FAILED.len() + LENGTH + CHECK,
            full.len() - 1,
        ] {
            assert_eq!(decode_failure(&full[..n]), None, "cut to {n} bytes");
        }
        // A byte changed in the text with the length left intact: damage,
        // not an entry with a different (still valid) reason.
        let mut flipped = full.clone();
        let at = FAILED.len() + LENGTH + CHECK + 1;
        assert!(flipped[at].is_ascii());
        flipped[at] = if flipped[at] == b'a' { b'b' } else { b'a' };
        assert_eq!(decode_failure(&flipped), None, "same length, changed byte");
        // Text that is not UTF-8, and a marker followed by text that is not
        // one (a shader entry's magic, say) are no entry either.
        assert_eq!(
            decode_failure(&[FAILED.as_slice(), &[0xff, 0xfe]].concat()),
            None
        );
        assert_eq!(decode_failure(b"VSH1a scalar memory read as data"), None);
        // An entry of the previous layout (no checksum) is no entry: it is
        // translated again rather than read with its text's first bytes as
        // a checksum.
        let old = [b"VSF1".as_slice(), &full[FAILED.len()..]].concat();
        assert_eq!(decode_failure(&old), None);
    }

    /// A cache whose entry cannot be written (here: the entry's path is a
    /// directory, so the rename onto it fails) must not turn a refusal into
    /// an OS error: the reason is the answer, the cache is only a shortcut.
    /// Nor a translation (before the port it did too), and the answer is
    /// the translation already made, not a second one.
    #[test]
    fn a_cache_that_cannot_be_written_still_says_why() {
        let _serial = serial();
        use xtp_pipeline::CacheKey;
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data".to_owned())
        };
        let ok = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        // Block each entry's own path with a directory before the first call.
        let block = |container: &[u8]| {
            let input = [&build[..], container].concat();
            let key = CacheKey::new(&Translate::new(fail), &input);
            std::fs::create_dir_all(cache.path_for(&key)).unwrap();
        };
        let refused = b"a container that does not translate".as_slice();
        block(refused);
        assert_eq!(
            cached(&cache, &build, refused, fail),
            Err("a scalar memory read as data".to_owned()),
            "the reason, not the OS error"
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "translated once");
        let translated = b"a container that translates".as_slice();
        block(translated);
        assert_eq!(cached(&cache, &build, translated, ok), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2, "translated once");
    }

    /// The acceptance test of the port: a refusal is kept under the key a
    /// translation would have, the build and the target. A container is
    /// translated once per pair, then its refusal comes from the cache; a
    /// rebuilt translator or another target translates it again (and may
    /// take it: a 64-lane host runs what a 32-lane one refuses).
    #[test]
    fn refusals_are_kept_by_build_and_target() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let container = b"a wave64 program that reads across its halves".as_slice();
        let (build, rebuilt) = ([7u8; 32], [8u8; 32]);
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let runs = AtomicU64::new(0);
        // Taken for 64-lane hosts, and by the rebuilt translator for any.
        let get = |build: &[u8; 32], target: Target| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                if target.runs(64) || *build == rebuilt {
                    Ok(shader())
                } else {
                    Err("it needs 64-lane waves".to_owned())
                }
            };
            cached(&cache, &key(build, target), container, translate)
        };
        let refused = Err("it needs 64-lane waves".to_owned());
        let runs = || runs.load(Ordering::Relaxed);
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(runs(), 1, "the second refusal came from the cache");

        assert_eq!(get(&build, Target::ANY), Ok(shader()), "another target");
        assert_eq!(runs(), 2);
        assert_eq!(get(&rebuilt, nvidia), Ok(shader()), "another build");
        assert_eq!(runs(), 3);

        // Each is its own entry, found again.
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(get(&build, Target::ANY), Ok(shader()));
        assert_eq!(get(&rebuilt, nvidia), Ok(shader()));
        assert_eq!(runs(), 3);
        assert_eq!(walk(dir.path()).len(), 3);
    }

    /// What `host_shader` keeps: the translator's own refusal, with its
    /// text, translated once.
    #[test]
    fn the_translators_refusal_is_kept_with_its_text() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |c: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            xtp_shader::emit::translate_for(c, Target::ANY).map_err(|e| e.to_string())
        };
        let container = b"not a shader container".as_slice();
        let refused = Err("not a DXBC container".to_owned());
        for _ in 0..2 {
            let got = cached(&cache, &key(&[7; 32], Target::ANY), container, translate);
            assert_eq!(got, refused);
        }
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second from the cache");
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    /// Translations are kept by the wave sizes they were made for: one
    /// container's wave64 program, translated for an NVIDIA host (two
    /// 32-lane waves, `WaveSize(32)`) and for any host (`WaveSize(64)`),
    /// is two entries, each found again by its own target.
    #[cfg(windows)]
    #[test]
    fn translations_are_kept_by_target() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let (c, native) = wave64_in_half_program();
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let runs = AtomicU64::new(0);
        let get = |target: Target| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                xtp_shader::emit::translate_program_for(&c, &native, target)
                    .map_err(|(_, e)| e.to_string())
            };
            let shader = cached(&cache, &key(&build, target), container, translate).unwrap();
            crate::pipeline::host::Needs::of(&shader.dxil)
                .unwrap()
                .lanes
        };
        assert_eq!(get(nvidia), (32, 32));
        assert_eq!(get(Target::ANY), (64, 64));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(get(nvidia), (32, 32), "from its own entry");
        assert_eq!(get(Target::ANY), (64, 64));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        let rdna = Target {
            lanes_min: 32,
            lanes_max: 64,
            stride_table: false,
        };
        assert_ne!(key(&build, nvidia), key(&build, rdna));
        assert_ne!(key(&build, rdna), key(&build, Target::ANY));
        assert_ne!(key(&build, nvidia), key(&[8; 32], nvidia));
        assert_eq!(key(&build, nvidia), key(&build, nvidia));
    }

    /// The key is the translator's, not the executable's: this run's build
    /// key is made from the fingerprint of `xtp-shader`'s sources, this
    /// file and the settings alone, nothing else of the build. So two
    /// builds that differ only outside the translator share their entries
    /// (`apps/xtp/tests/precook.rs` runs two executables that do), and a
    /// change to a file of the translator, to the encoding or to a setting
    /// translates again.
    #[test]
    fn the_key_is_the_translators_not_the_executables() {
        let _serial = serial();
        let settings = [
            xtp_shader::emit::loop_limit(),
            xtp_shader::emit::loop_budget(),
            xtp_shader::emit::dxil_opt(),
            u32::from(xtp_shader::emit::raw_structured()),
        ];
        let ours = build_of(xtp_shader::SOURCES, ENCODING, settings);
        assert_eq!(*build(), ours, "nothing but these parts");

        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let get = |build: [u8; 32]| cached(&cache, &key(&build, Target::ANY), container, translate);
        let runs = || runs.load(Ordering::Relaxed);
        assert_eq!(get(ours), Ok(shader()));
        // Another build of the same translator: the key is made again from
        // the same parts, and finds the entry.
        let rebuilt = build_of(xtp_shader::SOURCES, ENCODING, settings);
        assert_eq!(get(rebuilt), Ok(shader()));
        assert_eq!(runs(), 1, "shared by both builds");

        // A file of the translator changed: its fingerprint does
        // (`xtp_shader`'s fingerprint tests), and so does the key.
        let fingerprint = u128::from_str_radix(xtp_shader::SOURCES, 16).unwrap();
        let changed = format!("{:032x}", fingerprint ^ 1);
        assert_eq!(get(build_of(&changed, ENCODING, settings)), Ok(shader()));
        assert_eq!(runs(), 2, "a changed translator translates again");
        assert_ne!(
            build_of(xtp_shader::SOURCES, b"another encoding", settings),
            ours
        );
        for i in 0..settings.len() {
            let mut other = settings;
            other[i] ^= 1;
            assert_ne!(
                build_of(xtp_shader::SOURCES, ENCODING, other),
                ours,
                "setting {i}"
            );
        }
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
