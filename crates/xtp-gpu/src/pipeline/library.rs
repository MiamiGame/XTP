//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//! 3. Beside them, the whole descriptions of the graphics and mesh
//!    pipelines the title creates ([`super::record`]), from which `xtp
//!    precook` makes them into the library: those of the whole game after
//!    one session, and again after a translator change.
//!
//! Loads and stores of different pipelines run side by side: the library
//! is thread-safe and synchronises itself, but for two loads of the same
//! pipeline at once, which "may modify the state of that pipeline within
//! the library in a non-thread-safe manner" (D3D12's remarks on
//! `CreatePipelineLibrary`): one pipeline's name is used by one thread at a
//! time ([`NAMES`]). Serialising takes the library alone, so that its size
//! cannot change between asking for it and writing. With every call
//! serialised, the workers made some 35 pipelines a second, most of them
//! loads (a Series S session of 2026-11-25, `20261125-022031`: 1,465
//! loaded and 607 made, some 400 ms each on 14 workers), and its world
//! dropped some 900 draws a frame "being made" for a minute.
//!
//! - A save counts as done once its file is written: the pipelines it
//!   holds stop counting as added then, not before, so that a save that
//!   fails is made again by the next one.
//! - Generations: a library cannot drop an entry, and every translator
//!   change adds a name for each pipeline whose code changed, beside the
//!   old one. `xtp precook` therefore makes a new generation when the
//!   translator changed or the file grew past a bound ([`rebuild`]): the
//!   library starts empty, what it asks for is taken from the generation
//!   before when that has it (loaded there, stored here) and made
//!   otherwise, and the save replaces the file with the new generation
//!   alone. What only sessions asked for is made again when one asks.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The folder the pipeline cache is kept in, once set.
pub(crate) fn dir() -> Option<&'static std::path::Path> {
    DIR.get().map(PathBuf::as_path)
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// Microseconds pipelines waited for the library (their name, or a save),
/// and spent loading from it, since the start (the library's line).
static LOCK_US: AtomicU64 = AtomicU64::new(0);
static LOAD_US: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// While a new generation is made ([`rebuild`]): the one before, read
    /// only.
    previous: Option<Generation>,
}

/// A generation of the library read only, and the blob it was made from.
struct Generation {
    library: ID3D12PipelineLibrary1,
    _blob: Vec<u8>,
}

// SAFETY: D3D12 objects are free-threaded. The library synchronises its
// calls itself but for loads of one pipeline, which the name locks
// ([`NAMES`]) keep apart, and serialising is made with the library's write
// lock held, alone.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// Locks by a pipeline's name: one thread at a time loads or stores a
/// pipeline of a name (one of these locks: a few names share one).
static NAMES: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

/// The name lock of `name`.
fn name_lock(name: &str) -> &'static Mutex<()> {
    let h = name.bytes().fold(0usize, |h, b| {
        h.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    &NAMES[h % NAMES.len()]
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// The library is opened on first use: once opened, [`rebuild`] comes too
/// late.
static OPENED: AtomicBool = AtomicBool::new(false);
static REBUILD: AtomicBool = AtomicBool::new(false);

fn library() -> Option<&'static RwLock<Library>> {
    static LIBRARY: OnceLock<Option<RwLock<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            OPENED.store(true, Ordering::SeqCst);
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = if REBUILD.load(Ordering::SeqCst) {
                new_generation(&device, blob)
            } else {
                open(&device, blob)
            };
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(RwLock::new)
        })
        .as_ref()
}

/// Makes the library a new generation, the file's library its generation
/// before (see the module): asked before the library's first use (`xtp
/// precook`'s start), else too late, which it says.
pub(crate) fn rebuild() -> bool {
    REBUILD.store(true, Ordering::SeqCst);
    !OPENED.load(Ordering::SeqCst)
}

/// The library file's size, if there is one.
pub(crate) fn file_bytes() -> Option<u64> {
    let dir = DIR.get()?;
    std::fs::metadata(dir.join(LIBRARY_FILE))
        .ok()
        .map(|m| m.len())
}

/// For tests: whether the host keeps pipelines in a library (it may have
/// none).
#[cfg(test)]
pub(crate) fn available() -> bool {
    library().is_some()
}

/// The host's library of `bytes`, which must outlive it.
fn make(device: &ID3D12Device1, bytes: &[u8]) -> windows::core::Result<ID3D12PipelineLibrary1> {
    // SAFETY: the callers keep the bytes beside the library.
    unsafe {
        device
            .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
            .cast()
    }
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    if !blob.is_empty() {
        match make(device, &blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: AtomicU64::new(0),
                    previous: None,
                });
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(device, &[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: AtomicU64::new(0),
        previous: None,
    })
}

/// An empty library, the one in `blob` its generation before (none if the
/// host rejects it).
fn new_generation(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let previous = if blob.is_empty() {
        None
    } else {
        match make(device, &blob) {
            Ok(library) => Some(Generation {
                library,
                _blob: blob,
            }),
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected: the new generation starts from nothing");
                None
            }
        }
    };
    tracing::info!(
        before = previous.as_ref().map_or(0, |g| g._blob.len()),
        "host pipeline library: a new generation, from the one before"
    );
    let library = make(device, &[]).ok()?;
    Some(Library {
        library,
        _blob: Vec::new(),
        added: AtomicU64::new(0),
        previous,
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    found_in(library, name, load, create).map(|(pso, _)| pso)
}

/// Where a pipeline came from ([`found_in`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Library,
    /// The generation before (stored into the library).
    Previous,
    Made,
}

/// [`pipeline`] in `library`, and where it came from.
fn found_in(
    library: &RwLock<Library>,
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<(ID3D12PipelineState, Source)> {
    let wide = HSTRING::from(name);
    let since = |t: std::time::Instant| t.elapsed().as_micros() as u64;
    // The name's lock, then the library's (a save takes it alone).
    let take = || {
        let asked = std::time::Instant::now();
        let held = name_lock(name).lock().unwrap_or_else(|e| e.into_inner());
        let l = library.read().unwrap_or_else(|e| e.into_inner());
        LOCK_US.fetch_add(since(asked), Ordering::Relaxed);
        (held, l)
    };
    let store = |l: &Library, pso: &ID3D12PipelineState| {
        // SAFETY: a live pipeline; a name already stored (a description
        // that differs, or another thread's) is refused, which leaves the
        // library as it was.
        if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) }.is_ok() {
            l.added.fetch_add(1, Ordering::Relaxed);
        }
    };
    let loaded = {
        let (_name, l) = take();
        let started = std::time::Instant::now();
        let mut loaded = load(&l.library, PCWSTR(wide.as_ptr())).map(|pso| (pso, Source::Library));
        if loaded.is_err()
            && let Some(previous) = &l.previous
            && let Ok(pso) = load(&previous.library, PCWSTR(wide.as_ptr()))
        {
            store(&l, &pso);
            loaded = Ok((pso, Source::Previous));
        }
        LOAD_US.fetch_add(since(started), Ordering::Relaxed);
        loaded
    };
    if let Ok(found) = loaded {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(found);
    }
    // Compiled outside the locks: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let (_name, l) = take();
    store(&l, &pso);
    Ok((pso, Source::Made))
}

/// What a save did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Saved {
    /// Nothing to write (nothing added, the early list as it was), or not
    /// yet time to, or no library or cache folder.
    Nothing,
    Written,
    /// The reason; what was to be written still is.
    Failed(String),
}

/// Writes the library if pipelines were added since the last save, and the
/// early list with it or if it changed, once `SAVE_EVERY` has passed (or
/// `now`). The library is serialised only when it holds something new: it
/// takes the lock every load waits for, and the cache's library had grown
/// to 2.3 GB (2026-11-25), which a session wrote again every 20 s while it
/// drew new pipelines that the library already held.
///
/// One save at a time ([`write_library`]): a worker's gives way to one
/// under way, one asked `now` (at the end of a run or of a precook) waits
/// for it.
pub(crate) fn save(now: bool) -> Saved {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else {
        return Saved::Nothing;
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return Saved::Nothing;
        }
        *last = Some(std::time::Instant::now());
    }
    let saved = write_library(library, dir, now, |bytes| {
        // Under the save's lock: whether the list changed since the last
        // save that wrote it.
        let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
        if bytes.is_none() && !used_changed {
            return Ok(false);
        }
        let used: Vec<String> = USED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flatten()
            .cloned()
            .collect();
        if let Err(e) = write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            if used_changed {
                USED_CHANGED.store(true, Ordering::Relaxed);
            }
            return Err(e);
        }
        let (created, unkept, distinct) = super::record::created_counts();
        tracing::info!(
            library = bytes.is_some(),
            bytes = bytes.unwrap_or(0),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            created_unkept = unkept,
            created_distinct = distinct,
            recorded = super::record::kept(),
            lock_wait_ms = LOCK_US.load(Ordering::Relaxed) / 1000,
            load_ms = LOAD_US.load(Ordering::Relaxed) / 1000,
            "host pipeline library saved"
        );
        Ok(true)
    });
    if let Saved::Failed(e) = &saved {
        tracing::warn!(error = %e, "host pipeline library not saved");
    }
    saved
}

/// Writes `library` into `dir` if pipelines were added to it, then runs
/// `then` with its size (`None`: none written), which says whether it
/// wrote something too. The pipelines the library holds stop counting as
/// added once both are done, not before.
///
/// One at a time, from what it serialises to that count: two at once would
/// both write what was added before them, then both stop counting it (the
/// count would wrap), and share the temporary files. With `wait`, it waits
/// for one under way; without, it gives way to it ([`Saved::Nothing`]).
fn write_library(
    library: &RwLock<Library>,
    dir: &std::path::Path,
    wait: bool,
    then: impl FnOnce(Option<usize>) -> std::io::Result<bool>,
) -> Saved {
    static SAVING: Mutex<()> = Mutex::new(());
    let _saving = if wait {
        SAVING.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match SAVING.try_lock() {
            Ok(saving) => saving,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Saved::Nothing,
        }
    };
    let serialised = {
        // Alone: no store between the size and the bytes.
        let l = library.write().unwrap_or_else(|e| e.into_inner());
        let added = l.added.load(Ordering::Relaxed);
        if added == 0 {
            None
        } else {
            // SAFETY: a buffer of the size the library asks for.
            let bytes = unsafe {
                let mut bytes = vec![0u8; l.library.GetSerializedSize()];
                l.library.Serialize(&mut bytes).map(|()| bytes)
            };
            match bytes {
                Ok(bytes) => Some((bytes, added)),
                Err(e) => return Saved::Failed(format!("not serialised: {e}")),
            }
        }
    };
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| match &serialised {
            Some((bytes, _)) => write_atomically(&dir.join(LIBRARY_FILE), bytes),
            None => Ok(()),
        })
        .and_then(|()| then(serialised.as_ref().map(|(b, _)| b.len())));
    match (written, serialised) {
        (Err(e), _) => Saved::Failed(e.to_string()),
        (Ok(_), Some((_, added))) => {
            // Those stored meanwhile still count, for the next save.
            let l = library.read().unwrap_or_else(|e| e.into_inner());
            l.added.fetch_sub(added, Ordering::Relaxed);
            Saved::Written
        }
        (Ok(true), None) => Saved::Written,
        (Ok(false), None) => Saved::Nothing,
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. One save at a time uses it
/// ([`save`]).
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let test = "pipeline::library::tests::a_rejected_blob_gives_an_empty_library";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = gpu.device.cast::<ID3D12Device1>().expect("ID3D12Device1");
        // The layer reports the blob it cannot read; the library falls back.
        let library = crate::testgpu::expect_layer_errors(|| {
            open(&device, b"not a pipeline library".repeat(16)).expect("an empty library")
        });
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    /// Host pipelines of a few names, each asked for by several threads at
    /// once: every ask gets its pipeline (loads of one name kept apart,
    /// stores beside loads), most from the library, which still serialises
    /// after.
    #[test]
    fn pipelines_are_loaded_and_stored_side_by_side() {
        let test = "pipeline::library::tests::pipelines_are_loaded_and_stored_side_by_side";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        if library().is_none() {
            return;
        }
        let shaders: Vec<_> = [1, 2, 4, 8]
            .map(|n| {
                super::super::precook::tests::translated(
                    "compute",
                    xtp_shader::HwStage::Cs,
                    Some(&format!("[numthreads({n}, 1, 1)] void main() {{}}")),
                    windows::core::s!("cs_5_1"),
                    format!("library test: compute {n}").as_bytes(),
                )
            })
            .into();
        let loaded = LOADED.load(Ordering::Relaxed);
        std::thread::scope(|s| {
            for t in 0..8 {
                let shaders = &shaders;
                s.spawn(move || {
                    for k in 0..12 {
                        let shader = &shaders[(t + k) % shaders.len()];
                        let made = super::super::host::HostPipeline::new(
                            std::slice::from_ref(shader),
                            None,
                        );
                        assert!(made.is_some(), "a pipeline for {shader:?}");
                    }
                });
            }
        });
        // 96 asks of 4 pipelines: each compiled by at most the 8 threads.
        assert!(LOADED.load(Ordering::Relaxed) - loaded >= 96 - 4 * 8);
        let l = library()
            .unwrap()
            .write()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live library, alone.
        assert!(unsafe { l.library.GetSerializedSize() } > 0);
    }

    /// A compute kernel of `n` threads whose code holds its root signature.
    struct Kernel {
        code: Vec<u8>,
        root: ID3D12RootSignature,
    }

    fn kernel(device: &ID3D12Device, n: u32) -> Kernel {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let source = format!("[RootSignature(\"\")] [numthreads({n}, 1, 1)] void main() {{}}");
        let mut blob = None;
        // SAFETY: the source lives through compilation and the blob owns
        // its bytes.
        let code = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::PCSTR::null(),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: the code holds the root signature.
        let root = unsafe { device.CreateRootSignature(0, &code) }.unwrap();
        Kernel { code, root }
    }

    /// `k`'s pipeline named `name` in `library`, and where it came from.
    fn ask(library: &RwLock<Library>, device: &ID3D12Device, k: &Kernel, name: &str) -> Source {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a copy of a live root signature, not released by the
            // description.
            pRootSignature: unsafe { std::mem::transmute_copy(&k.root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: k.code.as_ptr().cast(),
                BytecodeLength: k.code.len(),
            },
            ..Default::default()
        };
        found_in(
            library,
            name,
            // SAFETY: the description points at live data.
            |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { device.CreateComputePipelineState(&desc) },
        )
        .unwrap()
        .1
    }

    /// A save that fails leaves the library's pipelines to save: the next
    /// save writes them, and only then do they stop counting as added.
    #[test]
    fn a_failed_save_is_made_again() {
        let test = "pipeline::library::tests::a_failed_save_is_made_again";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Vec::new()).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        // A file where the folder should be.
        let not_a_folder = dir.path().join("pipelines");
        std::fs::write(&not_a_folder, b"in the way").unwrap();
        assert!(matches!(
            write_library(&library, &not_a_folder, true, |_| Ok(true)),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        // The early list not written: neither is the save.
        let good = dir.path().join("cache");
        assert!(matches!(
            write_library(&library, &good, true, |_| Err(std::io::Error::other(
                "full"
            ))),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        let mut size = None;
        assert_eq!(
            write_library(&library, &good, true, |bytes| {
                size = bytes;
                Ok(true)
            }),
            Saved::Written
        );
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
        let written = std::fs::metadata(good.join(LIBRARY_FILE)).unwrap().len();
        assert_eq!(Some(written as usize), size);
        assert_eq!(
            write_library(&library, &good, true, |_| Ok(false)),
            Saved::Nothing,
            "nothing new"
        );
    }

    /// Two saves at once: one asked to wait writes after the one under way,
    /// which counts what it wrote first, so that what was added stops
    /// counting once and the count does not wrap; one that does not wait
    /// gives way.
    #[test]
    fn saves_at_once_count_what_they_wrote_once() {
        let test = "pipeline::library::tests::saves_at_once_count_what_they_wrote_once";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Vec::new()).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        let (inside, go) = (std::sync::Barrier::new(2), std::sync::Barrier::new(2));
        std::thread::scope(|s| {
            let first = s.spawn(|| {
                write_library(&library, dir.path(), true, |_| {
                    inside.wait();
                    go.wait();
                    Ok(true)
                })
            });
            inside.wait();
            // The first is between its write and its count.
            assert_eq!(
                write_library(&library, dir.path(), false, |_| Ok(true)),
                Saved::Nothing,
                "gives way"
            );
            let second = s.spawn(|| write_library(&library, dir.path(), true, |_| Ok(false)));
            go.wait();
            assert_eq!(first.join().unwrap(), Saved::Written);
            assert_eq!(
                second.join().unwrap(),
                Saved::Nothing,
                "nothing new by then"
            );
        });
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
    }

    /// A new generation takes what is asked for from the one before, and
    /// keeps only that: what was not asked for is made again later.
    #[test]
    fn a_new_generation_keeps_what_is_asked_for() {
        let test = "pipeline::library::tests::a_new_generation_keeps_what_is_asked_for";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let dir = tempfile::tempdir().unwrap();
        let saved = |library: &RwLock<Library>| {
            assert_eq!(
                write_library(library, dir.path(), true, |_| Ok(true)),
                Saved::Written
            );
            std::fs::read(dir.path().join(LIBRARY_FILE)).unwrap()
        };
        let old = RwLock::new(open(&device, Vec::new()).unwrap());
        assert_eq!(ask(&old, &host.device, &a, "a"), Source::Made);
        assert_eq!(ask(&old, &host.device, &b, "b"), Source::Made);
        let new = RwLock::new(new_generation(&device, saved(&old)).unwrap());
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Previous);
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Library);
        let next = RwLock::new(open(&device, saved(&new)).unwrap());
        assert_eq!(ask(&next, &host.device, &a, "a"), Source::Library);
        assert_eq!(
            ask(&next, &host.device, &b, "b"),
            Source::Made,
            "not asked for"
        );
    }
}
