//! `xtp precook`: every shader container of the title's pack translated
//! into the shader cache the title reads, and every compute program's host
//! pipeline made into the pipeline library, before a session, so that the
//! first session after a translator change does not wait for them in the
//! world. Each container is made ready as the title's first use of it
//! makes it, through the same calls (`xtp_gpu::pipeline::precook`). Then
//! the graphics and mesh pipelines earlier sessions made, which also need
//! the title's graphics state: sessions keep their descriptions beside the
//! pipeline library (`recorded-pipelines.bin`), and the precook makes each
//! one again into the library, as the session did.
//!
//! - The pack: as `xtp shader-survey` takes it (the path given, else
//!   `XTP_SAMPLE_SHADERS`, else `shaders.rpf` in the game folder). A
//!   container found twice is done once.
//! - The machine: the precook takes the cores it is given (`--jobs`, all by
//!   default) at below-normal priority, and never runs beside a game
//!   session. It waits while the session lock (`--lock`, else
//!   `XTP_SESSION_LOCK`) is held by a live process or another `xtp.exe`
//!   runs, then holds the lock itself, so that a session waits for it in
//!   turn. `cargo xtask play` runs it under the lock its session holds
//!   (`--lock-holder`), before the title. An `xtp.exe` that starts anyway
//!   pauses it until it ends.
//! - Up to date: after a complete pass, a note beside the cache
//!   (`shaders/precook.txt`) says for which translation key (the
//!   translator, its settings and the host's wave sizes), pack and whether
//!   with pipelines, and with pipelines what they came from: the file of
//!   kept descriptions (its BLAKE3) and the library file (its size). A
//!   precook that finds its own note stops there, in seconds (the pack and
//!   the kept descriptions are read and hashed, the host device made); one
//!   whose note differs only by the kept descriptions makes those alone;
//!   one whose library file is gone or smaller than noted passes again,
//!   as does `--force`. The host rejects the pipeline library after a
//!   driver update, which the note does not see: `--force` then.
//! - Generations: after a translator change (the note's key is another),
//!   or once the library file passes `LIBRARY_REBUILD_BYTES` (since the
//!   pass the note is of, or to twice its size then), the pass makes the
//!   library a new generation, which holds what it makes and nothing of the
//!   translators before (`xtp_gpu::pipeline::precook`). This is looked at
//!   before the note: a library that grew past its bound makes a whole
//!   pass even when the note still holds.
//! - The note says the kept descriptions that were made again: the file as
//!   the precook read it, not as it is at the end, which a session may
//!   have added to meanwhile (the next precook makes those).
//! - A pass whose library is not saved, or whose kept descriptions are not
//!   read, writes no note and fails (its exit code says so).
//! - The kept pipelines take at most `XTP_PRECOOK_SECONDS` (default
//!   [`REPLAY_SECONDS`]): those not started by then are left to the
//!   session, and the note is not written, so the next precook goes on
//!   (what this one made, it finds in the library). A pipeline whose
//!   shaders no longer translate is refused at once, as in a session.
//! - `--no-wait` starts at once, without the lock (tests, or when you know
//!   the machine is free).
//! - The pack's effects (`xtp_gpu::pipeline::effects`) name the shaders of
//!   each of their passes: the report says how many of the shader sets of
//!   those that draw the kept descriptions have, and how many sets they
//!   have that no pass names. A session keeps every pipeline the title
//!   creates, so after one that loaded the world this says whether the
//!   title creates them all while it loads.
#![cfg_attr(not(windows), allow(dead_code))]

use std::borrow::Cow;
use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::shader::{MAX_FILE, Source, sources};

/// What `xtp precook` was asked.
pub struct Options {
    pub pack: Option<PathBuf>,
    pub jobs: Option<usize>,
    /// Compute host pipelines too.
    pub pipelines: bool,
    /// Pass again even when the note says the cache is up to date.
    pub force: bool,
    /// Where to write one line per container.
    pub list: Option<PathBuf>,
    /// The session lock, else `XTP_SESSION_LOCK`.
    pub lock: Option<PathBuf>,
    /// The process holding the session lock for this precook.
    pub lock_holder: Option<u32>,
    /// Wait for the machine to be free (no session, no other `xtp.exe`).
    pub wait: bool,
}

/// The note's name, in the shader cache's folder.
const NOTE: &str = "precook.txt";
/// The note's first line; another version of the note is no note. Since
/// version 2 a pass with pipelines makes the kept graphics and mesh ones
/// too; since version 3 the note says what they came from.
const NOTE_FORMAT: &str = "xtp precook 3";
/// The time the kept graphics and mesh pipelines may take by default, in
/// seconds (`XTP_PRECOOK_SECONDS`).
const REPLAY_SECONDS: u64 = 900;
/// How often a waiting precook looks again, and a running one looks for a
/// title that started anyway.
const LOOK_EVERY: Duration = Duration::from_secs(5);
/// How often a running precook says how far it is.
const PROGRESS_EVERY: Duration = Duration::from_secs(30);

pub fn run(o: Options) -> Result<()> {
    #[cfg(not(windows))]
    {
        let _ = o;
        anyhow::bail!("xtp precook needs Windows (a host D3D12 device)");
    }
    #[cfg(windows)]
    precook(o)
}

#[cfg(windows)]
fn precook(o: Options) -> Result<()> {
    use xtp_gpu::pipeline::precook as gpu;

    below_normal();
    let path = crate::survey::pack_path(o.pack.clone())?;
    if let Some(out) = &o.list {
        // Names of the title's programs stay out of the repository.
        xtp_core::ensure_outside_repo(out)?;
    }
    let cfg = xtp_core::Config::load()?;
    let shaders = cfg.cache_subdir("shaders")?;
    let lock = o
        .lock
        .clone()
        .or_else(|| std::env::var_os("XTP_SESSION_LOCK").map(PathBuf::from))
        .filter(|p| !p.as_os_str().is_empty());
    let machine = Machine {
        lock,
        holder: o.lock_holder,
        me: std::process::id(),
    };
    let _held = if o.wait { machine.take()? } else { None };

    let started = Instant::now();
    let size = std::fs::metadata(&path)
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    anyhow::ensure!(size <= MAX_FILE, "{} is larger than 2 GiB", path.display());
    let data = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let pack = blake3::hash(&data);
    let read = started.elapsed().as_secs_f64();
    let ready = gpu::start().map_err(|e| anyhow::anyhow!("no host D3D12 device: {e}"))?;
    let device = started.elapsed().as_secs_f64() - read;
    println!(
        "precook      {}: {} bytes, blake3 {pack} (read in {read:.1} s)",
        path.display(),
        data.len()
    );
    println!(
        "host         waves of {} to {} lanes; translations kept under {} (device in {device:.1} s)",
        ready.lanes.0,
        ready.lanes.1,
        &ready.key[..16]
    );
    // The note first: opening the cache reads every entry's header (25,889
    // after a pass over the pack), which a precook that is up to date does
    // not need. Its check stays a matter of seconds whatever the cache
    // holds (2026-11-25: one that opened the cache first took 85 s on the
    // PC, the same check right after a pass 1 s).
    // The pipeline cache beside the shader cache, where `use_shader_cache`
    // keeps it (set here first: the note reads it before the shader cache
    // is opened).
    xtp_gpu::pipeline::use_pipeline_cache(shaders.with_file_name("pipelines"));
    let note_path = shaders.join(NOTE);
    let cache = |pipelines: bool| -> Result<Option<Cache>> {
        pipelines
            .then(gpu::cache)
            .transpose()
            .map(|c| {
                c.map(|c| Cache {
                    recorded: c.recorded,
                    library: c.library,
                })
            })
            .map_err(anyhow::Error::msg)
    };
    let note = Note {
        key: ready.key.clone(),
        pack: pack.to_hex().to_string(),
        pipelines: cache(o.pipelines)?,
    };
    let found = std::fs::read_to_string(&note_path)
        .ok()
        .and_then(|text| Note::parse(&text));
    // Whether the library is made a new generation comes first: a library
    // that grew past its bound while the note still holds is one too.
    let generation = note.new_generation(found.as_ref(), gpu::LIBRARY_REBUILD_BYTES);
    let plan = if o.force || generation.is_some() {
        // A new generation holds what this pass makes: all of it.
        Plan::Whole
    } else {
        note.plan(found.as_ref())
    };
    if plan == Plan::Nothing {
        println!(
            "up to date   for this translator, its settings, this pack and the kept pipelines ({:.1} s; --force passes again)",
            started.elapsed().as_secs_f64()
        );
        return Ok(());
    }
    if let Some(why) = generation {
        anyhow::ensure!(
            gpu::rebuild_library(),
            "the pipeline library was opened before its new generation"
        );
        println!(
            "library      a new generation ({why}; {} bytes before)",
            note.pipelines.as_ref().map_or(0, |c| c.library)
        );
    }
    let opening = Instant::now();
    xtp_hle::modules::d3d12x::use_shader_cache(shaders.clone());
    println!(
        "cache        {} (opened in {:.1} s)",
        shaders.display(),
        opening.elapsed().as_secs_f64()
    );

    let threads = o
        .jobs
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .max(1);
    let pass = match plan {
        Plan::Whole => pass(
            &data,
            o.pipelines,
            threads,
            o.wait.then_some(&machine),
            |c, s, p| gpu::container(c, s, p).into(),
        ),
        // The containers are as the note says: the kept pipelines alone.
        Plan::Nothing | Plan::Kept => Pass { items: Vec::new() },
    };
    let counts = gpu::counts();
    let replay = o.pipelines.then(|| {
        replay(
            threads,
            o.wait.then_some(&machine),
            replay_bound(),
            &data,
            derive_wanted(),
        )
    });
    let saved = gpu::finish();
    let seconds = started.elapsed().as_secs_f64();
    print!(
        "{}",
        report(
            &pass,
            plan,
            &counts,
            replay.as_ref(),
            o.pipelines,
            threads,
            seconds,
            gpu::refusals()
        )
    );
    if let Some(out) = &o.list {
        std::fs::write(out, list(&pass)).with_context(|| format!("writing {}", out.display()))?;
        println!("list         {}", out.display());
    }
    // The note only after a whole pass: refusals are answers kept in the
    // cache too, so a pass with some is whole; one whose kept pipelines
    // ran out of time is not, nor one whose kept descriptions were not read
    // or whose library was not saved, which fail. It says the descriptions
    // that were made again: the file as the replay read it (after its
    // rewrite), not as it is now, which a session may have added to
    // meanwhile; and the library file as it is after its save.
    match (&saved, replay.as_ref().and_then(|r| r.error.as_ref())) {
        (Err(e), _) => {
            println!("note         not written: the pipeline library was not saved");
            anyhow::bail!("the pipeline library was not saved: {e}");
        }
        (Ok(()), Some(e)) => {
            println!("note         not written: the kept pipelines were not read");
            anyhow::bail!("the kept pipelines were not read: {e}");
        }
        (Ok(()), None) if replay.as_ref().is_some_and(|r| r.left > 0) => {
            println!("note         not written: the kept pipelines are not all made");
        }
        (Ok(()), None) => {
            let pipelines = match (cache(o.pipelines)?, &replay) {
                (Some(now), Some(r)) => Some(Cache {
                    recorded: r.digest.clone(),
                    library: now.library,
                }),
                (now, _) => now,
            };
            let done = Note { pipelines, ..note };
            write_atomically(&note_path, done.text().as_bytes())
                .with_context(|| format!("writing {}", note_path.display()))?;
        }
    }
    Ok(())
}

/// Below-normal priority for this process and the threads it starts: the
/// precook gives way to anything interactive.
#[cfg(windows)]
fn below_normal() {
    use windows_sys::Win32::System::Threading::{
        BELOW_NORMAL_PRIORITY_CLASS, GetCurrentProcess, SetPriorityClass,
    };
    // SAFETY: changes our own process's priority class only.
    if unsafe { SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) } == 0 {
        eprintln!(
            "precook: priority not lowered ({})",
            std::io::Error::last_os_error()
        );
    }
}

/// What the note says: a complete pass was made for this key and pack,
/// with pipelines from this cache (the kept descriptions and the library
/// as they were after it), or without.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Note {
    key: String,
    pack: String,
    pipelines: Option<Cache>,
}

/// What the pipelines came from (`xtp_gpu::pipeline::precook::Cache`):
/// the BLAKE3 of the file of kept descriptions (`none` without one) and
/// the library file's size.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Cache {
    recorded: String,
    library: u64,
}

/// What a precook has to do, by its note ([`Note::plan`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plan {
    Nothing,
    /// The kept descriptions changed: those alone.
    Kept,
    Whole,
}

impl Note {
    fn text(&self) -> String {
        let mut o = format!("{NOTE_FORMAT}\nkey {}\npack {}\n", self.key, self.pack);
        match &self.pipelines {
            Some(c) => {
                let _ = write!(
                    o,
                    "pipelines yes\nrecorded {}\nlibrary {}\n",
                    c.recorded, c.library
                );
            }
            None => o.push_str("pipelines no\n"),
        }
        o
    }

    /// The note a file reading `text` holds; `None` if it is none (another
    /// version, damage).
    fn parse(text: &str) -> Option<Note> {
        let mut lines = text.lines();
        (lines.next()? == NOTE_FORMAT).then_some(())?;
        let mut field = |name: &str| {
            lines
                .next()?
                .strip_prefix(name)?
                .strip_prefix(' ')
                .map(str::to_owned)
        };
        let key = field("key")?;
        let pack = field("pack")?;
        let pipelines = match field("pipelines")?.as_str() {
            "no" => None,
            "yes" => Some(Cache {
                recorded: field("recorded")?,
                library: field("library")?.parse().ok()?,
            }),
            _ => return None,
        };
        let note = Note {
            key,
            pack,
            pipelines,
        };
        (note.text() == text).then_some(note)
    }

    /// Why this pass makes the pipeline library a new generation, if it does
    /// (with pipelines only): another translator than the one `found`
    /// notes, or a library file past `bound` that grew past it since that
    /// pass, or to twice its size then. A library already past the bound
    /// after a new generation holds the title's pipelines of this
    /// translator: making it again would not shrink it.
    fn new_generation(&self, found: Option<&Note>, bound: u64) -> Option<&'static str> {
        let now = self.pipelines.as_ref()?.library;
        if found.is_some_and(|f| f.key != self.key) {
            return Some("another translator");
        }
        let then = found.and_then(|f| f.pipelines.as_ref()).map(|c| c.library);
        (now > bound && then.is_none_or(|then| then <= bound || now >= then.saturating_mul(2)))
            .then_some("past its bound")
    }

    /// What this pass has to do after the one `found` notes: nothing if it
    /// was for the same key and pack, with pipelines if this one makes
    /// them, from the same kept descriptions and a library file at least
    /// as large (a session adds to it); the kept descriptions alone if only
    /// they differ; else everything.
    fn plan(&self, found: Option<&Note>) -> Plan {
        let Some(found) = found else {
            return Plan::Whole;
        };
        if found.key != self.key || found.pack != self.pack {
            return Plan::Whole;
        }
        match (&self.pipelines, &found.pipelines) {
            (None, _) => Plan::Nothing,
            (Some(_), None) => Plan::Whole,
            (Some(now), Some(then)) if now.library < then.library => Plan::Whole,
            (Some(now), Some(then)) if now.recorded != then.recorded => Plan::Kept,
            (Some(_), Some(_)) => Plan::Nothing,
        }
    }
}

/// What became of one container of the pack.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Done {
    /// The record does not unpack.
    Corrupt,
    /// The same container as an earlier record's.
    Repeated,
    /// Precooked: its stage, and the outcome.
    Cooked(Option<xtp_shader::HwStage>, Outcome),
    /// The translator panicked on it (a bug: counted, not fatal).
    Panicked(String),
}

/// A container's outcome, as `xtp_gpu::pipeline::precook` says it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Ready,
    Refused(String),
    NoPipeline,
}

#[cfg(windows)]
impl From<xtp_gpu::pipeline::precook::Outcome> for Outcome {
    fn from(o: xtp_gpu::pipeline::precook::Outcome) -> Outcome {
        use xtp_gpu::pipeline::precook::Outcome as Gpu;
        match o {
            Gpu::Ready => Outcome::Ready,
            Gpu::Refused(reason) => Outcome::Refused(reason),
            Gpu::NoPipeline => Outcome::NoPipeline,
        }
    }
}

/// One record of the pack and what became of its container.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Item {
    offset: usize,
    name: Option<String>,
    /// The container's BLAKE3 in hexadecimal (its first 16 digits are the
    /// `hash=` of the pipeline log); empty for a corrupt record.
    hash: String,
    done: Done,
}

/// A pass over a pack.
struct Pass {
    items: Vec<Item>,
}

/// The container bytes of `source`, as the title passes them: a container
/// stored as is ends where its header says, not at the end of the file.
fn container_bytes<'a>(source: &'a Source<'_>) -> Option<Cow<'a, [u8]>> {
    match source {
        Source::Raw { data, .. } => {
            let size = data.get(24..28)?;
            let size = u32::from_le_bytes(size.try_into().ok()?) as usize;
            Some(Cow::Borrowed(data.get(..size)?))
        }
        Source::Packed(_) => source.bytes(),
    }
}

/// Precooks every container of the pack in `data` on `threads` threads with
/// `cook` (container, stage, pipelines), pausing while `machine` says a
/// title runs.
fn pass<C>(data: &[u8], pipelines: bool, threads: usize, machine: Option<&Machine>, cook: C) -> Pass
where
    C: Fn(&[u8], Option<xtp_shader::HwStage>, bool) -> Outcome + Sync,
{
    let sources = sources(data);
    let seen = Mutex::new(std::collections::HashSet::new());
    let mut items = parallel(sources.len(), threads, machine, "records", None, |i| {
        item(&sources[i], &seen, pipelines, &cook)
    });
    // In the pack's order (`sources`'s), whichever thread took each.
    items.sort_by_key(|&(i, _)| i);
    let mut items: Vec<Item> = items.into_iter().map(|(_, item)| item).collect();
    first_takes_the_outcome(&mut items);
    Pass { items }
}

/// Runs `work` on each of `0..total` on `threads` threads, and gives what
/// each gave, by index, in no order. Says how far it is ("precook: <n> of
/// <total> <what>") every [`PROGRESS_EVERY`], pauses while `machine` says
/// a title runs, and starts nothing after `until`: the indices not run are
/// not in what it gives.
fn parallel<T, W>(
    total: usize,
    threads: usize,
    machine: Option<&Machine>,
    what: &str,
    until: Option<Instant>,
    work: W,
) -> Vec<(usize, T)>
where
    T: Send,
    W: Fn(usize) -> T + Sync,
{
    let next = AtomicUsize::new(0);
    let finished = AtomicUsize::new(0);
    let paused = AtomicBool::new(false);
    let done = AtomicBool::new(false);
    let started = Instant::now();
    std::thread::scope(|s| {
        // Says how far it is, and pauses the workers while a title runs.
        s.spawn(|| {
            let mut said = Instant::now();
            let mut looked = Instant::now() - LOOK_EVERY;
            while !done.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(200));
                if let Some(machine) = machine
                    && looked.elapsed() >= LOOK_EVERY
                {
                    looked = Instant::now();
                    let why = machine.busy();
                    if why.is_some() != paused.load(Ordering::Relaxed) {
                        match &why {
                            Some(why) => println!("precook: paused: {why}"),
                            None => println!("precook: going on"),
                        }
                        paused.store(why.is_some(), Ordering::Relaxed);
                    }
                }
                if said.elapsed() >= PROGRESS_EVERY {
                    said = Instant::now();
                    let n = finished.load(Ordering::Relaxed);
                    println!(
                        "precook: {n} of {total} {what} ({:.0} %), {:.0} s",
                        n as f64 * 100.0 / total.max(1) as f64,
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        });
        let workers: Vec<_> = (0..threads.clamp(1, total.max(1)))
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        while paused.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_millis(500));
                        }
                        if until.is_some_and(|t| Instant::now() >= t) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= total {
                            break;
                        }
                        out.push((i, work(i)));
                        finished.fetch_add(1, Ordering::Relaxed);
                    }
                    out
                })
            })
            .collect();
        let items = workers
            .into_iter()
            .flat_map(|w| w.join().expect("precook thread"))
            .collect();
        done.store(true, Ordering::Relaxed);
        items
    })
}

/// The time the kept pipelines may take: `XTP_PRECOOK_SECONDS`, else
/// [`REPLAY_SECONDS`].
fn replay_bound() -> Duration {
    let seconds = match std::env::var("XTP_PRECOOK_SECONDS") {
        Err(_) => REPLAY_SECONDS,
        Ok(text) => text.trim().parse().unwrap_or_else(|_| {
            eprintln!("precook: XTP_PRECOOK_SECONDS expects seconds, not {text:?}");
            REPLAY_SECONDS
        }),
    };
    Duration::from_secs(seconds)
}

/// What became of the graphics and mesh pipelines sessions kept.
#[derive(Debug, Default, Clone, PartialEq)]
struct Replay {
    /// The kept pipelines to make (each once), and of them those a session
    /// drew with (made first).
    kept: usize,
    drawn: usize,
    ready: usize,
    untranslated: usize,
    refused: usize,
    /// Translator bugs (counted, not fatal).
    panicked: usize,
    /// Not started: the time bound ran out first.
    left: usize,
    /// Kept but not made again: entries that do not read back, pipelines
    /// whose containers are not kept, repeats.
    damaged: usize,
    missing: usize,
    repeated: usize,
    /// Host pipelines made, and found in the library, meanwhile.
    made: u64,
    loaded: u64,
    seconds: f64,
    bound: Duration,
    /// Where they are kept, and why they could not be read.
    path: Option<PathBuf>,
    error: Option<String>,
    /// The file was rewritten past its bound: its size before, and the
    /// descriptions that went.
    compacted: Option<(u64, usize)>,
    /// The file was of an earlier version, started again: its size.
    earlier: Option<u64>,
    /// The file is past its bound but another process has it open.
    not_compacted: bool,
    /// The BLAKE3 of the file as read (the descriptions made), for the
    /// note.
    digest: String,
    /// After the kept pipelines, those derived from the pack's effects
    /// (`xtp_gpu::pipeline::precook::derive`), and of them those ready.
    derived: usize,
    derived_ready: usize,
    /// What the pack's effects say of the kept pipelines ([`Coverage`]);
    /// why not, if they were not read.
    coverage: Option<Result<Coverage, String>>,
}

/// What the pack's effects say against the kept descriptions, and what was
/// derived from them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Coverage {
    effects: usize,
    passes: usize,
    /// Passes that name programs not in the pack.
    outside: usize,
    /// The passes that draw, and their distinct shader sets.
    draws: usize,
    sets: usize,
    /// Of those sets, how many the kept descriptions have; and how many
    /// sets they have that no pass names.
    covered: usize,
    other: usize,
    /// The sets without a kept description that got derived ones, and how
    /// many.
    derived_sets: usize,
    derived: usize,
}

/// The pack's effects (`xtp_gpu::pipeline::effects`), and its records,
/// whose containers they name.
fn pack_effects(
    data: &[u8],
) -> Result<
    (
        xtp_gpu::pipeline::effects::Effects,
        Vec<xtp_game::sgd5::Record<'_>>,
    ),
    String,
> {
    use xtp_gpu::pipeline::effects;
    let records: Vec<xtp_game::sgd5::Record<'_>> = xtp_game::sgd5::records(data).collect();
    let found: Vec<effects::Record> = records
        .iter()
        .map(|r| effects::Record {
            marker: r.offset,
            name: r.name.map(str::len),
        })
        .collect();
    Ok((effects::read(data, &found)?, records))
}

/// Whether descriptions are derived from the pack's effects for the shader
/// sets no kept description has (`XTP_PRECOOK_DERIVE`, unless `0`).
fn derive_wanted() -> bool {
    std::env::var("XTP_PRECOOK_DERIVE").map_or(true, |v| v.trim() != "0")
}

impl Replay {
    /// Counts the `outcomes` of the pipelines made (each at most once, by
    /// its index: the kept ones, then the derived ones); the others were
    /// left.
    fn count(&mut self, outcomes: &[(usize, Replayed)]) {
        for &(i, o) in outcomes {
            if i >= self.kept {
                self.derived_ready += usize::from(o == Replayed::Ready);
                continue;
            }
            match o {
                Replayed::Ready => self.ready += 1,
                Replayed::Untranslated => self.untranslated += 1,
                Replayed::Refused => self.refused += 1,
                Replayed::Panicked => self.panicked += 1,
            }
        }
        self.left = (self.kept + self.derived).saturating_sub(outcomes.len());
    }
}

/// A kept pipeline's outcome, as `xtp_gpu::pipeline::precook` says it, or
/// the translator's panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Replayed {
    Ready,
    Untranslated,
    Refused,
    Panicked,
}

#[cfg(windows)]
impl From<xtp_gpu::pipeline::precook::Replayed> for Replayed {
    fn from(o: xtp_gpu::pipeline::precook::Replayed) -> Replayed {
        use xtp_gpu::pipeline::precook::Replayed as Gpu;
        match o {
            Gpu::Ready => Replayed::Ready,
            Gpu::Untranslated => Replayed::Untranslated,
            Gpu::Refused => Replayed::Refused,
        }
    }
}

/// Makes the graphics and mesh pipelines sessions kept into the library,
/// then (with `derive`) those derived from the effects of the pack in
/// `data` for the shader sets no kept one has, on `threads` threads, for
/// `bound` at most, pausing while `machine` says a title runs.
#[cfg(windows)]
fn replay(
    threads: usize,
    machine: Option<&Machine>,
    bound: Duration,
    data: &[u8],
    derive: bool,
) -> Replay {
    use xtp_gpu::pipeline::precook as gpu;

    let started = Instant::now();
    let before = gpu::counts();
    let mut out = Replay {
        bound,
        ..Replay::default()
    };
    let mut recorded = match gpu::recorded() {
        Ok(recorded) => recorded,
        Err(e) => {
            out.error = Some(e);
            return out;
        }
    };
    out.coverage = Some(pack_effects(data).map(|(effects, records)| {
        let d = gpu::derive(&mut recorded, &effects, derive, |i| {
            records.get(i)?.unpack()
        });
        Coverage {
            effects: effects.effects.len(),
            passes: effects.passes.len(),
            outside: effects.passes.iter().filter(|p| p.outside).count(),
            draws: effects.passes.iter().filter(|p| p.draws()).count(),
            sets: d.sets,
            covered: d.covered,
            other: d.other,
            derived_sets: d.derived_sets,
            derived: d.descriptions,
        }
    }));
    out.derived = recorded.derived;
    out.kept = recorded.len() - recorded.derived;
    out.damaged = recorded.damaged;
    out.missing = recorded.missing;
    out.repeated = recorded.repeated;
    out.drawn = recorded.drawn;
    out.path = recorded.path.clone();
    out.compacted = recorded.compacted;
    out.earlier = recorded.earlier;
    out.not_compacted = recorded.not_compacted;
    out.digest = recorded.digest.clone();
    if !recorded.is_empty() {
        println!(
            "precook: {} graphics and mesh pipelines kept by sessions and {} derived ({} s at most)",
            out.kept,
            out.derived,
            bound.as_secs()
        );
    }
    let outcomes: Vec<(usize, Replayed)> = parallel(
        recorded.len(),
        threads,
        machine,
        "kept pipelines",
        Some(started + bound),
        |i| match catch_unwind(AssertUnwindSafe(|| gpu::replay(&recorded, i))) {
            Ok(o) => o.into(),
            Err(_) => Replayed::Panicked,
        },
    );
    out.count(&outcomes);
    let after = gpu::counts();
    out.made = after.made - before.made;
    out.loaded = after.loaded - before.loaded;
    out.seconds = started.elapsed().as_secs_f64();
    out
}

/// Gives a repeated container's outcome to its first record in pack order:
/// the thread that cooked it may have taken a later one.
fn first_takes_the_outcome(items: &mut [Item]) {
    let mut first = std::collections::HashMap::new();
    for k in 0..items.len() {
        if items[k].hash.is_empty() {
            continue;
        }
        match first.get(&items[k].hash) {
            None => {
                first.insert(items[k].hash.clone(), k);
            }
            Some(&i) if items[k].done != Done::Repeated => {
                items[i].done = std::mem::replace(&mut items[k].done, Done::Repeated);
            }
            Some(_) => {}
        }
    }
}

/// Precooks one record's container, once per distinct container.
fn item<C>(
    source: &Source<'_>,
    seen: &Mutex<std::collections::HashSet<[u8; 32]>>,
    pipelines: bool,
    cook: &C,
) -> Item
where
    C: Fn(&[u8], Option<xtp_shader::HwStage>, bool) -> Outcome,
{
    let (offset, name) = match source {
        Source::Raw { offset, .. } => (*offset, None),
        Source::Packed(r) => (r.offset, r.name.map(str::to_owned)),
    };
    let Some(bytes) = container_bytes(source) else {
        return Item {
            offset,
            name,
            hash: String::new(),
            done: Done::Corrupt,
        };
    };
    let full = blake3::hash(&bytes);
    let hash = full.to_hex().to_string();
    let first = seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(*full.as_bytes());
    if !first {
        return Item {
            offset,
            name,
            hash,
            done: Done::Repeated,
        };
    }
    let stage = stage_of(&bytes);
    // A panic is a translator bug: counted, not fatal.
    let done = match catch_unwind(AssertUnwindSafe(|| cook(&bytes, stage, pipelines))) {
        Ok(outcome) => Done::Cooked(stage, outcome),
        Err(panic) => Done::Panicked(
            panic
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default(),
        ),
    };
    Item {
        offset,
        name,
        hash,
        done,
    }
}

/// The stage of the program a translation of `container` takes.
fn stage_of(container: &[u8]) -> Option<xtp_shader::HwStage> {
    #[cfg(windows)]
    {
        xtp_gpu::pipeline::precook::stage(container)
    }
    #[cfg(not(windows))]
    {
        let _ = container;
        None
    }
}

/// The pass's counts.
#[derive(Debug, Default, PartialEq, Eq)]
struct Totals {
    records: usize,
    containers: usize,
    repeated: usize,
    corrupt: usize,
    ready: usize,
    refused: usize,
    compute: usize,
    no_pipeline: usize,
    panicked: usize,
    /// Refusal reasons, by how many containers each stops.
    reasons: std::collections::BTreeMap<String, usize>,
}

impl Totals {
    fn of(pass: &Pass) -> Totals {
        let mut t = Totals {
            records: pass.items.len(),
            ..Totals::default()
        };
        for item in &pass.items {
            match &item.done {
                Done::Corrupt => t.corrupt += 1,
                Done::Repeated => t.repeated += 1,
                Done::Panicked(_) => {
                    t.containers += 1;
                    t.panicked += 1;
                }
                Done::Cooked(stage, outcome) => {
                    t.containers += 1;
                    t.compute += usize::from(*stage == Some(xtp_shader::HwStage::Cs));
                    match outcome {
                        Outcome::Ready => t.ready += 1,
                        Outcome::Refused(reason) => {
                            t.refused += 1;
                            *t.reasons.entry(reason.clone()).or_default() += 1;
                        }
                        Outcome::NoPipeline => t.no_pipeline += 1,
                    }
                }
            }
        }
        t
    }
}

/// The text report: what was translated and found, refused and made, and
/// the time.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn report(
    pass: &Pass,
    plan: Plan,
    counts: &xtp_gpu::pipeline::precook::Counts,
    replay: Option<&Replay>,
    pipelines: bool,
    threads: usize,
    seconds: f64,
    refusals: Option<String>,
) -> String {
    let t = Totals::of(pass);
    let mut o = String::new();
    if plan != Plan::Whole {
        let _ = writeln!(
            o,
            "containers   as the note says (this translator and pack): only the kept pipelines changed"
        );
        if let Some(r) = replay {
            o.push_str(&replay_lines(r));
        }
        let _ = writeln!(
            o,
            "time         {seconds:.1} s on {threads} threads at below-normal priority"
        );
        return o;
    }
    let _ = writeln!(
        o,
        "containers   {} in {} records ({} repeated{})",
        t.containers,
        t.records,
        t.repeated,
        if t.corrupt > 0 {
            format!(", {} corrupt", t.corrupt)
        } else {
            String::new()
        }
    );
    let _ = writeln!(o, "translated   {}", counts.translated);
    let _ = writeln!(o, "from cache   {}", counts.cached);
    let _ = writeln!(
        o,
        "refused      {} (kept in the cache: the title gets the same answer at once)",
        t.refused
    );
    let mut reasons: Vec<_> = t.reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (reason, n) in reasons.iter().take(8) {
        let _ = writeln!(o, "  {n:>7}  {reason}");
    }
    if reasons.len() > 8 {
        let _ = writeln!(o, "  {:>7}  other reasons", reasons.len() - 8);
    }
    if t.panicked > 0 {
        let _ = writeln!(o, "panicked     {} (translator bugs; not kept)", t.panicked);
    }
    if pipelines {
        let _ = writeln!(
            o,
            "compute      {} programs: {} host pipelines made, {} from the library, {} without one",
            t.compute, counts.made, counts.loaded, t.no_pipeline
        );
        if let Some(r) = replay {
            o.push_str(&replay_lines(r));
        }
        if let Some(r) = refusals {
            let _ = writeln!(o, "  {r}");
        }
    } else {
        let _ = writeln!(o, "compute      no host pipelines (--no-pipelines)");
    }
    let _ = writeln!(
        o,
        "time         {seconds:.1} s on {threads} threads at below-normal priority"
    );
    o
}

/// The report's lines on the kept graphics and mesh pipelines: how many,
/// what became of them, what was not made and why.
fn replay_lines(r: &Replay) -> String {
    let mut o = String::new();
    if let Some(e) = &r.error {
        let _ = writeln!(o, "graphics     0 kept pipelines (not read: {e})");
        return o;
    }
    let _ = writeln!(
        o,
        "graphics     {} kept pipelines ({} drawn by sessions, made first): {} ready ({} host \
         pipelines made, {} from the library), {} with shaders that do not translate, {} \
         refused, in {:.1} s",
        r.kept, r.drawn, r.ready, r.made, r.loaded, r.untranslated, r.refused, r.seconds
    );
    if r.panicked > 0 {
        let _ = writeln!(o, "  {:>7}  panicked (translator bugs)", r.panicked);
    }
    if r.left > 0 {
        let _ = writeln!(
            o,
            "  {:>7}  not made: {} s passed (XTP_PRECOOK_SECONDS); the next precook goes on",
            r.left,
            r.bound.as_secs()
        );
    }
    if let Some((before, dropped)) = r.compacted {
        let _ = writeln!(
            o,
            "  {dropped:>7}  dropped when the file was rewritten ({before} bytes before; the oldest go first)"
        );
    }
    if let Some(before) = r.earlier {
        let _ = writeln!(
            o,
            "  the file was of an earlier version ({before} bytes): started again, sessions keep their pipelines anew"
        );
    }
    if r.not_compacted {
        let _ = writeln!(
            o,
            "  the file is past its bound but another process has it open: rewritten by a later precook"
        );
    }
    for (n, what) in [
        (r.missing, "without their containers"),
        (r.damaged, "that do not read back"),
        (
            r.repeated,
            "repeated (descriptions this build reads as one pipeline)",
        ),
    ] {
        if n > 0 {
            let _ = writeln!(o, "  {n:>7}  {what}");
        }
    }
    if let Some(path) = &r.path {
        let _ = writeln!(o, "  kept in {}", path.display());
    }
    match &r.coverage {
        Some(Ok(c)) => {
            let _ = writeln!(
                o,
                "effects      {} effects, {} passes ({} name programs not in the pack): {} \
                 draw, with {} shader sets; the kept descriptions have {} of them ({:.1} %), \
                 and {} sets no pass names",
                c.effects,
                c.passes,
                c.outside,
                c.draws,
                c.sets,
                c.covered,
                c.covered as f64 * 100.0 / c.sets.max(1) as f64,
                c.other
            );
            let _ = writeln!(
                o,
                "derived      {} descriptions for {} shader sets without a kept one (their \
                 passes' shaders with the states kept for passes like them): {} ready",
                c.derived, c.derived_sets, r.derived_ready
            );
        }
        Some(Err(e)) => {
            let _ = writeln!(o, "effects      not read: {e}");
        }
        None => {}
    }
    o
}

/// One line per record: its offset, name, container hash, stage and
/// outcome.
fn list(pass: &Pass) -> String {
    let mut o = String::new();
    for item in &pass.items {
        let what = match &item.done {
            Done::Corrupt => "corrupt".to_owned(),
            Done::Repeated => "repeated".to_owned(),
            Done::Panicked(text) => format!("panicked: {text}"),
            Done::Cooked(stage, outcome) => {
                let stage = stage.map_or("-".to_owned(), |s| format!("{s:?}"));
                match outcome {
                    Outcome::Ready => format!("{stage} ready"),
                    Outcome::Refused(reason) => format!("{stage} refused: {reason}"),
                    Outcome::NoPipeline => format!("{stage} no host pipeline"),
                }
            }
        };
        let _ = writeln!(
            o,
            "{:#x}\t{}\t{}\t{what}",
            item.offset,
            item.name.as_deref().unwrap_or("-"),
            item.hash.get(..16).unwrap_or("-")
        );
    }
    o
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// The machine a precook runs on: the session lock, the process holding
/// it for us (if any), and our own process.
struct Machine {
    lock: Option<PathBuf>,
    holder: Option<u32>,
    me: u32,
}

/// The lock while this precook holds it: removed when dropped.
struct Held(PathBuf);

impl Drop for Held {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Machine {
    /// Waits until the machine is free, then takes the lock unless it is
    /// held for us (or there is none).
    fn take(&self) -> Result<Option<Held>> {
        let mut told: Option<String> = None;
        loop {
            if let Some(why) = self.busy() {
                if told.as_ref() != Some(&why) {
                    println!("precook: waiting: {why}");
                    told = Some(why);
                }
                std::thread::sleep(LOOK_EVERY);
                continue;
            }
            let (Some(path), None) = (&self.lock, self.holder) else {
                return Ok(None);
            };
            match std::fs::File::create_new(path) {
                Ok(mut file) => {
                    writeln!(file, "{}\nxtp precook", self.me)
                        .with_context(|| format!("writing {}", path.display()))?;
                    return Ok(Some(Held(path.clone())));
                }
                // Taken meanwhile: look again.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
            }
        }
    }

    /// Why the machine is not free for a precook now (`None`: it is): the
    /// lock held by another live process, or another `xtp.exe` running. A
    /// lock whose process is gone is stale: it is removed.
    fn busy(&self) -> Option<String> {
        if let Some(path) = &self.lock
            && let Ok(text) = std::fs::read_to_string(path)
        {
            match lock_holder(&text) {
                Some(pid) if pid == self.me || Some(pid) == self.holder => {}
                Some(pid) if !alive(pid) => {
                    println!(
                        "precook: removing the stale {} (pid {pid} is gone)",
                        path.display()
                    );
                    let _ = std::fs::remove_file(path);
                }
                Some(pid) => return Some(format!("{} is held (pid {pid})", path.display())),
                None => return Some(format!("{} is held", path.display())),
            }
        }
        xtp_processes(&tasks("IMAGENAME eq xtp.exe"))
            .into_iter()
            .find(|&pid| pid != self.me)
            .map(|pid| format!("xtp.exe runs (pid {pid})"))
    }
}

/// The process id on a lock's first line.
fn lock_holder(text: &str) -> Option<u32> {
    text.lines().next()?.trim().parse().ok()
}

/// `tasklist`'s CSV lines for processes matching `filter`; empty elsewhere.
fn tasks(filter: &str) -> String {
    if !cfg!(windows) {
        return String::new();
    }
    std::process::Command::new("tasklist")
        .args(["/FI", filter, "/NH", "/FO", "CSV"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// The process ids of `xtp.exe` lines in `tasklist` CSV output.
fn xtp_processes(csv: &str) -> Vec<u32> {
    csv.lines()
        .filter_map(|line| {
            let mut fields = line.trim().split("\",\"");
            let name = fields.next()?.trim_start_matches('"');
            let pid = fields.next()?.trim_matches('"');
            name.eq_ignore_ascii_case("xtp.exe")
                .then(|| pid.parse().ok())
                .flatten()
        })
        .collect()
}

fn alive(pid: u32) -> bool {
    !cfg!(windows) || tasks(&format!("PID eq {pid}")).contains(&format!("\"{pid}\""))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The note covers its key, pack and pipelines: a pass with pipelines
    /// covers one without; new kept descriptions are made alone; a library
    /// file gone or smaller, another key or pack, or no note, pass again.
    #[test]
    fn the_note_covers_its_key_pack_and_pipelines() {
        fn cache(recorded: &str, library: u64) -> Option<Cache> {
            Some(Cache {
                recorded: recorded.into(),
                library,
            })
        }
        let note = Note {
            key: "ab".repeat(32),
            pack: "cd".repeat(32),
            pipelines: cache("01", 1000),
        };
        let text = note.text();
        assert!(text.starts_with("xtp precook 3\n"));
        assert_eq!(Note::parse(&text), Some(note.clone()));
        assert_eq!(note.plan(Some(&note)), Plan::Nothing);
        assert_eq!(note.plan(None), Plan::Whole);
        let with = |change: fn(&mut Note)| {
            let mut n = note.clone();
            change(&mut n);
            n
        };
        let other_key = with(|n| n.key = "ef".repeat(32));
        assert_eq!(
            other_key.plan(Some(&note)),
            Plan::Whole,
            "another translator or target"
        );
        let other_pack = with(|n| n.pack = "00".repeat(32));
        assert_eq!(other_pack.plan(Some(&note)), Plan::Whole, "another pack");
        // A session kept more descriptions, and made them into the library.
        let more = with(|n| n.pipelines = cache("02", 1500));
        assert_eq!(more.plan(Some(&note)), Plan::Kept);
        // A session added to the library alone (compute pipelines).
        let larger = with(|n| n.pipelines = cache("01", 1500));
        assert_eq!(larger.plan(Some(&note)), Plan::Nothing);
        // The library file removed, or another one in its place.
        let gone = with(|n| n.pipelines = cache("01", 0));
        assert_eq!(gone.plan(Some(&note)), Plan::Whole);
        // A pass without pipelines is covered by one with them, not the
        // other way round.
        let without = with(|n| n.pipelines = None);
        assert_eq!(without.plan(Some(&note)), Plan::Nothing);
        assert_eq!(note.plan(Some(&without)), Plan::Whole);
        assert_eq!(Note::parse(&without.text()), Some(without.clone()));
        // Not notes.
        assert_eq!(Note::parse(""), None);
        assert_eq!(Note::parse(&text.replace("precook 3", "precook 2")), None);
        assert_eq!(
            Note::parse(&text.replace("library 1000", "library many")),
            None
        );
        assert_eq!(Note::parse(&format!("{text}more\n")), None);
    }

    /// A new generation for another translator, or for a library that grew
    /// past its bound, whatever the rest of the note says (a session that
    /// adds compute pipelines alone leaves the note as it was); not for one
    /// that a new generation left past the bound, until it doubles.
    #[test]
    fn a_library_past_its_bound_is_made_again_before_the_note_holds() {
        let note = |key: &str, library| Note {
            key: key.into(),
            pack: "pack".into(),
            pipelines: Some(Cache {
                recorded: "01".into(),
                library,
            }),
        };
        let bound = 2000;
        let then = note("k", 1500);
        assert_eq!(note("k", 1800).new_generation(Some(&then), bound), None);
        let grown = note("k", 2500);
        assert_eq!(grown.plan(Some(&then)), Plan::Nothing, "the note holds");
        assert_eq!(
            grown.new_generation(Some(&then), bound),
            Some("past its bound"),
            "looked at first"
        );
        assert_eq!(
            note("j", 10).new_generation(Some(&then), bound),
            Some("another translator")
        );
        assert_eq!(grown.new_generation(None, bound), Some("past its bound"));
        // A new generation that is past the bound by itself.
        let big = note("k", 3000);
        assert_eq!(note("k", 3500).new_generation(Some(&big), bound), None);
        assert_eq!(
            note("k", 6000).new_generation(Some(&big), bound),
            Some("past its bound")
        );
        // Without pipelines, the library is not this pass's business.
        let without = Note {
            pipelines: None,
            ..note("j", 0)
        };
        assert_eq!(without.new_generation(Some(&then), bound), None);
    }

    /// The outcomes of the kept pipelines and of the derived ones (after
    /// them) are counted apart, the pipelines not started as left; the
    /// report says the pack's coverage and what was derived.
    #[test]
    fn kept_and_derived_pipelines_are_counted_apart() {
        let mut r = Replay {
            kept: 3,
            derived: 4,
            coverage: Some(Ok(Coverage {
                effects: 2,
                passes: 9,
                outside: 1,
                draws: 6,
                sets: 3,
                covered: 2,
                other: 1,
                derived_sets: 1,
                derived: 4,
            })),
            ..Replay::default()
        };
        r.count(&[
            (0, Replayed::Ready),
            (2, Replayed::Untranslated),
            (3, Replayed::Ready),
            (5, Replayed::Refused),
            (1, Replayed::Ready),
        ]);
        assert_eq!(
            (r.ready, r.untranslated, r.derived_ready, r.left),
            (2, 1, 1, 2)
        );
        let text = replay_lines(&r);
        assert!(text.contains(
            "6 draw, with 3 shader sets; the kept descriptions have 2 of them (66.7 %), and 1 sets no pass names"
        ));
        assert!(text.contains("derived      4 descriptions for 1 shader sets"));
        assert!(text.contains("): 1 ready"));
    }

    #[test]
    fn tasklist_lines_give_xtp_process_ids() {
        let csv = "\"xtp.exe\",\"1234\",\"Console\",\"1\",\"812,345 K\"\r\n\
                   \"XTP.EXE\",\"77\",\"Console\",\"1\",\"1 K\"\r\n\
                   \"xtp-precook.exe\",\"99\",\"Console\",\"1\",\"1 K\"\r\n";
        assert_eq!(xtp_processes(csv), [1234, 77]);
        assert!(
            xtp_processes("INFO: No tasks are running which match the specified criteria.")
                .is_empty()
        );
        assert_eq!(lock_holder("4242\ncargo xtask play\n"), Some(4242));
        assert_eq!(lock_holder(" 7 \n"), Some(7));
        assert_eq!(lock_holder("someone\n"), None);
        assert_eq!(lock_holder(""), None);
    }

    /// A lock held for us (by the session that runs us) or by us is no
    /// reason to wait; one held by a live process is; a stale one is
    /// removed.
    #[cfg(windows)]
    #[test]
    fn the_lock_is_ours_or_someone_elses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("measure.lock");
        let me = std::process::id();
        let machine = |holder| Machine {
            lock: Some(path.clone()),
            holder,
            // Not this test process: a live one that is not "us".
            me: u32::MAX,
        };
        let busy = |m: &Machine| {
            // Other xtp.exe processes may run on this machine: only the
            // lock's answer counts here.
            m.busy().filter(|why| why.contains("measure.lock"))
        };
        assert_eq!(busy(&machine(None)), None, "no lock file");
        std::fs::write(&path, format!("{me}\ncargo xtask play\n")).unwrap();
        assert_eq!(busy(&machine(Some(me))), None, "held for us");
        assert!(
            busy(&machine(None)).is_some_and(|why| why.contains(&format!("pid {me}"))),
            "held by a live process"
        );
        std::fs::write(&path, "garbage\n").unwrap();
        assert!(busy(&machine(None)).is_some(), "held, by whom unknown");
        // A process id no live process has (ids are multiples of 4).
        std::fs::write(&path, "4294967293\n").unwrap();
        assert_eq!(busy(&machine(None)), None, "stale");
        assert!(!path.exists(), "the stale lock is removed");
    }

    /// A pack with a record that does not unpack, two records holding the
    /// same container and one other: each distinct container is cooked
    /// once, in pack order, and the list says what became of each record.
    #[test]
    fn a_pass_cooks_each_container_once() {
        let a = container(b"first");
        let b = container(b"second");
        let mut file = b"SGD5".to_vec();
        for (name, payload) in [("CS_A", &a), ("PS_B", &b), ("CS_A_again", &a)] {
            file.extend(record(name, payload));
        }
        // A record whose stream is not deflate.
        file.extend(broken_record("PS_Broken"));
        let cooked = Mutex::new(Vec::new());
        let cook = |c: &[u8], _: Option<xtp_shader::HwStage>, _: bool| {
            cooked.lock().unwrap().push(c.to_vec());
            if c == b.as_slice() {
                Outcome::Refused("a scalar memory read as data".into())
            } else {
                Outcome::Ready
            }
        };
        let pass = pass(&file, true, 3, None, cook);
        let mut seen = cooked.into_inner().unwrap();
        seen.sort();
        let mut want = vec![a.clone(), b.clone()];
        want.sort();
        assert_eq!(seen, want, "each container once");
        let t = Totals::of(&pass);
        assert_eq!(
            (t.records, t.containers, t.repeated, t.corrupt),
            (4, 2, 1, 1)
        );
        assert_eq!((t.ready, t.refused), (1, 1));
        assert_eq!(t.reasons.get("a scalar memory read as data"), Some(&1));
        let list = list(&pass);
        let lines: Vec<&str> = list.lines().collect();
        assert_eq!(lines.len(), 4);
        let hash_a = &blake3::hash(&a).to_hex()[..16];
        assert!(
            lines[0].contains("CS_A") && lines[0].contains(hash_a) && lines[0].ends_with("- ready")
        );
        assert!(lines[1].ends_with("- refused: a scalar memory read as data"));
        assert!(lines[2].contains("CS_A_again") && lines[2].ends_with("repeated"));
        assert!(lines[3].ends_with("corrupt"));
    }

    /// Whichever record a thread cooked, the first one in pack order shows
    /// the outcome and the others are repeats.
    #[test]
    fn the_first_record_takes_a_repeated_containers_outcome() {
        let item = |hash: &str, done| Item {
            offset: 0,
            name: None,
            hash: hash.to_owned(),
            done,
        };
        let mut items = vec![
            item("aa", Done::Repeated),
            item("", Done::Corrupt),
            item("bb", Done::Cooked(None, Outcome::NoPipeline)),
            item("aa", Done::Cooked(None, Outcome::Ready)),
            item("aa", Done::Repeated),
        ];
        first_takes_the_outcome(&mut items);
        let done: Vec<&Done> = items.iter().map(|i| &i.done).collect();
        assert_eq!(
            done,
            [
                &Done::Cooked(None, Outcome::Ready),
                &Done::Corrupt,
                &Done::Cooked(None, Outcome::NoPipeline),
                &Done::Repeated,
                &Done::Repeated,
            ]
        );
    }

    /// A container stored as is is cut where its header says it ends.
    #[test]
    fn raw_containers_end_where_their_header_says() {
        let c = container(b"payload");
        let mut file = c.clone();
        file.extend_from_slice(b"what follows in the file");
        let raw = Source::Raw {
            offset: 0,
            data: &file,
        };
        assert_eq!(container_bytes(&raw).as_deref(), Some(c.as_slice()));
        let short = Source::Raw {
            offset: 0,
            data: &c[..20],
        };
        assert_eq!(container_bytes(&short), None);
    }

    /// A DXBC container with one `DXIL` part holding `data`.
    fn container(data: &[u8]) -> Vec<u8> {
        let total = 0x24 + 8 + data.len();
        let mut out = b"DXBC".to_vec();
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&[1, 0, 0, 0]);
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0x24u32.to_le_bytes());
        out.extend_from_slice(b"DXIL");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        out
    }

    /// An SGD5 record holding `payload` as two stored deflate blocks, split
    /// after its second byte so that a container's magic does not appear
    /// as is in the file (see `survey`'s tests).
    fn record(name: &str, payload: &[u8]) -> Vec<u8> {
        let mut packed = Vec::new();
        for (last, part) in [(0u8, &payload[..2]), (1, &payload[2..])] {
            let len = u16::try_from(part.len()).unwrap();
            packed.push(last);
            packed.extend_from_slice(&len.to_le_bytes());
            packed.extend_from_slice(&(!len).to_le_bytes());
            packed.extend_from_slice(part);
        }
        framed(name, &packed, payload.len())
    }

    /// A record whose stream does not inflate.
    fn broken_record(name: &str) -> Vec<u8> {
        framed(name, &[0xff; 16], 64)
    }

    fn framed(name: &str, packed: &[u8], unpacked: usize) -> Vec<u8> {
        let n = (packed.len() + 12) as u32;
        let mut out = vec![0x11, 0x22, name.len() as u8 + 1];
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0, 0x32]);
        for v in [n, 0x000D_DCE7, n, unpacked as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(packed);
        out
    }
}
