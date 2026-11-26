//! `xtp`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
mod precook;
mod preflight;
mod replay;
mod run;
mod shader;
mod survey;
mod translate;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "xtp",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
        /// Emit stable JSON instead of the text summary.
        #[arg(long)]
        json: bool,
    },
    /// Classify imports as the loader binds them.
    Census {
        /// Each folder is a separate title, scanned without subfolders.
        /// A file is one image with no title DLLs available. Defaults to
        /// the configured game and test-title folders.
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List missing and table-stubbed imports separately, most used first.
        #[arg(long)]
        missing: bool,
        /// Emit stable JSON instead of the text table.
        #[arg(long)]
        json: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found.
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`XTP_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader
        /// (translated for this host's wave sizes unless --host-lanes says).
        #[arg(long)]
        pipelines: bool,
        /// Translate for a host whose waves are MIN-MAX lanes (32-32 for
        /// NVIDIA, 32-64 for AMD RDNA): shaders that need another wave size
        /// are listed as failures. Default: any wave size.
        #[arg(long, value_name = "MIN-MAX", value_parser = parse_lanes)]
        host_lanes: Option<(u32, u32)>,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Survey a shader pack (your own `shaders.rpf`): every program through
    /// the translator, counted once under its first blocker and grouped by
    /// the backlog item that removes it (docs/analysis/shader.md, S0).
    /// Defaults to `XTP_SAMPLE_SHADERS`, else `shaders.rpf` in the
    /// configured game folder.
    ShaderSurvey {
        pack: Option<PathBuf>,
        /// Also write the survey as stable JSON (one program per line, with
        /// the hash of its translation), to compare two runs as text.
        #[arg(long, value_name = "FILE")]
        json: Option<PathBuf>,
        /// Check every translation with dxil.dll's validator (`XTP_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Validate the translations of every Nth record only.
        #[arg(long, value_name = "N")]
        validate_every: Option<usize>,
        /// Translate for a host whose waves are MIN-MAX lanes (32-32 for
        /// NVIDIA, 32-64 for AMD RDNA). Default: any wave size.
        #[arg(long, value_name = "MIN-MAX", value_parser = parse_lanes)]
        host_lanes: Option<(u32, u32)>,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Fill the shader cache before a session: every program of the pack
    /// translated as the title would translate it, into the cache it reads,
    /// and every compute program's host pipeline made into the pipeline
    /// library. Waits while a game session runs (the session lock, another
    /// xtp.exe), then holds the lock and runs at below-normal priority; a
    /// pass already made for this translator, its switches and this pack
    /// is not made again. The pack as shader-survey takes it.
    Precook {
        pack: Option<PathBuf>,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Translations only, no compute host pipelines.
        #[arg(long)]
        no_pipelines: bool,
        /// Pass again even when the cache is up to date (after a driver
        /// update, which makes the host drop the pipeline library).
        #[arg(long)]
        force: bool,
        /// Also write one line per record: its container's hash as the
        /// pipeline log prints it, its stage and what became of it.
        #[arg(long, value_name = "FILE")]
        list: Option<PathBuf>,
        /// The session lock to wait for and hold (default: XTP_SESSION_LOCK).
        #[arg(long, value_name = "FILE")]
        lock: Option<PathBuf>,
        /// The process that holds the session lock for this precook
        /// (`cargo xtask play` passes its own).
        #[arg(long, value_name = "PID")]
        lock_holder: Option<u32>,
        /// Start at once, without waiting for the machine or taking the
        /// lock (tests, or when you know no session runs).
        #[arg(long)]
        no_wait: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        /// A title executable or a package folder with a manifest or a single .exe.
        #[arg(value_name = "EXE|FOLDER")]
        target: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// The console model the title sees (device type, memory); else
        /// `XTP_CONSOLE`; else Series S, whose smaller budgets keep the
        /// host GPU's memory manager out of the long stalls Series X hits
        /// (docs/analysis/series-s.md).
        #[arg(long, value_enum)]
        console: Option<Console>,
        /// Drive the title by this script (autopilot/, docs/dev/autopilot.md).
        #[arg(long, value_name = "SCRIPT")]
        autopilot: Option<PathBuf>,
        /// Record the title's GPU work into this folder (outside the
        /// repository: it is game data), as `XTP_RECORD` does; diagnostics
        /// builds (docs/analysis/frame-replay.md).
        #[arg(long, value_name = "DIR")]
        record: Option<PathBuf>,
        /// The frames to record: `<first>[:<count>]`, or `capture[:<count>]`
        /// for the first frames `XTP_CAPTURE_WORLD` lists (3 by default), as
        /// `XTP_RECORD_FRAMES`.
        #[arg(long, value_name = "N[:COUNT]", requires = "record")]
        record_frames: Option<String>,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Replay a recording (`run --record`) without the title, printing one
    /// JSON line per frame; diagnostics builds (docs/analysis/frame-replay.md).
    Replay {
        /// The recording's folder.
        dir: PathBuf,
        /// The frames to report, by session number: `A..B` or `A`.
        #[arg(long, value_name = "A..B")]
        frames: Option<String>,
        /// Replay the recorded frames this many times.
        #[arg(long = "loop", value_name = "K", default_value_t = 1)]
        loops: u32,
        /// Where captures go (`XTP_CAPTURE`): the recording's `replay`
        /// folder by default.
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Values written into title memory wherever the replay writes it:
        /// `<address>[+<offset>]=<f32|u32|u8|hex>:<values>[@<frame>]`.
        #[arg(long = "poke", value_name = "POKE")]
        pokes: Vec<String>,
        /// Pokes from a file, one per line (`#` starts a comment).
        #[arg(long, value_name = "FILE")]
        pokes_file: Option<PathBuf>,
        /// A number of the frame's pass dump (`XTP_PASS_DUMP`) in each line:
        /// `<name>=ds` (the largest depth target), `<name>=<program hash>`
        /// (its pass's first output) or `<name>=<W>x<H>:f<format>`.
        #[arg(long = "stat", value_name = "NAME=WHAT")]
        stats: Vec<String>,
        /// Write the pages the session wrote back after each submission
        /// over the replay's own (exact session memory, fixes to GPU
        /// results hidden).
        #[arg(long)]
        session_gpu_pages: bool,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

impl Console {
    /// `--console`, else `XTP_CONSOLE` (`series-x` or `series-s`), else
    /// Series S.
    fn pick(flag: Option<Console>, env: Option<&str>) -> Result<Console> {
        if let Some(console) = flag {
            return Ok(console);
        }
        match env.map(str::trim).filter(|v| !v.is_empty()) {
            None => Ok(Console::SeriesS),
            Some(value) => Console::from_str(value, true)
                .map_err(|_| anyhow::anyhow!("XTP_CONSOLE={value}: expected series-x or series-s")),
        }
    }
}

/// The default log filter: `precook` makes thousands of shaders and
/// pipelines, and says itself what became of them (a refusal reason is
/// still logged once).
fn log_filter(command: Option<&str>) -> &'static str {
    match command {
        Some("precook") => "warn,xtp_gpu::pipeline=error,xtp_gpu::pipeline::host=warn",
        _ => "info",
    }
}

fn main() -> Result<()> {
    xtp_core::logging::init(log_filter(std::env::args().nth(1).as_deref()));
    // The audio recorder `run` starts for `XTP_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if xtp_hle::modules::audio::capture::child() {
        return Ok(());
    }
    // An `XTP_*` set to `ab` that cannot alternate (a translation switch,
    // or one not read through `xtp_core::ab`) stops the run here.
    xtp_core::ab::check_environment().map_err(anyhow::Error::msg)?;
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
            json,
        } => inspect::run(&file, imports, exports, json),
        Command::Census {
            paths,
            strict,
            missing,
            json,
        } => census::run(paths, strict, missing, json),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
        } => shader::run(paths, verbose || disasm, disasm),
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            host_lanes,
        } => translate::run(paths, validate, pipelines, list, jobs, host_lanes),
        Command::ShaderSurvey {
            pack,
            json,
            validate,
            validate_every,
            host_lanes,
            jobs,
        } => survey::run(
            pack,
            json,
            validate_every.or(validate.then_some(1)),
            jobs,
            host_lanes,
        ),
        Command::Precook {
            pack,
            jobs,
            no_pipelines,
            force,
            list,
            lock,
            lock_holder,
            no_wait,
        } => precook::run(precook::Options {
            pack,
            jobs,
            pipelines: !no_pipelines,
            force,
            list,
            lock,
            lock_holder,
            wait: !no_wait,
        }),
        Command::Run {
            target,
            unimplemented,
            strict,
            execute,
            console,
            autopilot,
            record,
            record_frames,
            title_args: _,
        } => {
            let env = std::env::var("XTP_CONSOLE").ok();
            let console = Console::pick(console, env.as_deref())?;
            if let Some(dir) = record {
                record_into(&dir, record_frames.as_deref())?;
            }
            run::run(&target, unimplemented, strict, execute, console, autopilot)
        }
        Command::Replay {
            dir,
            frames,
            loops,
            out,
            pokes,
            pokes_file,
            stats,
            session_gpu_pages,
        } => replay::run(replay::Args {
            dir,
            frames,
            loops,
            out,
            pokes,
            poke_file: pokes_file,
            stats,
            session_gpu_pages,
        }),
        Command::Config => show_config(),
    }
}

/// `run --record`: the variables the recorder reads (`XTP_RECORD`,
/// `XTP_RECORD_FRAMES`), checked here first.
fn record_into(dir: &std::path::Path, frames: Option<&str>) -> Result<()> {
    anyhow::ensure!(
        cfg!(feature = "diagnostics"),
        "--record needs a diagnostics build (--features diagnostics)"
    );
    xtp_core::ensure_outside_repo(dir)?;
    // SAFETY: set before the title or any thread of ours starts.
    unsafe {
        std::env::set_var("XTP_RECORD", dir);
        if let Some(frames) = frames {
            std::env::set_var("XTP_RECORD_FRAMES", frames);
        }
    }
    Ok(())
}

/// `MIN-MAX` lanes per wave, or one size `N`.
fn parse_lanes(s: &str) -> Result<(u32, u32), String> {
    let (a, b) = s.split_once('-').unwrap_or((s, s));
    let n = |x: &str| {
        x.trim()
            .parse::<u32>()
            .ok()
            .filter(|n| n.is_power_of_two() && (4..=128).contains(n))
            .ok_or_else(|| format!("{x:?} is not a wave size (4 to 128, a power of two)"))
    };
    let (a, b) = (n(a)?, n(b)?);
    if a > b {
        return Err(format!("{a} is more than {b}"));
    }
    Ok((a, b))
}

fn show_config() -> Result<()> {
    let cfg = xtp_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = xtp_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = xtp_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see xtp.example.toml)"
    );
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn console_of(args: &[&str]) -> Option<Console> {
        let cli = Cli::try_parse_from(args).unwrap();
        let Command::Run { console, .. } = cli.command else {
            panic!("not run");
        };
        console
    }

    #[test]
    fn precook_takes_its_options() {
        let cli = Cli::try_parse_from([
            "xtp",
            "precook",
            "shaders.rpf",
            "--jobs",
            "4",
            "--no-pipelines",
            "--lock-holder",
            "12",
        ])
        .unwrap();
        let Command::Precook {
            pack,
            jobs,
            no_pipelines,
            force,
            lock_holder,
            no_wait,
            ..
        } = cli.command
        else {
            panic!("not precook");
        };
        assert_eq!(pack, Some(PathBuf::from("shaders.rpf")));
        assert_eq!((jobs, no_pipelines, force), (Some(4), true, false));
        assert_eq!((lock_holder, no_wait), (Some(12), false));
        assert_eq!(log_filter(Some("run")), "info");
        assert!(log_filter(Some("precook")).starts_with("warn"));
    }

    #[test]
    fn replay_takes_its_options() {
        let cli = Cli::try_parse_from([
            "xtp",
            "replay",
            "E:/r/one",
            "--frames",
            "618..620",
            "--loop",
            "3",
            "--poke",
            "0x10=f32:1",
            "--poke",
            "0x20=u8:2@619",
            "--stat",
            "depth=ds",
        ])
        .unwrap();
        let Command::Replay {
            dir,
            frames,
            loops,
            pokes,
            stats,
            session_gpu_pages,
            ..
        } = cli.command
        else {
            panic!("not replay");
        };
        assert_eq!(dir, PathBuf::from("E:/r/one"));
        assert_eq!((frames.as_deref(), loops), (Some("618..620"), 3));
        assert_eq!(pokes.len(), 2);
        assert_eq!(stats, ["depth=ds"]);
        assert!(!session_gpu_pages);
        let cli = Cli::try_parse_from([
            "xtp",
            "run",
            "game.exe",
            "--record",
            "E:/r",
            "--record-frames",
            "618:3",
        ]);
        assert!(cli.is_ok());
        assert!(Cli::try_parse_from(["xtp", "run", "game.exe", "--record-frames", "618"]).is_err());
    }

    #[test]
    fn series_s_unless_told() {
        let flag = console_of(&["xtp", "run", "game.exe"]);
        assert_eq!(flag, None);
        assert_eq!(Console::pick(flag, None).unwrap(), Console::SeriesS);
        assert_eq!(Console::pick(flag, Some(" ")).unwrap(), Console::SeriesS);
    }

    #[test]
    fn the_flag_then_the_variable_bring_series_x_back() {
        let flag = console_of(&["xtp", "run", "game.exe", "--console", "series-x"]);
        assert_eq!(flag, Some(Console::SeriesX));
        assert_eq!(
            Console::pick(flag, Some("series-s")).unwrap(),
            Console::SeriesX
        );
        assert_eq!(
            Console::pick(None, Some("Series-X")).unwrap(),
            Console::SeriesX
        );
        assert!(Console::pick(None, Some("one")).is_err());
    }
}
