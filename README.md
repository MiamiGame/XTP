# XTP

Runs BTP 6 (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; XTP supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

## Status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `XTP_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog, closed captions and high contrast from Windows settings | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads with trigger rumble, keyboard, keyboard and mouse as a pad (keys by place, mouse as the right stick) | Trigger rumble and the mouse are untested in play |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode (the default) | The world loads in seconds and runs (150 s in a row measured, about 1.7 fps) | The world, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # policy and commit-message checks on every commit
copy xtp.example.toml xtp.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p xtp -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p xtp
target\release\xtp.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\xtp\data`), derived
data under the cache folder.

Input: XInput pads; Xbox One and Series pads get their trigger motors too
(through Windows.Gaming.Input). The keyboard and mouse play as a pad while
the title's window has focus: a pad of their own while no pad is connected,
on top of the first pad otherwise. Keys go by place, not by label: the
letters below are a US keyboard's, and on AZERTY the key in W's place (Z)
pushes the stick forward just the same.

| Keys and mouse | Pad |
|------|-----|
| W A S D | left stick |
| mouse, I J K L | right stick |
| arrows | D-pad |
| Enter or Space, Backspace or R, X, Y | A, B, X, Y |
| Q / E, wheel down / up | left / right shoulder (a press a notch) |
| Z / C, right / left mouse button | left / right trigger |
| F, H or the middle mouse button | left stick click, right stick click |
| Esc, Tab | Menu, View |

The mouse plays while the cursor is held in the title's window, hidden:
from when the window is active until Alt+Tab or Esc (a menu's cursor is
yours); a click in the window takes it back, and is not a shot.
`[input]` in `xtp.toml` sets the mouse's sensitivity, curve, dead zone and
inversion, and other bindings (see `xtp.example.toml`), by control:
`a`, `b`, `x`, `y`, `menu`, `view`, `dpad_up` (`_down`, `_left`, `_right`),
`left_shoulder`, `right_shoulder`, `left_trigger`, `right_trigger`,
`left_stick_click`, `right_stick_click`, `left_stick_up` (`_down`, `_left`,
`_right`) and `right_stick_up` (likewise).

## Commands

```
xtp inspect <exe|dll> [--imports] [--exports]
xtp census [paths...] [--missing] [--strict]
xtp lib-exports <import.lib> [--list]
xtp shader-info [paths...] [--verbose] [--disasm]
xtp translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
xtp run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-s|series-x] [-- <title arguments>]
xtp config
```

The title sees an Xbox Series S unless `--console series-x` or
`XTP_CONSOLE=series-x` says otherwise: with Series S budgets the world
loads in seconds, where Series X's push the host GPU's resident memory
past the point at which Windows slows every new allocation, and loading
takes minutes (docs/analysis/series-s.md, stall.md). `cargo xtask play`
passes `XTP_CONSOLE` on as `--console`.

Logging: `XTP_LOG=xtp_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p xtp --features diagnostics   # target\diag\xtp.exe
```

| Variable | Effect |
|----------|--------|
| `XTP_LOG=<target>=debug,...` | debug and trace output |
| `XTP_STACKS=20,60` | every thread's stack at those seconds |
| `XTP_WATCH=<rva>,...` | logs a title u32 when it changes |
| `XTP_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not (keys by place, named as in `[input.bindings]`) |
| `XTP_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `XTP_CAPTURE=<dir>` | saves presented frames (powers of two, or `XTP_CAPTURE_FRAMES=<list>`) |
| `XTP_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `XTP_D3D12_DEBUG=1` | host debug layer |
| `XTP_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `XTP_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `XTP_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `XTP_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `XTP_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `XTP_ADAPTER=<index>\|<name>` | (every build) host GPU to run on: its index in the startup log's `host adapter` lines, or part of its name (`XTP_ADAPTER=radeon`); by default the first high-performance hardware adapter, a discrete GPU before an integrated one |
| `XTP_PIPELINE_WAIT_MS=<graphics>,<compute>` | (every build) milliseconds per frame title threads may wait for pipelines being made before their draws and dispatches are dropped for that frame (default `2,8`) |
| `XTP_SHADER_LOOP_LIMIT=<n>` | (every build) translated shader loops stop after `n` iterations per lane instead of 16384 (2^14, under the host GPU watchdog); loops that only reread memory until it changes (spin waits) after 4096, or `n` if lower (own shader cache entries); a lower `n` shows whether loops hitting the limit make world frames slow, a higher one whether a wrong result comes from a legitimate loop the default cut short |
| `XTP_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `XTP_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |
| `XTP_WATCHDOG_HOOK=1` | every build: keeps the title's own watchdog from ending it after about 30 s without progress (slow world frames), and logs a warning (at most every 30 s) when it would have; a real deadlock then hangs for ever, with no `crashes.log` entry; known builds only (1.0.2.1), otherwise a warning and nothing patched |
| `XTP_SKIP_INTRO=1` | every build: cuts the intro movie of the start-up front end (sent to its last frame when it starts), so the world starts loading about 13 s sooner (docs/analysis/intro-skip.md); known builds only (1.0.2.1), otherwise a warning and nothing changed |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder. The loader also prints the first 16 title C++ throws
(decorated type name, readable MSVC `std::exception` message and title
`module+rva`) to stderr,
and maps legacy debugger thread names to Windows thread descriptions.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first: it
has the rules every change must follow (nothing from the game, no keys, no
console SDK material, nothing that circumvents DRM, GPL-3.0-or-later, no code
from the projects we only use as references), the commit and pull request
conventions and how review works. [docs/dev/START_HERE.md](docs/dev/START_HERE.md)
maps the code and lists good first issues. Before opening a pull request, run

```powershell
cargo xtask pr-check     # what CI checks, with a summary
```

Report vulnerabilities privately as [SECURITY.md](SECURITY.md) describes.
Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

XTP started from [VIctory](https://github.com/Marvisak/VIctory) by Marvisak
(commit `ffe310e`); see [NOTICE.md](NOTICE.md) for the origin of the code and
the projects used as references only.
