# Contributing without a gaming PC

XTP runs one Xbox Series title on Windows. In this repository it is "the
title" (the analysis docs and `crates/xtp-game` use its code name, BTP 6).
Running it under XTP needs a strong PC and GPU, but much of the work does
not: the shader translator, static reverse engineering, analysis, driver
coverage and tests all run on an ordinary machine. This page says what you
can do there, with the exact commands.

Read [CONTRIBUTING.md](../CONTRIBUTING.md) first. Everything below follows
its rules.

Contents: [who it is for](#who-it-is-for) · [setup](#setup) ·
[tracks](#tracks) · [what needs the game running](#what-needs-the-game-running) ·
[how to submit](#how-to-submit)

## Who it is for

You own the title, but your PC cannot run it under XTP. You need:

| What | Why |
|------|-----|
| Your own, legally obtained copy of the title | Some tracks read its shader pack or its executable |
| Windows 10 or 11, x64 | XTP and its tools are Windows programs |
| A PC that builds Rust: about 16 GB of RAM, several GiB of free disk | Building and testing the workspace |
| Optional: any Direct3D 12 GPU | Only for track (d), driver coverage |
| Optional: an LLM | Welcome, under the rules below |

**Your copy, your files.** Each contributor works from their own legally
owned copy. The project does not provide game files, does not help anyone
obtain, dump, decrypt or unpack them, and does not answer such questions in
issues. XTP never writes to a title's files (CONTRIBUTING.md, rule 4).

### The rules, in short

- **English** in code, comments, docs, commits, issues and pull requests.
- **Do not name the title**, its publisher or its setting. Say "the title"
  and describe the behaviour ("Writing about titles" in CONTRIBUTING.md).
- **No game data** anywhere on GitHub: no shader bytes, disassembly or
  decompiled code, strings or record names from the title, file lists,
  captures, screenshots ([docs/asset-policy.md](asset-policy.md)).
- **What you may share:** program hashes (16 hex digits, as the logs and
  docs print them), counts, the translator's own refusal messages, RVAs of
  a known build with behaviour in your own words, byte-pattern signatures
  of functions we hook, small constants.

Several tools print names from the title or your local paths. Keep their
output on your machine and copy only counts, hashes and reasons from it:

| Output | What it holds |
|--------|---------------|
| `xtp shader-info --verbose` / `--disasm` | record names, the console code |
| `xtp translate` failure lines | an example record, after `(e.g.` |
| `xtp translate --list`, `xtp precook --list FILE` | one line per record, with its name |
| `xtp shader-survey --json FILE` | one line per program |
| the first line of `shader-survey` and `precook` | the pack's local path |
| `xtp inspect` | the executable's debug-file path (`pdb`) |

The tools refuse to write `--json` and `--list` files inside the
repository. Keep a work folder outside it (`C:\xtp-work` below).

### Using an LLM

AI-assisted contributions are allowed. CONTRIBUTING.md, "Provenance": say
in the pull request whether an AI tool wrote part of the change (the
template's `AI assistance:` line). You answer for every line, its
provenance included, as if you had typed it. What the LLM writes for you
follows the same rules as what you write: no title names, no game data,
nothing copied from xcl or AnyPS5 (rule 6). Read its output before you post
it.

## Setup

### Toolchain

From CONTRIBUTING.md, "Setup":

- [rustup](https://rustup.rs). The toolchain, the `x86_64-pc-windows-msvc`
  target and the `rustfmt` and `clippy` components are pinned in
  `rust-toolchain.toml`; rustup installs them on first use.
- Visual Studio 2022 Build Tools with the "Desktop development with C++"
  workload: the MSVC linker and the Windows SDK (which also brings the DXIL
  validator, below).
- Git.

```powershell
git clone https://github.com/<you>/XTP.git      # your fork
cd XTP
git remote add upstream https://github.com/MiamiGame/XTP.git
cargo xtask install-hooks                        # policy and commit-message checks
copy xtp.example.toml xtp.toml                   # then set its paths, outside the repo
```

### Build and test

```powershell
cargo build --release -p xtp            # target\release\xtp.exe
cargo test --workspace                  # needs no game data
Set-Alias xtp "$PWD\target\release\xtp.exe"   # this page writes `xtp ...`
```

`cargo run --release -p xtp -- <command>` works as well.

### Point XTP at your copy

`xtp.toml` (git-ignored), section `[paths]`; an environment variable takes
precedence over the file, and `XTP_CONFIG` selects another file:

| Setting | Variable | What |
|---------|----------|------|
| `game_dir` | `XTP_GAME_DIR` | your title folder |
| `cache_dir` | `XTP_CACHE_DIR` | derived data (default `%LOCALAPPDATA%\xtp\cache`) |
| `data_dir` | `XTP_DATA_DIR` | what titles save (default `%LOCALAPPDATA%\xtp\data`) |
| – | `XTP_SAMPLE_SHADERS` | the title's shader pack, for `shader-survey`, `precook` and the opt-in survey test |

Every path must be outside the repository: the tools refuse otherwise.
With no pack given, `shader-survey` and `precook` take `XTP_SAMPLE_SHADERS`,
else the shader pack in `game_dir` (`xtp shader-survey --help` names the
file). Check what XTP sees:

```powershell
xtp config
$env:XTP_SAMPLE_SHADERS = '<the shader pack in your title folder>'
```

### The DXIL validator

`--validate` checks each translation with `dxil.dll`, found at `XTP_DXIL`
if set, else in the newest Windows SDK
(`C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\dxil.dll`), else
on the DLL search path (`crates/xtp-shader/src/emit/validator.rs`). The
Build Tools workload installs one. The figures in
[docs/analysis/w6-validation.md](analysis/w6-validation.md) used validator
1.9 from SDK 26100.

### Tests on WARP (no GPU)

WARP is Windows' software Direct3D 12 device. GPU tests take their device
from `XTP_TEST_ADAPTER`: `warp`, `hardware` (the default) or `none`
(`crates/xtp-gpu/src/testgpu.rs`).

```powershell
$env:XTP_TEST_ADAPTER = 'warp'
cargo test -p xtp-shader -p xtp-gpu --locked
```

- Without a device, an `xtp-gpu` or `xtp-shader` test prints
  `SKIPPED <test>: <reason>` and passes, unless `XTP_REQUIRE_GPU` is set:
  then it fails.
- CI runs the `xtp-gpu` tests on WARP under the D3D12 debug layer. Locally,
  with Windows' optional "Graphics Tools" feature installed:

  ```powershell
  $env:XTP_TEST_ADAPTER='warp'; $env:XTP_D3D12_DEBUG='1'; $env:RUST_TEST_THREADS='1'
  cargo test -p xtp-gpu --features diagnostics --locked
  ```

- Translated shaders need shader model 6.6. XTP does not load the Agility
  SDK and relies on Windows 11's D3D12 runtime for it
  (`crates/xtp-gpu/src/dxgi/adapter.rs`), so on Windows 10 tests that need
  SM 6.6 may skip.

Long commands take `--jobs N` (`translate`, `shader-survey`, `precook`) to
leave cores for you; `precook` also runs at below-normal priority.

## Tracks

Open issues for this page carry the label
[`no-gaming-pc`](https://github.com/MiamiGame/XTP/issues?q=is%3Aissue+is%3Aopen+label%3Ano-gaming-pc).
Comment on an issue before you start, so that two people do not do the same
work.

| Track | Needs | Also look for |
|-------|-------|---------------|
| (a) Shader translation | your shader pack | `area/shader` |
| (b) CPU-side reverse engineering | your executable, Ghidra or IDA | `area/hle` |
| (c) Shader role map | your shader pack, the analysis docs | `documentation` |
| (d) AMD and Intel driver coverage | your shader pack, a DX12 GPU | `area/gpu` |
| (e) Tests and tools | nothing from the title | `area/tests` |

### (a) Shader translation without a GPU

**What.** `crates/xtp-shader` recompiles each console (GFX10) program to
DXIL when the title creates a pipeline. A program it refuses has its draws
and dispatches dropped, so what it draws is missing from the image.

**Why now.** The last measured survey translates 25,768 of the pack's
26,698 programs for a 32-lane host and refuses 930
([w6-validation.md](analysis/w6-validation.md)). The goal is that every
program of the pack translates
([refusals.md](analysis/refusals.md), introduction). Your own survey of
`main` gives today's figure.

**Commands.**

```powershell
xtp shader-survey --host-lanes 32-32 --json C:\xtp-work\before.json
```

- The report counts each program once: translated, or under its first
  blocker, grouped by the backlog item that removes it
  ([shader.md](analysis/shader.md), sections 4 and 6).
- `--host-lanes 32-32` translates for NVIDIA's 32-lane waves, as the docs'
  figures do; `32-64` is AMD RDNA's. Without it, the translation is for any
  host.
- `--validate` checks every translation with `dxil.dll`;
  `--validate-every N` checks every Nth record only.
- About 5 minutes on the project's 16-vCPU cloud machine
  ([refusals.md](analysis/refusals.md), section 9); longer on fewer cores.
- The JSON has one line per program with the hash of its translation, so
  two runs compare as text:

  ```powershell
  git diff --no-index C:\xtp-work\before.json C:\xtp-work\after.json
  ```

**How to work on a class.**

1. Pick a blocker class from your survey. Read its method and files in
   [refusals.md](analysis/refusals.md), section 3 (its counts predate
   later fixes: trust your survey), and the backlog in
   [shader.md](analysis/shader.md), section 6.
2. To read one program, map hashes to records locally, then read the
   record's code (both files stay on your machine):

   ```powershell
   xtp precook --no-pipelines --list C:\xtp-work\records.txt   # offset, name, hash, outcome
   xtp shader-info $env:XTP_SAMPLE_SHADERS --disasm > C:\xtp-work\disasm.txt
   ```

   `precook` writes the translations into your cache folder (about 1.1 GiB
   for the whole pack) and needs a Direct3D 12 device; without a GPU,
   `$env:XTP_ADAPTER = 'basic render'` names WARP.
3. Fix the translator. Write a test that rebuilds the pattern from
   hand-encoded instructions, never from the program's bytes; see
   `crates/xtp-shader/src/emit/lightloop_tests.rs`. It translates, the
   DXIL validates, and it runs on WARP where it executes.
4. Survey again. Translations rise, none is lost (translated to refused),
   and no other translation's hash changes unless you explain why.

The opt-in test `survey::tests::the_pack_survey_keeps_its_floor` runs the
survey on `XTP_SAMPLE_SHADERS`: `cargo test --release -p xtp survey`.

**A good contribution:** one class or one pattern of it, synthetic tests,
and the survey before and after as counts in the pull request ("+N
translated, 0 lost, M outputs changed because ...").

### (b) Reverse engineering on the CPU side

**What.** Static analysis of your own copy's executable, with Ghidra, IDA
or similar, for interoperability: what the title's code writes into a GPU
constant record, which system calls it makes and with what, what a hook
needs.

**Why now.** Some of what the image lacks is a value the CPU writes.
[docs/analysis/title-cpu-image.md](analysis/title-cpu-image.md) is the
model for this work: code named by RVA, behaviour in our own words.

**Commands.**

```powershell
xtp inspect "<title folder>\<executable>"      # blake3: your build's identity
xtp census "<title folder>" --missing          # missing and table-stubbed imports, listed separately
xtp census "<title folder>" --missing --strict # disable host pass-through
```

- The docs' RVAs are for build 1.0.2.1 (image base `0x140000000`), whose
  executable's BLAKE3 is in `crates/xtp-game/src/lib.rs`
  (`KNOWN_BUILDS`). If `xtp inspect` prints another BLAKE3, your RVAs
  differ: say so.
- `census` with no path takes `game_dir` and `test_titles`; `--json` gives
  stable JSON. An import we do not serve is a well-bounded change
  ([START_HERE](dev/START_HERE.md)): write it from public documentation and
  observed behaviour, never from console SDK material (rule 3), with a test
  that needs no title.
- `xtp lib-exports <import.lib> --list` lists the `dll!export` pairs an
  import library describes; use only libraries you may use (the Windows
  SDK's, never a console SDK's).

**How to report.** In the issue, or as a `docs/analysis` note in a pull
request:

- the build (version or BLAKE3);
- each function or datum by RVA;
- what it does, in your own words: what it reads, what it writes and at
  which offsets, under which condition, who calls it.

Allowed: RVAs, byte-pattern signatures, small constants. Not allowed:
decompiled or disassembled code, strings or tables copied from the
executable.

**A good contribution:** a question answered with RVAs and behaviour that
someone else can check in their own copy in minutes.

### (c) Shader role map

**What.** A table that says, per program hash, what the program does
("clusters the lights in world space", "temporal resolve"), its stage, and
the evidence.

**Why now.** The image work turns on which dropped program matters. The
analysis docs already name many programs by hash, each in its own tables;
one map makes all of them readable.

The [shader role map](analysis/shader-roles.md) starts with the documented
world-frame passes and their light chain.

**How.**

1. Start from the hashes in [light-chain.md](analysis/light-chain.md)
   (sections 3 and 4), [pix-world-frame.md](analysis/pix-world-frame.md)
   (sections 4 and 11), [world-black.md](analysis/world-black.md),
   [gpu-frame.md](analysis/gpu-frame.md) and
   [refusals.md](analysis/refusals.md), section 6.
2. Find each program's record with the `precook --list` file of track (a)
   and read its code locally with `shader-info --disasm`: what it reads and
   writes, its group size, its outputs' formats, where it sits in the
   docs' pass tables.
3. Write the role in your own words, with a confidence and where it comes
   from (a doc section, or your reading).

Never copy record names, shader entry names or other strings from the pack
into the map: they are the title's text.

**A good contribution:** a Markdown table (hash, stage, role, evidence,
confidence) in a docs pull request, starting with the programs of one world
frame.

### (d) AMD and Intel driver coverage with a modest DX12 GPU

**What.** Create XTP's host pipelines with your GPU's driver, without
running the game, and report what the driver accepts.

**Why now.** The measurements in `docs/analysis` are on NVIDIA GPUs (an RTX
4080 SUPER and a cloud L4). There a sample of the pack's translations, all
four stages, got every pipeline it asked for
([pso-refusals.md](analysis/pso-refusals.md)). AMD's and Intel's drivers,
and their wave sizes, are not measured.

**Commands.** Both need the shader pack (not the rest of the title folder)
and a Direct3D 12 device. Set `XTP_ADAPTER` to part of your GPU's name
(`radeon`, `intel`) so that both use the same one: without it, `precook`
takes a discrete GPU before an integrated one, and `translate` the adapter
that drives the main display.

```powershell
xtp precook --force --list C:\xtp-work\precook.txt
xtp translate $env:XTP_SAMPLE_SHADERS --validate --pipelines
```

- `precook` translates every program as the title would, for your GPU's
  wave sizes, and makes every compute program's host pipeline. Its report:
  `host` (your wave sizes), `translated`, `refused` with its main reasons,
  and `compute N programs: M host pipelines made, L from the library, K
  without one`. Graphics and mesh pipelines it makes only from descriptions
  that game sessions recorded, so its `graphics` line stays at 0 for you.
  `--force` passes again over an up-to-date cache (after a driver update,
  too); `--no-pipelines` translates only.
- `translate --pipelines` creates a host pipeline for every emitted vertex,
  pixel, mesh and compute shader, with a stub partner shader where the
  stage needs one, and reports per stage `emitted`, `validated` and
  `pipelines`, then each failure reason with its count.
- Expect tens of minutes; `--jobs N` keeps your PC usable.

**What to report:** GPU, driver version, Windows build; the `host` line;
the counts above; each `pipeline:` failure reason with its count, without
the `(e.g. ...)` part; and, from the precook list, the hashes and outcomes
(not the names) of programs your driver refuses.

### (e) Tests and tools

**What.** Tests that need no title and run on WARP, test helpers, and the
`xtp` and `xtask` tools.

**Why now.** Most GPU tests run on WARP; gaps in what they check are listed
in [docs/analysis/hardware-tests.md](analysis/hardware-tests.md), section
3.

**How.** The WARP commands of [Setup](#tests-on-warp-no-gpu), then
`cargo xtask pr-check`. Tests follow CONTRIBUTING.md, "Tests": synthetic
inputs, a skip through `xtp_testkit::skip` when an input is missing, and
assertions on properties, never on snapshots of game content.

**A good contribution:** a test that would have caught a real bug, or a
helper that removes a class of silent skips.

## What needs the game running

Without running the title you cannot:

- run it (`xtp run --execute`, `cargo xtask play`) and judge its image,
  frame time or GPU hangs;
- capture or probe frames (`XTP_CAPTURE`, `XTP_DUMP_TARGETS`, pass dumps,
  frame probes, PIX), or read a run log with `cargo xtask frames`;
- give `precook` graphics and mesh pipelines to make: sessions record
  them;
- check that a translation is right in the title. A validated translation
  is not yet a correct one ([es-gs.md](analysis/es-gs.md)).

Frame recordings, captures and render-target dumps are game data. They are
never committed or attached, and maintainers do not share theirs either.
What they showed reaches `docs/analysis` as hashes, counts and
descriptions.

## How to submit

- Comment on the issue first.
- Branch `<area>/<topic>` in your fork. Commits and the pull request title
  are `<area>: <summary>` ([CONTRIBUTING.md, "Commits and pull request
  titles"](../CONTRIBUTING.md#commits-and-pull-request-titles)).
- Run `cargo xtask pr-check` before you push.
- Fill in the pull request template ([CONTRIBUTING.md, "Pull
  requests"](../CONTRIBUTING.md#pull-requests)):
  - tick a [Provenance](../CONTRIBUTING.md#provenance) box, and list what
    you read (xcl and AnyPS5 for behaviour only, rule 6);
  - fill in the `AI assistance:` line;
  - `Ran on a title: no` is a valid answer.
- Findings without code (RE results, driver counts, role notes) can be a
  comment on the issue; a maintainer may fold them into `docs/analysis`.

Before you post anything, reread it: English, "the title" and no title
names, no game data. `cargo xtask policy` checks files, not issues or
pull request text.
