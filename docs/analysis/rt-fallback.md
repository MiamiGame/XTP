# The title's choice between ray tracing and its fallback, and how to force it

Task rtfallback (d5499893), 2026-11-25. The title is BTP 6, build 1.0.2.1.
Static analysis of its executable only (Ghidra 12.2 headless on the cloud L4
VM, raw-byte queries on the same machine): RVAs (image base 0x140000000),
pass numbers and behaviour in our own words, no code. Programs are named by
hash (the runtime's 16-digit hash, as the logs print it), their roles from
rt-standin.md and refusals.md. The queries and their outputs stay outside
the repository, under `E:\xtp-data\re\` (section 6).

The question (rt-standin.md §3): the pack carries non-ray-traced variants
of the gathers that leave the irradiance cache and the probes at 0, and
they translate, but none runs; the title's data has no switch for it.
Where does the code pick the traced path, and can XTP make it pick the
other?

## 1. Summary

- **One predicate decides, everywhere: RVA `0x4fd8550`, "is ray-tracing
  scene `i` ready".** It reads only the title's own state: its ray-tracing
  manager (the pointer at RVA `0x8527dc8`), the manager's scene `i`, and
  whether that scene has something to trace (section 2). It asks neither
  the system nor the device: there is no capability XTP could answer "no"
  to, which is why the device never sees a feature query (rt-standin.md
  §3).
- **The probe system takes its fallback when the predicate says no.** Its
  primary gather runs pass 17 of its technique, the fallback gather
  `81560230fcde2b0a` (translates), instead of binding the structure and
  running pass 10 or 11 (`aeb93e0888250bd3`, `3592426e3e9142d8`, refused);
  the transparent surfaces' probes run pass 73, the fallback gather and
  shade `29dc558346584906` (translates), instead of pass 68 or 69
  (`3d121c93c7f9a057`, `36f04a4ed0ad1d9b`). The irradiance cache has no
  fallback: without the predicate its gather (pass 42 or 43) and its
  shadow rays (pass 61 or 62) are not dispatched at all (section 3).
- **The voxel variants are not this choice.** None of the probe system's
  16 dispatch steps selects the voxel passes (16, 46, 71) on either side
  of the predicate: the voxel path, if the title still uses it, is chosen
  elsewhere, and forcing "no ray tracing" does not reach it.
- **46 call sites ask the same predicate** (45 calls, one tail jump): the
  indirect diffuse (screen probes) system, the probe system, the lighting
  functions after them, and about twenty others (section 4). Making the
  predicate answer "no" switches each of them to its non-traced branch at
  once, which is what the task asks; what each of the others does on that
  branch is section 4's open list.
- **The switch**: `XTP_RT_PATH`, off by default (section 5). It makes the
  predicate and its twin (`0x4fd8570`, the same test on a given manager)
  answer 0 by a title-specific patch by RVA in the per-build hooks of
  `crates/xtp-game` (`rt_path.rs`). `fallback` does so for a whole run;
  `ab` alternates it with the title's own answer within one run.
- **The A/B (section 8): the fallback runs, does not brighten the scene,
  and stays opt-in.** In phase B, pass 17 and pass 73 run in every frame,
  and none of the 13 trace programs is dispatched any more: the particle
  collision, reflections, screen probes and local light shadows also stop.
  On the one view both arms show, the lit HDR before fog moves by +0.3 %
  (mean) and +0.08 % (P99), within what phase A drifts on its own. The
  irradiance atlas reads 0 in both arms. The fallback costs about 17 %
  more frame time (a world-only median of 249 ms against 213 ms). Nothing
  crashed or asserted.

## 2. The predicate

`0x4fd8550(i)` returns a byte, true when:

1. the ray-tracing manager exists (the 8-byte pointer at RVA `0x8527dc8`
   is not null), and
2. its scene `i` exists (entry `i` of the 8-byte table the manager points
   to at +0x08), and
3. either one of the manager's two counts at +0x9c and +0xac is not 0, or
   the object at +0xd8 exists and its word at +0x114 is not 0.

Every caller read so far passes `i = 0`.

**Its twin, `0x4fd8570(manager, i)`**, makes the same test (points 2 and
3) on the manager it is given. The wrappers that bind the structure for a
pass ask both (`0x4fd8630`, `0x4fd86a0`, `0x4fd8c10`: first `0x4fd8550`,
then `0x4fd8570` on the manager at `0x8527dc8`); the binding of the probe
system's structure (`0x200e810`) asks the twin alone, for scene
`1 + far`, where `far` is the byte at RVA `0x8527e02` (read by
`0x5154180`): the "far field" variant of each traced pass is chosen by the
same byte (pass `base + far`). Thirteen other call sites ask the twin
directly (`0x049be10`, `0x04e7570`, `0x06b6bd0`, `0x06b6ef0` two,
`0x15ccdc0`, `0x200e810`, `0x326d970`, `0x3598fe0`, `0x4fd8660`,
`0x4fd8740`, `0x4fd8c50`, `0x5154190`, `0x5374ff0`); the last asks it
about another manager (the pointer at RVA `0x84d5f40`), then hands over
to the predicate by a tail jump, for the scene `0x5154160` names. No
copy of the test made in line was found among the callers read.

## 3. The probe system's dispatch steps

The probe system's effect is loaded by name at start-up (`0x5708f50`), its
pointer kept at RVA `0x876c688` and its technique's index at `0x876c690`.
Each step binds a program as the effect's pass `first + k`, `first` read
from the technique's entry (the rule of title-cpu-image.md §2.3); in this
effect pass `k` is the effect's `k`-th program record, which every pass
number below confirms against the programs' roles. The steps are the 16
functions from `0x35744e0` to `0x3581fb0`; those that ask the predicate:

| Step | Predicate true | Predicate false |
|---|---|---|
| `0x35744e0`, primary probes' gather | binds the structure (`0x200e810`), pass 10 + far: `aeb93e0888250bd3` / `3592426e3e9142d8` | pass 17: `81560230fcde2b0a` |
| `0x35797d0`, irradiance cache's gather, then its shadow rays | the structure, pass 42 + far (`a74a0746279f960d` / `b58eb3ca6ee46427`); pass 61 + far (`5f13c9550dc4ab39` / `f607780aa8e502d0`) | neither dispatched |
| `0x357a8f0`, irradiance cache's shadow rays (second site) | pass 61 + far | not dispatched |
| `0x3581760`, transparent surfaces' probes | pass 68 + far: `3d121c93c7f9a057` / `36f04a4ed0ad1d9b` | pass 73: `29dc558346584906` |
| `0x3574bf0`, `0x357acb0` | move the traced passes' textures into the state those passes need | skip that |
| `0x3575930` | binds the structure (`0x5154190`) and the traced passes' inputs | skips them |
| `0x3580e20`, `0x3581fb0` (both also want the byte at `0x876aa10` to be 0) | the transparent surfaces' traced passes and their inputs | return before them |

The transparent surfaces' step also takes its fallback when the byte at
RVA `0x876aa10` is 1, whatever the predicate; one function writes it
(`0x0cd3b40`), not read yet.

## 4. The other callers

Call sites of `0x4fd8550`, by the `.pdata` function holding them:

- the indirect diffuse (screen probes) system, whose effect is at RVA
  `0x876a740` (loaded by `0x0cc4b00`): `0x3567d20`, `0x3568230`,
  `0x356b6b0` (two), `0x356bd00`, `0x356f2b0`;
- the probe system (section 3) and the steps after it: `0x35744e0`,
  `0x3574bf0`, `0x3575930`, `0x35797d0` (two), `0x357a8f0`, `0x357acb0`,
  `0x357b990`, `0x3580e20`, `0x3581760`, `0x3581fb0`, `0x3582a00`,
  `0x3583000`, `0x3584170`, `0x358df90`, `0x35915a0` (two), `0x3598fe0`,
  `0x359a320` (two);
- elsewhere: `0x027f8b0`, `0x027fec0`, `0x04e69f0`, `0x06b6cd0`,
  `0x06b6d50`, `0x06b6eb0` (two), `0x0a52d90`, `0x0ad53a0`, `0x0c4ce60`,
  `0x15b4ab0`, `0x16c8260`, `0x202f8d0`, `0x4fd82b0`, `0x4fd8630`,
  `0x4fd86a0`, `0x4fd8c10`, `0x5374ff0` (a call and a tail jump),
  `0x55798b0`.

What the screen probes do without the predicate is the first thing to
read: their fallback rays (`a953d5b93efe8424`) are refused (wave64 lanes
in a group over 64, rt-standin.md §3), so that hole would stay.

## 5. Forcing the fallback: `XTP_RT_PATH`

The cleanest switch the code allows: the entries of the predicate and of
its twin made to return 0 ("no scene ready"), so that every caller takes
the branch it takes before the title's ray-tracing scene exists. Nothing
is faked on the GPU side: the structure is still built (its builder's
dispatches are dropped as now), nobody asks for it. Code that reads the
manager's fields itself, without either function (the manager's own
functions, among the 37 references to its pointer), is not affected.

Written by task rthook (65fa56c1), 2026-11-25: `crates/xtp-game/src/rt_path.rs`,
applied from `apps/xtp/src/run.rs` once the image is mapped, before any
title code runs (next to the watchdog and assertion hooks).

- **Values**:
  - unset or `bvh`: nothing is written, the title answers itself;
  - `fallback`: both functions answer 0 for the whole run;
  - `ab`, `ab:<phase s>` or `ab:<phase s>:<settle s>`: the A/B clock of
    `xtp_core::ab` alternates the two. Phase A (`XTP_RT_PATH=0` in the
    A/B lines) is the title's own answer, phase B (`=1`) the fallback. It
    shares the phases of any other switch set to `ab` (`XTP_RT_STANDIN`,
    for one).
- **What is written**: each function's first two bytes become a short
  jump to the function's own "return 0". For the predicate that is
  `0x4fd8562`, the path it takes without a manager; for the twin,
  `0x4fd8581`, the path without scene `i`. Nothing else is written.
- **Why two bytes and not four**: the predicate's first instruction is two
  bytes long, the twin's four, and both functions are leaves (no frame, no
  unwind data). A jump that covers no more than the first instruction is
  one aligned, atomic two-byte store. While title threads run:
  - no thread is ever inside a written byte;
  - a thread that took the jump is in the title's "return 0", which is
    never written, so taking the jump out cannot strand it.

  A "return 0" of our own written over four bytes (two instructions) would
  break the second point. A thread stopped between those two instructions
  would resume, once the patch is taken out, in the middle of the
  predicate's second instruction. So `ab` can flip the path in a running
  session.
- **Checks before writing**: the build's BLAKE3, then an FNV-1a hash of
  each function's bytes from its entry through its "return 0" (0x15 and
  0x14 bytes). A mismatch writes nothing. The title's bytes stay outside
  the repository. The opt-in test `rt_path::tests::sites_in_the_sample_game`
  (`XTP_SAMPLE_GAME=<title folder>`) checks the hashes against a dump.
  On 2026-11-25 the image was mapped at `0x7ffa75f40000` (relocated) with
  `xtp run` without `--execute`. Both sites matched and were written:
  `ray-tracing scene predicate answers no … sites=2`.
- **In `ab`**: a thread of ours (`xtp-rt-path`) reads the clock every 5 ms
  and flips both entries after the present that ends a phase. That is not
  a frame boundary of the title, so the frame across a flip can see both
  answers. The phase's settling time (3 s by default) leaves that frame
  out of the measurements.
- **Log lines**:
  - `ray-tracing scene predicate answers no (XTP_RT_PATH=fallback)`;
  - `ray-tracing path (XTP_RT_PATH=ab): phase A …, phase B …`, then
    `ray-tracing scene predicate: <path> (A/B arm <A|B>)` at each flip;
  - a refusal says why: `ray-tracing path not switched (XTP_RT_PATH): …`
    (unknown build, bytes that are not the expected ones).
- **Expected**: in the probe and transparent probe passes, the fallback
  programs in the "first bind" lines and no drop of `3592426e3e9142d8`,
  `36f04a4ed0ad1d9b`, `b58eb3ca6ee46427` or `f607780aa8e502d0`; the
  irradiance cache's atlas stays 0 (no gather), so whether the light
  mixers (`ab619d0b7b7684d2`, `6a62009bc6a4146e`) still read it on that
  branch decides what the image gains.

## 6. Method and material

- **Strings and references**: the effects' names are plain strings in
  `.rdata`, each referenced once, by its system's start-up function; a
  RIP-relative reference scan of the code then gives every function using
  the effect's pointer (40 sites for the probe system).
- **Decompilation**: Ghidra 12.2 headless on a project imported without
  analysis (title-cpu-image.md §5), on the L4 VM at idle priority.
- **Queries**: `E:\xtp-data\re\scripts\ghidra\ReaFind.java` (strings with
  their references, references, direct callers, immediates) and
  `ReaQuery.java` (decompilation), driven from the PC by
  `E:\xtp-data\re\out\rtfallback\start-find.ps1` and `start-q.ps1`;
  outputs in `E:\xtp-data\re\out\rtfallback\vm\`; the probe effect's
  program records in order in `E:\xtp-data\re\out\rtfallback\ilp-records.tsv`.

## 7. What comes next

- **The hook**: written (section 5), `XTP_RT_PATH`, off by default.
- **The A/B on the same frames**, one Series S session with
  `XTP_RT_PATH=ab`: run by task rtab, section 8. The plan it followed was
  to record in each phase:
  - the "first bind" lines of the fallback programs;
  - the `rt: dropped traces in a frame` lines: the probe gathers and the
    cache's shadow rays should fall to 0;
  - in the pass dump, the probes' writes at pass 17 and the transparent
    probe volume's at pass 73, and the light mixers' inputs (non-zero?);
  - the irradiance cache on its own: its gather is not dispatched on this
    branch, so whether its atlas stays at 0 and whether the mixers still
    read it are measured apart from the probes;
  - the luminance chain, G-buffer → lit HDR → presented, as in
    world-black.md §10.1, and whether the street gets brighter.

  Upstream (watch 25, item 348): on an RTX 4070 SUPER both probe
  structures stayed at 0 over 553 readings, even with their address
  fixed. Nobody upstream has measured the forced predicate.
- **The default**: `fallback` becomes the default only if the A/B shows a
  clearly better image with no crash and no assertion; otherwise it stays
  opt-in, and this page says why. Section 8 ran it: no brighter image, so
  it stays opt-in.
- **How to confirm the path changed** (watch 23, items 328–330): no
  upstream test has shown the title's non-traced path running. The first
  write into the probe outputs by `81560230fcde2b0a`, then into the
  transparent probe volume by `29dc558346584906`, is the proof; the voxel
  volume is not expected to be written (section 1).
- **A darker room is possible**: on an RTX 5080, with the builder off,
  two ray passes darkened a room once the reflection tiles were made.
  Here the irradiance cache loses its gather on the fallback branch, and
  the screen probes' fallback rays are refused: the A/B says whether the
  light mixers' reads of the cache and of the screen probes make the
  image darker than with the traced passes dropped, and which callers of
  section 4 then need reading.
- **Section 4's callers**, starting with the screen probes and the
  lighting functions after the probe system, and the byte at `0x876aa10`.

## 8. The A/B: bvh against fallback in one session

Task rtab (ffc6d315), 2026-11-25, on the person's PC (RTX 4080 SUPER),
Series S, through the PC's FIFO queue (`queue-session.ps1`).

### 8.1 Set-up

- **Build**: `main` at `a5a7823` (the hook of section 5), plus this task's
  two fixes to `rt_standin.rs`, plus a measurement patch that stays outside
  the repository (`E:\xtp-data\runs\rtab\measurement-build.patch`). The
  patch is cxfog2's diagnostic `99e13fc` (world-black.md §10.1), widened
  to outputs of every width and to one-channel textures (the channel
  itself), so that the probes' and the irradiance cache's outputs are
  measured too.
- **Method**: cxfog2's, world-black.md §10.1. For each output:
  - the mean and the nearest-rank 99th percentile of
    `0.2126 R + 0.7152 G + 0.0722 B` over the dump's 24 sampled rows;
  - for the presented image, the same over every RGB pixel of the PNG
    with the dump's frame number, in code units.

  The analysis script (`E:\xtp-data\runs\rtab\rtab_measure.py`) gives
  back §10.1's frame 704 exactly: 0.06454895 / 0.27407169 before fog,
  0.03769958 / 0.15534353 presented.
- **Environment**:
  - `XTP_RT_PATH=ab:20:8`: phases of 20 s, the first 8 s of each
    settling;
  - `XTP_TEXTURE_POOL=1`, raw structured reads, intro skipped;
  - the autopilot to gameplay, then 90 s;
  - `XTP_CAPTURE_WORLD=2000:8:18:12`, `XTP_PASS_DUMP=0`;
  - `XTP_PASS_INPUTS=ab619d0b7b7684d2`: the light mixer, which binds the
    irradiance atlas.
- **Sessions**:
  - rtab-1 was built without the `diagnostics` feature, so it has no
    captures. It gives the frame time and the stability (run log
    `E:\xtp-data\cache\play\20261125-064144\run.log`).
  - rtab-2 gives the captures (`E:\xtp-data\runs\rtab-2`, run log
    `…\20261125-065136\run.log`, report
    `E:\xtp-data\runs\rtab\rtab-2-report.txt`).
- **Phases in rtab-2**, by present:

  | Phase | Arm | Presents |
  |---|---|---|
  | A0 | A | 1–560 (loading) |
  | B1 | B | 561–660 |
  | A2 | A | 661–734 |
  | B3 | B | 735–804 |
  | A4 | A | 805–898 |

  The first frame with 2,000 draws or more, 594, falls in B1. The dump of
  a listed frame `n` is `passes-<n+1>.txt` (its run-log drop line names
  `n`).
- **The builder, for phase A**: `main`'s. cxbvh's builder work (the
  `BVH builder admitted` / `held` lines) is not in this build, and no such
  line is in either log. refusals.md records the builder's programs as
  refused on `main` for addresses outside the root parameters. In these
  runs, 21 programs were refused for "an address not from a root
  parameter" (9 of them traces in rtab-2). So phase A measures the traced
  passes dropped, not lighting from a complete structure. cxbvh's
  witness of the normal path with the builder held (L4, frame 4096: a very
  dark world with a few light sources) is
  `E:\xtp-data\cloud-runs\20261125-072637-game`.

### 8.2 The path changes

- **The fallback programs run from the first world frame of B1**. The
  first bind of `81560230fcde2b0a` and `29dc558346584906` is at frame 594
  in rtab-2 and frame 578 in rtab-1. Every B dump with the probe system's
  passes (627, 639, 651, 759, 771, 783, 795) has pass 17 (1 of 1 op ran)
  and pass 73 (9 of 9); no A dump has either. Dumps 603, 615, 746 and 747
  (B) have none of the probe system's passes.
- **Phase B dispatches no trace program at all.** Each trace program's
  dropped dispatches, per dump, tied to it by its pipeline's address (run
  log `first bind`):

  | Program (role) | Each A dump (663, 699, 711, 723, 735, 807) | Each B dump (627, 639, 651, 759, 771, 783, 795) |
  |---|---|---|
  | `caf96a3229145fa6` (particle collision) | 30 to 70 | 0 |
  | `3592426e3e9142d8` (probes, gather) | 1 | 0 |
  | `36f04a4ed0ad1d9b` (transparent surfaces' probes) | 1 | 0 |
  | `b58eb3ca6ee46427` (irradiance cache, gather) | 1 | 0 |
  | `f607780aa8e502d0` (irradiance cache, shadow rays) | 2 | 0 |
  | `f22a76fcb967ee59` (screen probes) | 1 | 0 |
  | `6d7fe7745ab9c6d8` (reflections) | 2 | 0 |
  | `32ed072b9f845048` (local lights, tiled shadows) | 1 | 0 |
  | `559d9f0184e15fe3` (shadow mask) | 1 in 807 | 0 |

  `rt: dropped traces in a frame` is logged for every listed A frame and
  never for a B frame. Section 4's other callers (particles, reflections,
  screen probes, local lights) follow the predicate as well.

### 8.3 Probes, irradiance cache, lighting

Mean / P99, as defined in 8.1. "Settled" is past the phase's first 8 s,
by the time the run log gives for the dump's write.

- **Pass 17** (`81560230fcde2b0a`). It writes one 2304×576 one-channel
  16-bit float texture, and a second output without a host copy (not
  sampled). The texture holds 0, except a few per cent of texels at
  about 10000 (the mean is about 10000 times the share). These are the
  gather's values, not light.

  | B dump | Share not 0 | Mean / P99 |
  |---|---|---|
  | 627 | 1.4 % | 136.72 / 10000 |
  | 639 | 2.3 % | 234.38 / 10000 |
  | 651 | 2.3 % | 234.38 / 10000 |
  | 759 | 2.8 % | 283.20 / 10000 |
  | 771 | 3.0 % | 299.48 / 10000 |
  | 783 | 3.1 % | 309.24 / 10000 |
  | 795 | 4.0 % | 400.39 / 10000 |

  In A the gather (pass 10/11) is dispatched and dropped: nothing is
  written.
- **Pass 73** (`29dc558346584906`). It writes two 32×16 textures:

  | B dump | RGBA8 mean / P99 | R11G11B10 float mean / P99 |
  |---|---|---|
  | 627 | 0.529 / 0.847 | 0.041 / 0.169 |
  | 639 | 0.550 / 0.847 | 0.029 / 0.251 |
  | 651 | 0.553 / 0.847 | 0.032 / 0.332 |
  | 759 | 0.500 / 0.847 | 0.187 / 0.544 |
  | 771 | 0.513 / 0.847 | 0.168 / 0.484 |
  | 783 | 0.512 / 0.847 | 0.171 / 0.485 |
  | 795 | 0.527 / 0.847 | 0.144 / 0.552 |

  In A, pass 68/69 is dropped.
- **Irradiance cache, measured apart.** The atlas, 1280×2560, one 16-bit
  float channel, is 0 / 0 with no texel other than 0 in every dump of
  both arms, in two places:
  - after its copy, `c519f7a7946e3702`;
  - as the light mixer `ab619d0b7b7684d2` reads it (s0+832, both of its
    dispatches).

  The cache has no fallback gather (section 3), and its traced gather is
  dropped in A.
- **The mixer's other inputs** do not follow the arm:
  - s0+576, 128×128 RGBA16F, rises steadily through the run:
    - mean 0.00222 (627, B), 0.00255 (663, A), 0.00297 (735, A),
      0.00302 (759, B), 0.00312 (807, A);
    - P99 from 0.0325 to 0.0345.
  - s0+936, 256×256 RGBA16F, changes with the shot in both arms (mean 23
    to 88).
- **The deferred lighting `1cd9bc634417a981`** runs in one dump only,
  807 (A4, settling). P861 has 14 ops, of which 8 are indirect
  rectangles. It leaves the scene colour (1280×720 RGBA16F) at 0 / 0: red,
  green and blue are 0 on every sampled texel, and only alpha is non-zero,
  on 6.2 % of them. This is pixcap's
  finding (pix-world-frame.md §1), here in A; no B dump has a pass with
  this program.

  In every other dump, the material resolve (`f0c91b4d2578f5bf`) is
  followed by one other writer of the scene colour before the sky, in
  both arms: the compute pass `348e1e8c414c7eca` (11 dispatches).

### 8.4 The luminance chain

**The same view in both arms**: dumps 675, 687, 735 (A) and 747, 759 (B).
They show one still view behind a title card (the cards of the opening).
The presented image is the card, so it says nothing of the lighting.

| Dump | Arm | After `348e1e8c414c7eca` | Lit HDR before fog | After fog | Presented |
|---|---|---|---|---|---|
| 675 | A, settling (3.9 s) | 0.00089269 / 0.00189512 | 0.00090015 / 0.00189660 | 0.01104532 / 0.02612823 | 0.12561250 / 0.93893804 |
| 687 | A, settling | 0.00092638 / 0.00196731 | 0.00093811 / 0.00196982 | 0.01115747 / 0.02643230 | 0.11471583 / 0.14260314 |
| 735 | A, settled (A2's last frame) | 0.00095444 / 0.00202751 | 0.00096425 / 0.00203023 | 0.01067229 / 0.02655887 | 0.11453475 / 0.12888784 |
| 747 | B, settling (2.4 s) | no material resolve in the frame | 0.00096264 / 0.00202805 | 0.01124730 / 0.02717043 | 0.11453346 / 0.12888784 |
| 759 | B, settling (4.6 s) | 0.00095539 / 0.00202887 | 0.00096740 / 0.00203187 | 0.01122156 / 0.02735868 | 0.11625647 / 0.15181020 |

From 735 (A) to 759 (B), the lit HDR before fog changes by +0.3 % (mean)
and +0.08 % (P99). Over phase A alone, from 675 to 735, it rose by 7 %.
After fog, B is 5 % above 735 and 0.6–1.8 % above 675 and 687: within
A's own spread.

**The other dumps** show different shots of the opening, one per phase.
They are listed for completeness, not compared:

| Dump | Arm | Lit HDR before fog | After fog | Presented |
|---|---|---|---|---|
| 627 | B, settled | 0.00048831 / 0.00243747 | 0.00681797 / 0.03708506 | 0.00284508 / 0.02383294 |
| 639 | B, settled | 0.00001519 / 0.00001902 | 0.00722226 / 0.02994347 | 0.01434220 / 0.06008471 |
| 651 | B, settled | 0.00003419 / 0.00032891 | 0.00111442 / 0.00268275 | 0.00125801 / 0.00392157 (black) |
| 663 | A, settling | no fog in the frame (sky pass 0.09319670 / 3.81782937) | – | 0.04710535 / 0.16118588 |
| 699 | A, settled | no fog in the frame (sky pass 0.00892851 / 0.01154337) | – | 0.06214589 / 0.21802902 |
| 711 | A, settled | 0.18063104 / 6.22330713 | 0.03379093 / 0.11904912 | 0.09673241 / 0.41109333 (street, poles and wires) |
| 723 | A, settled | 0.00725480 / 0.00862142 | 0.03280483 / 0.11563386 | 0.11455917 / 0.12972157 |
| 771 | B, settled (8.7 s) | 0.00707753 / 0.01100019 | 0.00989981 / 0.03611754 | 0.01383704 / 0.11510353 |
| 783 | B, settled | 0.00323944 / 0.01113751 | 0.01072787 / 0.04895948 | 0.01933869 / 0.12257098 (dark street, lights) |
| 795 | B, settled | 0.00146415 / 0.01169774 | 0.00721895 / 0.05063149 | 0.01232025 / 0.07003765 |
| 807 | A, settling | 0.00166691 / 0.00558832 | 0.01136596 / 0.04046361 | 0.11453461 / 0.12888784 |

The opening cuts from shot to shot, and no still shot of the world spans
a flip with both sides settled. So the only same-frame comparison is the
one above, and its B side is still settling.

### 8.5 Stability and frame time

- **Stability**: both sessions ran to the autopilot's `exit after` (title
  exit code 0) through five phases, four flips in the world:
  - no crash;
  - no title assertion (one would end the process, asserts.rs);
  - no panic and no ERROR line in either log.
- **Frame time**, rtab-1 (no dumps), world phases only (A0 was loading,
  at about 29 fps):

  | | A (A2, A4) | B (B1, B3) | B/A |
  |---|---|---|---|
  | Settled presents | 126 in 23.9 s, 5.27 fps | 94 in 23.7 s, 3.97 fps | −25 % |
  | Frame lines past settling, median of 4: fps | 4.7 | 4.05 | −14 % |
  | ms/frame | 213.1 | 249.2 | +17 % |
  | ms on the GPU | 161.0 | 190.7 | +18 % |
  | Longest frame, per phase | 300, 314 ms | 334, 333 ms | |

  Each phase shows another shot of the opening, so part of the difference
  is the shot. Phase B also runs work that A drops: pass 17 and pass 73
  run, while their traced passes in A are refused. xtask's own summary
  reads the same eight lines but takes the upper of the two middle
  values: A 4.7 fps and 213.3 ms, B 4.2 fps and 258.4 ms (+21 %).

  rtab-2's phase rates (with dumps): B1 3.47, A2 3.27, B3 2.98,
  A4 4.49 fps.

### 8.6 The decision: `fallback` stays opt-in

The task's rule: the fallback becomes the default if it brightens the
scene and breaks nothing. It breaks nothing seen. It does not brighten:
- on the shared view, the lit HDR moves 0.3 % against a 7 % drift within
  phase A;
- the irradiance atlas and the mixer's read of it are 0 in both arms;
- no measured output after the probes follows the arm.

It also costs about 17 % of frame time. `XTP_RT_PATH` stays off by default
(unset is the title's own answer), and `rt_path.rs` says so.

Why it does not brighten. These are leads, not proven:
- the irradiance cache has no fallback (section 3);
- in A, the deferred lighting by rectangles writes 0 where it runs
  (pix-world-frame.md); in B it was not seen at all;
- what reads pass 17's 2304×576 output, and its value of about 10000, was
  not traced;
- the screen probes' fallback rays are refused (section 4).

What would settle it:
- the same A/B once cxbvh's builder lands (phase A with a structure);
- a still gameplay view held for two settled phases, instead of the
  opening's cuts;
- the readers of pass 17's output.
