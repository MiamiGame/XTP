# One world frame in PIX: where the light goes and where meshes vanish

Task pixcap (eccb80f1), 2026-11-25, on the person's PC (RTX 4080 SUPER,
Series S mode). One world frame of `main` captured with PIX for Windows
2611.19 and walked pass by pass, against the person's reference for the
start of the game: night, but you see well; street lamps light the street
and the pavement, the facades take a teal and green ambient light,
characters are lit from the side, palm trees, signs and cars are readable.
Programs are named by hash only; captures, tables and screenshots stay
under `E:\xtp-data\pix\`, none is in the repository.

## 1. Verdict

Measured on frames 805 and 806 of session `world-1` (about 60 frames into
the world, the opening text on screen) and on frame 1007 of session
`world-2` (about 400 frames later, another shot of the opening, §9).

- **The light disappears at the deferred lighting** (frame 805's P835,
  806's P733, 1007's P741): 232 indirect rectangle draws (183 pixel
  programs, `1cd9bc634417a981` among them), all of them run, write the
  RGBA16F scene colour with blending off, and leave it **0 on every texel**
  in frame 805, and on 91 % of it in 1007 (the 8.7 % left at a fiftieth of
  what the material resolve wrote there). The pass before it, the material
  resolve `f0c91b4d2578f5bf`, had written exactly the texels the geometry
  covers. Three more lighting passes (434 draws) add nothing. What reaches
  the screen afterwards is the sky (behind the geometry), the fog and the
  lights' own sprites and coronas drawn later: the person's "only light
  points and coronas" (§5).
- **The lighting has almost nothing to light.** After the G-buffer
  (mesh-shader draws, 97 % of its ops run) the main depth covers **11.8 %**
  of the screen in frame 805, **18 %** in 806, **17 %** in 1007; the depth
  pre-pass alone, **8.3 %**. The albedo target holds a non-zero colour on
  **3.6 %** of the texels in 805 (mean 0.0005), 9.4 % in 806, and **0 %**
  in 1007. The street's ground, the facades and the main character are not
  in the depth: they are missing, not dark (§6).
- **Where meshes vanish (lead, not proven).** Between the depth pre-pass
  and the G-buffer, every frame drops almost every dispatch of two
  programs: `48426bae580a080c` runs 1 of 922 dispatches in frame 1007 (1 of
  1,186 in 806), `969c1cef8b783b39` 1 of 139. The 921 dropped ones go
  through another pipeline object, which has no host pipeline 400 frames
  into the world, and the same happens without PIX (cxfog2's sessions: 1 of
  153 and 1 of 235). They sit where a GPU-driven renderer culls and prepares
  the mesh draws that follow; the G-buffer's indirect mesh dispatches take
  their counts from the GPU (§6).
- **The albedo of the materials drawn is about 0.** Frame 806's first
  G-buffer pass (P351, 285 draws, 59 programs) writes its albedo with a
  maximum of **1/255** in red, green and blue while its other targets hold
  normal values; later passes write some real albedo (up to 0.44); in frame
  1007 none does. The four main pixel programs of P351 take no integer
  through an interpolated attribute (checked in their DXIL, §7).
- **Dropped work.** Frame 805 drops 4,848 of its 16,321 draws and
  dispatches, 806 drops 1,972 of 13,562, 1007 drops 1,340 of 10,567: in
  frames 805 and 806 mostly pipelines still being made, in 1007 mostly the
  921 of `48426bae580a080c`'s pass and 236 indirect dispatches (§6.3).
- **Ray tracing:** each frame drops its local-light shadow traces (2
  programs), the shadow cache, the probe and irradiance gathers (3), the
  screen probes, the reflections and the particle collision (§5.3): the
  teal and green ambient of the reference would come from those probes.
- **Exposure.** The auto exposure the post passes read climbs to about
  **214** (inverse 0.0047) in both sessions: a scene with no lit geometry
  is pushed up until the sky saturates; frame 1007 reaches the screen white
  and grey (§9).
- **Tools.** PIX captures XTP; PIX's replay needs Windows Developer Mode,
  which is off on the PC (the person's decision). RenderDoc captures XTP
  too, but both of its replays of a world frame hung the GPU. The values
  here come from XTP's own pass dump of the same frames (`XTP_PASS_DUMP`),
  the structure (every call, pipeline state, shaders) from the PIX capture
  (§2).

## 2. How the frame was captured and read

- **Build:** `main` at `10e7f2d` (be56d45 plus the fog fix and the
  ray-tracing drop log, its stand-in off), diagnostics build.
- **Session:** `tools/pix/capture-world.ps1 -Name world-1 -WorldDraws 2000
  -WorldAfter 60 -WorldCount 4 -Settle 0 -EnvPairs
  'XTP_PASS_DUMP=0;XTP_PASS_INPUTS=f0c91b4d2578f5bf'`. It waits for the
  PC's game slot as `session.ps1` does (`E:\xtp-data\measure.lock`, no
  game `xtp.exe`), copies the golden data, runs the precook, and starts
  `xtp.exe run … --autopilot autopilot/gameplay.txt --console series-s`
  through `xtp-pix capture`, which launches it for GPU capture with the PIX
  API (`PixApi.h`, PIX 2611). Environment: `XTP_SKIP_INTRO=1`,
  `XTP_RAW_STRUCTURED=1`, Series S, and `XTP_D3D12_DRED=1`, so that XTP
  names its host textures and buffers and puts a marker before each draw
  and dispatch with its program hashes (`draw.rs`, `op_text`): that is how
  a PIX event is tied to a program. `XTP_CAPTURE_WORLD=2000:60:4:1` dumps
  four frames, starting 60 frames after the first frame with 2,000 draws;
  as soon as the first one's PNG appears, `xtp-pix` takes a one-frame
  capture (present to present) and ends the title.
- **The PIX frame is frame 806 of the dump.** Its draws and dispatches
  match frame 806's ops kind by kind (688 mesh dispatches, 3,668 indirect
  mesh dispatches, 2,041 dispatches, 799 indirect dispatches, 256 indirect
  rectangle draws, 200 draws, 78 indexed draws, 142 ray-tracing drop
  markers): the capture has XTP's own measurement of every pass of the
  same frame. Frame 806's dump ran out of its read-back budget after its
  G-buffer, so the light chain is read in frame 805, the same shot one
  frame earlier, complete.
- **PIX's replay is not available on this PC.** Every PIX analysis
  (`xtp-pix walk`, and `pixtool open-capture … save-resource` alike) stops
  with "This feature requires developer mode to be enabled on the analysis
  device" (`E_PIX_FEATURE_REQUIRES_DEVELOPER_MODE`). What PIX gives without
  a replay: every call of every queue with its arguments (`xtp-pix
  events`), the views and pipeline state bound at each draw and dispatch
  (`xtp-pix walk --no-replay`), and the shaders of any pipeline state
  object (`xtp-pix pso`, then `dxc -dumpbin`).
- **RenderDoc** (1.47, `E:\tools\gpu\RenderDoc`) captures XTP through
  `xtp-rdoc` (frame 864 of session `world-rd1`, 2.3 GB), but both replays
  on the RTX 4080 ended in `DXGI_ERROR_DEVICE_HUNG`; a replay on WARP needs
  more memory than the PC has free beside a session. No replay was tried
  again.
- **Checked on a known frame:** `tools/pix/selftest.ps1` renders, on WARP
  only, a draw of a constant colour into an RGBA16F target with depth, and
  a dispatch that reads it and writes 4 × red into an R32F texture. Both
  tools capture it; `xtp-rdoc` reads back 0.294125 (the colour's
  luminance), depth 0.25 on all 32,768 texels, the dispatch's input at
  0.294125 and its output at 2.0, exactly; `xtp-pix` passes its capture
  and its walk without replay, and reports its replay as skipped.
- **Numbers.** "Non-zero" is the dump's "lit": the share of sampled texels
  with a channel other than 0 (24 rows across each target). Luminance is
  0.2126 R + 0.7152 G + 0.0722 B, means over all sampled texels. Formats are
  DXGI's: f10 RGBA16F, f19 D32S8, f27/f28 RGBA8, f44 R24G8 (the material
  class depth), f87 BGRA8, f24 RGB10A2.

## 3. The frame

- One direct queue, 305 command lists in 35 submissions, 141,660 events:
  14,524 draws and dispatches (8,023 dispatches, 4,791 `ExecuteIndirect`,
  1,105 `DispatchMesh`, 466 draws, 139 indexed draws), 25,676 barriers,
  16,550 texture copies.
- The world's geometry is drawn by mesh shaders: 688 direct mesh
  dispatches and 3,668 indirect ones (XTP's `ExecuteIndirect` of up to 33
  records with a GPU-written count). Of the indirect ones, about 1,500 draw
  shadow maps (R16 depth, 256² to 1536²), 179 the depth pre-pass into the
  main depth, 269 the G-buffer; 1,670 bind targets the event list alone
  does not show.
- The G-buffer: seven 1280×720 targets (albedo RGBA8 typeless, five RGBA8,
  one RG16F) on the D32S8 main depth, written by direct and indirect mesh
  dispatches, nearly all with depth writes off after the pre-pass.
- The deferred material and lighting path: tile classification compute,
  the material resolve `f0c91b4d2578f5bf` (one 160×90 dispatch), then the
  lighting as indirect rectangle draws tested EQUAL against the material
  class depth (R24G8).

## 4. The passes (frame 805)

| Pass | What | Ran | Output (1280×720 unless said) | Non-zero | Mean |
|---|---|---|---|---|---|
| P52, P58 | depth pre-pass (165 ops, mesh) | all | main depth | 8.3 % | 0.083 |
| P474–P548 | G-buffer (12 passes of mesh draws, and the passes among them) | 1,767 of 1,824 ops | main depth / albedo | 11.8 % / 3.6 % | 0.084 / 0.0005 |
| P552–P680 | decals and material-class passes | all | albedo, depth | unchanged | |
| P707 | material resolve `f0c91b4d2578f5bf` | 1/1 | scene colour RGBA16F | 11.8 % | 0.0003 |
| P707 | (same) | | packed material BGRA8, RGBA8 | 100 % | 0.55–0.66 |
| P902 | half-resolution buffer (640×360 R11G11B10), 186 indirect dispatches | 25 of 186 | | 0.0 % | 0 |
| **P835** | **deferred lighting, 232 rectangles** | **232/232** | **scene colour** | **0.0 %** | **0** |
| P843, P844, P846 | more lighting (235, 171, 28 draws) | all | scene colour | 0.0 % | 0 |
| P889 | sky `348e1e8c414c7eca` (11 dispatches) | all | scene colour | 88.2 % | 0.002 |
| P890 | sky dome draws | all | scene colour | 88.2 % | 0.002 |
| P896 | fog reconstruction (the two programs fixed in `e1b75d7`) | 2/2 | 480×272 | 85–100 % | |
| P904 | fog application `577248728544d74b` | 1/1 | scene colour | 100 % | 0.005 |
| P909–P924 | draws after the fog | all | scene colour | 100 % | 0.005 |
| P954 | final composite `731a39171b568298` | 1/1 | scene colour | 100 % | 0.005 |
| P959 | FidelityFX stand-in (bilinear) | 1/1 | 1920×1080 RGBA16F | 100 % | 0.005 |
| P1022 | tone mapping | 42/42 | 1920×1080 RGB10A2 | 64 % | 0.008 |
| P1026 | scene plane | 1/1 | 1920×1080 RGBA8 | 58 % | 0.008 |
| P1028 | interface plane (opening text) | 63/63 | 3840×2160 RGBA8 | 16 % | |

Two 16-byte buffers are read as exposure (exposure, its inverse, the
bounds): the lighting draws read one holding [1.17, 0.85, 8, 0.125]; 17
other reads in the frame get [213.7, 0.0047, 8, 0.125] (216.4 in frame
1007).

## 5. Where the light disappears

### 5.1 The lighting pass

- The 232 draws of P835 are indirect rectangle draws (XTP's expansion,
  `rectangles.rs`), vertex `25ebc92075b62822` and 183 pixel programs, one
  per material class and light type. Their pipeline state, from the PIX
  capture (object 20525, pixel `1cd9bc634417a981`): **blending off**
  (the result replaces the target), two targets **RGBA16F** (the scene
  colour) and **R32_UINT**, depth **D24S8** tested **EQUAL** with writes
  off, no culling. Format and blend are what the title asks; nothing
  between P707 and P835 clears or copies into the scene colour.
- So the draws cover the pixels and compute 0 there. They read
  (slot descriptions in the dump): the G-buffer (albedo, three RGBA8
  targets), the depth and a linear depth, the per-tile light lists (10×6
  R8_UINT, R32_UINT, R8), two 1280×720 R9G9B9E5 views (the light textures
  of `world-black.md` §5), a 640×360 R16 target, light probes (384×384,
  256×256 RGBA16F), a 2048×1824 R8_UINT texture, and the title's 1×1
  textures filled by the CPU, which hold (0, 0, 0, 1).

### 5.2 The material resolve's inputs (frame 805, read just before P707)

| Slot | Texture | Non-zero | Mean (RGB) |
|---|---|---|---|
| s0+432, s0+880 | main depth | 11.8 % | 0.084 |
| s0+992 | linear depth R32F | 11.8 % | 0.084 |
| s0+584 | albedo | 3.6 % | 0.0005 (max 0.46) |
| s0+752 | G-buffer RGBA8 | 3.8 % | 0.0001 (max 0.13) |
| s0+1000, s0+1168 | G-buffer RGBA8 | 4.0 % | 0.02 |
| s0+1264 | G-buffer RGBA8 | **0 %** | 0 |
| s0+872 | normals | 100 % | 0.5 (empty normal) |
| s0+416, s0+664, s0+1312 | 1×1, CPU-filled | 100 % | (0, 0, 0, 1) |
| s0+488, s0+1016 | 1×1 UINT, CPU-filled | 100 % | (0, 0, 0, 255) |

The material resolve reads what the G-buffer holds: geometry on 11.8 % of
the screen, albedo on 3.6 %.

### 5.3 Ray tracing in the frame

Frame 806's drop markers name the traces XTP cannot run (`rt_standin.rs`):
local-light shadows (`32ed072b9f845048`, `559d9f0184e15fe3`, writing
1.4 MB and 2.1 MB masks), the shadow cache (`f607780aa8e502d0`), the probe
and irradiance gathers (`3592426e3e9142d8`, `36f04a4ed0ad1d9b`,
`b58eb3ca6ee46427`), the screen probes (`f22a76fcb967ee59`), the
reflections (`6d7fe7745ab9c6d8`) and the particle collision
(`caf96a3229145fa6`, 82 dispatches). Their outputs are buffers. rtdark's
A/B (task 558d346d) found its default stand-in without effect and the
irradiance caches at 0, read by the light mixers.

## 6. Where meshes vanish

### 6.1 Coverage

The depth pre-pass leaves 8.3 % of the main depth written, the G-buffer
11.8 % (805) or 18 % (806); the albedo 3.6 % or 9.4 %. The view (a street
with poles, wires, awnings and distant lights) should have ground on most
of the lower half and facades on both sides. What the depth holds is what
the presented image shows: poles, wires and a few props.

### 6.2 The draws

The G-buffer draws are in the frame and run: 1,767 of 1,824 ops in frame
805's G-buffer passes, 285 of 302 in 806's first one. In the PIX capture,
590 draws and dispatches bind the main depth; 204 of them have depth
writes on (the pre-pass and the first G-buffer pipelines), the others test
it with writes off. The draws that are missing are not
dropped draws: they are mesh dispatches that draw little. Their counts
and arguments are written on the GPU by compute before them (XTP's
`ExecuteIndirect` reads up to 33 records and a GPU-written count), which
a capture shows only through a replay.

### 6.3 What is dropped

| Frame | Ops | Dropped | Of which |
|---|---|---|---|
| 805 | 16,321 | 4,848 | 4,300 dispatches and 531 indirect dispatches without a host pipeline (being made), 16 indirect mesh dispatches |
| 806 | 13,562 | 1,972 | 1,424 dispatches, 523 indirect dispatches, 24 indirect mesh dispatches, 1 indexed rectangle draw |
| 1007 | 10,567 | 1,340 | 1,080 dispatches, 236 indirect dispatches, 23 indirect mesh dispatches, 1 indexed rectangle draw |

- **The pass before the G-buffer.** In frame 1007, P34 (list 7, after the
  depth pre-pass, before the G-buffer) runs one dispatch of
  `48426bae580a080c` (3997×1×1) and drops 921 that go through another
  pipeline object, with no host pipeline 400 frames into the world; P38
  runs 1 of 139 of `969c1cef8b783b39`. Frame 806 has the same at P319 (1 of
  1,186) and P324 (1 of 205); so do cxfog2's sessions without PIX
  (`c58683-luma1` frame 704: 1 of 153; `c58683-luma-ctl1` frame 645: 1 of
  235). Both programs translate (the RenderDoc session's log binds them at
  frames 801 and 810): what is missing is the other object's pipeline. The
  dispatch of `48426bae580a080c` that runs reads a 998,400-byte placed
  buffer and writes a 1.5 MB buffer; the G-buffer's indirect mesh
  dispatches read their records elsewhere, so the link from these drops to
  the missing meshes is a lead, not shown.
- **Pipelines being made.** The session `world-rd1`'s log (up to frame
  864) counts, per frame, 392 indirect dispatches dropped "being made
  (heavy)", 79 that do not translate, 32 indirect mesh dispatches that do
  not translate; its refusals are led by "a structured buffer of unknown
  stride" (31 compute programs; `XTP_STRIDE_TABLE`, off by default) and "an
  address not from a root parameter" (25, the traces of §5.3).

## 7. The albedo of the first G-buffer pass

- Frame 806, P351 (285 draws, mostly indirect mesh dispatches): albedo
  maximum (0.0039, 0.0039, 0.0039), that is 1/255, alpha 0.502; one
  RGBA8 target RGB 0, another's green at most 1/255; the remaining
  targets normal (maximum 1.0). P417 and P419 (pixel programs
  `63051e97862e7dad`, `57a27c5e2e1d62a9` among others) raise the albedo
  to a maximum of 0.44. The pipelines write the albedo through an
  R8G8B8A8_UNORM_SRGB view of the typeless texture, as the title asks
  (PSOs 19145 and 19387 alike), and the dump reads its bytes: 1/255 there
  is a linear albedo of at most 0.0003.
- The pixel programs of P351 with the most draws: `09eb0a73df63b87d` (32),
  `d42a2d8a9d2c954a` (24), `6590393ed93ff071` (21), `94d281151ca58b4f`
  (18). Their translated DXIL, taken from the capture (`xtp-pix pso`,
  pipeline objects 19145, 19140, 19152, 19126): every pixel input is a
  float; the one read as a constant (`TEXCOORD17`) is declared
  `nointerpolation`; no interpolated input reaches integer arithmetic or a
  heap index (the `and`/`or` on two inputs of 19126 are XTP's lowering of
  the console's min/max, its NaN and signed-zero handling).
- `09eb0a73df63b87d`'s albedo depends on its interpolated `COLOR2` (four
  uses) and `COLOR1`; a program that writes real albedo (`63051e97…`)
  reads neither `COLOR2` nor `TEXCOORD10`. Its mesh shader computes
  `COLOR2` with multiply-adds of values read from buffers, 0 on one path
  (a `phi` with 0, taken only by threads without a vertex). Next: the
  texture and buffer inputs of that program in a late frame (§10).
- **The view heap's mirror (watch item 181).** The buffers `COLOR2` is
  computed from are taken from the heap at `(u9 + 128) >> 5`, `(u9 + 160)
  >> 5` and nearby (and `(u8 + 160) >> 5`). In frame 806 the program's u9
  is 0x23d8260: slots **1,174,547 to 1,174,555, past the 737,856 slots XTP
  mirrors** of the title's 1,245,184, so they read zeros. Counting every
  mesh dispatch by where its u9 table falls:

  | Frame | Pass | Ops past the mirror | Depth after | Albedo after |
  |---|---|---|---|---|
  | 806 | P62, depth pre-pass | 180 of 180 | 8.3 % | |
  | 806 | P351, first G-buffer pass | 222 of 222 | 18.0 % | 7.7 %, at most 1/255 |
  | 806 | P353, P354, P356, P364, P370 | 0 of 324 | unchanged | unchanged |
  | 806 | P358 | 140 of 140 | unchanged | unchanged |
  | 806 | P417 | 0 of 124 | unchanged | 9.3 %, up to 0.44 |
  | 805 | P58, depth pre-pass | 16 of 157 | 8.3 % | |
  | 805 | P474, first G-buffer pass | 0 of 184 | 11.8 % | 3.2 %, about 0 |
  | 1007 | P26, depth pre-pass | 99 of 99 | 16.3 % | |
  | 1007 | P462, P480 | 0 of 66 | unchanged | 0 % |

  Draws past the mirror read null descriptors, and a draw that the depth
  pre-pass missed adds nothing afterwards (the G-buffer tests depth EQUAL).
  But passes wholly inside the mirror (805's P474, 1007's P462 and P480)
  leave the albedo at 0 too: the mirror explains part of the black, not
  all of it.

## 8. Tools and commands

In the repository, `tools/pix/`:

| Script | What |
|---|---|
| `capture-world.ps1 -Name N [-With pix\|renderdoc] [-EnvPairs …] [-NoPrecook] [-Detach]` | one PC session to a captured world frame, under the slot lock, then the walks listed in `<run>\walks.txt` |
| `build.ps1` | builds the clients below from `E:\xtp-data\pix\src` into `E:\xtp-data\pix\bin` (Visual Studio C++ tools, PIX 2611, RenderDoc 1.47, whose API headers it fetches at the DLL's commit) |
| `selftest.ps1` | both clients on a known WARP frame |

The clients are C++ and two summaries Python, file types the repository's
policy (`xtask/src/policy.rs`) does not admit yet; until the person
decides, their sources stay in `E:\xtp-data\pix\src`:

| Source | What |
|---|---|
| `xtp-pix.cpp`, `walk.inl`, `texels.h` | PIX API client: `capture` (launch for GPU capture, trigger on an `XTP_CAPTURE_WORLD` PNG, one-frame capture), `events` (every call), `walk` (per-pass statistics with a replay; `--no-replay`: views and pipeline state), `pso` (a pipeline's shaders, for `dxc -dumpbin`) |
| `d3d12-newer.h` | stand-ins for five types of a newer `d3d12.h` that the PIX headers name in parts the client does not use |
| `xtp-rdoc.cpp` | RenderDoc client: `capture` (injection and target control), `walk` (replay API) |
| `selftest.cpp` | the WARP frame of known values |
| `events.py`, `passes.py` | an event list by call kind and XTP program; a walk table as a pass list |

The sessions' files: `E:\xtp-data\pix\world-1\` (`world.wpix`,
`events.tsv`, `views.tsv`, `frames\passes-80{4,5,6}.txt`, `pso\`),
`E:\xtp-data\pix\world-2\` (`frames\passes-100{6,7}.txt`, a later
`world.wpix`), `E:\xtp-data\pix\world-rd1\` (`world.rdc`, the title's log
in `capture.out.txt`).

## 9. What a later frame says

Session `world-2`: the same build, no precook,
`XTP_CAPTURE_WORLD=2000:400:2:1` (two frames dumped 400 frames after the
first frame with 2,000 draws). Frame 1006 is a transition (2,582 ops);
frame 1007 another shot of the opening, letterboxed, which reaches the
screen white and grey.

| Pass | What | Ran | Output | Non-zero | Mean (RGB) |
|---|---|---|---|---|---|
| P20, P26 | depth pre-pass | all | main depth | 8.3 %, 16.3 % | |
| P34, P38 | `48426bae580a080c`, `969c1cef8b783b39` | **1 of 922, 1 of 139** | buffers | | |
| P457–P566 | G-buffer | 115 of 128 in P457, all after | main depth / albedo | 17.3 % / **0 %** | |
| P594 | material resolve | 1/1 | scene colour | 17.2 % | (1.15, 1.68, 2.34) |
| **P741** | **deferred lighting, 232 rectangles** | **232/232** | scene colour | **8.7 %** | (0.024, 0.037, 0.059) |
| P784 | sky | 11/11 | scene colour | 91.4 % | (3.66, 8.02, 13.04) |
| P801 | fog | 1/1 | scene colour | 99.9 % | (5.21, 8.30, 13.08) |
| P847 | final composite | 1/1 | scene colour | 99.9 % | (5.79, 9.76, 15.80) |
| P915 | tone mapping | 36/36 | 1920×1080 RGB10A2 | 100 % | (0.68, 0.71, 0.72) |

The exposure read by the post passes is 216.4 (inverse 0.0046). The
material resolve's inputs, read just before it: albedo 0 %, the G-buffer's
RGBA8 targets 0 % to 1 %, depth 17.3 %, the CPU-filled 1×1 textures
(0, 0, 0, 1). The picture of frames 805 and 806 holds 400 frames later:
geometry on a sixth of the screen, no albedo, the lighting removing light
rather than adding it, and the sky, multiplied by the exposure, making the
image.

## 10. Next

- The lighting's inputs just before its draws (session `pixcap-light-1`,
  `XTP_PASS_INPUTS=1cd9bc634417a981`, 400 frames into the world, queued):
  which of the light lists, the R9G9B9E5 light textures, the probes or the
  shadow masks reads 0.
- The program of the pipeline object behind the 921 dropped dispatches of
  P34, and why it has no host pipeline (the same session's log).
- `COLOR2` of mesh program `db4835a364449a45` (task e404dd85, flowfix).
- PIX's replay, if the person turns on Windows Developer Mode: the captures
  above replay as they are, with per-draw depth and inputs (`xtp-pix walk
  --accessed --inputs --depth-diff`).

## 11. The black at its source, in two live sessions

Task imgsrc (8724fde2), 2026-11-25, on the person's PC (RTX 4080 SUPER,
Series S mode). Two world sessions of this task's branch (`main` f74e985
plus the probe below, diagnostics build): `imgsrc-1` (07:58) and
`imgsrc-2` (08:07), each through `E:\tools\stall-probe\queue-session.ps1`
with `XTP_TEXTURE_POOL=1`, `XTP_CAPTURE_WORLD=2000:60:4:1`,
`XTP_PASS_DUMP=0`, `XTP_FRAME_PROBE` and `XTP_PASS_INPUTS` (the lighting
pixel `1cd9bc634417a981` in the first, the lighting compute
`34fb51b8a0312e82` in the second). Their files stay in
`E:\xtp-data\pix\imgsrc-1\` and `imgsrc-2\`; the title's logs in
`E:\xtp-data\cache\play\20261125-065811\` and `20261125-070757\`.

### 11.1 The probe

`XTP_FRAME_PROBE=<program>[,…]` (`crates/xtp-gpu/src/frameprobe.rs`,
diagnostics builds) writes `probe-<n>.txt` beside the pass dump for each
listed frame. For every op of a listed program, and every descriptor its
stage reads through a table:

- the table's byte offset as the title gave it (its user data), the title
  slot, whether it lies past the 737,856 slots the host heap mirrors, and
  the host slot the shader reads instead when `draw.rs` makes the table
  afresh in dynamic slots;
- the view the title wrote there, and for a mirrored slot the view the
  host slot was made from;
- for a buffer view: the memory there (private, or a mapped view of
  physical pages, and at how many other addresses its 64 KiB page shows),
  the host copy it resolves to, and its bytes three times: title memory
  when the op is recorded, title memory when its list is submitted (what
  the refresh copies), and the host copy on the GPU just before the op (a
  copy recorded in the list where the op stands);
- for a small texture the title placed (the 1×1 ones): its first texels in
  title memory.

Each frame also gets the order of the R9G9B9E5 decodings against their
writers (upstream watch item 360, §11.4). The title records a frame's lists
while the frame before is presented, so the probe reports a frame once
every list it probed has run; `probe-<n>` holds the ops whose lists the
pass dump reports as `passes-<n+1>` (same op counts).

### 11.2 Albedo: `COLOR2` is read as the title writes it

**Heap index.** The mesh `db4835a364449a45` reads `COLOR2`'s word through
`(u9 + 128) >> 5`, that is title slot `u9 / 32 + 4` (its table's entry
`s1+32`; the index word it uses comes from `s1+64`, `(u9 + 256) >> 5`).

| Frame | Draws of the mesh | `s1+32` past the mirror | Table made afresh |
|---|---|---|---|
| 805 (world-1) | 34 | 0 | (not probed) |
| 806 (world-1) | 38 | 32 | (not probed) |
| 673 (imgsrc-1) | 9 | 1 | 9 |
| 676 (imgsrc-1) | 39 | 13 | 39 |
| 660 (imgsrc-2) | 9 | 9 | 9 |
| 663 (imgsrc-2) | 41 | 30 | 41 |

Every probed draw's `u9` table is made afresh in dynamic slots (it holds
constant buffer views, `s1+88`, which `draw.rs` `State::tables` always
makes afresh, and a table reaching past the mirror is made afresh too),
and the shader gets the new offset as `u9` (host slots 761,680 to
945,562). So the draws past the mirror do not read null descriptors
there, and §7's line saying so does not hold for table reads: watch item
355 (VIctory #2121) does not explain `COLOR2`. It still matters for slots
shaders compute without a table.

**The view and the copy.** `s1+32` is a placed buffer view (the title's
"view of memory at an address") of a mapped view of physical pages: for
example 64 bytes at `0x3e15bac0` (a page shown at no other address) or 768
bytes at `0x735fa5a40` (a pool whose pages show at two other addresses).
XTP reads it from the host copy of the title buffer holding that address.
In `imgsrc-2` every probed draw ran and was read back: for all 60 draws of
frames 660 to 663, the GPU's bytes of `s1+32` and `s1+64` just before the
draw equal title memory at submission, which equals title memory at
recording: the host copy is up to date, and the GPU reads what the title
holds there.

**The bytes.** Each 64-byte element is an instance record: a 3×4 matrix,
1.0, the word `COLOR2` decodes (offset 52), 1.0, a packed word. The word
holds real colours with alpha 255 on some draws (`ff070e14`, `ff165b44`,
`ff2e567a`, `ff415f7f`, `ff182d41`), `00ffffff` (white, alpha 0) on the
characters' draws, and `0n000000` (n from 0 to 3) on most instanced draws
of frames 663 and 676: the "0 or 1" words flowfix inferred are what the
title writes there.

**`COLOR2` does not darken the albedo.** In the pixel `09eb0a73df63b87d`
(PSO 19145) the albedo is `saturate(T) × lerp(B, C × COLOR2.rgb, k ×
COLOR2.a)`: a texture `T`, a base `B` and a colour `C` made from textures,
and `COLOR2` as a tint weighted by its own alpha. With `COLOR2` = (0, 0,
0, n/255) or (1, 1, 1, 0) the tint's weight is about 0, and the albedo is
about `saturate(T) × B`.

**What `T` and `B` read: textures picked by index.** Their handles come
from words of raw buffers (`RawBufferLoad`, then `CreateHandleFromHeap`):
textures the shader picks by index, which the pass of `descriptor/host.rs`
over such textures uploads, a placed one's slot staying null until then.
In both sessions, from the moment the world loads (06:58:37 and 07:08:22
UTC) to the end, that pass's staging memory sits at its cap
(`staging_mib=1023` of `PASS_STAGING`, 1 GiB), 14,000 to 19,000 slots wait
for their first upload with 3.7 to 4.0 GB of textures left, and about
13,000 are served (the 5-second "textures shaders pick by index" lines of
the run logs). A material whose `T` or `B` is among the waiting slots
reads 0: that is the lead for the black albedo, to be confirmed by an A/B
on that cap (§11.5).

### 11.3 Lighting: the light it is given is 0 at its writer

**What `1cd9bc634417a981` computes** (PSO 20525's pixel, read in its
DXIL): `RGB = e × (D × albedo + S)`, alpha its luminance, where `e` is
word 1 of the 16-byte exposure buffer (`s0+168`: 0.79 to 0.85, not 0),
`albedo` the G-buffer's `s0+584`, and the diffuse light `D = L × (1 + k) +
X` with `L = e0 × k' × texel(s0+1008)`: `s0+1008` is one of the two
R9G9B9E5 views (`0x97bf50000` in world-1, `0x97b110000` in imgsrc-1). `S`
(the additive term) reads the other one, `s0+456`, and `X` the light loop
(tile lists, probes, shadow inputs).

**Its inputs, just before it** (imgsrc-1, pass P618 of `passes-675`; its
output is 0.2 % non-zero, at most 0.0014, while the albedo is non-zero on
12 %):

| Slot | What | Non-zero | Values |
|---|---|---|---|
| `s0+584` | albedo RGBA8 | 12.0 % | max 0.80 |
| `s0+1008` | R9G9B9E5 view (RGBA16F stand-in), the diffuse light | 0 % | RGB 0 |
| `s0+456` | R9G9B9E5 view, the other light | 100 % (alpha) | RGB at most 0.008 |
| `s0+168` | exposure buffer | | [1.27, 0.79, 8, 0.125] |
| `s1+80` | 160×90×13 R32_UINT, tile light lists | 0 % | 0 in every slice |
| `s0+968` | 7680×90 R32_UINT | 100 % | all `0xffffffff` |
| `s0+1160` | 2048×1824 R8_UINT | 0 % | 0 |
| `s1+56`, `s1+48`, `s1+64` | 10×6 R32_UINT, R8_UINT, R8 | 60 %, 100 %, 0 % | up to 3, 1, 0 |
| `s0+760`, `s0+1280`, `s0+808` | 1280×720 R8, R8, RGBA8 | 0 % | 0 |
| `s0+928`, `s0+1248` | 384×384 RGBA16F probes | 100 % | means 0.002 and 0.13–0.16 |
| `s0+936` | 256×256 RGBA16F | 58 % | ±12,280, 16 texels NaN or infinite |
| `s0+664`, `s0+1312`, `s0+488` | the 1×1 textures | 100 % | (0, 0, 0, 1) |

**The input that zeroes it: `s0+1008`, 0 from its writer on.** Its memory
is an R32_UINT texture that the compute `34fb51b8a0312e82` (160×180
groups, P604 of `passes-675`, P567 of `passes-662`) writes with eight
others: there, every texel is 0 (0 too in its twin; 2.4 to 6.3 % of the
third R32 output non-zero). That compute accumulates light over frames: it
reads its own history through four R9G9B9E5 views (all 0, or at most
0.009) and two 1280×720 R11G11B10 textures of the frame (`s1+24`,
`s1+88`), which are **0 on every texel**. Those two are written by P564,
the indirect dispatches `2dc364df5d3278b8` (28,800 groups: its record
holds `0x7080` once the classification `bad01f53f53248e5` has counted) and
`940c2635dc1312f4` (0 groups); they write 0. Among their inputs, the
2048×120 R9G9B9E5 view at `0x17bf40000` is the output of the computes
`8cb27ccf2d50c606` (3,680 groups, P556) and `17e736204a05a105` (P559),
which write 0 too. The local lights are missing as well: the tile lists
the lighting loops over hold no light in any tile.

**The 1×1 textures are black in title memory**: `00 00 00 ff` at
`0x957000000` and `0x957010000`, as the title wrote them, not an upload
that went missing.

### 11.4 The R9G9B9E5 decodings run before their writers (watch item 360)

`resource::decode_now` records a stand-in's decoding at the frame's first
read of the view and runs it in a list of XTP's own at the start of the
next title submission. In imgsrc-1's frame 674, the R9G9B9E5 view at
`0x97d8e0000` is first read in a list submitted in submission 3276,
recorded when 3,272 had been made: its decoding runs at the start of
submission 3273, while its writer (`dce085c07b9414b6`, R32_UINT) and its
readers (the lighting among them) run in submission 3276. The same holds
for `0x97d520000` (written by `34fb51b8a0312e82` and `71883e40620a567a`,
decoded at 3273, all of them in 3276) and for the 2048×120 and 1280×720
pairs of frames 675 and 676. So the stand-ins show the frame before's
light, and the compute pairs that write one texture while reading the
other (§11.3) read last frame's half. It is a real lag, but not the black:
the textures behind them are written 0 in the same frame.

### 11.5 What to fix, and where

- **The light chain's zero** (§11.3): the computes writing the 2048×120
  light data (`8cb27ccf2d50c606`, `17e736204a05a105`, `6bb493a39685ddcc`)
  and the per-pixel lighting `2dc364df5d3278b8` write 0 although their
  dispatches run. Next: their inputs just before them
  (`XTP_PASS_INPUTS=8cb27ccf2d50c606`, then `2dc364df5d3278b8`, and
  `XTP_FRAME_PROBE` on the same programs for their buffers: light
  descriptions, counts), down to the first input that is 0 or the first
  program whose translation loses the value.
- **Decode in the reading list** (§11.4, `resource.rs`): run the
  R9G9B9E5 decoding in the list that reads the view, after its writer, as
  the copy path of `alias_copy` already does (VIctory #1868's in-list
  pack), instead of once a frame before the next submission.
- **Textures picked by index** (§11.2, `descriptor/host.rs`): an A/B on
  `PASS_STAGING` (1 GiB, held for each texture's life) in one session: the
  waiting slots, and the G-buffer's albedo (non-zero share and mean) with
  the cap raised or the staging released after each upload.
