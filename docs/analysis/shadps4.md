# shadPS4's shader recompiler and XTP's refusal classes

The person asked whether anyone outside the Xbox projects converts AMD
shaders well. shadPS4, the PS4 emulator, translates GCN (the PS4's ISA, an
ancestor of the Series' RDNA2) to SPIR-V for Vulkan. This report says what
its recompiler does about each class of program XTP refuses, whether that
needs raw GPU pointers (which DXIL lacks), how XTP would do the same on
D3D12, and which three techniques to port first.

- **Read:** `github.com/shadps4-emu/shadPS4` at `5c3b6910b9` (`main`,
  2026-11-23), a sparse shallow clone made with `gh` in the session's
  scratchpad: `src/shader_recompiler/` (123 files), `src/video_core/buffer_cache/`,
  the settings default read through the GitHub API, and the history of the
  files below (GitHub API). Nothing was built or run.
- **Paths:** shadPS4 paths are under `src/shader_recompiler/` unless they
  start with `src/`; line numbers are at `5c3b6910b9`. XTP paths are at
  `ab88646` (main).
- **No code is copied.** Mechanisms are described in our own words.
- **XTP's numbers** come from the world session `20261125-000534`
  (`E:\xtp-data\cache\play\20261125-000534\run.log`, present's build with
  the image switches): one "shader does not translate" line per program
  (556 programs), and the "dropped N/frame without a host pipeline" lines,
  which list the top 40 programs every few seconds. "Dispatches a frame"
  below is each program's last such figure, summed per class: a lower
  bound, since a program under the top 40's floor (about 4 a frame) does
  not appear. The census (7d248d0b, `docs/analysis/refusals.md`, not yet
  delivered) will give the exact counts and passes.
- **What the programs are** was looked up in the private table of the
  pack's 26,503 shader records (`E:\xtp-data\re\out\shader-names.tsv`,
  outside the repository). Only kinds of work are named here, never a
  record name.

## 1. Summary

shadPS4 is good at what the PS4 needs, and its model explains why its
passes do not map one to one onto XTP. A PS4 program has no root
signature: it gets 16 user-data dwords and finds every descriptor by
reading memory through pointers in them. So shadPS4 **walks those pointer
chains on the CPU at every draw** (a JIT-compiled x86-64 walker), copies
every dword the program reads into a flat buffer, reads the descriptors
from that copy on the CPU, and compiles a shader variant for their
contents. XTP knows from the D3D12 root signature where every table is,
and its tracker (`resource.rs`) is more general and refuses where
shadPS4 asserts or guesses.

| XTP refusal | Programs | Dispatches a frame | What they are | shadPS4's answer | Raw pointers? | Verdict for XTP |
|---|---:|---:|---|---|---|---|
| an address in an SGPR that is no root parameter | 34 compute | 223 (22 seen) | **None is in the title's pack**: the console system library's ray-tracing structure builder (§3.1) | Flattening of extended user data (`flatten_extended_userdata_pass.cpp`) | No | **Port the idea**: treat the spilled table as user data (no memory walk at all) |
| an address not from a root parameter | 13 compute | 69 (1 seen) | Ray-tracing consumers: traced sun and light shadows, probe gathers, binned tracing, a particle collision (the 69) | Walker if the chain is evaluable, else device address (off by default) or assertion | Device-address path: yes | Page directory (the fork's `by_address`), long tail |
| a scalar memory read as data | 19 mesh, 6 compute, 5 pixel, 4 vertex | 110 + 84 + 11 + 31 | Characters and skinned shadows (mesh, meshdata d687ecdb); character skinning and its ray-tracing resolves (compute); vegetation impostors (vertex); deferred surfaces (pixel) | No such class: descriptor words are data in its flat copy | No | Flat constant buffer filled at draw time, or R2's heap shadow; meshdata and rawfix cover the mesh forms |
| a lane mask read as data | 6 pixel, 5 mesh, 2 compute | 16 + 43 | Characters (mesh, pixel); skinning for the ray-tracing structure | Masks are host-subgroup ballots: on 32-lane hosts the high dword is 0 (silent) | No | Ideas only; XTP stays fail-closed |
| wave64 lanes in a group of more than 64 threads | 23 compute | 4.7 (1 seen) | **19 of 23: tiled cascaded-shadow rendering and classification**; heap defragmentation, alpha tiles, ray-tile sorting, a matrix product | `lower_wave64_pass.cpp`: ballots, fixed-lane reads and `v_mbcnt` exchanged through LDS, only outside loops and thread-dependent branches | No | **Port the idea** where XTP proves every thread reaches the exchange |
| an address from a phi | 8 compute | 15 (3 seen) | Hair-strand interpolation (6), volumetric scattering (2) | Assertion (not handled) | — | **First find why they still refuse**: `mesh-wave64.md` §9.2 has them translating since `e1b75d7` (§3.6) |
| a descriptor read from the first lane of computed VGPRs | 4 mesh | 40 (3 seen; 86–120 in present-1) | **Static G-buffer geometry (2) and vegetation (2)** | Sees through `v_readfirstlane`; one value number per load address (`srt_gvn_table.h`); a join of two loads asserts | No | **Port** the fork's `lane_copies`; shadPS4 gives the rule behind it |

The answer to the person's question: yes, shadPS4 translates AMD shaders
well for its platform, but none of its passes can be dropped into XTP:
they rest on a flat guest address space in the emulator's process, a
CPU pass over memory at every draw, and Vulkan device addresses. Three
techniques carry over, rewritten for D3D12 (§5). In several places XTP is
already ahead (§2.3).

## 2. How shadPS4's recompiler works

### 2.1 The pass order

`recompiler.cpp` 141–235 (`TranslateProgram`): decode, control-flow graph,
translation to an SSA IR (one console lane per host invocation, SGPRs as
uniform values), then, in this order: `LowerFp64ToFp32` when the host has
no fp64, `LowerMovRelPass`, `SsaRewritePass`, `ConstantPropagationPass`,
`ReadLaneEliminationPass` (188), the tessellation transforms,
`RingAccessElimination`, `ResourceDiscoverPass` (198),
`FlattenExtendedUserdataPass` (199), `ResourcePatchingPass` (201),
`LowerBufferFormatToRaw`, the shared-memory passes, `PhiSimplificationPass`,
`InverseBallotEliminationPass`; then structurization and, on the
structured program, `SharedMemoryBarrierPass`, `LowerWave64BallotPass`
(228) and clean-ups.

### 2.2 The model

- **User data and the flat buffer.** A program's 16 user-data dwords
  (`resource.h` 14, `NUM_USER_DATA_REGS`) are the start of a per-draw flat
  buffer. `s_load_dword*` becomes `ReadConst(base, offset)`
  (`frontend/translate/scalar_memory.cpp` 40–59) and `s_buffer_load_dword*`
  `ReadConstBuffer(V#, offset)`.
- **The walker** (`ir/passes/flatten_extended_userdata_pass.cpp` 746–870).
  Every `ReadConst` whose base comes from user data or from another
  `ReadConst` is a node of a tree rooted at a user-data pair. Nodes with
  the same base and offset get one value number (`ir/srt_gvn_table.h`), so
  equal loads share one slot. The pass emits x86-64 code (Xbyak, 636–682)
  that, at each draw, follows the tree through guest memory and stores
  every dword the program reads, in tree order, into the flat buffer after
  dword 16; each `ReadConst` is then a read of its slot. An offset may be
  computed if it uses only user data, earlier slots and add, sub, mul,
  shifts, and/or/xor/not, min/max and bit-field extract (488–510): the
  walker evaluates it at the draw. A fault in the walker is caught by a
  signal handler that patches the faulting load to give 0 from then on
  (71–135).
- **Descriptors on the CPU.** `ResourceDiscoverPass` collects the dwords of
  every V#, T# and S# operand, sees through `v_readfirstlane_b32` (308) and
  through a few compiler idioms that patch descriptors in registers (a
  stride OR-ed into dword 1, a V# around a constant table in the code, a
  cube viewed as an array, sampler clamp and anisotropy edits: 24–270).
  Every dword must come from user data or a load, else an assertion
  (289). `ResourcePatchingPass` turns each operand into a binding index
  and records where its dwords sit in the flat buffer (`SharpFetch`,
  `resource.h` 25; `SharpLocationFromSource`, 165). At bind time the host
  reads the descriptors from the flat buffer and binds the buffer cache's
  copy of the memory they name.
- **Specialization.** The shader is compiled per descriptor contents: a
  buffer's stride, format, swizzle; an image's type and format
  (`specialization.h` 24, 79). `LowerBufferFormatToRaw` then reads typed
  buffers as raw bytes and converts in the shader with the format known at
  compile time.
- **Device addresses** (`backend/spirv/spirv_emit_context.cpp` 1219,
  `DefineGetBdaPointer`; 1278, `DefineReadConst`). A `ReadConst` the walker
  did not place (slot 0) is read through a page table of 64-bit Vulkan
  buffer device addresses, one per sparse page of guest memory. A page
  with no address sets its bit in a fault buffer and reads 0; a compute
  pass collects the faults and the buffer cache makes those pages resident
  for later draws (`src/video_core/buffer_cache/fault_manager.cpp`). This
  path is **off by default**: `direct_memory_access_enabled{false}` in
  `src/core/emulator_settings.h` 423. Off, an unplaced `ReadConst` reads
  flat-buffer dword 0, user-data dword 0, silently
  (`backend/spirv/emit_spirv_context_get_set.cpp` 55–65).
- **Lanes and masks.** A lane mask is the host subgroup's ballot, packed
  into 64 bits (`backend/spirv/emit_spirv_warp.cpp` 51–66). On a 32-lane
  host its high dword is 0. Only in compute, and only in blocks outside
  every loop and every branch whose condition depends on the thread id,
  does `LowerWave64BallotPass` rebuild console wave64 masks (§3.5).

### 2.3 Where XTP is already ahead

- **Tracking.** XTP's tracker follows joins, selects, per-lane picks,
  per-lane loads (`global_load_*`), waterfall loops and the code's own
  constant tables (`resource.rs` `Source`, 44–80). shadPS4 recognises
  eight fixed patterns and asserts on the rest, and a failed assertion
  shuts the emulator down (`src/common/assert.cpp`), where XTP refuses one
  program with a reason.
- **Lane spills.** `ReadLaneEliminationPass` (`ir/passes/readlane_elimination_pass.cpp`
  101) turns a `v_readlane_b32` at a constant lane back into the value a
  matching `v_writelane_b32` saved, through phis. XTP's `ir/unspill.rs`
  (unspill 8190a07f, from the fork's pick 9) does the same and refuses
  where the restored value could depend on the path a lane took; shadPS4
  does not check that. Nothing to take.
- **Divergence.** shadPS4 calls a branch divergent only if its condition
  depends on the thread id or lane id (`lower_wave64_pass.cpp` 14,
  `shared_memory_barrier_pass.cpp` 73). A branch on per-lane loaded data
  counts as uniform, so a group barrier can be placed where some waves do
  not arrive. XTP proves barrier reach (`emit/lower/lanes.rs` 45,
  `group_reads_meet`).
- **Fail-closed.** Device addresses off, the walker's patched faults, and
  ballots outside uniform blocks are all silent approximations. XTP's rule
  is to refuse instead (`shader.md` §7, risk 5).

## 3. XTP's refusal classes, one by one

### 3.1 "an address in an SGPR that is no root parameter" (34 compute)

**XTP.** `resource.rs` 909 (`why_no_base`): the address's computed term
is an entry SGPR that holds no root parameter. All 34 originally refused
at the first reads of the table. **None of the 34 is in
the title's shader pack**, while every other program refused in the
session is. With the fork's measurement (fxceless's VIctory fork, `578d459`,
`victory-shader/src/spill.rs`: "every compute program of the console's
ray-tracing structure builder" keeps its root parameters in a spilled
table, 118 of 118 loads on a declared parameter), these are the console
system library's builder programs. When a compute root signature needs
more dwords than the program's user SGPRs (`COMPUTE_PGM_RSRC2`), the
console keeps none of them in registers: s0 and s1 & 0xff hold the 40-bit
address of a table, and the parameter the layout declares at user-data
index N ≥ 16 is at byte 4 × (N − 16).

The 16-digit BLAKE3 hashes in the 20261125-000534 run match 34 of the 52
compute containers embedded in the console's BVH builder library. All 34
declare four user SGPRs and their first root parameter at dword 16, and
each reads the table early in the program. To reproduce, run
`xtp shader-info` on that library from your own console's system files.

**shadPS4.** This is the PS4's "extended user data": user data beyond 16
dwords lives behind a pointer in user data. `FlattenExtendedUserdataPass`
exists for it (its first version is "Implement shader resource tables",
#1165, `9ec75c3feb`): the walker copies the table's dwords into the flat
buffer at each draw.

**Raw pointers:** no. The walker runs on the CPU, the shader reads an
ordinary bound buffer.

**XTP on D3D12: better than a walker.** XTP is the console's D3D12 driver:
the table holds the root arguments of `SetComputeRoot*`, which XTP already
captures per slot and binds as user data by root slot and dword
(`emit/mod.rs` 326, `UserData { index, slot, root, dword }`). Rewriting
each load through s0:s1 at byte 4 × (N − 16) as user-data dword N makes a
spilled parameter a root parameter like any other; no memory is read and
no CPU pass runs. The fork's `spill.rs` does this rewrite (loads with a
constant, dword-aligned offset; others left to refuse). What shadPS4 adds:
the value-number slot sharing, and offsets computed from user data, which
the fork leaves alone. One D3D12 limit: root constants count against the
64 DWORDs of a root signature. If index N goes past what is left, the user
data goes in a root CBV of a per-dispatch upload instead.

XTP now tracks each loaded SSA word by root parameter and dword
(`resource/spill.rs`, ported from VIctory `27c3dc9d5`). It packs the
constants and descriptor-table offsets the shader reads into the existing
per-dispatch root constants. Root views use their existing slot bindings.
At `6e99955` on the L4 all 34 reached a later instruction. Thirteen reached emission
and stopped at a root UAV or typed root view. The other 21 stopped during
address tracking. None emitted DXIL.

Les 13 refus de vue racine disparaissent avec une table D3D12 d'un UAV
brut par slot. La vue couvre la copie hôte du tampon et le shader ajoute
la distance entre le début aligné de la vue et l'adresse donnée par le
titre. Cela sert aussi pour les compteurs. Les accès par demi-mot
préservent les octets voisins. Un DispatchX en unités de threads transmet
les trois dimensions au shader du constructeur. Dans un groupe de queue,
les écritures et les comptages d'EXEC respectent ces limites, y compris
lorsque deux vagues hôtes exécutent une vague console et échangent via LDS.

Origine du premier accès refusé dans les 21 autres programmes du relevé
initial. Les préfixes identifient les empreintes BLAKE3 de la session.

| Programmes | Première adresse | Résolution |
|---|---|---|
| 6 · `9205d265`, `894c9165`, `a63e2dfc`, `198f6dfd`, `cb9647e7`, `4b5aedc0` | Une constante donne une adresse en unités de 256 octets. Le champ lu tient dans les huit bits libérés. | Les deux moitiés doivent provenir du même décalage. L'hôte lie la copie du tampon et le shader lit le champ dans une SRV brute. |
| 8 · `a197f50c`, `f28a33a8`, `c85822de`, `6f6a359d`, `162216b9`, `7bb5b1b9`, `3df629ff`, `ffd93c8a` | Deux dwords consécutifs de constantes forment une adresse complète. | Les lectures à déplacement fixe utilisent une SRV si les mots appartiennent au même paramètre. Pour une copie en VGPR, les lanes qui lisent doivent avoir exécuté les deux copies. Plusieurs variantes calculent ensuite un index avec une retenue 64 bits à partir de données lues sur le GPU. |
| 2 · `190fbf0e`, `f6e47efa` | Une paire intérieure aux constantes déportées sert de pointeur. Le premier programme lui ajoute un index par lane dès la première lecture. | La paire seule peut désigner une vue. Les accès suivants parcourent la mémoire et comprennent des opérations atomiques globales. |
| 2 · `5279ae36`, `35e51bb2` | La lane reconstruit une adresse et y insère un numéro de champ. | La paire varie entre lanes. Une vue uniforme liée au début du dispatch ne suffit pas. |
| 3 · `5e687674`, `5062bc7a`, `3cea4534` | Le tampon source contient lui-même l'adresse de la prochaine ressource. | Les deux premiers lisent à une adresse indexée par lane, le troisième y écrit. La ressource de destination n'est connue qu'à l'exécution sur le GPU. |

Pour ces parcours indirects, le fork associe les pages d'adresses du titre
à des vues utilisables par le shader. XTP dispose des copies côté hôte.
Le shader aurait besoin de la correspondance décrite en §5.4 pour choisir
la copie, vérifier les limites et conserver l'identité des écritures et
des opérations atomiques.

Sur le L4 en vagues de 32 lanes, `translate --list --validate --pipelines`
sur `1084cf6` donne **21 DXIL validés et 21 pipelines créés parmi les
34 programmes de la session (+21, 0 perdu)**. Ce sont les 13 programmes
à vue racine, les six à adresse compactée, ainsi que `162216b9` et
`3df629ff`. Les 13 autres s'arrêtent au suivi d'adresse. Parmi eux,
`5e687674` reste le refus de l'item 312. Les trois autres empreintes
suivies par cet item (`caa78b8c`, `28918b71`, `4ddd6f97`) ont un
pipeline. Sur les 52 conteneurs de la bibliothèque, le même relevé passe
de 0 à 30 DXIL validés et pipelines créés.

Le pack du titre et la bibliothèque système se mesurent séparément. Sur la
ferme, la comparaison de `6e99955` et `1084cf6` avec
`shader-survey --host-lanes 32-32 --json` donne les mêmes résultats pour
les 26 698 programmes, entrée par entrée. Les 25 725 programmes traduits
ont tous conservé la même empreinte DXIL.
Après intégration de `main` (`d24d377`), le même relevé sur la ferme
compare cette base à `8e29dce`. Les 25 768 traductions produisent le même
DXIL et les 930 refus gardent leur catégorie.

La veille 26 (item 362, VIctory #2118) signale qu'un constructeur exécuté
en partie peut écrire une structure incomplète. Au chargement de la DLL,
XTP inventorie ses conteneurs et reconnaît leurs copies par contenu. La
première création d'un pipeline du constructeur lance la préparation de
toute la bibliothèque. Si un pipeline est en préparation au premier
dispatch, tous les dispatchs du constructeur sont ignorés pour cette image,
sans attendre les compilateurs. Un seul refus bloque le constructeur pour
la session, y compris les programmes déjà prêts. Le journal indique
`BVH builder held` avec une empreinte
refusée, ou `BVH builder admitted` quand tous les pipelines sont prêts.
Pour le test RT A/B (`ffc6d315`), relever ce verdict sur le chemin A :
avec `held`, la mesure ne représente pas un BVH construit par le GPU.

**Image impact:** the largest dispatch count of all classes (223 a frame),
but the builder only makes the ray-tracing structure. Nothing is lit by it
until the trace programs of §3.2 translate too.

### 3.2 "an address not from a root parameter" (13 compute)

**XTP.** `why_no_base`, last arm: the base is computed by an instruction
the tracker does not follow (not an addition of root parameters, not a
load, not a first-lane read, not a join). The 13 refuse in mid-program
(0xc4 to 0x233c). All are ray-tracing consumers: traced sun shadows (4),
probe and irradiance-cache gathers and traces (4), screen-probe and
binned tracing (2), traced shadows for local lights and a shadow mask
(2), and a particle collision against the scene's ray-tracing structure
(1, the 69 dispatches a frame). They walk structures that lie in plain
title memory, outside any view (the fork's `memory.rs` notes).

**shadPS4.** If the address is evaluable from user data with the allowed
operations, the walker flattens the read. A 64-bit address computed
otherwise asserts in the flattening pass (`flatten_extended_userdata_pass.cpp`
832), unless it is a first-lane read, which takes the device-address path
(off by default, and then user-data dword 0 is read instead, §2.2).
Pointers read out of memory per lane (structure nodes) are beyond the
walker.

In the same run, all 13 have their first root parameter at user-data
dword 4. At the failing instruction, six read through an SGPR address
built from user data, and seven issue a per-lane global load. The
spill-table rule applies to zero of the 2,682 compute programs in the
title's pack.

**Raw pointers:** the device-address path, yes.

**XTP on D3D12.** A page directory: per 64 KiB page of title address
space, the number of a host copy (or of a UMA window over title memory,
`docs/analysis/uma-feasibility.md`) and the page's offset in it; per copy
a raw view in a run of heap slots that SM 6.6 shaders index
(`ResourceDescriptorHeap`); a fault list the shaders append to and the
host reads once a frame. This is the fork's `by_address` design
(`victory-shader/src/resource.rs` 878, `victory-gpu/src/memory.rs`), and
shadPS4's page table and fault buffer are the same idea on Vulkan. In
both, an access to a page not yet there reads 0 and only the next frame
gets it: a lost access.

### 3.3 "a scalar memory read as data" (34: 19 mesh, 6 compute, 5 pixel, 4 vertex)

**XTP.** `emit/lower/smem.rs` 214 (`descriptor_data`): an `s_load_dword*`
from descriptor memory whose dwords the code also reads as numbers, in a
form the size rebuild does not cover. meshdata (d687ecdb) classified the
19 mesh programs: form A, a `GetDimensions` of a structured buffer and
then an `s_buffer_load` of its last element (rawfix's `Fields::Smem` arm,
`6b3ef55`, ccf43697); form B, the fork's `dead` (mask dwords joined with
numbers and never read).

**shadPS4.** The class does not exist there: in its model every dword a
program reads through user data, descriptors included, is copied into the
flat buffer, and reading a V#'s record count as a number is a read of a
flat-buffer slot. A `s_buffer_load` through a V# is a storage-buffer load.

**Raw pointers:** no.

**XTP on D3D12.** Two routes, both without pointers:
- *R2, heap shadow* (`shader.md` R2): bind the console heap's bytes as one
  raw SRV and read descriptor dwords from it. One binding, any index, per
  lane included.
- *A draw-time flat constant buffer*, shadPS4's route: where the address
  is uniform and rooted at root parameters (a table base plus a constant
  or a root-constant index), the GPU layer reads the dwords at bind time
  (descriptor heaps are written by the CPU, so there is no GPU-write
  hazard) and appends them to the user data, or to a per-draw CBV. AnyPS5's
  measurement argues for a constant buffer over a storage buffer (§6).

The mesh programs are meshdata's and rawfix's. The 15 others are the next
question: character skinning in compute (2 programs, 84 dispatches a
frame), ray-tracing resolves for characters (4), vegetation impostors in
vertex programs (2 seen, 31 a frame) and deferred surfaces in pixel
programs (1 seen, 11 a frame). The skinning and resolve programs share
their material families with meshdata's character programs, so the same
two forms are likely. Translate them after rawfix and meshdata land;
whatever still stops on this class is where the flat buffer pays.

### 3.4 "a lane mask read as data" (6 pixel, 5 mesh, 2 compute)

**XTP.** `emit/lower/mask.rs` 35 (`LANE_MASK`): a 64-bit lane mask used as
a number (a population count, a compare with a constant, a shift, a
store) in a program the per-lane form lowers. On a 32-lane host a console
wave64 pixel or mesh program has no exact form. The programs: characters
(mesh, 43 dispatches a frame; pixel, 16) and two skinning programs for the
ray-tracing structure.

**shadPS4.** Masks are host-subgroup ballots everywhere. On a 32-lane host
the high dword is 0. That is right for idioms that do not depend on the
wave size (compaction: count the ballot, add it to a counter once per
wave, place each lane by its `v_mbcnt`), and wrong where the code
compares a mask with a 64-lane constant or reads lanes 32 to 63. No pass
checks which case a program is.

**Raw pointers:** no. D3D12's `WaveActiveBallot` behaves the same.

**XTP:** ideas only. A proof that a mask flows only into a wave-size-free
idiom would let XTP take host-wave semantics for it; without one, the
refusal stays. For mesh and compute, AnyPS5's lane pairs (§6) are the
exact answer.

### 3.5 Wave64 lanes in a group of more than 64 threads (23 compute)

**XTP.** `emit/lower/mod.rs` 577–600. present's pick 12 (`group_reads`,
landed in `bd0cc66`) takes the groups whose only reads across halves are
lane reads at a fixed lane, between group barriers every thread reaches;
`group_reads_meet` also wants no loop at all, barrier blocks in a chain of
dominance and LDS ordered by barriers (present's comment on 179cd4fc). No
mask held whole (`masks_held`), no `v_readfirstlane_b32`. Which condition
each of the 23 fails was not looked at; of the 43 programs
`docs/wave64-on-wave32.md` read for this refusal, "nearly all count mask
bits (`v_mbcnt`) and read a mask as a number". **19 of the 23 are the
tiled cascaded-shadow renderer and its classification**; the others
defragment a heap, build alpha tiles,
sort ray tiles and multiply matrices. Only the last appears in the drop
lines (4.7 a frame); each of the others runs fewer than about 4
dispatches a frame, if any. The census has to say whether they run.

**shadPS4.** `ir/passes/lower_wave64_pass.cpp` (#4995 `bacdcc745c`, #5116
`2f40f87b5b`), for compute on hosts whose subgroups are not 64 (104–106):
- a ballot whose 64-bit value is used: each host subgroup writes its
  32-bit ballot to its own LDS slot, group barrier, each thread reads the
  64-bit pair of its console wave, barrier;
- `v_readlane` at a constant lane: the same exchange of the read lane;
- `v_mbcnt` against such a mask: recomputed from the flat thread index
  modulo 64.

A console wave is the host subgroups 2k and 2k + 1 (177), which assumes
host subgroups hold consecutive flat ids, the assumption XTP's
`thread_id` makes too (`emit/lower/lanes.rs` 368) and hwtests questions
for NVIDIA's 8 × 8 groups. It applies only in blocks outside every loop
and every thread-dependent branch (`FindUniformBlocks`, 38); elsewhere it
logs a warning and keeps the 32-lane ballot (124).

**Raw pointers:** no. LDS and group barriers are `groupshared` and
`GroupMemoryBarrierWithGroupSync`.

**XTP:** extend `group_reads` to masks held whole, and to `v_mbcnt`
against them, when every ballot whose value is used sits where
`group_reads_meet` proves every thread arrives: exchange the two 32-lane
halves through LDS between two group barriers, as shadPS4 does, with
XTP's own reach proof instead of shadPS4's thread-id test. Both forms are
for programs without loops around the exchange, so whether the shadow
programs gain depends on where their ballots are. If inside the loops
over their tiles, neither reaches them; for those without LDS,
`docs/wave64-on-wave32.md`'s planned "one console wave per host group" is
exact and complete.

### 3.6 "an address from a phi" (8 compute)

**XTP.** `why_no_base`: the address joins different values at a block.
The 8: hair-strand interpolation (6, 15 dispatches a frame) and
volumetric scattering reconstruction (2).

**A discrepancy to settle first.** These are the 8 hashes of mesh64's
code-table joins (`docs/analysis/mesh-wave64.md` §9.2: the fork's
`code_join`, ported in `e1b75d7`), whose survey over the pack went from
28 refusals of this class to 0. Yet session `20261125-000534` refuses all
8, and the 2 "address read from the first lane of a per-lane value" of
the same section (`5ed77e9e8f084dea`, `0a5d4f9047b3dce4`) as well. Both
builds the session can have run hold `e1b75d7`: present's `bacfcc6`, and
main's `E:\XTP\target\diag\xtp.exe`, written at 01:05 local time, just
before the session started (01:05:34). So either the session ran an
older binary after all, or the run-time translation differs from the
survey's (the layout the title gives at pipeline creation, a cached
result). Nothing of shadPS4 is needed until that is known.

**shadPS4.** Not handled: a load whose base is a phi asserts in the
flattening pass (832) before any other path is tried. `PhiSimplificationPass`
runs later and only folds unary operations and duplicate phi cycles.

**XTP**, for joins `code_join` does not take: either a per-path address
(each predecessor's `Address`, the load lowered as a select between them,
as `Source::Phi` already does for descriptors), when every arm is rooted
at a root parameter; or the page directory of §3.2, which takes any title
address.

### 3.7 "a descriptor read from the first lane of computed VGPRs" (4 mesh)

**XTP.** `resource.rs` 1464 (`per_lane`), the last refusal arm (1503): a
waterfall reads a descriptor from the first lane of a VGPR the tracker
does not follow. The 4 are static G-buffer geometry (2) and vegetation
(2). rea's note on meshdata's task counts 86 to 120 dropped dispatches a
frame for them in present-1's frames 677 and 683 (40 seen in this
session): the G-buffer's scenery draws, and meshdata suspects a good part
of the draws P293 loses (the street and buildings missing from the main
depth, `world-black.md` §2). The fork's
answer is `lane_copies` (`victory-shader/src/resource.rs` 2104): the
console compiler loads a descriptor on both sides of a per-lane branch,
the join is per lane so it lives in VGPRs (`v_mov_b32` from the SGPRs),
and the access loops over distinct values; when every scalar those moves
copy is the same dword of loads from one constant address, the lanes
agree and the descriptor is any of the loads.

**shadPS4.** Half of it. `ResourceDiscoverPass` sees through
`v_readfirstlane_b32` to the VGPR's value, and in its SSA IR a `v_mov_b32`
from an SGPR is that SGPR's value, so a plain copy resolves. Loads of one
base and offset share one value number (`srt_gvn_table.h`). But the
per-lane join of two loads is a phi, not a load, and asserts (289).

**Raw pointers:** no.

**XTP:** port the fork's `lane_copies`, critically reviewed. Its rule is
shadPS4's value numbering applied through the join: loads at equal
`Address` (`resource.rs` 29) are one descriptor. XTP already compares
addresses, so nothing of shadPS4's code is needed.

## 4. The key constraint: raw pointers

DXIL has no pointers: a shader reaches memory only through descriptors
(root CBVs and SRVs whose GPU address the CPU sets, or heap views, indexed
at run time with SM 6.6). For each shadPS4 technique:

| Technique (shadPS4 file) | Needs device addresses? | D3D12 form for XTP |
|---|---|---|
| Extended user-data flattening (`ir/passes/flatten_extended_userdata_pass.cpp`, `ir/passes/srt.h`) | No: a CPU walker and a bound buffer | Spilled tables: rename loads as user data (no walk). Other uniform chains rooted at root parameters: read on the CPU at bind time into user data or a per-draw CBV |
| Load value numbering (`ir/srt_gvn_table.h`) | No | Equal `Address` = same descriptor (`lane_copies`) |
| Descriptor discovery and patching (`ir/passes/resource_discover_pass.cpp`, `ir/passes/resource_patching_pass.cpp`, `resource.h`) | No | XTP's root-signature tracking already does this, more generally |
| Specialization on descriptor contents (`specialization.h`), typed buffers as raw (`ir/passes/lower_buffer_format_to_raw.cpp`) | No | XTP reads strides at run time (stride table, 84451453) and raw views (`XTP_RAW_STRUCTURED`); per-descriptor variants would multiply pipelines |
| Dynamic `ReadConst` through the page table (`backend/spirv/spirv_emit_context.cpp` 1219, 1278; `ir/passes/shader_info_collection_pass.cpp` 233, 263; `src/video_core/buffer_cache/fault_manager.cpp`) | **Yes** (`PhysicalStorageBuffer64`) | Page directory to (copy or UMA window, offset), raw views in heap slots indexed by SM 6.6, fault list read per frame: the fork's `by_address` |
| Wave64 ballots through LDS (`ir/passes/lower_wave64_pass.cpp`) | No | `groupshared` slots and group barriers |
| Lane-spill elimination (`ir/passes/readlane_elimination_pass.cpp`) | No | Already in XTP (`ir/unspill.rs`) |
| Ring accesses as attributes (`ir/passes/ring_access_elimination.cpp`) | No | For merged ES + GS (`shader.md` G1, G2); not one of these classes |
| LDS barriers for lockstep waves, LDS too large to a storage buffer (`ir/passes/shared_memory_barrier_pass.cpp`, `ir/passes/shared_memory_to_storage_pass.cpp`, `ir/passes/shared_memory_simplify_pass.cpp`) | No | Barriers: mesh-fix's LDS order inside a console wave. Storage fallback: an `RWByteAddressBuffer` slice per group, if a program ever needs more than D3D12's 32 KiB of `groupshared` |

So the only technique that needs raw pointers is the fallback for what
nothing else pins, and D3D12 can emulate it with a page directory over
bindless raw views. Everything else works with root parameters and heap
descriptors.

## 5. Port plan

Ranked by expected image impact, following the census's order (G-buffer,
lighting, shadows, post-process first; ray tracing last).

### 5.1 Equal loads are one descriptor, through first-lane copies

- **Gain:** static G-buffer geometry and vegetation (4 mesh programs,
  86–120 dispatches a frame in present-1, 40 seen here). meshdata has
  already extracted 38 candidate programs of the pack
  (`E:\xtp-data\runs\meshdata\lanecopy-hashes.txt`): the survey of this
  item.
- **Source:** the fork's `lane_copies` (`victory-shader/src/resource.rs`
  2104, `578d459`, fxceless, GPL-3.0: take with "Ported from fxceless's
  VIctory fork (commit 578d459)"). shadPS4's `ir/srt_gvn_table.h` and
  `ir/passes/resource_discover_pass.cpp` (GPL-2.0-or-later) give the
  principle; no line of them is needed.
- **Lands in:** `crates/xtp-shader/src/resource.rs` (`per_lane`, a new arm
  before the refusal at 1503) and its tests; `emit/waterfall_tests.rs` for
  a GPU test of two loads joined per lane.
- **Owner:** meshdata (d687ecdb) owns `resource.rs` and offered to take
  it in its comment; otherwise a task after meshdata.
- **Effort:** small (one tracker arm, unit tests, a survey before and
  after on the L4).

### 5.2 Wave64 ballots exchanged through LDS where every thread arrives

- **Gain:** up to 19 cascaded-shadow programs (the sun's shadow maps) and
  4 others, **if** their ballots sit outside their loops. First
  step: disassemble the 19 (`xtp shader-info --disasm`) and mark where
  each ballot whose value is used and each `v_mbcnt` sits relative to the
  loops and barriers. If inside the loops, skip this item and take
  `docs/wave64-on-wave32.md`'s "one console wave per host group" for the
  ones without LDS.
- **Source:** shadPS4's `ir/passes/lower_wave64_pass.cpp` (GPL-2.0-or-later,
  #4995, #5116): the slot layout and the two-barrier exchange. XTP's reach
  proof (`group_reads_meet`) replaces shadPS4's thread-id test.
- **Lands in:** `crates/xtp-shader/src/emit/lower/mod.rs` (the
  `group_reads` condition, 577), `emit/lower/lanes.rs` (the exchange),
  `emit/lower/mask.rs`; tests in `emit/split_tests.rs`. No task in
  progress owns these files (present and stride, done, did).
- **Effort:** small to medium.
- **Licence:** a port of `lower_wave64_pass.cpp` lines would keep its
  notice ("Ported from shadPS4 (src/shader_recompiler/ir/passes/lower_wave64_pass.cpp,
  commit 5c3b6910b9)"), but XTP emits DXIL from its own IR, so only the
  design transfers.

### 5.3 Flattened user data: spilled tables and draw-time constants

- **Gain:** (a) the 34 programs of the ray-tracing structure builder (223
  dispatches a frame), a prerequisite for every ray-traced effect; (b) the
  non-mesh "scalar memory read as data" programs that still stop after
  rawfix and meshdata, if their reads are uniform and rooted at root
  parameters (character skinning 84, vegetation impostors 31, surfaces 11
  dispatches a frame).
- **Source:** shadPS4's `ir/passes/flatten_extended_userdata_pass.cpp`
  and `ir/passes/srt.h` (GPL-2.0-or-later) for the model; the fork's
  `spill.rs` (GPL-3.0) for the console's spill layout. XTP needs no JIT:
  (a) is a rename in the translator; (b) is a short list of (root
  parameter, offset) reads the GPU layer performs at bind time.
- **Lands in:** (a) a spill rewrite before tracking (`crates/xtp-shader/src/ir.rs`
  or a new `resource/spill.rs`), `native.rs` (layout entries past the user
  SGPRs), `emit/mod.rs` (`UserData` past index 16), and
  `crates/xtp-gpu/src/pipeline/host.rs` (root-constant count, root-CBV
  fallback past 64 DWORDs). (b) `emit/lower/smem.rs` (rawfix's) and
  `crates/xtp-gpu/src/draw.rs` (shared: add lines only), or R2's heap
  shadow instead (`shader.md` R2).
- **Effort:** (a) small to medium; the builder programs come from the
  system library at run time. Its DLL can be passed directly to
  `xtp translate` for an offline survey. (b) medium, translator
  and GPU layer.

### 5.4 Not in the top 3

- **Page directory** (§3.2): the long tail of ray-tracing consumers (13
  programs), and whatever of §3.6's joins and first-lane addresses (8 + 2)
  still refuses once their discrepancy is settled. Large: a new `xtp-gpu`
  memory module, a lowering for raw addresses, faults, after the UMA
  decision (`gpu.md` G15–G18). Take the fork's D3D12 design, reviewed.
  shadPS4's fault path is a reference for bounding the read-back: the
  shaders set bits in a page bitmap, and a compute pass compacts it into
  at most 1,024 pages before the CPU reads them
  (`src/video_core/buffer_cache/fault_manager.cpp`, GPL-2.0-or-later).
- **Do not port:** the default fallback that reads user-data dword 0 for
  a load nothing placed; the walker's patched faults; ballots kept at 32
  lanes outside uniform blocks; the thread-id-only divergence test;
  assertions on unknown descriptor forms; per-descriptor shader variants.

## 6. Kyty-derived PS5 work (ideas only unless the person's)

`docs/analysis/anyps5.md` §2 sets the rule: KytyPS5's layer of AnyPS5 is
GPL-2.0-only (ideas only); the person's AnyKyty commits are portable.

- **KytyPS5's SRT walker** (`core/kyty/graphics/shader/recompiler/ir/passes/SrtWalker.h`,
  `SrtWalker.cpp`, 2,159 lines, read at `E:\xtp-refs\AnyPS5` `7de61e3451`):
  also an x86-64 JIT walker. Two ideas shadPS4's lacks: it asks whether
  the GPU produces a page before trusting its bytes, and, when it runs
  ahead of the GPU, takes the value of a write still queued in command
  order. Both are outside AnyKyty's `#if defined(ANYPS5_KYTY)` blocks, so
  KytyPS5's: ideas only. They matter to XTP only if 5.3 (b) ever reads
  title memory rather than descriptor heaps.
- **The person's flat-table binding** (`521ca7e58f`, `f749c16ef2`,
  2026-11-21, portable): a flattened table with no searched parts and at
  most 4,096 words is bound as a uniform (constant) buffer, not a storage
  buffer, and scalar loads of read-only buffers go through uniform views.
  The commit measures 5.4 ms of GPU time for one full-screen pass at 4K
  before, because NVIDIA gave every constant a vector register and a
  memory wait. For XTP: put 5.3 (b)'s words in root constants or a root
  CBV, never in a raw SRV.
- **Lane pairs** (KytyPS5, `anyps5.md` §3, ideas only): one console wave64
  as one 32-lane host wave, each thread holding lanes i and i + 32. It
  answers masks read whole in mesh and compute (§3.4, §3.5) without
  barriers; it does not apply to pixel programs.
- AnyPS5 carries shadPS4's `fault_buffer_process.comp` (GPL-2.0-or-later),
  the shader behind §2.2's fault collection.

## 7. Licence and provenance

shadPS4's `LICENSE` is the GPL-2.0 text. Every file read carries
`SPDX-License-Identifier: GPL-2.0-or-later`, except `ir/passes/ssa_rewrite_pass.cpp`
(MIT, Philip Rebohle). `REUSE.toml` overrides nothing under
`src/shader_recompiler/`. No GPL-2.0-only file was found. GPL-2.0-or-later
code may be combined into XTP's GPL-3.0-or-later with its notices kept;
credit as "Ported from shadPS4 (<path>, commit 5c3b6910b9)".

| File (under `src/shader_recompiler/` unless `src/`) | SPDX | Copyright | Origin |
|---|---|---|---|
| `ir/passes/flatten_extended_userdata_pass.cpp` | GPL-2.0-or-later | 2024-2026 shadPS4 Emulator Project | #1165 `9ec75c3feb` (2024-11-01); last `d1cb90dbbe` |
| `ir/passes/srt.h` | GPL-2.0-or-later | 2024 shadPS4 | `9ec75c3feb`; last `a9f8eaf778` |
| `ir/srt_gvn_table.h` | GPL-2.0-or-later | 2024 shadPS4 ("inspiration from spirv-opt") | `9ec75c3feb`; last `507464a531` |
| `ir/passes/resource_discover_pass.cpp` | GPL-2.0-or-later | 2026 shadPS4 | #4782 `1cf28cbc96`; last `8bafaa5aca` |
| `ir/passes/resource_patching_pass.cpp`, `ir/passes/resource_pass.h` | GPL-2.0-or-later | 2024-2026 shadPS4 | #4782; last `f46a693c80` |
| `resource.h`, `specialization.h`, `info.h` | GPL-2.0-or-later | 2024-2026 shadPS4 | last `8bafaa5aca` (`resource.h`) |
| `ir/passes/lower_wave64_pass.cpp` | GPL-2.0-or-later | 2026 shadPS4 | #4995 `bacdcc745c`, #5116 `2f40f87b5b`; last `4788f0b108` |
| `ir/passes/readlane_elimination_pass.cpp` | GPL-2.0-or-later | 2025 shadPS4 | #2667 `1f9ac53c28`; last `eeecd103f6` |
| `ir/passes/lower_buffer_format_to_raw.cpp` | GPL-2.0-or-later | 2025 shadPS4 | #2363 `cfe249debe`; last `64c820f3a0` |
| `ir/passes/ring_access_elimination.cpp` | GPL-2.0-or-later | 2024 shadPS4 | `927bb0c175`; last `a146ff2efc` |
| `ir/passes/shared_memory_barrier_pass.cpp`, `_to_storage_pass.cpp`, `_simplify_pass.cpp` | GPL-2.0-or-later | 2024-2025 shadPS4 | last `08225c2b35`, `f3e4c49695` |
| `ir/passes/inverse_ballot_elimination_pass.cpp`, `ir/passes/phi_simplification_pass.cpp` | GPL-2.0-or-later | 2026 shadPS4 | `507464a531`; last `d9cf41ba0a`, `524be7c1d4` |
| `backend/spirv/spirv_emit_context.cpp`, `backend/spirv/emit_spirv_context_get_set.cpp`, `backend/spirv/emit_spirv_warp.cpp` | GPL-2.0-or-later | 2024-2026 shadPS4 | last `2b9ca0d9f6` (`spirv_emit_context.cpp`) |
| `src/video_core/buffer_cache/fault_manager.cpp`, `buffer_cache.cpp` | GPL-2.0-or-later | 2024-2025 shadPS4 | — |
| yuzu-derived: `ir/passes/constant_propagation_pass.cpp`, `ir/passes/lower_fp64_to_fp32.cpp`, `frontend/structured_control_flow.cpp`, `ir/condition.h`, `ir/breadth_first_search.h`, `ir/post_order.cpp`, `ir/program.cpp`, `backend/spirv/emit_spirv_atomic.cpp`, `emit_spirv_barriers.cpp`, `emit_spirv_shared_memory.cpp` | GPL-2.0-or-later | 2021–2023 yuzu Emulator Project (some with shadPS4 too) | none is part of the plan |

None of the three items of §5 needs shadPS4's code: XTP's IR, DXIL emitter
and root-signature model differ, so its passes transfer as designs. If a
worker does port lines, the notice and the credit line above go with them.

## 8. Method and limits

- shadPS4: the pass order in `recompiler.cpp`; read in full: the
  flattening, discovery, patching, wave64, readlane, buffer-format and
  LDS-barrier passes, `srt.h`, `srt_gvn_table.h`, `resource.h`,
  `resource_pass.h` and the scalar-memory translation; read in part: the
  other files of §7 (the device-address and ballot emission, the
  specialization keys, the fault manager's set-up, the ring, LDS-storage,
  phi and ballot clean-up passes). The device-address default was read in
  `src/core/emulator_settings.h` and `src/common/assert.cpp`'s failure
  path through the GitHub API, as was each file's history. Nothing was
  built or run.
- XTP: `crates/xtp-shader/src/resource.rs`, `emit/lower/smem.rs`,
  `emit/lower/mod.rs`, `emit/lower/lanes.rs`, `emit/lower/mask.rs`,
  `emit/mod.rs`, `ir/unspill.rs`; `docs/wave64-on-wave32.md`,
  `docs/analysis/shader.md`, `anyps5.md`, `mesh-wave64.md`,
  `uma-feasibility.md`; the private fork study
  (`E:\xtp-data\upstream\fork-fxceless.md` §3.2, §3.3) and the fork's
  `spill.rs` and `memory.rs` (`E:\xtp-refs\fork-fxceless`); meshdata's
  and rea's comments on d687ecdb.
- Numbers: one session's log (see the header). The 34 builder programs
  were placed by their absence from the pack, their refusal offsets and
  the fork's measurement, not by disassembly; the 23 wave64 programs'
  ballot positions were not looked at (5.2's first step); the phi and
  first-lane discrepancy of §3.6 was found, not explained.
- Not done: measuring anything on a GPU, the census's per-pass attribution
  (7d248d0b), and shadPS4's tessellation, geometry and fetch-shader paths,
  which map to none of these classes.
