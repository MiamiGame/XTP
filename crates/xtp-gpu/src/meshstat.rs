//! Draws that run on the host with nothing to show (`XTP_MESH_STATS=1`, a
//! diagnostic of diagnostics builds, off by default). The world's frame line
//! counts about 3,200 draws a frame and drops about 300 for want of a host
//! pipeline, yet half its meshes are missing: walls, shelves, floors. This
//! sorts each frame's draws and mesh dispatches by why they could not show,
//! so that one session tells which cause hides them:
//!
//! - **dropped**: no host pipeline (a vertex or mesh program refused, or not
//!   made yet), or another reason; by program.
//! - **no arguments**: an indirect execution whose records, as the GPU read
//!   them, draw nothing (a count of 0, or 0 vertices, indices, instances or
//!   groups in every record). Split by who wrote them: GPU work of its own
//!   submission, GPU work of an earlier one, or nothing XTP ran (the title's
//!   culling dispatches write them, and some do not translate: the
//!   dispatches dropped earlier in the frame are named beside them).
//! - **indices zero**, **vertices zero**: an indexed draw whose indices, or
//!   a draw whose every vertex buffer, hold only zeros where the GPU read
//!   them: data never written, or not where the draw looks.
//! - **stale**: a range a draw reads (vertex, index or buffer views, root
//!   views, indirect arguments) whose host copy differs from title memory
//!   once its submission's refreshes are done, where no GPU work still in
//!   flight writes it: a write the tracking missed. Counted again if the
//!   page also shows at other addresses (`watch::views_of`): the aliases of
//!   fxceless's VIctory fork #571, where a DirectStorage write into a
//!   physical page was signalled at one of its views only.
//! - **another copy**: a range a draw reads that GPU work earlier in its
//!   submission writes through another host copy of the same memory
//!   (another title buffer over it): the draw's copy took title memory
//!   before that work ran. The path of GPU-written vertices (skinning).
//! - **has data**: none of these.
//!
//! Beside the classes: mesh dispatches of one group, (1, 1, 1), direct or
//! every record of an indirect one, by program. xcl #724 found meshes drawn
//! so empty, from a descriptor table read one register too late (a mesh
//! program's SGPR base off by one; its ideas only, none of its code).
//!
//! Three moments. Recording: each draw, mesh dispatch, dispatch and indirect
//! execution notes what it reads and what its shaders write (`draw.rs` calls
//! [`start`], [`read`], [`wrote`], [`root_view`]), into its list's recording
//! ([`Probe`], `HostList::mesh_probe`). Submission, once the refreshes are
//! done ([`submitting`]): host copies against title memory, in the order the
//! GPU runs the work. Completion, before the copy-back ([`finished`]): what
//! the GPU read, its own writes in the submission included. A frame line
//! sums it up ([`log`]), with the programs most often in each class and the
//! first few cases of each in full.
//!
//! shortcut: a range is sampled, its first [`SAMPLE`] bytes (a direct
//! draw's from its first vertex or index): a range zero there but written
//! further on counts as zero, and a missed write past the sample is not
//! seen.
//!
//! shortcut: the writers XTP sees are the unordered-access buffer views of
//! the work that ran; GPU buffer copies are not among them, but at
//! completion arguments that differ from title memory count as written by
//! their submission whatever wrote them.
//!
//! shortcut: copies in the card's memory (`XTP_CARD_BUFFERS=1`) cannot be
//! mapped: their ranges are counted as not checked.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;

use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// Bytes of each range read and compared.
pub(crate) const SAMPLE: usize = 4096;

/// Records of an indirect execution read at completion, at most.
const RECORDS: usize = 64;

/// Cases of each class logged in full, over the run.
const EXAMPLES: u32 = 4;

/// Whether the diagnostic is on (`XTP_MESH_STATS=1`).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XTP_MESH_STATS").is_ok_and(|v| v.trim() == "1"))
}

/// What a range read is to the work that reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum What {
    /// A vertex buffer view, its stride.
    Vertex { stride: u32 },
    /// The index buffer view, its format (`DXGI_FORMAT`).
    Index { format: u32 },
    /// A buffer view of a table, or a root view.
    View,
    /// Indirect records.
    Args,
    /// An indirect execution's record count.
    Count,
}

/// The work being recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Draw {
        vertices: u32,
        first: u32,
    },
    Indexed {
        indices: u32,
        first: u32,
    },
    /// `groups`: the groups in all (the product of the three dimensions).
    Mesh {
        groups: u32,
    },
    /// PC numbering of `kind`: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh
    /// dispatch.
    Indirect {
        kind: u32,
        stride: u32,
        max: u32,
    },
    Dispatch,
}

impl Op {
    /// Draws and mesh dispatches, direct or indirect: work that shows.
    fn draws(self) -> bool {
        !matches!(self, Op::Dispatch | Op::Indirect { kind: 2, .. })
    }

    fn label(self) -> &'static str {
        match self {
            Op::Draw { .. } => "draw",
            Op::Indexed { .. } => "indexed draw",
            Op::Mesh { .. } => "mesh dispatch",
            Op::Indirect { kind: 0, .. } => "indirect draw",
            Op::Indirect { kind: 1, .. } => "indirect indexed draw",
            Op::Indirect { kind: 2, .. } => "indirect dispatch",
            Op::Indirect { kind: 10, .. } => "indirect mesh dispatch",
            Op::Indirect { .. } => "indirect, another kind",
            Op::Dispatch => "dispatch",
        }
    }
}

/// A range of a host copy that work reads or writes.
#[derive(Clone)]
struct Range {
    what: What,
    shadow: Arc<Shadow>,
    address: u64,
    len: u64,
}

/// What one draw, dispatch or indirect execution of a list read and wrote,
/// as recorded; a dispatch that did not run keeps only its program, so that
/// the executions after it can name it.
#[derive(Clone)]
pub(crate) struct Probe {
    op: Op,
    ran: bool,
    program: Program,
    reads: Vec<Range>,
    writes: Vec<Range>,
}

/// The ranges of the work being recorded on this thread, from [`start`] to
/// the drop of what it returns.
#[derive(Default)]
struct Open {
    reads: Vec<Range>,
    writes: Vec<Range>,
}

thread_local! {
    static OPEN: RefCell<Option<Open>> = const { RefCell::new(None) };
}

/// Work being recorded into `list`, from [`start`]: its ranges go into the
/// list when this drops, if it ran ([`Started::ran`]); a draw that did not
/// run is counted as dropped.
pub(crate) struct Started<'a> {
    list: Option<&'a HostList>,
    pipeline: usize,
    op: Op,
    ran: bool,
}

/// Starts recording `op` of `pipeline` (a title pipeline) into `list`; the
/// ranges [`read`] and [`wrote`] note until the returned value drops are
/// its. Nothing is noted while the diagnostic is off.
pub(crate) fn start(list: &HostList, pipeline: usize, op: Op) -> Started<'_> {
    let list = on().then_some(list);
    if list.is_some() {
        OPEN.with(|o| *o.borrow_mut() = Some(Open::default()));
    }
    Started {
        list,
        pipeline,
        op,
        ran: false,
    }
}

impl Started<'_> {
    /// The work runs on the host.
    pub(crate) fn ran(&mut self) {
        self.ran = true;
    }
}

impl Drop for Started<'_> {
    fn drop(&mut self) {
        let Some(list) = self.list else {
            return;
        };
        let open = OPEN.with(|o| o.borrow_mut().take()).unwrap_or_default();
        let program = program_of(self.pipeline);
        if !self.ran && self.op.draws() {
            let no_pipeline = crate::pipeline::host_pipeline(self.pipeline as This).is_none();
            stats().dropped(self.op, program, no_pipeline);
            return;
        }
        let (reads, writes) = if self.ran {
            (open.reads, open.writes)
        } else {
            (Vec::new(), Vec::new())
        };
        list.mesh_probe(Probe {
            op: self.op,
            ran: self.ran,
            program,
            reads,
            writes,
        });
    }
}

/// What work is known by: its host pipeline's first shader (vertex, mesh
/// or compute), or without a host pipeline the title pipeline's address,
/// which the log's `first bind` lines give the shaders of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Program {
    Shader(u64),
    Pipeline(usize),
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Program::Shader(hash) => write!(f, "{hash:016x}"),
            Program::Pipeline(at) => write!(f, "pipeline {at:#x}"),
        }
    }
}

fn program_of(pipeline: usize) -> Program {
    crate::pipeline::host_pipeline(pipeline as This)
        .and_then(|p| p.stages.first())
        .and_then(|s| u64::from_str_radix(&s.shader.hash, 16).ok())
        .map_or(Program::Pipeline(pipeline), Program::Shader)
}

/// The work being recorded reads `[address, address + len)` of `shadow`.
pub(crate) fn read(what: What, shadow: &Arc<Shadow>, address: u64, len: u64) {
    note(false, what, shadow, address, len);
}

/// The work being recorded may write `[address, address + len)` of
/// `shadow` (unordered access).
pub(crate) fn wrote(shadow: &Arc<Shadow>, address: u64, len: u64) {
    note(true, What::View, shadow, address, len);
}

/// The work being recorded reads a root view at `address`, if its buffer
/// has a host copy already (the first use makes it after this).
pub(crate) fn root_view(address: u64) {
    if !on() {
        return;
    }
    if let Some(shadow) = crate::shadow::existing(address, 1) {
        let len = shadow.left(address).min(SAMPLE as u64);
        read(What::View, &shadow, address, len);
    }
}

fn note(write: bool, what: What, shadow: &Arc<Shadow>, address: u64, len: u64) {
    if !on() || len == 0 {
        return;
    }
    OPEN.with(|o| {
        if let Some(open) = o.borrow_mut().as_mut() {
            let range = Range {
                what,
                shadow: shadow.clone(),
                address,
                len,
            };
            if write {
                open.writes.push(range);
            } else {
                open.reads.push(range);
            }
        }
    });
}

/// Why a draw that ran showed nothing, or that it had data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Class {
    /// Its records draw nothing; nothing XTP ran wrote them.
    NoArgsUnwritten,
    /// Its records draw nothing; GPU work of its submission wrote them.
    NoArgsHere,
    /// Its records draw nothing; GPU work of an earlier submission wrote
    /// them.
    NoArgsBefore,
    IndicesZero,
    VerticesZero,
    Stale,
    OtherCopy,
    HasData,
    /// Dropped: no host pipeline.
    NoPipeline,
    /// Dropped for another reason.
    Dropped,
}

const CLASSES: usize = 10;

impl Class {
    fn text(self) -> &'static str {
        match self {
            Class::NoArgsUnwritten => "without arguments, not written by GPU work that ran",
            Class::NoArgsHere => "without arguments, written by its own submission",
            Class::NoArgsBefore => "without arguments, written by an earlier submission",
            Class::IndicesZero => "indices all zero",
            Class::VerticesZero => "vertices all zero",
            Class::Stale => "a host copy older than title memory",
            Class::OtherCopy => "GPU output of its submission through another host copy",
            Class::HasData => "has data",
            Class::NoPipeline => "dropped without a host pipeline",
            Class::Dropped => "dropped otherwise",
        }
    }
}

/// Who last wrote a range on the GPU, as recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Writer {
    /// The host copy written (its address).
    shadow: usize,
    program: Program,
    submission: u64,
    /// The bytes of the view it wrote through (the world declares views of
    /// its 174 MB heap, over everything).
    declared: u64,
}

/// Buffer views written through views up to this size are followed to the
/// draws that read them ([`Stats::fed`]); bigger ones cover too much.
const FOLLOWED: u64 = 64 << 20;

/// How a draw reads GPU output: through vertex or index buffers (else a
/// buffer view), through the host copy the GPU wrote (else another), in the
/// submission that wrote it (else a later one).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Fed {
    writer: Program,
    reader: Program,
    label: &'static str,
    buffers: bool,
    same_copy: bool,
    same_submission: bool,
}

/// Disjoint ranges of title memory `[start, end)` by start, each with its
/// last writer.
type Written = BTreeMap<u64, (u64, Writer)>;

/// Notes `[start, end)` as written by `writer`, over what it held.
fn assign(map: &mut Written, start: u64, end: u64, writer: Writer) {
    if start >= end {
        return;
    }
    if let Some((&s, &(e, old))) = map.range(..start).next_back()
        && e > start
    {
        map.insert(s, (start, old));
        if e > end {
            map.insert(end, (e, old));
        }
    }
    let inside: Vec<u64> = map.range(start..end).map(|(&s, _)| s).collect();
    for s in inside {
        if let Some((e, old)) = map.remove(&s)
            && e > end
        {
            map.insert(end, (e, old));
        }
    }
    map.insert(start, (end, writer));
}

/// The writers of the parts of `[start, end)` that were written.
fn writers(map: &Written, start: u64, end: u64) -> Vec<Writer> {
    let before = map
        .range(..start)
        .next_back()
        .filter(|(_, (e, _))| *e > start);
    before
        .into_iter()
        .chain(map.range(start..end))
        .map(|(_, (_, w))| *w)
        .collect()
}

/// What a submission found of one draw that ran, kept for its completion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Verdict {
    /// A range differs from title memory where no GPU work in flight
    /// writes it, and whether on a page shown at other addresses too.
    stale: bool,
    stale_aliased: bool,
    /// GPU work earlier in the submission writes a range it reads through
    /// another host copy.
    other_copy: bool,
    /// Its records: written by GPU work of its submission, of an earlier
    /// one.
    args_here: bool,
    args_before: bool,
    /// The dispatches dropped earlier in its frame (programs).
    dropped_before: Vec<Program>,
    /// A first stale range: its address, bytes that differ, other views.
    first_stale: Option<(u64, usize, usize)>,
    /// The GPU output it reads (reader and label left to [`finished`]).
    fed: Vec<Fed>,
}

#[derive(Default)]
struct Counts {
    /// Work that ran, by [`Op::label`].
    ran: BTreeMap<&'static str, u64>,
    classes: [u64; CLASSES],
    stale_aliased: u64,
    /// Ranges compared at submission; of them on pages shown at other
    /// addresses too; not compared (card copies, unreadable memory).
    reads: u64,
    aliased_reads: u64,
    unchecked: u64,
}

/// The diagnostic's state.
#[derive(Default)]
struct Stats {
    counts: Counts,
    /// Per class, program and kind of work: how often, since the last line.
    top: HashMap<(Class, Program, &'static str), u64>,
    /// Dropped dispatches before executions without arguments that nothing
    /// ran wrote: how often, by program, since the last line.
    culprits: HashMap<Program, u64>,
    /// Draws reading GPU output (skinned vertices, culled arguments): how
    /// often, since the last line.
    fed: HashMap<Fed, u64>,
    /// Mesh dispatches of one group ([`one_group`]), by program and kind.
    single: HashMap<(Program, &'static str), u64>,
    written: Written,
    /// Submissions prepared and not finished, with what they found, in the
    /// order of their probes.
    pending: HashMap<u64, Vec<Option<Verdict>>>,
    in_flight: BTreeSet<u64>,
    /// The dispatches dropped in the current frame: its number, programs.
    dropped: (u64, Vec<Program>),
    examples: [u32; CLASSES],
}

fn stats() -> std::sync::MutexGuard<'static, Stats> {
    static STATS: OnceLock<Mutex<Stats>> = OnceLock::new();
    STATS
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// The most ranges [`Stats::written`] keeps before it starts again.
const WRITTEN_MAX: usize = 1 << 20;

impl Stats {
    fn dropped(&mut self, op: Op, program: Program, no_pipeline: bool) {
        let class = if no_pipeline {
            Class::NoPipeline
        } else {
            Class::Dropped
        };
        self.count(class, program, op.label());
    }

    fn count(&mut self, class: Class, program: Program, label: &'static str) {
        self.counts.classes[class as usize] += 1;
        *self.top.entry((class, program, label)).or_default() += 1;
    }

    fn note_written(&mut self, ranges: &[Range], program: Program, submission: u64) {
        if self.written.len() > WRITTEN_MAX {
            self.written.clear();
        }
        for r in ranges {
            let writer = Writer {
                shadow: Arc::as_ptr(&r.shadow) as usize,
                program,
                submission,
                declared: r.len,
            };
            assign(&mut self.written, r.address, r.address + r.len, writer);
        }
    }
}

/// The bytes of `[address, address + n)` of a host copy, as the CPU sees
/// them; `None` for a copy in the card's memory or outside the copy.
fn host_bytes(shadow: &Shadow, address: u64, n: usize) -> Option<Vec<u8>> {
    let n = n.min(usize::try_from(shadow.left(address)).ok()?);
    if shadow.carded() || n == 0 {
        return None;
    }
    let offset = usize::try_from(shadow.offset(address)).ok()?;
    let mut data = std::ptr::null_mut();
    // SAFETY: a buffer in system memory, mapped persistently already (maps
    // nest); nothing is written through this mapping.
    unsafe { shadow.buffer.Map(0, None, Some(&mut data)) }.ok()?;
    // SAFETY: `offset + n` bytes lie in the buffer (`left`).
    let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>().add(offset), n) }.to_vec();
    // SAFETY: as above; the empty range says nothing was written.
    unsafe {
        shadow
            .buffer
            .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }))
    };
    Some(bytes)
}

/// Title memory `[address, address + n)`; `None` where it cannot be read
/// (unmapped since, or never committed).
fn title_bytes(address: u64, n: usize) -> Option<Vec<u8>> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let _pin = crate::pin_copy_back(address as usize, n);
    let mut out = vec![0u8; n];
    let mut read = 0usize;
    // SAFETY: the kernel copies into our buffer of `n` bytes, and fails
    // instead of faulting where the source cannot be read.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            address as *const _,
            out.as_mut_ptr().cast(),
            n,
            &mut read,
        )
    } != 0;
    (ok && read == n).then_some(out)
}

fn zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Bytes per index of an index format.
fn index_size(format: u32) -> u64 {
    // DXGI_FORMAT_R16_UINT; R32_UINT otherwise.
    if format == 57 { 2 } else { 4 }
}

/// Where in a range `op` reads first, and how many bytes are sampled there.
fn sample(op: Op, r: &Range) -> (u64, usize) {
    sample_at(op, r.what, r.address, r.len)
}

fn sample_at(op: Op, what: What, address: u64, len: u64) -> (u64, usize) {
    let skip = match (op, what) {
        (Op::Draw { vertices, first }, What::Vertex { stride }) => Some((
            u64::from(first) * u64::from(stride),
            u64::from(vertices) * u64::from(stride),
        )),
        (Op::Indexed { indices, first }, What::Index { format }) => {
            let size = index_size(format);
            Some((u64::from(first) * size, u64::from(indices) * size))
        }
        _ => None,
    };
    let (from, n) = match skip {
        Some((from, n)) if from < len => (from, n.min(len - from)),
        Some(_) => (len, 0),
        None => (0, len),
    };
    (address + from, n.min(SAMPLE as u64) as usize)
}

/// An indirect execution's records as read: whether every active one does
/// nothing, and the first that does something's first word and count
/// (start vertex or index, and vertices or indices), for draws.
fn records_empty(
    kind: u32,
    stride: u32,
    max: u32,
    bytes: &[u8],
    count: Option<u32>,
) -> (bool, Option<(u32, u32)>) {
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let active = count.map_or(max, |c| c.min(max)) as usize;
    let stride = stride.max(4) as usize;
    let seen = active.min(bytes.len() / stride).min(RECORDS);
    for i in 0..seen {
        let at = i * stride;
        match kind {
            // Count, instances, start, ...
            0 | 1 => {
                if word(at) != 0 && word(at + 4) != 0 {
                    return (false, Some((word(at + 8), word(at))));
                }
            }
            _ => {
                if word(at) != 0 && word(at + 4) != 0 && word(at + 8) != 0 {
                    return (false, None);
                }
            }
        }
    }
    // Records past those read are taken as drawing something.
    (seen == active, None)
}

/// Whether every active record of an indirect dispatch, as read, is one
/// group, (1, 1, 1): xcl #724's empty meshes were drawn so, from a
/// descriptor table read one register too late (a mesh program's SGPR base
/// off by one). False for no active record or records past those read.
fn records_one_group(stride: u32, max: u32, bytes: &[u8], count: Option<u32>) -> bool {
    let stride = stride.max(12) as usize;
    let active = count.map_or(max, |c| c.min(max)) as usize;
    let seen = active.min(bytes.len() / stride).min(RECORDS);
    let one = |at: usize| {
        bytes
            .get(at..at + 12)
            .is_some_and(|b| b.chunks(4).all(|w| w == [1, 0, 0, 0]))
    };
    seen > 0 && seen == active && (0..seen).all(|i| one(i * stride))
}

/// Whether a mesh dispatch that ran was one group (direct), or every record
/// of an indirect one is, as the GPU left them.
fn one_group(probe: &Probe) -> bool {
    match probe.op {
        Op::Mesh { groups } => groups == 1,
        Op::Indirect {
            kind: 10,
            stride,
            max,
        } => {
            let find = |what: What| probe.reads.iter().find(|r| r.what == what);
            let count = find(What::Count)
                .and_then(|r| host_bytes(&r.shadow, r.address, 4))
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            find(What::Args).is_some_and(|r| {
                let n = (stride.max(12) as usize)
                    .saturating_mul(RECORDS)
                    .min(r.len as usize);
                host_bytes(&r.shadow, r.address, n)
                    .is_some_and(|b| records_one_group(stride, max, &b, count))
            })
        }
        _ => false,
    }
}

/// A submission's lists are about to run, their refreshes done: their
/// draws' ranges are compared with title memory, in the order the GPU runs
/// them.
pub(crate) fn submitting(runs: &[&HostList], submission: u64) {
    if !on() {
        return;
    }
    let frame = crate::frame::current();
    let mut s = stats();
    let s = &mut *s;
    if s.dropped.0 != frame {
        s.dropped = (frame, Vec::new());
    }
    let mut verdicts = Vec::new();
    let mut cache = Compared::new();
    for list in runs {
        for probe in list.mesh_probes().iter() {
            if !probe.ran {
                if !s.dropped.1.contains(&probe.program) {
                    s.dropped.1.push(probe.program);
                }
                verdicts.push(None);
                continue;
            }
            if !probe.op.draws() {
                s.note_written(&probe.writes, probe.program, submission);
                verdicts.push(None);
                continue;
            }
            let v = judge(s, probe, submission, &mut cache);
            s.note_written(&probe.writes, probe.program, submission);
            verdicts.push(Some(v));
        }
    }
    s.pending.insert(submission, verdicts);
    s.in_flight.insert(submission);
}

/// What a draw's ranges hold at its submission.
fn judge(s: &mut Stats, probe: &Probe, submission: u64, cache: &mut Compared) -> Verdict {
    let mut v = Verdict {
        dropped_before: s.dropped.1.clone(),
        ..Verdict::default()
    };
    for r in &probe.reads {
        let (at, n) = sample(probe.op, r);
        if n == 0 {
            continue;
        }
        let shadow = Arc::as_ptr(&r.shadow) as usize;
        let mut in_flight = false;
        let buffers = matches!(r.what, What::Vertex { .. } | What::Index { .. });
        for w in writers(&s.written, at, at + n as u64) {
            if w.submission == submission {
                v.other_copy |= w.shadow != shadow;
                v.args_here |= r.what == What::Args;
            } else {
                in_flight |= s.in_flight.contains(&w.submission);
                v.args_before |= r.what == What::Args;
            }
            let followed = buffers || (r.what == What::View && w.declared <= FOLLOWED);
            let fed = Fed {
                writer: w.program,
                reader: Program::Shader(0),
                label: "",
                buffers,
                same_copy: w.shadow == shadow,
                same_submission: w.submission == submission,
            };
            if followed && !v.fed.contains(&fed) {
                v.fed.push(fed);
            }
        }
        // Each range once a submission: draws share most of theirs.
        let compared = *cache.entry((shadow, at, n)).or_insert_with(|| {
            let views = crate::watch::views_of(at as usize);
            let (host, title) = (host_bytes(&r.shadow, at, n)?, title_bytes(at, n)?);
            let differ = host.iter().zip(&title).filter(|(a, b)| a != b).count();
            Some((differ, views))
        });
        let Some((differ, views)) = compared else {
            s.counts.unchecked += 1;
            continue;
        };
        s.counts.reads += 1;
        s.counts.aliased_reads += u64::from(views > 0);
        if differ > 0 && !in_flight {
            v.stale = true;
            v.stale_aliased |= views > 0;
            v.first_stale.get_or_insert((at, differ, views));
        }
    }
    v
}

/// What comparing a range found in a submission, by (host copy, address,
/// bytes): the bytes that differ and the page's other views, or `None` if
/// it could not be read.
type Compared = HashMap<(usize, u64, usize), Option<(usize, usize)>>;

/// A submission has run, its copy-back not done yet: what its draws read,
/// as the GPU left it, decides their class.
pub(crate) fn finished<'a>(submission: u64, probes: impl Iterator<Item = &'a Arc<Vec<Probe>>>) {
    if !on() {
        return;
    }
    let mut s = stats();
    let s = &mut *s;
    s.in_flight.remove(&submission);
    let Some(verdicts) = s.pending.remove(&submission) else {
        return;
    };
    let all = probes.flat_map(|p| p.iter());
    for (probe, verdict) in all.zip(verdicts) {
        let Some(v) = verdict else {
            continue;
        };
        *s.counts.ran.entry(probe.op.label()).or_default() += 1;
        let (class, detail) = classify(probe, &v);
        if one_group(probe) {
            *s.single
                .entry((probe.program, probe.op.label()))
                .or_default() += 1;
        }
        s.count(class, probe.program, probe.op.label());
        if v.stale_aliased {
            s.counts.stale_aliased += 1;
        }
        if class == Class::NoArgsUnwritten {
            for &program in &v.dropped_before {
                *s.culprits.entry(program).or_default() += 1;
            }
        }
        for fed in &v.fed {
            let fed = Fed {
                reader: probe.program,
                label: probe.op.label(),
                ..*fed
            };
            *s.fed.entry(fed).or_default() += 1;
        }
        let example = &mut s.examples[class as usize];
        if class != Class::HasData && *example < EXAMPLES {
            *example += 1;
            let first = probe.reads.first().map_or(0, |r| r.address);
            tracing::info!(
                class = class.text(),
                op = probe.op.label(),
                program = %probe.program,
                at = format_args!("{first:#x}"),
                reads = probe.reads.len(),
                stale = ?v.first_stale.map(|(a, d, w)| format!("{a:#x}: {d} bytes, {w} other views")),
                dropped_before = ?v.dropped_before.iter().take(8).map(ToString::to_string).collect::<Vec<_>>(),
                "{detail}meshes: a case"
            );
        }
    }
}

/// A draw's class at completion, and a detail for its log line.
fn classify(probe: &Probe, v: &Verdict) -> (Class, String) {
    let find = |what: fn(What) -> bool| probe.reads.iter().find(|r| what(r.what));
    let mut first_index = None;
    if let Op::Indirect { kind, stride, max } = probe.op {
        let records = find(|w| w == What::Args).and_then(|r| {
            let n = (stride.max(4) as usize)
                .saturating_mul(RECORDS)
                .min(r.len as usize);
            Some((r, host_bytes(&r.shadow, r.address, n)?))
        });
        let count = find(|w| w == What::Count)
            .and_then(|r| host_bytes(&r.shadow, r.address, 4))
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        if let Some((r, bytes)) = records {
            let (empty, first) = records_empty(kind, stride, max, &bytes, count);
            if empty {
                // Arguments that differ from title memory before the
                // copy-back were written by this submission's GPU work.
                let here =
                    v.args_here || title_bytes(r.address, bytes.len()).is_some_and(|t| t != bytes);
                let class = match (here, v.args_before) {
                    (true, _) => Class::NoArgsHere,
                    (false, true) => Class::NoArgsBefore,
                    (false, false) => Class::NoArgsUnwritten,
                };
                return (class, format!("count {count:?}, max {max}: "));
            }
            first_index = first;
        }
    }
    let indexed = match probe.op {
        Op::Indexed { first, indices } => Some((first, indices)),
        Op::Indirect { kind: 1, .. } => first_index,
        _ => None,
    };
    if let Some((first, indices)) = indexed
        && let Some(r) = find(|w| matches!(w, What::Index { .. }))
        && let What::Index { format } = r.what
    {
        let size = index_size(format);
        let from = u64::from(first) * size;
        let n = (u64::from(indices) * size).min(SAMPLE as u64) as usize;
        if from < r.len
            && let Some(bytes) = host_bytes(&r.shadow, r.address + from, n)
            && zero(&bytes)
        {
            return (Class::IndicesZero, String::new());
        }
    }
    let vertices: Vec<&Range> = probe
        .reads
        .iter()
        .filter(|r| matches!(r.what, What::Vertex { .. }))
        .collect();
    let vertex_zero = |r: &&Range| {
        let (at, n) = sample(probe.op, r);
        n > 0 && host_bytes(&r.shadow, at, n).is_some_and(|b| zero(&b))
    };
    if !vertices.is_empty() && vertices.iter().all(vertex_zero) {
        return (Class::VerticesZero, String::new());
    }
    if v.stale {
        return (Class::Stale, String::new());
    }
    if v.other_copy {
        return (Class::OtherCopy, String::new());
    }
    (Class::HasData, String::new())
}

/// What the frame line sums up: the counts since the last [`take`].
#[derive(Default)]
pub(crate) struct Taken {
    counts: Counts,
    top: HashMap<(Class, Program, &'static str), u64>,
    culprits: HashMap<Program, u64>,
    fed: HashMap<Fed, u64>,
    single: HashMap<(Program, &'static str), u64>,
}

/// The counts since the last call, over the frames the frame-rate line
/// counts them for (it takes them every second, and logs one in a few).
pub(crate) fn take() -> Taken {
    if !on() {
        return Taken::default();
    }
    let mut s = stats();
    Taken {
        counts: std::mem::take(&mut s.counts),
        top: std::mem::take(&mut s.top),
        culprits: std::mem::take(&mut s.culprits),
        fed: std::mem::take(&mut s.fed),
        single: std::mem::take(&mut s.single),
    }
}

/// The frame line of what [`take`] took, over `frames` frames, then the
/// programs most often in each class.
pub(crate) fn log(taken: Taken, frames: f64) {
    if !on() || frames <= 0.0 {
        return;
    }
    let Taken {
        counts,
        top,
        culprits,
        fed,
        single,
    } = taken;
    let per = |n: u64| n as f64 / frames;
    let c = |class: Class| per(counts.classes[class as usize]);
    let ran: u64 = counts.ran.values().sum();
    let kinds: Vec<String> = counts
        .ran
        .iter()
        .map(|(label, n)| format!("{label} {:.0}", per(*n)))
        .collect();
    tracing::info!(
        "meshes per frame: {:.0} ran ({}); nothing to draw: {:.1} without arguments ({:.1} not \
         written by GPU work that ran, {:.1} written by their own submission, {:.1} by an \
         earlier one), {:.1} indices all zero, {:.1} vertices all zero; {:.1} read a host copy \
         older than title memory ({:.1} on pages shown at other addresses too), {:.1} GPU output \
         of their submission through another host copy; {:.1} have data; dropped {:.1} without a \
         host pipeline, {:.1} otherwise; ranges compared {:.0} ({:.0} on pages shown at other \
         addresses too), not compared {:.0}",
        per(ran),
        kinds.join(", "),
        c(Class::NoArgsUnwritten) + c(Class::NoArgsHere) + c(Class::NoArgsBefore),
        c(Class::NoArgsUnwritten),
        c(Class::NoArgsHere),
        c(Class::NoArgsBefore),
        c(Class::IndicesZero),
        c(Class::VerticesZero),
        c(Class::Stale),
        per(counts.stale_aliased),
        c(Class::OtherCopy),
        c(Class::HasData),
        c(Class::NoPipeline),
        c(Class::Dropped),
        per(counts.reads),
        per(counts.aliased_reads),
        per(counts.unchecked),
    );
    let mut top: Vec<((Class, Program, &'static str), u64)> = top.into_iter().collect();
    top.sort_unstable_by(|a, b| a.0.0.cmp(&b.0.0).then(b.1.cmp(&a.1)).then(a.0.cmp(&b.0)));
    let mut shown = [0; CLASSES];
    for ((class, program, label), n) in top {
        let seen = &mut shown[class as usize];
        if class == Class::HasData || *seen == 6 {
            continue;
        }
        *seen += 1;
        tracing::info!(
            "meshes per frame: {}: {label} {program}: {:.1}",
            class.text(),
            per(n)
        );
    }
    let mut culprits: Vec<(Program, u64)> = culprits.into_iter().collect();
    culprits.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    if !culprits.is_empty() {
        let named: Vec<String> = culprits
            .iter()
            .take(8)
            .map(|(program, n)| format!("{program} {:.1}", per(*n)))
            .collect();
        tracing::info!(
            "meshes per frame: dispatches dropped earlier in the frame of executions without \
             arguments that nothing ran wrote: {}",
            named.join(", ")
        );
    }
    // The GPU output draws read, by writer: vertex and index buffers first,
    // then buffer views.
    let mut fed: Vec<(Fed, u64)> = fed.into_iter().collect();
    fed.sort_unstable_by(|a, b| {
        b.0.buffers
            .cmp(&a.0.buffers)
            .then(b.1.cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });
    let mut shown = [0; 2];
    for (f, n) in fed {
        let seen = &mut shown[usize::from(f.buffers)];
        if *seen == 8 {
            continue;
        }
        *seen += 1;
        tracing::info!(
            "meshes per frame: GPU output of {} read through {} by {} {}, through {} \
             host copy, {} submission: {:.1}",
            f.writer,
            if f.buffers {
                "vertex or index buffers"
            } else {
                "buffer views"
            },
            f.label,
            f.reader,
            if f.same_copy { "the same" } else { "another" },
            if f.same_submission {
                "in its"
            } else {
                "after its"
            },
            per(n)
        );
    }
    // Mesh dispatches of one group, by program (xcl #724).
    let mut single: Vec<((Program, &'static str), u64)> = single.into_iter().collect();
    single.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for ((program, label), n) in single.into_iter().take(8) {
        tracing::info!(
            "meshes per frame: {label} of one group (1, 1, 1): {program}: {:.1}",
            per(n)
        );
    }
}

/// For DirectStorage's raw destinations (title addresses): of the 64 KiB
/// pages of `[at, at + len)`, how many the title shows at other addresses
/// too, and at how many other addresses in all (`watch::views_of`).
pub fn other_views(at: usize, len: usize) -> (usize, usize) {
    const PAGE: usize = crate::watch::PAGE;
    let (first, end) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    (first..end)
        .map(|page| crate::watch::views_of(page * PAGE))
        .filter(|&n| n > 0)
        .fold((0, 0), |(pages, views), n| (pages + 1, views + n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(shadow: usize, submission: u64) -> Writer {
        Writer {
            shadow,
            program: Program::Shader(0),
            submission,
            declared: 0,
        }
    }

    #[test]
    fn assigned_ranges_stay_disjoint_with_the_last_writer() {
        let mut map = Written::new();
        assign(&mut map, 100, 200, writer(1, 1));
        assign(&mut map, 150, 160, writer(2, 2));
        assign(&mut map, 190, 300, writer(3, 3));
        let spans: Vec<(u64, u64, usize)> =
            map.iter().map(|(&s, &(e, w))| (s, e, w.shadow)).collect();
        assert_eq!(
            spans,
            [(100, 150, 1), (150, 160, 2), (160, 190, 1), (190, 300, 3)]
        );
        let shadows = |map: &Written, a, b| -> Vec<usize> {
            writers(map, a, b).iter().map(|w| w.shadow).collect()
        };
        assert_eq!(shadows(&map, 155, 156), [2]);
        assert_eq!(shadows(&map, 140, 195), [1, 2, 1, 3]);
        assert_eq!(shadows(&map, 0, 100), Vec::<usize>::new());
        assert_eq!(shadows(&map, 300, 400), Vec::<usize>::new());
        // A range over all of them replaces them.
        assign(&mut map, 50, 400, writer(4, 4));
        assert_eq!(map.len(), 1);
        assert_eq!(shadows(&map, 120, 121), [4]);
    }

    fn words(records: &[[u32; 5]], stride: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for r in records {
            let mut bytes: Vec<u8> = r.iter().flat_map(|w| w.to_le_bytes()).collect();
            bytes.resize(stride, 0);
            out.extend(bytes);
        }
        out
    }

    #[test]
    fn records_that_draw_nothing_are_told_apart() {
        let none = words(&[[0, 1, 0, 0, 0], [5, 0, 0, 0, 0]], 20);
        assert_eq!(records_empty(1, 20, 2, &none, None), (true, None));
        let one = words(&[[0, 1, 0, 0, 0], [36, 2, 600, 0, 0]], 20);
        assert_eq!(
            records_empty(1, 20, 2, &one, None),
            (false, Some((600, 36)))
        );
        // A count of 1 leaves only the first record, which draws nothing.
        assert_eq!(records_empty(1, 20, 2, &one, Some(1)), (true, None));
        assert_eq!(records_empty(1, 20, 2, &one, Some(0)), (true, None));
        // Groups: any zero dimension dispatches nothing.
        let groups = words(&[[4, 1, 0, 0, 0], [4, 1, 1, 0, 0]], 12);
        assert_eq!(records_empty(10, 12, 1, &groups, None), (true, None));
        assert_eq!(records_empty(10, 12, 2, &groups, None), (false, None));
        // More records than read: the rest are taken as drawing.
        let empty = vec![0u8; 16 * RECORDS];
        assert_eq!(records_empty(0, 16, 1000, &empty, None), (false, None));
    }

    #[test]
    fn dispatches_of_one_group_are_told_apart() {
        let ones = words(&[[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], 12);
        assert!(records_one_group(12, 2, &ones, None));
        let mixed = words(&[[1, 1, 1, 0, 0], [2, 1, 1, 0, 0]], 12);
        assert!(!records_one_group(12, 2, &mixed, None));
        // Only the active records count; none active is not one group.
        assert!(records_one_group(12, 2, &mixed, Some(1)));
        assert!(!records_one_group(12, 2, &ones, Some(0)));
        // Records past those read: not known to be one group.
        assert!(!records_one_group(12, 1000, &ones, None));
    }

    #[test]
    fn samples_start_where_direct_draws_read() {
        let draw = Op::Draw {
            vertices: 3,
            first: 10,
        };
        let vertex = What::Vertex { stride: 32 };
        assert_eq!(sample_at(draw, vertex, 0x1000, 0x10000), (0x1000 + 320, 96));
        // Past the view: nothing read there.
        assert_eq!(sample_at(draw, vertex, 0x1000, 320), (0x1000 + 320, 0));
        let indexed = Op::Indexed {
            indices: 100_000,
            first: 6,
        };
        let r16 = What::Index { format: 57 };
        assert_eq!(sample_at(indexed, r16, 0x2000, 1 << 20), (0x200c, SAMPLE));
        assert_eq!(index_size(42), 4);
        // Other ranges from their start, at most a sample.
        assert_eq!(
            sample_at(Op::Mesh { groups: 1 }, What::View, 0x3000, 100),
            (0x3000, 100)
        );
        assert_eq!(
            sample_at(indexed, vertex, 0x3000, 1 << 20),
            (0x3000, SAMPLE)
        );
    }
}
