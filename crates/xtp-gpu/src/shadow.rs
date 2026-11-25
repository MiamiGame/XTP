//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. After a submission, [`crate::readmap`] keeps a completed
//! physical page behind inaccessible title views until its first CPU access.
//! Learned pages, and memory whose aliases cannot all be guarded, receive
//! the result during submission completion ([`Shadow::gpu_writes`]).
//!
//! A range is copied at most once until the title writes it (W1): a
//! refresh asks the write watch once for its whole range, not once per gap
//! between the ranges GPU work in flight writes, and a page is copied again
//! only if written since, also where a refresh took only part of it (the
//! part is kept, [`Taken`]; before, such a page was copied at every
//! refresh). Copies between title memory and the copies are plain loads and
//! stores, which a fault on a page the title unmapped ends ([`try_copy`]):
//! the kernel's copy (`ReadProcessMemory`, `WriteProcessMemory`) they
//! replace costs a system call per range and ran at 0.25 to 0.8 GB/s in the
//! world (run-world, world-frame session C). The copy-back reads a run of
//! pieces the GPU changed at once (Marvisak/VIctory#1124) and writes only
//! the bytes the GPU changed and the title left alone (a three-way merge
//! ported from simnJS/AnyPS5), piece by piece, each byte once per
//! submission however many of its ranges overlap there
//! ([`Shadow::copy_back`]). `XTP_SHADOW_ONCE=0` brings
//! back all of it as before: kernel copies, an asking per gap, partial
//! pages copied at every refresh, the copy-back read piece by piece.
//!
//! The list's buffer copies and memory writes that touch a copy run on
//! the GPU at their place in the list ([`copy`], [`write`]), so they see
//! what shaders before them wrote and later commands see their result.
//!
//! With `XTP_CARD_BUFFERS=1`, big title buffers have their copy in the
//! card's own memory ([`cards`], off by default;
//! the idea comes from fxceless's VIctory fork, commit 578d459, its
//! `cards`; reimplemented, none of its code is taken): shaders use it there
//! without going through PCIe, and what they wrote is found on the card.
//! The CPU's side, which refreshes fill and the three-way merge takes as
//! its "before", stays in system memory. Around each submission's lists,
//! in its own `ExecuteCommandLists` ([`cards_around`]): before them, what
//! the CPU side took goes up and each piece of the ranges its shaders may
//! write is hashed on the card; after them, each is hashed again, and the
//! pieces whose hash changed are copied into system memory. Its copy-back
//! merges only those ([`Shadow::card_back`]): no snapshot and no compare on
//! the CPU, which cost about 42 and 80 ms a Series S world frame
//! (docs/analysis/lazy-copyback.md).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.
//!
//! shortcut: copies have no tracked resource state. Like every other use,
//! the GPU copies and writes take them as common (draws rely on implicit
//! promotion), with an unordered access barrier first so that earlier
//! shader writes are done; the debug layer may report the state.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::{HostList, host, transition, uav_barrier};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// What shaders use: system memory, or the card's ([`Card`]).
    pub(crate) buffer: ID3D12Resource,
    /// The copy's bytes as the CPU sees them: `buffer`'s, or with a
    /// [`Card`] its CPU side's.
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: what the copy last took
    /// of it, and after which write-watch asking.
    copied: Mutex<Vec<Taken>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges GPU work of submissions not finished yet writes (its shaders,
    /// or its buffer copies): title memory there is older than the copy,
    /// so no refresh copies it over them.
    busy: Mutex<Vec<Busy>>,
    /// Per submission not finished yet whose shaders write the copy: the
    /// copy's bytes before it ran, over the ranges they write. What the GPU
    /// wrote differs.
    before: Mutex<Snapshots>,
    // A completed mirror can serve as the next preimage until the host copy
    // changes. Track by title page, including the partial pages at its ends.
    versions: Vec<std::sync::atomic::AtomicU64>,
    /// For the first-touch census ([`census`]): whether the title memory
    /// copied back is a view of physical pages, asked once: unknown, a
    /// view, other memory.
    #[cfg(any(test, feature = "diagnostics"))]
    kind: AtomicU8,
    /// For a copy in the card's memory ([`cards`]): its CPU side and what
    /// goes between the two.
    card: Option<Card>,
}

/// What a copy in the card's memory ([`cards`]) has beside
/// `Shadow::buffer`, which is in the card's memory then.
///
/// Its CPU side (`staging`, which `Shadow::data` maps) holds title memory
/// as refreshes took it and copy-backs merged it. The card side equals it
/// but where shaders of a submission not copied back yet wrote, once
/// `dirty` has gone up: a refresh leaves those ranges alone (as it leaves
/// a system memory copy's), and so does every other write into the CPU
/// side. So the CPU side over a range a submission declares is the card
/// side's bytes from before its shaders ran, the "before" of the three-way
/// merge, without a snapshot.
struct Card {
    /// The CPU's side, in system memory.
    staging: ID3D12Resource,
    /// System memory, mapped at `after_data`, where the card copies what a
    /// submission's GPU work wrote, at its offsets in the copy: the pieces
    /// whose hash changed, and the destinations of its GPU copies.
    after: ID3D12Resource,
    after_data: *mut u8,
    /// What the card side lacks of the CPU side, as offsets `[from, to)`:
    /// it goes up before the next submission's lists. Ranges refreshes
    /// copied, and the title's bytes a copy-back's merge kept.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// The work of submissions not copied back yet on this copy.
    parts: Mutex<Vec<Part>>,
    /// The card side's bytes, held against [`card_limits`]' budget.
    bytes: u64,
}

impl Drop for Card {
    fn drop(&mut self) {
        CARD_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

/// A submission's work on one copy in the card's memory ([`cards_around`]).
struct Part {
    submission: u64,
    /// The slot of its plan ([`CardWork`]).
    slot: usize,
    /// Where the card wrote its pieces' records: the slot's memory, which
    /// stays until the part is done with.
    records: *const u8,
    /// The pieces of the ranges its shaders may write, as offsets `[from,
    /// to)` in the copy (each in one page of title memory), with their
    /// record's index.
    spans: Vec<(usize, usize, u32)>,
}

/// What a copy holds of a page of title memory as the title wrote it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Taken {
    /// The write-watch asking after which the copy took it (0: never).
    asked: u64,
    /// The bytes taken then, as offsets in the page: the page's part in
    /// the buffer, or for a page a refresh took in part, that part.
    from: u32,
    to: u32,
}

impl Taken {
    /// Whether the copy holds the page's bytes `[from, to)` (offsets in the
    /// page) as title memory has them, the page last found written at the
    /// asking `write` (0: never).
    fn holds(&self, from: usize, to: usize, write: u64) -> bool {
        self.asked != 0
            && write <= self.asked
            && self.from as usize <= from
            && to <= self.to as usize
    }

    /// The page's bytes `[from, to)` (offsets in it), taken after the
    /// asking `now`.
    fn new(now: u64, from: usize, to: usize) -> Taken {
        Taken {
            asked: now,
            from: from as u32,
            to: to as u32,
        }
    }
}

/// A range of a copy that GPU work in flight writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Busy {
    writer: u64,
    /// The submission ([`submission`]).
    submission: u64,
    /// Title addresses `[from, to)`.
    from: usize,
    to: usize,
    /// Its shaders write it (else a buffer copy does).
    shaders: bool,
}

/// A copy's snapshots: those of submissions not finished yet, and the
/// mirrors that served earlier ones.
#[derive(Default)]
struct Snapshots {
    live: Vec<Before>,
    /// Mirrors kept for good, for the next snapshots of the copy: their
    /// pages are touched already, so a snapshot there takes no page fault.
    /// Before, each snapshot took fresh memory from a pool of 16 buffers
    /// that freed the rest, and merged its pieces into new buffers as later
    /// ranges overlapped them: 76.5 ms and about 74 000 page faults a world
    /// frame (docs/analysis/30fps.md §4.1).
    spare: Vec<Mirror>,
}

/// Memory of a copy's size that holds a snapshot's bytes at their offsets in
/// the copy, kept for good.
struct Mirror {
    bytes: Box<[u8]>,
    versions: Vec<u64>,
}

impl Mirror {
    fn new(len: usize) -> Mirror {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Mirrors, 1);
            add(C::MirrorBytes, len as u64);
        }
        MIRRORS.fetch_add(len as u64, Ordering::Relaxed);
        Mirror {
            bytes: vec![0; len].into_boxed_slice(),
            versions: Vec::new(),
        }
    }
}

impl Drop for Mirror {
    fn drop(&mut self) {
        MIRRORS.fetch_sub(self.bytes.len() as u64, Ordering::Relaxed);
    }
}

/// The bytes of all snapshot mirrors held (for the copystat line).
pub(crate) static MIRRORS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a finished snapshot leaves its mirror to the copy's next ones:
/// unless `XTP_SNAPSHOT_MIRRORS=0`, which gives each snapshot fresh memory
/// as before (without the old pool's merges), to time the two.
/// `XTP_SNAPSHOT_MIRRORS=ab[:<lines>]` (diagnostics) alternates them within
/// one run, switching every `<lines>` copystat lines (3 by default; kept
/// first), each line measuring one way ([`mirror_phase`]).
pub(crate) fn mirrors_kept() -> bool {
    mirror_setting();
    !FRESH.load(Ordering::Relaxed)
}

/// Each snapshot in fresh memory (`XTP_SNAPSHOT_MIRRORS=0`, or this phase
/// of an A/B run).
static FRESH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `XTP_SNAPSHOT_MIRRORS`, read once: the lines of an A/B phase (0: no A/B).
fn mirror_setting() -> usize {
    static SETTING: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| {
        let value = std::env::var("XTP_SNAPSHOT_MIRRORS").unwrap_or_default();
        let value = value.trim();
        let lines = match value.split_once(':') {
            _ if value == "0" => {
                FRESH.store(true, Ordering::Relaxed);
                0
            }
            Some(("ab", n)) => n.trim().parse().unwrap_or(3).max(1),
            None if value == "ab" => 3,
            _ => 0,
        };
        if lines > 0 || value == "0" {
            tracing::info!(value, "snapshot mirrors");
        }
        lines
    })
}

/// For the copystat line, once per line: how the line's snapshots were
/// made (`kept`, `fresh`, the line's place in its A/B phase), then the next
/// line's way.
#[cfg(feature = "diagnostics")]
pub(crate) fn mirror_phase() -> String {
    static LINE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let lines = mirror_setting();
    let way = if mirrors_kept() { "kept" } else { "fresh" };
    if lines == 0 {
        return way.into();
    }
    let line = LINE.fetch_add(1, Ordering::Relaxed);
    let place = line % lines;
    if place + 1 == lines {
        FRESH.store((line / lines).is_multiple_of(2), Ordering::Relaxed);
    }
    format!("{way}, A/B line {} of {lines}", place + 1)
}

/// A submission's snapshot of a copy ([`Shadow::writing`]).
struct Before {
    submission: u64,
    /// The snapshot's bytes, each at its offset in the copy.
    mirror: Mirror,
    /// The offsets taken, as sorted ranges that neither overlap nor touch.
    /// They cover the ranges the submission declared, not the gaps between
    /// them (a refresh may change a gap before a later declaration covers
    /// it). A copy-back makes what it put back equal to the copy
    /// ([`Shadow::put_back`]).
    taken: Vec<(usize, usize)>,
    /// Title pages a copy-back of the submission put back only in part
    /// (its range did not cover them whole), for a later one that covers
    /// them whole to note current ([`Shadow::mark_current`]).
    partial: Vec<usize>,
}

impl Before {
    /// Covers the copy's `[from, to)` (offsets): bytes this submission took
    /// already keep their value (the copy is unchanged there since, nothing
    /// refreshes a busy range), the rest is taken from `host` now, each
    /// byte once. Returns the bytes taken now.
    #[cfg(test)]
    fn cover<'a>(
        &mut self,
        from: usize,
        to: usize,
        host: impl Fn(usize, usize) -> &'a [u8],
    ) -> usize {
        self.cover_gaps(from, to, |mirror, a, b| {
            mirror.bytes[a..b].copy_from_slice(host(a, b));
            b - a
        })
    }

    fn cover_gaps(
        &mut self,
        from: usize,
        to: usize,
        mut take: impl FnMut(&mut Mirror, usize, usize) -> usize,
    ) -> usize {
        // The ranges taken that overlap or touch `[from, to)`.
        let first = self.taken.partition_point(|&(_, end)| end < from);
        let last = first
            + self.taken[first..]
                .iter()
                .take_while(|&&(at, _)| at <= to)
                .count();
        let (mut at, mut took) = (from, 0);
        for &(a, end) in &self.taken[first..last] {
            if at < a {
                took += take(&mut self.mirror, at, a);
            }
            at = at.max(end);
        }
        if at < to {
            took += take(&mut self.mirror, at, to);
        }
        let start = match self.taken.get(first) {
            Some(&(a, _)) if first < last => a.min(from),
            _ => from,
        };
        let end = self.taken[first..last]
            .last()
            .map_or(to, |&(_, end)| end.max(to));
        self.taken.splice(first..last, [(start, end)]);
        took
    }

    /// The snapshot's bytes of `[from, to)` (offsets), if it took them.
    fn bytes(&mut self, from: usize, to: usize) -> Option<&mut [u8]> {
        let i = self.taken.partition_point(|&(_, end)| end < to);
        let &(at, end) = self.taken.get(i)?;
        (at <= from && to <= end).then(|| &mut self.mirror.bytes[from..to])
    }
}

/// Numbers the host's submissions, for what they leave in the copies until
/// they finish (snapshots, busy ranges).
static SUBMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A new submission's number (higher than any before), taken as its
/// preparation starts.
pub(crate) fn submission() -> u64 {
    SUBMISSIONS.fetch_add(1, Ordering::Relaxed) + 1
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written by the submitting thread, and by the thread finishing an earlier
// submission only in ranges that one's GPU work wrote (which the submitting
// thread leaves alone until it has finished: `host::Batch::settle`).
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(
        buffer: ID3D12Resource,
        data: *mut u8,
        gpu: u64,
        start: u64,
        len: u64,
        card: Option<Card>,
    ) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![Taken::default(); pages as usize]),
            watched: AtomicU8::new(0),
            busy: Mutex::default(),
            before: Mutex::default(),
            versions: if crate::readmap::enabled() {
                (0..pages)
                    .map(|_| std::sync::atomic::AtomicU64::new(1))
                    .collect()
            } else {
                Vec::new()
            },
            #[cfg(any(test, feature = "diagnostics"))]
            kind: AtomicU8::new(0),
            card,
        }
    }

    /// Whether the copy is in the card's memory ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Whether the title memory at `at` (in the buffer) is a view of
    /// physical pages, which the guard of view memory keeps read-only
    /// ([`census`]). Asked of the system once per copy: a query in a big
    /// view walks its page tables to the region's end.
    #[cfg(any(test, feature = "diagnostics"))]
    fn census_view(&self, at: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        let kind = match self.kind.load(Ordering::Relaxed) {
            0 => {
                // SAFETY: plain data filled in by the query.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: a valid out pointer of the right size.
                let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
                let kind = if n != 0 && info.Type == MEM_MAPPED {
                    1
                } else {
                    2
                };
                self.kind.store(kind, Ordering::Relaxed);
                kind
            }
            kind => kind,
        };
        kind == 1
    }

    fn busy(&self) -> std::sync::MutexGuard<'_, Vec<Busy>> {
        self.busy.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: in a copy in system memory, what shaders wrote is found by
    /// keeping the range's bytes before the submission (a copy per declared
    /// range and submission); a copy in the card's memory finds it on the
    /// card ([`cards_around`]), with no copy. Big buffers only
    /// ([`card_limits`]): smaller ones and loose memory keep the snapshot.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At the preparation of `submission`, after the range's refresh: its
    /// shaders write this range. Its bytes are kept to tell later what the
    /// GPU wrote: one snapshot per copy and submission (submissions in
    /// flight each need the bytes from before their own GPU work: one
    /// snapshot shared by them gave a later one bytes from before an
    /// earlier one, which lost GPU writes and undid title writes), grown to
    /// cover every range the submission declares (the world declares
    /// overlapping views of its 174 MB heap dozens of times), each byte
    /// taken once, in a mirror of the copy kept for good ([`Snapshots`]: no
    /// fresh memory, so no page faults once the mirrors are warm).
    #[cfg(test)]
    pub(crate) fn writing(&self, submission: u64, address: usize, len: usize) {
        self.writing_by(submission, address, len, 0);
    }

    pub(crate) fn writing_by(&self, submission: u64, address: usize, len: usize, writer: u64) {
        self.busy().push(Busy {
            writer,
            submission,
            from: address,
            to: address + len,
            shaders: true,
        });
        if self.card.is_some() {
            // The card tells what its shaders wrote ([`cards_around`]),
            // from its CPU side, which nothing changes there meanwhile.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::CardDeclared, 1);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: inside the mapping; GPU work in flight does not write it
        // (a step declaring a range first finishes the submissions whose
        // GPU work writes it).
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut snapshots = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Snapshots { live, spare } = &mut *snapshots;
        let i = match live.iter().position(|b| b.submission == submission) {
            Some(i) => i,
            None => {
                let mirror = spare
                    .pop()
                    .unwrap_or_else(|| Mirror::new(self.len as usize));
                live.push(Before {
                    submission,
                    mirror,
                    taken: Vec::new(),
                    partial: Vec::new(),
                });
                live.len() - 1
            }
        };
        let _took = live[i].cover_gaps(from, to, |mirror, a, b| {
            if self.versions.is_empty() {
                mirror.bytes[a..b].copy_from_slice(host(a, b));
                return b - a;
            }
            mirror.versions.resize(self.versions.len(), 0);
            let start = self.start as usize;
            let mut at = a;
            let mut bytes = 0;
            while at < b {
                let page = (start + at) / PAGE;
                let next = b.min((page + 1) * PAGE - start);
                let index = page - start / PAGE;
                if mirror.versions[index] != self.versions[index].load(Ordering::Acquire) {
                    mirror.bytes[at..next].copy_from_slice(host(at, next));
                    bytes += next - at;
                }
                at = next;
            }
            bytes
        });
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Snapshots, 1);
            add(C::SnapshotBytes, _took as u64);
            add(C::SnapshotUs, started.elapsed().as_micros() as u64);
        }
    }

    pub(crate) fn invalidate_mirror(&self, address: usize, len: usize) {
        if self.versions.is_empty() || len == 0 {
            return;
        }
        let base = self.start as usize / PAGE;
        for page in address / PAGE..(address + len).div_ceil(PAGE) {
            self.versions[page - base].fetch_add(1, Ordering::AcqRel);
        }
    }

    /// At the preparation of `submission`: a buffer copy it records on the
    /// GPU writes `[address, address + len)` of the copy, which refreshes
    /// leave alone until [`Shadow::write_back`] has put it in title memory.
    pub(crate) fn copying(&self, submission: u64, address: usize, len: usize) {
        self.invalidate_mirror(address, len);
        self.busy().push(Busy {
            writer: 0,
            submission,
            from: address,
            to: address + len,
            shaders: false,
        });
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). The title keeps
    /// writing while this runs (on the completion thread), so only the runs
    /// of bytes the GPU changed are written, not the pieces around them.
    ///
    /// Each byte goes back once per submission, whatever ranges over it
    /// the submission declared and in whatever order they come back: what
    /// a copy-back put back is in the snapshot after it ([`Shadow::put_back`]),
    /// so another range over it finds nothing the GPU changed there. (Before,
    /// a second range over bytes the first had taken from the title took
    /// them for the GPU's, and wrote them over what the title wrote since.
    /// The host merges the ranges of a submission that overlap or touch.)
    /// `[address, address + len)` covers whole ranges the submission
    /// declared: they are done with once it is back. Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, submission: u64, address: usize, len: usize) -> usize {
        if let Some(card) = &self.card {
            return self.card_back(card, submission, address, len);
        }
        let mut snapshots = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.invalidate_mirror(start + from, to - from);
        if let Some(snapshot) = snapshots
            .live
            .iter_mut()
            .find(|b| b.submission == submission)
            && from < to
            && let Some(before) = snapshot.bytes(from, to)
        {
            // By piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote, a run
            // of such pieces with one copy (Marvisak/VIctory#1124: through
            // the kernel a call cost tens to hundreds of microseconds however
            // small, and its world put back ~650 pieces a frame).
            let end_of = |at: usize| (at + PIECE - (start + at) % PIECE).min(to);
            // SAFETY: inside the mapping; the submitting thread leaves the
            // range alone until this submission has finished.
            let host =
                |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
            let differs =
                |before: &[u8], a: usize, b: usize| host(a, b) != &before[a - from..b - from];
            #[cfg(any(test, feature = "diagnostics"))]
            let census = census::before_back(submission, start + from, start + to, || {
                self.census_view(start + from)
            });
            let mut synced = Vec::new();
            #[cfg(feature = "diagnostics")]
            let scan_start = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let mut merging = std::time::Duration::ZERO;
            let mut at = from;
            while at < to {
                let mut end = end_of(at);
                if differs(before, at, end) {
                    // The pieces after it the GPU changed too, up to a run
                    // (one piece at a time as before W1).
                    while once() && end < to && end - at < RUN && differs(before, end, end_of(end))
                    {
                        end = end_of(end);
                    }
                    #[cfg(feature = "diagnostics")]
                    let merge_start = std::time::Instant::now();
                    let old = &mut before[at - from..end - from];
                    // SAFETY: inside the mapping; the submitting thread
                    // leaves the range alone until this submission has
                    // finished.
                    let host =
                        unsafe { std::slice::from_raw_parts_mut(self.data.add(at), end - at) };
                    if crate::readmap::enabled() {
                        written +=
                            self.put_mapped_range(submission, start + at, host, old, &mut synced);
                    } else {
                        written += self.put_back(at, end, host, old, &mut synced).0;
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        merging += merge_start.elapsed();
                    }
                }
                at = end;
            }
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(
                    C::BackScanNs,
                    scan_start.elapsed().saturating_sub(merging).as_nanos() as u64,
                );
                add(C::BackCompared, (to - from) as u64);
                // By kind of copy, to tell which would gain from the card:
                // windows of loose memory, then title buffers by size.
                let kind = if self.start.is_multiple_of(WINDOW) && self.len.is_multiple_of(WINDOW) {
                    C::BackComparedLoose
                } else if self.len < 1 << 20 {
                    C::BackComparedSmall
                } else if self.len < card_limits().0 {
                    C::BackComparedMid
                } else {
                    C::BackComparedBig
                };
                add(kind, (to - from) as u64);
            }
            // Pages now equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh). The
            // range's whole pages only, with those an earlier range of the
            // submission put back in part; the other part pages wait for a
            // range that covers them whole.
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let whole = |p: &usize| (first..last).contains(p);
            let mut current: Vec<usize> = synced
                .iter()
                .chain(&snapshot.partial)
                .copied()
                .filter(whole)
                .collect();
            current.sort_unstable();
            current.dedup();
            snapshot.partial.retain(|p| !whole(p));
            snapshot
                .partial
                .extend(synced.into_iter().filter(|p| !whole(p)));
            self.mark_current(&current);
            // Every compared byte now agrees with the host, including holes
            // which were unreadable in title memory (put_mapped_piece). Only
            // a completely covered page may supply a later full preimage.
            if !self.versions.is_empty() {
                for page in (start + from) / PAGE..(start + to).div_ceil(PAGE) {
                    let first = (page * PAGE).max(start);
                    let last = ((page + 1) * PAGE).min(start + self.len as usize);
                    if start + from <= first && last <= start + to {
                        let i = page - start / PAGE;
                        snapshot.mirror.versions[i] = self.versions[i].load(Ordering::Acquire);
                    }
                }
            }
            #[cfg(any(test, feature = "diagnostics"))]
            census::after_back(census, submission, start + from, start + to);
        }
        // Every step of the submission that declared a range in it is done
        // with it, and the submission's snapshot with its last range.
        let end = address + len;
        let mut busy = self.busy();
        busy.retain(|b| {
            !(b.shaders && b.submission == submission && address <= b.from && b.to <= end)
        });
        if !busy.iter().any(|b| b.shaders && b.submission == submission)
            && let Some(i) = snapshots
                .live
                .iter()
                .position(|b| b.submission == submission)
        {
            // Its mirror serves the copy's next snapshots (unless each is to
            // have fresh memory).
            let done = snapshots.live.swap_remove(i);
            if mirrors_kept() {
                snapshots.spare.push(done.mirror);
            } else {
                snapshots.spare.clear();
            }
        }
        written
    }

    /// Publishes a completed piece while its source still belongs to this
    /// submission. GPU copies replace every byte, shader writes merge with
    /// the preimage. Other views and other shadows observe the journal write.
    fn publish_piece(
        &self,
        address: usize,
        writer: u64,
        gpu: &mut [u8],
        before: Option<&[u8]>,
    ) -> Option<usize> {
        let start = self.start as usize;
        let from = address - start;
        let key = crate::readmap::Key {
            size: self.len,
            alignment: self.start.trailing_zeros().min(16),
            writer,
            offset: from as u64,
        };
        let (written, now) = crate::watch::publish(address, key, gpu, before)?;
        if address.is_multiple_of(PAGE) && gpu.len() == PAGE {
            lock(&self.copied)[address / PAGE - start / PAGE] = Taken::new(now, 0, PAGE);
        }
        #[cfg(feature = "diagnostics")]
        if written == 0 {
            crate::copystat::add(crate::copystat::C::DeferredBytes, gpu.len() as u64);
        }
        Some(written)
    }

    /// Publish each physical page independently. Unmapped pieces keep the
    /// GPU bytes in the shadow and the snapshot for its next submission.
    fn put_mapped_range(
        &self,
        submission: u64,
        address: usize,
        gpu: &mut [u8],
        before: &mut [u8],
        synced: &mut Vec<usize>,
    ) -> usize {
        let mut written = 0;
        let mut at = 0;
        while at < gpu.len() {
            let next = (at + PAGE - (address + at) % PAGE).min(gpu.len());
            written += self.put_mapped_piece(
                submission,
                address + at,
                &mut gpu[at..next],
                &mut before[at..next],
                synced,
            );
            at = next;
        }
        written
    }

    fn put_mapped_piece(
        &self,
        submission: u64,
        address: usize,
        gpu: &mut [u8],
        before: &mut [u8],
        synced: &mut Vec<usize>,
    ) -> usize {
        let start = self.start as usize;
        let from = address - start;
        let writer = self
            .busy()
            .iter()
            .find(|b| {
                b.shaders
                    && b.submission == submission
                    && b.from < address + gpu.len()
                    && address < b.to
            })
            .map_or(0, |b| b.writer);
        let written = self
            .publish_piece(address, writer, gpu, Some(before))
            .unwrap_or_else(|| self.put_back(from, from + gpu.len(), gpu, before, synced).0);
        before.copy_from_slice(gpu);
        written
    }

    /// A buffer copy `list` records on the GPU writes `[address, address +
    /// len)` of the copy: once the list has run, the range goes back into
    /// title memory, all of it ([`Shadow::write_back`]).
    pub(crate) fn gpu_copies(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_copy(self, address as usize, len as usize);
    }

    /// After the submission ran: writes `[address, address + len)` of the
    /// copy into title memory (unmapped pages left alone), as the GPU
    /// copies that wrote it would have. Unlike [`Shadow::copy_back`], no
    /// snapshot tells the GPU's bytes apart: the range is exactly what a
    /// copy wrote. `[address, address + len)` covers ranges the
    /// submission's copies declared ([`Shadow::copying`]): refreshes may
    /// take them again. Returns the bytes written.
    pub(crate) fn write_back(&self, submission: u64, address: usize, len: usize) -> usize {
        let written = self.write_copied(submission, address, len);
        let end = address + len;
        self.busy()
            .retain(|b| b.shaders || b.submission != submission || b.from < address || b.to > end);
        if let Some(card) = &self.card {
            self.card_settled(card, submission);
        }
        written
    }

    fn write_copied(&self, submission: u64, address: usize, len: usize) -> usize {
        let (from, to) = self.inside(address, len);
        if from >= to {
            return 0;
        }
        let start = self.start as usize;
        if let Some(card) = &self.card {
            // What the GPU copies wrote is on the card, which copied it
            // into `after` once the submission's lists had run
            // ([`cards_around`]): the CPU side takes it, the card side
            // holds it already.
            // SAFETY: inside both mappings; the submission has run, and no
            // later GPU work writes this range before it is written back.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    card.after_data.add(from),
                    self.data.add(from),
                    to - from,
                );
            }
        }
        // The submission, for the census only.
        #[cfg(not(any(test, feature = "diagnostics")))]
        let _ = submission;
        #[cfg(any(test, feature = "diagnostics"))]
        let census = census::before_back(submission, start + from, start + to, || {
            self.census_view(start + from)
        });
        if crate::readmap::enabled() {
            let mut p = from;
            let mut written = 0;
            let mut current = Vec::new();
            while p < to {
                let addr = start + p;
                let n = (PAGE - addr % PAGE).min(to - p);
                // SAFETY: this destination's fence completed. A later
                // overlapping GPU writer waits for its CPU consumption.
                let gpu = unsafe { std::slice::from_raw_parts_mut(self.data.add(p), n) };
                written += self
                    .publish_piece(addr, u64::MAX, gpu, None)
                    .unwrap_or_else(|| {
                        crate::watch::writing(addr, n);
                        write_title(addr, gpu.as_ptr(), n);
                        if n == PAGE {
                            current.push(addr / PAGE);
                        }
                        n
                    });
                p += n;
            }
            self.mark_current(&current);
            return written;
        }
        crate::watch::writing(start + from, to - from);
        // SAFETY: inside the mapping; the submitting thread leaves the
        // range alone until this submission has finished.
        let host = unsafe { self.data.add(from) };
        write_title(start + from, host, to - from);
        #[cfg(any(test, feature = "diagnostics"))]
        census::wrote(start + from, to - from);
        // Whole pages written are now as the copy has them, unless the
        // title wrote them meanwhile: our own write is not taken for the
        // title's.
        let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
        self.mark_current(&(first..last).collect::<Vec<_>>());
        #[cfg(any(test, feature = "diagnostics"))]
        census::after_back(census, submission, start + from, start + to);
        to - from
    }

    /// [`Shadow::copy_back`] of `[from, to)` (offsets in the copy), a run of
    /// pieces the GPU changed (`host`: the copy's bytes of them after the
    /// submission; `old`: before it): title memory read with one copy, where
    /// it can all be read (else the pieces that can: [`read_pieces`]), then
    /// piece by piece the GPU's bytes written into title memory where it
    /// lacks them, the title's taken into `host` elsewhere. Pieces that
    /// cannot be read are left alone; those read go into `synced` (page
    /// numbers), and `old` takes what `host` holds of them then: another
    /// range of the submission over them compares with that, and puts back
    /// only what this left. Returns the bytes written, and the parts read
    /// (offsets in the run).
    ///
    /// Unlike Marvisak/VIctory#1124, which writes back whole runs of pieces
    /// with one kernel call, only the bytes the GPU changed are written
    /// (with plain stores, which cost no call): the title writes these
    /// pages while this runs, and a piece written whole puts back the
    /// title's bytes as read before over its writes since.
    ///
    /// shortcut: what stays open (the private watch notes' review of
    /// Marvisak/VIctory#1124). A title write into bytes the GPU changed
    /// stays when it was made before this read title memory for the run
    /// (the three-way merge, [`Shadow::write_gpu_runs`]), but not when made
    /// after that read (up to 16 pieces) and before the piece's write: the
    /// merge compares with the bytes as read, and the GPU's go over it. Nor
    /// when it put back the very value the snapshot holds: the byte looks
    /// untouched. A title write into bytes the GPU left alone stays; the
    /// copy then holds the title's bytes as read, and the page is not
    /// current ([`Shadow::mark_current`] compares).
    fn put_back(
        &self,
        from: usize,
        to: usize,
        host: &mut [u8],
        old: &mut [u8],
        synced: &mut Vec<usize>,
    ) -> (usize, Vec<(usize, usize)>) {
        thread_local! {
            /// Title memory of a run, kept (already touched) for the next.
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let at = self.start as usize + from;
        TITLE.with_borrow_mut(|title| {
            title.resize(to - from, 0);
            let mut written = 0;
            let parts = read_pieces(at, title);
            for &(a, b) in &parts {
                let mut piece = a;
                while piece < b {
                    let end = (piece + PIECE - (at + piece) % PIECE).min(b);
                    #[cfg(test)]
                    tests::after_reading(at + piece);
                    let (host, old, title) = (
                        &mut host[piece..end],
                        &mut old[piece..end],
                        &title[piece..end],
                    );
                    written += self.write_gpu_runs(at + piece, host, old, title);
                    // Put back: the copy's bytes are the snapshot's now.
                    old.copy_from_slice(host);
                    synced.push((at + piece) / PAGE);
                    piece = end;
                }
            }
            (written, parts)
        })
    }

    /// Writes into title memory at `at` the runs of bytes the GPU changed
    /// in a piece of the copy (`host`; `old`, its bytes before the
    /// submission; `title`, title memory as just read) where the title
    /// left them alone; the copy takes the title's bytes everywhere else.
    /// Returns the bytes written.
    ///
    /// A three-way merge (ported from simnJS/AnyPS5 (65a1b01db4), its
    /// `CompareWriteBack`): a byte both changed, the GPU in the copy and the
    /// title in title memory since the snapshot, keeps the title's. Before,
    /// the GPU's went over it; title memory is the title's records first,
    /// and the copy-back comes late (once the submission has finished).
    fn write_gpu_runs(&self, at: usize, host: &mut [u8], old: &[u8], title: &[u8]) -> usize {
        let (mut written, mut opened) = (0, false);
        let mut i = 0;
        while i < host.len() {
            let run = first_change(host, old, i);
            host[i..run].copy_from_slice(&title[i..run]);
            if run == host.len() {
                break;
            }
            i = first_same(host, old, run);
            // In the run the GPU changed: the parts the title left alone go
            // back, those it wrote stay its own (and the copy takes them).
            let (title_run, old_run) = (&title[..i], &old[..i]);
            let mut j = run;
            while j < i {
                let theirs = first_change(title_run, old_run, j);
                if theirs > j {
                    if !opened {
                        // The piece's guard, once.
                        crate::watch::writing(at, host.len());
                        opened = true;
                    }
                    write_title(at + j, host[j..].as_ptr(), theirs - j);
                    written += theirs - j;
                    #[cfg(any(test, feature = "diagnostics"))]
                    census::wrote(at + j, theirs - j);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::BackRuns, 1);
                }
                j = first_same(title_run, old_run, theirs);
                host[theirs..j].copy_from_slice(&title[theirs..j]);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackKept, (j - theirs) as u64);
            }
        }
        written
    }

    /// After our own write into title memory: notes `pages` (title page
    /// numbers, all in the copy) current where title memory still equals
    /// the copy. The write watch is asked first, so that it does not report
    /// our write later; a title write before that asking is in title memory
    /// when it is compared (the page is not current then), one after it
    /// shows at a later asking.
    fn mark_current(&self, pages: &[usize]) {
        let (Some(&first), Some(&last)) = (pages.iter().min(), pages.iter().max()) else {
            return;
        };
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return;
        }
        let Some((now, _)) = crate::watch::writes(first * PAGE, (last + 1) * PAGE, true) else {
            return;
        };
        let start = self.start as usize;
        let mut title = vec![0u8; PAGE];
        let current: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&page| {
                let from = page * PAGE - start;
                // SAFETY: a whole page inside the mapping.
                let host = unsafe { std::slice::from_raw_parts(self.data.add(from), PAGE) };
                read_whole(page * PAGE, &mut title) && *host == *title
            })
            .collect();
        let base = start / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        for page in current {
            copied[page - base] = Taken::new(now, 0, PAGE);
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched), but where GPU work in flight writes the copy.
    /// Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .busy()
            .iter()
            .map(|b| (b.from, b.to))
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        busy.sort_unstable();
        let copied = if once() {
            self.update_once(from, to, &busy)
        } else {
            self.update_gaps(from, to, busy)
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Updates, 1);
            add(C::Updated, (to - from) as u64);
            add(C::Refreshed, copied as u64);
            add(C::SkippedBytes, (to - from).saturating_sub(copied) as u64);
            if copied == 0 {
                add(C::Skipped, 1);
            }
        }
        copied
    }

    /// [`Shadow::update`] of `[from, to)` around `busy` (sorted): one asking
    /// of the write watch for the whole range (the journal keeps what it
    /// finds in the busy ranges for later refreshes), then the pages
    /// written since their last copy in each gap.
    fn update_once(&self, from: usize, to: usize, busy: &[(usize, usize)]) -> usize {
        let mut gaps = Vec::with_capacity(busy.len() + 1);
        let mut at = from;
        for &(a, e) in busy {
            if a > at {
                gaps.push((at, a));
            }
            at = at.max(e);
        }
        if at < to {
            gaps.push((at, to));
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED && !whole_ranges() {
            #[cfg(test)]
            tests::ASKED.set(tests::ASKED.get() + 1);
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied: usize = gaps
                    .iter()
                    .map(|&(a, b)| self.copy_stale(a, b, now, &written))
                    .sum();
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        gaps.iter().map(|&(a, b)| self.copy(a, b)).sum()
    }

    /// [`Shadow::update`] as before W1 (`XTP_SHADOW_ONCE=0`): an asking per
    /// gap around `busy` (sorted).
    fn update_gaps(&self, from: usize, to: usize, busy: Vec<(usize, usize)>) -> usize {
        if !busy.is_empty() {
            // Around ranges GPU work is writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// Whether the title may have written a page of `[from, to)` since the
    /// copy last took it, or the copy holds only another part of the page
    /// (always, for memory that is neither write-watched nor guarded).
    /// Asks the write journal, which keeps the answer for the
    /// refresh that follows. (Marvisak/VIctory#595, for refreshes that
    /// overlap what an earlier submission's GPU work writes.)
    pub(crate) fn stale(&self, from: usize, to: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let Some((_, written)) = crate::watch::writes(from, to, true) else {
            return true;
        };
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        (from / PAGE..=(to - 1) / PAGE).any(|page| {
            let write = written
                .binary_search_by_key(&page, |&(p, _)| p)
                .map_or(0, |i| written[i].1);
            let at = page * PAGE;
            !copied[page - base].holds(from.max(at) - at, to.min(at + PAGE) - at, write)
        })
    }

    /// [`Shadow::update_gaps`] of `[from, to)` (inside the buffer) where no
    /// GPU work writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED && !whole_ranges() {
            #[cfg(test)]
            tests::ASKED.set(tests::ASKED.get() + 1);
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to)
    }

    /// Copies the parts in `[from, to)` of the pages written after the copy
    /// took them (`written`: last writes by page, in page order, from
    /// `from`'s page or before; `now`: this asking), whole pages within the
    /// buffer. A page only partly in the range is copied in part, and that
    /// part is kept ([`Taken`]): a refresh of it (or of less) finds it
    /// current until the title writes the page. (Before W1, such a page
    /// stayed stale, and was copied at every refresh: the ends of every
    /// range that does not start and end on a page, each submission.)
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let first = written.partition_point(|&(p, _)| p < from / PAGE);
        let mut writes = written[first..].iter().peekable();
        let (mut bytes, mut run) = (0, None::<usize>);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = page * PAGE;
            // The page's part in the range, and in the buffer (offsets in
            // the page).
            let (a, b) = (from.max(at) - at, to.min(at + PAGE) - at);
            let whole = (start.max(at) - at, end.min(at + PAGE) - at);
            let taken = &mut copied[page - base];
            if taken.holds(a, b, write) {
                if let Some(first) = run.take() {
                    bytes += self.copy((first * PAGE).max(from), at.min(to));
                }
                continue;
            }
            if (a, b) == whole {
                *taken = Taken::new(now, a, b);
            } else if once() {
                // What it holds still, unwritten since, if this part
                // overlaps or touches it: both.
                let near = taken.asked != 0
                    && write <= taken.asked
                    && a <= taken.to as usize
                    && taken.from as usize <= b;
                *taken = if near {
                    Taken::new(now, a.min(taken.from as usize), b.max(taken.to as usize))
                } else {
                    Taken::new(now, a, b)
                };
            }
            run.get_or_insert(page);
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy
    /// (zeros for what cannot be read). Returns the bytes copied.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        self.invalidate_mirror(from, to - from);
        if let Some(card) = &self.card {
            // Up to the card side before the next submission's lists.
            lock(&card.dirty).push((from - start, to - start));
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let host = unsafe { self.data.add(from - start) };
        if !plain() {
            crate::host::copy_mapped(host, from as *const u8, to - from, false);
            return to - from;
        }
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() == 2 {
            return to - from;
        }
        #[cfg(feature = "diagnostics")]
        let moved = std::time::Instant::now();
        let zeros = read_title(host, from, to - from);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::PlainNs, moved.elapsed().as_nanos() as u64);
            add(C::Plain, (to - from - zeros) as u64);
            add(C::Zeroed, zeros as u64);
        }
        let _ = zeros;
        to - from
    }
}

/// The first offset from `at` where `a` and `b` differ (their length if
/// none), 8 bytes at a time: the copy-back's byte loop cost about 1 ns a
/// byte, 8 ms for each 8 MiB range the GPU wrote here and there
/// (`world_frame_copies_cost`). A byte at a time as before W1
/// (`XTP_SHADOW_ONCE=0`).
fn first_change(a: &[u8], b: &[u8], mut at: usize) -> usize {
    if once() {
        while at + 8 <= a.len() {
            let x = word(a, at) ^ word(b, at);
            if x != 0 {
                return at + (x.trailing_zeros() / 8) as usize;
            }
            at += 8;
        }
    }
    while at < a.len() && a[at] == b[at] {
        at += 1;
    }
    at
}

/// The first offset from `at` where `a` and `b` are the same (their length
/// if none), 8 bytes at a time as [`first_change`].
fn first_same(a: &[u8], b: &[u8], mut at: usize) -> usize {
    const ONES: u64 = u64::from_le_bytes([1; 8]);
    const HIGHS: u64 = u64::from_le_bytes([0x80; 8]);
    if once() {
        while at + 8 <= a.len() {
            let x = word(a, at) ^ word(b, at);
            // The high bit of each zero byte of `x` (bytes above the first
            // may be marked wrongly, not the first).
            let zero = x.wrapping_sub(ONES) & !x & HIGHS;
            if zero != 0 {
                return at + (zero.trailing_zeros() / 8) as usize;
            }
            at += 8;
        }
    }
    while at < a.len() && a[at] != b[at] {
        at += 1;
    }
    at
}

/// The 8 bytes of `bytes` at `at`, the first lowest.
fn word(bytes: &[u8], at: usize) -> u64 {
    let mut w = [0; 8];
    w.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(w)
}

/// The copy-back's unit: title memory is compared, read and written back
/// by page.
const PIECE: usize = PAGE;
/// At most this many bytes of pieces the GPU changed are read at once.
const RUN: usize = 16 * PIECE;

/// Whether refreshes and copy-backs work as W1 has them (see the module):
/// unless `XTP_SHADOW_ONCE=0`.
fn once() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        std::env::var("XTP_SHADOW_ONCE")
            .ok()
            .is_none_or(|v| v.trim() != "0")
    })
}

/// Whether title memory is copied with plain loads and stores: with
/// [`once`], unless `XTP_COPY_MODE=kernel` (diagnostics) asks for the
/// kernel's copy to time it.
fn plain() -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::copystat::mode() == 3 {
        return false;
    }
    once()
}

// One `rep movsb` that a fault ends (see [`try_copy`]). Windows x64: the
// destination, source and length come in rcx, rdx and r8; rdi and rsi are
// the callee's to keep, hence the unwind data.
core::arch::global_asm!(
    ".text",
    ".p2align 4",
    ".globl xtp_gpu_try_copy",
    "xtp_gpu_try_copy:",
    ".seh_proc xtp_gpu_try_copy",
    "pushq %rdi",
    ".seh_pushreg %rdi",
    "pushq %rsi",
    ".seh_pushreg %rsi",
    ".seh_endprologue",
    "movq %rcx, %rdi",
    "movq %rdx, %rsi",
    "movq %r8, %rcx",
    ".globl xtp_gpu_try_copy_moving",
    "xtp_gpu_try_copy_moving:",
    "rep movsb",
    ".globl xtp_gpu_try_copy_stopped",
    "xtp_gpu_try_copy_stopped:",
    "movq %rcx, %rax",
    "popq %rsi",
    "popq %rdi",
    "retq",
    ".seh_endproc",
    options(att_syntax)
);

unsafe extern "C" {
    /// Copies `len` bytes from `src` to `dst`; returns how many were left
    /// when a fault stopped it (0: all of them copied).
    fn xtp_gpu_try_copy(dst: *mut u8, src: *const u8, len: usize) -> usize;
    /// Its `rep movsb`, and the instruction after it.
    safe static xtp_gpu_try_copy_moving: u8;
    safe static xtp_gpu_try_copy_stopped: u8;
}

/// The kernel's page, the unit of protections and of what can be mapped.
const KERNEL_PAGE: usize = 4096;

thread_local! {
    /// The address the last fault at the copy's instruction on this thread
    /// could not access, as the system reported it (0: not reported).
    static FAULT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The stores into guarded blocks [`stopped`] let go on in this copy.
    static OPENED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

pub(super) fn copy_instruction(ip: usize) -> bool {
    ip == &raw const xtp_gpu_try_copy_moving as usize
}

/// The copy's vectored handler: a fault at its `rep movsb` ends the copy
/// (the instruction after it returns what was left), the address that
/// faulted noted in [`FAULT`]. It looks at nothing else.
///
/// A store into a block the guard of view memory made read-only goes to
/// the guard first ([`crate::watch::writing`] opens it, as the guard's own
/// handler does), and on once the page is writable: both handlers come
/// first ahead of the crash reporter's, so the one registered last is
/// called first, and that depends on what ran first. Ended there, a copy
/// into title memory fell back to the kernel's ([`write_title`]), which
/// fails on a block a refresh guarded again meanwhile: the GPU's bytes
/// were lost without a word.
unsafe extern "system" fn stopped(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    };
    /// At most this many guarded blocks opened per copy (a refresh may
    /// guard a block again before the store is made again).
    const OPENINGS: u32 = 16;
    const WRITE: usize = 1;
    // SAFETY: the system passes the exception being dispatched.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    // The census's guards, from any code: the system took the guard off,
    // and the access goes on (the copy's too). A write that faults while
    // the census changes a block's protection goes on once it is done.
    #[cfg(any(test, feature = "diagnostics"))]
    if record.NumberParameters >= 2 {
        let (kind, at) = (
            record.ExceptionInformation[0],
            record.ExceptionInformation[1],
        );
        let going_on = match record.ExceptionCode {
            STATUS_GUARD_PAGE_VIOLATION => census::fault(kind == WRITE, at, context.Rip as usize),
            EXCEPTION_ACCESS_VIOLATION if kind == WRITE => census::settle(at, context.Rip as usize),
            _ => false,
        };
        if going_on {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
        && record.NumberParameters >= 2
        && record.ExceptionInformation[0] == 0
        && crate::watch::read_fault(
            record.ExceptionInformation[1],
            !copy_instruction(context.Rip as usize),
        )
    {
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    let faults = [
        EXCEPTION_ACCESS_VIOLATION,
        EXCEPTION_IN_PAGE_ERROR,
        STATUS_GUARD_PAGE_VIOLATION,
    ];
    if context.Rip != &raw const xtp_gpu_try_copy_moving as u64
        || !faults.contains(&record.ExceptionCode)
    {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Access violations, in-page errors and guard pages give the address.
    let at = if record.NumberParameters >= 2 {
        record.ExceptionInformation[1]
    } else {
        0
    };
    if record.ExceptionCode == EXCEPTION_ACCESS_VIOLATION
        && record.NumberParameters >= 2
        && record.ExceptionInformation[0] == WRITE
        && OPENED.get() < OPENINGS
    {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        // The faulting thread's last error stays its own (it may be the
        // title's): opening the other views of a page fails where one is
        // unmapped meanwhile, as the guard's handler knows.
        // SAFETY: plain reads and writes of this thread's last error.
        let error = unsafe { GetLastError() };
        let opened = crate::watch::write_fault(at, false);
        // SAFETY: as above.
        unsafe { SetLastError(error) };
        if opened {
            OPENED.set(OPENED.get() + 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    FAULT.set(at);
    #[cfg(test)]
    {
        // A string instruction may stop short of the faulting byte (fast
        // strings move blocks): as if this one had stopped `lag` bytes
        // before it.
        let lag = tests::LAG.replace(0) as u64;
        context.Rcx += lag;
        context.Rsi -= lag;
        context.Rdi -= lag;
    }
    context.Rip = &raw const xtp_gpu_try_copy_stopped as u64;
    EXCEPTION_CONTINUE_EXECUTION
}

/// Registers [`stopped`], once, first, ahead of the crash reporter's:
/// before the first plain copy, and before the census guards a page.
fn install() {
    use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
    static HANDLER: std::sync::Once = std::sync::Once::new();
    // SAFETY: a handler that only moves past the copy's instruction, opens
    // a guarded block as the guard's own handler does, or lets an access to
    // a page the census guarded go on.
    HANDLER.call_once(|| unsafe {
        #[cfg(any(test, feature = "diagnostics"))]
        census::find_our_image();
        AddVectoredExceptionHandler(1, Some(stopped));
    });
}

/// Whether `at` is committed memory that can be written now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && can_write(info.Protect)
}

/// Whether committed memory of protection `protect` can be written.
fn can_write(protect: u32) -> bool {
    use windows_sys::Win32::System::Memory::{
        PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READWRITE, PAGE_WRITECOPY,
    };
    let writes = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    protect & PAGE_GUARD == 0 && protect & writes != 0
}

/// Copies `len` bytes from `src` to `dst` with plain loads and stores;
/// returns how many were left from the first 4 KiB page of either that
/// could not be accessed (0: all of them copied), every byte before it
/// copied. The kernel's copy reports such a page instead of faulting, but
/// runs at 1 GB/s or less, and title memory can be unmapped or decommitted
/// under a copy at any time (by the title's other threads): here a
/// vectored handler turns a fault at the copy's `rep movsb` into its end
/// ([`stopped`]).
///
/// A string instruction's registers say how far it got, which may be short
/// of the faulting byte: the processor may move blocks at a time (fast
/// strings), and on a fault keeps only that the iterations before the
/// registers are done. So the copy is made again from there up to the 4
/// KiB page of the address the system reported, and only that page is out
/// of reach. (Before, the bytes between were taken for unreadable: zeros
/// in the copy, a readable piece left out of the copy-back.) The stores
/// past where it stopped (if any) are only what the copy would have
/// stored.
///
/// # Safety
///
/// `dst` and `src` are `len` bytes that do not overlap, of which every page
/// that is accessible may be read (`src`) and written (`dst`): title memory,
/// or memory of ours no reference points to meanwhile.
unsafe fn try_copy(dst: *mut u8, src: *const u8, len: usize) -> usize {
    install();
    // Every byte before `at` is copied; `[at, end)` is left to copy.
    let (mut at, mut end) = (0, len);
    while at < end {
        FAULT.set(0);
        OPENED.set(0);
        // SAFETY: as the caller vouches; a fault ends the copy.
        let left = unsafe { xtp_gpu_try_copy(dst.add(at), src.add(at), end - at) };
        if left == 0 {
            at = end;
            break;
        }
        let stop = end - left;
        // The faulting page, as an offset in the copy, if the system said
        // where in the part not copied yet; else where the copy stopped.
        let fault = FAULT.get();
        let page = [src as usize, dst as usize]
            .into_iter()
            .find(|&base| (base + stop..base + end).contains(&fault))
            .map_or(stop, |base| {
                (fault & !(KERNEL_PAGE - 1)).max(base + stop) - base
            });
        if page == stop {
            return len - stop;
        }
        (at, end) = (stop, page);
    }
    len - at
}

/// Remplit la mémoire d’upload par lectures directes (TG0000, VIctory #1010).
/// Le gestionnaire de XTP reprend après une page illisible et conserve les
/// octets lisibles de part et d’autre, même si le début n’est pas aligné.
///
/// # Safety
///
/// `dst` désigne `len` octets accessibles en écriture, réservés à la liste
/// et disjoints de la source. Les pages lisibles de `src` appartiennent au titre.
pub(super) unsafe fn fill_upload(dst: *mut u8, src: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    if crate::copystat::mode() != 0 {
        // Les modes de mesure gardent leur copie brute, absente ou noyau.
        crate::host::copy_mapped(dst, src as *const u8, len, false);
        return;
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let _zeros = read_title(dst, src, len);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::MoveUs, moved.elapsed().as_micros() as u64);
        if _zeros != 0 {
            add(C::Fallbacks, 1);
            add(C::Unmapped, _zeros as u64);
        }
    }
}

/// Copies title memory `[src, src + len)` into memory of ours at `dst`
/// with plain loads; what cannot be read is left as zeros (as the kernel's
/// copy did). Returns the bytes zeroed.
fn read_title(dst: *mut u8, src: usize, len: usize) -> usize {
    let (mut at, mut zeros) = (0, 0);
    while at < len {
        // SAFETY: our memory, `len` bytes; title memory a fault leaves alone.
        let left = unsafe { try_copy(dst.add(at), (src + at) as *const u8, len - at) };
        at = len - left;
        if left == 0 {
            break;
        }
        let n = out_of_reach(src + at, false).min(left);
        // SAFETY: inside our memory.
        unsafe { dst.add(at).write_bytes(0, n) };
        zeros += n;
        at += n;
    }
    zeros
}

/// Reads title memory at `src` into `dst`: with one copy where it can all
/// be read, else the pieces that can be (Marvisak/VIctory#1124: a range
/// shorter than a piece can still cross a piece boundary, and a read that
/// fails keeps the readable pieces). Returns the parts read, as offsets in
/// `dst`: whole pieces of title memory, but at `dst`'s ends.
fn read_pieces(src: usize, dst: &mut [u8]) -> Vec<(usize, usize)> {
    let len = dst.len();
    // The end of the piece holding `at` (an offset in `dst`).
    let piece_end = |at: usize| (at + PIECE - (src + at) % PIECE).min(len);
    let mut parts: Vec<(usize, usize)> = Vec::new();
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackRead, len as u64);
    if !plain() {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackReads, 1);
        if crate::host::read_into(src as *const u8, dst) {
            parts.push((0, len));
            return parts;
        }
        // Not all of it is mapped: piece by piece.
        let mut at = 0;
        while at < len {
            let end = piece_end(at);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackReads, 1);
            if crate::host::read_into((src + at) as *const u8, &mut dst[at..end]) {
                match parts.last_mut() {
                    Some(last) if last.1 == at => last.1 = end,
                    _ => parts.push((at, end)),
                }
            }
            at = end;
        }
        return parts;
    }
    let mut at = 0;
    while at < len {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackReads, 1);
        #[cfg(test)]
        tests::READS.set(tests::READS.get() + 1);
        // SAFETY: our buffer, from `at` to its end; title memory a fault
        // leaves alone.
        let left = unsafe { try_copy(dst[at..].as_mut_ptr(), (src + at) as *const u8, len - at) };
        if left == 0 {
            parts.push((at, len));
            break;
        }
        // Read up to the piece the fault is in, which is left out whole.
        let stop = len - left;
        let read = stop.saturating_sub((src + stop) % PIECE).max(at);
        if read > at {
            parts.push((at, read));
        }
        // Past what cannot be read, to the end of the piece it ends in.
        let past = stop + out_of_reach(src + stop, false).min(len - stop);
        at = if past < len && !(src + past).is_multiple_of(PIECE) {
            piece_end(past)
        } else {
            past
        };
    }
    parts
}

/// How many bytes cannot be read (with `write`, written) from `at`, where
/// a copy faulted: to the end of its region if that is not committed or
/// not readable (writable), else (a page that changed since, or failed
/// otherwise) to the end of its 4 KiB page. One query per region out of
/// reach: the kernel's copy tried each 4 KiB page of it.
fn out_of_reach(at: usize, write: bool) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let page_end = KERNEL_PAGE - at % KERNEL_PAGE;
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    let reachable = info.State == MEM_COMMIT
        && if write {
            can_write(info.Protect)
        } else {
            info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
        };
    if n == 0 || reachable {
        return page_end;
    }
    (info.BaseAddress as usize + info.RegionSize)
        .saturating_sub(at)
        .max(page_end)
}

/// Reads title memory at `src` into `dst`; false if a page of it cannot be
/// read.
fn read_whole(src: usize, dst: &mut [u8]) -> bool {
    if !plain() {
        return crate::host::read_into(src as *const u8, dst);
    }
    // SAFETY: our buffer; title memory a fault leaves alone.
    unsafe { try_copy(dst.as_mut_ptr(), src as *const u8, dst.len()) == 0 }
}

/// Writes `len` bytes of ours at `src` into title memory at `dst`, whose
/// guarded blocks the caller opened ([`crate::watch::writing`]; one a
/// refresh guarded again meanwhile is opened at the store's fault,
/// [`stopped`]): plain stores where it is mapped and writable, the pages
/// that are not left alone, as the kernel's copy left them before W1.
/// (Before, the kernel's copy took over from the first such page, and it
/// fails on a block a refresh guarded again meanwhile: the GPU's bytes
/// there were lost without a word.)
fn write_title(dst: usize, src: *const u8, len: usize) {
    if !plain() {
        crate::host::write_mapped(dst as *mut u8, src, len);
        return;
    }
    let (mut at, mut again) = (0, usize::MAX);
    while at < len {
        // SAFETY: title memory, as the caller vouches; a fault ends it.
        let left = unsafe { try_copy((dst + at) as *mut u8, src.wrapping_add(at), len - at) };
        if left == 0 {
            break;
        }
        at = len - left;
        // Writable since the fault (opened, or mapped again): once more.
        if at != again && writable(dst + at) {
            again = at;
            continue;
        }
        at += out_of_reach(dst + at, true).min(left);
    }
}

/// First-touch census of the copy-back (diagnostics; `XTP_COPYBACK_CENSUS`,
/// docs/analysis/lazy-copyback.md): of the title memory copy-backs cover,
/// how much the CPU touches before the GPU writes it again. The idea is xcl
/// issue #1640's (its census found about 96 % of the GPU's stores into
/// title memory overwritten before any CPU read); none of its code is
/// taken.
///
/// After each copy-back ([`Shadow::copy_back`], [`Shadow::write_back`]),
/// every 4 KiB page of its range gets a guard (`PAGE_GUARD`); the first
/// access to a page faults once into [`stopped`], which counts it (a read
/// or a write, by XTP's code or another's) and lets it go on: the system
/// takes the guard off. The next copy-back over the page counts what no
/// access reached as written again by the GPU first. Only what the copy-back
/// covers is counted: the GPU's bytes it writes back, and the whole range it
/// compares, which a copy-back on first touch would leave alone.
///
/// Which memory gets guards: with `XTP_COPYBACK_CENSUS=1`, only blocks the
/// guard of view memory keeps read-only ([`crate::watch`]). The runtime opens
/// those before every kernel write into title memory (file reads,
/// DirectStorage), which takes the census's guard off with the guard's
/// protection, so no system call fails on it. Read-write memory (write-
/// watched, or view blocks written since they were guarded) gets guards
/// with `XTP_COPYBACK_CENSUS=all` only: there a kernel write fails once on
/// the guard, which BTP 6's synchronous reads retry, but a DirectStorage or
/// `fread` read into it would fail.
///
/// shortcut: what the census does not see, and what it costs. A kernel read
/// of a guarded page (`ReadProcessMemory`: the title copies' and some
/// texture reads, `host::copy_mapped`) fails on the guard and takes it off
/// without a fault; the page then counts as not touched. `copy_mapped` reads
/// again piece by piece after a failure, and meets each other guard once
/// more: with a census, such copies give the GPU zeros there (1–5 MB a frame
/// in the Series S world; docs/analysis/lazy-copyback.md). A kernel or runtime write the journal
/// sees counts as written unseen. A write to a read-only view block opens
/// the whole block ([`crate::watch`]), its other guards with it: the rest of
/// its pages count as uncounted after a write. A guard the guard of view
/// memory takes off otherwise (the other views of a page it guards, a
/// remapping) goes unnoticed.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) mod census {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    use super::{KERNEL_PAGE, PAGE};

    /// The 4 KiB pages of a 64 KiB block (the census's unit, and that of
    /// protections).
    const SMALL: usize = PAGE / KERNEL_PAGE;
    /// Instruction addresses kept of the touches.
    const RIPS: usize = 32;

    /// Counters since the last line: bytes (of 4 KiB pages), but where said.
    #[derive(Clone, Copy)]
    pub(crate) enum K {
        /// Bytes of copied-back ranges guarded, and those left without a
        /// guard (read-write memory without `all`, memory neither watched
        /// nor a view, a protection change that failed).
        Armed,
        Unarmed,
        /// Guarded bytes a later copy-back covered before any touch: the
        /// GPU wrote them again first.
        Again,
        /// Guarded bytes touched first by a read, or a write, of code other
        /// than XTP's (the title's, the system's it calls), and by XTP's
        /// own (its copies, its texture hashing).
        Read,
        Write,
        Ours,
        /// Guarded bytes written without a fault (the journal saw it: a
        /// kernel or runtime write, another view of the page).
        Unseen,
        /// Guarded bytes uncounted after a write to their block: to a
        /// read-only block, which opens the whole block; to another, when
        /// the journal then finds the block written (it knows blocks, not
        /// pages).
        AfterWrite,
        /// Bytes of the 64 KiB blocks with a touch (a copy-back on first
        /// touch by block would compare those).
        Blocks,
        /// Bytes the GPU changed that a copy-back wrote into guarded pages;
        /// of them, in pages touched (or written unseen, or uncounted after
        /// a write) before a later copy-back, and in pages a later
        /// copy-back covered first.
        Gpu,
        GpuTouched,
        GpuAgain,
        /// Faults on guarded pages (a count: guard faults, and stores that
        /// fault on the read-only protection under a guard first),
        /// protection changes (a count), and the census's own time in its
        /// copy-back steps (microseconds).
        Faults,
        Protections,
        Us,
    }
    const COUNTERS: usize = 15;
    static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

    fn add(k: K, n: u64) {
        COUNT[k as usize].fetch_add(n, Relaxed);
    }

    /// The counters since the last call, cleared.
    pub(crate) fn take() -> [u64; COUNTERS] {
        std::array::from_fn(|i| COUNT[i].swap(0, Relaxed))
    }

    /// What a census guards: 0 nothing (no census), 1 read-only view
    /// blocks, 2 all.
    pub(crate) fn mode() -> u8 {
        #[cfg(test)]
        {
            let forced = super::tests::CENSUS.load(Relaxed);
            if forced != 0 {
                return forced - 1;
            }
        }
        static MODE: OnceLock<u8> = OnceLock::new();
        *MODE.get_or_init(|| {
            match std::env::var("XTP_COPYBACK_CENSUS")
                .as_deref()
                .map(str::trim)
            {
                Ok("1" | "on") => 1,
                Ok("all") => 2,
                _ => 0,
            }
        })
    }

    /// A 64 KiB block of title memory a census guarded.
    struct Block {
        /// The submission whose copy-back guarded it last.
        submission: u64,
        /// The journal's asking after that copy-back's own writes.
        asked: u64,
        /// Its 4 KiB pages counted (bit i: the i-th), and those guarded as
        /// far as the census knows.
        armed: u16,
        guarded: u16,
        /// The bytes the GPU wrote back into each page then.
        gpu: [u16; SMALL],
        /// Its protection under the guard is read-only (the guard of view
        /// memory's): a write opens the whole block.
        read_only: bool,
        /// A copy-back over it is under way: its faults are the copy-back's.
        busy: bool,
        /// A page of it was touched since it was guarded, and written (a
        /// guard fault on a store).
        touched: bool,
        wrote: bool,
    }

    struct Census {
        blocks: BTreeMap<usize, Block>,
        /// Every block a census guarded: a guard fault there is the
        /// census's, whatever it knows of the block since.
        ever: BTreeSet<usize>,
        /// Where touches came from: instruction address, touches.
        rips: Vec<(usize, u64)>,
    }

    static CENSUS: Mutex<Census> = Mutex::new(Census {
        blocks: BTreeMap::new(),
        ever: BTreeSet::new(),
        rips: Vec::new(),
    });

    /// Taken by the handler on the faulting thread: nothing holds it while
    /// it touches title memory.
    fn census() -> std::sync::MutexGuard<'static, Census> {
        CENSUS.lock().unwrap_or_else(|e| e.into_inner())
    }

    thread_local! {
        /// The GPU's bytes the copy-back under way on this thread wrote, as
        /// (4 KiB page number, bytes), while it is counted.
        static WROTE: std::cell::RefCell<Option<Vec<(usize, u32)>>> =
            const { std::cell::RefCell::new(None) };
    }

    /// XTP's executable image (its code faults as XTP's own), and the copy's
    /// instruction.
    static IMAGE: OnceLock<(usize, usize)> = OnceLock::new();

    /// Notes where XTP's image lies (with the handler's registration). In
    /// tests none: the tests' accesses stand in for the title's.
    pub(crate) fn find_our_image() {
        #[cfg(not(test))]
        {
            use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
            use windows_sys::Win32::System::ProcessStatus::{K32GetModuleInformation, MODULEINFO};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: the process's own executable; a sized structure.
            let image = unsafe {
                let module = GetModuleHandleW(std::ptr::null());
                let mut info: MODULEINFO = std::mem::zeroed();
                let ok = K32GetModuleInformation(
                    GetCurrentProcess(),
                    module,
                    &mut info,
                    size_of::<MODULEINFO>() as u32,
                ) != 0;
                let base = info.lpBaseOfDll as usize;
                if ok {
                    (base, base + info.SizeOfImage as usize)
                } else {
                    (0, 0)
                }
            };
            let _ = IMAGE.set(image);
        }
    }

    fn ours(rip: usize) -> bool {
        let (from, to) = IMAGE.get().copied().unwrap_or_default();
        rip == &raw const super::xtp_gpu_try_copy_moving as usize || (from..to).contains(&rip)
    }

    /// The bits of the 4 KiB pages `[from, to)` of a block.
    fn mask(from: usize, to: usize) -> u16 {
        ((((1u32 << to) - 1) & !((1u32 << from) - 1)) & 0xffff) as u16
    }

    fn bits(mask: u16) -> impl Iterator<Item = usize> {
        (0..SMALL).filter(move |i| mask & (1 << i) != 0)
    }

    /// The journal's last writing of `block`'s page (0: none known).
    fn last_write(written: &[(usize, u64)], block: usize) -> u64 {
        written
            .binary_search_by_key(&block, |&(p, _)| p)
            .map_or(0, |i| written[i].1)
    }

    /// How a copy-back takes part: whether the census counts it, and
    /// guards its range after.
    #[derive(Clone, Copy)]
    pub(crate) struct Back {
        counted: bool,
        view: bool,
    }

    /// Before a copy-back of `submission` writes title memory `[from, to)`
    /// (`view`: whether it is a view of physical pages, asked only with a
    /// census): what a census guarded there and no access reached, the GPU
    /// wrote again first. Faults there until [`after_back`] are the
    /// copy-back's own.
    pub(crate) fn before_back(
        submission: u64,
        from: usize,
        to: usize,
        view: impl FnOnce() -> bool,
    ) -> Back {
        let mode = mode();
        if mode == 0 || from >= to {
            return Back {
                counted: false,
                view: false,
            };
        }
        let view = view();
        if mode == 1 && !view {
            // Read-write memory, which a census leaves alone without `all`.
            add(
                K::Unarmed,
                (to - from).div_ceil(KERNEL_PAGE) as u64 * KERNEL_PAGE as u64,
            );
            return Back {
                counted: false,
                view,
            };
        }
        let started = std::time::Instant::now();
        // The journal as before the copy-back's own writes.
        let asked = crate::watch::writes(from, to, true);
        {
            let mut census = census();
            for (&block, b) in census.blocks.range_mut(from / PAGE..=(to - 1) / PAGE) {
                b.busy = true;
                if b.submission == submission {
                    // Another copy of the same title memory, back from the
                    // same submission: the same GPU work.
                    continue;
                }
                let written = asked.as_ref().map_or(0, |(_, w)| last_write(w, block));
                // Written since it was guarded: a write the guard did not see
                // (which may have taken the guards off), unless a store to
                // one of its pages faulted, which the journal cannot tell
                // apart (it knows blocks).
                let unseen = written > b.asked;
                for i in bits(b.armed & b.guarded) {
                    let (bytes, gpu) = match (unseen, b.wrote) {
                        (false, _) => (K::Again, K::GpuAgain),
                        (true, false) => (K::Unseen, K::GpuTouched),
                        (true, true) => (K::AfterWrite, K::GpuTouched),
                    };
                    add(bytes, KERNEL_PAGE as u64);
                    add(gpu, u64::from(b.gpu[i]));
                }
                b.armed = 0;
                if unseen {
                    b.guarded = 0;
                }
            }
        }
        WROTE.set(Some(Vec::new()));
        add(K::Us, started.elapsed().as_micros() as u64);
        Back {
            counted: true,
            view,
        }
    }

    /// The copy-back under way on this thread wrote the GPU's bytes into
    /// title memory `[at, at + len)`.
    pub(crate) fn wrote(at: usize, len: usize) {
        WROTE.with_borrow_mut(|wrote| {
            let Some(wrote) = wrote else {
                return;
            };
            let mut a = at;
            while a < at + len {
                let end = (a / KERNEL_PAGE + 1) * KERNEL_PAGE;
                let n = end.min(at + len) - a;
                match wrote.last_mut() {
                    Some((page, bytes)) if *page == a / KERNEL_PAGE => *bytes += n as u32,
                    _ => wrote.push((a / KERNEL_PAGE, n as u32)),
                }
                a += n;
            }
        });
    }

    /// After the copy-back of `[from, to)` by `submission` ([`before_back`]
    /// told how it takes part): guards each 4 KiB page of the range, for
    /// the first access to tell.
    pub(crate) fn after_back(back: Back, submission: u64, from: usize, to: usize) {
        if !back.counted {
            return;
        }
        super::install();
        let started = std::time::Instant::now();
        let mut wrote = WROTE.take().unwrap_or_default();
        wrote.sort_unstable();
        // The journal after the copy-back's own writes (its asking): a write
        // it finds now is another's, made while the copy-back ran.
        let asked = crate::watch::writes(from, to, true);
        let mut census = census();
        let census = &mut *census;
        if census.rips.capacity() < RIPS {
            // Room for the handler, which does not allocate.
            census.rips.reserve(RIPS);
        }
        for block in from / PAGE..=(to - 1) / PAGE {
            let at = block * PAGE;
            let (a, b) = (
                (from.max(at) - at) / KERNEL_PAGE,
                (to.min(at + PAGE) - at).div_ceil(KERNEL_PAGE),
            );
            let pages = mask(a, b);
            let bytes = ((b - a) * KERNEL_PAGE) as u64;
            let Some((now, written)) = &asked else {
                add(K::Unarmed, bytes);
                continue;
            };
            let mut gpu = [0u16; SMALL];
            let first = wrote.partition_point(|&(p, _)| p < at / KERNEL_PAGE);
            let end = (at + PAGE) / KERNEL_PAGE;
            for &(page, n) in wrote[first..].iter().take_while(|&&(p, _)| p < end) {
                let n = n.min(u32::from(u16::MAX)) as u16;
                gpu[page % SMALL] = gpu[page % SMALL].saturating_add(n);
            }
            let ours = gpu.iter().any(|&n| n != 0);
            // Where the copy-back wrote, its write and another's look alike.
            let theirs = !ours && last_write(written, block) == *now;
            let Census { blocks, ever, .. } = census;
            let entry = blocks.entry(block).or_insert(Block {
                submission,
                asked: 0,
                armed: 0,
                guarded: 0,
                gpu,
                read_only: back.view,
                busy: false,
                touched: false,
                wrote: false,
            });
            // Still guarded where it counts, and nothing took the guards
            // off: our write (the guard of view memory opens the block
            // first), another's while the copy-back ran.
            let intact = entry.guarded & pages == pages && !ours && !theirs;
            *entry = Block {
                submission,
                asked: *now,
                armed: pages,
                guarded: entry.guarded,
                gpu,
                read_only: entry.read_only,
                busy: false,
                touched: false,
                wrote: false,
            };
            if theirs {
                // Touched (written) as it came back: a touch before the
                // next GPU write.
                add(K::Unseen, bytes);
                add(K::GpuTouched, gpu.iter().map(|&n| u64::from(n)).sum());
                entry.armed = 0;
                continue;
            }
            if !intact {
                // Known as the census's before it is guarded: a fault there
                // waits for the lock, then finds it.
                ever.insert(block);
                match guard(at + a * KERNEL_PAGE, b - a, back.view) {
                    Some(read_only) => {
                        entry.guarded |= pages;
                        entry.read_only = read_only;
                    }
                    None => {
                        entry.armed = 0;
                        entry.guarded &= !pages;
                        add(K::Unarmed, bytes);
                        continue;
                    }
                }
            }
            add(K::Armed, bytes);
            add(K::Gpu, gpu.iter().map(|&n| u64::from(n)).sum());
        }
        add(K::Us, started.elapsed().as_micros() as u64);
    }

    /// Guards `pages` 4 KiB pages from `at`, keeping the protection under
    /// the guard: read-only for view memory, which the guard of view memory
    /// keeps so (read-write too with `all`), read-write for other memory
    /// (`all` only). Returns whether the protection under it is read-only,
    /// or `None` (the protection as it was) for any other protection.
    fn guard(at: usize, pages: usize, view: bool) -> Option<bool> {
        use windows_sys::Win32::System::Memory::{
            PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, VirtualProtect,
        };
        let len = pages * KERNEL_PAGE;
        let set = |protection: u32| {
            add(K::Protections, 1);
            let mut old = 0;
            // SAFETY: title memory; only its protection changes, and a
            // fault on the guard is the census's (the caller noted the
            // block first).
            let ok = unsafe { VirtualProtect(at as *const _, len, protection, &mut old) } != 0;
            ok.then_some(old)
        };
        let expected = if view { PAGE_READONLY } else { PAGE_READWRITE };
        let old = set(expected | PAGE_GUARD)?;
        let under = old & !PAGE_GUARD;
        if under == expected {
            return Some(view);
        }
        if view && under == PAGE_READWRITE && mode() == 2 {
            // A view block written since it was guarded: read-write.
            return set(PAGE_READWRITE | PAGE_GUARD).map(|_| false);
        }
        // Not what the guard or the title leaves there (another modifier,
        // no access): as it was.
        set(old);
        None
    }

    /// The handler, at a guard fault on `at` (`write`: a store) by the
    /// instruction at `rip`: whether the census guarded it, then counted
    /// and let go on (the system took the guard off).
    pub(crate) fn fault(write: bool, at: usize, rip: usize) -> bool {
        if mode() == 0 {
            return false;
        }
        let mut census = census();
        if !census.ever.contains(&(at / PAGE)) {
            return false;
        }
        touch(&mut census, write, at, rip);
        true
    }

    /// The handler, at a write fault on `at` by the instruction at `rip`:
    /// whether it is in a block the census guarded, and writable now, then
    /// let go on. A store to a read-only page faults on its protection
    /// before its guard (which stays): the census counts it here, and the
    /// guard of view memory then opens the block, its guards with it. A
    /// store also faults on a block the census made read-only for a moment
    /// (a protection it gives back): writable once that is done.
    pub(crate) fn settle(at: usize, rip: usize) -> bool {
        if mode() == 0 {
            return false;
        }
        {
            let mut census = census();
            if !census.ever.contains(&(at / PAGE)) {
                return false;
            }
            touch(&mut census, true, at, rip);
        }
        super::writable(at)
    }

    /// Counts the first access to the 4 KiB page of `at`, if guarded and
    /// counted, by the instruction at `rip`.
    fn touch(census: &mut Census, write: bool, at: usize, rip: usize) {
        add(K::Faults, 1);
        let Some(b) = census.blocks.get_mut(&(at / PAGE)) else {
            return;
        };
        let i = at % PAGE / KERNEL_PAGE;
        let bit = 1u16 << i;
        if b.armed & b.guarded & bit != 0 && !b.busy {
            let kind = if ours(rip) {
                K::Ours
            } else if write {
                K::Write
            } else {
                K::Read
            };
            add(kind, KERNEL_PAGE as u64);
            add(K::GpuTouched, u64::from(b.gpu[i]));
            b.armed &= !bit;
            b.wrote |= write;
            if !b.touched {
                b.touched = true;
                add(K::Blocks, PAGE as u64);
            }
        }
        b.guarded &= !bit;
        if write && b.read_only {
            // The guard of view memory opens the whole block for the store:
            // the other guards go with its read-only protection.
            if !b.busy {
                for j in bits(b.armed & b.guarded) {
                    add(K::AfterWrite, KERNEL_PAGE as u64);
                    add(K::GpuTouched, u64::from(b.gpu[j]));
                }
            }
            (b.armed, b.guarded) = (0, 0);
        }
        let rips = &mut census.rips;
        if let Some(r) = rips.iter_mut().find(|r| r.0 == rip) {
            r.1 += 1;
        } else if rips.len() < rips.capacity() {
            rips.push((rip, 1));
        } else if let Some(r) = rips.iter_mut().min_by_key(|r| r.1) {
            *r = (rip, 1);
        }
    }

    /// Logs the counters per frame over `frames` frames, and where the
    /// touches came from.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn log(frames: f64) {
        if mode() == 0 {
            return;
        }
        let c = take().map(|v| v as f64 / frames);
        let (armed, mut rips) = {
            let mut census = census();
            let armed: u64 = census
                .blocks
                .values()
                .map(|b| u64::from((b.armed & b.guarded).count_ones()))
                .sum();
            let rips = std::mem::take(&mut census.rips);
            (armed * KERNEL_PAGE as u64, rips)
        };
        let mb = |k: K| c[k as usize] / 1e6;
        tracing::info!(
            "copy-back census per frame (4 KiB pages): guarded {:.1} MB (not guarded {:.1} MB); \
             the GPU wrote {:.1} MB again before any touch; touched first {:.2} MB (reads \
             {:.2}, writes {:.2}, XTP's own {:.2}; {:.1} MB of 64 KiB blocks), written unseen \
             {:.2} MB, uncounted after a write {:.2} MB; the GPU's bytes written back {:.3} MB: \
             {:.3} MB touched, {:.3} MB written again first; {:.0} guard faults, {:.0} \
             protection changes, census {:.2} ms; {:.1} MB guarded now",
            mb(K::Armed),
            mb(K::Unarmed),
            mb(K::Again),
            mb(K::Read) + mb(K::Write) + mb(K::Ours),
            mb(K::Read),
            mb(K::Write),
            mb(K::Ours),
            mb(K::Blocks),
            mb(K::Unseen),
            mb(K::AfterWrite),
            mb(K::Gpu),
            mb(K::GpuTouched),
            mb(K::GpuAgain),
            c[K::Faults as usize],
            c[K::Protections as usize],
            c[K::Us as usize] / 1e3,
            armed as f64 / 1e6,
        );
        rips.sort_unstable_by_key(|r| std::cmp::Reverse(r.1));
        for (rip, n) in rips.into_iter().take(8) {
            tracing::info!(
                "  census touches at {}: {:.1}/frame",
                place(rip),
                n as f64 / frames
            );
        }
    }

    /// `rip` as module+offset, when a module the system knows holds it.
    #[cfg(feature = "diagnostics")]
    fn place(rip: usize) -> String {
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleFileNameW, GetModuleHandleExW,
        };
        let mut module = std::ptr::null_mut();
        // SAFETY: an address as the module name, with the flag that says so;
        // the reference count stays as it is.
        let found = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                rip as *const u16,
                &mut module,
            )
        } != 0;
        if !found {
            return format!("{rip:#x} (no module the system knows: the title's image?)");
        }
        let mut name = [0u16; 260];
        // SAFETY: a buffer of its stated length.
        let n =
            unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
        let path = String::from_utf16_lossy(&name[..n]);
        let file = path.rsplit('\\').next().unwrap_or(&path);
        format!("{file}+{:#x}", rip - module as usize)
    }
}

/// `XTP_FULL_REFRESH`, a diagnostic (slow): host copies brought up to date
/// whatever the write tracking (journal, guards, kernel) says, to tell
/// whether a copy the tracking left stale is what a picture lacks. The
/// idea is the fork's (watch item 226: with `VICTORY_FULL_REFRESH=1`, walls
/// that drew nothing came back in 3 runs of 3).
///
/// - `ranges`: each refresh copies its whole range from title memory, as
///   for memory that is not watched (the fork's switch).
/// - `<n>`: before every n-th submission's lists, every host copy is copied
///   whole from title memory, but where GPU work in flight writes, through
///   any copy ([`refresh_everything`]); also the memory no refresh covers
///   (buffers read through directly indexed heaps). `1` is every
///   submission: about 2 GB of copies each time in the Series S world
///   (its System pool), so `30`, about once a frame, keeps it usable.
///   First each 64 KiB piece is compared: one the copy holds otherwise
///   than title memory while the tracking has it current is a write the
///   tracking missed, counted on the copystat line, the first 64 logged
///   with the memory they are in (watch item 237: in the fork, missed
///   writes came through another view of the same physical pages). A
///   shader write outside the ranges it declared, into a copy in system
///   memory, counts too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FullRefresh {
    Off,
    Ranges,
    Every(u64),
}

fn full_refresh() -> FullRefresh {
    static SETTING: std::sync::OnceLock<FullRefresh> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| {
        let value = std::env::var("XTP_FULL_REFRESH").unwrap_or_default();
        let setting = match value.trim() {
            "" | "0" => FullRefresh::Off,
            "ranges" => FullRefresh::Ranges,
            n => match n.parse::<u64>() {
                Ok(n) if n > 0 => FullRefresh::Every(n),
                _ => {
                    tracing::warn!(
                        value,
                        "XTP_FULL_REFRESH: 0, ranges or a number of submissions"
                    );
                    FullRefresh::Off
                }
            },
        };
        if setting != FullRefresh::Off {
            tracing::info!(
                ?setting,
                "host copies refreshed whole, whatever the write tracking says (XTP_FULL_REFRESH)"
            );
        }
        setting
    })
}

/// Whether refreshes copy their whole range (`XTP_FULL_REFRESH=ranges`).
fn whole_ranges() -> bool {
    full_refresh() == FullRefresh::Ranges
}

/// With `XTP_FULL_REFRESH=<n>`, before `submission`'s lists if its number
/// is a multiple of `n`: every host copy (title buffers' and loose memory's)
/// copied whole from title memory, but where a submission not copied back
/// yet writes through any copy: there title memory is older than the GPU's
/// bytes. Returns the bytes copied.
pub(crate) fn refresh_due(submission: u64) -> bool {
    matches!(full_refresh(), FullRefresh::Every(n) if submission.is_multiple_of(n))
}

pub(crate) fn refresh_everything(submission: u64) -> usize {
    let FullRefresh::Every(n) = full_refresh() else {
        return 0;
    };
    if !submission.is_multiple_of(n) {
        return 0;
    }
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let mut copies: Vec<Arc<Shadow>> = buffers()
        .values()
        .filter_map(|e| e.shadow.clone())
        .collect();
    copies.extend(lock(&LOOSE).values().cloned());
    let mut busy = Vec::new();
    for shadow in &copies {
        busy.extend(shadow.busy().iter().map(|b| (b.from, b.to)));
    }
    let busy = joined(busy);
    let (mut bytes, mut missed) = (0, 0);
    for shadow in &copies {
        let (from, to) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        let mut at = from;
        let mut take = |a: usize, b: usize| {
            missed += shadow.missed(a, b);
            bytes += shadow.copy(a, b);
        };
        for &(a, e) in busy.iter().filter(|&&(a, e)| a < to && e > from) {
            if a > at {
                take(at, a);
            }
            at = at.max(e);
        }
        if at < to {
            take(at, to);
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::FullRefreshes, 1);
        add(C::FullRefreshBytes, bytes as u64);
        add(C::FullRefreshMissed, missed as u64);
        add(C::FullRefreshUs, started.elapsed().as_micros() as u64);
    }
    let _ = missed;
    bytes
}

impl Shadow {
    /// For [`refresh_everything`]: the bytes of the 64 KiB pieces of title
    /// memory `[from, to)` (within the copy) that the copy holds otherwise
    /// than title memory while the write tracking says it holds them as
    /// they are ([`Shadow::stale`] false): writes it missed. The first ones
    /// are logged with what memory they are in (a view of physical pages,
    /// and at how many other addresses the title shows that page, or other
    /// memory).
    /// Pieces that cannot all be read are left out.
    fn missed(&self, from: usize, to: usize) -> usize {
        /// Pieces logged at most, over the run.
        static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        thread_local! {
            static PIECE_BYTES: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let mut missed = 0;
        let mut at = from;
        PIECE_BYTES.with_borrow_mut(|title| {
            while at < to {
                let end = (at + PIECE - at % PIECE).min(to);
                title.resize(end - at, 0);
                // SAFETY: inside the copy's mapping; only this thread writes
                // the copy outside ranges in flight, which are left out.
                let held =
                    unsafe { std::slice::from_raw_parts(self.data.add(at - start), end - at) };
                if read_whole(at, title) && held != &title[..] && !self.stale(at, end) {
                    let differ = held
                        .iter()
                        .zip(title.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    missed += end - at;
                    if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
                        use windows_sys::Win32::System::Memory::{
                            MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
                        };
                        // SAFETY: plain data filled in by the query.
                        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        // SAFETY: a valid out pointer of the right size.
                        unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
                        tracing::info!(
                            at = format_args!("{at:#x}"),
                            len = end - at,
                            differ,
                            copy = format_args!("{start:#x}+{:#x}", self.len),
                            view = info.Type == MEM_MAPPED,
                            other_views = crate::watch::views_of(at),
                            protect = format_args!("{:#x}", info.Protect),
                            region = format_args!("{:#x}", info.AllocationBase as usize),
                            watched = self.watched.load(Ordering::Relaxed),
                            "a write the tracking missed (XTP_FULL_REFRESH)"
                        );
                    }
                }
                at = end;
            }
        });
        missed
    }
}

/// `mutex`'s guard, poisoned or not.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the copies of big title buffers are in the card's own memory
/// (see the module): with `XTP_CARD_BUFFERS=1`. Off by default, every copy
/// in system memory as before: there shaders read and write it through
/// PCIe, and finding what they wrote takes a snapshot of each declared
/// range before the submission and a compare after it, on the CPU: 41.9
/// and about 80 ms of a Series S world frame, for about 620 MB a frame
/// (docs/analysis/lazy-copyback.md, session lazyback-mirrors-ab).
///
/// shortcut: off until measured on the PC. Its one Series S session there
/// (n1b-cards, 2026-11-25 01:28) had a title submission run 19.7 s on the
/// GPU, and the title stopped itself (its watchdog, a write to 0 at RVA
/// `0x23cf683`); the cause is not known (the card's pool takes 512 MiB of
/// the card's memory more, near the VidMm admission threshold of
/// docs/analysis/stall.md?). On the cloud L4, 214 s of world ran with it
/// and no slow frame, but the merge of what the card found changed still
/// cost 71 to 78 ms a frame there, and copies in system memory compared
/// about 230 MB a frame.
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XTP_CARD_BUFFERS").is_ok_and(|v| v.trim() == "1");
        if on {
            tracing::info!("copies of big title buffers in the card's memory (XTP_CARD_BUFFERS=1)");
        }
        on
    })
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies at once: `XTP_CARD_BUFFERS_MIN_MB` (16) and
/// `XTP_CARD_BUFFERS_MB` (512). A buffer past the budget keeps its copy in
/// system memory, as before. The Series S world makes seven buffers of 24
/// MB or more, 383 MB in all (session lazyback-mirrors-ab).
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(default)
                << 20
        };
        (
            mb("XTP_CARD_BUFFERS_MIN_MB", 16).max(1 << 16),
            mb("XTP_CARD_BUFFERS_MB", 512),
        )
    })
}

/// For tests: bytes of pieces the card found changed, at copy-backs of
/// copies in its memory ([`Shadow::card_back`]).
#[cfg(test)]
pub(crate) static CARD_CHANGED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Bytes of the card sides alive.
static CARD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The copies in the card's memory, for [`cards_around`].
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Notes `shadow` for [`cards_around`] if it is in the card's memory.
fn carded(shadow: &Arc<Shadow>) {
    if shadow.card.is_some() {
        lock(&CARDED).push(Arc::downgrade(shadow));
    }
}

/// For the frame-rate line: the copies in the card's memory and their
/// bytes there.
pub(crate) fn card_summary() -> Option<String> {
    if !cards() {
        return None;
    }
    let copies = lock(&CARDED)
        .iter()
        .filter(|w| w.strong_count() > 0)
        .count();
    Some(format!(
        "{copies} title buffer copies on the card, {} MiB",
        CARD_BYTES.load(Ordering::Relaxed) >> 20
    ))
}

/// The card side of a copy of `size` bytes whose CPU side is `staging`,
/// and the rest of its [`Card`], if the copy is to be in the card's memory:
/// [`cards`] is on, `size` is within [`card_limits`] and below 4 GiB (the
/// card's passes count in 32 bits), and the card's passes could be made
/// ([`work`]).
fn card_for(
    host: &crate::host::Host,
    size: u64,
    staging: &ID3D12Resource,
) -> Option<(ID3D12Resource, Card)> {
    let (least, most) = card_limits();
    if !cards() || size < least || size >= 1 << 32 {
        return None;
    }
    let work = work()?;
    // `size` bytes of the budget, if they fit.
    let reserve = || {
        let fits = CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size <= most;
        if !fits {
            CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        }
        fits
    };
    let mut reserved = reserve();
    if !reserved {
        // Copies the title let go may still be held by plans that have run.
        lock(work).let_go();
        reserved = reserve();
    }
    if !reserved {
        tracing::info!(
            size,
            most,
            "a big title buffer's copy stays in system memory: the card's budget for them is \
             taken (XTP_CARD_BUFFERS_MB)"
        );
        return None;
    }
    let made = crate::host::buffers::card_buffer(&host.device, size)
        .and_then(|buffer| Some((buffer, host.shared_buffer(size)?)));
    let Some((buffer, (after, after_data))) = made else {
        CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    };
    let card = Card {
        staging: staging.clone(),
        after,
        after_data,
        // The card side's bytes are undefined until the CPU side's have
        // gone up: all of them first.
        dirty: Mutex::new(vec![(0, size as usize)]),
        parts: Mutex::default(),
        bytes: size,
    };
    Some((buffer, card))
}

impl Part {
    /// Whether the card found the piece of record `i` changed.
    fn changed(&self, i: u32) -> bool {
        // SAFETY: a record of the part's slot, which stays until the part is
        // done with; the card wrote it before the submission finished.
        let flag = unsafe {
            self.records
                .add(i as usize * RECORD + 8)
                .cast::<u32>()
                .read_volatile()
        };
        flag != 0
    }
}

impl Shadow {
    /// [`Shadow::copy_back`] of a copy in the card's memory. The pieces of
    /// `[address, address + len)` whose hash changed on the card while the
    /// submission's lists ran ([`cards_around`]) are merged as a system
    /// memory copy's ([`Shadow::put_back`]): the card's bytes, which it
    /// copied into `after`, against the CPU side's from before (which
    /// nothing wrote since), and title memory. The other pieces the GPU
    /// left as they were: nothing to compare. The CPU side takes what the
    /// merge made, and the card side the title's bytes it kept (`dirty`).
    fn card_back(&self, card: &Card, submission: u64, address: usize, len: usize) -> usize {
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        let mut written = 0;
        if from < to {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            // The range's pieces the card compared, and those it found
            // changed.
            let (compared, changed) = {
                let parts = lock(&card.parts);
                match parts.iter().find(|p| p.submission == submission) {
                    Some(part) => {
                        let inside = || {
                            part.spans
                                .iter()
                                .filter(move |&&(a, b, _)| a < to && b > from)
                        };
                        let compared: usize = inside().map(|&(a, b, _)| b - a).sum();
                        let changed: Vec<(usize, usize)> = inside()
                            .filter(|&&(_, _, i)| part.changed(i))
                            .map(|&(a, b, _)| (a.max(from), b.min(to)))
                            .collect();
                        (compared, changed)
                    }
                    None => {
                        tracing::warn!(
                            address = format_args!("{:#x}", start + from),
                            len = to - from,
                            "a range shaders wrote in a copy on the card was not compared: \
                             what they wrote there does not reach title memory"
                        );
                        (0, Vec::new())
                    }
                }
            };
            #[cfg(any(test, feature = "diagnostics"))]
            let census = census::before_back(submission, start + from, start + to, || {
                self.census_view(start + from)
            });
            let mut synced = Vec::new();
            let mut i = 0;
            while i < changed.len() {
                // With the pieces after it that touch it, up to a run.
                let (a, mut b) = changed[i];
                i += 1;
                while once() && i < changed.len() && changed[i].0 == b && b - a < RUN {
                    b = changed[i].1;
                    i += 1;
                }
                written += self.card_put_back(card, submission, a, b, &mut synced);
            }
            #[cfg(test)]
            CARD_CHANGED.fetch_add(
                changed.iter().map(|&(a, b)| b - a).sum::<usize>(),
                Ordering::Relaxed,
            );
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::CardCompared, compared as u64);
                add(
                    C::CardChanged,
                    changed.iter().map(|&(a, b)| b - a).sum::<usize>() as u64,
                );
                add(C::CardBackUs, started.elapsed().as_micros() as u64);
            }
            let _ = compared;
            // As a system memory copy's: the range's whole pages put back
            // are current where title memory equals the CPU side.
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            let mut current: Vec<usize> = synced
                .into_iter()
                .filter(|p| (first..last).contains(p))
                .collect();
            current.sort_unstable();
            current.dedup();
            self.mark_current(&current);
            #[cfg(any(test, feature = "diagnostics"))]
            census::after_back(census, submission, start + from, start + to);
        }
        let end = address + len;
        self.busy().retain(|b| {
            !(b.shaders && b.submission == submission && address <= b.from && b.to <= end)
        });
        self.card_settled(card, submission);
        written
    }

    /// [`Shadow::card_back`] of `[from, to)` (offsets), a run of pieces the
    /// card found changed. Returns the bytes written into title memory.
    fn card_put_back(
        &self,
        card: &Card,
        submission: u64,
        from: usize,
        to: usize,
        synced: &mut Vec<usize>,
    ) -> usize {
        thread_local! {
            /// A run's bytes as the card copied them, and as merged (kept,
            /// already touched, for the next).
            static RUN_BYTES: std::cell::RefCell<(Vec<u8>, Vec<u8>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        RUN_BYTES.with_borrow_mut(|(gpu, merged)| {
            let n = to - from;
            gpu.resize(n, 0);
            merged.resize(n, 0);
            // SAFETY: inside `after`'s mapping, which the card wrote before
            // the submission finished. A later submission's compare may copy
            // its pieces over this range's edges meanwhile: the same bytes,
            // as no GPU work writes the range before it is copied back.
            unsafe {
                std::ptr::copy_nonoverlapping(card.after_data.add(from), gpu.as_mut_ptr(), n)
            };
            merged.copy_from_slice(gpu);
            // SAFETY: inside the CPU side's mapping, which nothing else
            // writes in this range until the submission is copied back.
            let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(from), n) };
            let (written, read) = if crate::readmap::enabled() {
                // put_mapped_range updates the CPU mirror, including holes.
                // The GPU bytes in unreadable pieces stay unchanged in merged.
                (
                    self.put_mapped_range(
                        submission,
                        self.start as usize + from,
                        merged,
                        old,
                        synced,
                    ),
                    vec![(0, n)],
                )
            } else {
                self.put_back(from, to, merged, old, synced)
            };
            // Where title memory could not be read: the CPU side takes the
            // card's bytes, as a copy in system memory keeps the GPU's.
            let mut at = 0;
            for &(a, b) in read.iter().chain([&(n, n)]) {
                if at < a {
                    old[at..a].copy_from_slice(&gpu[at..a]);
                }
                at = b;
            }
            // The title's bytes the merge kept, in place of the GPU's or
            // beside them: the card side lacks them.
            let mut dirty = Vec::new();
            for &(a, b) in &read {
                let mut i = a;
                loop {
                    let x = first_change(&merged[..b], &gpu[..b], i);
                    if x == b {
                        break;
                    }
                    i = first_same(&merged[..b], &gpu[..b], x);
                    dirty.push((from + x, from + i));
                }
            }
            if !dirty.is_empty() {
                lock(&card.dirty).extend(dirty);
            }
            written
        })
    }

    /// After a copy-back or write-back of `submission` on a copy in the
    /// card's memory: once none of its ranges is left, its part is done
    /// with (its slot may serve another plan once its other parts are).
    fn card_settled(&self, card: &Card, submission: u64) {
        if self.busy().iter().any(|b| b.submission == submission) {
            return;
        }
        let part = {
            let mut parts = lock(&card.parts);
            parts
                .iter()
                .position(|p| p.submission == submission)
                .map(|i| parts.swap_remove(i))
        };
        if let (Some(part), Some(work)) = (part, work()) {
            lock(work).done(part.slot);
        }
    }

    /// `ranges` (offsets, sorted, apart) cut at the pages of title memory:
    /// the copy-back's pieces.
    fn pieces(&self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let start = self.start as usize;
        let mut out = Vec::new();
        for &(mut at, to) in ranges {
            while at < to {
                let end = (at + PIECE - (start + at) % PIECE).min(to);
                out.push((at, end));
                at = end;
            }
        }
        out
    }
}

/// `ranges` sorted, those that overlap or touch merged.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// One copy's work in a submission's plan ([`cards_around`]).
struct Job {
    shadow: Arc<Shadow>,
    /// Ranges of the CPU side that go up (offsets, sorted, apart).
    uploads: Vec<(usize, usize)>,
    /// The pieces of the ranges the submission's shaders may write, and of
    /// those its GPU copies write (offsets).
    spans: Vec<(usize, usize)>,
    fetches: Vec<(usize, usize)>,
    /// The first record of each.
    first_span: u32,
    first_fetch: u32,
}

/// For a submission about to run, the lists of ours that go first and last
/// in its `ExecuteCommandLists` (`Host::run_as`), for the copies in the
/// card's memory ([`cards`]); `(None, None)` when there is nothing to do.
///
/// - First: what their CPU side took since the last submission goes up to
///   their card side, then each piece of the ranges its shaders may write
///   is hashed on the card (128 bits of a mix of its bytes, [`CARD_PASS`]).
/// - Last, once its lists have run: each piece is hashed again; a piece
///   whose hash changed is copied into system memory (`Card::after`), and
///   so is what its GPU copies wrote.
///
/// Its copy-back then has what shaders changed, and only that, when the
/// submission's fence is reached ([`Shadow::card_back`]): no snapshot, no
/// compare on the CPU, no wait for the card. In one `ExecuteCommandLists`,
/// lists run one after the other with nothing between them, so the hashes
/// tell the card's bytes at the start and at the end of the submission's
/// lists. Hashing each submission's pieces both times, rather than keeping
/// a hash per piece across submissions, leaves each submission's result to
/// its own bytes: a later submission may write other bytes of a piece
/// before this one is copied back.
pub(crate) fn cards_around(
    submission: u64,
) -> (Option<ID3D12CommandList>, Option<ID3D12CommandList>) {
    if !cards() {
        return (None, None);
    }
    let shadows: Vec<Arc<Shadow>> = {
        let mut all = lock(&CARDED);
        all.retain(|w| w.strong_count() > 0);
        all.iter().filter_map(std::sync::Weak::upgrade).collect()
    };
    let mut jobs = Vec::new();
    for shadow in shadows {
        let Some(card) = &shadow.card else {
            continue;
        };
        let uploads = joined(std::mem::take(&mut *lock(&card.dirty)));
        let (mut written, mut copied) = (Vec::new(), Vec::new());
        for b in shadow.busy().iter().filter(|b| b.submission == submission) {
            let (from, to) = shadow.inside(b.from, b.to - b.from);
            if from < to {
                (if b.shaders { &mut written } else { &mut copied }).push((from, to));
            }
        }
        // Each byte once, as the host merges the ranges it copies back.
        let spans = shadow.pieces(&joined(written));
        let fetches = shadow.pieces(&joined(copied));
        if uploads.is_empty() && spans.is_empty() && fetches.is_empty() {
            continue;
        }
        jobs.push(Job {
            shadow,
            uploads,
            spans,
            fetches,
            first_span: 0,
            first_fetch: 0,
        });
    }
    if jobs.is_empty() {
        return (None, None);
    }
    // Copies are in the card's memory only once the passes are made.
    let Some(work) = work() else {
        return (None, None);
    };
    lock(work).plan(submission, jobs)
}

/// The record of a piece the card's passes take, in a slot's memory: its
/// offsets `[from, to)` in the copy (32 bits each), whether the compare
/// found it changed (32 bits), a word unused, then the hash of its bytes
/// before the submission's lists (128 bits).
const RECORD: usize = 32;

/// Words of the card's marks: two per slot ([`CardWork`]).
const MARKS: usize = 1 << 14;

/// The card's passes over the copies in its memory: one compute program,
/// four ways.
///
/// - 0: hashes each piece of its records into the record.
/// - 1: hashes each again, notes in the record whether the hash changed,
///   and copies a changed piece from `card` into `after`.
/// - 2: copies each piece into `after` (a GPU copy's destination).
/// - 3: sets word `first` of `after` (the marks) to `size` (a ticket).
///
/// A group of 256 threads takes a piece (at most 64 KiB), 16 bytes a step,
/// its ends widened to 16 bytes (inside the buffer, whose size is a
/// multiple of 64 KiB): a byte just outside a piece that changes only makes
/// a compare that finds nothing on the CPU. Each thread mixes its words
/// into 128 bits of its own, every step a bijection of them for a given
/// word, so a change of one word always changes the thread's result; the
/// group's results are joined by xor, which keeps that. Changes of several
/// words could cancel out with odds of about 2^-128.
const CARD_PASS: &str = r#"
#define RS "RootConstants(num32BitConstants=3, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Way : register(b0) {
    uint mode;
    uint first;  // the dispatch's first record (mode 3: the mark's word)
    uint size;   // the copy's bytes on the card (mode 3: the ticket)
};
RWByteAddressBuffer card : register(u0);
RWByteAddressBuffer after : register(u1);
RWByteAddressBuffer records : register(u2);
groupshared uint4 lanes[256];
groupshared uint changed;

uint4 mix(uint4 s) {
    s *= uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu);
    s ^= s >> 15;
    s.x += s.w;
    s.y += s.x;
    s.z += s.y;
    s.w += s.z;
    return s;
}

[RootSignature(RS)]
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    if (mode == 3u) {
        if (group.x == 0u && t == 0u) after.Store(first * 4u, size);
        return;
    }
    uint at = (first + group.x) * 32u;
    uint lo = records.Load(at) & ~15u;
    uint hi = min((records.Load(at + 4u) + 15u) & ~15u, size);
    if (mode != 2u) {
        uint4 h = uint4(t, t ^ 0x6A09E667u, 0xBB67AE85u - t, 0x3C6EF372u);
        for (uint b = lo + t * 16u; b < hi; b += 4096u) {
            h = mix(h ^ card.Load4(b));
        }
        lanes[t] = mix(h);
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint n = 128u; n > 0u; n >>= 1u) {
            if (t < n) {
                lanes[t] ^= lanes[t + n];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        uint4 sum = lanes[0];
        if (mode == 0u) {
            if (t == 0u) records.Store4(at + 16u, sum);
            return;
        }
        if (t == 0u) {
            changed = any(sum != records.Load4(at + 16u)) ? 1u : 0u;
            records.Store(at + 8u, changed);
        }
        GroupMemoryBarrierWithGroupSync();
        if (changed == 0u) return;
    }
    for (uint c = lo + t * 16u; c < hi; c += 4096u) {
        after.Store4(c, card.Load4(c));
    }
}
"#;

/// The ways of [`CARD_PASS`].
const HASH: u32 = 0;
const COMPARE: u32 = 1;
const FETCH: u32 = 2;
const MARK: u32 = 3;

/// What the card's passes need ([`CARD_PASS`]), and the lists of ours they
/// are recorded into, by slot: a submission's plan takes a free slot.
struct CardWork {
    device: ID3D12Device,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// Two words per slot (system memory, mapped): the ticket of the plan
    /// whose first list, and whose last, the card ended last.
    marks: (ID3D12Resource, *mut u32),
    slots: Vec<Slot>,
    /// The last ticket given.
    ticket: u32,
}

/// The lists and records of one plan at a time.
struct Slot {
    /// The first list and the last (with their allocators).
    lists: [(ID3D12CommandAllocator, ID3D12GraphicsCommandList); 2],
    /// Its records (buffer, mapping, how many it holds).
    records: Option<(ID3D12Resource, *mut u8, usize)>,
    /// Its last plan's ticket, and which of its lists that plan ran.
    ticket: u32,
    ran: [bool; 2],
    /// Copies whose part of the last plan is not done with yet.
    parts: usize,
    /// The copies its last plan's lists name, kept until they have run.
    keep: Vec<Arc<Shadow>>,
}

// SAFETY: D3D12 objects are free-threaded; the mappings are of buffers it
// holds, used under its lock (and by the parts its records serve).
unsafe impl Send for CardWork {}

static WORK: std::sync::OnceLock<Option<Mutex<CardWork>>> = std::sync::OnceLock::new();

/// The card's passes, made at the first copy in the card's memory; `None`
/// if they cannot be (every copy then stays in system memory).
fn work() -> Option<&'static Mutex<CardWork>> {
    WORK.get_or_init(|| {
        let made = host()
            .ok_or_else(windows::core::Error::empty)
            .and_then(CardWork::new);
        match made {
            Ok(work) => Some(Mutex::new(work)),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "copies of big title buffers stay in system memory: the card's passes could \
                     not be made"
                );
                None
            }
        }
    })
    .as_ref()
}

impl CardWork {
    fn new(host: &crate::host::Host) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let device = host.device.clone();
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers
        // ours.
        let compiled = unsafe {
            D3DCompile(
                CARD_PASS.as_ptr().cast(),
                CARD_PASS.len(),
                windows::core::s!("card passes"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            let text = errors.map_or_else(String::new, |errors| {
                // SAFETY: the blob's own bytes.
                let text = unsafe {
                    std::slice::from_raw_parts(
                        errors.GetBufferPointer().cast::<u8>(),
                        errors.GetBufferSize(),
                    )
                };
                String::from_utf8_lossy(text).into_owned()
            });
            let what = format!("the card's passes did not compile: {text}");
            return Err(windows::core::Error::new(e.code(), what));
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let (marks, marks_data) = host
            .shared_buffer((MARKS * 4) as u64)
            .ok_or_else(windows::core::Error::empty)?;
        crate::host::name(&marks, || "marks of the card's passes".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                device,
                root,
                pso,
                marks: (marks, marks_data.cast()),
                slots: Vec::new(),
                ticket: 0,
            })
        }
    }

    /// A part of slot `slot`'s plan is done with.
    fn done(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.parts = s.parts.saturating_sub(1);
        }
    }

    /// Whether slot `i` is free: the card has ended its last plan's lists,
    /// and that plan's parts are done with.
    fn free(&self, i: usize) -> bool {
        let s = &self.slots[i];
        // SAFETY: words inside the marks' mapping, which the card writes.
        let mark = |word: usize| unsafe { self.marks.1.add(word).read_volatile() };
        s.parts == 0 && (0..2).all(|l| !s.ran[l] || mark(2 * i + l) == s.ticket)
    }

    /// Lets go of the copies that free slots still hold.
    fn let_go(&mut self) {
        for i in 0..self.slots.len() {
            if self.free(i) {
                self.slots[i].keep.clear();
            }
        }
    }

    /// The lists of `submission`'s plan, `jobs` ([`cards_around`]). A
    /// failure to record them ends the run: the copies' card sides would
    /// hold what the GPU did not write, or miss what the title did.
    fn plan(
        &mut self,
        submission: u64,
        jobs: Vec<Job>,
    ) -> (Option<ID3D12CommandList>, Option<ID3D12CommandList>) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let planned = self.record(submission, jobs);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::CardPlans, 1);
            add(C::CardPlanUs, started.elapsed().as_micros() as u64);
        }
        planned.unwrap_or_else(|e| {
            crate::host::fail(format_args!(
                "the card's passes over title buffer copies could not be recorded ({e})"
            ))
        })
    }

    fn record(
        &mut self,
        submission: u64,
        mut jobs: Vec<Job>,
    ) -> windows::core::Result<(Option<ID3D12CommandList>, Option<ID3D12CommandList>)> {
        let mut records = 0u32;
        for job in &mut jobs {
            job.first_span = records;
            records += job.spans.len() as u32;
            job.first_fetch = records;
            records += job.fetches.len() as u32;
        }
        let i = self.slot(records as usize)?;
        self.ticket = self.ticket.wrapping_add(1).max(1);
        let ticket = self.ticket;
        let (buffer, data, _) = self.slots[i]
            .records
            .clone()
            .ok_or_else(windows::core::Error::empty)?;
        for job in &jobs {
            for (k, &(a, b)) in job.spans.iter().chain(&job.fetches).enumerate() {
                let at = (job.first_span as usize + k) * RECORD;
                // SAFETY: inside the slot's records, which no list of ours
                // uses now (the slot is free).
                unsafe {
                    data.add(at)
                        .cast::<[u32; 4]>()
                        .write_unaligned([a as u32, b as u32, 0, 0]);
                }
            }
        }
        // SAFETY: a plain query of a live buffer.
        let records_at = unsafe { buffer.GetGPUVirtualAddress() };
        let first = jobs
            .iter()
            .any(|j| !j.uploads.is_empty() || !j.spans.is_empty());
        let last = jobs
            .iter()
            .any(|j| !j.spans.is_empty() || !j.fetches.is_empty());
        let slot = &self.slots[i];
        let mut up = None;
        let mut back = None;
        // SAFETY: the slot's lists are not running (it is free); live
        // buffers (kept until the lists have run), ranges inside them.
        unsafe {
            if first {
                up = Some(self.record_first(&slot.lists[0], &jobs, records_at, 2 * i, ticket)?);
            }
            if last {
                back =
                    Some(self.record_last(&slot.lists[1], &jobs, records_at, 2 * i + 1, ticket)?);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            let sum = |f: fn(&Job) -> &Vec<(usize, usize)>| -> (u64, u64) {
                jobs.iter().flat_map(f).fold((0, 0), |(n, bytes), &(a, b)| {
                    (n + 1, bytes + (b - a) as u64)
                })
            };
            let (ups, up_bytes) = sum(|j| &j.uploads);
            add(C::CardUps, ups);
            add(C::CardUpBytes, up_bytes);
            add(C::CardSpans, sum(|j| &j.spans).0);
            add(C::CardFetched, sum(|j| &j.fetches).1);
        }
        let mut parts = 0;
        for job in &jobs {
            let Some(card) = &job.shadow.card else {
                continue;
            };
            if job.spans.is_empty() && job.fetches.is_empty() {
                continue;
            }
            parts += 1;
            let spans = job
                .spans
                .iter()
                .zip(job.first_span..)
                .map(|(&(a, b), k)| (a, b, k))
                .collect();
            lock(&card.parts).push(Part {
                submission,
                slot: i,
                records: data.cast_const(),
                spans,
            });
        }
        let slot = &mut self.slots[i];
        slot.ticket = ticket;
        slot.ran = [up.is_some(), back.is_some()];
        slot.parts = parts;
        slot.keep = jobs.into_iter().map(|j| j.shadow).collect();
        Ok((
            up.and_then(|l| l.cast().ok()),
            back.and_then(|l| l.cast().ok()),
        ))
    }

    /// A free slot for a plan of `records` records: one whose last plan's
    /// lists the card has ended and whose parts are done with, else a new
    /// one.
    fn slot(&mut self, records: usize) -> windows::core::Result<usize> {
        let i = match (0..self.slots.len()).find(|&i| self.free(i)) {
            Some(i) => i,
            None => {
                if 2 * (self.slots.len() + 1) > MARKS {
                    return Err(windows::core::Error::empty());
                }
                let lists = [self.list()?, self.list()?];
                self.slots.push(Slot {
                    lists,
                    records: None,
                    ticket: 0,
                    ran: [false; 2],
                    parts: 0,
                    keep: Vec::new(),
                });
                self.slots.len() - 1
            }
        };
        let slot = &mut self.slots[i];
        // Its lists have run: what they named may go.
        slot.keep.clear();
        if slot.records.as_ref().is_none_or(|r| r.2 < records) {
            let n = records.max(2048).next_power_of_two();
            let (buffer, data) = host()
                .and_then(|h| h.shared_buffer((n * RECORD) as u64))
                .ok_or_else(windows::core::Error::empty)?;
            crate::host::name(&buffer, || "records of the card's passes".into());
            slot.records = Some((buffer, data, n));
        }
        Ok(i)
    }

    /// A closed list of ours, with its allocator.
    fn list(&self) -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            list.Close()?;
            crate::host::name(&list, || {
                format!("xtp card passes {:#x}", list.as_raw() as usize)
            });
            Ok((allocator, list))
        }
    }

    /// The plan's first list: what the CPU sides took goes up (a copy per
    /// range), then the pieces are hashed. The card sides start in the
    /// common state (buffers decay to it between `ExecuteCommandLists`) and
    /// end in it, as the title's lists after it take them.
    ///
    /// # Safety
    ///
    /// The list is not running; the jobs' buffers stay alive until it has
    /// run; `records` holds the jobs' records.
    unsafe fn record_first(
        &self,
        (allocator, list): &(ID3D12CommandAllocator, ID3D12GraphicsCommandList),
        jobs: &[Job],
        records: u64,
        mark: usize,
        ticket: u32,
    ) -> windows::core::Result<ID3D12GraphicsCommandList> {
        let (common, dest, uav) = (
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        // SAFETY: as the caller vouches.
        unsafe {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            let mut copied = Vec::new();
            for job in jobs.iter().filter(|j| !j.uploads.is_empty()) {
                let Some(card) = &job.shadow.card else {
                    continue;
                };
                for &(a, b) in &job.uploads {
                    let (at, n) = (a as u64, (b - a) as u64);
                    list.CopyBufferRegion(&job.shadow.buffer, at, &card.staging, at, n);
                }
                let then = if job.spans.is_empty() { common } else { uav };
                copied.push(transition(&job.shadow.buffer, dest, then));
            }
            if !copied.is_empty() {
                list.ResourceBarrier(&copied);
            }
            self.bind(list, records);
            let hashed: Vec<&Job> = jobs.iter().filter(|j| !j.spans.is_empty()).collect();
            for job in &hashed {
                self.pass(list, HASH, job, job.first_span, job.spans.len());
            }
            if !hashed.is_empty() {
                let back: Vec<D3D12_RESOURCE_BARRIER> = hashed
                    .iter()
                    .map(|j| transition(&j.shadow.buffer, uav, common))
                    .collect();
                list.ResourceBarrier(&back);
            }
            self.mark(list, mark, ticket);
            list.Close()?;
        }
        Ok(list.clone())
    }

    /// The plan's last list, after the submission's: its shaders' writes
    /// done first, the pieces compared, those that changed copied out, and
    /// so are the destinations of its GPU copies.
    ///
    /// # Safety
    ///
    /// As [`CardWork::record_first`].
    unsafe fn record_last(
        &self,
        (allocator, list): &(ID3D12CommandAllocator, ID3D12GraphicsCommandList),
        jobs: &[Job],
        records: u64,
        mark: usize,
        ticket: u32,
    ) -> windows::core::Result<ID3D12GraphicsCommandList> {
        // SAFETY: as the caller vouches.
        unsafe {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            list.ResourceBarrier(&[uav_barrier()]);
            self.bind(list, records);
            for job in jobs {
                if !job.spans.is_empty() {
                    self.pass(list, COMPARE, job, job.first_span, job.spans.len());
                }
                if !job.fetches.is_empty() {
                    self.pass(list, FETCH, job, job.first_fetch, job.fetches.len());
                }
            }
            self.mark(list, mark, ticket);
            list.Close()?;
        }
        Ok(list.clone())
    }

    /// Binds the passes and their records.
    ///
    /// # Safety
    ///
    /// An open list; `records` a live buffer's address.
    unsafe fn bind(&self, list: &ID3D12GraphicsCommandList, records: u64) {
        // SAFETY: as the caller vouches; our own root signature.
        unsafe {
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRootUnorderedAccessView(3, records);
        }
    }

    /// A pass of way `way` over `count` records from `first`, one group per
    /// record, on `job`'s copy: its card side bound whole, and its `after`
    /// (the card side again for a hash, which writes none).
    ///
    /// # Safety
    ///
    /// The passes bound ([`CardWork::bind`]); the records lie inside the
    /// copy.
    unsafe fn pass(
        &self,
        list: &ID3D12GraphicsCommandList,
        way: u32,
        job: &Job,
        first: u32,
        count: usize,
    ) {
        let Some(card) = &job.shadow.card else {
            return;
        };
        let size = card.bytes as u32;
        let after = if way == HASH {
            &job.shadow.buffer
        } else {
            &card.after
        };
        // SAFETY: as the caller vouches.
        unsafe {
            list.SetComputeRootUnorderedAccessView(1, job.shadow.buffer.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, after.GetGPUVirtualAddress());
            let (mut first, mut left) = (first, count);
            while left > 0 {
                let n = left.min(65535);
                let constants = [way, first, size];
                list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
                list.Dispatch(n as u32, 1, 1);
                first += n as u32;
                left -= n;
            }
        }
    }

    /// Ends a list with its mark: the card sets word `word` of the marks to
    /// `ticket` once the list's work before it is done.
    ///
    /// # Safety
    ///
    /// The passes bound ([`CardWork::bind`]).
    unsafe fn mark(&self, list: &ID3D12GraphicsCommandList, word: usize, ticket: u32) {
        // SAFETY: as the caller vouches; the marks are ours, the word inside
        // them.
        unsafe {
            list.ResourceBarrier(&[uav_barrier()]);
            let marks = self.marks.0.GetGPUVirtualAddress();
            list.SetComputeRootUnorderedAccessView(1, marks);
            list.SetComputeRootUnorderedAccessView(2, marks);
            let constants = [MARK, word as u32, ticket];
            list.SetComputeRoot32BitConstants(0, 3, constants.as_ptr().cast(), 0);
            list.Dispatch(1, 1, 1);
        }
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let (key, entry_end) = {
        let map = buffers();
        let found = map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end);
        match found {
            Some((
                _,
                Entry {
                    shadow: Some(s), ..
                },
            )) => return Some(s.clone()),
            Some((&key, e)) => (key, e.end),
            None => {
                drop(map);
                return loose(address, end);
            }
        }
    };
    // Made without the map's lock held: making a buffer may wait (for a
    // heap, or VidMm), and every recording thread looks copies up here.
    let len = entry_end - key.0;
    // Rounded up so that views ending in the buffer's last 256 bytes fit.
    // Shaders write some buffers (unordered access), so not upload memory.
    let size = len.next_multiple_of(1 << 16);
    let host = host()?;
    let (staging, data) = host.shared_buffer(size)?;
    crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
    let (buffer, card) = match card_for(host, size, &staging) {
        Some((buffer, card)) => {
            crate::host::name(&buffer, || {
                format!("title buffer {:#x}+{len:#x} on the card", key.0)
            });
            (buffer, Some(card))
        }
        None => (staging, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let made = Arc::new(Shadow::new(buffer, data, gpu, key.0, len, card));
    let mut map = buffers();
    match map.get_mut(&key) {
        // Another thread made one meanwhile: that one (ours goes).
        Some(Entry {
            shadow: Some(s), ..
        }) => return Some(s.clone()),
        Some(entry) if entry.end == entry_end => entry.shadow = Some(made.clone()),
        // Released (or made again) meanwhile: ours serves this use only.
        _ => {
            drop(map);
            carded(&made);
            return Some(made);
        }
    }
    drop(map);
    carded(&made);
    if len >= 16 << 20 {
        tracing::info!(
            start = format_args!("{:#x}", key.0),
            len,
            card = made.carded(),
            "large buffer copied to the host"
        );
    } else {
        tracing::debug!(
            start = format_args!("{:#x}", key.0),
            len,
            "buffer copied to the host"
        );
    }
    Some(made)
}

/// The title buffers holding `address`, innermost first (the one [`find`]
/// copies), as (start, length); none for memory outside every buffer,
/// which [`find`] copies in windows. For logs: it makes no copy.
pub(crate) fn holders(address: u64) -> Vec<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .filter(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
        .collect()
}

/// The copy [`find`] gives `[address, address + len)`, if it is made
/// already.
pub(crate) fn existing(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let map = buffers();
    match map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
    {
        Some((_, entry)) => entry.shadow.clone(),
        None => {
            drop(map);
            let start = address & !(WINDOW - 1);
            let len = end.next_multiple_of(WINDOW) - start;
            let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
            map.get(&(start, len)).cloned()
        }
    }
}

/// A buffer copy of `len` bytes from title address `src` to `dst` in
/// `list`. When either side has a host copy, shaders may have written the
/// source earlier in the submission or read the destination later, so the
/// copy runs on the GPU between host copies (made for the other side if
/// need be) at its place in the list, and the destination goes back to
/// title memory once the list has run. Returns false, recording nothing,
/// when neither side has a copy: title memory is then all there is, and
/// the caller copies it on the CPU at submission.
pub(crate) fn copy(list: &HostList, dst: u64, src: u64, len: u64) -> bool {
    if len == 0 {
        return true;
    }
    let (from, to) = (existing(src, len), existing(dst, len));
    if from.is_none() && to.is_none() {
        return false;
    }
    let (Some(from), Some(to), Some(host)) = (
        from.or_else(|| find(src, len)),
        to.or_else(|| find(dst, len)),
        host(),
    ) else {
        return false;
    };
    let (at, from_at) = (to.offset(dst), from.offset(src));
    let same = from.buffer.as_raw() == to.buffer.as_raw();
    // Overlapping ranges of one buffer go through a buffer of the list's
    // own, as if the source were read whole first.
    let through = if same && from_at < at + len && at < from_at + len {
        let Some((through, _)) = host.shared_buffer(len) else {
            return false;
        };
        list.keep(&through);
        Some(through)
    } else {
        None
    };
    // The source as the title or earlier work of the submission left it.
    from.refresh(list, src, len);
    to.gpu_copies(list, dst, len);
    #[cfg(feature = "diagnostics")]
    list.note_read(
        || format!("copy {len:#x} bytes from {src:#x} to {dst:#x} on the GPU"),
        (from.gpu(src), len),
    );
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live buffers (the steps above keep both
    // copies until the list is reset); ranges inside both.
    unsafe {
        if let Some(through) = &through {
            let b = &from.buffer;
            list.list.ResourceBarrier(&[
                uav_barrier(),
                transition(b, common, source),
                transition(through, common, dest),
            ]);
            list.list.CopyBufferRegion(through, 0, b, from_at, len);
            list.list.ResourceBarrier(&[
                transition(b, source, dest),
                transition(through, dest, source),
            ]);
            list.list.CopyBufferRegion(b, at, through, 0, len);
            list.list.ResourceBarrier(&[transition(b, dest, common)]);
        } else if same {
            // One buffer is the source and the destination: copied within
            // it as a destination.
            let b = &to.buffer;
            list.list
                .ResourceBarrier(&[uav_barrier(), transition(b, common, dest)]);
            list.list.CopyBufferRegion(b, at, b, from_at, len);
            list.list.ResourceBarrier(&[transition(b, dest, common)]);
        } else {
            let (a, b) = (&from.buffer, &to.buffer);
            list.list.ResourceBarrier(&[
                uav_barrier(),
                transition(a, common, source),
                transition(b, common, dest),
            ]);
            list.list.CopyBufferRegion(b, at, a, from_at, len);
            list.list
                .ResourceBarrier(&[transition(a, source, common), transition(b, dest, common)]);
        }
    }
    true
}

/// A memory write in `list` of the low `bytes` (4 or 8) of `value` at
/// title address `address`: if the address has a host copy, also written
/// there on the GPU at this place in the list, so that later commands of
/// the list read it (title memory gets it once the list has run, as
/// before). Returns whether it was.
///
/// shortcut: only where a copy exists already (a word nothing on the GPU
/// read yet does not get one), and only values: timestamps are known when
/// the list has run.
pub(crate) fn write(list: &HostList, address: u64, value: u64, bytes: usize) -> bool {
    if !matches!(bytes, 4 | 8) || !address.is_multiple_of(4) {
        return false;
    }
    let Some(shadow) = existing(address, bytes as u64) else {
        return false;
    };
    let Ok(list2) = list.list.cast::<ID3D12GraphicsCommandList2>() else {
        return false;
    };
    let gpu = shadow.gpu(address);
    let words = [value as u32, (value >> 32) as u32];
    let params: Vec<D3D12_WRITEBUFFERIMMEDIATE_PARAMETER> = words[..bytes / 4]
        .iter()
        .zip(0..)
        .map(|(&word, i)| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: gpu + 4 * i,
            Value: word,
        })
        .collect();
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("write {value:#x} ({bytes} bytes) at {address:#x} on the GPU"));
    list.keep(&shadow.buffer);
    list.count_gpu_write(&shadow, address as usize, bytes);
    let b = &shadow.buffer;
    // SAFETY: an open list, a live buffer (kept until the list is reset)
    // and destinations inside it.
    unsafe {
        list2.ResourceBarrier(&[
            uav_barrier(),
            transition(
                b,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
        ]);
        list2.WriteBufferImmediate(params.len() as u32, params.as_ptr(), None);
        list2.ResourceBarrier(&[transition(
            b,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    true
}

/// Titles also use memory outside any buffer object as GPU memory (BTP 6's
/// interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    if let Some(shadow) = LOOSE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&(start, len))
    {
        return Some(shadow.clone());
    }
    // Made without the lock held, as in `find`.
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let made = Arc::new(Shadow::new(buffer, data, gpu, start, len, None));
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    // The first one made stays.
    let shadow = map.entry((start, len)).or_insert(made).clone();
    drop(map);
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Title memory as the runtime reserves it, write-watched and touched:
    /// `len` bytes at the start of a 4 MiB window of their own (one loose
    /// host copy, [`find`]). Never freed: the copy outlives the test.
    pub(super) fn watched(len: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let window = WINDOW as usize;
        assert!(len <= window);
        // SAFETY: a fresh reservation of our own.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                2 * window,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0, "write-watched memory");
        let at = base.next_multiple_of(window);
        // SAFETY: inside the reservation.
        unsafe { (at as *mut u8).write_bytes(0, window) };
        at
    }

    /// Title memory `[at, at + len)`.
    pub(super) fn title(at: usize, len: usize) -> Vec<u8> {
        // SAFETY: the test's memory.
        (0..len)
            .map(|i| unsafe { ((at + i) as *const u8).read_volatile() })
            .collect()
    }

    /// Writes title memory as the title's CPU does.
    pub(super) fn set_title(at: usize, len: usize, byte: u8) {
        for i in 0..len {
            // SAFETY: the test's memory.
            unsafe { ((at + i) as *mut u8).write_volatile(byte) };
        }
    }

    /// A title write while a copy-back runs, given the piece's address.
    type TitleWrite = Box<dyn FnMut(usize)>;

    thread_local! {
        /// What [`after_reading`] does on this thread.
        static AFTER_READING: std::cell::RefCell<Option<TitleWrite>> =
            const { std::cell::RefCell::new(None) };
        /// The plain reads [`read_pieces`] made on this thread.
        pub(super) static READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// The write-watch askings refreshes made on this thread.
        pub(super) static ASKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// How far short of the faulting byte the next fault at the copy's
        /// instruction on this thread stops it ([`stopped`]).
        pub(super) static LAG: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// The census's mode for this process ([`census::mode`]) plus one, 0
    /// to leave it to `XTP_COPYBACK_CENSUS`: set by the census tests, each
    /// in a process of its own ([`alone`]).
    pub(super) static CENSUS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    /// The census's counters since the last call.
    fn counted() -> [u64; 15] {
        census::take()
    }

    fn at(c: &[u64; 15], k: census::K) -> u64 {
        c[k as usize]
    }

    /// Runs `body` in a process of its own: the test binary again, with
    /// this test alone. Guarded blocks and the order of the vectored
    /// handlers are the process's, and the guard's own tests count the
    /// guarded blocks (they hold a lock of theirs for it).
    fn alone(test: &str, body: impl FnOnce()) {
        const CHILD: &str = "XTP_SHADOW_TEST_ALONE";
        if std::env::var_os(CHILD).is_some() {
            body();
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{test} in a process of its own: {stdout}\n{stderr}"
        );
    }

    /// A view of a pagefile section of `len` bytes, mapped read-write, as
    /// the runtime maps physical pages: memory the guard watches. Never
    /// unmapped: copies made of it outlive the test.
    fn view(len: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };
        // SAFETY: a fresh section, mapped whole; the view keeps it.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null(), "a section");
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            windows_sys::Win32::Foundation::CloseHandle(section);
            assert_ne!(view, 0, "a view");
            view
        }
    }

    /// Whether the kernel's write watch has a page of `[at, at + PAGE)`
    /// written (asked without clearing).
    fn kernel_marked(at: usize) -> bool {
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        let mut found = [0usize; PAGE / KERNEL_PAGE];
        let (mut count, mut granularity) = (found.len(), 0);
        // SAFETY: room for `count` addresses; no reset.
        let failed = unsafe {
            GetWriteWatch(
                0,
                at as *const _,
                PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        assert!(!failed, "write-watched memory");
        count > 0
    }

    /// Makes `[at, at + len)` of title memory unmapped (decommitted).
    fn unmap(at: usize, len: usize) {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        // SAFETY: the test's own memory, which nothing else uses.
        assert_ne!(unsafe { VirtualFree(at as *mut _, len, MEM_DECOMMIT) }, 0);
    }

    /// Called by [`Shadow::copy_back`] between its read of a piece of title
    /// memory at `at` and its write.
    pub(super) fn after_reading(at: usize) {
        AFTER_READING.with_borrow_mut(|f| {
            if let Some(f) = f {
                f(at);
            }
        });
    }

    impl Shadow {
        /// The copy's bytes at title addresses `[at, at + len)`.
        pub(super) fn host_bytes(&self, at: usize, len: usize) -> Vec<u8> {
            let from = at - self.start as usize;
            assert!(from + len <= self.len as usize);
            // SAFETY: inside the mapping.
            unsafe { std::slice::from_raw_parts(self.data.add(from), len) }.to_vec()
        }

        /// The snapshot mirrors the copy holds, for submissions in flight
        /// or kept for the next.
        pub(crate) fn snapshots(&self) -> usize {
            let snapshots = self.before.lock().unwrap();
            snapshots.live.len() + snapshots.spare.len()
        }

        /// Writes the copy at title addresses `[at, at + len)`, as the GPU
        /// does.
        pub(super) fn set_host(&self, at: usize, len: usize, byte: u8) {
            let from = at - self.start as usize;
            assert!(from + len <= self.len as usize);
            // SAFETY: inside the mapping; tests stand in for the GPU.
            unsafe { self.data.add(from).write_bytes(byte, len) };
        }
    }

    /// `XTP_FULL_REFRESH=1` (a diagnostic): before a submission, a copy the
    /// write tracking left stale is taken whole from title memory again,
    /// but where GPU work in flight writes it. In a process of its own,
    /// where the switch is read afresh.
    #[test]
    fn a_full_refresh_takes_every_copy_whole_but_what_gpu_work_writes() {
        const CHILD: &str = "XTP_SHADOW_TEST_FULL_REFRESH";
        let test = "shadow::tests::a_full_refresh_takes_every_copy_whole_but_what_gpu_work_writes";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if std::env::var_os(CHILD).is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env("XTP_FULL_REFRESH", "1")
                .output()
                .unwrap();
            let (stdout, stderr) = (
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
            );
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "{test} in a process of its own: {stdout}\n{stderr}"
            );
            return;
        }
        assert_eq!(full_refresh(), FullRefresh::Every(1));
        let window = watched(4 * PAGE);
        let shadow = find(window as u64, (4 * PAGE) as u64).expect("a host copy");
        set_title(window, 4 * PAGE, 0x11);
        shadow.update(window, 4 * PAGE);
        // Stale, and the tracking does not know: the copy holds other bytes
        // than title memory.
        shadow.set_host(window, 4 * PAGE, 0x22);
        assert!(!shadow.stale(window, window + 4 * PAGE));
        // The four pieces count as writes the tracking missed; the busy
        // page is left out by the caller, not by the count.
        assert_eq!(shadow.missed(window, window + 4 * PAGE), 4 * PAGE);
        // GPU work in flight writes the second page.
        let writing = submission();
        shadow.writing(writing, window + PAGE, PAGE);
        let next = submission();
        assert!(refresh_due(next));
        refresh_everything(next);
        let lists = cards_around(next);
        assert_eq!(
            shadow.missed(window, window + PAGE),
            0,
            "taken: nothing missed now"
        );
        assert!(
            lists.0.is_none() && lists.1.is_none(),
            "no copy on the card"
        );
        let all =
            |at: usize, len: usize, byte: u8| shadow.host_bytes(at, len).iter().all(|&b| b == byte);
        assert!(all(window, PAGE, 0x11), "taken again");
        assert!(
            all(window + PAGE, PAGE, 0x22),
            "what the GPU writes, left alone"
        );
        assert!(all(window + 2 * PAGE, 2 * PAGE, 0x11), "taken again");
        assert_eq!(shadow.copy_back(writing, window + PAGE, PAGE), 0);

        // A GPU word has no shader/copy Busy entry. The independent
        // submission's whole-memory sweep must consume the earlier label
        // and its readers before it touches their shadow.
        let [first, independent] = std::array::from_fn(|_| host.new_list().unwrap());
        assert!(write(&first, window as u64, 0xdead_beef, 4));
        first.defer_write(window as _, crate::queue::Written::Value(0xdead_beef), 4);
        first.close();
        independent.close();
        let mut gate = crate::host::submissions::Gate::close(host);
        host.execute(&[&first]);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(80));
                gate.open();
            });
            host.execute(&[&independent]);
            assert_eq!(title(window, 4), 0xdead_beefu32.to_le_bytes());
        });
        host.drain();
        assert_eq!(shadow.host_bytes(window, 4), 0xdead_beefu32.to_le_bytes());
    }

    #[test]
    fn learned_copy_back_keeps_the_completed_submission_until_first_touch() {
        let test =
            "shadow::tests::learned_copy_back_keeps_the_completed_submission_until_first_touch";
        alone(test, || {
            let Some(gpu) = crate::testgpu::device(test) else {
                return;
            };
            let host = gpu.host();
            const REOPEN: &str = "XTP_TEST_REOPEN_READMAP";
            let local = tempfile::tempdir().unwrap();
            let previous = std::env::var_os(REOPEN);
            let path = previous
                .as_ref()
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| local.path().join("one-build.v1"));
            crate::configure_copy_back(Some(path.clone()));
            crate::watch::tests::with_physical_views(|at, alias, private| {
                register(1, at as u64, (4 * PAGE) as u64);
                let s = find(at as u64, (4 * PAGE) as u64).unwrap();
                // The fourth page is a write-only GPU destination. It never
                // went through an ordinary refresh or its write guard.
                s.update(at, 3 * PAGE);
                let publish = |address, byte| {
                    let id = submission();
                    s.writing_by(id, address, PAGE, 123);
                    s.set_host(address + 7, 1, byte);
                    s.copy_back(id, address, PAGE)
                };
                if previous.is_some() {
                    // A new process, with fresh virtual addresses and no live
                    // owners. The same writer/offset is eager on its first use.
                    assert_eq!(publish(at, 0x51), PAGE);
                    assert_eq!(title(private + 7, 1), [0x51]);
                    assert_eq!(publish(at + 3 * PAGE, 4), 0);
                    assert_eq!(title(private + 3 * PAGE + 7, 1), [0]);
                    host.drain();
                    unregister(1, at as u64);
                    return;
                }
                assert_eq!(publish(at, 0x51), 0);
                // Reuse the D3D buffer for B before the CPU consumes A.
                let b = submission();
                s.writing_by(b, at, PAGE, 123);
                s.set_host(at + 7, 1, 0x72);
                assert_eq!(title(private + 7, 1), [0]);
                {
                    let gate = std::sync::Barrier::new(2);
                    let read = |address: usize| {
                        gate.wait();
                        // SAFETY: the test holds this CPU view across the read.
                        unsafe { (address as *const u8).read_volatile() }
                    };
                    std::thread::scope(|readers| {
                        let a = readers.spawn(|| read(alias + 7));
                        assert_eq!(read(at + 7), 0x51);
                        assert_eq!(a.join().unwrap(), 0x51);
                    });
                }
                assert_eq!(s.copy_back(b, at, PAGE), PAGE);
                assert_eq!(title(private + 7, 1), [0x72]);
                crate::save_copy_back_map();

                // Identical allocation with a different page offset stays
                // cold. A kernel read must resolve it just like a CPU load.
                assert_eq!(publish(at + PAGE, 0xa1), 0);
                let mut byte = [0; 1];
                assert!(crate::host::read_into((alias + PAGE + 7) as _, &mut byte));
                assert_eq!(byte, [0xa1]);
                // Une lecture interne ne doit pas rendre le prochain résultat immédiat.
                assert_eq!(publish(at + PAGE, 0xa2), 0);
                assert_eq!(publish(at + 2 * PAGE, 0xb1), 0);
                // A DirectStorage destination is pinned before the first
                // byte is written, including writes that restore old values.
                let physical = private + 2 * PAGE;
                {
                    let _pin = crate::pin_copy_back(physical, PAGE);
                    assert_eq!(title(physical + 7, 1), [0xb1]);
                    set_title(physical + 7, 1, 0);
                    crate::watch::writing(at + 2 * PAGE, PAGE);
                }
                assert_eq!(title(alias + 2 * PAGE + 7, 1), [0]);
                // Replacing an untouched result updates the pending data
                // without publishing either result to the CPU's backing.
                assert_eq!(publish(at + 3 * PAGE, 3), 0);
                assert_eq!(publish(at + 3 * PAGE, 4), 0);
                assert_eq!(title(private + 3 * PAGE + 7, 1), [0]);
                assert_eq!(title(alias + 3 * PAGE + 7, 1), [4]);
                host.drain();
                unregister(1, at as u64);
            });
            if previous.is_none() {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([test, "--exact", "--nocapture", "--test-threads=1"])
                    .env(REOPEN, path)
                    .output()
                    .unwrap();
                assert!(
                    out.status.success(),
                    "fresh process: {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        });
    }

    /// The card's passes compile, and their pipeline is made.
    #[test]
    fn the_cards_passes_are_made() {
        let test = "shadow::tests::the_cards_passes_are_made";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if let Err(e) = CardWork::new(host) {
            panic!("{e}");
        }
    }

    /// The ranges a plan of the card's passes takes are each byte once:
    /// those that overlap or touch are joined, as the host joins the ranges
    /// it copies back.
    #[test]
    fn a_cards_plan_takes_each_byte_once() {
        let ranges = vec![(10, 20), (0, 5), (5, 8), (15, 30), (40, 41), (30, 30)];
        assert_eq!(joined(ranges), [(0, 8), (10, 30), (40, 41)]);
        assert!(joined(Vec::new()).is_empty());
    }

    /// A submission's snapshot takes the bytes of each range it declares
    /// once, when first declared, and not the gaps between ranges. A range
    /// over what it took already takes only the rest: each byte is copied
    /// once (before, a range overlapping a piece merged both into a new
    /// buffer, copying the piece again).
    #[test]
    fn a_snapshot_keeps_what_it_took_first_and_skips_the_gaps() {
        let mut host = vec![1u8; 64];
        let mut before = Before {
            submission: 1,
            mirror: Mirror::new(64),
            taken: Vec::new(),
            partial: Vec::new(),
        };
        let cover = |before: &mut Before, host: &[u8], from: usize, to: usize| {
            before.cover(from, to, |a, b| &host[a..b])
        };
        assert_eq!(cover(&mut before, &host, 8, 16), 8);
        assert_eq!(cover(&mut before, &host, 32, 40), 8);
        host.fill(2);
        // Overlapping the first and touching the second: merged; what was
        // taken stays 1, the gap between them is taken now.
        assert_eq!(cover(&mut before, &host, 12, 32), 16, "the gap only");
        host.fill(3);
        assert_eq!(cover(&mut before, &host, 48, 50), 2);
        assert_eq!(cover(&mut before, &host, 10, 14), 0, "taken already");
        assert_eq!(cover(&mut before, &host, 8, 40), 0, "taken already");
        assert_eq!(before.taken, [(8, 40), (48, 50)]);
        let mut want = vec![1u8; 8];
        want.extend([2u8; 16]);
        want.extend([1u8; 8]);
        assert_eq!(before.bytes(8, 40).as_deref(), Some(&want[..]));
        assert_eq!(before.bytes(14, 18).as_deref(), Some(&[1, 1, 2, 2][..]));
        assert_eq!(before.bytes(40, 41), None, "past what it took");
        assert_eq!(before.bytes(47, 49), None);
        assert_eq!(before.bytes(48, 50).as_deref(), Some(&[3, 3][..]));
        // A range over both and the gap between them: one range.
        assert_eq!(cover(&mut before, &host, 30, 60), 18);
        assert_eq!(before.taken, [(8, 60)]);
        assert_eq!(
            before.bytes(39, 49).as_deref(),
            Some(&[1, 3, 3, 3, 3, 3, 3, 3, 3, 3][..])
        );
    }

    /// A copy's snapshot mirror outlives its submission: the next
    /// submission's snapshot takes it again (its pages touched already, no
    /// fresh memory), and two submissions in flight each have their own.
    #[test]
    fn snapshot_mirrors_are_kept_for_the_next_submissions() {
        let test = "shadow::tests::snapshot_mirrors_are_kept_for_the_next_submissions";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(2 * PAGE);
        let shadow = find(window as u64, (2 * PAGE) as u64).expect("a host copy");
        shadow.update(window, 2 * PAGE);
        let counts = |s: &Shadow| {
            let snapshots = s.before.lock().unwrap();
            (snapshots.live.len(), snapshots.spare.len())
        };
        let (a, b) = (submission(), submission());
        shadow.writing(a, window, PAGE);
        shadow.writing(b, window + PAGE, PAGE);
        assert_eq!(counts(&shadow), (2, 0), "one mirror each");
        let held = shadow.before.lock().unwrap().live[0].mirror.bytes.as_ptr();
        shadow.set_host(window + 0x10, 4, 0x61);
        assert_eq!(shadow.copy_back(a, window, PAGE), 4);
        assert_eq!(counts(&shadow), (1, 1), "a's mirror kept");
        let next = submission();
        shadow.writing(next, window, PAGE);
        assert_eq!(counts(&shadow), (2, 0));
        let snapshots = shadow.before.lock().unwrap();
        let taken = snapshots
            .live
            .iter()
            .find(|s| s.submission == next)
            .unwrap();
        assert_eq!(taken.mirror.bytes.as_ptr(), held, "the same memory");
        assert_eq!(&taken.mirror.bytes[0x10..0x14], &[0x61; 4], "today's bytes");
        drop(snapshots);
        assert_eq!(shadow.copy_back(b, window + PAGE, PAGE), 0);
        assert_eq!(shadow.copy_back(next, window, PAGE), 0);
        assert_eq!(counts(&shadow), (0, 2));
    }

    /// A refresh leaves alone what a GPU buffer copy in flight writes into
    /// the host copy, even where the title wrote since: the copy's bytes go
    /// to title memory when its submission finishes, and only then may a
    /// refresh take them back.
    #[test]
    fn a_refresh_leaves_what_a_gpu_copy_in_flight_writes() {
        let test = "shadow::tests::a_refresh_leaves_what_a_gpu_copy_in_flight_writes";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(PAGE);
        let shadow = find(window as u64, PAGE as u64).expect("a host copy");
        shadow.update(window, PAGE);
        let (dst, len) = (window + 0x100, 0x100);
        let copy = submission();
        shadow.copying(copy, dst, len);
        // The GPU copy ran; the title writes the page, the destination too.
        shadow.set_host(dst, len, 0xcc);
        set_title(window, PAGE, 0x22);
        shadow.update(window, PAGE);
        assert!(shadow.host_bytes(dst, len).iter().all(|&b| b == 0xcc));
        assert!(shadow.host_bytes(window, 0x100).iter().all(|&b| b == 0x22));
        assert!(
            shadow
                .host_bytes(dst + len, 0x100)
                .iter()
                .all(|&b| b == 0x22)
        );
        assert_eq!(shadow.write_back(copy, window, PAGE), PAGE);
        assert!(title(dst, len).iter().all(|&b| b == 0xcc));
        // Written back: refreshes take the range again.
        set_title(dst, len, 0x33);
        shadow.update(window, PAGE);
        assert!(shadow.host_bytes(dst, len).iter().all(|&b| b == 0x33));
    }

    /// The title writes a piece of title memory while it is copied back,
    /// between the copy-back's read and its write: its bytes stay (only
    /// the runs the GPU changed are written), and the page is not taken
    /// for current, so the next refresh brings the title's write.
    #[test]
    fn a_copy_back_keeps_title_writes_made_while_it_runs() {
        let test = "shadow::tests::a_copy_back_keeps_title_writes_made_while_it_runs";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(2 * PAGE);
        let shadow = find(window as u64, PAGE as u64).expect("a host copy");
        let page = window + PAGE;
        shadow.update(window, 2 * PAGE);
        assert!(!shadow.stale(page, page + PAGE));
        let gpu = submission();
        shadow.writing(gpu, page, PAGE);
        // The GPU writes 16 bytes; the title 16 others meanwhile.
        shadow.set_host(page, 16, 0xab);
        AFTER_READING.set(Some(Box::new(move |at| {
            if at == page {
                set_title(page + 0x100, 16, 0x5c);
            }
        })));
        let written = shadow.copy_back(gpu, page, PAGE);
        AFTER_READING.set(None);
        assert_eq!(written, 16, "only the GPU's bytes");
        assert!(title(page, 16).iter().all(|&b| b == 0xab));
        assert!(
            title(page + 0x100, 16).iter().all(|&b| b == 0x5c),
            "the title's write is kept"
        );
        assert!(
            shadow.stale(page, page + PAGE),
            "not current: the title wrote it"
        );
        shadow.update(page, PAGE);
        assert!(
            shadow
                .host_bytes(page + 0x100, 16)
                .iter()
                .all(|&b| b == 0x5c)
        );
        assert!(shadow.host_bytes(page, 16).iter().all(|&b| b == 0xab));
        // Without a title write meanwhile, the page is current after it.
        let gpu = submission();
        shadow.writing(gpu, page, PAGE);
        shadow.set_host(page + 0x200, 8, 0xcd);
        assert_eq!(shadow.copy_back(gpu, page, PAGE), 8);
        assert!(title(page + 0x200, 8).iter().all(|&b| b == 0xcd));
        assert!(
            !shadow.stale(page, page + PAGE),
            "current after our own write"
        );
    }

    /// W1: a range is copied once, then not again by the submissions after
    /// it (the rest of the frame) until the title writes it: neither its
    /// whole pages nor the pages it takes only part of (before W1 those
    /// were copied at every refresh). A write brings back the page written,
    /// and only it; a part of a page next to what the copy holds of it is
    /// copied, then held with it.
    #[test]
    fn a_range_is_copied_once_until_the_title_writes_it() {
        let test = "shadow::tests::a_range_is_copied_once_until_the_title_writes_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(3 * PAGE);
        let shadow = find(window as u64, (3 * PAGE) as u64).expect("a host copy");
        // From inside the first page to inside the third, as views are.
        let (from, len) = (window + 0x100, 2 * PAGE);
        set_title(window, 3 * PAGE, 0x11);
        assert_eq!(shadow.update(from, len), len, "copied first");
        assert_eq!(shadow.host_bytes(from, len), title(from, len));
        for submission in 0..3 {
            assert_eq!(
                shadow.update(from, len),
                0,
                "submission {submission} after it"
            );
        }
        assert_eq!(shadow.update(from + 0x200, 0x100), 0, "a part of it");
        assert!(!shadow.stale(from, from + len));
        // The title writes the third page: its part in the range is copied
        // again, the rest is not.
        set_title(window + 2 * PAGE + 0x10, 4, 0x22);
        assert!(shadow.stale(from, from + len));
        assert!(
            !shadow.stale(from, window + 2 * PAGE),
            "the pages not written"
        );
        assert_eq!(shadow.update(from, len), 0x100);
        assert_eq!(
            shadow.host_bytes(window + 2 * PAGE, 0x100),
            title(window + 2 * PAGE, 0x100)
        );
        assert_eq!(shadow.update(from, len), 0);
        // The rest of the first page, which touches what it holds.
        assert_eq!(shadow.update(window, 0x100), 0x100);
        assert_eq!(shadow.update(window, PAGE), 0, "the first page whole");
        assert_eq!(shadow.host_bytes(window, PAGE), title(window, PAGE));
        // A part of the third page apart from what it holds: copied, and
        // held instead.
        assert_eq!(shadow.update(window + 2 * PAGE + 0x400, 0x100), 0x100);
        assert_eq!(shadow.update(window + 2 * PAGE + 0x400, 0x100), 0);
        assert_eq!(shadow.update(window + 2 * PAGE, 0x100), 0x100);
    }

    /// A refresh around what shaders in flight write: one asking for the
    /// whole range (W1; before, one per gap around it), the pages written
    /// since copied but where the GPU work writes, which keeps the GPU's
    /// bytes; the copy-back puts them in title memory.
    #[test]
    fn a_refresh_leaves_what_shaders_in_flight_write() {
        let test = "shadow::tests::a_refresh_leaves_what_shaders_in_flight_write";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(3 * PAGE);
        let shadow = find(window as u64, (3 * PAGE) as u64).expect("a host copy");
        shadow.update(window, 3 * PAGE);
        let gpu = submission();
        shadow.writing(gpu, window + PAGE, PAGE);
        shadow.set_host(window + PAGE, PAGE, 0xc1);
        // The title writes the pages around it (where both write, the title's
        // bytes stay: `where_the_gpu_and_the_title_both_wrote_the_titles_stay`).
        set_title(window, PAGE, 0x33);
        set_title(window + 2 * PAGE, PAGE, 0x33);
        ASKED.set(0);
        assert_eq!(shadow.update(window, 3 * PAGE), 2 * PAGE);
        assert_eq!(
            ASKED.get(),
            if once() { 1 } else { 2 },
            "write-watch askings"
        );
        assert!(shadow.host_bytes(window, PAGE).iter().all(|&b| b == 0x33));
        assert!(
            shadow
                .host_bytes(window + PAGE, PAGE)
                .iter()
                .all(|&b| b == 0xc1)
        );
        assert!(
            shadow
                .host_bytes(window + 2 * PAGE, PAGE)
                .iter()
                .all(|&b| b == 0x33)
        );
        assert_eq!(shadow.copy_back(gpu, window + PAGE, PAGE), PAGE);
        assert!(title(window + PAGE, PAGE).iter().all(|&b| b == 0xc1));
        assert!(!shadow.stale(window, window + 3 * PAGE), "all current");
        assert_eq!(shadow.update(window, 3 * PAGE), 0);
    }

    /// Plain copies of title memory read what the kernel's copy read, with
    /// zeros where a page is decommitted or not readable, from any offset;
    /// a copy stops at the first byte it cannot read.
    #[test]
    fn plain_copies_read_as_the_kernel_did() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualProtect,
        };
        let len = 4 * PAGE;
        // SAFETY: a fresh reservation of our own, never freed.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        for i in 0..len {
            // SAFETY: the test's memory.
            unsafe { ((base + i) as *mut u8).write((i * 7 + 3) as u8) };
        }
        // The second page decommitted, the first 4 KiB of the third not
        // readable.
        unmap(base + PAGE, PAGE);
        let mut old = 0;
        // SAFETY: the test's memory.
        let protected =
            unsafe { VirtualProtect((base + 2 * PAGE) as *const _, 4096, PAGE_NOACCESS, &mut old) };
        assert_ne!(protected, 0);
        let (src, n) = (base + 0x40, len - 0x80);
        let (mut plain, mut kernel) = (vec![0xeeu8; n], vec![0xeeu8; n]);
        let zeros = read_title(plain.as_mut_ptr(), src, n);
        crate::host::copy_mapped(kernel.as_mut_ptr(), src as *const u8, n, false);
        assert!(plain == kernel, "as the kernel's copy");
        assert_eq!(zeros, PAGE + 4096);
        let mut upload = vec![0xee; n];
        // SAFETY: le vecteur couvre toute la destination, hors de la réservation.
        unsafe { fill_upload(upload.as_mut_ptr(), src, n) };
        assert!(
            upload == kernel,
            "le remplissage doit reprendre après les pages illisibles"
        );
        assert_eq!(&plain[..0x40], &title(src, 0x40)[..]);
        let mut buffer = vec![0u8; 2 * PAGE];
        // SAFETY: our buffer; title memory a fault leaves alone.
        let left = unsafe { try_copy(buffer.as_mut_ptr(), (base + 0x10) as *const u8, 2 * PAGE) };
        assert_eq!(left, PAGE + 0x10, "stopped at the decommitted page");
        assert!(!read_whole(base + PAGE - 0x10, &mut buffer[..0x20]));
        assert!(read_whole(base + 3 * PAGE, &mut buffer[..PAGE]));
        // Writes leave alone the pages they cannot write, and write what is
        // mapped past them.
        let ours = vec![0x5au8; 3 * PAGE];
        write_title(base + 0x20, ours.as_ptr(), 3 * PAGE);
        assert!(title(base + 0x20, PAGE - 0x20).iter().all(|&b| b == 0x5a));
        assert!(
            title(base + 2 * PAGE + 4096, PAGE - 4096)
                .iter()
                .all(|&b| b == 0x5a)
        );
    }

    /// The copy-back's scans by 8 bytes find what the byte loop found: the
    /// first byte that differs, and the first that does not.
    #[test]
    fn the_copy_backs_scans_find_what_the_byte_loop_found() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..200 {
            let len = (next() % 100) as usize;
            let a: Vec<u8> = (0..len).map(|_| (next() % 3) as u8).collect();
            let b: Vec<u8> = (0..len).map(|_| (next() % 3) as u8).collect();
            for at in 0..=len {
                let change = (at..len).find(|&i| a[i] != b[i]).unwrap_or(len);
                let same = (at..len).find(|&i| a[i] == b[i]).unwrap_or(len);
                assert_eq!(first_change(&a, &b, at), change, "{a:?} {b:?} from {at}");
                assert_eq!(first_same(&a, &b, at), same, "{a:?} {b:?} from {at}");
            }
        }
    }

    /// A copy-back reads a run of pieces the GPU changed with one copy and
    /// keeps the readable pieces of a run that crosses an unmapped page; a
    /// range shorter than a piece across a piece boundary puts back both
    /// sides (the fallback fix of Marvisak/VIctory#1124).
    #[test]
    fn a_copy_back_keeps_the_readable_pieces_of_a_run() {
        let test = "shadow::tests::a_copy_back_keeps_the_readable_pieces_of_a_run";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(4 * PAGE);
        let shadow = find(window as u64, (4 * PAGE) as u64).expect("a host copy");
        shadow.update(window, 4 * PAGE);
        let (short, long) = ((window + PAGE - 0x80, 0x100), (window + PAGE, 3 * PAGE));
        let gpu = submission();
        shadow.writing(gpu, short.0, short.1);
        shadow.writing(gpu, long.0, long.1);
        // The GPU writes across the first boundary, in the third page and
        // in the fourth; the title unmaps the third.
        shadow.set_host(window + PAGE - 0x40, 0x80, 0xa1);
        shadow.set_host(window + 2 * PAGE + 0x10, 0x10, 0xa2);
        shadow.set_host(window + 3 * PAGE + 0x20, 0x10, 0xa3);
        unmap(window + 2 * PAGE, PAGE);
        READS.set(0);
        assert_eq!(shadow.copy_back(gpu, short.0, short.1), 0x80);
        assert_eq!(READS.get(), 1, "both pieces with one copy");
        assert!(title(window + PAGE - 0x40, 0x80).iter().all(|&b| b == 0xa1));
        READS.set(0);
        // The second page has the GPU's bytes already: only the fourth's go.
        assert_eq!(shadow.copy_back(gpu, long.0, long.1), 0x10);
        assert_eq!(READS.get(), 2, "the run, then past the unmapped page");
        assert!(
            title(window + 3 * PAGE + 0x20, 0x10)
                .iter()
                .all(|&b| b == 0xa3)
        );
        assert!(!shadow.stale(window + PAGE, window + 2 * PAGE));
        assert!(!shadow.stale(window + 3 * PAGE, window + 4 * PAGE));
    }

    /// A run of pieces is read at once, then put back piece by piece: a
    /// title write into a later piece after the read keeps its bytes (only
    /// the GPU's are written; Marvisak/VIctory#1124 wrote whole pieces as
    /// read), and that page is not taken for current.
    #[test]
    fn a_copy_back_run_keeps_title_writes_made_after_its_read() {
        let test = "shadow::tests::a_copy_back_run_keeps_title_writes_made_after_its_read";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(2 * PAGE);
        let shadow = find(window as u64, (2 * PAGE) as u64).expect("a host copy");
        let second = window + PAGE;
        shadow.update(window, 2 * PAGE);
        let gpu = submission();
        shadow.writing(gpu, window, 2 * PAGE);
        shadow.set_host(window + 0x10, 8, 0xb1);
        shadow.set_host(second + 0x10, 8, 0xb2);
        READS.set(0);
        AFTER_READING.set(Some(Box::new(move |at| {
            if at == second {
                set_title(second + 0x100, 8, 0x5d);
            }
        })));
        let written = shadow.copy_back(gpu, window, 2 * PAGE);
        AFTER_READING.set(None);
        assert_eq!(written, 16, "only the GPU's bytes");
        assert_eq!(READS.get(), 1, "the run read at once");
        assert!(title(window + 0x10, 8).iter().all(|&b| b == 0xb1));
        assert!(title(second + 0x10, 8).iter().all(|&b| b == 0xb2));
        assert!(
            title(second + 0x100, 8).iter().all(|&b| b == 0x5d),
            "the title's write is kept"
        );
        assert!(!shadow.stale(window, second), "current after our write");
        assert!(
            shadow.stale(second, second + PAGE),
            "not current: the title wrote it"
        );
        shadow.update(window, 2 * PAGE);
        assert!(
            shadow
                .host_bytes(second + 0x100, 8)
                .iter()
                .all(|&b| b == 0x5d)
        );
    }

    /// The copy-back is a three-way merge (ported from simnJS/AnyPS5
    /// (65a1b01db4)): where the GPU and, since the snapshot, the title both
    /// changed bytes, the title's stay and the copy takes them; the GPU's
    /// bytes around them go back. Before, the GPU's went over the title's.
    #[test]
    fn where_the_gpu_and_the_title_both_wrote_the_titles_stay() {
        let test = "shadow::tests::where_the_gpu_and_the_title_both_wrote_the_titles_stay";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(PAGE);
        let shadow = find(window as u64, PAGE as u64).expect("a host copy");
        shadow.update(window, PAGE);
        let gpu = submission();
        shadow.writing(gpu, window, PAGE);
        // The GPU writes 32 bytes; the title 8 of them meanwhile.
        shadow.set_host(window + 0x100, 32, 0xa7);
        set_title(window + 0x108, 8, 0x5e);
        assert_eq!(
            shadow.copy_back(gpu, window, PAGE),
            24,
            "the GPU's bytes but the title's 8"
        );
        assert!(title(window + 0x100, 8).iter().all(|&b| b == 0xa7));
        assert!(
            title(window + 0x108, 8).iter().all(|&b| b == 0x5e),
            "the title's write stays"
        );
        assert!(title(window + 0x110, 16).iter().all(|&b| b == 0xa7));
        assert!(
            shadow
                .host_bytes(window + 0x108, 8)
                .iter()
                .all(|&b| b == 0x5e),
            "the copy takes it"
        );
        assert!(title(window, 0x100).iter().all(|&b| b == 0), "nothing else");
        assert!(!shadow.stale(window, window + PAGE), "current");
    }

    /// Two ranges of one submission that overlap (the world declares
    /// overlapping views of its 174 MB heap dozens of times) go back once
    /// per byte. The first copy-back takes the title's bytes into the copy
    /// where the GPU left it alone; the second must not take them for the
    /// GPU's. Before, the snapshot kept the bytes from before the
    /// submission, and the second copy-back wrote the title's older bytes
    /// over what it wrote between the two, and took the page for current.
    #[test]
    fn overlapping_ranges_of_a_submission_go_back_once() {
        let test = "shadow::tests::overlapping_ranges_of_a_submission_go_back_once";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(2 * PAGE);
        let shadow = find(window as u64, (2 * PAGE) as u64).expect("a host copy");
        shadow.update(window, 2 * PAGE);
        let (small, big) = ((window, 0x1000), (window, 2 * PAGE));
        let gpu = submission();
        shadow.writing(gpu, small.0, small.1);
        shadow.writing(gpu, big.0, big.1);
        // The GPU writes in both ranges, then in the second only, in both
        // pages; the title writes A meanwhile, in the first range.
        shadow.set_host(window + 0x10, 16, 0xab);
        shadow.set_host(window + 0x2000, 16, 0xac);
        shadow.set_host(window + PAGE + 0x10, 16, 0xad);
        let x = window + 0x100;
        set_title(x, 8, 0xa0);
        assert_eq!(shadow.copy_back(gpu, small.0, small.1), 16);
        assert!(title(x, 8).iter().all(|&b| b == 0xa0));
        // Between the two copy-backs, the title writes B there.
        set_title(x, 8, 0xb0);
        assert_eq!(
            shadow.copy_back(gpu, big.0, big.1),
            32,
            "the GPU's bytes the first left, and only them"
        );
        assert!(
            title(x, 8).iter().all(|&b| b == 0xb0),
            "the title's newer write stays"
        );
        assert!(title(window + 0x10, 16).iter().all(|&b| b == 0xab));
        assert!(title(window + 0x2000, 16).iter().all(|&b| b == 0xac));
        assert!(title(window + PAGE + 0x10, 16).iter().all(|&b| b == 0xad));
        // The second copy-back took B into the copy: both pages current.
        assert!(shadow.host_bytes(x, 8).iter().all(|&b| b == 0xb0));
        assert!(!shadow.stale(window, window + 2 * PAGE));
        assert_eq!(shadow.update(window, 2 * PAGE), 0);
    }

    /// A title write into a page GPU work in flight writes, which the GPU
    /// leaves alone. The refresh around the busy range asks the write watch
    /// for it too (W1), which clears the kernel's mark, and does not copy
    /// it; the copy-back finds nothing the GPU changed there and leaves it
    /// too. Only the journal keeps the write, and the next refresh copies
    /// the page. (Marvisak/VIctory#1322 found such writes never copied in
    /// its tree.)
    #[test]
    fn a_title_write_under_a_busy_range_reaches_the_copy_later() {
        let test = "shadow::tests::a_title_write_under_a_busy_range_reaches_the_copy_later";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(3 * PAGE);
        let shadow = find(window as u64, (3 * PAGE) as u64).expect("a host copy");
        let page = window + PAGE;
        shadow.update(window, 3 * PAGE);
        let gpu = submission();
        // Shaders write the first two pages; the GPU changes the first.
        shadow.writing(gpu, window, 2 * PAGE);
        shadow.set_host(window + 0x10, 16, 0xc2);
        set_title(page + 0x40, 8, 0x44);
        set_title(window + 2 * PAGE, 8, 0x45);
        // A refresh meanwhile copies the third page only.
        assert_eq!(shadow.update(window, 3 * PAGE), PAGE);
        assert!(shadow.host_bytes(page + 0x40, 8).iter().all(|&b| b == 0));
        if once() {
            assert!(!kernel_marked(page), "asked for the busy page too");
        }
        assert_eq!(shadow.copy_back(gpu, window, 2 * PAGE), 16);
        assert!(title(window + 0x10, 16).iter().all(|&b| b == 0xc2));
        assert!(
            title(page + 0x40, 8).iter().all(|&b| b == 0x44),
            "the title's write stays"
        );
        assert!(!shadow.stale(window, page), "current after our write");
        assert!(shadow.stale(page, page + PAGE), "the journal keeps it");
        assert_eq!(shadow.update(window, 3 * PAGE), PAGE);
        assert!(shadow.host_bytes(page + 0x40, 8).iter().all(|&b| b == 0x44));
        assert_eq!(shadow.update(window, 3 * PAGE), 0);
    }

    /// A plain copy that stops short of the byte that faulted (fast strings
    /// may: the registers then say less was copied than was) is made again
    /// up to the faulting page. What comes before that page is read, not
    /// zeroed, and a copy-back's read keeps the piece before an unmapped
    /// one. Before, the bytes from where the copy stopped to the faulting
    /// page read as zeros, and that piece was left out of the copy-back.
    #[test]
    fn a_copy_that_stops_short_of_its_fault_is_made_again_up_to_it() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const LAG_BYTES: usize = 0x40;
        let len = 4 * PAGE;
        // SAFETY: a fresh reservation of our own, never freed.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        for i in 0..len {
            // SAFETY: the test's memory.
            unsafe { ((base + i) as *mut u8).write((i * 13 + 5) as u8) };
        }
        unmap(base + 2 * PAGE, PAGE);
        let (mut ours, mut kernel) = (vec![0xeeu8; len], vec![0xeeu8; len]);
        LAG.set(LAG_BYTES);
        let zeros = read_title(ours.as_mut_ptr(), base, len);
        assert_eq!(LAG.get(), 0, "the copy faulted");
        crate::host::copy_mapped(kernel.as_mut_ptr(), base as *const u8, len, false);
        assert!(ours == kernel, "as the kernel's copy");
        assert_eq!(zeros, PAGE);
        // A copy-back's read (with plain copies).
        if plain() {
            LAG.set(LAG_BYTES);
            let mut read = vec![0u8; len];
            assert_eq!(
                read_pieces(base, &mut read),
                [(0, 2 * PAGE), (3 * PAGE, len)]
            );
            assert_eq!(LAG.get(), 0, "the copy faulted");
            assert_eq!(&read[..2 * PAGE], &title(base, 2 * PAGE)[..]);
        }
        LAG.set(LAG_BYTES);
        let (src, n) = (base + PAGE + 0x10, 2 * PAGE);
        let mut buffer = vec![0u8; n];
        // SAFETY: our buffer; title memory a fault leaves alone.
        let left = unsafe { try_copy(buffer.as_mut_ptr(), src as *const u8, n) };
        assert_eq!(left, PAGE + 0x10, "from the faulting page");
        assert_eq!(&buffer[..n - left], &title(src, n - left)[..]);
        LAG.set(0);
    }

    /// A store of the copy into a block the guard made read-only goes on,
    /// whichever vectored handler the system calls first. With the guard's
    /// registered first (the first asking for view memory before the first
    /// plain copy), the copy's is called first, and gives the store to the
    /// guard. Before, it ended the copy there, and the copy-back fell back
    /// to the kernel's write, which fails on a block guarded again
    /// meanwhile.
    #[test]
    fn a_store_into_a_guarded_block_goes_on_whichever_handler_is_first() {
        let test = "shadow::tests::a_store_into_a_guarded_block_goes_on_whichever_handler_is_first";
        alone(test, || {
            let view = view(2 * PAGE);
            // An asking guards the blocks (and registers the guard's
            // handler), then a plain copy registers the copy's.
            crate::watch::writes(view, view + 2 * PAGE, true).expect("view memory");
            assert!(!writable(view) && !writable(view + PAGE), "guarded");
            let mut read = [0u8; 16];
            // SAFETY: our buffer; the test's view.
            let left = unsafe { try_copy(read.as_mut_ptr(), view as *const u8, 16) };
            assert_eq!(left, 0);
            let ours = [0x5au8; 0x100];
            // SAFETY: the test's view; a fault ends the copy.
            let left = unsafe { try_copy((view + PAGE + 0x10) as *mut u8, ours.as_ptr(), 0x100) };
            assert_eq!(left, 0, "the store went on");
            assert!(title(view + PAGE + 0x10, 0x100).iter().all(|&b| b == 0x5a));
            assert!(writable(view + PAGE), "the guard opened the block");
            assert!(!writable(view), "and only it");
            // Noted written, as the guard's handler notes a write.
            let (now, written) = crate::watch::writes(view, view + 2 * PAGE, true).unwrap();
            assert!(written.contains(&(view / PAGE + 1, now)));
            assert!(!written.contains(&(view / PAGE, now)));
        });
    }

    /// W1 on view memory, which the guard watches (its blocks read-only
    /// until written) instead of the kernel: a range is copied once, then
    /// not again until the title writes it, neither its whole pages nor the
    /// pages it takes part of, and a write brings back the page written. A
    /// refresh around what shaders in flight write leaves that alone with
    /// one asking; the copy-back puts the GPU's bytes in title memory, keeps
    /// the title's write next to them and makes the page current.
    #[test]
    fn w1_on_guarded_memory() {
        let test = "shadow::tests::w1_on_guarded_memory";
        alone(test, || {
            let Some(_gpu) = crate::testgpu::device(test) else {
                return;
            };
            // A window of a view's own: one loose host copy.
            let mapped = view(2 * WINDOW as usize);
            let window = mapped.next_multiple_of(WINDOW as usize);
            let shadow = find(window as u64, (3 * PAGE) as u64).expect("a host copy");
            let (from, len) = (window + 0x100, 2 * PAGE);
            set_title(window, 3 * PAGE, 0x11);
            assert_eq!(shadow.update(from, len), len, "copied first");
            assert_eq!(shadow.host_bytes(from, len), title(from, len));
            assert!(!writable(window), "guarded");
            for submission in 0..3 {
                assert_eq!(
                    shadow.update(from, len),
                    0,
                    "submission {submission} after it"
                );
            }
            assert!(!shadow.stale(from, from + len));
            // The title writes the third page: the guard takes the fault.
            set_title(window + 2 * PAGE + 0x10, 4, 0x22);
            assert!(shadow.stale(from, from + len));
            assert!(
                !shadow.stale(from, window + 2 * PAGE),
                "the pages not written"
            );
            assert_eq!(shadow.update(from, len), 0x100);
            assert_eq!(
                shadow.host_bytes(window + 2 * PAGE, 0x100),
                title(window + 2 * PAGE, 0x100)
            );
            assert_eq!(shadow.update(from, len), 0);
            // Shaders in flight write the second page; the title writes the
            // first and, next to what the GPU changes, the second.
            let gpu = submission();
            shadow.writing(gpu, window + PAGE, PAGE);
            shadow.set_host(window + PAGE + 0x40, 16, 0xc3);
            set_title(window + 0x200, 8, 0x33);
            set_title(window + PAGE + 0x80, 8, 0x34);
            ASKED.set(0);
            assert_eq!(
                shadow.update(from, len),
                PAGE - 0x100,
                "the first page's part, not the busy page"
            );
            assert_eq!(ASKED.get(), 1, "one asking");
            assert!(
                shadow
                    .host_bytes(window + 0x200, 8)
                    .iter()
                    .all(|&b| b == 0x33)
            );
            assert_eq!(shadow.copy_back(gpu, window + PAGE, PAGE), 16);
            assert!(title(window + PAGE + 0x40, 16).iter().all(|&b| b == 0xc3));
            assert!(
                title(window + PAGE + 0x80, 8).iter().all(|&b| b == 0x34),
                "the title's write stays"
            );
            assert_eq!(
                shadow.host_bytes(window + PAGE, PAGE),
                title(window + PAGE, PAGE)
            );
            assert!(!shadow.stale(window + PAGE, window + 2 * PAGE), "current");
            assert_eq!(shadow.update(from, len), 0);
        });
    }

    /// The census on view memory (`XTP_COPYBACK_CENSUS=1`): a copy-back
    /// guards every 4 KiB page of its range; the first access to a page is
    /// counted once (a read, a write, XTP's own copy) and sees the GPU's
    /// bytes; a write to a read-only block leaves the rest of it uncounted;
    /// the next copy-back over the range counts what no access reached as
    /// written again by the GPU first. XTP's own copy goes on through the
    /// guards (before, a guard fault ended it: zeros in the copy). Without
    /// a census nothing is guarded.
    #[test]
    fn the_census_counts_the_first_touch_of_what_came_back() {
        use census::K;
        let test = "shadow::tests::the_census_counts_the_first_touch_of_what_came_back";
        alone(test, || {
            let Some(_gpu) = crate::testgpu::device(test) else {
                return;
            };
            let mapped = view(2 * WINDOW as usize);
            let window = mapped.next_multiple_of(WINDOW as usize);
            let shadow = find(window as u64, (3 * PAGE) as u64).expect("a host copy");
            let len = 3 * PAGE;
            shadow.update(window, len);
            // No census: nothing counted, nothing guarded.
            let gpu = submission();
            shadow.writing(gpu, window, len);
            shadow.set_host(window + 0x40, 4, 0x99);
            assert_eq!(shadow.copy_back(gpu, window, len), 4);
            assert_eq!(counted(), [0; 15]);
            assert!(title(window + 0x40, 4).iter().all(|&b| b == 0x99));
            assert_eq!(counted(), [0; 15], "no guard fault");
            CENSUS.store(2, Ordering::SeqCst);
            // The GPU writes the first page of the first block and the third
            // of the third.
            let gpu = submission();
            shadow.writing(gpu, window, len);
            shadow.set_host(window + 0x10, 16, 0xab);
            shadow.set_host(window + 2 * PAGE + 0x2010, 8, 0xcd);
            assert_eq!(shadow.copy_back(gpu, window, len), 24);
            let c = counted();
            assert_eq!(at(&c, K::Armed), len as u64, "every page guarded");
            assert_eq!(at(&c, K::Unarmed), 0);
            assert_eq!(at(&c, K::Gpu), 24);
            assert_eq!(
                at(&c, K::Faults),
                0,
                "the copy-back's own access is not a touch"
            );
            // Reads see the GPU's bytes; a page faults once.
            assert!(title(window + 0x10, 16).iter().all(|&b| b == 0xab));
            assert!(title(window + 0x10, 16).iter().all(|&b| b == 0xab));
            let _ = title(window + KERNEL_PAGE, 4);
            assert!(
                title(window + 2 * PAGE + 0x2010, 8)
                    .iter()
                    .all(|&b| b == 0xcd)
            );
            // A write to the third block: the guard of view memory opens it
            // whole.
            set_title(window + 2 * PAGE + 0x5000, 4, 0x11);
            let c = counted();
            assert_eq!(at(&c, K::Read), 3 * KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::Write), KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::AfterWrite), 14 * KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::Blocks), 2 * PAGE as u64);
            assert_eq!(at(&c, K::GpuTouched), 24);
            assert_eq!(at(&c, K::Faults), 4);
            // The next submission writes the range again, the GPU changing
            // nothing: the first block's 14 untouched pages and the second
            // block were written again first.
            let next = submission();
            shadow.writing(next, window, len);
            assert_eq!(shadow.copy_back(next, window, len), 0);
            let c = counted();
            assert_eq!(at(&c, K::Again), 30 * KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::GpuAgain), 0);
            assert_eq!(at(&c, K::Unseen), 0);
            assert_eq!(at(&c, K::Armed), len as u64);
            // XTP's own copy through the guarded second block reads it all.
            let mut read = vec![0xeeu8; PAGE];
            assert_eq!(
                read_title(read.as_mut_ptr(), window + PAGE, PAGE),
                0,
                "no zeros"
            );
            assert_eq!(read, title(window + PAGE, PAGE));
            let c = counted();
            assert_eq!(at(&c, K::Ours), PAGE as u64);
            assert_eq!(at(&c, K::Read), 0, "the test's own reads came after");
            CENSUS.store(0, Ordering::SeqCst);
        });
    }

    /// The census on write-watched memory: left alone with
    /// `XTP_COPYBACK_CENSUS=1` (a kernel write into it would fail on the
    /// guard), guarded with `all`. A write there opens one page only, and a
    /// write the guard does not see (the kernel's) counts as written unseen
    /// at the next copy-back.
    #[test]
    fn the_census_guards_write_watched_memory_with_all_only() {
        use census::K;
        let test = "shadow::tests::the_census_guards_write_watched_memory_with_all_only";
        alone(test, || {
            let Some(_gpu) = crate::testgpu::device(test) else {
                return;
            };
            let len = 2 * PAGE;
            let window = watched(len);
            let shadow = find(window as u64, len as u64).expect("a host copy");
            shadow.update(window, len);
            CENSUS.store(2, Ordering::SeqCst);
            let gpu = submission();
            shadow.writing(gpu, window, len);
            shadow.set_host(window + 0x10, 8, 0xa1);
            assert_eq!(shadow.copy_back(gpu, window, len), 8);
            let c = counted();
            assert_eq!(at(&c, K::Armed), 0);
            assert_eq!(at(&c, K::Unarmed), len as u64);
            assert!(title(window + 0x10, 8).iter().all(|&b| b == 0xa1));
            assert_eq!(at(&counted(), K::Faults), 0, "not guarded");
            CENSUS.store(3, Ordering::SeqCst);
            let gpu = submission();
            shadow.writing(gpu, window, len);
            shadow.set_host(window + PAGE + 0x3000, 8, 0xa2);
            assert_eq!(shadow.copy_back(gpu, window, len), 8);
            assert_eq!(at(&counted(), K::Armed), len as u64);
            // A title write opens its page only; a read of another page of
            // the block still faults.
            set_title(window + PAGE + 0x3004, 2, 0x5a);
            let _ = title(window + PAGE, 4);
            let c = counted();
            assert_eq!(at(&c, K::Write), KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::Read), KERNEL_PAGE as u64);
            assert_eq!(at(&c, K::AfterWrite), 0);
            assert_eq!(at(&c, K::GpuTouched), 8);
            // A kernel write into the first block (its guard fails it once,
            // then the system has taken the guard off).
            use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let ours = [0x77u8; 4];
            let mut done = 0;
            for _ in 0..2 {
                // SAFETY: the test's memory, our own buffer.
                unsafe {
                    WriteProcessMemory(
                        GetCurrentProcess(),
                        (window + 0x100) as *const _,
                        ours.as_ptr().cast(),
                        ours.len(),
                        &mut done,
                    )
                };
            }
            assert_eq!(title(window + 0x100, 4), ours, "written the second time");
            let c = counted();
            assert_eq!(at(&c, K::Faults), 0, "the kernel's access does not fault");
            let next = submission();
            shadow.writing(next, window, len);
            assert_eq!(shadow.copy_back(next, window, len), 0);
            let c = counted();
            assert_eq!(
                at(&c, K::Unseen),
                16 * KERNEL_PAGE as u64,
                "the first block"
            );
            assert_eq!(
                at(&c, K::AfterWrite),
                14 * KERNEL_PAGE as u64,
                "the second's others: its write shows in the journal too"
            );
            assert_eq!(at(&c, K::Again), 0);
            CENSUS.store(0, Ordering::SeqCst);
        });
    }

    /// Benchmark (`--ignored --nocapture`; `XTP_SHADOW_ONCE=0` times the
    /// path before W1, `XTP_COPY_MODE=kernel` W1 with the kernel's
    /// copies): a world-like frame of 39 submissions on one 64 MiB host
    /// copy, as in world-frame session C (copies 17.9 MB, copy-back 1.1 MB
    /// a frame). Each submission refreshes the range as its views merge
    /// (from and to inside pages) and 30 constant views of 600 bytes on
    /// their own (as after steps that write title memory), while the
    /// previous one's shaders write 8 MiB of it, after the title wrote 8
    /// scattered pages and a ring of 200 constant blocks of 600 bytes; then
    /// the previous one is copied back, its shaders having written 200 runs
    /// of 16 bytes over the 8 MiB and a 16 KiB block.
    #[test]
    #[ignore = "benchmark"]
    fn world_frame_copies_cost() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        let test = "shadow::tests::world_frame_copies_cost";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        const LEN: usize = 64 << 20;
        const SUBMISSIONS: usize = 39;
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let window = WINDOW as usize;
        // SAFETY: a fresh reservation of our own, never freed.
        let reserved = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN + window,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(reserved, 0);
        let base = reserved.next_multiple_of(window);
        set_title(base, LEN, 0);
        let shadow = find(base as u64, LEN as u64).expect("a host copy");
        shadow.update(base, LEN);
        let busy = (base + (16 << 20), 8 << 20);
        let pages = LEN / PAGE;
        let mut ring = 0;
        let (mut refresh, mut back) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let (mut copied, mut written) = (0, 0);
        let mut previous = None;
        for s in 0..SUBMISSIONS {
            let byte = (s % 250) as u8 + 1;
            if s == 0 {
                // Once, the pages the merged range takes only part of.
                set_title(base + 0x200, 16, byte);
                set_title(base + LEN - 0x400, 16, byte);
            }
            for p in 0..8 {
                set_title(
                    base + (s * 8 + p) * 7_919 % pages * PAGE + 0x40,
                    0x100,
                    byte,
                );
            }
            for _ in 0..200 {
                set_title(base + (48 << 20) + ring, 600, byte);
                ring = (ring + 600) % (8 << 20);
            }
            let started = std::time::Instant::now();
            copied += shadow.update(base + 0x100, LEN - 0x200);
            for v in 0..30 {
                copied += shadow.update(base + (48 << 20) + v * 262_144 + 0x40, 600);
            }
            refresh += started.elapsed();
            if let Some(gpu) = previous.take() {
                let started = std::time::Instant::now();
                written += shadow.copy_back(gpu, busy.0, busy.1);
                back += started.elapsed();
            }
            let gpu = submission();
            shadow.writing(gpu, busy.0, busy.1);
            for r in 0..200 {
                shadow.set_host(busy.0 + r * 40_960 + s % 64, 16, byte);
            }
            shadow.set_host(busy.0 + (4 << 20), 16 << 10, byte);
            previous = Some(gpu);
        }
        if let Some(gpu) = previous {
            let started = std::time::Instant::now();
            written += shadow.copy_back(gpu, busy.0, busy.1);
            back += started.elapsed();
        }
        println!(
            "BENCH once={} plain={}: per frame of {SUBMISSIONS} submissions, refreshes {:.2} ms \
             ({:.1} MB copied), copy-back {:.2} ms ({:.2} MB written)",
            once(),
            plain(),
            refresh.as_secs_f64() * 1e3,
            copied as f64 / 1e6,
            back.as_secs_f64() * 1e3,
            written as f64 / 1e6
        );
    }

    /// A copy is made without the buffer map's lock held: while one thread
    /// waits to make a buffer's copy (a heap, VidMm), another finds the
    /// copies of other buffers. Before, every recording thread queued on
    /// that lock behind the one waiting (frame-prof's session A).
    #[test]
    fn a_copy_is_made_without_the_buffer_lock_held() {
        let test = "shadow::tests::a_copy_is_made_without_the_buffer_lock_held";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let window = watched(2 << 20);
        // Two title buffers (the resources are only keys here), the first
        // of a size no other test asks for.
        let (slow, quick) = ((window, 0x7_0123u64), (window + (1 << 20), 0x100u64));
        register(0x5eed_0001, slow.0 as u64, slow.1);
        register(0x5eed_0002, quick.0 as u64, quick.1);
        let (release, held) = std::sync::mpsc::channel();
        *crate::host::buffers::HOLD.lock().unwrap() =
            Some((slow.1.next_multiple_of(1 << 16), held));
        let waiting = std::thread::spawn(move || find(slow.0 as u64, 16).is_some());
        // The first is being made; the second is found meanwhile.
        while crate::host::buffers::HOLD.lock().unwrap().is_some() {
            std::thread::yield_now();
        }
        let (found, done) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = found.send(find(quick.0 as u64, 16).is_some());
        });
        let quick_found = done.recv_timeout(std::time::Duration::from_secs(10));
        release.send(()).unwrap();
        assert_eq!(quick_found, Ok(true), "not held up by the other's making");
        assert!(waiting.join().unwrap());
        let first = find(slow.0 as u64, 16).unwrap();
        assert!(
            Arc::ptr_eq(&first, &find(slow.0 as u64 + 8, 8).unwrap()),
            "made once"
        );
        unregister(0x5eed_0001, slow.0 as u64);
        unregister(0x5eed_0002, quick.0 as u64);
    }
}
