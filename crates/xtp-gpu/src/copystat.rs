//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `XTP_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only), and
//! `kernel` has the host copies' refreshes and copy-backs go through the
//! kernel's copy (`ReadProcessMemory`, `WriteProcessMemory`) as before W1,
//! to time it against their plain copies (`XTP_SHADOW_ONCE=0` brings back
//! all of W1's before). `XTP_COPYBACK_CENSUS=1` (or `all`) adds the
//! copy-back's first-touch census ([`crate::shadow::census`]) to the line.
//! A second line gives the work of the copies in the card's memory
//! ([`crate::shadow::cards`]); the first line's snapshots and copy-back
//! compare are those of the copies in system memory.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    /// Host copy updates (merged per copy at each flush) that copied
    /// nothing, and the bytes updates did not copy: current in the copy,
    /// or written by GPU work in flight.
    Skipped,
    SkippedBytes,
    /// Host copy updates; the bytes of the ranges they covered; of those,
    /// the bytes copied from title memory.
    Updates,
    Updated,
    Refreshed,
    /// Of those, bytes read with plain loads, and bytes that could not be
    /// read (zeros); the plain loads' time in nanoseconds (small copies
    /// take less than a microsecond each).
    Plain,
    Zeroed,
    PlainNs,
    /// Copy-back: bytes of shader-written ranges compared with their
    /// snapshot; bytes of title memory read where the GPU changed them,
    /// and the copies (or kernel calls) that read them; runs of GPU bytes
    /// written into title memory; bytes the GPU changed that the title
    /// wrote too since the snapshot, which keep the title's (the copy-back's
    /// three-way merge).
    BackCompared,
    BackRead,
    BackReads,
    BackRuns,
    BackKept,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Time in the guard of view memory (part of `WatchUs`), the system
    /// queries it made (is it a view, is it a placeholder), its protection
    /// changes, and the blocks it skipped as unmapped.
    GuardUs,
    ViewQueries,
    Protections,
    Absent,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Snapshots of shader-written ranges before their submission
    /// (`Shadow::writing`): declarations, bytes copied, their time in
    /// microseconds; snapshot mirrors made (each of a host copy's size,
    /// kept for good), and their bytes.
    Snapshots,
    SnapshotBytes,
    SnapshotUs,
    Mirrors,
    MirrorBytes,
    /// Copies in the card's memory (`shadow::cards`): shader-written ranges
    /// declared (no snapshot taken); plans of submissions and their time
    /// in microseconds; ranges of CPU sides sent up and their bytes; pieces
    /// hashed before and after the lists; bytes of GPU copies' destinations
    /// fetched; at copy-back, bytes of pieces compared on the card, bytes
    /// of those the card found changed, and the copy-back's time there in
    /// microseconds.
    CardDeclared,
    CardPlans,
    CardPlanUs,
    CardUps,
    CardUpBytes,
    CardSpans,
    CardFetched,
    CardCompared,
    CardChanged,
    CardBackUs,
    /// `XTP_FULL_REFRESH=<n>`: host copies taken whole before a
    /// submission, their bytes and time in microseconds.
    FullRefreshes,
    FullRefreshBytes,
    FullRefreshUs,
    /// Of those, bytes of 64 KiB pieces the copies held otherwise than
    /// title memory while the write tracking had them current: missed
    /// writes.
    FullRefreshMissed,
    /// Of `BackCompared` (copies in system memory): bytes in windows of
    /// loose memory, in title buffers under 1 MiB, under the card's least
    /// size, and of that size or more (past the card's budget).
    BackComparedLoose,
    BackComparedSmall,
    BackComparedMid,
    BackComparedBig,
    ReadFaults,
    WriteFaults,
    ReadFaultNs,
    KernelReadPages,
    KernelReadNs,
    DeferredBytes,
    CapacityCopies,
    BackScanNs,
    Count,
}
const COUNTERS: usize = C::Count as usize;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all, 3 the
/// host copies' through the kernel.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("XTP_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        Ok("kernel") => 3,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} updates over {:.1} MB copied {:.1} MB ({:.1} MB plain in {:.2} ms, {:.1} MB unreadable) \
         and skipped \
         {:.1} MB ({:.0} updates whole); copy-back compared {:.1} MB, read {:.1} MB in {:.0} \
         reads, wrote {:.0} runs, kept {:.0} title bytes; {:.0} runs {:.1} ms; {:.0} queries \
         {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; guard {:.2} ms, {:.0} view queries, \
         {:.0} protection changes, {:.0} unmapped blocks skipped; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; snapshots {:.0} over {:.1} MB in \
         {:.2} ms, {:.1} mirrors made ({:.1} MB; {:.0} MB held; mirrors {})",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Updates as usize],
        mb(c[C::Updated as usize]),
        mb(c[C::Refreshed as usize]),
        mb(c[C::Plain as usize]),
        c[C::PlainNs as usize] / 1e6,
        mb(c[C::Zeroed as usize]),
        mb(c[C::SkippedBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::BackCompared as usize]),
        mb(c[C::BackRead as usize]),
        c[C::BackReads as usize],
        c[C::BackRuns as usize],
        c[C::BackKept as usize],
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        ms(c[C::GuardUs as usize]),
        c[C::ViewQueries as usize],
        c[C::Protections as usize],
        c[C::Absent as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        c[C::Snapshots as usize],
        mb(c[C::SnapshotBytes as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::Mirrors as usize],
        mb(c[C::MirrorBytes as usize]),
        crate::shadow::MIRRORS.load(Relaxed) as f64 / 1e6,
        crate::shadow::mirror_phase(),
    );
    if c[C::BackCompared as usize] > 0.0 {
        tracing::info!(
            "snapshot plus CPU compare per frame: {:.3} ms ({:.3} ms scanning)",
            ms(c[C::SnapshotUs as usize]) + c[C::BackScanNs as usize] / 1e6,
            c[C::BackScanNs as usize] / 1e6
        );
    }
    if crate::readmap::enabled() {
        tracing::info!(
            frames = frames as u64,
            pending_bytes = crate::readmap::PENDING_BYTES.load(Relaxed),
            capacity_copies = (c[C::CapacityCopies as usize] * frames).round() as u64,
            read_faults = (c[C::ReadFaults as usize] * frames).round() as u64,
            write_faults = (c[C::WriteFaults as usize] * frames).round() as u64,
            handler_us = (c[C::ReadFaultNs as usize] * frames / 1e3).round() as u64,
            kernel_pages = (c[C::KernelReadPages as usize] * frames).round() as u64,
            kernel_us = (c[C::KernelReadNs as usize] * frames / 1e3).round() as u64,
            "copy-back map per frame: {:.2} MB deferred, {:.1} read and {:.1} write faults, {:.3} ms in handlers",
            mb(c[C::DeferredBytes as usize]),
            c[C::ReadFaults as usize],
            c[C::WriteFaults as usize],
            c[C::ReadFaultNs as usize] / 1e6
        );
    }
    if c[C::FullRefreshes as usize] > 0.0 {
        tracing::info!(
            "full refreshes per frame (XTP_FULL_REFRESH): {:.1}, {:.1} MB in {:.1} ms, {:.2} MB              of pieces the write tracking missed",
            c[C::FullRefreshes as usize],
            mb(c[C::FullRefreshBytes as usize]),
            ms(c[C::FullRefreshUs as usize]),
            mb(c[C::FullRefreshMissed as usize]),
        );
    }
    if c[C::BackCompared as usize] > 0.0 {
        tracing::info!(
            "copy-back compared in system memory per frame, by copy: loose windows {:.1} MB,              buffers under 1 MiB {:.1} MB, under the card's least size {:.1} MB, past the              card's budget {:.1} MB",
            mb(c[C::BackComparedLoose as usize]),
            mb(c[C::BackComparedSmall as usize]),
            mb(c[C::BackComparedMid as usize]),
            mb(c[C::BackComparedBig as usize]),
        );
    }
    if let Some(cards) = crate::shadow::card_summary() {
        tracing::info!(
            "copies on the card per frame ({cards}): {:.0} ranges declared, no snapshot; {:.1} \
             plans in {:.2} ms; up {:.1} MB in {:.0} ranges; {:.0} pieces hashed before and \
             after; fetched {:.2} MB; at copy-back compared {:.1} MB on the card, {:.2} MB \
             changed, merged in {:.2} ms",
            c[C::CardDeclared as usize],
            c[C::CardPlans as usize],
            ms(c[C::CardPlanUs as usize]),
            mb(c[C::CardUpBytes as usize]),
            c[C::CardUps as usize],
            c[C::CardSpans as usize],
            mb(c[C::CardFetched as usize]),
            mb(c[C::CardCompared as usize]),
            mb(c[C::CardChanged as usize]),
            ms(c[C::CardBackUs as usize]),
        );
    }
    // With `XTP_COPYBACK_CENSUS`: what the CPU touches of what came back.
    crate::shadow::census::log(frames);
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
