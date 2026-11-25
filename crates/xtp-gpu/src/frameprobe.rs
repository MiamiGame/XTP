//! Diagnostics: what chosen draws and dispatches read through their
//! descriptor tables, at the moment they run, and the order in which the
//! R9G9B9E5 stand-ins are decoded against their writers
//! (docs/analysis/pix-world-frame.md §11).
//!
//! `XTP_FRAME_PROBE=<program>[,<program>...]` (16 hex digits each), with
//! `XTP_CAPTURE=<folder>` and frames listed by `XTP_CAPTURE_FRAMES` or
//! `XTP_CAPTURE_WORLD`, writes `probe-<n>.txt` for each listed frame `n`
//! once its lists have run. For each op of a listed program (the first
//! [`OPS`] a frame), and each descriptor its stage reads through a table:
//! - where the descriptor is: the table's byte offset the title gave (its
//!   user data), the title slot, whether that slot is past the host heap's
//!   mirrored slots, and the host slot the shader reads (in a table made
//!   afresh in dynamic slots, `draw::State::tables`, when it has one);
//! - the view the title wrote there, and for a mirrored slot the view the
//!   host slot was last made from;
//! - for a buffer view: the title range, the memory there (private,
//!   mapped: a view of physical pages; its protection), at how many other
//!   addresses the title shows its 64 KiB page (`watch::views_of`), and the
//!   host copy it resolves to;
//! - the range's bytes three times: title memory when the op is recorded,
//!   title memory when its list is submitted (what the submission's
//!   refresh takes), and the host copy on the GPU just before the op
//!   (copied in the list where the op stands). [`BYTES`] at most each,
//!   counted (non-zero bytes) and shown as words up to [`SHOWN`] bytes.
//!
//! Every listed frame also gets the R9G9B9E5 timeline (upstream watch item
//! 360): each texture an op reads through a table in title format 67 (an
//! R9G9B9E5 view, which the host keeps as an RGBA16F stand-in that
//! `resource::decode_now` fills on the GPU, in a list of XTP's own run
//! at the start of the next title submission), each texture an op writes
//! through unordered access at the same title address, and the title
//! submissions with their lists in order. The report names, per R9G9B9E5
//! texture, the submission whose start runs its decoding (the first after
//! the frame's first read of it) and the submissions of its writers.
//!
//! shortcut: the decoding is inferred from where the first read stands
//! (`resource::decode_now` is not told about), so a texture whose
//! decoding could not be made is reported as decoded too.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use xtp_shader::ResourceType;

use crate::host::HostList;
use crate::object::This;

/// Ops probed a frame, at most.
const OPS: usize = 96;
/// Bytes of a range snapshot (title memory) or copied back (host copy).
const BYTES: u64 = 64 << 10;
/// Bytes of a snapshot shown as words.
const SHOWN: usize = 512;
/// Read-back memory a frame.
const BUDGET: u64 = 48 << 20;
/// The title format of R9G9B9E5.
const R9G9B9E5: u32 = 67;
/// Presents after a listed frame at which it is reported even if some
/// probed op's list was never submitted.
const LATE: u64 = 4;

/// A descriptor a stage reads through a table, as `draw::State` resolves
/// it for the op.
pub(crate) struct Table {
    pub(crate) slot: u8,
    pub(crate) offset_dwords: u32,
    pub(crate) kind: ResourceType,
    /// The title's descriptor.
    pub(crate) handle: u64,
    /// Its slot in the title's heap.
    pub(crate) index: Option<u32>,
    /// The host heap's slots mirroring the title's.
    pub(crate) mirrored: Option<u32>,
    /// The table's byte offset in its heap, as the title gave it.
    pub(crate) title_offset: Option<u32>,
    /// The byte offset the shader gets instead, for a table made afresh.
    pub(crate) fresh_offset: Option<u32>,
    /// For a mirrored slot: the view its host slot was made from.
    pub(crate) host_view: Option<String>,
}

/// A range's bytes, title memory or host copy.
struct Snapshot {
    bytes: Vec<u8>,
}

impl Snapshot {
    fn text(&self, len: u64) -> String {
        let nz = self.bytes.iter().filter(|&&b| b != 0).count();
        let words: Vec<String> = self.bytes[..self.bytes.len().min(SHOWN)]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| format!("{:08x}", u32::from_le_bytes(w)))
            .collect();
        let more = if self.bytes.len() > SHOWN { " …" } else { "" };
        format!(
            "non-zero {nz} of {} bytes{} [{}{more}]",
            self.bytes.len(),
            if (self.bytes.len() as u64) < len {
                format!(" (first {} of {len})", self.bytes.len())
            } else {
                String::new()
            },
            words.join(" ")
        )
    }
}

/// One descriptor of a probed op.
struct Read {
    text: String,
    /// The buffer range it views: title address and bytes.
    range: Option<(u64, u64)>,
    at_record: Option<Snapshot>,
    at_submit: Option<Snapshot>,
    /// Where the host copy's bytes are in the read-back buffer, or why
    /// they are not.
    gpu: Result<(u64, u64), &'static str>,
}

struct Op {
    list: usize,
    /// The title submission that ran its list, once one has.
    submission: Option<u64>,
    program: String,
    reads: Vec<Read>,
}

/// A texture an op of a listed frame reads or writes through a table.
struct TextureUse {
    list: usize,
    after: u64,
    program: String,
    address: u64,
    title_format: u32,
    host: String,
    write: bool,
}

/// What a listed frame gathers until its report.
struct Frame {
    frame: u64,
    matched: usize,
    ops: Vec<Op>,
    textures: Vec<TextureUse>,
    readback: Option<ID3D12Resource>,
    used: u64,
    /// Title submissions made while it was recorded or ran: number, lists.
    submissions: Vec<(u64, Vec<usize>)>,
}

static FRAMES: Mutex<Vec<Frame>> = Mutex::new(Vec::new());
/// Title submissions so far.
static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The program whose stage this thread prepares (`begin`).
    static PROGRAM: RefCell<String> = const { RefCell::new(String::new()) };
}

fn frames() -> MutexGuard<'static, Vec<Frame>> {
    FRAMES.lock().unwrap_or_else(|e| e.into_inner())
}

fn key(list: &HostList) -> usize {
    std::ptr::from_ref(list) as usize
}

/// The programs `XTP_FRAME_PROBE` lists.
fn programs() -> &'static [String] {
    static LIST: OnceLock<Vec<String>> = OnceLock::new();
    LIST.get_or_init(|| parse(&std::env::var("XTP_FRAME_PROBE").unwrap_or_default()))
}

fn parse(text: &str) -> Vec<String> {
    text.split(',')
        .map(|h| h.trim().to_ascii_lowercase())
        .filter(|h| h.len() == 16 && h.chars().all(|c| c.is_ascii_hexdigit()))
        .collect()
}

/// The frame being recorded, if the probe runs and lists it.
fn recording() -> Option<u64> {
    if programs().is_empty() || crate::capture::folder().is_none() {
        return None;
    }
    let frame = crate::queue::presented() + 1;
    crate::capture::listed(frame).then_some(frame)
}

fn frame_mut(frames: &mut Vec<Frame>, frame: u64) -> &mut Frame {
    if let Some(i) = frames.iter().position(|f| f.frame == frame) {
        return &mut frames[i];
    }
    frames.push(Frame {
        frame,
        matched: 0,
        ops: Vec::new(),
        textures: Vec::new(),
        readback: None,
        used: 0,
        submissions: Vec::new(),
    });
    frames.last_mut().unwrap()
}

/// `draw::State::bind`, before a stage's tables are prepared: the program
/// the texture uses that follow ([`texture`]) belong to.
pub(crate) fn begin(program: &str) {
    if recording().is_some() {
        PROGRAM.with(|p| program.clone_into(&mut p.borrow_mut()));
    }
}

/// Whether a stage of `program` is probed in the frame being recorded.
pub(crate) fn wanted(program: &str) -> bool {
    recording().is_some() && programs().iter().any(|p| p == program)
}

/// `draw::State::bind`, once a stage of a probed program has its tables
/// (`tables`) for the op about to be recorded in `list`.
pub(crate) fn stage(list: &HostList, program: &str, tables: Vec<Table>) {
    let Some(frame) = recording() else {
        return;
    };
    {
        let mut frames = frames();
        let f = frame_mut(&mut frames, frame);
        f.matched += 1;
        if f.ops.len() >= OPS {
            return;
        }
    }
    // Looked up without the lock: finding a host copy may make one, which
    // may wait.
    let found: Vec<(Read, Option<CopyFrom>)> = tables.iter().map(read).collect();
    let mut frames = frames();
    let f = frame_mut(&mut frames, frame);
    let reads = found
        .into_iter()
        .map(|(mut r, from)| {
            if let Some((s, address, len)) = from {
                r.gpu = copy(f, list, &s, address, len);
            }
            r
        })
        .collect();
    f.ops.push(Op {
        list: key(list),
        submission: None,
        program: program.to_owned(),
        reads,
    });
}

/// A host copy, and the title range of it to copy back.
type CopyFrom = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// What `t` reads, its title bytes taken now, and the host copy whose
/// bytes the GPU reads.
fn read(t: &Table) -> (Read, Option<CopyFrom>) {
    let mut text = format!("s{}+{} {:?}:", t.slot, t.offset_dwords, t.kind);
    if let Some(offset) = t.title_offset {
        let _ = write!(text, " table +{offset:#x}");
    }
    match (t.index, t.mirrored) {
        (Some(i), Some(m)) if i >= m => {
            let _ = write!(text, " title slot {i} (past the {m} mirrored)");
        }
        (Some(i), _) => {
            let _ = write!(text, " title slot {i}");
        }
        (None, _) => text.push_str(" outside the bound heaps"),
    }
    match t.fresh_offset {
        Some(offset) => {
            let slot = offset / 32 + t.offset_dwords / 8;
            let _ = write!(
                text,
                ", made afresh: the shader gets +{offset:#x}, host slot {slot}"
            );
        }
        None => text.push_str(", host slot = title slot"),
    }
    let view = crate::descriptor::view(t.handle);
    let _ = write!(text, "\n      view: {}", view_text(view.as_ref()));
    if let Some(h) = &t.host_view {
        let _ = write!(text, "\n      host slot made from: {h}");
    }
    // A small texture the title placed (the 1×1 ones the CPU fills): its
    // first texels as title memory holds them, whatever the swizzle.
    if let Some(crate::descriptor::View::Placed(_, r, d)) = &view
        && r[0] != 1
        && u64::from(r[4]) * u64::from(r[6]) <= 16
    {
        let address = u64::from(d[10]) | u64::from(d[11]) << 32;
        let _ = write!(
            text,
            "\n      texels in title memory {address:#x}: {}; {}",
            memory_text(address),
            snapshot(address, 64).map_or_else(|| "not readable".to_owned(), |s| s.text(64))
        );
    }
    let range = view
        .as_ref()
        .and_then(crate::descriptor::host::buffer_range)
        .filter(|&(_, len)| len > 0);
    let mut gpu = Err("not a buffer view");
    let mut at_record = None;
    let mut from = None;
    if let Some((address, len)) = range {
        let _ = write!(
            text,
            "\n      memory {address:#x}+{len:#x}: {}, its 64 KiB page shown at {} other addresses",
            memory_text(address),
            crate::watch::views_of(address as usize)
        );
        let take = len.min(BYTES);
        match crate::shadow::find(address, take) {
            Some(s) => {
                let _ = write!(
                    text,
                    "; host copy at +{:#x} ({} bytes to its end{})",
                    s.offset(address),
                    s.left(address),
                    if s.carded() { ", card" } else { "" }
                );
                let n = take.min(s.left(address));
                from = Some((s, address, n));
            }
            None => {
                text.push_str("; no host copy");
                gpu = Err("no host copy");
            }
        }
        at_record = snapshot(address, take);
    }
    let r = Read {
        text,
        range,
        at_record,
        at_submit: None,
        gpu,
    };
    (r, from)
}

/// A view in a few words.
fn view_text(view: Option<&crate::descriptor::View>) -> String {
    use crate::descriptor::View;
    match view {
        None => "none (no view of XTP's at the title's descriptor)".into(),
        Some(View::Placed(uav, r, d)) => format!(
            "placed uav={uav} dim {} {}x{} f{} layout {:#x} at {:#x}, view words {:x?}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32,
            &d[..10]
        ),
        Some(View::Shader(r, desc)) => format!(
            "resource {:#x} ({}) view {desc:x?}",
            r,
            host_text(*r as This)
        ),
        Some(View::Unordered(r, counter, desc)) => format!(
            "uav of {:#x} ({}) counter {counter:#x} view {desc:x?}",
            r,
            host_text(*r as This)
        ),
        Some(View::Constant(address, size)) => format!("constant {address:#x}+{size}"),
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A title resource's host object, in a few words.
fn host_text(resource: This) -> String {
    match crate::resource::host_resource(resource) {
        Some(t) => {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "host {}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        }
        None => match crate::resource::buffer_bytes(resource) {
            Some((p, len)) => format!("buffer {:#x}+{len:#x}", p as usize),
            None => "no host object".into(),
        },
    }
}

/// The title memory at `address`: its kind, state and protection.
fn memory_text(address: u64) -> String {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, MEM_RESERVE, MEMORY_BASIC_INFORMATION,
        VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size; a query-only address.
    let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
    if n == 0 {
        return "not queried".into();
    }
    let kind = match info.Type {
        MEM_PRIVATE => "private",
        MEM_MAPPED => "mapped (a view of physical pages)",
        MEM_IMAGE => "image",
        _ => "free",
    };
    let state = match info.State {
        MEM_COMMIT => "committed",
        MEM_RESERVE => "reserved",
        _ => "free",
    };
    format!(
        "{kind}, {state}, protection {:#x}, region {:#x}+{:#x} of allocation {:#x}",
        info.Protect, info.BaseAddress as usize, info.RegionSize, info.AllocationBase as usize
    )
}

/// `len` bytes of title memory at `address`, if they can be read.
fn snapshot(address: u64, len: u64) -> Option<Snapshot> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery,
    };
    let (start, end) = (address as usize, (address + len) as usize);
    let mut p = start;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size; a query-only address.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let readable = matches!(
            info.Protect,
            PAGE_READONLY
                | PAGE_READWRITE
                | PAGE_WRITECOPY
                | PAGE_EXECUTE_READ
                | PAGE_EXECUTE_READWRITE
        );
        if n == 0 || info.State != MEM_COMMIT || !readable {
            return None;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    let mut bytes = vec![0u8; len as usize];
    // SAFETY: `len` committed, readable bytes of title memory (checked
    // just above); the title may write them meanwhile, which a diagnostic
    // accepts.
    unsafe { std::ptr::copy_nonoverlapping(start as *const u8, bytes.as_mut_ptr(), bytes.len()) };
    Some(Snapshot { bytes })
}

/// Records, in `list` where it stands, a copy of `len` bytes of the host
/// copy `s` at title `address` into the frame's read-back buffer.
fn copy(
    f: &mut Frame,
    list: &HostList,
    s: &crate::shadow::Shadow,
    address: u64,
    len: u64,
) -> Result<(u64, u64), &'static str> {
    let host = crate::host::host().ok_or("no host")?;
    if f.readback.is_none() {
        f.readback = Some(readback_buffer(&host.device).map_err(|_| "no read-back buffer")?);
    }
    let at = f.used.next_multiple_of(16);
    if at + len > BUDGET {
        return Err("read-back budget spent");
    }
    f.used = at + len;
    let readback = f.readback.as_ref().ok_or("no read-back buffer")?;
    let (common, source) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    );
    // SAFETY: an open list; the host copy and our read-back buffer, which
    // the list keeps, the ranges inside both. Host copies are taken as
    // common, as `shadow` takes them (and `passdump::sample_buffer`).
    unsafe {
        list.list.ResourceBarrier(&[
            crate::host::uav_barrier(),
            crate::host::transition(&s.buffer, common, source),
        ]);
        list.list
            .CopyBufferRegion(readback, at, &s.buffer, s.offset(address), len);
        list.list
            .ResourceBarrier(&[crate::host::transition(&s.buffer, source, common)]);
    }
    list.keep(&s.buffer);
    list.keep(readback);
    Ok((at, len))
}

fn readback_buffer(device: &ID3D12Device) -> windows::core::Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: BUDGET,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; the out pointer is ours.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut buffer,
        )?;
    }
    let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
    crate::host::name(&buffer, || "XTP_FRAME_PROBE read-back".to_owned());
    Ok(buffer)
}

/// `draw::readable`: an op being prepared in `list` reads (or, with
/// `write`, writes through unordered access) the title texture `resource`.
pub(crate) fn texture(list: &HostList, resource: This, write: bool) {
    let Some(frame) = recording() else {
        return;
    };
    let Some((data, desc)) = crate::resource::texture_memory(resource) else {
        return;
    };
    let title_format = desc.format();
    if !write && title_format != R9G9B9E5 {
        return;
    }
    let use_ = TextureUse {
        list: key(list),
        after: SUBMISSIONS.load(Ordering::Relaxed),
        program: PROGRAM.with(|p| p.borrow().clone()),
        address: data as u64,
        title_format,
        host: host_text(resource),
        write,
    };
    frame_mut(&mut frames(), frame).textures.push(use_);
}

/// `Host::submit`: a title submission of `runs`, in order, before its
/// refreshes. Takes the probed ranges' title memory for the ops of these
/// lists, and reports the frames presented since, once what was submitted
/// so far has run.
pub(crate) fn submitted(runs: &[&HostList]) {
    if programs().is_empty() {
        return;
    }
    let number = SUBMISSIONS.fetch_add(1, Ordering::Relaxed) + 1;
    let lists: Vec<usize> = runs.iter().map(|l| key(l)).collect();
    let presented = crate::queue::presented();
    let done: Vec<Frame> = {
        let mut frames = frames();
        // Before this submission is counted: what `Host::after` waits for
        // below is the submissions made so far, not this one. The title
        // records a frame's lists while the one before is presented, so
        // a frame is ready once every op it probed has had its list
        // submitted, or [`LATE`] presents after it at most.
        let (done, kept) = frames.drain(..).partition(|f: &Frame| {
            f.frame <= presented && f.ops.iter().all(|o| o.submission.is_some())
                || presented >= f.frame + LATE
        });
        *frames = kept;
        for f in frames.iter_mut() {
            f.submissions.push((number, lists.clone()));
            for op in f
                .ops
                .iter_mut()
                .filter(|o| o.submission.is_none() && lists.contains(&o.list))
            {
                op.submission = Some(number);
                for r in &mut op.reads {
                    if let Some((address, len)) = r.range {
                        r.at_submit = snapshot(address, len.min(BYTES));
                    }
                }
            }
        }
        done
    };
    if done.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else {
        return;
    };
    for f in done {
        host.after(move || write_report(f));
    }
}

fn write_report(f: Frame) {
    let Some(folder) = crate::capture::folder() else {
        return;
    };
    let gpu = f.readback.as_ref().and_then(|r| {
        let mut p = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: f.used as usize,
        };
        // SAFETY: our read-back buffer, written by lists that have run.
        unsafe { r.Map(0, Some(&range), Some(&mut p)) }.ok()?;
        // SAFETY: the mapping holds `used` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(p.cast::<u8>(), f.used as usize) }.to_vec();
        // SAFETY: mapped just above; nothing written.
        unsafe { r.Unmap(0, Some(&D3D12_RANGE::default())) };
        Some(bytes)
    });
    let text = report(&f, gpu.as_deref());
    let path = folder.join(format!("probe-{}.txt", f.frame));
    let _ = std::fs::create_dir_all(folder);
    match std::fs::write(&path, text) {
        Ok(()) => tracing::info!(path = %path.display(), "frame probe written"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame probe not written"),
    }
}

/// The first submission after `after` that holds `list`.
fn submission_of(f: &Frame, list: usize, after: u64) -> Option<u64> {
    f.submissions
        .iter()
        .find(|(n, lists)| *n > after && lists.contains(&list))
        .map(|&(n, _)| n)
}

fn report(f: &Frame, gpu: Option<&[u8]>) -> String {
    let mut out = String::new();
    // Lists numbered in submission order, from 1.
    let mut numbers: HashMap<usize, usize> = HashMap::new();
    for (_, lists) in &f.submissions {
        for &l in lists {
            let n = numbers.len() + 1;
            numbers.entry(l).or_insert(n);
        }
    }
    let list_name = |list: usize, after: u64| {
        let sub = submission_of(f, list, after)
            .map_or_else(|| "not submitted".to_owned(), |n| format!("submission {n}"));
        match numbers.get(&list) {
            Some(n) => format!("L{n} ({sub})"),
            None => format!("list {list:#x} ({sub})"),
        }
    };
    let _ = writeln!(
        out,
        "XTP frame probe of frame {}: programs {:?}; {} ops probed of {} matching; title submissions {}..{}.",
        f.frame,
        programs(),
        f.ops.len(),
        f.matched,
        f.submissions.first().map_or(0, |s| s.0),
        f.submissions.last().map_or(0, |s| s.0),
    );
    let _ = writeln!(
        out,
        "Bytes: title memory when the op was recorded, when its list was submitted (before the submission's refresh), and the host copy on the GPU just before the op; words in hex, little-endian."
    );
    let _ = writeln!(out, "\n== Ops");
    for (i, op) in f.ops.iter().enumerate() {
        let _ = writeln!(
            out,
            "op {} {} {}",
            i + 1,
            match (op.submission, numbers.get(&op.list)) {
                (Some(s), Some(n)) => format!("L{n} (submission {s})"),
                _ => format!("list {:#x} (not submitted)", op.list),
            },
            op.program
        );
        for r in &op.reads {
            let _ = writeln!(out, "    {}", r.text);
            let Some((_, len)) = r.range else {
                continue;
            };
            let record = r.at_record.as_ref();
            let _ = writeln!(
                out,
                "      at record: {}",
                record.map_or_else(|| "not readable".to_owned(), |s| s.text(len))
            );
            let submit = r.at_submit.as_ref();
            let same = |a: Option<&Snapshot>, b: Option<&Snapshot>| {
                a.zip(b).is_some_and(|(a, b)| a.bytes == b.bytes)
            };
            let _ = writeln!(
                out,
                "      at submit: {}",
                match submit {
                    None => "not taken".to_owned(),
                    Some(_) if same(submit, record) => "as at record".to_owned(),
                    Some(s) => s.text(len),
                }
            );
            let ran = op.submission.is_some();
            let on_gpu = match (&r.gpu, gpu) {
                (Err(why), _) => Err(*why),
                // Its list never ran: the read-back bytes are not its.
                (Ok(_), _) if !ran => Err("its list was not submitted"),
                (Ok(_), None) => Err("not read back"),
                (Ok((at, n)), Some(bytes)) => bytes
                    .get(*at as usize..(at + n) as usize)
                    .map(|b| Snapshot { bytes: b.to_vec() })
                    .ok_or("not read back"),
            };
            let _ = writeln!(
                out,
                "      on the GPU: {}",
                match &on_gpu {
                    Err(why) => (*why).to_owned(),
                    Ok(s) if same(Some(s), submit) => "as at submit".to_owned(),
                    Ok(s) => {
                        let differ = submit.map_or(0, |t| {
                            t.bytes.iter().zip(&s.bytes).filter(|(a, b)| a != b).count()
                        });
                        format!("{} ({differ} bytes differ from submit)", s.text(len))
                    }
                }
            );
        }
    }
    let _ = writeln!(out, "\n== R9G9B9E5 reads and the writes at their address");
    let mut seen: Vec<u64> = Vec::new();
    for t in f.textures.iter().filter(|t| !t.write) {
        if seen.contains(&t.address) {
            continue;
        }
        seen.push(t.address);
        let _ = writeln!(
            out,
            "texture {:#x} (title format {}, {}):",
            t.address, t.title_format, t.host
        );
        let decoded = t.after + 1;
        let _ = writeln!(
            out,
            "  first read by {} in {}: its decoding runs at the start of submission {decoded}",
            t.program,
            list_name(t.list, t.after),
        );
        // Each program and list once, with how many uses.
        let mut uses: Vec<(bool, &str, usize, u64, u32, usize)> = Vec::new();
        for u in f.textures.iter().filter(|u| u.address == t.address) {
            let sub = submission_of(f, u.list, u.after).unwrap_or(0);
            match uses
                .iter_mut()
                .find(|x| x.0 == u.write && x.1 == u.program && x.2 == u.list)
            {
                Some(x) => x.5 += 1,
                None => uses.push((u.write, &u.program, u.list, sub, u.title_format, 1)),
            }
        }
        for &(write, program, list, _, format, n) in &uses {
            let _ = writeln!(
                out,
                "  {} by {program} in {} ({n} uses, title format {format})",
                if write { "written" } else { "read" },
                list_name(list, 0)
            );
        }
        let late: Vec<u64> = uses
            .iter()
            .filter(|x| x.0 && x.3 >= decoded)
            .map(|x| x.3)
            .collect();
        let _ = writeln!(
            out,
            "  => {}",
            if !uses.iter().any(|x| x.0) {
                "no unordered-access writer at this address in the frame's tables".to_owned()
            } else if late.is_empty() {
                "every writer's list ran before the decoding".to_owned()
            } else {
                format!(
                    "the decoding runs before writers in submissions {late:?}: their texels show a frame later"
                )
            }
        );
    }
    let _ = writeln!(out, "\n== Title submissions");
    for (n, lists) in &f.submissions {
        let names: Vec<String> = lists
            .iter()
            .map(|l| {
                numbers
                    .get(l)
                    .map_or_else(|| format!("{l:#x}"), |n| format!("L{n}"))
            })
            .collect();
        let _ = writeln!(out, "submission {n}: {}", names.join(" "));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_parse() {
        assert_eq!(
            parse("db4835a364449a45, 1CD9BC634417A981,short,zz4835a364449a45"),
            vec!["db4835a364449a45", "1cd9bc634417a981"]
        );
        assert!(parse("").is_empty());
    }

    #[test]
    fn snapshots_count_and_show_words() {
        let s = Snapshot {
            bytes: vec![1, 0, 0, 0, 0, 0, 0, 0, 0xff, 0x80, 0, 1],
        };
        assert_eq!(
            s.text(12),
            "non-zero 4 of 12 bytes [00000001 00000000 010080ff]"
        );
        assert_eq!(
            s.text(64),
            "non-zero 4 of 12 bytes (first 12 of 64) [00000001 00000000 010080ff]"
        );
    }

    #[test]
    fn title_memory_reads_back() {
        let data: Vec<u8> = (0..64).collect();
        let s = snapshot(data.as_ptr() as u64, 64).unwrap();
        assert_eq!(s.bytes, data);
        assert!(snapshot(0x10, 16).is_none());
    }
}
