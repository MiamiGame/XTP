//! The graphics and mesh pipelines sessions made, kept beside the pipeline
//! library so that `xtp precook` makes them again before the next session
//! ([`super::precook`]). After a translator change every host pipeline has
//! a new name in the library (its name hashes the translated code), and the
//! precook could make only the compute ones, which need their container
//! alone: the others need the title's graphics state, which only a session
//! sees. A Series S session then dropped some 970 draws a frame "being
//! made" while its world loaded (run `20261125-022031`).
//!
//! - What is kept: each title pipeline with a graphics state that the title
//!   creates, at its creation, whether it is ever drawn with or not: its
//!   shaders' containers and the subobjects of the title's stream as the
//!   title gave them, in its order. The title creates its pipelines while it
//!   loads, long before it draws with them, so a session that reaches the
//!   world keeps those of the whole game, not only those its frames drew
//!   with ([`created`]). The title's thread only hands the description over
//!   (its shaders are shared by content, as its pipeline shares them); a
//!   thread of ours keeps it. With `XTP_KEEP_CREATED=0`, a description is
//!   kept only once a pipeline worker made or tried to make the pipeline (a
//!   first draw with it, or a start at creation from the early list), as
//!   before. A subobject without a pointer is kept byte for
//!   byte, the input layout with its semantic names; the root signature is
//!   ours (the translations' bindings define it) and the cached blob is the
//!   driver's. A stream that cannot be kept whole is not kept at all: one
//!   with a subobject of unknown size (the rest of the stream is not read),
//!   a stream output with declarations or view instancing with views, which
//!   point at data that is not taken in. The precook gives the subobjects to
//!   `host::GraphicsState::read` again, so that the state, its description,
//!   the host pipeline's name in the library and the pipeline's early key
//!   are the session's, as this build reads them.
//! - Kept once per description, by its own key: the BLAKE3 of its shaders'
//!   containers and its stream as kept. Two descriptions that this build
//!   reads as one state are both kept, since a later build may tell them
//!   apart; the precook makes each state once.
//! - Not kept per translator: a description is the title's, the same for
//!   every translator, and a file per translator would be empty exactly
//!   when the precook needs it, after a change. The precook skips what no
//!   longer translates.
//! - Compute pipelines are not kept: the precook makes every compute
//!   program of the pack from its container alone.
//! - The file, [`FILE`]: versioned ([`MAGIC`]) and append-only, a list of
//!   entries: a container, by its BLAKE3, or a pipeline, by its key, each
//!   once. An entry is written whole by one call and carries its length and
//!   the start of its digest. Our threads write it, never the title's,
//!   with the file locked against other processes for each write, at its
//!   end as it is then: a session first takes in what another one appended
//!   meanwhile. An entry that does not read back (its digest) is skipped,
//!   and a pipeline whose containers do not is kept again when a session
//!   makes it; the part of an entry a run left when it ended while writing
//!   is cut off.
//! - Versions: version 1 kept the part of a stream before a subobject it
//!   could not keep as if it were the whole stream, so that its entries may
//!   describe another pipeline than the title's; version 2 keeps whole
//!   streams only. A file of version 1 is started again, empty (by the
//!   first session that keeps a pipeline, or by `xtp precook`); a file of a
//!   later version is left as it is, and nothing is kept in it.
//! - Room: the file stops growing at [`MAX_BYTES`]. Past [`COMPACT_AT`],
//!   `xtp precook` rewrites it ([`compact`]): entries that do not read back,
//!   repeats and containers no pipeline uses go, and then the oldest
//!   pipelines until it is under [`COMPACT_TO`]. A pipeline that goes is
//!   kept again when a session makes it. A file that another process has
//!   open (a session that started anyway) is not rewritten: the session
//!   would go on writing into the file it opened, which the rewrite
//!   replaces. A session opens the file under the lock that the rewrite
//!   holds until the new file is in place, so that it never opens the one
//!   being replaced; one that cannot open the file while it is rewritten
//!   tries again [`RETRY`] later.
//! - A file that cannot be opened (its folder, say) is tried again
//!   [`RETRY`] later, and a full one is looked at again as often.

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_INPUT_CLASSIFICATION, D3D12_INPUT_ELEMENT_DESC, D3D12_INPUT_LAYOUT_DESC,
    D3D12_STREAM_OUTPUT_DESC, D3D12_VIEW_INSTANCING_DESC,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::PCSTR;

use super::Shader;
use super::host::GraphicsState;

/// The file, in the pipeline cache's folder.
pub(crate) const FILE: &str = "recorded-pipelines.bin";
/// The file's first bytes: its format, then its version (2; see the
/// module).
const MAGIC: &[u8; 8] = b"XTPREC\x00\x02";

/// What a file's first bytes say it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    /// This build's.
    Ours,
    /// An earlier version's, which this build starts again.
    Earlier,
    /// A later version's, or no file of ours: left as it is.
    Other,
}

/// The version of a file whose first bytes are `head` (all of them, or its
/// first [`MAGIC`]`.len()`).
fn version(head: &[u8]) -> Version {
    let n = MAGIC.len() - 1;
    match head.get(..MAGIC.len()) {
        Some(h) if h == MAGIC => Version::Ours,
        Some(h) if h[..n] == MAGIC[..n] && h[n] >= 1 && h[n] < MAGIC[n] => Version::Earlier,
        _ => Version::Other,
    }
}

/// Windows' "the file is open in another process in a way that excludes
/// this one" (`ERROR_SHARING_VIOLATION`).
const SHARING_VIOLATION: i32 = 32;
/// The share mode that lets no other handle read or write the file while
/// it is open, but lets it be renamed over (`FILE_SHARE_DELETE`).
const SHARE_DELETE_ONLY: u32 = 4;
/// The file stops growing at this size (a session's pipelines take a few
/// megabytes, their containers some ten).
const MAX_BYTES: u64 = 1 << 30;
/// The precook rewrites a file past this size, down to [`COMPACT_TO`].
pub(crate) const COMPACT_AT: u64 = MAX_BYTES / 4 * 3;
pub(crate) const COMPACT_TO: u64 = MAX_BYTES / 2;
/// A file that could not be opened is tried again this much later.
const RETRY: Duration = Duration::from_secs(30);
/// An entry's payload is at most this long (a container takes a few
/// hundred KiB at most); a longer one is damage.
const MAX_ENTRY: u64 = 64 << 20;

/// Entry tags: a shader container, a pipeline.
const CONTAINER: u8 = b'S';
const PIPELINE: u8 = b'P';
/// An entry's head: its tag, its payload's length (u32) and the first
/// [`CHECK`] bytes of the payload's BLAKE3.
const CHECK: usize = 8;
const HEAD: usize = 1 + 4 + CHECK;

/// The stages a kept pipeline's shaders are of, by their number in the
/// file.
const STAGES: [&str; 8] = [
    "vertex",
    "pixel",
    "domain",
    "hull",
    "geometry",
    "compute",
    "amplification",
    "mesh",
];

/// Subobject types with pointers that are checked rather than kept: the
/// input layout (its elements are kept apart), the stream output and view
/// instancing (a stream is kept only when they hold nothing).
const STREAM_OUTPUT: u32 = 7;
const INPUT_LAYOUT: u32 = 12;
const VIEW_INSTANCING: u32 = 22;

/// Whether a subobject of `kind` is kept byte for byte: of a known size
/// (`super::subobject`) and without pointers. Root signature (0), shaders
/// (1 to 6, 23, 24), stream output (7), input layout (12, kept apart),
/// cached blob (19) and view instancing (22) hold pointers.
fn as_is(kind: u32) -> bool {
    matches!(kind, 8..=11 | 13..=18 | 20 | 21)
}

/// Whether byte `i` of the data (after its type) of a subobject of `kind`
/// kept byte for byte is padding: the title leaves whatever was there,
/// which would make one state two descriptions (a session to the world
/// created 14,784 descriptions
/// that the one before had kept with other padding in their blend and
/// depth-stencil, 2026-11-25), so it is kept as zeros. Blend (8): three
/// bytes after each target's write mask, four at the end; depth-stencil
/// (11, 21): two after the stencil masks, and four at the end of the
/// second; sample description (17): four at the end.
fn padding(kind: u32, i: usize) -> bool {
    match kind {
        8 => i >= 328 || (i >= 8 && (i - 8) % 40 >= 37),
        11 => (18..20).contains(&i),
        21 => (18..20).contains(&i) || (56..60).contains(&i),
        17 => (8..12).contains(&i),
        _ => false,
    }
}

/// The graphics subobjects of a title's pipeline stream, as kept: each its
/// type (u8), its length (u16) and its data, in the stream's order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Stream(Box<[u8]>);

/// A stream's subobjects being taken in, at the title's creation call
/// ([`Capture::take`]).
#[derive(Debug, Default)]
pub(crate) struct Capture {
    bytes: Vec<u8>,
    /// No subobject was left out (the stream is kept only then).
    whole: bool,
    /// Its shaders, shared by content as the title's pipeline shares them
    /// (`super::shader`), to keep the description at creation
    /// ([`created`]); none when descriptions are kept once made.
    shaders: Vec<Arc<Shader>>,
}

impl Capture {
    pub(crate) fn new() -> Capture {
        Capture {
            whole: true,
            ..Capture::default()
        }
    }

    /// Takes in one subobject of the title's stream: its type `kind`, at
    /// `at`, `size` bytes with its type and padding (`super::subobject`).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind` and `size` bytes.
    pub(crate) unsafe fn take(&mut self, kind: u32, at: *const u8, size: usize) {
        match kind {
            INPUT_LAYOUT => {
                // SAFETY: an input layout subobject holds its description
                // at +8.
                let layout =
                    unsafe { at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned() };
                if layout.pInputElementDescs.is_null() {
                    // `GraphicsState::read` takes nothing in either.
                    return;
                }
                // SAFETY: the title passes this many elements.
                let elements = unsafe {
                    std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    )
                };
                let mut data = (elements.len() as u32).to_le_bytes().to_vec();
                for e in elements {
                    // SAFETY: each element names its semantic with a C
                    // string.
                    let name = unsafe { CStr::from_ptr(e.SemanticName.0.cast()) }.to_bytes();
                    let Ok(len) = u16::try_from(name.len()) else {
                        self.whole = false;
                        return;
                    };
                    data.extend(len.to_le_bytes());
                    data.extend(name);
                    for v in [
                        e.SemanticIndex,
                        e.Format.0 as u32,
                        e.InputSlot,
                        e.AlignedByteOffset,
                        e.InputSlotClass.0 as u32,
                        e.InstanceDataStepRate,
                    ] {
                        data.extend(v.to_le_bytes());
                    }
                }
                self.push(kind, &data);
            }
            STREAM_OUTPUT => {
                // SAFETY: a stream output subobject holds its description
                // at +8.
                let so = unsafe {
                    at.add(8)
                        .cast::<D3D12_STREAM_OUTPUT_DESC>()
                        .read_unaligned()
                };
                if so.NumEntries != 0 || so.NumStrides != 0 {
                    // Its declarations are not taken in.
                    self.whole = false;
                }
            }
            VIEW_INSTANCING => {
                // SAFETY: a view instancing subobject holds its description
                // at +8.
                let views = unsafe {
                    at.add(8)
                        .cast::<D3D12_VIEW_INSTANCING_DESC>()
                        .read_unaligned()
                };
                if views.ViewInstanceCount != 0 {
                    self.whole = false;
                }
            }
            _ if as_is(kind) && size > 4 => {
                // SAFETY: the subobject's data follows its type, `size` in
                // all.
                let mut data = unsafe { std::slice::from_raw_parts(at.add(4), size - 4) }.to_vec();
                for (i, byte) in data.iter_mut().enumerate() {
                    if padding(kind, i) {
                        *byte = 0;
                    }
                }
                self.push(kind, &data);
            }
            // The shaders: kept apart, by their containers.
            _ => {
                if let Some((_, Some(stage))) = super::subobject(kind)
                    && at_creation()
                {
                    // SAFETY: a shader subobject holds its bytecode at +8.
                    let code = unsafe { at.add(8).cast::<super::Bytecode>().read_unaligned() };
                    self.shaders.extend(super::copy(stage, &code));
                }
                // The root signature (ours), the cached blob (the
                // driver's): not kept.
            }
        }
    }

    /// The stream holds a subobject of unknown size, and the rest of it is
    /// not read: it cannot be kept.
    pub(crate) fn unknown(&mut self) {
        self.whole = false;
    }

    fn push(&mut self, kind: u32, data: &[u8]) {
        let (Ok(kind), Ok(len)) = (u8::try_from(kind), u16::try_from(data.len())) else {
            self.whole = false;
            return;
        };
        self.bytes.push(kind);
        self.bytes.extend(len.to_le_bytes());
        self.bytes.extend(data);
    }

    /// The stream as kept; `None` if a subobject could not be kept. At the
    /// title's creation call: the description is handed over to be kept
    /// ([`created`]).
    pub(crate) fn finish(self) -> Option<Stream> {
        let Capture {
            bytes,
            whole,
            shaders,
        } = self;
        let stream = whole.then(|| Stream(bytes.into_boxed_slice()));
        created(shaders, stream.as_ref());
        stream
    }
}

impl Stream {
    /// The stream as kept.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.0
    }

    /// The graphics state the stream describes: each subobject given to
    /// `GraphicsState::read` again, in the stream's order, as the title's
    /// creation call gave it; `None` if one does not decode.
    pub(crate) fn state(&self) -> Option<GraphicsState> {
        let mut g = GraphicsState::default();
        let mut r = Reader(&self.0);
        while !r.0.is_empty() {
            let kind = u32::from(r.u8()?);
            let len = usize::from(r.u16()?);
            let data = r.bytes(len)?;
            if kind == INPUT_LAYOUT {
                let (_names, elements) = input_layout(data)?;
                let mut sub = vec![0u8; 24];
                sub[..4].copy_from_slice(&kind.to_le_bytes());
                let desc = D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: elements.as_ptr(),
                    NumElements: elements.len() as u32,
                };
                // SAFETY: 16 bytes at +8 of a 24-byte buffer.
                unsafe {
                    sub.as_mut_ptr()
                        .add(8)
                        .cast::<D3D12_INPUT_LAYOUT_DESC>()
                        .write_unaligned(desc);
                }
                // SAFETY: a whole input layout subobject, its elements and
                // their names alive while `read` copies them.
                unsafe { g.read(kind, sub.as_ptr()) };
                continue;
            }
            let (size, None) = super::subobject(kind)? else {
                return None;
            };
            if !as_is(kind) || len + 4 != size {
                return None;
            }
            let mut sub = kind.to_le_bytes().to_vec();
            sub.extend(data);
            // SAFETY: a whole subobject of this kind: its type, then its
            // `size - 4` bytes of data.
            unsafe { g.read(kind, sub.as_ptr()) };
        }
        Some(g)
    }
}

/// An input layout as kept: its names, and its elements pointing at them.
fn input_layout(data: &[u8]) -> Option<(Vec<CString>, Vec<D3D12_INPUT_ELEMENT_DESC>)> {
    let mut r = Reader(data);
    let n = r.u32()? as usize;
    // D3D12 takes 32 elements at most.
    if n > 64 {
        return None;
    }
    let mut names = Vec::with_capacity(n);
    let mut fields = Vec::with_capacity(n);
    for _ in 0..n {
        let len = usize::from(r.u16()?);
        names.push(CString::new(r.bytes(len)?).ok()?);
        let mut f = [0u32; 6];
        for v in &mut f {
            *v = r.u32()?;
        }
        fields.push(f);
    }
    if !r.0.is_empty() {
        return None;
    }
    let elements = names
        .iter()
        .zip(&fields)
        .map(|(name, f)| D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(name.as_ptr().cast()),
            SemanticIndex: f[0],
            Format: DXGI_FORMAT(f[1] as i32),
            InputSlot: f[2],
            AlignedByteOffset: f[3],
            InputSlotClass: D3D12_INPUT_CLASSIFICATION(f[4] as i32),
            InstanceDataStepRate: f[5],
        })
        .collect();
    Some((names, elements))
}

/// Little-endian fields read off the front of a byte slice.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if n > self.0.len() {
            return None;
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Some(head)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.bytes(N)?.try_into().ok()
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.array()?))
    }
}

/// An entry of the file: its head, then `payload`.
fn entry(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEAD + payload.len());
    out.push(tag);
    out.extend((payload.len() as u32).to_le_bytes());
    out.extend(&blake3::hash(payload).as_bytes()[..CHECK]);
    out.extend(payload);
    out
}

/// A pipeline's shaders as its entry lists them: stage number, container
/// BLAKE3.
type Listed = Vec<(u8, [u8; 32])>;

/// What a pipeline entry holds after its key: its shaders (their count,
/// then stage number and container BLAKE3 each), then its stream.
fn body(shaders: &[(u8, [u8; 32])], stream: &[u8]) -> Vec<u8> {
    let mut out = vec![shaders.len() as u8];
    for (stage, hash) in shaders {
        out.push(*stage);
        out.extend(hash);
    }
    out.extend(stream);
    out
}

/// The shaders a pipeline entry's `body` lists, and its stream.
fn read_body(body: &[u8]) -> Option<(Listed, &[u8])> {
    let mut r = Reader(body);
    let n = r.u8()?;
    let shaders = (0..n)
        .map(|_| {
            let stage = r.u8()?;
            (usize::from(stage) < STAGES.len()).then_some(())?;
            Some((stage, r.array::<32>()?))
        })
        .collect::<Option<Listed>>()?;
    Some((shaders, r.0))
}

/// A description's key: the start of its entry body's BLAKE3.
fn key_of(body: &[u8]) -> [u8; 16] {
    let mut key = [0u8; 16];
    key.copy_from_slice(&blake3::hash(body).as_bytes()[..16]);
    key
}

/// A pipeline entry's payload: its key, then its body.
fn pipeline_payload(key: &[u8; 16], body: &[u8]) -> Vec<u8> {
    let mut out = key.to_vec();
    out.extend(body);
    out
}

/// What became of a description to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keep {
    Written,
    /// The file holds it already.
    Held,
    /// The file is full ([`MAX_BYTES`]).
    Full,
}

/// The lock of one file across processes: a named mutex, by the file's
/// full path (a lock on the file itself would be `File::lock`, newer than
/// this workspace's Rust). A process that ends holding it leaves it
/// abandoned, which the next one takes.
struct Lock(windows_sys::Win32::Foundation::HANDLE);

// SAFETY: a kernel handle, usable from any thread; each hold takes and
// releases the mutex on one thread ([`Lock::hold`]).
unsafe impl Send for Lock {}

impl Lock {
    /// The lock of the file at `path`, which need not exist yet (a session
    /// opens the file under it).
    fn of(path: &Path) -> std::io::Result<Lock> {
        let full = std::path::absolute(path)?;
        let digest = blake3::hash(full.to_string_lossy().to_lowercase().as_bytes());
        let name: Vec<u16> = format!("Local\\xtp-recorded-pipelines-{}", &digest.to_hex()[..32])
            .encode_utf16()
            .chain([0])
            .collect();
        // SAFETY: a NUL-terminated name; no security attributes.
        let handle = unsafe {
            windows_sys::Win32::System::Threading::CreateMutexW(std::ptr::null(), 0, name.as_ptr())
        };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Lock(handle))
    }

    /// Waits for the lock, held until the guard drops (on this thread).
    fn hold(&self) -> std::io::Result<Held> {
        use windows_sys::Win32::Foundation::{WAIT_ABANDONED, WAIT_OBJECT_0};
        // SAFETY: our live mutex handle.
        match unsafe {
            windows_sys::Win32::System::Threading::WaitForSingleObject(
                self.0,
                windows_sys::Win32::System::Threading::INFINITE,
            )
        } {
            // Abandoned: a process ended holding it, maybe in the middle of
            // a write, which the next look cuts off.
            WAIT_OBJECT_0 | WAIT_ABANDONED => Ok(Held(self.0)),
            _ => Err(std::io::Error::last_os_error()),
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        // SAFETY: our handle, closed once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

/// A held [`Lock`], released when dropped. It must not outlive its lock,
/// nor leave its thread.
struct Held(windows_sys::Win32::Foundation::HANDLE);

impl Drop for Held {
    fn drop(&mut self) {
        // SAFETY: the mutex this thread holds.
        unsafe { windows_sys::Win32::System::Threading::ReleaseMutex(self.0) };
    }
}

/// A process's view of the file, open for writing, and what it holds.
struct Writer {
    file: File,
    lock: Lock,
    /// The end of what it has taken in: other processes may have written
    /// past it (zero before the first look).
    end: u64,
    containers: HashSet<[u8; 32]>,
    pipelines: HashSet<[u8; 16]>,
    /// When the file was found full: nothing more is written until it is
    /// looked at again, [`RETRY`] later.
    full: Option<Instant>,
}

impl Writer {
    /// The file at `path`, made with its folder if there is none, with what
    /// it holds taken in. It is opened under the lock, which a rewrite
    /// holds until the new file is in place ([`compact`]).
    fn open(path: &Path) -> std::io::Result<Writer> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let lock = Lock::of(path)?;
        let held = lock.hold()?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let mut writer = Writer {
            file,
            lock,
            end: 0,
            containers: HashSet::new(),
            pipelines: HashSet::new(),
            full: None,
        };
        let caught = writer.catch_up();
        drop(held);
        caught?;
        Ok(writer)
    }

    /// Runs `work` with the file locked against other processes.
    fn locked<T>(
        &mut self,
        work: impl FnOnce(&mut Writer) -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        let held = self.lock.hold()?;
        let done = work(self);
        drop(held);
        done
    }

    /// Takes in what the file holds past [`Writer::end`] (all of it the
    /// first time, or once it was rewritten), its file locked: the entries
    /// that read back, and the part of an entry a run left at the end cut
    /// off.
    fn catch_up(&mut self) -> std::io::Result<()> {
        let len = self.file.metadata()?.len();
        if self.end == 0 || len < self.end {
            self.containers.clear();
            self.pipelines.clear();
            let mut magic = [0u8; MAGIC.len()];
            let head = len.min(MAGIC.len() as u64) as usize;
            self.file.seek(SeekFrom::Start(0))?;
            self.file.read_exact(&mut magic[..head])?;
            let again = match version(&magic[..head]) {
                Version::Ours => false,
                // New, or its magic cut short.
                Version::Other if magic[..head] == MAGIC[..head] => true,
                Version::Earlier => {
                    tracing::info!(
                        bytes = len,
                        "recorded pipelines: a file of an earlier version, started again"
                    );
                    true
                }
                Version::Other => {
                    return Err(std::io::Error::new(
                        ErrorKind::InvalidData,
                        "another format (another version of XTP): nothing is kept in it",
                    ));
                }
            };
            self.end = MAGIC.len() as u64;
            if again {
                self.file.set_len(0)?;
                self.file.seek(SeekFrom::Start(0))?;
                self.file.write_all(MAGIC)?;
                return Ok(());
            }
        }
        if len == self.end {
            return Ok(());
        }
        self.file.seek(SeekFrom::Start(self.end))?;
        let mut reader = BufReader::new(&self.file);
        let mut at = self.end;
        let mut payload = Vec::new();
        let mut taken = Vec::new();
        while at + HEAD as u64 <= len {
            let mut head = [0u8; HEAD];
            reader.read_exact(&mut head)?;
            let n = u64::from(u32::from_le_bytes([head[1], head[2], head[3], head[4]]));
            let end = at + HEAD as u64 + n;
            if end > len || n > MAX_ENTRY || !matches!(head[0], CONTAINER | PIPELINE) {
                break;
            }
            payload.resize(n as usize, 0);
            reader.read_exact(&mut payload)?;
            if blake3::hash(&payload).as_bytes()[..CHECK] == head[1 + 4..] {
                taken.push((head[0], std::mem::take(&mut payload)));
            }
            at = end;
        }
        drop(reader);
        for (tag, payload) in taken {
            self.take_in(tag, &payload);
        }
        if at < len {
            tracing::info!(
                bytes = len - at,
                "recorded pipelines: the end of an entry left unfinished is cut off"
            );
            self.file.set_len(at)?;
        }
        self.end = at;
        Ok(())
    }

    /// Takes in an entry that reads back: a container, or a pipeline whose
    /// containers the file holds (one whose containers do not read back is
    /// kept again, with them).
    fn take_in(&mut self, tag: u8, payload: &[u8]) {
        match tag {
            CONTAINER => {
                if let Some(hash) = payload.get(..32).and_then(|h| h.try_into().ok()) {
                    self.containers.insert(hash);
                }
            }
            _ => {
                if let Some(body) = payload.get(16..)
                    && let Some((shaders, _)) = read_body(body)
                    && shaders.iter().all(|(_, h)| self.containers.contains(h))
                {
                    self.pipelines.insert(key_of(body));
                }
            }
        }
    }

    /// Keeps the description of key `key` and entry body `body`, of
    /// `shaders` (`listed` as the body lists them), after those of its
    /// containers the file does not hold.
    fn keep(
        &mut self,
        key: [u8; 16],
        shaders: &[Arc<Shader>],
        listed: &[(u8, [u8; 32])],
        body: &[u8],
    ) -> std::io::Result<Keep> {
        if self.pipelines.contains(&key) {
            return Ok(Keep::Held);
        }
        if self.full.is_some_and(|t| t.elapsed() < RETRY) {
            return Ok(Keep::Full);
        }
        self.locked(|w| {
            // What other processes wrote meanwhile, and the end they left.
            w.catch_up()?;
            if w.pipelines.contains(&key) {
                return Ok(Keep::Held);
            }
            // Its containers first: a pipeline entry is only read back with
            // them.
            let mut bytes = Vec::new();
            let mut new = Vec::new();
            for ((_, hash), s) in listed.iter().zip(shaders) {
                if !w.containers.contains(hash) && !new.contains(hash) {
                    let mut payload = hash.to_vec();
                    payload.extend(&s.container);
                    bytes.extend(entry(CONTAINER, &payload));
                    new.push(*hash);
                }
            }
            bytes.extend(entry(PIPELINE, &pipeline_payload(&key, body)));
            if w.end + bytes.len() as u64 > MAX_BYTES {
                if w.full.is_none() {
                    tracing::warn!(
                        bytes = w.end,
                        "recorded pipelines: the file is full; no more pipelines are kept until \
                         `xtp precook` makes room"
                    );
                }
                w.full = Some(Instant::now());
                return Ok(Keep::Full);
            }
            w.full = None;
            w.file.seek(SeekFrom::Start(w.end))?;
            w.file.write_all(&bytes)?;
            w.end += bytes.len() as u64;
            w.containers.extend(new);
            w.pipelines.insert(key);
            Ok(Keep::Written)
        })
    }
}

/// The session's file: open, not yet (or tried again from `retry` on), or
/// of another version.
enum Slot {
    Closed { retry: Option<Instant> },
    Open(Writer),
    Refused,
}

static WRITER: Mutex<Slot> = Mutex::new(Slot::Closed { retry: None });

/// Runs `work` on the session's file, opened (again) if need be; `None`
/// without a pipeline cache folder, if the file cannot be opened now, or if
/// `work` fails, which closes it (it is opened again [`RETRY`] later, and
/// what a failed write left is cut off then).
fn with_writer<T>(work: impl FnOnce(&mut Writer) -> std::io::Result<T>) -> Option<T> {
    let path = super::library::dir()?.join(FILE);
    let mut slot = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    if let Slot::Closed { retry } = &*slot {
        if retry.is_some_and(|t| Instant::now() < t) {
            return None;
        }
        *slot = match Writer::open(&path) {
            Ok(w) => {
                tracing::info!(
                    pipelines = w.pipelines.len(),
                    containers = w.containers.len(),
                    bytes = w.end,
                    "recorded pipelines: kept so far"
                );
                Slot::Open(w)
            }
            Err(e) if e.kind() == ErrorKind::InvalidData => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    "recorded pipelines: this run's pipelines are not kept"
                );
                Slot::Refused
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    path = %path.display(),
                    retry_s = RETRY.as_secs(),
                    "recorded pipelines: no file yet; tried again later"
                );
                Slot::Closed {
                    retry: Some(Instant::now() + RETRY),
                }
            }
        };
    }
    let Slot::Open(w) = &mut *slot else {
        return None;
    };
    match work(w) {
        Ok(done) => Some(done),
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                retry_s = RETRY.as_secs(),
                "recorded pipelines: not written; the file is opened again later"
            );
            *slot = Slot::Closed {
                retry: Some(Instant::now() + RETRY),
            };
            None
        }
    }
}

/// Pipelines kept since the start.
static KEPT: AtomicU64 = AtomicU64::new(0);

/// Pipelines kept since the start (for the library's line).
pub(crate) fn kept() -> u64 {
    KEPT.load(Ordering::Relaxed)
}

/// `shaders` as a pipeline entry lists them: stage number, container
/// BLAKE3; `None` for a stage the file has no number for.
fn listed(shaders: &[Arc<Shader>]) -> Option<Listed> {
    shaders
        .iter()
        .map(|s| {
            let stage = STAGES.iter().position(|&n| n == s.stage)?;
            Some((stage as u8, *blake3::hash(&s.container).as_bytes()))
        })
        .collect()
}

/// A title pipeline of `shaders` with the graphics state `stream` was
/// made, or tried: its description is kept unless it is already, with
/// those of its shaders' containers that are not. Nothing to do when
/// descriptions are kept at creation ([`created`]): this one was.
pub(crate) fn made(shaders: &[Arc<Shader>], stream: &Stream) {
    if at_creation() && keeper().is_some() {
        return;
    }
    // Hashed outside the lock: a container takes up to some 100 KiB.
    let Some(listed) = listed(shaders) else {
        return;
    };
    let body = body(&listed, &stream.0);
    let key = key_of(&body);
    if with_writer(|w| w.keep(key, shaders, &listed, &body)) == Some(Keep::Written) {
        KEPT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether descriptions are kept at the title's creation of a pipeline
/// (`XTP_KEEP_CREATED`, unless `0`), or once a worker made it.
fn at_creation() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XTP_KEEP_CREATED").map_or(true, |v| v.trim() != "0");
        if !on {
            tracing::info!(
                "recorded pipelines: kept once made, not at creation (XTP_KEEP_CREATED=0)"
            );
        }
        on
    })
}

/// Title pipelines with a graphics state created since the start
/// ([`created`]), and of them those whose stream could not be kept whole.
static CREATED: AtomicU64 = AtomicU64::new(0);
static UNKEPT: AtomicU64 = AtomicU64::new(0);
/// The distinct descriptions among those created, as [`Keeper`] tells
/// them apart.
static DISTINCT: AtomicU64 = AtomicU64::new(0);

/// What the title created since the start (for the library's line):
/// pipelines with a graphics state, those not kept whole, and the distinct
/// descriptions among the others.
pub(crate) fn created_counts() -> (u64, u64, u64) {
    (
        CREATED.load(Ordering::Relaxed),
        UNKEPT.load(Ordering::Relaxed),
        DISTINCT.load(Ordering::Relaxed),
    )
}

/// A description the title created, on its way to the file.
struct Created {
    shaders: Vec<Arc<Shader>>,
    stream: Stream,
}

/// At the title's creation of a pipeline of `shaders` with the graphics
/// state `stream` (`None`: not kept whole): counted, and handed to the
/// thread that keeps descriptions ([`keeper`]), unless it is a compute
/// pipeline (the precook makes those from the pack). The title's thread
/// does nothing else.
fn created(shaders: Vec<Arc<Shader>>, stream: Option<&Stream>) {
    if shaders.is_empty() || matches!(&*shaders, [cs] if cs.stage == "compute") {
        return;
    }
    CREATED.fetch_add(1, Ordering::Relaxed);
    let Some(stream) = stream else {
        UNKEPT.fetch_add(1, Ordering::Relaxed);
        return;
    };
    if let Some(keeper) = keeper() {
        let _ = keeper.send(Created {
            shaders,
            stream: stream.clone(),
        });
    }
}

/// The thread that keeps the descriptions the title creates, started on
/// first use; `None` if it could not be started (descriptions are then kept
/// once made, as with `XTP_KEEP_CREATED=0`).
fn keeper() -> Option<&'static mpsc::Sender<Created>> {
    static SENDER: OnceLock<Option<mpsc::Sender<Created>>> = OnceLock::new();
    SENDER
        .get_or_init(|| {
            let (send, receive) = mpsc::channel();
            let started = std::thread::Builder::new()
                .name("xtp-record".into())
                .spawn(move || keep_created(&receive));
            match started {
                Ok(_) => Some(send),
                Err(e) => {
                    tracing::warn!(error = %e, "recorded pipelines: no thread to keep them at creation");
                    None
                }
            }
        })
        .as_ref()
}

/// How often the keeper says how far the title's creations are, while they
/// go on, and how long without one it takes for them to have stopped.
const SAY_EVERY: Duration = Duration::from_secs(10);
const QUIET: Duration = Duration::from_secs(5);

/// The keeper's loop: each description received kept once ([`Keeper`]),
/// and how many the title created said every [`SAY_EVERY`] while it
/// creates them and once it stops ([`QUIET`]).
fn keep_created(receive: &mpsc::Receiver<Created>) {
    let mut keeper = Keeper::default();
    let mut first: Option<Instant> = None;
    let mut said = Instant::now();
    let mut new = false;
    let say = |keeper: &Keeper, first: Option<Instant>, why: &str| {
        let (created, unkept, distinct) = created_counts();
        tracing::info!(
            created,
            unkept,
            distinct,
            repeated = keeper.repeated,
            kept = KEPT.load(Ordering::Relaxed),
            frame = crate::frame::current(),
            seconds = first.map_or(0.0, |t| t.elapsed().as_secs_f64()),
            "title pipelines created ({why})"
        );
    };
    loop {
        match receive.recv_timeout(QUIET) {
            Ok(c) => {
                first.get_or_insert_with(Instant::now);
                new = true;
                keeper.take(c, |key, shaders, listed, body| {
                    with_writer(|w| w.keep(key, shaders, listed, body))
                });
                if said.elapsed() >= SAY_EVERY {
                    say(&keeper, first, "so far");
                    said = Instant::now();
                    new = false;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if new {
                    say(&keeper, first, "none for 5 s");
                    said = Instant::now();
                    new = false;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// What the keeper knows: each shader's container BLAKE3, by the shared
/// shader (hashed once, not once per pipeline), and the descriptions it
/// took already.
#[derive(Default)]
struct Keeper {
    hashes: HashMap<usize, (Arc<Shader>, [u8; 32])>,
    seen: HashSet<[u8; 16]>,
    /// Descriptions created again (the same shaders and stream).
    repeated: u64,
}

impl Keeper {
    /// Takes in a created description: counted once among the distinct
    /// ones, and kept by `keep` (key, shaders, listed, body) the first
    /// time.
    fn take(
        &mut self,
        c: Created,
        keep: impl FnOnce([u8; 16], &[Arc<Shader>], &[(u8, [u8; 32])], &[u8]) -> Option<Keep>,
    ) {
        let listed = c
            .shaders
            .iter()
            .map(|s| {
                let stage = STAGES.iter().position(|&n| n == s.stage)? as u8;
                let (_, hash) = self
                    .hashes
                    .entry(Arc::as_ptr(s) as usize)
                    .or_insert_with(|| (s.clone(), *blake3::hash(&s.container).as_bytes()));
                Some((stage, *hash))
            })
            .collect::<Option<Listed>>();
        let Some(listed) = listed else {
            return;
        };
        let body = body(&listed, &c.stream.0);
        let key = key_of(&body);
        if !self.seen.insert(key) {
            self.repeated += 1;
            return;
        }
        DISTINCT.fetch_add(1, Ordering::Relaxed);
        if keep(key, &c.shaders, &listed, &body) == Some(Keep::Written) {
            KEPT.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A kept pipeline: its shaders (stage, container BLAKE3) and its stream.
pub(crate) struct Kept {
    pub(crate) shaders: Vec<(&'static str, [u8; 32])>,
    pub(crate) stream: Stream,
}

impl Kept {
    /// Its entry body, as written ([`body`]).
    fn body(&self) -> Option<Vec<u8>> {
        let listed = self
            .shaders
            .iter()
            .map(|(stage, hash)| Some((STAGES.iter().position(|s| s == stage)? as u8, *hash)))
            .collect::<Option<Listed>>()?;
        Some(body(&listed, &self.stream.0))
    }
}

/// What the file holds, as the precook reads it.
#[derive(Default)]
pub(crate) struct Contents {
    /// In the file's order: the oldest first.
    pub(crate) pipelines: Vec<Kept>,
    containers: HashMap<[u8; 32], Vec<u8>>,
    /// Entries that do not read back (their digest, or what they hold).
    pub(crate) damaged: usize,
    /// The file's size.
    pub(crate) bytes: u64,
    /// The BLAKE3 of the file's bytes as read, in hexadecimal; `none`
    /// without a file.
    pub(crate) digest: String,
    /// The file is of an earlier version: what it holds is not read, and
    /// the precook starts it again ([`compact`]).
    pub(crate) earlier: bool,
}

impl Contents {
    /// The container whose BLAKE3 is `hash`, if the file holds it.
    pub(crate) fn container(&self, hash: &[u8; 32]) -> Option<&[u8]> {
        self.containers.get(hash).map(Vec::as_slice)
    }
}

/// Reads the file in `dir`: empty if there is none.
pub(crate) fn read(dir: &Path) -> Result<Contents, String> {
    let path = dir.join(FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Ok(Contents {
                digest: "none".into(),
                ..Contents::default()
            });
        }
        Err(e) => return Err(format!("reading {}: {e}", path.display())),
    };
    parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// What the file's `bytes` hold.
fn parse(bytes: &[u8]) -> Result<Contents, String> {
    let mut out = Contents {
        bytes: bytes.len() as u64,
        digest: blake3::hash(bytes).to_hex().to_string(),
        ..Contents::default()
    };
    match version(bytes) {
        Version::Ours => {}
        Version::Earlier => {
            out.earlier = true;
            return Ok(out);
        }
        // Empty, or its magic cut short: as a session leaves a new file.
        Version::Other if bytes == &MAGIC[..bytes.len().min(MAGIC.len())] => return Ok(out),
        Version::Other => return Err("another format (another version of XTP)".into()),
    }
    let mut r = Reader(&bytes[MAGIC.len()..]);
    while r.0.len() >= HEAD {
        let tag = r.u8().unwrap_or_default();
        let n = r.u32().unwrap_or_default() as usize;
        let check = r.array::<CHECK>().unwrap_or_default();
        let Some(payload) = r.bytes(n) else {
            break;
        };
        if blake3::hash(payload).as_bytes()[..CHECK] != check {
            out.damaged += 1;
            continue;
        }
        let mut p = Reader(payload);
        match tag {
            CONTAINER => match p.array::<32>() {
                Some(hash) => {
                    out.containers.insert(hash, p.0.to_vec());
                }
                None => out.damaged += 1,
            },
            PIPELINE => match pipeline(payload) {
                Some(kept) => out.pipelines.push(kept),
                None => out.damaged += 1,
            },
            _ => out.damaged += 1,
        }
    }
    // The part of an entry a run left when it ended while writing.
    if !r.0.is_empty() {
        out.damaged += 1;
    }
    Ok(out)
}

/// A pipeline entry's payload ([`pipeline_payload`]).
fn pipeline(payload: &[u8]) -> Option<Kept> {
    let (shaders, stream) = read_body(payload.get(16..)?)?;
    Some(Kept {
        shaders: shaders
            .into_iter()
            .map(|(stage, hash)| (STAGES[usize::from(stage)], hash))
            .collect(),
        stream: Stream(stream.into()),
    })
}

/// What a rewrite of the file did ([`compact`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Compacted {
    /// The file's size before and after.
    pub(crate) before: u64,
    pub(crate) after: u64,
    /// Descriptions kept, and those that went: repeats, those whose
    /// containers did not read back, then the oldest for room.
    pub(crate) kept: usize,
    pub(crate) dropped: usize,
    /// The file was of an earlier version, started again.
    pub(crate) earlier: bool,
}

/// Rewrites the file in `dir` with what reads back in it, each description
/// once with the containers it uses, newest first until it would pass `to`
/// bytes (the oldest go), in the file's order; a file of an earlier version
/// is started again, empty. Only `xtp precook` does, and only while no
/// other process has the file open: `None` if one has (a session that
/// started anyway would go on writing into the file it opened). The lock
/// is held until the new file is in place, and sessions open the file
/// under it ([`Writer::open`]).
pub(crate) fn compact(dir: &Path, to: u64) -> Result<Option<Compacted>, String> {
    use std::os::windows::fs::OpenOptionsExt;
    let path = dir.join(FILE);
    let failed = |e: std::io::Error| format!("{}: {e}", path.display());
    let lock = Lock::of(&path).map_err(failed)?;
    let held = lock.hold().map_err(failed)?;
    // Alone: another process's handle refuses this one, and this one
    // refuses theirs while it is open.
    let mut file = match OpenOptions::new()
        .read(true)
        .write(true)
        .share_mode(SHARE_DELETE_ONLY)
        .open(&path)
    {
        Ok(file) => file,
        Err(e) if e.raw_os_error() == Some(SHARING_VIOLATION) => return Ok(None),
        Err(e) => return Err(failed(e)),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(failed)?;
    let contents = parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    // Newest first, each description once, while there is room.
    let mut chosen = Vec::new();
    let mut keys = HashSet::new();
    let mut used = HashSet::new();
    let mut size = MAGIC.len() as u64;
    for p in contents.pipelines.iter().rev() {
        let Some(body) = p.body() else {
            continue;
        };
        if !p
            .shaders
            .iter()
            .all(|(_, h)| contents.containers.contains_key(h))
            || !keys.insert(key_of(&body))
        {
            continue;
        }
        let new: Vec<&[u8; 32]> = p
            .shaders
            .iter()
            .map(|(_, h)| h)
            .filter(|h| !used.contains(*h))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        let cost = (HEAD + 16 + body.len()) as u64
            + new
                .iter()
                .map(|h| (HEAD + 32 + contents.containers[*h].len()) as u64)
                .sum::<u64>();
        if size + cost > to {
            break;
        }
        size += cost;
        used.extend(new.into_iter().copied());
        chosen.push((p, body));
    }
    chosen.reverse();
    let mut out = MAGIC.to_vec();
    let mut written = HashSet::new();
    for (p, body) in &chosen {
        for (_, hash) in &p.shaders {
            if written.insert(*hash) {
                let mut payload = hash.to_vec();
                payload.extend(&contents.containers[hash]);
                out.extend(entry(CONTAINER, &payload));
            }
        }
        out.extend(entry(PIPELINE, &pipeline_payload(&key_of(body), body)));
    }
    let tmp = path.with_extension("tmp");
    drop(file);
    std::fs::write(&tmp, &out)
        .and_then(|()| std::fs::rename(&tmp, &path))
        .map_err(failed)?;
    drop(held);
    Ok(Some(Compacted {
        before: bytes.len() as u64,
        after: out.len() as u64,
        kept: chosen.len(),
        dropped: contents.pipelines.len() - chosen.len(),
        earlier: contents.earlier,
    }))
}

/// For tests: keeps a pipeline in the file in `dir` as a session's worker
/// does ([`made`]): whether it was written.
#[cfg(test)]
pub(crate) fn keep_in(dir: &Path, shaders: &[Arc<Shader>], stream: &Stream) -> bool {
    let mut w = Writer::open(&dir.join(FILE)).expect("the file");
    let listed = listed(shaders).expect("known stages");
    let body = body(&listed, &stream.0);
    w.keep(key_of(&body), shaders, &listed, &body)
        .expect("written")
        == Keep::Written
}

/// For tests: the stream of `subobjects`, each its type and the data after
/// it.
#[cfg(test)]
pub(crate) fn stream_of(subobjects: &[(u32, Vec<u8>)]) -> Stream {
    let mut c = Capture::new();
    for (kind, data) in subobjects {
        c.push(*kind, data);
    }
    c.finish().expect("short subobjects")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::S_OK;

    /// A stream as BTP 6 builds it (`super::super::tests`'s layout:
    /// title+0x86f090): root signature, a vertex and a pixel shader, blend,
    /// sample mask, rasterizer, depth-stencil 1, sample description, input
    /// layout (two elements), topology, render target and depth formats.
    /// The blend's and depth's padding hold garbage, as a title's may.
    struct TitleStream {
        bytes: Vec<u8>,
        _vs: Vec<u8>,
        _ps: Vec<u8>,
        _names: [CString; 2],
        _elements: Vec<D3D12_INPUT_ELEMENT_DESC>,
    }

    fn title_stream(salt: u8) -> TitleStream {
        let vs = vec![1, 2, 3, salt];
        let ps = vec![4u8; 5];
        let names = [c"POSITION".to_owned(), c"TEXCOORD".to_owned()];
        let elements = vec![
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(names[0].as_ptr().cast()),
                SemanticIndex: 0,
                Format: DXGI_FORMAT(6), // R32G32B32_FLOAT
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(0),
                InstanceDataStepRate: 0,
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(names[1].as_ptr().cast()),
                SemanticIndex: 1,
                Format: DXGI_FORMAT(34), // R16G16_FLOAT
                InputSlot: 1,
                AlignedByteOffset: u32::MAX,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(1),
                InstanceDataStepRate: 1,
            },
        ];
        let mut s = vec![0xccu8; 0x2b0];
        for (at, kind) in [
            (0x0, 0),
            (0x10, 1),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            words(&mut s, at, &[kind, 0]);
        }
        // No root signature; the vertex and pixel shaders' code, no domain,
        // hull or geometry shader.
        quad(&mut s, 0x8, 0);
        quad(&mut s, 0x18, vs.as_ptr() as u64);
        quad(&mut s, 0x20, vs.len() as u64);
        quad(&mut s, 0x30, ps.as_ptr() as u64);
        quad(&mut s, 0x38, ps.len() as u64);
        for at in [0x48, 0x50, 0x60, 0x68, 0x78, 0x80] {
            quad(&mut s, at, 0);
        }
        // Blend: alpha to coverage `salt`, no independent blend; target 0
        // blends one + one (the other targets' bytes are garbage).
        words(&mut s, 0x8c, &[u32::from(salt), 0]);
        words(&mut s, 0x94, &[1, 0, 2, 2, 1, 2, 1, 1, 4, 0x0f]);
        // Sample mask, then a rasterizer: solid, no culling.
        words(&mut s, 0x1dc, &[u32::MAX]);
        words(&mut s, 0x1e4, &[3, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
        // Depth-stencil 1: depth on, writes, less-equal, no stencil; its
        // masks then garbage padding, the faces, depth bounds on.
        let d = 0x214;
        words(&mut s, d, &[1, 1, 4, 0]);
        s[d + 16..d + 20].copy_from_slice(&[0xff, 0xff, 0xab, 0xcd]);
        words(&mut s, d + 20, &[1, 1, 1, 8, 1, 1, 1, 8, 1]);
        // Sample description: one sample.
        words(&mut s, 0x254, &[1, 0]);
        // Input layout.
        quad(&mut s, 0x268, elements.as_ptr() as u64);
        quad(&mut s, 0x270, elements.len() as u64);
        // Topology type 3 (triangles), one render target of format 28,
        // depth format 45 (D24_UNORM_S8_UINT).
        words(&mut s, 0x27c, &[3]);
        words(&mut s, 0x284, &[28, 0, 0, 0, 0, 0, 0, 0, 1]);
        words(&mut s, 0x2ac, &[45]);
        TitleStream {
            bytes: s,
            _vs: vs,
            _ps: ps,
            _names: names,
            _elements: elements,
        }
    }

    fn words(s: &mut [u8], at: usize, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            s[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    fn quad(s: &mut [u8], at: usize, v: u64) {
        s[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// The title's pipeline from `stream`, through the device method.
    fn create(stream: &TitleStream) -> &'static super::super::Pipeline {
        let desc = super::super::StreamDesc {
            size: stream.bytes.len(),
            stream: stream.bytes.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = super::super::create_stream(
            std::ptr::null(),
            &desc,
            &super::super::PIPELINE_IIDS[0],
            &mut p,
        );
        assert_eq!(hr, S_OK);
        super::super::pipeline(p)
    }

    /// The state rebuilt from what was kept is the one the title's stream
    /// gave: the same description (so the same name in the library) and
    /// the same early key, the input layout's names included.
    #[test]
    fn a_kept_stream_gives_the_titles_state_again() {
        let title = title_stream(0);
        let p = create(&title);
        let made = p.host.graphics.as_deref().unwrap();
        let stream = p.host.stream.as_ref().expect("the stream is kept");
        let again = stream.state().expect("it decodes");
        assert_eq!(again.describe(), made.describe());
        assert!(made.describe().contains("TEXCOORD"));
        let shaders = p.shaders.clone();
        assert_eq!(
            super::super::make::key(&shaders, Some(&again)),
            p.host.key,
            "the same early key"
        );
        // Another blend is another state.
        let other = title_stream(1);
        let q = create(&other);
        assert_ne!(
            q.host.stream.as_ref().unwrap().state().unwrap().describe(),
            made.describe()
        );
    }

    /// A stream whose subobject does not decode gives no state.
    #[test]
    fn damaged_streams_give_no_state() {
        let mut c = Capture::new();
        c.push(9, &[1, 2, 3]); // a sample mask of three bytes
        assert!(c.finish().unwrap().state().is_none());
        let mut c = Capture::new();
        c.push(0, &[0; 12]); // a root signature
        assert!(c.finish().unwrap().state().is_none());
        assert!(Stream(Box::new([12, 4, 0, 9, 0, 0, 0])).state().is_none());
        let mut c = Capture::new();
        c.push(9, &[0; 70_000]);
        assert!(c.finish().is_none(), "too long to keep");
        assert!(Capture::new().finish().unwrap().state().is_some());
    }

    /// A stream is kept only whole: not with a subobject of unknown size
    /// (the rest is not read), a stream output that declares outputs or
    /// view instancing with views; empty ones are their defaults.
    #[test]
    fn streams_that_cannot_be_kept_whole_are_not_kept() {
        let taken = |kind: u32, sub: &[u8]| {
            let mut c = Capture::new();
            // SAFETY: a whole subobject of `sub.len()` bytes.
            unsafe { c.take(kind, sub.as_ptr(), sub.len()) };
            c.finish()
        };
        let mut so = [0u8; 40];
        so[..4].copy_from_slice(&STREAM_OUTPUT.to_le_bytes());
        assert_eq!(taken(STREAM_OUTPUT, &so), Some(Stream(Box::new([]))));
        so[16] = 2; // two declarations
        assert_eq!(taken(STREAM_OUTPUT, &so), None);
        let mut views = [0u8; 32];
        views[..4].copy_from_slice(&VIEW_INSTANCING.to_le_bytes());
        assert!(taken(VIEW_INSTANCING, &views).is_some());
        views[8] = 2; // two views
        assert_eq!(taken(VIEW_INSTANCING, &views), None);
        let mut c = Capture::new();
        c.push(9, &[0xff; 4]);
        c.unknown();
        assert!(c.finish().is_none());

        // The title's stream with a console subobject the device does not
        // size: made, not kept.
        let mut title = title_stream(3);
        words(&mut title.bytes, 0x2a8, &[30, 0]);
        let p = create(&title);
        assert!(p.host.graphics.is_some());
        assert!(p.host.stream.is_none(), "not kept: its end was not read");
    }

    /// A description's key is its own: the same shaders and stream give the
    /// same key, any other byte another.
    #[test]
    fn keys_are_the_descriptions_own() {
        let shaders = [(0u8, [1u8; 32]), (1, [2; 32])];
        let a = body(&shaders, &[9, 4, 0, 1, 2, 3, 4]);
        assert_eq!(key_of(&a), key_of(&a.clone()));
        assert_ne!(key_of(&a), key_of(&body(&shaders, &[9, 4, 0, 1, 2, 3, 5])));
        assert_ne!(
            key_of(&a),
            key_of(&body(&shaders[..1], &[9, 4, 0, 1, 2, 3, 4]))
        );
        assert_eq!(
            read_body(&a),
            Some((shaders.to_vec(), &[9u8, 4, 0, 1, 2, 3, 4][..]))
        );
        assert_eq!(read_body(&[1, 8]), None, "no stage 8");
    }

    fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
        super::super::shader(stage, bytes)
    }

    /// `w` keeps `shaders` with `stream` as a session's worker does.
    fn keep(w: &mut Writer, shaders: &[Arc<Shader>], stream: &Stream) -> Keep {
        let listed = listed(shaders).unwrap();
        let body = body(&listed, &stream.0);
        w.keep(key_of(&body), shaders, &listed, &body).unwrap()
    }

    /// The stream of `title`'s pipeline.
    fn kept_stream(title: &TitleStream) -> Stream {
        create(title).host.stream.clone().unwrap()
    }

    /// What a session keeps, a precook reads back: each description once
    /// (by its key, also across runs), each container once, in a file a
    /// run left half written as well; another version's file is left
    /// alone.
    #[test]
    fn pipelines_are_kept_once_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(7));
        let (vs, ps) = (
            shader("vertex", b"record vs"),
            shader("pixel", b"record ps"),
        );
        let ps2 = shader("pixel", b"record ps 2");
        {
            let mut w = Writer::open(&path).unwrap();
            assert_eq!(
                keep(&mut w, &[vs.clone(), ps.clone()], &stream),
                Keep::Written
            );
            assert_eq!(keep(&mut w, &[vs.clone(), ps.clone()], &stream), Keep::Held);
            assert_eq!(
                keep(&mut w, &[vs.clone(), ps2.clone()], &stream),
                Keep::Written
            );
        }
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.damaged), (2, 0));
        assert_eq!(read.containers.len(), 3, "the vertex shader once");
        let first = &read.pipelines[0];
        assert_eq!(first.shaders[0].0, "vertex");
        assert_eq!(read.container(&first.shaders[1].1), Some(&b"record ps"[..]));
        assert_eq!(first.stream, stream);

        // A run that ended while writing: its half entry is cut off when
        // the next run opens the file, which knows what it holds.
        let whole = std::fs::metadata(&path).unwrap().len();
        let mut half = entry(PIPELINE, &pipeline_payload(&[9; 16], &body(&[], &stream.0)));
        half.truncate(half.len() / 2);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&half)
            .unwrap();
        assert_eq!(super::read(dir.path()).unwrap().damaged, 1);
        {
            let mut w = Writer::open(&path).unwrap();
            assert_eq!(w.end, whole, "cut off");
            assert_eq!(std::fs::metadata(&path).unwrap().len(), whole);
            assert_eq!((w.pipelines.len(), w.containers.len()), (2, 3));
            assert_eq!(
                keep(&mut w, &[vs.clone(), ps2.clone()], &stream),
                Keep::Held,
                "kept by the run before"
            );
            assert_eq!(
                keep(&mut w, std::slice::from_ref(&ps2), &stream),
                Keep::Written
            );
        }
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.damaged), (3, 0));
        assert_eq!(read.containers.len(), 3);

        // A damaged entry is skipped; the others still read.
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        let parsed = parse(&bytes).unwrap();
        assert_eq!((parsed.pipelines.len(), parsed.damaged), (2, 1));

        // A later version's file, or no file of ours.
        for other in [&b"XTPREC\x00\x03 and more"[..], b"something else"] {
            std::fs::write(&path, other).unwrap();
            assert_eq!(
                Writer::open(&path).err().map(|e| e.kind()),
                Some(ErrorKind::InvalidData)
            );
            assert!(super::read(dir.path()).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), other, "left as it is");
        }
        // No file: nothing kept.
        let empty = tempfile::tempdir().unwrap();
        let none = super::read(empty.path()).unwrap();
        assert_eq!((none.pipelines.len(), none.digest.as_str()), (0, "none"));
    }

    /// Version 1 kept the part of a stream before a subobject it could not
    /// keep as the whole stream: its descriptions are not read, and a file
    /// of version 1 is started again, by a session or by the precook.
    #[test]
    fn a_file_of_version_1_is_started_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(13));
        let shaders = [shader("vertex", b"v1 vs"), shader("pixel", b"v1 ps")];
        assert_eq!(
            keep(&mut Writer::open(&path).unwrap(), &shaders, &stream),
            Keep::Written
        );
        let mut v1 = std::fs::read(&path).unwrap();
        v1[MAGIC.len() - 1] = 1;
        std::fs::write(&path, &v1).unwrap();
        let read = super::read(dir.path()).unwrap();
        assert!(read.earlier);
        assert_eq!(read.pipelines.len(), 0, "not read");
        assert_eq!(read.digest, blake3::hash(&v1).to_hex().to_string());

        // A session starts it again, and keeps what it makes.
        let mut w = Writer::open(&path).unwrap();
        assert_eq!(w.pipelines.len(), 0);
        assert_eq!(keep(&mut w, &shaders, &stream), Keep::Written);
        drop(w);
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.earlier, read.pipelines.len()), (false, 1));

        // The precook starts it again too.
        std::fs::write(&path, &v1).unwrap();
        let c = compact(dir.path(), u64::MAX).unwrap().unwrap();
        assert!(c.earlier);
        assert_eq!((c.kept, c.after), (0, MAGIC.len() as u64));
        assert_eq!(std::fs::read(&path).unwrap(), MAGIC);
    }

    /// A rewrite leaves a file that another process has open as it is (it
    /// would go on writing into the file it opened); once that one is
    /// closed, the rewrite is made, and a session opened after it keeps
    /// into the new file.
    #[test]
    fn a_file_another_process_has_open_is_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(14));
        let a = [shader("vertex", b"open vs"), shader("pixel", b"open ps a")];
        let b = [shader("vertex", b"open vs"), shader("pixel", b"open ps b")];
        let mut session = Writer::open(&path).unwrap();
        assert_eq!(keep(&mut session, &a, &stream), Keep::Written);
        let before = std::fs::read(&path).unwrap();
        assert_eq!(compact(dir.path(), 0).unwrap(), None, "open in a session");
        assert_eq!(std::fs::read(&path).unwrap(), before, "left as it is");
        drop(session);
        let c = compact(dir.path(), 0).unwrap().expect("rewritten");
        assert_eq!((c.kept, c.dropped), (0, 1));
        let mut session = Writer::open(&path).unwrap();
        assert_eq!(keep(&mut session, &b, &stream), Keep::Written);
        assert_eq!(super::read(dir.path()).unwrap().pipelines.len(), 1);
    }

    /// The first session of a fresh cache makes the folder the file goes
    /// in: its first pipelines are kept.
    #[test]
    fn a_fresh_cache_gets_its_folder() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("cache").join("pipelines");
        let mut w = Writer::open(&folder.join(FILE)).unwrap();
        let stream = kept_stream(&title_stream(8));
        let shaders = [shader("vertex", b"fresh vs"), shader("pixel", b"fresh ps")];
        assert_eq!(keep(&mut w, &shaders, &stream), Keep::Written);
        assert_eq!(super::read(&folder).unwrap().pipelines.len(), 1);
    }

    /// Two descriptions this build reads as one state (they differ in a
    /// node mask, which it does not read) are both kept: a later build may
    /// tell them apart.
    #[test]
    fn variants_this_build_reads_as_one_are_both_kept() {
        let title = title_stream(9);
        let stream = kept_stream(&title);
        let mut bytes = stream.0.to_vec();
        bytes.extend([18, 4, 0, 1, 0, 0, 0]); // node mask 1
        let other = Stream(bytes.into_boxed_slice());
        assert_eq!(
            other.state().unwrap().describe(),
            stream.state().unwrap().describe(),
            "one state for this build"
        );
        let dir = tempfile::tempdir().unwrap();
        let mut w = Writer::open(&dir.path().join(FILE)).unwrap();
        let shaders = [
            shader("vertex", b"variant vs"),
            shader("pixel", b"variant ps"),
        ];
        assert_eq!(keep(&mut w, &shaders, &stream), Keep::Written);
        assert_eq!(keep(&mut w, &shaders, &other), Keep::Written);
        let read = super::read(dir.path()).unwrap();
        assert_eq!(read.pipelines.len(), 2);
        assert_eq!(read.pipelines[1].stream, other);
    }

    /// Two processes keeping pipelines in one file: each writes at the end
    /// the other left, and knows what the other kept.
    #[test]
    fn processes_sharing_the_file_lose_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(10));
        let shaders: Vec<_> = (0..3u8)
            .map(|i| [shader("vertex", b"shared vs"), shader("pixel", &[b'p', i])])
            .collect();
        let mut a = Writer::open(&path).unwrap();
        let mut b = Writer::open(&path).unwrap();
        assert_eq!(keep(&mut a, &shaders[0], &stream), Keep::Written);
        assert_eq!(keep(&mut b, &shaders[1], &stream), Keep::Written);
        assert_eq!(keep(&mut a, &shaders[2], &stream), Keep::Written);
        assert_eq!(keep(&mut b, &shaders[0], &stream), Keep::Held, "a's");
        assert_eq!(keep(&mut b, &shaders[2], &stream), Keep::Held, "a's too");
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.damaged), (3, 0));
        assert_eq!(read.containers.len(), 4, "the vertex shader once");
    }

    /// A pipeline whose container does not read back any more is kept
    /// again, with it, when a session makes it.
    #[test]
    fn a_pipeline_whose_container_is_damaged_is_kept_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(11));
        let shaders = [
            shader("vertex", b"damaged vs"),
            shader("pixel", b"damaged ps"),
        ];
        assert_eq!(
            keep(&mut Writer::open(&path).unwrap(), &shaders, &stream),
            Keep::Written
        );
        // The first container's last byte.
        let mut bytes = std::fs::read(&path).unwrap();
        let at = MAGIC.len() + HEAD + 32 + b"damaged vs".len() - 1;
        bytes[at] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();
        let mut w = Writer::open(&path).unwrap();
        assert_eq!((w.pipelines.len(), w.containers.len()), (0, 1));
        assert_eq!(keep(&mut w, &shaders, &stream), Keep::Written);
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.damaged), (2, 1));
        assert!(
            read.pipelines[1]
                .shaders
                .iter()
                .all(|(_, h)| read.container(h).is_some())
        );
    }

    /// Two streams that differ only in the padding of their blend and
    /// depth-stencil are kept as one.
    #[test]
    fn padding_is_kept_as_zeros() {
        let a = title_stream(18);
        let mut b = title_stream(18);
        // The padding after target 6's write mask, and the depth-stencil's.
        b.bytes[0x88 + 4 + 8 + 40 * 6 + 37] = 0x5a;
        b.bytes[0x214 + 18] = 0x77;
        assert_ne!(a.bytes, b.bytes);
        let (sa, sb) = (kept_stream(&a), kept_stream(&b));
        assert_eq!(sa, sb);
        assert_eq!(
            sa.state().unwrap().describe(),
            create(&a).host.graphics.as_deref().unwrap().describe()
        );
    }

    /// At creation, a stream's shaders are taken in as the title's pipeline
    /// shares them (the same shared containers), in the stream's order.
    #[test]
    fn a_streams_shaders_are_taken_in_at_creation() {
        let title = title_stream(15);
        let p = create(&title);
        let mut c = Capture::new();
        for at in [0x10, 0x28, 0x40] {
            // SAFETY: the vertex, pixel and (empty) domain subobjects of a
            // live stream.
            unsafe { c.take(title.bytes[at] as u32, title.bytes.as_ptr().add(at), 24) };
        }
        assert_eq!(c.shaders.len(), 2, "no domain shader");
        assert!(
            c.shaders
                .iter()
                .zip(&p.shaders)
                .all(|(a, b)| Arc::ptr_eq(a, b))
        );
        assert_eq!(c.shaders[0].stage, "vertex");
    }

    /// The keeper keeps each description the title creates once, however
    /// often it is created, and hashes each shared container once.
    #[test]
    fn the_keeper_keeps_each_created_description_once() {
        let dir = tempfile::tempdir().unwrap();
        let mut w = Writer::open(&dir.path().join(FILE)).unwrap();
        let stream = kept_stream(&title_stream(16));
        let other = kept_stream(&title_stream(17));
        let vs = shader("vertex", b"created vs");
        let ps = shader("pixel", b"created ps");
        let mut keeper = Keeper::default();
        let mut kept = Vec::new();
        for s in [&stream, &stream, &other, &stream] {
            let c = Created {
                shaders: vec![vs.clone(), ps.clone()],
                stream: s.clone(),
            };
            keeper.take(c, |key, shaders, listed, body| {
                let done = w.keep(key, shaders, listed, body).unwrap();
                kept.push(done);
                Some(done)
            });
        }
        assert_eq!(kept, [Keep::Written, Keep::Written], "each once");
        assert_eq!((keeper.repeated, keeper.seen.len()), (2, 2));
        assert_eq!(keeper.hashes.len(), 2, "each container hashed once");
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.containers.len()), (2, 2));
        assert_eq!(read.pipelines[1].stream, other);
        // Kept as a worker keeps it after making it: the same key.
        assert_eq!(keep(&mut w, &[vs, ps], &stream), Keep::Held);
    }

    /// A rewrite keeps each description that reads back once, newest first
    /// within its room, with the containers they use; a session's file
    /// opened after it knows what it holds.
    #[test]
    fn a_rewrite_keeps_the_newest_within_its_room() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE);
        let stream = kept_stream(&title_stream(12));
        let pixel = |i: u8| shader("pixel", &[i; 1000]);
        let vs = shader("vertex", b"rewrite vs");
        {
            let mut w = Writer::open(&path).unwrap();
            for i in 0..4 {
                assert_eq!(
                    keep(&mut w, &[vs.clone(), pixel(i)], &stream),
                    Keep::Written
                );
            }
        }
        // A repeat, as a former build's key kept it.
        let listed = listed(&[vs.clone(), pixel(3)]).unwrap();
        let repeat = entry(
            PIPELINE,
            &pipeline_payload(&[7; 16], &body(&listed, &stream.0)),
        );
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(&repeat)
            .unwrap();
        let before = std::fs::metadata(&path).unwrap().len();
        // Room for about two.
        let c = compact(dir.path(), 4000).unwrap().expect("no session");
        assert_eq!((c.before, c.kept, c.dropped), (before, 2, 3));
        assert_eq!(c.after, std::fs::metadata(&path).unwrap().len());
        let read = super::read(dir.path()).unwrap();
        assert_eq!((read.pipelines.len(), read.damaged), (2, 0));
        assert_eq!(
            read.containers.len(),
            3,
            "the vertex shader and two pixel ones"
        );
        let first = read.pipelines[0].shaders[1].1;
        assert_eq!(
            read.container(&first),
            Some(&[2u8; 1000][..]),
            "the oldest kept"
        );
        let mut w = Writer::open(&path).unwrap();
        assert_eq!(w.pipelines.len(), 2);
        assert_eq!(keep(&mut w, &[vs.clone(), pixel(3)], &stream), Keep::Held);
        assert_eq!(
            keep(&mut w, &[vs.clone(), pixel(0)], &stream),
            Keep::Written,
            "went"
        );
    }
}
