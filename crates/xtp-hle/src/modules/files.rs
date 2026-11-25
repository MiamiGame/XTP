//! Virtual drives over host folders: on the console, title storage is a
//! drive of its own, and titles build paths on that assumption.
//!
//! Every title import that takes a path is bound here. A path on a virtual
//! drive (`X:\..`, `X:/..`, `X:`, `\\?\X:\..`) is rewritten to the host
//! folder behind it and handed to the host's own function; any other path
//! goes to the host unchanged. A rewritten narrow path goes to the host's
//! wide twin, so host folders outside the ANSI code page work; never to our
//! wide wrapper, which would rewrite the host path a second time when its
//! drive letter is mounted too. Letters are chosen among those the host
//! does not use, so no host drive is hidden.
//!
//! The package drive, `G:` (also named `GAME:`), is read-only: its folder
//! is the person's game dump, and every measurement on it must stay valid.
//! What the title changes there goes to a copy-on-write [`overlay`] in the
//! title's data folder: opening a package file for writing copies that
//! file there and opens the copy, reads take the overlay first, deletions
//! leave whiteouts, listings merge both layers. Host paths that lead into
//! the package folder (built from the title's module path, or relative
//! ones after a change of folder) take the same way.
//!
//! A console-looking path that no mount serves (`Q:\..` where the host has
//! no `Q:`, or a named prefix such as `NAME:\..`) is logged once.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

mod overlay;

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, OsString, c_char, c_void};
use std::io;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_ALREADY_EXISTS, ERROR_FILE_EXISTS, ERROR_FILE_NOT_FOUND,
    ERROR_INVALID_NAME, ERROR_NO_MORE_FILES, ERROR_PATH_NOT_FOUND, GetLastError,
    INVALID_HANDLE_VALUE, SetLastError,
};
use windows_sys::Win32::Globalization::{
    CP_ACP, CSTR_EQUAL, CompareStringOrdinal, MultiByteToWideChar,
};
use windows_sys::Win32::Storage::FileSystem::{
    GetFullPathNameW, GetLogicalDrives, WIN32_FIND_DATAW,
};

use self::overlay::{Must, Overlay, under};
use crate::{HleExport, HleModule, hle_export};

/// The host's own function: what the import would have been bound to.
macro_rules! host {
    ($dll:literal, $name:literal: $ty:ty) => {{
        static ADDRESS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let address = *ADDRESS.get_or_init(|| {
            crate::host::function($dll, $name)
                .unwrap_or_else(|| panic!("the host lacks {}!{}", $dll, $name))
        });
        // SAFETY: the host exports this function with this signature.
        unsafe { std::mem::transmute::<usize, $ty>(address) }
    }};
}
pub(crate) use host;

/// A mounted drive.
struct Drive {
    /// Upper-case letter.
    letter: u16,
    /// Host root (UTF-16, no trailing separator, no NUL).
    root: Vec<u16>,
    /// Set for a read-only package drive.
    package: Option<Arc<Package>>,
}

static DRIVES: Mutex<Vec<Drive>> = Mutex::new(Vec::new());

fn drives() -> MutexGuard<'static, Vec<Drive>> {
    DRIVES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The console's package drive.
const PACKAGE: char = 'G';

/// The console's named prefixes and the drive each names.
const PREFIXES: &[(&str, char)] = &[("GAME", PACKAGE)];

/// Gives the host folder `root` a drive letter (the same one if it already
/// has one): the highest letter neither the host nor another mount uses.
pub fn mount(root: &Path) -> Option<char> {
    mount_free(root, None)
}

fn mount_free(root: &Path, overlay: Option<OverlayAt>) -> Option<char> {
    let wide = root_wide(root);
    let mut drives = drives();
    let package = overlay.is_some();
    if let Some(drive) = drives
        .iter()
        .find(|d| d.root == wide && d.package.is_some() == package)
    {
        return char::from_u32(drive.letter.into());
    }
    // SAFETY: no arguments.
    let host = unsafe { GetLogicalDrives() };
    let letter = (b'D'..=b'Z').rev().find(|&l| {
        host >> (l - b'A') & 1 == 0 && !drives.iter().any(|d| d.letter == u16::from(l))
    })?;
    tracing::info!(drive = %char::from(letter), root = %root.display(), read_only = package, "virtual drive");
    let package = overlay.map(|at| Arc::new(Package::new(&wide, at)));
    drives.push(Drive {
        letter: letter.into(),
        root: wide,
        package,
    });
    Some(letter.into())
}

/// Gives the host folder `root` a drive letter the console fixes (titles
/// open their package as `G:\..` and temporary files as `T:\..`). A host
/// drive of that letter is hidden from the title. `G:` is read-only, its
/// overlay the `overlay` folder beside the title's storage folder, in its
/// data folder (see [`mount_package_at`]).
pub fn mount_at(letter: char, root: &Path) {
    let package = letter
        .eq_ignore_ascii_case(&PACKAGE)
        .then_some(OverlayAt::TitleData);
    mount_fixed(letter, root, package);
}

/// Mounts the package folder `root` read-only at `letter`: what the title
/// changes there goes to an overlay in the folder `overlay`.
pub fn mount_package_at(letter: char, root: &Path, overlay: &Path) {
    let overlay = std::path::absolute(overlay).unwrap_or_else(|_| overlay.to_owned());
    mount_fixed(letter, root, Some(OverlayAt::Folder(overlay)));
}

fn mount_fixed(letter: char, root: &Path, overlay: Option<OverlayAt>) {
    let letter = letter.to_ascii_uppercase();
    let Some(index) = u32::from(letter)
        .checked_sub(u32::from('A'))
        .filter(|i| *i < 26)
    else {
        return;
    };
    // SAFETY: no arguments.
    let hides_host_drive = unsafe { GetLogicalDrives() } >> index & 1 != 0;
    let read_only = overlay.is_some();
    tracing::info!(drive = %letter, root = %root.display(), hides_host_drive, read_only, "virtual drive");
    let wide = root_wide(root);
    let package = overlay.map(|at| Arc::new(Package::new(&wide, at)));
    let mut drives = drives();
    drives.retain(|d| u32::from(d.letter) != u32::from(letter));
    drives.push(Drive {
        letter: letter as u16,
        root: wide,
        package,
    });
}

/// `root` as UTF-16, absolute, without trailing separators (a drive root
/// keeps its own).
fn root_wide(root: &Path) -> Vec<u16> {
    let root = std::path::absolute(root).unwrap_or_else(|_| root.to_owned());
    let mut wide: Vec<u16> = root.as_os_str().encode_wide().collect();
    while wide.len() > 3 && wide.last().is_some_and(|c| SEPARATORS.contains(c)) {
        wide.pop();
    }
    wide
}

const SEPARATORS: [u16; 2] = [b'\\' as u16, b'/' as u16];
const VERBATIM: [u16; 4] = [b'\\' as u16, b'\\' as u16, b'?' as u16, b'\\' as u16];

/// Host paths from this length on take the `\\?\` form, which lifts the
/// host's MAX_PATH limit (248 characters for a directory, 260 for a file).
const LONG: usize = 248;

/// `root` and the title's `tail` as a `\\?\` path. That form turns off the
/// host's own clean-up, so `/` becomes `\` and `.` and `..` are resolved
/// here (`..` stops at the root).
fn verbatim(root: &[u16], tail: &[u16]) -> Vec<u16> {
    let backslash = u16::from(b'\\');
    let mut out = if root.starts_with(&VERBATIM) {
        root.to_vec()
    } else if let Some(unc) = root.strip_prefix(&[backslash; 2][..]) {
        let mut out = VERBATIM.to_vec();
        out.extend("UNC\\".encode_utf16());
        out.extend_from_slice(unc);
        out
    } else {
        [&VERBATIM[..], root].concat()
    };
    while out.last() == Some(&backslash) {
        out.pop();
    }
    let dot = u16::from(b'.');
    let mut parts: Vec<&[u16]> = Vec::new();
    for part in tail.split(|c| SEPARATORS.contains(c)) {
        match part {
            [] => {}
            [d] if *d == dot => {}
            [d, e] if [*d, *e] == [dot; 2] => drop(parts.pop()),
            _ => parts.push(part),
        }
    }
    for part in parts {
        out.push(backslash);
        out.extend_from_slice(part);
    }
    out
}

/// The host path, NUL-terminated, of the title's `tail` on the drive over
/// `root`; `prefixed` when the title wrote it in the `\\?\` form.
fn on_drive(root: &[u16], prefixed: bool, tail: &[u16]) -> Vec<u16> {
    if !prefixed && root.len() + tail.len() >= LONG {
        let mut out = verbatim(root, tail);
        out.push(0);
        return out;
    }
    let mut out = if prefixed {
        VERBATIM.to_vec()
    } else {
        Vec::new()
    };
    out.extend_from_slice(root);
    // `X:` and `X:name` are relative to the root.
    if !tail.first().is_some_and(|c| SEPARATORS.contains(c)) {
        out.push(b'\\'.into());
    }
    out.extend(tail.iter().map(|&c| {
        if prefixed && c == u16::from(b'/') {
            b'\\'.into()
        } else {
            c
        }
    }));
    out.push(0);
    out
}

/// A clean host path (absolute, no `.` or `..`) as the host takes it:
/// NUL-terminated, in the `\\?\` form from [`LONG`] on.
fn host_wide(path: &Path) -> Vec<u16> {
    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let mut out = if wide.len() >= LONG && !wide.starts_with(&VERBATIM) {
        verbatim(&wide, &[])
    } else {
        wide
    };
    out.push(0);
    out
}

// Packages.

/// Where a package's overlay lives.
enum OverlayAt {
    Folder(PathBuf),
    /// The `overlay` folder beside the title's storage folder, in its data
    /// folder: known once the title is.
    TitleData,
}

/// The overlay folder of a title whose storage folder is `storage`.
fn beside_storage(storage: &Path) -> Option<PathBuf> {
    (!storage.as_os_str().is_empty()).then(|| storage.with_file_name("overlay"))
}

/// A read-only package folder and the overlay its changes go to.
struct Package {
    lower: PathBuf,
    at: OverlayAt,
    overlay: Mutex<Option<Arc<Overlay>>>,
    /// An overlay that does not open was reported.
    reported: AtomicBool,
}

/// What `rel` names in a package: a name with `:` (a stream, or a drive
/// in the middle of a path) is refused.
fn valid(rel: &str) -> Result<(), u32> {
    if rel.contains(':') {
        Err(ERROR_INVALID_NAME)
    } else {
        Ok(())
    }
}

/// The Win32 error behind an overlay failure.
fn code(error: io::Error) -> u32 {
    error
        .raw_os_error()
        .map_or(ERROR_ACCESS_DENIED, |c| c as u32)
}

impl Package {
    fn new(root: &[u16], at: OverlayAt) -> Package {
        Package {
            lower: PathBuf::from(OsString::from_wide(root)),
            at,
            overlay: Mutex::new(None),
            reported: AtomicBool::new(false),
        }
    }

    fn slot(&self) -> MutexGuard<'_, Option<Arc<Overlay>>> {
        self.overlay.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The overlay, opened on first use. None before the title's data
    /// folder is known, or when it does not open: the package then only
    /// reads, and every change fails.
    fn overlay(&self) -> Option<Arc<Overlay>> {
        let mut slot = self.slot();
        if let Some(overlay) = &*slot {
            return Some(overlay.clone());
        }
        let folder = match &self.at {
            OverlayAt::Folder(folder) => folder.clone(),
            OverlayAt::TitleData => beside_storage(&crate::runtime::title().storage)?,
        };
        match Overlay::open(&self.lower, &folder) {
            Ok(overlay) => {
                tracing::info!(package = %self.lower.display(), overlay = %folder.display(), "package overlay");
                let overlay = Arc::new(overlay);
                *slot = Some(overlay.clone());
                Some(overlay)
            }
            Err(error) => {
                if !self.reported.swap(true, Ordering::Relaxed) {
                    tracing::error!(overlay = %folder.display(), %error, "the package overlay does not open: changes to the package fail");
                }
                None
            }
        }
    }

    /// The overlay if it is open already.
    fn opened(&self) -> Option<Arc<Overlay>> {
        self.slot().clone()
    }

    /// The host path to read `rel` at, the overlay's first; Err when a
    /// whiteout hides it or it is not there.
    fn read(&self, rel: &str) -> Result<Vec<u16>, u32> {
        valid(rel)?;
        match self.overlay() {
            Some(overlay) => overlay.read_path(rel).map(|p| host_wide(&p)).map_err(code),
            None => Ok(host_wide(&under(&self.lower, rel))),
        }
    }

    /// The host path for `rel` whether it is there or not: where it reads,
    /// else where the host finds nothing (the upper path of a hidden one).
    fn read_or_absent(&self, rel: &str) -> Vec<u16> {
        self.read(rel).unwrap_or_else(|_| match self.opened() {
            Some(overlay) if overlay.hidden(rel) => host_wide(&overlay.upper_path(rel)),
            _ => host_wide(&under(&self.lower, rel)),
        })
    }

    /// Runs `op`, a change of `rel`, on the overlay.
    fn with<T>(&self, rel: &str, op: impl FnOnce(&Overlay) -> io::Result<T>) -> Result<T, u32> {
        valid(rel)?;
        let overlay = self.overlay().ok_or(ERROR_ACCESS_DENIED)?;
        op(&overlay).map_err(code)
    }

    /// Makes `rel` ready for a change; the host's call then goes to the
    /// returned path.
    fn prepare(&self, rel: &str, must: Must, keep: bool) -> Result<(Vec<u16>, Done), u32> {
        valid(rel)?;
        let overlay = self.overlay().ok_or(ERROR_ACCESS_DENIED)?;
        let prepared = overlay.prepare(rel, must, keep).map_err(code)?;
        let path = host_wide(&prepared.path);
        let done = Done {
            overlay,
            rel: rel.to_owned(),
            existed: prepared.existed,
            present: prepared.present,
        };
        Ok((path, done))
    }

    /// Makes `rel` ready to receive a file (moved, copied or linked there);
    /// `replace` lets it take another file's place.
    fn destination(&self, rel: &str, replace: bool) -> Result<(Vec<u16>, Done), u32> {
        valid(rel)?;
        let overlay = self.overlay().ok_or(ERROR_ACCESS_DENIED)?;
        if let Some(entry) = overlay.entry(rel) {
            if !replace {
                return Err(ERROR_ALREADY_EXISTS);
            }
            if entry.is_dir() {
                return Err(ERROR_ACCESS_DENIED);
            }
        }
        self.prepare(rel, Must::Either, false)
    }

    /// Plans an open of `rel`.
    fn open(&self, rel: &str, intent: Intent) -> Result<Open, u32> {
        valid(rel)?;
        let Some(overlay) = self.overlay() else {
            // Without its overlay the package only reads.
            let path = under(&self.lower, rel);
            let creates = intent.must != Must::Exist && std::fs::symlink_metadata(&path).is_err();
            if intent.changes || creates {
                return Err(ERROR_ACCESS_DENIED);
            }
            return Ok(Open::read(host_wide(&path)));
        };
        if !intent.changes && (intent.must == Must::Exist || overlay.exists(rel)) {
            let path = overlay.read_path(rel).map_err(code)?;
            return Ok(Open::read(host_wide(&path)));
        }
        let (host, done) = self.prepare(rel, intent.must, intent.keep)?;
        Ok(Open {
            host: Some(host),
            package: Some(Opened::Change(done)),
        })
    }

    /// Moves `rel` out of the package to the host path `dest`: its upper
    /// entry moves, a package file is copied; the package entry is hidden
    /// after.
    fn move_out(&self, rel: &str, dest: *const u16, flags: u32) -> Result<(), u32> {
        let mv = host!("kernel32.dll", "MoveFileExW": extern "system" fn(*const u16, *const u16, u32) -> i32);
        let copy = host!("kernel32.dll", "CopyFileW": extern "system" fn(*const u16, *const u16, i32) -> i32);
        valid(rel)?;
        let overlay = self.overlay().ok_or(ERROR_ACCESS_DENIED)?;
        let Some(entry) = overlay.entry(rel) else {
            return Err(code(overlay.missing(rel)));
        };
        if entry.is_dir() {
            overlay.movable(rel).map_err(code)?;
        }
        if overlay.in_upper(rel).is_some() {
            let from = host_wide(&overlay.upper_path(rel));
            if mv(from.as_ptr(), dest, flags | MOVEFILE_COPY_ALLOWED) == 0 {
                return Err(last_error());
            }
        } else {
            let from = host_wide(&overlay.lower_path(rel));
            let fail_if_exists = i32::from(flags & MOVEFILE_REPLACE_EXISTING == 0);
            if copy(from.as_ptr(), dest, fail_if_exists) == 0 {
                // As a move words it.
                return Err(match last_error() {
                    ERROR_FILE_EXISTS => ERROR_ALREADY_EXISTS,
                    e => e,
                });
            }
        }
        overlay.hide(rel).map_err(code)
    }

    /// Plans a search of the folder `dir` for `pattern` (its last
    /// component, wildcards and all); `list` is the host's search.
    fn search(
        &self,
        dir: &str,
        pattern: &[u16],
        list: impl Fn(&[u16]) -> Vec<WIN32_FIND_DATAW>,
    ) -> Search {
        if let Err(e) = valid(dir) {
            return Search::Fail(e);
        }
        let at = |root: &Path| {
            let mut path = under(root, dir).into_os_string();
            path.push("\\");
            path.push(OsString::from_wide(pattern));
            host_wide(Path::new(&path))
        };
        let Some(overlay) = self.overlay() else {
            return Search::Host(at(&self.lower));
        };
        if overlay.hidden(dir) {
            return Search::Fail(ERROR_PATH_NOT_FOUND);
        }
        if !overlay.needs_merge(dir) {
            return Search::Host(at(&self.lower));
        }
        if !overlay.is_dir(dir) {
            return Search::Fail(ERROR_PATH_NOT_FOUND);
        }
        let upper = match overlay.in_upper(dir) {
            Some(m) if m.is_dir() => list(&at(overlay.upper_root())),
            _ => Vec::new(),
        };
        let lower = match overlay.in_lower(dir) {
            Some(m) if m.is_dir() => list(&at(&self.lower)),
            _ => Vec::new(),
        };
        let entries = overlay.merge(dir, upper, lower, found_name);
        if entries.is_empty() {
            return Search::Fail(ERROR_FILE_NOT_FOUND);
        }
        Search::Merged(entries)
    }
}

/// A change made ready in a package, recorded once the host's call
/// succeeds.
struct Done {
    overlay: Arc<Overlay>,
    rel: String,
    /// The entry was there before.
    existed: bool,
    /// Its upper entry is there; if not, the host's call creates it.
    present: bool,
}

impl Done {
    /// Records the change; a file deleted on close leaves its package
    /// entry hidden instead.
    fn finish(self, delete_on_close: bool) {
        let recorded = if delete_on_close {
            self.overlay.hide(&self.rel)
        } else {
            self.overlay.commit(&self.rel)
        };
        if let Err(error) = recorded {
            tracing::warn!(file = self.rel, %error, "package overlay: recording a change failed");
        }
    }
}

fn finish(done: Option<Done>) {
    if let Some(done) = done {
        done.finish(false);
    }
}

// Title paths.

/// What a title path names.
enum Target {
    /// Not ours: the title's path goes to the host as it is.
    Host,
    /// On a writable virtual drive: the host path, NUL-terminated.
    Drive(Vec<u16>),
    /// In a package: the package and the path below its root.
    Package(Arc<Package>, String),
}

/// What the title path `path` (without its NUL) names. Every title path
/// is traced at debug level.
fn classify(path: &[u16]) -> Target {
    let target = target_of(path);
    tracing::debug!(
        path = %String::from_utf16_lossy(path),
        host = match &target {
            Target::Host => None,
            Target::Drive(host) => Some(String::from_utf16_lossy(&host[..host.len() - 1])),
            Target::Package(_, rel) => Some(format!("package \\{rel}")),
        },
        "title path"
    );
    target
}

fn ascii_upper(c: u16) -> u16 {
    if (u16::from(b'a')..=u16::from(b'z')).contains(&c) {
        c - 32
    } else {
        c
    }
}

fn target_of(path: &[u16]) -> Target {
    let (prefixed, rest) = match path.strip_prefix(&VERBATIM[..]) {
        Some(rest) => (true, rest),
        None => (false, path),
    };
    if let Some((name, tail)) = console_prefix(rest) {
        let letter = match name {
            [letter] => ascii_upper(*letter),
            _ => PREFIXES
                .iter()
                .find(|(prefix, _)| {
                    prefix
                        .encode_utf16()
                        .eq(name.iter().map(|&c| ascii_upper(c)))
                })
                .map_or(0, |&(_, letter)| letter as u16),
        };
        let mounted = drives()
            .iter()
            .find(|d| d.letter == letter)
            .map(|d| match &d.package {
                Some(package) => Target::Package(package.clone(), relative(tail)),
                None => Target::Drive(on_drive(&d.root, prefixed, tail)),
            });
        if let Some(target) = mounted {
            return target;
        }
        let host_drive = name.len() == 1
            // SAFETY: no arguments.
            && unsafe { GetLogicalDrives() } >> (letter - u16::from(b'A')) & 1 != 0;
        if !host_drive {
            unmapped(path);
        }
    }
    in_package(path)
}

/// A console-looking prefix: the name before the path's first `:` (a drive
/// letter, or a word such as `GAME`), and what follows the `:`.
fn console_prefix(path: &[u16]) -> Option<(&[u16], &[u16])> {
    let colon = path.iter().position(|&c| c == u16::from(b':'))?;
    let (name, tail) = (&path[..colon], &path[colon + 1..]);
    let alphabetic = |c: &u16| u8::try_from(*c).is_ok_and(|c| c.is_ascii_alphabetic());
    let word = |c: &u16| u8::try_from(*c).is_ok_and(|c| c.is_ascii_alphanumeric() || c == b'_');
    (name.first().is_some_and(alphabetic) && name.iter().all(word)).then_some((name, tail))
}

/// The path below a drive's root: components joined by `\`, without empty
/// ones and `.`, `..` stopping at the root, and trailing dots and spaces
/// dropped as the host drops them.
fn relative(tail: &[u16]) -> String {
    let text = String::from_utf16_lossy(tail);
    let mut parts: Vec<&str> = Vec::new();
    for part in text.split(['\\', '/']) {
        match part {
            "" | "." => {}
            ".." => drop(parts.pop()),
            _ => {
                let part = part.trim_end_matches(['.', ' ']);
                if !part.is_empty() {
                    parts.push(part);
                }
            }
        }
    }
    parts.join("\\")
}

/// A host path (absolute, or relative to the current folder) that leads
/// into a package folder or its overlay is the package's.
fn in_package(path: &[u16]) -> Target {
    if path.is_empty() {
        return Target::Host;
    }
    let packages: Vec<Arc<Package>> = drives().iter().filter_map(|d| d.package.clone()).collect();
    if packages.is_empty() {
        return Target::Host;
    }
    let Some(full) = full_path(path) else {
        return Target::Host;
    };
    let full = full.strip_prefix(&VERBATIM[..]).unwrap_or(&full[..]);
    for package in packages {
        let upper = package.opened().map(|o| wide_of(o.upper_root()));
        for root in [Some(wide_of(&package.lower)), upper].into_iter().flatten() {
            if let Some(tail) = below(full, &root) {
                let rel = relative(tail);
                return Target::Package(package, rel);
            }
        }
    }
    Target::Host
}

fn wide_of(path: &Path) -> Vec<u16> {
    path.as_os_str().encode_wide().collect()
}

/// What follows `root` in `full`, names compared as the host compares
/// them; None if `full` is not inside `root`.
fn below<'a>(full: &'a [u16], root: &[u16]) -> Option<&'a [u16]> {
    let root = root.strip_prefix(&VERBATIM[..]).unwrap_or(root);
    let head = full.get(..root.len())?;
    let tail = &full[root.len()..];
    let len = i32::try_from(root.len()).ok()?;
    // SAFETY: two strings of `len` units.
    let same = unsafe { CompareStringOrdinal(head.as_ptr(), len, root.as_ptr(), len, 1) };
    let boundary = root.last().is_some_and(|c| SEPARATORS.contains(c))
        || tail.first().is_none_or(|c| SEPARATORS.contains(c));
    (same == CSTR_EQUAL && boundary).then_some(tail)
}

/// The host's absolute form of `path`.
fn full_path(path: &[u16]) -> Option<Vec<u16>> {
    let path: Vec<u16> = path.iter().copied().chain([0]).collect();
    let mut buffer = vec![0u16; 300];
    loop {
        let size = u32::try_from(buffer.len()).ok()?;
        // SAFETY: a NUL-terminated path and a buffer of `size` units.
        let n = unsafe {
            GetFullPathNameW(
                path.as_ptr(),
                size,
                buffer.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        } as usize;
        if n == 0 {
            return None;
        }
        if n < buffer.len() {
            buffer.truncate(n);
            return Some(buffer);
        }
        buffer.resize(n, 0);
    }
}

/// Console-looking paths no mount serves, each logged once.
static UNMAPPED: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// Logs `path`, a console-looking path that no mount serves, the first
/// time; true then.
fn unmapped(path: &[u16]) -> bool {
    let text = String::from_utf16_lossy(path);
    let mut seen = UNMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() >= 1000 || !seen.insert(text.to_lowercase()) {
        return false;
    }
    tracing::warn!(path = %text, "unmapped console path");
    true
}

/// The host path, NUL-terminated, for a title path (without its NUL) that
/// a mount serves: for a package path, where it reads.
pub(crate) fn translate(path: &[u16]) -> Option<Vec<u16>> {
    match classify(path) {
        Target::Host => None,
        Target::Drive(host) => Some(host),
        Target::Package(package, rel) => Some(package.read_or_absent(&rel)),
    }
}

/// A title's wide path without its NUL; empty for none.
fn units<'a>(path: *const u16) -> &'a [u16] {
    if path.is_null() {
        return &[];
    }
    // SAFETY: the title passes a NUL-terminated path.
    unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        std::slice::from_raw_parts(path, len)
    }
}

/// A title's narrow path, decoded as the host decodes it (ANSI code page),
/// without its NUL; None for none.
fn decode(path: *const c_char) -> Option<Vec<u16>> {
    if path.is_null() {
        return None;
    }
    // SAFETY: the title passes a NUL-terminated path.
    let bytes = unsafe { CStr::from_ptr(path) }.to_bytes();
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    let len = i32::try_from(bytes.len()).ok()?;
    // SAFETY: sizing call, then a call with room for `n` units.
    unsafe {
        let n = MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), len, std::ptr::null_mut(), 0);
        let mut wide = vec![0u16; usize::try_from(n).ok().filter(|&n| n > 0)?];
        MultiByteToWideChar(CP_ACP, 0, bytes.as_ptr(), len, wide.as_mut_ptr(), n);
        Some(wide)
    }
}

/// An ASCII mode string (`"rb"`, `"w, ccs=UTF-8"`) for a wide CRT call.
fn widen(mode: *const c_char) -> Vec<u16> {
    if mode.is_null() {
        return vec![0];
    }
    // SAFETY: a NUL-terminated mode string.
    let bytes = unsafe { CStr::from_ptr(mode) }.to_bytes();
    bytes.iter().map(|&b| u16::from(b)).chain([0]).collect()
}

/// The path the host gets: ours if there is one, else the title's.
fn pick(title: *const u16, host: &Option<Vec<u16>>) -> *const u16 {
    host.as_ref().map_or(title, |h| h.as_ptr())
}

/// Where a title's read of `path` goes: `Ok(None)` the title's own path,
/// `Ok(Some)` our host path; `Err` the error the title gets.
fn reading(path: &[u16]) -> Result<Option<Vec<u16>>, u32> {
    match classify(path) {
        Target::Host => Ok(None),
        Target::Drive(host) => Ok(Some(host)),
        Target::Package(package, rel) => package.read(&rel).map(Some),
    }
}

/// Where a title's change of `path` that keeps its content goes
/// (attributes, times): as [`reading`], with the change to record once the
/// host's call succeeds.
fn changing(path: &[u16]) -> Result<(Option<Vec<u16>>, Option<Done>), u32> {
    match classify(path) {
        Target::Host => Ok((None, None)),
        Target::Drive(host) => Ok((Some(host), None)),
        Target::Package(package, rel) => package
            .prepare(&rel, Must::Exist, true)
            .map(|(host, done)| (Some(host), Some(done))),
    }
}

/// Where a title path receives a file (moved, copied or linked there);
/// `replace` lets it take another file's place.
fn receiving(target: Target, replace: bool) -> Result<(Option<Vec<u16>>, Option<Done>), u32> {
    match target {
        Target::Host => Ok((None, None)),
        Target::Drive(host) => Ok((Some(host), None)),
        Target::Package(package, rel) => package
            .destination(&rel, replace)
            .map(|(host, done)| (Some(host), Some(done))),
    }
}

// Opens.

const GENERIC_WRITE: u32 = 0x4000_0000;
const OPEN_EXISTING: u32 = 3;
const CREATE_NEW: u32 = 1;
const CREATE_ALWAYS: u32 = 2;
const OPEN_ALWAYS: u32 = 4;
const TRUNCATE_EXISTING: u32 = 5;
const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;

/// What an open asks of the file.
#[derive(Clone, Copy, Debug)]
struct Intent {
    /// It may change: written, truncated, created, its attributes set.
    changes: bool,
    /// Its content stays; a package file is copied up first.
    keep: bool,
    must: Must,
    delete_on_close: bool,
}

impl Intent {
    /// `CreateFile*`.
    fn win32(access: u32, disposition: u32, flags: u32) -> Intent {
        // GENERIC_WRITE and _ALL, MAXIMUM_ALLOWED, FILE_WRITE_DATA,
        // FILE_APPEND_DATA, FILE_WRITE_EA, FILE_DELETE_CHILD,
        // FILE_WRITE_ATTRIBUTES, DELETE, WRITE_DAC, WRITE_OWNER.
        const WRITES: u32 = GENERIC_WRITE
            | 0x1000_0000
            | 0x0200_0000
            | 0x2
            | 0x4
            | 0x10
            | 0x40
            | 0x100
            | 0x1_0000
            | 0x4_0000
            | 0x8_0000;
        let delete_on_close = flags & FILE_FLAG_DELETE_ON_CLOSE != 0;
        let (must, keep) = match disposition {
            CREATE_NEW => (Must::NotExist, false),
            CREATE_ALWAYS => (Must::Either, false),
            OPEN_ALWAYS => (Must::Either, true),
            TRUNCATE_EXISTING => (Must::Exist, false),
            _ => (Must::Exist, true),
        };
        Intent {
            changes: access & WRITES != 0 || !keep || delete_on_close,
            keep: keep || delete_on_close,
            must,
            delete_on_close,
        }
    }

    /// `fopen` and its kin: `"r"`, `"w+b"`, `"a, ccs=UTF-8"`.
    fn mode(mode: &[u16]) -> Intent {
        let mode = &mode[..mode
            .iter()
            .position(|&c| c == u16::from(b','))
            .unwrap_or(mode.len())];
        let has = |c: u8| mode.contains(&u16::from(c));
        let delete_on_close = has(b'D');
        let first = mode.iter().find(|&&c| c != u16::from(b' '));
        let (changes, must, keep) = match first.and_then(|&c| u8::try_from(c).ok()) {
            Some(b'w') if has(b'x') => (true, Must::NotExist, false),
            Some(b'w') => (true, Must::Either, false),
            Some(b'a') => (true, Must::Either, true),
            Some(b'r') => (has(b'+'), Must::Exist, true),
            // The host refuses it.
            _ => (false, Must::Exist, true),
        };
        Intent {
            changes: changes || delete_on_close,
            keep,
            must,
            delete_on_close,
        }
    }

    /// `_open`: `_O_RDONLY`, `_O_CREAT | _O_TRUNC | _O_WRONLY`, ...
    fn oflag(oflag: i32) -> Intent {
        const ACCESS: i32 = 3;
        const TEMPORARY: i32 = 0x40;
        const CREAT: i32 = 0x100;
        const TRUNC: i32 = 0x200;
        const EXCL: i32 = 0x400;
        let must = match (oflag & CREAT != 0, oflag & EXCL != 0) {
            (true, true) => Must::NotExist,
            (true, false) => Must::Either,
            _ => Must::Exist,
        };
        let truncates = oflag & TRUNC != 0;
        let delete_on_close = oflag & TEMPORARY != 0;
        Intent {
            changes: oflag & ACCESS != 0 || truncates || must == Must::NotExist || delete_on_close,
            // The host's open needs the file there.
            keep: !truncates || must == Must::Exist,
            must,
            delete_on_close,
        }
    }

    /// `_Fiopen`: an `ios_base::openmode`.
    fn ios(mode: i32) -> Intent {
        const IN: i32 = 1;
        const OUT: i32 = 2;
        const APP: i32 = 8;
        const TRUNC: i32 = 0x10;
        const NOCREATE: i32 = 0x40;
        const NOREPLACE: i32 = 0x80;
        let changes = mode & (OUT | APP | TRUNC) != 0;
        let updates = mode & IN != 0 && mode & (TRUNC | APP) == 0;
        let must = if !changes || mode & NOCREATE != 0 || updates {
            Must::Exist
        } else if mode & NOREPLACE != 0 {
            Must::NotExist
        } else {
            Must::Either
        };
        Intent {
            changes,
            keep: mode & APP != 0 || mode & (IN | TRUNC) == IN || must == Must::Exist,
            must,
            delete_on_close: false,
        }
    }
}

/// How a package open goes.
enum Opened {
    /// It only reads.
    Read,
    Change(Done),
}

/// A title's open, planned.
struct Open {
    /// The host path; None: the title's own.
    host: Option<Vec<u16>>,
    package: Option<Opened>,
}

/// Where a title's open of `path` goes.
fn opening(path: &[u16], intent: Intent) -> Result<Open, u32> {
    match classify(path) {
        Target::Host => Ok(Open {
            host: None,
            package: None,
        }),
        Target::Drive(host) => Ok(Open {
            host: Some(host),
            package: None,
        }),
        Target::Package(package, rel) => package.open(&rel, intent),
    }
}

impl Open {
    fn read(host: Vec<u16>) -> Open {
        Open {
            host: Some(host),
            package: Some(Opened::Read),
        }
    }

    /// The disposition the host gets for the title's `asked` (Win32 opens),
    /// and the last error a success leaves, as the title's disposition
    /// would have left it.
    fn disposition(&self, asked: u32) -> (u32, Option<u32>) {
        match &self.package {
            // A read can never create.
            Some(Opened::Read) => (
                OPEN_EXISTING,
                (asked == OPEN_ALWAYS).then_some(ERROR_ALREADY_EXISTS),
            ),
            Some(Opened::Change(done)) if !done.present => match asked {
                TRUNCATE_EXISTING => (CREATE_ALWAYS, None),
                CREATE_ALWAYS if done.existed => (CREATE_ALWAYS, Some(ERROR_ALREADY_EXISTS)),
                _ => (asked, None),
            },
            _ => (asked, None),
        }
    }

    /// Runs the host's open: `call` gets the path and the disposition (0
    /// for a CRT open), `ok` tells whether it succeeded.
    fn run<T>(
        self,
        title: *const u16,
        asked: u32,
        intent: Intent,
        call: impl FnOnce(*const u16, u32) -> T,
        ok: impl FnOnce(&T) -> bool,
    ) -> T {
        let (disposition, then) = self.disposition(asked);
        let result = call(pick(title, &self.host), disposition);
        if ok(&result) {
            if let Some(error) = then {
                // SAFETY: sets this thread's last error.
                unsafe { SetLastError(error) };
            }
            if let Some(Opened::Change(done)) = self.package {
                done.finish(intent.delete_on_close);
            }
        }
        result
    }
}

// Errors.

/// Fails a Win32 call: sets the last error and returns `value`.
fn fail<T>(code: u32, value: T) -> T {
    // SAFETY: sets this thread's last error.
    unsafe { SetLastError(code) };
    value
}

fn last_error() -> u32 {
    // SAFETY: reads this thread's last error.
    unsafe { GetLastError() }
}

fn win32(result: Result<(), u32>) -> i32 {
    match result {
        Ok(()) => 1,
        Err(e) => fail(e, 0),
    }
}

/// `errno` for a Win32 error, as the CRT maps it.
fn errno_of(code: u32) -> i32 {
    const ENOENT: i32 = 2;
    const EACCES: i32 = 13;
    const EEXIST: i32 = 17;
    const EXDEV: i32 = 18;
    const EINVAL: i32 = 22;
    const ENOSPC: i32 = 28;
    const ENOTEMPTY: i32 = 41;
    match code {
        2 | 3 | 15 | 18 | 53 | 67 | 161 | 206 => ENOENT,
        5 | 16 | 19..=36 | 65 | 82 | 83 | 108 | 132 | 158 | 167 => EACCES,
        17 => EXDEV,
        80 | 183 => EEXIST,
        112 => ENOSPC,
        145 => ENOTEMPTY,
        _ => EINVAL,
    }
}

/// Sets the CRT's `errno` and `_doserrno` for a Win32 error; returns the
/// `errno`.
fn set_errno(code: u32) -> i32 {
    let errno = errno_of(code);
    let errno_slot = host!("ucrtbase.dll", "_errno": extern "C" fn() -> *mut i32)();
    let dos_slot = host!("ucrtbase.dll", "__doserrno": extern "C" fn() -> *mut u32)();
    // SAFETY: the CRT's slots for this thread.
    unsafe {
        *errno_slot = errno;
        *dos_slot = code;
    }
    errno
}

/// Fails a CRT call: sets `errno` and returns `value`.
fn crt_fail<T>(code: u32, value: T) -> T {
    set_errno(code);
    value
}

/// Fails a CRT `_s` open: no stream, and the `errno` returned.
fn crt_fail_s(out: *mut File, code: u32) -> i32 {
    if !out.is_null() {
        // SAFETY: the title's out pointer.
        unsafe { *out = std::ptr::null_mut() };
    }
    set_errno(code)
}

fn crt(result: Result<(), u32>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(e) => crt_fail(e, -1),
    }
}

fn hresult(code: u32) -> i32 {
    (0x8007_0000 | (code & 0xFFFF)) as i32
}

type Handle = *mut c_void;
type File = *mut c_void;

type CreateFileWFn =
    extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, Handle) -> Handle;

// kernel32 and its file API sets.

pub extern "system" fn CreateFileW(
    path: *const u16,
    access: u32,
    share: u32,
    security: *const c_void,
    disposition: u32,
    flags: u32,
    template: Handle,
) -> Handle {
    let create = host!("kernel32.dll", "CreateFileW": CreateFileWFn);
    let intent = Intent::win32(access, disposition, flags);
    match opening(units(path), intent) {
        Ok(open) => open.run(
            path,
            disposition,
            intent,
            |p, d| create(p, access, share, security, d, flags, template),
            |h| *h != INVALID_HANDLE_VALUE,
        ),
        Err(e) => fail(e, INVALID_HANDLE_VALUE),
    }
}

pub extern "system" fn CreateFileA(
    path: *const c_char,
    access: u32,
    share: u32,
    security: *const c_void,
    disposition: u32,
    flags: u32,
    template: Handle,
) -> Handle {
    let narrow = host!("kernel32.dll", "CreateFileA": extern "system" fn(*const c_char, u32, u32, *const c_void, u32, u32, Handle) -> Handle);
    let create = host!("kernel32.dll", "CreateFileW": CreateFileWFn);
    let intent = Intent::win32(access, disposition, flags);
    let host_open = || narrow(path, access, share, security, disposition, flags, template);
    let Some(units) = decode(path) else {
        return host_open();
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => host_open(),
        Ok(open) => open.run(
            std::ptr::null(),
            disposition,
            intent,
            |p, d| create(p, access, share, security, d, flags, template),
            |h| *h != INVALID_HANDLE_VALUE,
        ),
        Err(e) => fail(e, INVALID_HANDLE_VALUE),
    }
}

/// Used by Bink.
pub extern "system" fn CreateFile2(
    path: *const u16,
    access: u32,
    share: u32,
    disposition: u32,
    params: *const c_void,
) -> Handle {
    let create = host!("kernel32.dll", "CreateFile2": extern "system" fn(*const u16, u32, u32, u32, *const c_void) -> Handle);
    // CREATEFILE2_EXTENDED_PARAMETERS: dwSize, dwFileAttributes,
    // dwFileFlags, ...
    let flags = if params.is_null() {
        0
    } else {
        // SAFETY: the title's parameters.
        unsafe { params.cast::<u32>().add(2).read_unaligned() }
    };
    let intent = Intent::win32(access, disposition, flags);
    match opening(units(path), intent) {
        Ok(open) => open.run(
            path,
            disposition,
            intent,
            |p, d| create(p, access, share, d, params),
            |h| *h != INVALID_HANDLE_VALUE,
        ),
        Err(e) => fail(e, INVALID_HANDLE_VALUE),
    }
}

pub extern "system" fn DeleteFileA(path: *const c_char) -> i32 {
    let narrow = host!("kernel32.dll", "DeleteFileA": extern "system" fn(*const c_char) -> i32);
    let delete = host!("kernel32.dll", "DeleteFileW": extern "system" fn(*const u16) -> i32);
    match decode(path).map(|units| classify(&units)) {
        None | Some(Target::Host) => narrow(path),
        Some(Target::Drive(host)) => delete(host.as_ptr()),
        Some(Target::Package(package, rel)) => win32(package.with(&rel, |o| o.delete_file(&rel))),
    }
}

pub extern "system" fn DeleteFileW(path: *const u16) -> i32 {
    let delete = host!("kernel32.dll", "DeleteFileW": extern "system" fn(*const u16) -> i32);
    match classify(units(path)) {
        Target::Host => delete(path),
        Target::Drive(host) => delete(host.as_ptr()),
        Target::Package(package, rel) => win32(package.with(&rel, |o| o.delete_file(&rel))),
    }
}

pub extern "system" fn RemoveDirectoryW(path: *const u16) -> i32 {
    let remove = host!("kernel32.dll", "RemoveDirectoryW": extern "system" fn(*const u16) -> i32);
    match classify(units(path)) {
        Target::Host => remove(path),
        Target::Drive(host) => remove(host.as_ptr()),
        Target::Package(package, rel) => win32(package.with(&rel, |o| o.remove_dir(&rel))),
    }
}

/// The security attributes do not reach the overlay.
pub extern "system" fn CreateDirectoryW(path: *const u16, security: *const c_void) -> i32 {
    let create = host!("kernel32.dll", "CreateDirectoryW": extern "system" fn(*const u16, *const c_void) -> i32);
    match classify(units(path)) {
        Target::Host => create(path, security),
        Target::Drive(host) => create(host.as_ptr(), security),
        Target::Package(package, rel) => win32(package.with(&rel, |o| o.create_dir(&rel))),
    }
}

pub extern "system" fn SetFileAttributesW(path: *const u16, attributes: u32) -> i32 {
    let set =
        host!("kernel32.dll", "SetFileAttributesW": extern "system" fn(*const u16, u32) -> i32);
    match changing(units(path)) {
        Ok((host, done)) => {
            let ok = set(pick(path, &host), attributes);
            if ok != 0 {
                finish(done);
            }
            ok
        }
        Err(e) => fail(e, 0),
    }
}

const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

pub extern "system" fn GetFileAttributesA(path: *const c_char) -> u32 {
    let narrow =
        host!("kernel32.dll", "GetFileAttributesA": extern "system" fn(*const c_char) -> u32);
    let get = host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32);
    match decode(path).map(|units| reading(&units)) {
        None | Some(Ok(None)) => narrow(path),
        Some(Ok(Some(host))) => get(host.as_ptr()),
        Some(Err(e)) => fail(e, INVALID_FILE_ATTRIBUTES),
    }
}

pub extern "system" fn GetFileAttributesW(path: *const u16) -> u32 {
    let get = host!("kernel32.dll", "GetFileAttributesW": extern "system" fn(*const u16) -> u32);
    match reading(units(path)) {
        Ok(host) => get(pick(path, &host)),
        Err(e) => fail(e, INVALID_FILE_ATTRIBUTES),
    }
}

pub extern "system" fn GetFileAttributesExW(
    path: *const u16,
    level: i32,
    info: *mut c_void,
) -> i32 {
    let get = host!("kernel32.dll", "GetFileAttributesExW": extern "system" fn(*const u16, i32, *mut c_void) -> i32);
    match reading(units(path)) {
        Ok(host) => get(pick(path, &host), level, info),
        Err(e) => fail(e, 0),
    }
}

pub extern "system" fn GetDiskFreeSpaceExW(
    path: *const u16,
    available: *mut u64,
    total: *mut u64,
    free: *mut u64,
) -> i32 {
    let get = host!("kernel32.dll", "GetDiskFreeSpaceExW": extern "system" fn(*const u16, *mut u64, *mut u64, *mut u64) -> i32);
    match reading(units(path)) {
        Ok(host) => get(pick(path, &host), available, total, free),
        Err(e) => fail(e, 0),
    }
}

/// Wants a volume root: a virtual drive answers for the host volume its
/// folder is on.
pub extern "system" fn GetDiskFreeSpaceW(
    root: *const u16,
    sectors_per_cluster: *mut u32,
    bytes_per_sector: *mut u32,
    free_clusters: *mut u32,
    clusters: *mut u32,
) -> i32 {
    let get = host!("kernel32.dll", "GetDiskFreeSpaceW": extern "system" fn(*const u16, *mut u32, *mut u32, *mut u32, *mut u32) -> i32);
    let mut host = match reading(units(root)) {
        Ok(host) => host,
        Err(e) => return fail(e, 0),
    };
    if let Some(host) = &mut host
        && let [letter, colon, sep, ..] = host[..]
        && colon == u16::from(b':')
        && SEPARATORS.contains(&sep)
    {
        *host = vec![letter, colon, sep, 0];
    }
    get(
        pick(root, &host),
        sectors_per_cluster,
        bytes_per_sector,
        free_clusters,
        clusters,
    )
}

type FindFirstFn =
    extern "system" fn(*const u16, i32, *mut c_void, i32, *mut c_void, u32) -> Handle;

pub extern "system" fn FindFirstFileW(path: *const u16, data: *mut c_void) -> Handle {
    // FindExInfoStandard, FindExSearchNameMatch.
    FindFirstFileExW(path, 0, data, 0, std::ptr::null_mut(), 0)
}

/// A search in a package folder lists both layers, merged, when the overlay
/// holds something there; the title then walks our handle with
/// [`FindNextFileW`].
pub extern "system" fn FindFirstFileExW(
    path: *const u16,
    level: i32,
    data: *mut c_void,
    search: i32,
    filter: *mut c_void,
    flags: u32,
) -> Handle {
    let find = host!("kernel32.dll", "FindFirstFileExW": FindFirstFn);
    let units = units(path);
    let host = match classify(units) {
        Target::Host => None,
        Target::Drive(host) => Some(host),
        Target::Package(whole, rel) => {
            let (folder, pattern) = split_search(units);
            let Target::Package(package, dir) = classify(&folder) else {
                let host = whole.read_or_absent(&rel);
                return find(host.as_ptr(), level, data, search, filter, flags);
            };
            let host_list = |p: &[u16]| list(p, level, search, filter, flags);
            return match package.search(&dir, pattern, host_list) {
                Search::Host(host) => find(host.as_ptr(), level, data, search, filter, flags),
                Search::Merged(entries) => start(entries, data),
                Search::Fail(e) => fail(e, INVALID_HANDLE_VALUE),
            };
        }
    };
    find(pick(path, &host), level, data, search, filter, flags)
}

/// A search path's folder (with its last separator, so a root stays a
/// root) and its last component, the pattern.
fn split_search(path: &[u16]) -> (Vec<u16>, &[u16]) {
    if let Some(at) = path.iter().rposition(|c| SEPARATORS.contains(c)) {
        return (path[..=at].to_vec(), &path[at + 1..]);
    }
    match path.iter().position(|&c| c == u16::from(b':')) {
        Some(colon) => (path[..=colon].to_vec(), &path[colon + 1..]),
        None => (vec![u16::from(b'.')], path),
    }
}

/// A package search, planned.
enum Search {
    /// The host's search, on one folder.
    Host(Vec<u16>),
    /// Both layers' entries, merged.
    Merged(Vec<WIN32_FIND_DATAW>),
    Fail(u32),
}

fn found_name(entry: &WIN32_FIND_DATAW) -> String {
    let name = &entry.cFileName;
    let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
    String::from_utf16_lossy(&name[..len])
}

/// The host's whole listing for a search path (NUL-terminated).
fn list(
    path: &[u16],
    level: i32,
    search: i32,
    filter: *mut c_void,
    flags: u32,
) -> Vec<WIN32_FIND_DATAW> {
    let first = host!("kernel32.dll", "FindFirstFileExW": FindFirstFn);
    let next =
        host!("kernel32.dll", "FindNextFileW": extern "system" fn(Handle, *mut c_void) -> i32);
    let close = host!("kernel32.dll", "FindClose": extern "system" fn(Handle) -> i32);
    let mut data = WIN32_FIND_DATAW::default();
    let handle = first(
        path.as_ptr(),
        level,
        (&raw mut data).cast(),
        search,
        filter,
        flags,
    );
    if handle == INVALID_HANDLE_VALUE {
        return Vec::new();
    }
    let mut out = vec![data];
    while next(handle, (&raw mut data).cast()) != 0 {
        out.push(data);
    }
    close(handle);
    out
}

/// A merged listing the title walks with [`FindNextFileW`].
struct Merged(std::vec::IntoIter<WIN32_FIND_DATAW>);

/// The merged listings, by the handle the title holds.
static SEARCHES: Mutex<BTreeMap<usize, Box<Merged>>> = Mutex::new(BTreeMap::new());

fn searches() -> MutexGuard<'static, BTreeMap<usize, Box<Merged>>> {
    SEARCHES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Hands the title the first entry, and a handle for the others.
fn start(entries: Vec<WIN32_FIND_DATAW>, data: *mut c_void) -> Handle {
    let mut entries = entries.into_iter();
    let Some(first) = entries.next() else {
        return fail(ERROR_FILE_NOT_FOUND, INVALID_HANDLE_VALUE);
    };
    // SAFETY: the title's WIN32_FIND_DATAW.
    unsafe { data.cast::<WIN32_FIND_DATAW>().write_unaligned(first) };
    let merged = Box::new(Merged(entries));
    // The box's address is the handle: no live host handle has it.
    let handle: Handle = (&raw const *merged).cast_mut().cast();
    searches().insert(handle as usize, merged);
    handle
}

pub extern "system" fn FindNextFileW(handle: Handle, data: *mut c_void) -> i32 {
    if let Some(merged) = searches().get_mut(&(handle as usize)) {
        return match merged.0.next() {
            Some(entry) => {
                // SAFETY: the title's WIN32_FIND_DATAW.
                unsafe { data.cast::<WIN32_FIND_DATAW>().write_unaligned(entry) };
                1
            }
            None => fail(ERROR_NO_MORE_FILES, 0),
        };
    }
    host!("kernel32.dll", "FindNextFileW": extern "system" fn(Handle, *mut c_void) -> i32)(
        handle, data,
    )
}

pub extern "system" fn FindClose(handle: Handle) -> i32 {
    if searches().remove(&(handle as usize)).is_some() {
        return 1;
    }
    host!("kernel32.dll", "FindClose": extern "system" fn(Handle) -> i32)(handle)
}

const MOVEFILE_REPLACE_EXISTING: u32 = 1;
const MOVEFILE_COPY_ALLOWED: u32 = 2;
const MOVEFILE_DELAY_UNTIL_REBOOT: u32 = 4;

/// Renames title paths: within a package through the overlay; out of one
/// by moving its upper file or copying the package's; into one as a new
/// upper file.
fn rename(
    from: *const u16,
    source: Target,
    to: *const u16,
    target: Target,
    flags: u32,
) -> Result<(), u32> {
    let mv = host!("kernel32.dll", "MoveFileExW": extern "system" fn(*const u16, *const u16, u32) -> i32);
    let moved = |a: *const u16, b: *const u16, flags: u32| {
        if mv(a, b, flags) != 0 {
            Ok(())
        } else {
            Err(last_error())
        }
    };
    let replace = flags & MOVEFILE_REPLACE_EXISTING != 0;
    let packaged = matches!(source, Target::Package(..)) || matches!(target, Target::Package(..));
    if packaged && (to.is_null() || flags & MOVEFILE_DELAY_UNTIL_REBOOT != 0) {
        // A package entry cannot wait for a restart.
        return Err(ERROR_ACCESS_DENIED);
    }
    if let (Target::Package(a, from_rel), Target::Package(b, to_rel)) = (&source, &target)
        && Arc::ptr_eq(a, b)
    {
        valid(to_rel)?;
        return a.with(from_rel, |o| o.rename(from_rel, to_rel, replace));
    }
    let into_package = matches!(target, Target::Package(..));
    let (dest, done) = receiving(target, replace)?;
    let dest = pick(to, &dest);
    // Into the overlay's folder, which may be on another volume.
    let flags = if into_package {
        flags | MOVEFILE_COPY_ALLOWED
    } else {
        flags
    };
    match source {
        Target::Host => moved(from, dest, flags)?,
        Target::Drive(host) => moved(host.as_ptr(), dest, flags)?,
        Target::Package(package, rel) => package.move_out(&rel, dest, flags)?,
    }
    finish(done);
    Ok(())
}

pub extern "system" fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32 {
    let (source, target) = (classify(units(from)), classify(units(to)));
    win32(rename(from, source, to, target, flags))
}

/// Copies to a package path make a new upper file.
pub extern "system" fn CopyFile2(from: *const u16, to: *const u16, params: *const c_void) -> i32 {
    const COPY_FILE_FAIL_IF_EXISTS: u32 = 1;
    let copy = host!("kernel32.dll", "CopyFile2": extern "system" fn(*const u16, *const u16, *const c_void) -> i32);
    let source = match reading(units(from)) {
        Ok(source) => source,
        Err(e) => return hresult(e),
    };
    // COPYFILE2_EXTENDED_PARAMETERS: dwSize, dwCopyFlags, ...
    let flags = if params.is_null() {
        0
    } else {
        // SAFETY: the title's parameters.
        unsafe { params.cast::<u32>().add(1).read_unaligned() }
    };
    let replace = flags & COPY_FILE_FAIL_IF_EXISTS == 0;
    let (dest, done) = match receiving(classify(units(to)), replace) {
        Ok(dest) => dest,
        // As a copy words it.
        Err(ERROR_ALREADY_EXISTS) => return hresult(ERROR_FILE_EXISTS),
        Err(e) => return hresult(e),
    };
    let result = copy(pick(from, &source), pick(to, &dest), params);
    if result >= 0 {
        finish(done);
    }
    result
}

/// A link to a package file would let writes through it reach the
/// package: it links to the file's upper copy.
pub extern "system" fn CreateHardLinkW(
    new: *const u16,
    existing: *const u16,
    security: *const c_void,
) -> i32 {
    let link = host!("kernel32.dll", "CreateHardLinkW": extern "system" fn(*const u16, *const u16, *const c_void) -> i32);
    let (existing_host, copied) = match changing(units(existing)) {
        Ok(existing) => existing,
        Err(e) => return fail(e, 0),
    };
    finish(copied);
    let (new_host, done) = match receiving(classify(units(new)), false) {
        Ok(new) => new,
        Err(e) => return fail(e, 0),
    };
    let ok = link(
        pick(new, &new_host),
        pick(existing, &existing_host),
        security,
    );
    if ok != 0 {
        finish(done);
    }
    ok
}

/// Following a link into the package could write to it: such links are
/// refused. Other targets, relative ones included, stay as they are.
pub extern "system" fn CreateSymbolicLinkW(link: *const u16, target: *const u16, flags: u32) -> u8 {
    let create = host!("kernel32.dll", "CreateSymbolicLinkW": extern "system" fn(*const u16, *const u16, u32) -> u8);
    let target_host = match classify(units(target)) {
        Target::Host => None,
        Target::Drive(host) => Some(host),
        Target::Package(..) => return fail(ERROR_ACCESS_DENIED, 0),
    };
    let (link_host, done) = match receiving(classify(units(link)), false) {
        Ok(link) => link,
        Err(e) => return fail(e, 0),
    };
    let ok = create(pick(link, &link_host), pick(target, &target_host), flags);
    if ok != 0 {
        finish(done);
    }
    ok
}

pub extern "system" fn SetCurrentDirectoryA(path: *const c_char) -> i32 {
    let narrow =
        host!("kernel32.dll", "SetCurrentDirectoryA": extern "system" fn(*const c_char) -> i32);
    let set = host!("kernel32.dll", "SetCurrentDirectoryW": extern "system" fn(*const u16) -> i32);
    match decode(path).map(|units| reading(&units)) {
        None | Some(Ok(None)) => narrow(path),
        Some(Ok(Some(host))) => set(host.as_ptr()),
        Some(Err(e)) => fail(e, 0),
    }
}

// The C runtime (ucrtbase, which the title's CRT imports bind to).

pub extern "C" fn _wfopen(path: *const u16, mode: *const u16) -> File {
    let open = host!("ucrtbase.dll", "_wfopen": extern "C" fn(*const u16, *const u16) -> File);
    let intent = Intent::mode(units(mode));
    match opening(units(path), intent) {
        Ok(plan) => plan.run(path, 0, intent, |p, _| open(p, mode), |f| !f.is_null()),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

pub extern "C" fn _wfopen_s(out: *mut File, path: *const u16, mode: *const u16) -> i32 {
    let open =
        host!("ucrtbase.dll", "_wfopen_s": extern "C" fn(*mut File, *const u16, *const u16) -> i32);
    let intent = Intent::mode(units(mode));
    match opening(units(path), intent) {
        Ok(plan) => plan.run(path, 0, intent, |p, _| open(out, p, mode), |e| *e == 0),
        Err(e) => crt_fail_s(out, e),
    }
}

pub extern "C" fn _wfsopen(path: *const u16, mode: *const u16, share: i32) -> File {
    let open =
        host!("ucrtbase.dll", "_wfsopen": extern "C" fn(*const u16, *const u16, i32) -> File);
    let intent = Intent::mode(units(mode));
    match opening(units(path), intent) {
        Ok(plan) => plan.run(
            path,
            0,
            intent,
            |p, _| open(p, mode, share),
            |f| !f.is_null(),
        ),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

pub extern "C" fn _fsopen(path: *const c_char, mode: *const c_char, share: i32) -> File {
    let narrow =
        host!("ucrtbase.dll", "_fsopen": extern "C" fn(*const c_char, *const c_char, i32) -> File);
    let open =
        host!("ucrtbase.dll", "_wfsopen": extern "C" fn(*const u16, *const u16, i32) -> File);
    let wide_mode = widen(mode);
    let intent = Intent::mode(&wide_mode[..wide_mode.len() - 1]);
    let Some(units) = decode(path) else {
        return narrow(path, mode, share);
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => narrow(path, mode, share),
        Ok(plan) => plan.run(
            std::ptr::null(),
            0,
            intent,
            |p, _| open(p, wide_mode.as_ptr(), share),
            |f| !f.is_null(),
        ),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

// Kernel reads into title memory. The GPU side leaves blocks of title GPU
// buffers read-only to see the title's next write, and the kernel's writes
// fail there instead of faulting into its handler; BTP 6's
// synchronous reads (title+0x313fc0) read straight into 4 KiB-aligned
// destinations and retry a failed `ReadFile` forever. So the destination
// is opened first, and held open until the read is done: a copy guarding it
// again meanwhile cut reads (docs/analysis/archive-null.md). An overlapped
// read still under way when `ReadFile` returns keeps it open until it
// completes.

/// `ERROR_INVALID_USER_BUFFER`: the kernel could not write the destination
/// while it read.
const INVALID_USER_BUFFER: u32 = 1784;
const IO_PENDING: u32 = 997;

/// Le statut du noyau reste associé à la garde de la source ou de la destination.
type PendingIo = (usize, Box<dyn Send>);
static PENDING_IO: Mutex<Vec<PendingIo>> = Mutex::new(Vec::new());
static IO_PENDING_CHANGED: std::sync::Condvar = std::sync::Condvar::new();

fn hold_until_done(overlapped: usize, buffer: impl Send + 'static) {
    static WATCHER: std::sync::Once = std::sync::Once::new();
    WATCHER.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("xtp-overlapped-io".into())
            .spawn(|| {
                loop {
                    {
                        let mut pending = PENDING_IO.lock().unwrap_or_else(|e| e.into_inner());
                        while pending.is_empty() {
                            pending = IO_PENDING_CHANGED
                                .wait(pending)
                                .unwrap_or_else(|e| e.into_inner());
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    finish_overlapped_io();
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(
                "overlapped buffers are released at the next I/O call: no thread to watch them ({e})"
            );
        }
    });
    PENDING_IO
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((overlapped, Box::new(buffer)));
    IO_PENDING_CHANGED.notify_one();
}

/// Windows remplace STATUS_PENDING dans le premier champ de l'OVERLAPPED
/// à la fin de l'E/S. Le titre peut ensuite libérer cette structure.
fn finish_overlapped_io() {
    const STATUS_PENDING: usize = 0x103;
    let mut pending = PENDING_IO.lock().unwrap_or_else(|e| e.into_inner());
    if pending.is_empty() {
        return;
    }
    pending.retain(|&(overlapped, _)| read_own_usize(overlapped) == Some(STATUS_PENDING));
}

/// The `usize` at `at` in our own memory, or `None` where it cannot be read.
fn read_own_usize(at: usize) -> Option<usize> {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let _pin = xtp_gpu::pin_copy_back(at, size_of::<usize>());
    let (mut value, mut read) = (0usize, 0usize);
    // SAFETY: reads into a local; the call fails, rather than faults, on
    // memory that cannot be read.
    let ok = unsafe {
        ReadProcessMemory(
            GetCurrentProcess(),
            at as *const c_void,
            (&raw mut value).cast(),
            size_of::<usize>(),
            &mut read,
        )
    };
    (ok != 0 && read == size_of::<usize>()).then_some(value)
}

pub extern "system" fn ReadFile(
    file: *mut c_void,
    buffer: *mut u8,
    len: u32,
    read: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    let host = host!("kernel32.dll", "ReadFile": extern "system" fn(*mut c_void, *mut u8, u32, *mut u32, *mut c_void) -> i32);
    finish_overlapped_io();
    let writing = xtp_gpu::kernel_will_write(buffer as usize, len as usize);
    let mut done = host(file, buffer, len, read, overlapped);
    // SAFETY: plain reads and writes of this thread's last error.
    let mut error = unsafe { GetLastError() };
    if done == 0 && error == INVALID_USER_BUFFER {
        // Not expected with the destination held open: once more, and said.
        tracing::warn!(
            len,
            "ReadFile: a read was cut (ERROR_INVALID_USER_BUFFER); reading it again"
        );
        done = host(file, buffer, len, read, overlapped);
        // SAFETY: as above.
        error = unsafe { GetLastError() };
    }
    if done == 0 && error == IO_PENDING && !overlapped.is_null() {
        hold_until_done(overlapped as usize, writing);
    } else {
        drop(writing);
    }
    // SAFETY: as above; the title reads the read's error, not ours.
    unsafe { SetLastError(error) };
    done
}

pub extern "system" fn WriteFile(
    file: *mut c_void,
    buffer: *const u8,
    len: u32,
    written: *mut u32,
    overlapped: *mut c_void,
) -> i32 {
    let host = host!("kernel32.dll", "WriteFile": extern "system" fn(*mut c_void, *const u8, u32, *mut u32, *mut c_void) -> i32);
    finish_overlapped_io();
    let pin = xtp_gpu::pin_title_read(buffer as usize, len as usize);
    let result = host(file, buffer, len, written, overlapped);
    // SAFETY: sauvegarde de la dernière erreur sur le thread appelant.
    let error = unsafe { GetLastError() };
    if let Some(pin) = pin
        && result == 0
        && error == IO_PENDING
        && !overlapped.is_null()
    {
        hold_until_done(overlapped as usize, pin);
    }
    // SAFETY: le titre doit recevoir l'erreur de WriteFile.
    unsafe { SetLastError(error) };
    result
}

pub extern "C" fn fwrite(buffer: *const u8, size: usize, count: usize, stream: File) -> usize {
    let _pin = xtp_gpu::pin_title_read(buffer as usize, size.saturating_mul(count));
    host!("ucrtbase.dll", "fwrite": extern "C" fn(*const u8, usize, usize, File) -> usize)(
        buffer, size, count, stream,
    )
}

#[cfg(test)]
pub(super) fn check_fwrite_from(buffer: *const u8, len: usize, expected: u8) {
    let dir = tempfile::tempdir().unwrap();
    let mut path = wide_of(&dir.path().join("pending.bin"));
    path.push(0);
    let mode: Vec<_> = "w+b\0".encode_utf16().collect();
    let file = host!("ucrtbase.dll", "_wfopen": extern "C" fn(*const u16, *const u16) -> File)(
        path.as_ptr(),
        mode.as_ptr(),
    );
    assert!(!file.is_null());
    assert_eq!(fwrite(buffer, 1, len, file), len);
    host!("ucrtbase.dll", "rewind": extern "C" fn(File))(file);
    let mut bytes = vec![0; len];
    assert_eq!(fread(bytes.as_mut_ptr(), 1, len, file), len);
    assert!(bytes.iter().all(|&byte| byte == expected));
    assert_eq!(
        host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)(file),
        0
    );
}

/// The host CRT reads large requests straight into the caller's buffer.
pub extern "C" fn fread(buffer: *mut u8, size: usize, count: usize, stream: File) -> usize {
    finish_overlapped_io();
    let _writing = xtp_gpu::kernel_will_write(buffer as usize, size.saturating_mul(count));
    host!("ucrtbase.dll", "fread": extern "C" fn(*mut u8, usize, usize, File) -> usize)(
        buffer, size, count, stream,
    )
}

pub extern "C" fn fopen(path: *const c_char, mode: *const c_char) -> File {
    let narrow =
        host!("ucrtbase.dll", "fopen": extern "C" fn(*const c_char, *const c_char) -> File);
    let open = host!("ucrtbase.dll", "_wfopen": extern "C" fn(*const u16, *const u16) -> File);
    let wide_mode = widen(mode);
    let intent = Intent::mode(&wide_mode[..wide_mode.len() - 1]);
    let Some(units) = decode(path) else {
        return narrow(path, mode);
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => narrow(path, mode),
        Ok(plan) => plan.run(
            std::ptr::null(),
            0,
            intent,
            |p, _| open(p, wide_mode.as_ptr()),
            |f| !f.is_null(),
        ),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

pub extern "C" fn fopen_s(out: *mut File, path: *const c_char, mode: *const c_char) -> i32 {
    let narrow = host!("ucrtbase.dll", "fopen_s": extern "C" fn(*mut File, *const c_char, *const c_char) -> i32);
    let open =
        host!("ucrtbase.dll", "_wfopen_s": extern "C" fn(*mut File, *const u16, *const u16) -> i32);
    let wide_mode = widen(mode);
    let intent = Intent::mode(&wide_mode[..wide_mode.len() - 1]);
    let Some(units) = decode(path) else {
        return narrow(out, path, mode);
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => narrow(out, path, mode),
        Ok(plan) => plan.run(
            std::ptr::null(),
            0,
            intent,
            |p, _| open(out, p, wide_mode.as_ptr()),
            |e| *e == 0,
        ),
        Err(e) => crt_fail_s(out, e),
    }
}

/// `_open(path, flags[, mode])` is variadic; on x64 the optional integer
/// travels in the same register either way, so it is passed on as is.
pub extern "C" fn _open(path: *const c_char, flags: i32, mode: i32) -> i32 {
    let narrow = host!("ucrtbase.dll", "_open": extern "C" fn(*const c_char, i32, i32) -> i32);
    let open = host!("ucrtbase.dll", "_wopen": extern "C" fn(*const u16, i32, i32) -> i32);
    let intent = Intent::oflag(flags);
    let Some(units) = decode(path) else {
        return narrow(path, flags, mode);
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => narrow(path, flags, mode),
        Ok(plan) => plan.run(
            std::ptr::null(),
            0,
            intent,
            |p, _| open(p, flags, mode),
            |fd| *fd >= 0,
        ),
        Err(e) => crt_fail(e, -1),
    }
}

/// See [`_open`]; used by libHttpClient.
pub extern "C" fn _wopen(path: *const u16, flags: i32, mode: i32) -> i32 {
    let open = host!("ucrtbase.dll", "_wopen": extern "C" fn(*const u16, i32, i32) -> i32);
    let intent = Intent::oflag(flags);
    match opening(units(path), intent) {
        Ok(plan) => plan.run(path, 0, intent, |p, _| open(p, flags, mode), |fd| *fd >= 0),
        Err(e) => crt_fail(e, -1),
    }
}

pub extern "C" fn _wstat64(path: *const u16, stat: *mut c_void) -> i32 {
    let get = host!("ucrtbase.dll", "_wstat64": extern "C" fn(*const u16, *mut c_void) -> i32);
    match reading(units(path)) {
        Ok(host) => get(pick(path, &host), stat),
        Err(e) => crt_fail(e, -1),
    }
}

pub extern "C" fn _stat64i32(path: *const c_char, stat: *mut c_void) -> i32 {
    let narrow =
        host!("ucrtbase.dll", "_stat64i32": extern "C" fn(*const c_char, *mut c_void) -> i32);
    let get = host!("ucrtbase.dll", "_wstat64i32": extern "C" fn(*const u16, *mut c_void) -> i32);
    match decode(path).map(|units| reading(&units)) {
        None | Some(Ok(None)) => narrow(path, stat),
        Some(Ok(Some(host))) => get(host.as_ptr(), stat),
        Some(Err(e)) => crt_fail(e, -1),
    }
}

pub extern "C" fn remove(path: *const c_char) -> i32 {
    let narrow = host!("ucrtbase.dll", "remove": extern "C" fn(*const c_char) -> i32);
    let delete = host!("ucrtbase.dll", "_wremove": extern "C" fn(*const u16) -> i32);
    match decode(path).map(|units| classify(&units)) {
        None | Some(Target::Host) => narrow(path),
        Some(Target::Drive(host)) => delete(host.as_ptr()),
        Some(Target::Package(package, rel)) => crt(package.with(&rel, |o| o.delete_file(&rel))),
    }
}

pub extern "C" fn _wremove(path: *const u16) -> i32 {
    let delete = host!("ucrtbase.dll", "_wremove": extern "C" fn(*const u16) -> i32);
    match classify(units(path)) {
        Target::Host => delete(path),
        Target::Drive(host) => delete(host.as_ptr()),
        Target::Package(package, rel) => crt(package.with(&rel, |o| o.delete_file(&rel))),
    }
}

pub extern "C" fn _wrmdir(path: *const u16) -> i32 {
    let remove = host!("ucrtbase.dll", "_wrmdir": extern "C" fn(*const u16) -> i32);
    match classify(units(path)) {
        Target::Host => remove(path),
        Target::Drive(host) => remove(host.as_ptr()),
        Target::Package(package, rel) => crt(package.with(&rel, |o| o.remove_dir(&rel))),
    }
}

pub extern "C" fn _wrename(from: *const u16, to: *const u16) -> i32 {
    let host_rename =
        host!("ucrtbase.dll", "_wrename": extern "C" fn(*const u16, *const u16) -> i32);
    let (source, target) = (classify(units(from)), classify(units(to)));
    if matches!(source, Target::Package(..)) || matches!(target, Target::Package(..)) {
        // The CRT's own rename is this move.
        return crt(rename(from, source, to, target, MOVEFILE_COPY_ALLOWED));
    }
    let path = |title: *const u16, target: &Target| match target {
        Target::Drive(host) => host.as_ptr(),
        _ => title,
    };
    host_rename(path(from, &source), path(to, &target))
}

pub extern "C" fn _wchdir(path: *const u16) -> i32 {
    let set = host!("ucrtbase.dll", "_wchdir": extern "C" fn(*const u16) -> i32);
    match reading(units(path)) {
        Ok(host) => set(pick(path, &host)),
        Err(e) => crt_fail(e, -1),
    }
}

pub extern "C" fn _wchmod(path: *const u16, mode: i32) -> i32 {
    let set = host!("ucrtbase.dll", "_wchmod": extern "C" fn(*const u16, i32) -> i32);
    match changing(units(path)) {
        Ok((host, done)) => {
            let result = set(pick(path, &host), mode);
            if result == 0 {
                finish(done);
            }
            result
        }
        Err(e) => crt_fail(e, -1),
    }
}

pub extern "C" fn _wutime64(path: *const u16, times: *mut c_void) -> i32 {
    let set = host!("ucrtbase.dll", "_wutime64": extern "C" fn(*const u16, *mut c_void) -> i32);
    match changing(units(path)) {
        Ok((host, done)) => {
            let result = set(pick(path, &host), times);
            if result == 0 {
                finish(done);
            }
            result
        }
        Err(e) => crt_fail(e, -1),
    }
}

// The C++ library: `std::basic_filebuf::open` goes through `_Fiopen`.

const FIOPEN: &str = "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z";
const FIOPEN_WIDE: &str = "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z";

/// `_Fiopen(const wchar_t* path, ios_base::openmode, int protection)`.
extern "C" fn fiopen_wide(path: *const u16, mode: i32, protection: i32) -> File {
    let open = host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z": extern "C" fn(*const u16, i32, i32) -> File);
    let intent = Intent::ios(mode);
    match opening(units(path), intent) {
        Ok(plan) => plan.run(
            path,
            0,
            intent,
            |p, _| open(p, mode, protection),
            |f| !f.is_null(),
        ),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

/// `_Fiopen(const char* path, ios_base::openmode, int protection)`.
extern "C" fn fiopen(path: *const c_char, mode: i32, protection: i32) -> File {
    let narrow = host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEBDHH@Z": extern "C" fn(*const c_char, i32, i32) -> File);
    let open = host!("msvcp140.dll", "?_Fiopen@std@@YAPEAU_iobuf@@PEB_WHH@Z": extern "C" fn(*const u16, i32, i32) -> File);
    let intent = Intent::ios(mode);
    let Some(units) = decode(path) else {
        return narrow(path, mode, protection);
    };
    match opening(&units, intent) {
        Ok(Open { host: None, .. }) => narrow(path, mode, protection),
        Ok(plan) => plan.run(
            std::ptr::null(),
            0,
            intent,
            |p, _| open(p, mode, protection),
            |f| !f.is_null(),
        ),
        Err(e) => crt_fail(e, std::ptr::null_mut()),
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "files",
        dlls: &[
            "kernel32.dll",
            "kernelbase.dll",
            "api-ms-win-core-file-*",
            "api-ms-win-core-processenvironment-*",
            "ucrtbase.dll",
            "api-ms-win-crt-stdio-*",
            "api-ms-win-crt-filesystem-*",
            "api-ms-win-crt-time-*",
            "msvcp140.dll",
        ],
        exports: vec![
            hle_export!(ReadFile),
            hle_export!(WriteFile),
            hle_export!(fwrite),
            hle_export!(fread),
            hle_export!(CreateFileW),
            hle_export!(CreateFileA),
            hle_export!(CreateFile2),
            hle_export!(DeleteFileA),
            hle_export!(DeleteFileW),
            hle_export!(RemoveDirectoryW),
            hle_export!(SetFileAttributesW),
            hle_export!(GetFileAttributesA),
            hle_export!(CreateDirectoryW),
            hle_export!(GetFileAttributesW),
            hle_export!(GetFileAttributesExW),
            hle_export!(GetDiskFreeSpaceExW),
            hle_export!(GetDiskFreeSpaceW),
            hle_export!(FindFirstFileW),
            hle_export!(FindFirstFileExW),
            hle_export!(FindNextFileW),
            hle_export!(FindClose),
            hle_export!(MoveFileExW),
            hle_export!(CopyFile2),
            hle_export!(CreateHardLinkW),
            hle_export!(CreateSymbolicLinkW),
            hle_export!(SetCurrentDirectoryA),
            hle_export!(_wfopen),
            hle_export!(_wfopen_s),
            hle_export!(_wfsopen),
            hle_export!(_fsopen),
            hle_export!(fopen),
            hle_export!(fopen_s),
            hle_export!(_open),
            hle_export!(_wopen),
            hle_export!(_wstat64),
            hle_export!(_stat64i32),
            hle_export!(remove),
            hle_export!(_wremove),
            hle_export!(_wrmdir),
            hle_export!(_wrename),
            hle_export!(_wchdir),
            hle_export!(_wchmod),
            hle_export!(_wutime64),
            HleExport {
                name: FIOPEN,
                address: fiopen as *const () as usize,
            },
            HleExport {
                name: FIOPEN_WIDE,
                address: fiopen_wide as *const () as usize,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    fn w(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0]).collect()
    }

    /// Tests that mount the console's fixed letters take turns.
    static FIXED: Mutex<()> = Mutex::new(());

    fn fixed_letters() -> MutexGuard<'static, ()> {
        FIXED.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn unmount(letter: char) {
        drives().retain(|d| u32::from(d.letter) != u32::from(letter));
    }

    #[test]
    fn kernel_reads_reach_the_buffer() {
        use std::os::windows::io::AsRawHandle;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let (mut out, mut n) = ([0u8; 10], 0);
        let handle = file.as_raw_handle().cast();
        assert_ne!(
            ReadFile(handle, out.as_mut_ptr(), 10, &mut n, std::ptr::null_mut()),
            0
        );
        assert_eq!((&out, n), (b"0123456789", 10));
    }

    /// An overlapped read still under way keeps its destination open until
    /// the kernel sets the `OVERLAPPED`'s status; then the watching thread
    /// lets it go, without another read.
    #[test]
    fn overlapped_reads_hold_their_destination_until_they_complete() {
        let mut status: usize = 0x103; // STATUS_PENDING
        let at = (&raw mut status) as usize;
        let destination = [0u8; 64];
        let writing = xtp_gpu::kernel_will_write(destination.as_ptr() as usize, 64);
        hold_until_done(at, writing);
        let held = || PENDING_IO.lock().unwrap().iter().any(|&(o, _)| o == at);
        std::thread::sleep(std::time::Duration::from_millis(20));
        finish_overlapped_io();
        assert!(held(), "still pending");
        // SAFETY: our local, as the kernel writes the status on completion.
        unsafe { (&raw mut status).write_volatile(0) };
        let waited = std::time::Instant::now();
        while held() && waited.elapsed() < std::time::Duration::from_secs(2) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!held(), "completed: let go by the watching thread");
    }

    fn text(v: &[u16]) -> String {
        String::from_utf16_lossy(&v[..v.len() - 1])
    }

    #[test]
    fn rewrites_only_paths_on_mounted_drives() {
        let _fixed = fixed_letters();
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        let letter = mount(dir.path()).unwrap();
        assert_eq!(
            mount(dir.path()),
            Some(letter),
            "mounting twice reuses the letter"
        );
        let tr = |s: String| translate(&w(&s)[..s.encode_utf16().count()]).map(|v| text(&v));
        assert_eq!(
            tr(format!("{letter}:/a/b.txt")),
            Some(format!("{root}/a/b.txt"))
        );
        let lower = letter.to_ascii_lowercase();
        assert_eq!(tr(format!("{lower}:\\a")), Some(format!("{root}\\a")));
        assert_eq!(tr(format!("{letter}:")), Some(format!("{root}\\")));
        assert_eq!(
            tr(format!(r"\\?\{letter}:/a/b")),
            Some(format!(r"\\?\{root}\a\b"))
        );
        assert_eq!(tr(r"C:\Windows".into()), None);
        assert_eq!(tr("relative/path".into()), None);
        assert_eq!(tr(String::new()), None);

        // Fixed letters, as the console's package drive; a later mount of
        // the same letter replaces the earlier one.
        let package = tempfile::tempdir().unwrap();
        let overlay = tempfile::tempdir().unwrap();
        mount_at('g', dir.path());
        mount_package_at('G', package.path(), overlay.path());
        assert_eq!(
            tr(r"G:\appdata0.rpf".into()),
            Some(format!(r"{}\appdata0.rpf", package.path().display()))
        );
        assert_ne!(mount(dir.path()), Some('G'), "free letters skip fixed ones");

        // Long ones take the `\\?\` form, cleaned up as the host would.
        let long = "d".repeat(250);
        assert_eq!(
            tr(format!(r"G:/a/./b/../{long}//f")),
            Some(format!(r"\\?\{}\a\{long}\f", package.path().display()))
        );
        assert_eq!(
            verbatim(&w(r"\\server\share\")[..15], &w("/x/../../y")[..10]),
            w(r"\\?\UNC\server\share\y")[..22].to_vec()
        );
        unmount('G');
    }

    #[test]
    fn long_title_paths_work_on_the_host() {
        let dir = tempfile::tempdir().unwrap();
        let letter = mount(dir.path()).unwrap();
        let deep = format!("{letter}:/{}/{}", "a".repeat(200), "b".repeat(100));
        let null = std::ptr::null_mut();
        let first = w(&deep[..203]);
        assert_ne!(CreateDirectoryW(first.as_ptr(), null), 0);
        assert_ne!(CreateDirectoryW(w(&deep).as_ptr(), null), 0);
        let file = format!("{deep}/../{}/f.txt", "b".repeat(100));
        // `N`: a child of another test holding the file (see
        // `writes_renames_and_deletes_leave_the_package_byte_identical`)
        // would fail its delete.
        let f = _wfopen(w(&file).as_ptr(), w("wbN").as_ptr());
        assert!(!f.is_null(), "{file}");
        host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)(f);
        assert_ne!(DeleteFileW(w(&format!("{deep}/f.txt")).as_ptr()), 0);
        assert_ne!(RemoveDirectoryW(w(&deep).as_ptr()), 0);
    }

    #[test]
    fn title_file_calls_land_in_the_host_folder() {
        let dir = tempfile::tempdir().unwrap();
        let letter = mount(dir.path()).unwrap();
        const CREATE_ALWAYS: u32 = 2;
        let close = host!("kernel32.dll", "CloseHandle": extern "system" fn(Handle) -> i32);
        let fclose = host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32);

        let path = w(&format!(r"{letter}:\created.bin"));
        let null = std::ptr::null_mut();
        let h = CreateFileW(
            path.as_ptr(),
            GENERIC_WRITE,
            0,
            null,
            CREATE_ALWAYS,
            0,
            null,
        );
        assert_ne!(h as isize, -1);
        close(h);
        assert!(dir.path().join("created.bin").is_file());

        // `N`: a child of another test holding the file (see
        // `writes_renames_and_deletes_leave_the_package_byte_identical`)
        // would fail its `remove` below.
        let opened = format!("{letter}:/opened.txt\0");
        let f = fopen(opened.as_ptr().cast(), c"wbN".as_ptr());
        assert!(!f.is_null());
        fclose(f);
        assert!(dir.path().join("opened.txt").is_file());

        let path = format!("{letter}:/stream.txt\0");
        const OUT: i32 = 2;
        let f = fiopen(path.as_ptr().cast(), OUT, 0x40);
        assert!(!f.is_null());
        fclose(f);
        assert!(dir.path().join("stream.txt").is_file());

        // The file `fopen` made, not the stream's: a stream's open mode
        // cannot ask for `N`, so a child may hold that one.
        assert_eq!(remove(opened.as_ptr().cast()), 0);
        assert!(!dir.path().join("opened.txt").exists());
        let f = fiopen_wide(w(&format!("{letter}:/wide.txt")).as_ptr(), OUT, 0x40);
        assert!(!f.is_null());
        fclose(f);
        assert!(dir.path().join("wide.txt").is_file());

        let path = w(&format!(r"{letter}:\created.bin"));
        let h = CreateFile2(path.as_ptr(), GENERIC_WRITE, 0, OPEN_EXISTING, null);
        assert_ne!(h as isize, -1);
        close(h);
        assert_ne!(GetFileAttributesW(path.as_ptr()), INVALID_FILE_ATTRIBUTES);

        const O_WRONLY: i32 = 1;
        const O_CREAT: i32 = 0x100;
        const S_IWRITE: i32 = 0x80;
        let path = format!("{letter}:/low.bin\0");
        let fd = _open(path.as_ptr().cast(), O_WRONLY | O_CREAT, S_IWRITE);
        assert!(fd >= 0);
        host!("ucrtbase.dll", "_close": extern "C" fn(i32) -> i32)(fd);
        assert!(dir.path().join("low.bin").is_file());

        // Other paths pass through untouched.
        let host_path = format!("{}\0", dir.path().join("plain.txt").display());
        let f = fopen(host_path.as_ptr().cast(), c"wb".as_ptr());
        assert!(!f.is_null());
        fclose(f);
        assert!(dir.path().join("plain.txt").is_file());
    }

    /// A narrow path is rewritten once, as its wide twin is. The inner
    /// drive's folder lies on the outer virtual drive, as the title folder
    /// does when it sits on a host `G:` drive and is mounted as `G:`. The
    /// outer letter is no host drive, so the host finds nothing there; a
    /// narrow call that reaches the outer folder rewrote the path twice.
    #[test]
    fn narrow_paths_are_rewritten_once() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("inner");
        std::fs::create_dir(&folder).unwrap();
        std::fs::write(folder.join("present.txt"), b"").unwrap();
        let outer = mount(dir.path()).unwrap();
        let inner = mount(Path::new(&format!(r"{outer}:\inner"))).unwrap();
        let path = |name: &str| format!("{inner}:/{name}\0");

        let wide = GetFileAttributesW(w(&format!("{inner}:/present.txt")).as_ptr());
        assert_eq!(wide, INVALID_FILE_ATTRIBUTES);
        let narrow = GetFileAttributesA(path("present.txt").as_ptr().cast());
        assert_eq!(narrow, wide, "GetFileAttributesA agrees with its wide twin");

        const CREATE_ALWAYS: u32 = 2;
        let null = std::ptr::null_mut();
        let h = CreateFileA(
            path("a.bin").as_ptr().cast(),
            GENERIC_WRITE,
            0,
            null,
            CREATE_ALWAYS,
            0,
            null,
        );
        assert_eq!(h as isize, -1);
        assert!(fopen(path("b.txt").as_ptr().cast(), c"wb".as_ptr()).is_null());
        let mut f: File = std::ptr::null_mut();
        let error = fopen_s(&mut f, path("c.txt").as_ptr().cast(), c"wb".as_ptr());
        assert_ne!(error, 0);
        const OUT: i32 = 2;
        assert!(fiopen(path("d.txt").as_ptr().cast(), OUT, 0x40).is_null());

        // Nothing was made behind a path rewritten twice.
        let names: Vec<_> = std::fs::read_dir(&folder)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["present.txt"]);
    }

    /// A package folder mounted read-only, its overlay in a folder of its
    /// own; unmounted when dropped.
    struct Mounted {
        letter: char,
        package: PathBuf,
        overlay: PathBuf,
        _dir: tempfile::TempDir,
    }

    impl Mounted {
        /// At a free letter, or at `letter`.
        fn new(letter: Option<char>) -> Mounted {
            let dir = tempfile::tempdir().unwrap();
            let package = dir.path().join("package");
            std::fs::create_dir_all(package.join("data/sub")).unwrap();
            for (file, bytes) in [
                ("appdata.rpf", "package bytes"),
                ("readme.txt", "read me"),
                ("data/level.dat", "level"),
                ("data/gone.txt", "gone"),
                ("data/sub/deep.txt", "deep"),
            ] {
                std::fs::write(package.join(file), bytes).unwrap();
            }
            let overlay = dir.path().join("overlay");
            let letter = match letter {
                Some(letter) => {
                    mount_package_at(letter, &package, &overlay);
                    letter
                }
                None => mount_free(&package, Some(OverlayAt::Folder(overlay.clone())))
                    .expect("a free drive letter"),
            };
            Mounted {
                letter,
                package,
                overlay,
                _dir: dir,
            }
        }

        /// `rel` on the drive, as the title writes it, NUL-terminated.
        fn path(&self, rel: &str) -> Vec<u16> {
            w(&format!("{}:\\{rel}", self.letter))
        }

        fn narrow(&self, rel: &str) -> String {
            format!("{}:/{rel}\0", self.letter)
        }
    }

    impl Drop for Mounted {
        fn drop(&mut self) {
            unmount(self.letter);
        }
    }

    /// Everything below `dir`: each entry and, for a file, its bytes, its
    /// read-only flag and its last write.
    fn snapshot(dir: &Path) -> BTreeMap<PathBuf, Option<(Vec<u8>, bool, SystemTime)>> {
        let mut out = BTreeMap::new();
        let mut todo = vec![dir.to_owned()];
        while let Some(at) = todo.pop() {
            for entry in std::fs::read_dir(&at).unwrap() {
                let path = entry.unwrap().path();
                let rel = path.strip_prefix(dir).unwrap().to_owned();
                let meta = std::fs::metadata(&path).unwrap();
                if meta.is_dir() {
                    todo.push(path);
                    out.insert(rel, None);
                } else {
                    let bytes = std::fs::read(&path).unwrap();
                    let readonly = meta.permissions().readonly();
                    out.insert(rel, Some((bytes, readonly, meta.modified().unwrap())));
                }
            }
        }
        out
    }

    const GENERIC_READ: u32 = 0x8000_0000;

    /// A file's bytes, read through the title's API; None if it does not
    /// open.
    fn contents(path: &[u16]) -> Option<Vec<u8>> {
        let close = host!("kernel32.dll", "CloseHandle": extern "system" fn(Handle) -> i32);
        let null = std::ptr::null_mut();
        let h = CreateFileW(path.as_ptr(), GENERIC_READ, 1, null, OPEN_EXISTING, 0, null);
        if h == INVALID_HANDLE_VALUE {
            return None;
        }
        let (mut out, mut buffer) = (Vec::new(), [0u8; 256]);
        loop {
            let mut n = 0;
            assert_ne!(ReadFile(h, buffer.as_mut_ptr(), 256, &mut n, null), 0);
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buffer[..n as usize]);
        }
        close(h);
        Some(out)
    }

    /// The names a title's search finds, `.` and `..` left out.
    fn listing(pattern: &[u16]) -> Vec<String> {
        let mut data = WIN32_FIND_DATAW::default();
        let h = FindFirstFileW(pattern.as_ptr(), (&raw mut data).cast());
        if h == INVALID_HANDLE_VALUE {
            return Vec::new();
        }
        let mut names = vec![found_name(&data)];
        while FindNextFileW(h, (&raw mut data).cast()) != 0 {
            names.push(found_name(&data));
        }
        assert_eq!(last_error(), ERROR_NO_MORE_FILES);
        assert_ne!(FindClose(h), 0);
        names.retain(|n| n != "." && n != "..");
        names
    }

    #[test]
    fn writes_renames_and_deletes_leave_the_package_byte_identical() {
        let m = Mounted::new(None);
        let before = snapshot(&m.package);
        let null = std::ptr::null_mut();
        let close = host!("kernel32.dll", "CloseHandle": extern "system" fn(Handle) -> i32);
        let write = host!("kernel32.dll", "WriteFile": extern "system" fn(Handle, *const u8, u32, *mut u32, *mut c_void) -> i32);
        let fclose = host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32);
        let fwrite =
            host!("ucrtbase.dll", "fwrite": extern "C" fn(*const u8, usize, usize, File) -> usize);

        // Writing a package file writes its copy.
        let path = m.path("appdata.rpf");
        let access = GENERIC_READ | GENERIC_WRITE;
        let h = CreateFileW(path.as_ptr(), access, 0, null, OPEN_EXISTING, 0, null);
        assert_ne!(h, INVALID_HANDLE_VALUE);
        let mut n = 0;
        assert_ne!(write(h, b"changed".as_ptr(), 7, &mut n, null), 0);
        close(h);
        assert_eq!(contents(&path).unwrap(), b"changed bytes");

        // The CRT opens below ask for `N` and `_O_NOINHERIT`. A CRT handle
        // is inheritable by default, and other tests in this binary start
        // child processes (xgamesave, xmem): a child started while one is
        // open keeps a duplicate, without FILE_SHARE_DELETE, until it
        // exits, and the reads, renames and deletes below then fail with
        // ERROR_SHARING_VIOLATION. The flag changes nothing the overlay
        // does: the mode and the flags reach the host CRT as they are.

        // So does `fopen`, and a new file lives in the overlay only.
        let f = fopen(m.narrow("data/level.dat").as_ptr().cast(), c"abN".as_ptr());
        assert!(!f.is_null());
        assert_eq!(fwrite(b"+more".as_ptr(), 1, 5, f), 5);
        fclose(f);
        assert_eq!(contents(&m.path("data\\level.dat")).unwrap(), b"level+more");
        assert!(m.overlay.join("files\\data\\level.dat").is_file());
        let f = _wfopen(m.path("new.txt").as_ptr(), w("wN").as_ptr());
        assert!(!f.is_null());
        fclose(f);
        assert!(m.overlay.join("files\\new.txt").is_file());
        const O_WRONLY: i32 = 1;
        const O_NOINHERIT: i32 = 0x80;
        const O_TRUNC: i32 = 0x200;
        let fd = _wopen(
            m.path("data\\sub\\deep.txt").as_ptr(),
            O_WRONLY | O_TRUNC | O_NOINHERIT,
            0,
        );
        assert!(fd >= 0);
        host!("ucrtbase.dll", "_close": extern "C" fn(i32) -> i32)(fd);
        assert_eq!(contents(&m.path("data\\sub\\deep.txt")).unwrap(), b"");

        // Creating over a package file: it was there, as the title is told.
        let path = m.path("readme.txt");
        let h = CreateFile2(path.as_ptr(), GENERIC_WRITE, 0, CREATE_ALWAYS, null);
        assert_ne!(h, INVALID_HANDLE_VALUE);
        assert_eq!(last_error(), ERROR_ALREADY_EXISTS);
        close(h);
        assert_eq!(contents(&path).unwrap(), b"");
        let h = CreateFileW(path.as_ptr(), GENERIC_WRITE, 0, null, CREATE_NEW, 0, null);
        assert_eq!((h, last_error()), (INVALID_HANDLE_VALUE, ERROR_FILE_EXISTS));

        // Deletes hide package files.
        assert_ne!(DeleteFileW(m.path("data\\gone.txt").as_ptr()), 0);
        let gone = GetFileAttributesW(m.path("data\\gone.txt").as_ptr());
        assert_eq!(
            (gone, last_error()),
            (INVALID_FILE_ATTRIBUTES, ERROR_FILE_NOT_FOUND)
        );
        assert_eq!(remove(m.narrow("new.txt").as_ptr().cast()), 0);
        assert_eq!(contents(&m.path("new.txt")), None);

        // Renames.
        let (deep, moved) = (m.path("data\\sub\\deep.txt"), m.path("moved.txt"));
        assert_ne!(MoveFileExW(deep.as_ptr(), moved.as_ptr(), 0), 0);
        assert_eq!(contents(&moved).unwrap(), b"");
        assert_eq!(contents(&deep), None);
        let readme = m.path("readme.txt");
        assert_eq!(MoveFileExW(moved.as_ptr(), readme.as_ptr(), 0), 0);
        assert_eq!(last_error(), ERROR_ALREADY_EXISTS);
        let rpf = m.path("data\\appdata.rpf");
        assert_eq!(_wrename(m.path("appdata.rpf").as_ptr(), rpf.as_ptr()), 0);
        assert_eq!(contents(&rpf).unwrap(), b"changed bytes");
        assert_eq!(contents(&m.path("appdata.rpf")), None);
        let level = m.path("data\\level.dat");
        assert_ne!(
            MoveFileExW(level.as_ptr(), m.path("level.dat").as_ptr(), 0),
            0
        );
        assert_ne!(
            MoveFileExW(m.path("level.dat").as_ptr(), level.as_ptr(), 0),
            0
        );

        // Attributes and times change on the copies.
        const READONLY: u32 = 1;
        assert_ne!(SetFileAttributesW(level.as_ptr(), READONLY), 0);
        assert_eq!(GetFileAttributesW(level.as_ptr()) & READONLY, READONLY);
        assert_eq!(_wchmod(moved.as_ptr(), 0x100), 0);
        assert_eq!(GetFileAttributesW(moved.as_ptr()) & READONLY, READONLY);
        assert_eq!(_wutime64(readme.as_ptr(), null), 0);
        assert_eq!(
            CopyFile2(moved.as_ptr(), m.path("copy.txt").as_ptr(), null),
            0
        );

        // Folders.
        assert_ne!(CreateDirectoryW(m.path("data\\made").as_ptr(), null), 0);
        assert_eq!(_wrmdir(m.path("data\\made").as_ptr()), 0);
        assert_ne!(RemoveDirectoryW(m.path("data\\sub").as_ptr()), 0);

        // Listings merge both layers, without what was deleted.
        assert_eq!(
            listing(&m.path("*")),
            ["copy.txt", "data", "moved.txt", "readme.txt"]
        );
        assert_eq!(listing(&m.path("data\\*")), ["appdata.rpf", "level.dat"]);
        assert_eq!(listing(&m.path("data\\*.dat")), ["level.dat"]);
        assert_eq!(listing(&m.path("data\\sub\\*")), Vec::<String>::new());
        assert_eq!(last_error(), ERROR_PATH_NOT_FOUND);

        assert_eq!(snapshot(&m.package), before, "the package is untouched");
        // So the temporary folder goes.
        const NORMAL: u32 = 0x80;
        for rel in ["data\\level.dat", "moved.txt", "copy.txt"] {
            assert_ne!(SetFileAttributesW(m.path(rel).as_ptr(), NORMAL), 0);
        }
    }

    #[test]
    fn host_paths_into_the_package_go_through_the_overlay() {
        let m = Mounted::new(None);
        let before = snapshot(&m.package);
        let null = std::ptr::null_mut();
        let close = host!("kernel32.dll", "CloseHandle": extern "system" fn(Handle) -> i32);
        let host_path = |rel: &str| w(&m.package.join(rel).display().to_string());

        let readme = host_path("data\\..\\readme.txt");
        let h = CreateFileW(
            readme.as_ptr(),
            GENERIC_WRITE,
            0,
            null,
            CREATE_ALWAYS,
            0,
            null,
        );
        assert_ne!(h, INVALID_HANDLE_VALUE);
        close(h);
        assert_eq!(contents(&m.path("readme.txt")).unwrap(), b"");
        assert_ne!(DeleteFileW(host_path("data\\gone.txt").as_ptr()), 0);
        assert_eq!(contents(&m.path("data\\gone.txt")), None);
        let f = fopen(
            format!("{}\0", m.package.join("own.txt").display())
                .as_ptr()
                .cast(),
            c"w".as_ptr(),
        );
        assert!(!f.is_null());
        host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)(f);
        assert!(m.overlay.join("files\\own.txt").is_file());
        assert_eq!(snapshot(&m.package), before);
    }

    #[test]
    fn game_and_g_name_the_same_package_file() {
        let _fixed = fixed_letters();
        let m = Mounted::new(Some('G'));
        let tr = |s: &str| translate(&w(s)[..s.encode_utf16().count()]).map(|v| text(&v));
        let lower = format!(r"{}\readme.txt", m.package.display());
        assert_eq!(tr(r"G:\readme.txt"), Some(lower.clone()));
        assert_eq!(tr(r"GAME:\readme.txt"), Some(lower.clone()));
        assert_eq!(tr("game:/readme.txt"), Some(lower));

        // `N`, or a child another test starts while the copy is open keeps
        // it open with write access, and the read below, which shares only
        // reading, fails with ERROR_SHARING_VIOLATION (xtp-farm-3 at
        // 2a63867; see `writes_renames_and_deletes_leave_the_package_byte_identical`).
        let f = _wfopen(w(r"GAME:\readme.txt").as_ptr(), w("aN").as_ptr());
        assert!(!f.is_null());
        let fwrite =
            host!("ucrtbase.dll", "fwrite": extern "C" fn(*const u8, usize, usize, File) -> usize);
        assert_eq!(fwrite(b"!".as_ptr(), 1, 1, f), 1);
        host!("ucrtbase.dll", "fclose": extern "C" fn(File) -> i32)(f);
        assert_eq!(contents(&w(r"G:\readme.txt")).unwrap(), b"read me!");
        assert_eq!(
            std::fs::read(m.package.join("readme.txt")).unwrap(),
            b"read me"
        );
    }

    #[test]
    fn unmapped_console_paths_are_logged_once() {
        let seen = |s: &str| {
            UNMAPPED
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .contains(&s.to_lowercase())
        };
        let units = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        let named = r"QZXW:\save\slot.dat";
        assert!(matches!(classify(&units(named)), Target::Host));
        assert!(seen(named));
        assert!(!unmapped(&units(named)), "a second sight is not logged");

        // A letter neither mounted nor the host's.
        // SAFETY: no arguments.
        let host_has_b = unsafe { GetLogicalDrives() } & 0b10 != 0;
        if !host_has_b {
            classify(&units(r"B:\x"));
            assert!(seen(r"B:\x"));
            assert!(!unmapped(&units(r"B:\x")));
        }
        // Host drives' paths are no console paths.
        classify(&units(r"C:\Windows\win.ini"));
        assert!(!seen(r"C:\Windows\win.ini"));
    }

    #[test]
    fn a_package_without_its_overlay_only_reads() {
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("package");
        std::fs::create_dir(&package).unwrap();
        std::fs::write(package.join("readme.txt"), "read me").unwrap();
        // An overlay folder whose whiteout list is unreadable.
        let overlay = dir.path().join("overlay");
        std::fs::create_dir_all(overlay.join("whiteouts.txt")).unwrap();
        let letter = mount_free(&package, Some(OverlayAt::Folder(overlay))).unwrap();
        let path = |rel: &str| w(&format!("{letter}:\\{rel}"));
        let null = std::ptr::null_mut();

        assert_eq!(contents(&path("readme.txt")).unwrap(), b"read me");
        let h = CreateFileW(
            path("readme.txt").as_ptr(),
            GENERIC_WRITE,
            0,
            null,
            OPEN_EXISTING,
            0,
            null,
        );
        assert_eq!(
            (h, last_error()),
            (INVALID_HANDLE_VALUE, ERROR_ACCESS_DENIED)
        );
        let h = CreateFileW(
            path("new.txt").as_ptr(),
            GENERIC_READ,
            0,
            null,
            OPEN_ALWAYS,
            0,
            null,
        );
        assert_eq!(
            (h, last_error()),
            (INVALID_HANDLE_VALUE, ERROR_ACCESS_DENIED)
        );
        assert_eq!(DeleteFileW(path("readme.txt").as_ptr()), 0);
        assert!(_wfopen(path("readme.txt").as_ptr(), w("r+").as_ptr()).is_null());
        let names: Vec<_> = std::fs::read_dir(&package).unwrap().collect();
        assert_eq!(names.len(), 1);
        assert_eq!(
            std::fs::read(package.join("readme.txt")).unwrap(),
            b"read me"
        );
        unmount(letter);
    }

    #[test]
    fn the_title_overlay_lives_in_its_data_folder() {
        assert_eq!(
            beside_storage(Path::new(r"D:\xtp\data\0000ABCD\storage")),
            Some(PathBuf::from(r"D:\xtp\data\0000ABCD\overlay"))
        );
        assert_eq!(beside_storage(Path::new("")), None);
    }

    #[test]
    fn opens_ask_what_their_modes_say() {
        let mode = |s: &str| {
            let i = Intent::mode(&w(s)[..s.len()]);
            (i.changes, i.must, i.keep)
        };
        assert_eq!(mode("rb"), (false, Must::Exist, true));
        assert_eq!(mode("r+b"), (true, Must::Exist, true));
        assert_eq!(mode("wb"), (true, Must::Either, false));
        assert_eq!(mode("wx"), (true, Must::NotExist, false));
        assert_eq!(mode("a, ccs=UTF-8"), (true, Must::Either, true));
        let ios = |m: i32| {
            let i = Intent::ios(m);
            (i.changes, i.must, i.keep)
        };
        assert_eq!(ios(1), (false, Must::Exist, true), "in");
        assert_eq!(ios(2), (true, Must::Either, false), "out");
        assert_eq!(ios(3), (true, Must::Exist, true), "in | out");
        assert_eq!(ios(0x12), (true, Must::Either, false), "out | trunc");
        assert_eq!(ios(0xa), (true, Must::Either, true), "out | app");
        assert_eq!(ios(0x42), (true, Must::Exist, true), "out | _Nocreate");
        assert_eq!(ios(0x82), (true, Must::NotExist, false), "out | _Noreplace");
        let oflag = |f: i32| {
            let i = Intent::oflag(f);
            (i.changes, i.must, i.keep)
        };
        assert_eq!(oflag(0), (false, Must::Exist, true), "O_RDONLY");
        assert_eq!(
            oflag(0x301),
            (true, Must::Either, false),
            "O_WRONLY | O_CREAT | O_TRUNC"
        );
        assert_eq!(
            oflag(0x502),
            (true, Must::NotExist, true),
            "O_RDWR | O_CREAT | O_EXCL"
        );
        let win32 = |a: u32, d: u32| {
            let i = Intent::win32(a, d, 0);
            (i.changes, i.must, i.keep)
        };
        assert_eq!(
            win32(GENERIC_READ, OPEN_EXISTING),
            (false, Must::Exist, true)
        );
        assert_eq!(
            win32(GENERIC_READ, OPEN_ALWAYS),
            (false, Must::Either, true)
        );
        assert_eq!(
            win32(0x100, OPEN_EXISTING),
            (true, Must::Exist, true),
            "FILE_WRITE_ATTRIBUTES"
        );
        assert_eq!(
            win32(GENERIC_WRITE, TRUNCATE_EXISTING),
            (true, Must::Exist, false)
        );
    }
}
