//! An opt-in switch of the title's ray tracing to its non-traced fallback,
//! off by default. `XTP_RT_PATH=fallback` makes the title's "ray-tracing
//! scene ready" predicate answer no from the start; `bvh` (or unset) leaves
//! it as it is; `ab` (or `ab:<phase s>[:<settle s>]`) alternates the two
//! within one run on the A/B clock of `xtp_core::ab`: phase A runs the
//! title's own answer (`XTP_RT_PATH=0` in the A/B lines), phase B the
//! fallback (`=1`).
//!
//! Why it stays off by default (rt-fallback.md §8, one Series S session
//! with `ab`): in phase B the fallback passes run and no trace program is
//! dispatched any more, but the scene is no brighter, and frames take
//! about 17 % longer.
//! - On the one view both arms showed, the lit HDR before fog moved by
//!   0.3 %.
//! - The irradiance atlas read 0 in both arms.
//!
//! Why the switch exists (docs/analysis/rt-fallback.md). The title chooses between its traced
//! passes, which walk a ray-tracing structure that XTP cannot translate yet
//! (their dispatches are dropped), and their non-traced fallbacks, which
//! translate, by one predicate that reads only its own state: no system or
//! device query that XTP could answer instead. 46 call sites ask it: the
//! screen probes, the probe system (pass 17 instead of 10/11 for the
//! primary gather, pass 73 instead of 68/69 for the transparent surfaces;
//! the irradiance cache's gather and shadow rays are then not dispatched at
//! all), the lighting after them and about twenty others.
//!
//! What the title does (1.0.2.1, by RVA). `0x4fd8550(i)` loads the title's
//! ray-tracing manager (the pointer at `0x8527dc8`); without one it returns
//! 0 (its own `return 0`, at `0x4fd8562`), otherwise it jumps to its twin
//! `0x4fd8570(manager, i)`. The twin returns 0 (at `0x4fd8581`) when the
//! manager has no scene `i`, and otherwise whether that scene has something
//! to trace. Both are leaf functions: no stack frame, no unwind data. The
//! predicate's first instruction is two bytes long, the twin's four.
//!
//! The hook. Each function's first two bytes become a short jump to its own
//! `return 0`, the path the title takes before its ray-tracing scene
//! exists; nothing else is written, and the code the jumps land on is the
//! title's. Putting the jumps in or taking them out is one aligned two-byte
//! store per function, which is safe while title threads run:
//! - a thread at an entry runs either the jump or the original first
//!   instruction, whole (the store is atomic, and the jump covers no more
//!   than that instruction's bytes);
//! - a thread past an entry is in bytes the hook never writes;
//! - a thread that took a jump is in the title's `return 0`, which stays.
//!
//! (The analysis guessed an aligned four-byte store; a `return 0` of our own
//! at the entry would leave a thread between its two instructions to resume
//! in the middle of the title's first one when it is taken out.)
//!
//! Before anything is written, the build's BLAKE3 is matched and each
//! function's bytes, from its entry through its `return 0`, must give a
//! known FNV-1a hash. The title's bytes are not kept here, only that hash.
//!
//! In `ab`, a thread of ours follows the clock and flips both entries within
//! a few milliseconds of the present that ends a phase. That is not a frame
//! boundary of the title: the frame across a flip can see both answers.
//! The phase's settling time leaves it out of the measurements.
//!
//! The trade-off: every caller takes its non-traced branch at once, which
//! is what the switch is for. What each caller outside the probe system
//! does on that branch is the analysis' open list (rt-fallback.md §4).
//! The irradiance cache has no fallback; the screen probes' fallback rays
//! are refused by the translator, so that hole stays. Code that reads the
//! manager's fields without asking either function is not affected.

use std::ffi::OsStr;
use std::fmt;

use crate::{BUILD_1_0_2_1, KnownBuild};

/// `XTP_RT_PATH=fallback|bvh|ab[:<phase s>[:<settle s>]]`.
pub const ENV: &str = "XTP_RT_PATH";

/// The value that makes the predicate answer no.
pub const FALLBACK: &str = "fallback";

/// The value that leaves the title's own predicate (the default).
pub const BVH: &str = "bvh";

/// Which of its paths the title takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    /// The title's own answer: its traced passes once its scene is ready.
    Bvh,
    /// "No scene ready": its non-traced fallbacks.
    Fallback,
}

impl Choice {
    /// The choice of an A/B arm's value: `true` (phase B) is the fallback.
    pub fn of(fallback: bool) -> Choice {
        if fallback {
            Choice::Fallback
        } else {
            Choice::Bvh
        }
    }
}

impl fmt::Display for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Choice::Bvh => "bvh (the title's own answer)",
            Choice::Fallback => "fallback (no ray-tracing scene ready)",
        })
    }
}

/// What [`ENV`] asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asked {
    /// Unset or empty: nothing written.
    Nothing,
    /// `bvh` (any case): nothing written either.
    Bvh,
    /// `fallback` (any case): the predicate answers no for the whole run.
    Fallback,
    /// `ab` or `ab:…`: alternate, as `xtp_core::ab` reads the plan.
    Ab,
    /// Something else: nothing written, and a warning.
    Unknown(String),
}

/// Reads [`ENV`]'s `value`.
pub fn asked(value: Option<&OsStr>) -> Asked {
    let Some(value) = value else {
        return Asked::Nothing;
    };
    let Some(text) = value.to_str().map(str::trim) else {
        return Asked::Unknown(value.to_string_lossy().into_owned());
    };
    if text.is_empty() {
        Asked::Nothing
    } else if text.eq_ignore_ascii_case(BVH) {
        Asked::Bvh
    } else if text.eq_ignore_ascii_case(FALLBACK) {
        Asked::Fallback
    } else if text == "ab" || text.starts_with("ab:") {
        // What `xtp_core::ab` takes for an A/B; it reads the plan itself.
        Asked::Ab
    } else {
        Asked::Unknown(text.to_owned())
    }
}

/// One function's entry, made to jump to its own `return 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Site {
    /// What it is, for the log.
    pub what: &'static str,
    /// The function's entry.
    pub rva: usize,
    /// How many bytes from the entry are checked: through the end of the
    /// function's `return 0`.
    pub checked: usize,
    /// FNV-1a (64-bit, [`fnv1a`]) of those bytes as the build has them.
    pub fnv1a: u64,
    /// The function's own `return 0`, where the jump lands.
    pub returns_false: usize,
}

impl Site {
    /// The two bytes written over the entry: `jmp short` to
    /// [`Self::returns_false`].
    pub const fn jump(&self) -> [u8; 2] {
        let rel = self.returns_false as isize - (self.rva as isize + 2);
        assert!(rel >= 0 && rel <= i8::MAX as isize, "out of a short jump");
        [0xEB, rel as u8]
    }
}

/// The hook for one build.
#[derive(Debug, Clone, Copy)]
pub struct Hook {
    pub build: &'static KnownBuild,
    /// Written together, in this order.
    pub sites: &'static [Site],
}

/// Builds the hook knows. Any other build is refused: its offsets would be
/// someone else's code.
pub const HOOKS: &[Hook] = &[Hook {
    build: &BUILD_1_0_2_1,
    sites: SITES_1_0_2_1,
}];

const SITES_1_0_2_1: &[Site] = &[
    Site {
        what: "ray-tracing scene ready, on the title's manager",
        rva: 0x4fd8550,
        checked: 0x15,
        fnv1a: 0xf209_d254_3805_1062,
        returns_false: 0x4fd8562,
    },
    Site {
        what: "ray-tracing scene ready, on a given manager",
        rva: 0x4fd8570,
        checked: 0x14,
        fnv1a: 0x53fa_2342_f974_a7d0,
        returns_false: 0x4fd8581,
    },
];

/// 64-bit FNV-1a of `bytes`.
pub const fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        i += 1;
    }
    hash
}

/// Why the hook was not applied, or not wholly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotApplied {
    /// No sites are known for this build: nothing written.
    UnknownBuild,
    /// A site's bytes are not the build's, or lie outside the image:
    /// nothing written.
    Mismatch { what: &'static str, rva: usize },
    /// Already installed in this process: nothing written.
    Twice,
    /// Making a site writable failed after `written` sites were written.
    Write {
        what: &'static str,
        rva: usize,
        written: usize,
        os_error: i32,
    },
}

impl fmt::Display for NotApplied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnknownBuild => write!(f, "no sites are known for this build"),
            Self::Mismatch { what, rva } => write!(
                f,
                "the bytes at {rva:#x} ({what}) are not the expected ones; nothing written"
            ),
            Self::Twice => write!(f, "already installed; nothing written"),
            Self::Write {
                what,
                rva,
                written,
                os_error,
            } => write!(
                f,
                "making {rva:#x} ({what}) writable failed after {written} site(s): {}",
                std::io::Error::from_raw_os_error(os_error)
            ),
        }
    }
}

impl std::error::Error for NotApplied {}

/// The hook for the executable whose BLAKE3 is `blake3_hex` (any case).
pub fn hook_for(blake3_hex: &str) -> Result<&'static Hook, NotApplied> {
    HOOKS
        .iter()
        .find(|h| h.build.blake3.eq_ignore_ascii_case(blake3_hex))
        .ok_or(NotApplied::UnknownBuild)
}

/// A hook checked against an image: what each of its sites holds on either
/// path.
#[derive(Debug, Clone)]
pub struct Armed {
    hook: &'static Hook,
    /// Each site's first two bytes, as the image had them.
    originals: Vec<[u8; 2]>,
}

impl Hook {
    /// Checks every site: its bytes hash as the build's, its entry is
    /// two-byte aligned. `bytes_at` gives the image's `len` bytes at `rva`,
    /// `None` outside it. Writes nothing.
    fn arm_with<'a>(
        &'static self,
        mut bytes_at: impl FnMut(usize, usize) -> Option<&'a [u8]>,
    ) -> Result<Armed, NotApplied> {
        let mut originals = Vec::with_capacity(self.sites.len());
        for s in self.sites {
            let mismatch = NotApplied::Mismatch {
                what: s.what,
                rva: s.rva,
            };
            let bytes = bytes_at(s.rva, s.checked).ok_or(mismatch)?;
            if !s.rva.is_multiple_of(2) || s.checked < 2 || fnv1a(bytes) != s.fnv1a {
                return Err(mismatch);
            }
            originals.push([bytes[0], bytes[1]]);
        }
        Ok(Armed {
            hook: self,
            originals,
        })
    }

    /// [`Self::arm_with`] on `image`, laid out as mapped (indexed by RVA).
    pub fn arm(&'static self, image: &[u8]) -> Result<Armed, NotApplied> {
        self.arm_with(|rva, len| image.get(rva..rva.checked_add(len)?))
    }
}

impl Armed {
    pub fn hook(&self) -> &'static Hook {
        self.hook
    }

    /// Each site's RVA and the two bytes it holds on `choice`: the jump on
    /// the fallback, the build's own on bvh.
    pub fn words(&self, choice: Choice) -> impl Iterator<Item = (usize, [u8; 2])> + '_ {
        self.hook
            .sites
            .iter()
            .zip(&self.originals)
            .map(move |(s, original)| {
                let word = match choice {
                    Choice::Bvh => *original,
                    Choice::Fallback => s.jump(),
                };
                (s.rva, word)
            })
    }

    /// Writes `choice` into `image`, laid out as mapped.
    pub fn apply_to(&self, image: &mut [u8], choice: Choice) {
        for (rva, word) in self.words(choice) {
            image[rva..rva + 2].copy_from_slice(&word);
        }
    }
}

#[cfg(windows)]
pub use mapped::{Installed, apply_if_asked, install};

#[cfg(windows)]
mod mapped {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::time::Duration;

    use super::{Armed, Asked, Choice, ENV, Hook, NotApplied, asked, hook_for};

    /// How often the A/B thread looks at the clock.
    const POLL: Duration = Duration::from_millis(5);

    /// A hook checked against the mapped image, ready to switch.
    #[derive(Debug)]
    pub struct Installed {
        base: usize,
        armed: Armed,
    }

    /// Checks every site of `hook` in the image mapped at `base`, reading
    /// only the checked bytes, and keeps each entry's first two bytes.
    /// Writes nothing: [`Installed::set`] does.
    ///
    /// # Safety
    /// `base..base + size` is the mapped image of the build `hook` names (or
    /// memory laid out like it), and stays mapped while the result lives.
    pub unsafe fn install(
        hook: &'static Hook,
        base: usize,
        size: usize,
    ) -> Result<Installed, NotApplied> {
        let armed = hook.arm_with(|rva, len| {
            rva.checked_add(len).filter(|&end| end <= size)?;
            // SAFETY: inside the image, in code of the build the hash names,
            // which is readable.
            Some(unsafe { std::slice::from_raw_parts((base + rva) as *const u8, len) })
        })?;
        Ok(Installed { base, armed })
    }

    impl Installed {
        /// Puts `choice` in: every site, in order. Safe while title threads
        /// run the sites (the module doc).
        pub fn set(&self, choice: Choice) -> Result<(), NotApplied> {
            let sites = self.armed.hook().sites;
            for (written, ((rva, word), s)) in self.armed.words(choice).zip(sites).enumerate() {
                // SAFETY: a two-byte aligned entry inside the image, checked
                // by `install`, whose contract keeps it mapped.
                unsafe { write_word(self.base + rva, word) }.map_err(|e| NotApplied::Write {
                    what: s.what,
                    rva,
                    written,
                    os_error: e.raw_os_error().unwrap_or(0),
                })?;
            }
            Ok(())
        }
    }

    /// Stores `word` at `at` in one atomic write.
    ///
    /// # Safety
    /// `at` is two-byte aligned mapped code, where two bytes may be
    /// replaced whole while threads run it.
    unsafe fn write_word(at: usize, word: [u8; 2]) -> std::io::Result<()> {
        use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
        use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let mut old = 0;
        // SAFETY: mapped, per the caller; only the protection of this page
        // changes (execution stays allowed), and it is restored below.
        if unsafe { VirtualProtect(at as _, 2, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: writable now and aligned, per the caller; an atomic store,
        // so a thread fetching it sees the old two bytes or the new ones.
        // Then the old protection back, and the instruction cache flushed.
        unsafe {
            AtomicU16::from_ptr(at as *mut u16).store(u16::from_le_bytes(word), Ordering::SeqCst);
            VirtualProtect(at as _, 2, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as _, 2);
        }
        Ok(())
    }

    static INSTALLED: OnceLock<Installed> = OnceLock::new();

    /// Applies the hook to the mapped title as [`ENV`] asks, and logs the
    /// outcome: the fallback at once, or a thread that follows the A/B
    /// clock. Does nothing when it is unset or `bvh`.
    ///
    /// # Safety
    /// `base..base + size` is the mapped image of the executable whose
    /// original BLAKE3 is `blake3_hex`, mapped for the rest of the process.
    pub unsafe fn apply_if_asked(blake3_hex: &str, base: usize, size: u32) {
        let asked = asked(std::env::var_os(ENV).as_deref());
        match &asked {
            Asked::Nothing => return,
            Asked::Bvh => {
                tracing::info!("{ENV}=bvh: the title's ray-tracing predicate is left as it is");
                return;
            }
            Asked::Unknown(v) => {
                tracing::warn!(
                    "{ENV}={v} is not known (fallback, bvh or ab[:<phase s>[:<settle s>]]): the title's ray-tracing predicate is left as it is"
                );
                return;
            }
            Asked::Fallback | Asked::Ab => {}
        }
        let installed = hook_for(blake3_hex).and_then(|hook| {
            // SAFETY: forwarded from this function's contract.
            unsafe { install(hook, base, size as usize) }
        });
        let installed = match installed {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!("ray-tracing path not switched ({ENV}): {e}");
                return;
            }
        };
        if INSTALLED.set(installed).is_err() {
            tracing::warn!(
                "ray-tracing path not switched ({ENV}): {}",
                NotApplied::Twice
            );
            return;
        }
        let installed = INSTALLED.get().expect("set just above");
        let build = installed.armed.hook().build.label;
        let sites = installed.armed.hook().sites.len();
        if asked == Asked::Fallback {
            match installed.set(Choice::Fallback) {
                Ok(()) => tracing::info!(
                    build,
                    sites,
                    "ray-tracing scene predicate answers no ({ENV}=fallback): the title takes its non-traced paths"
                ),
                Err(e) => tracing::warn!("ray-tracing path not switched ({ENV}): {e}"),
            }
            return;
        }
        let switch = xtp_core::ab::switch(ENV, false);
        if !switch.alternates() {
            tracing::warn!(
                "{ENV}: no A/B (its value is not one xtp_core::ab takes, see above); the title's ray-tracing predicate is left as it is"
            );
            return;
        }
        tracing::info!(
            build,
            sites,
            "ray-tracing path ({ENV}=ab): phase A {}, phase B {}",
            Choice::Bvh,
            Choice::Fallback
        );
        let spawned = std::thread::Builder::new()
            .name("xtp-rt-path".into())
            .spawn(move || {
                let mut now = Choice::Bvh;
                loop {
                    std::thread::sleep(POLL);
                    let want = Choice::of(switch.get());
                    if want == now {
                        continue;
                    }
                    if let Err(e) = installed.set(want) {
                        tracing::warn!("{ENV}: the A/B stops here: {e}");
                        return;
                    }
                    now = want;
                    let arm = xtp_core::ab::current();
                    tracing::info!("ray-tracing scene predicate: {now} (A/B arm {arm})");
                }
            });
        if let Err(e) = spawned {
            tracing::warn!("{ENV}: no A/B thread, the title's predicate is left as it is: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HASH: &str = BUILD_1_0_2_1.blake3;

    /// Two functions of our own, laid out like the title's two: the first
    /// one's first instruction is two bytes long, the second one's four;
    /// each returns 1, and has a `return 0` of its own further on, 0x12 and
    /// 0x11 bytes after its entry; int3 elsewhere.
    const CODE: [u8; 0x40] = {
        let mut c = [0xCC; 0x40];
        // 0x00: mov al, 1; ret
        c[0x00] = 0xB0;
        c[0x01] = 0x01;
        c[0x02] = 0xC3;
        // 0x12: xor eax, eax; ret
        c[0x12] = 0x31;
        c[0x13] = 0xC0;
        c[0x14] = 0xC3;
        // 0x20: mov ax, 1; ret
        c[0x20] = 0x66;
        c[0x21] = 0xB8;
        c[0x22] = 0x01;
        c[0x23] = 0x00;
        c[0x24] = 0xC3;
        // 0x31: xor eax, eax; ret
        c[0x31] = 0x31;
        c[0x32] = 0xC0;
        c[0x33] = 0xC3;
        c
    };

    /// Where [`CODE`] goes in a test image.
    const AT: usize = 0x100;

    const TEST_SITES: &[Site] = &[
        Site {
            what: "test function one",
            rva: AT,
            checked: 0x15,
            fnv1a: fnv1a(CODE.split_at(0x15).0),
            returns_false: AT + 0x12,
        },
        Site {
            what: "test function two",
            rva: AT + 0x20,
            checked: 0x14,
            fnv1a: fnv1a(CODE.split_at(0x20).1.split_at(0x14).0),
            returns_false: AT + 0x31,
        },
    ];

    static TEST_HOOK: Hook = Hook {
        build: &BUILD_1_0_2_1,
        sites: TEST_SITES,
    };

    fn image() -> Vec<u8> {
        let mut image = vec![0u8; 0x200];
        image[AT..AT + CODE.len()].copy_from_slice(&CODE);
        image
    }

    #[test]
    fn the_switch_takes_fallback_bvh_and_ab() {
        let a = |v: &str| asked(Some(OsStr::new(v)));
        assert_eq!(asked(None), Asked::Nothing);
        assert_eq!(a(""), Asked::Nothing);
        assert_eq!(a(" "), Asked::Nothing);
        assert_eq!(a("fallback"), Asked::Fallback);
        assert_eq!(a(" Fallback "), Asked::Fallback);
        assert_eq!(a("bvh"), Asked::Bvh);
        assert_eq!(a("BVH"), Asked::Bvh);
        assert_eq!(a("ab"), Asked::Ab);
        assert_eq!(a("ab:30"), Asked::Ab);
        assert_eq!(a("ab:30:5"), Asked::Ab);
        for other in ["1", "0", "on", "AB", "abc", "fallbacks", "rt"] {
            assert_eq!(a(other), Asked::Unknown(other.to_owned()), "{other}");
        }
        assert_eq!(Choice::of(false), Choice::Bvh);
        assert_eq!(Choice::of(true), Choice::Fallback);
    }

    #[test]
    fn only_the_known_build_has_a_hook() {
        assert_eq!(
            hook_for(HASH).map(|h| h.build.label),
            Ok(BUILD_1_0_2_1.label)
        );
        assert!(hook_for(&HASH.to_ascii_uppercase()).is_ok());
        for other in [
            "0".repeat(64),
            HASH[..16].to_owned(),
            format!("{}0000", &HASH[..60]),
            String::new(),
        ] {
            assert_eq!(hook_for(&other).err(), Some(NotApplied::UnknownBuild));
        }
        for hook in HOOKS {
            assert_eq!(
                crate::identify(hook.build.blake3).map(|b| b.blake3),
                Some(hook.build.blake3)
            );
        }
    }

    /// The sites are the analysis' two functions; each jump lands on the
    /// function's own `return 0`, inside the checked bytes and past the two
    /// it writes; no site is written across a page.
    #[test]
    fn sites_jump_to_their_own_return_inside_the_checked_bytes() {
        let hook = hook_for(HASH).unwrap();
        let entries: Vec<usize> = hook.sites.iter().map(|s| s.rva).collect();
        assert_eq!(entries, [0x4fd8550, 0x4fd8570]);
        for hook in HOOKS.iter().chain([&TEST_HOOK]) {
            for (i, s) in hook.sites.iter().enumerate() {
                assert_eq!(s.rva % 2, 0, "{}", s.what);
                assert_eq!(s.rva / 0x1000, (s.rva + 1) / 0x1000, "{}", s.what);
                // A `return 0` takes at least three bytes.
                assert!(s.returns_false >= s.rva + 2, "{}", s.what);
                assert!(s.returns_false + 3 <= s.rva + s.checked, "{}", s.what);
                let [op, rel] = s.jump();
                assert_eq!(op, 0xEB, "{}", s.what);
                assert_eq!(s.rva + 2 + rel as usize, s.returns_false, "{}", s.what);
                for t in &hook.sites[i + 1..] {
                    let apart = s.rva + s.checked <= t.rva || t.rva + t.checked <= s.rva;
                    assert!(apart, "{} overlaps {}", s.what, t.what);
                }
            }
        }
        let hook = hook_for(HASH).unwrap();
        assert_eq!(hook.sites[0].jump(), [0xEB, 0x10]);
        assert_eq!(hook.sites[1].jump(), [0xEB, 0x0F]);
    }

    #[test]
    fn fnv1a_is_the_64_bit_one() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn the_fallback_jumps_and_bvh_puts_the_bytes_back() {
        let original = image();
        let armed = TEST_HOOK.arm(&original).unwrap();
        let mut img = original.clone();
        armed.apply_to(&mut img, Choice::Fallback);
        assert_eq!(img[AT..AT + 2], [0xEB, 0x10]);
        assert_eq!(img[AT + 0x20..AT + 0x22], [0xEB, 0x0F]);
        // Only the two bytes of each entry changed.
        let changed: Vec<usize> = (0..img.len()).filter(|&i| img[i] != original[i]).collect();
        assert_eq!(changed, [AT, AT + 1, AT + 0x20, AT + 0x21]);
        // Patched, the image no longer arms: a second install is refused.
        assert_eq!(
            TEST_HOOK.arm(&img).err(),
            Some(NotApplied::Mismatch {
                what: "test function one",
                rva: AT
            })
        );
        armed.apply_to(&mut img, Choice::Bvh);
        assert_eq!(img, original);
        // Back and forth.
        for _ in 0..3 {
            armed.apply_to(&mut img, Choice::Fallback);
            armed.apply_to(&mut img, Choice::Bvh);
        }
        assert_eq!(img, original);
    }

    #[test]
    fn any_differing_checked_byte_refuses_both_sites() {
        for s in TEST_SITES {
            for i in 0..s.checked {
                let mut img = image();
                img[s.rva + i] ^= 0x01;
                assert_eq!(
                    TEST_HOOK.arm(&img).err(),
                    Some(NotApplied::Mismatch {
                        what: s.what,
                        rva: s.rva
                    }),
                    "{} byte {i}",
                    s.what
                );
            }
        }
        // A byte past the checked ones does not matter.
        let mut img = image();
        img[AT + 0x15] ^= 0x01;
        assert!(TEST_HOOK.arm(&img).is_ok());
        // An image that ends inside a site.
        let mut cut = image();
        cut.truncate(AT + 0x20 + 0x13);
        assert_eq!(
            TEST_HOOK.arm(&cut).err(),
            Some(NotApplied::Mismatch {
                what: "test function two",
                rva: AT + 0x20
            })
        );
        // The real hook on an image that ends before its sites.
        let hook = hook_for(HASH).unwrap();
        assert!(matches!(
            hook.arm(&[0u8; 0x100]),
            Err(NotApplied::Mismatch { rva: 0x4fd8550, .. })
        ));
    }

    /// [`install`] and [`Installed::set`] on real code of our own, in a page
    /// made `PAGE_EXECUTE_READ` like an image's `.text`: the functions
    /// answer 1 on bvh and 0 on the fallback, the page keeps its
    /// protection, and two threads calling them while the path flips (4000
    /// times at least, and until each thread has made 1000 rounds of calls)
    /// only ever get 0 or 1.
    #[cfg(windows)]
    #[test]
    fn set_flips_live_code_and_keeps_its_protection() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
        use std::time::{Duration, Instant};

        use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ,
            PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect, VirtualQuery,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        // SAFETY: a fresh page, freed at the end.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        // SAFETY: inside the page, writable for now; then made code.
        unsafe {
            std::ptr::copy_nonoverlapping(CODE.as_ptr(), (base + AT) as *mut u8, CODE.len());
            let mut old = 0;
            assert_ne!(
                VirtualProtect(base as _, 0x1000, PAGE_EXECUTE_READ, &mut old),
                0
            );
            FlushInstructionCache(GetCurrentProcess(), base as _, 0x1000);
        }
        let protection = || {
            // SAFETY: a plain query into a local the call fills.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: as above.
            let n = unsafe {
                VirtualQuery(
                    (base + AT) as _,
                    &mut info,
                    size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            assert_ne!(n, 0);
            info.Protect
        };
        type Predicate = extern "system" fn() -> u8;
        // SAFETY: the two functions of CODE, which take nothing and return
        // a byte in al.
        let (one, two) = unsafe {
            (
                std::mem::transmute::<usize, Predicate>(base + AT),
                std::mem::transmute::<usize, Predicate>(base + AT + 0x20),
            )
        };
        assert_eq!((one(), two()), (1, 1));

        // SAFETY: the page is laid out as TEST_HOOK says, and freed only
        // after `installed` is gone.
        let installed = unsafe { install(&TEST_HOOK, base, 0x1000) }.unwrap();
        assert_eq!((one(), two()), (1, 1), "install writes nothing");
        installed.set(Choice::Fallback).unwrap();
        assert_eq!((one(), two()), (0, 0));
        assert_eq!(protection(), PAGE_EXECUTE_READ);
        installed.set(Choice::Bvh).unwrap();
        assert_eq!((one(), two()), (1, 1));
        assert_eq!(protection(), PAGE_EXECUTE_READ);

        // Flips go on until each caller has run a while beside them: on a
        // loaded machine a caller may start late (the check's farm, where
        // 4000 flips once ended before either had made a call).
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new([AtomicU64::new(0), AtomicU64::new(0)]);
        let callers: Vec<_> = (0..2)
            .map(|n| {
                let (stop, calls) = (Arc::clone(&stop), Arc::clone(&calls));
                std::thread::spawn(move || {
                    let mut seen = [0u64; 2];
                    while !stop.load(SeqCst) {
                        for answer in [one(), two()] {
                            assert!(answer <= 1, "answered {answer}");
                            seen[answer as usize] += 1;
                        }
                        calls[n].fetch_add(1, SeqCst);
                    }
                    seen
                })
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut flips = 0u64;
        while flips < 4000 || calls.iter().any(|c| c.load(SeqCst) < 1000) {
            if Instant::now() > deadline {
                break;
            }
            installed.set(Choice::of(flips.is_multiple_of(2))).unwrap();
            flips += 1;
            std::thread::yield_now();
        }
        stop.store(true, SeqCst);
        for (n, caller) in callers.into_iter().enumerate() {
            let seen = caller.join().expect("a caller failed");
            assert!(
                seen[0] + seen[1] > 0,
                "caller {n} made no call in 60 s of {flips} flips"
            );
        }
        installed.set(Choice::Bvh).unwrap();
        assert_eq!((one(), two()), (1, 1), "bvh put back");

        drop(installed);
        // SAFETY: our page, no longer run.
        assert_ne!(unsafe { VirtualFree(base as _, 0, MEM_RELEASE) }, 0);
    }

    /// Against a real dump, when `XTP_SAMPLE_GAME` names its folder: for a
    /// build the hook knows, each site hashes as expected at its RVA, in
    /// executable code; the hook arms on that copy, and the fallback then
    /// bvh gives the image back. Messages name offsets only, never the
    /// title's bytes.
    #[test]
    fn sites_in_the_sample_game() {
        let Some(dir) = std::env::var_os("XTP_SAMPLE_GAME") else {
            println!("SKIPPED sites_in_the_sample_game: XTP_SAMPLE_GAME is not set");
            return;
        };
        let exe = std::path::Path::new(&dir).join("game_scarlett_master.exe");
        let Ok(file) = std::fs::read(&exe) else {
            println!("SKIPPED sites_in_the_sample_game: no {}", exe.display());
            return;
        };
        let hash = blake3::hash(&file).to_hex();
        let Ok(hook) = hook_for(&hash) else {
            println!("build {hash} has no ray-tracing path hook: nothing to check");
            return;
        };
        let info = xtp_pe::ImageInfo::parse(&file).expect("parsing the executable");
        let mut image = vec![0u8; info.size_of_image as usize];
        for s in &info.sections {
            let len = s.file_size.min(s.virtual_size) as usize;
            let (from, to) = (s.file_offset as usize, s.rva as usize);
            image[to..to + len].copy_from_slice(&file[from..from + len]);
        }
        for s in hook.sites {
            let code = info.sections.iter().any(|c| {
                let start = c.rva as usize;
                (start..start + c.virtual_size as usize).contains(&s.rva)
                    && (start..=start + c.virtual_size as usize).contains(&(s.rva + s.checked))
                    && c.characteristics & 0x2000_0000 != 0 // IMAGE_SCN_MEM_EXECUTE
            });
            assert!(code, "{} at {:#x} is not in executable code", s.what, s.rva);
        }
        let armed = hook.arm(&image).expect("the sites hash as the build's");
        let original = image.clone();
        armed.apply_to(&mut image, Choice::Fallback);
        assert!(hook.arm(&image).is_err(), "armed again once patched");
        armed.apply_to(&mut image, Choice::Bvh);
        assert!(image == original, "bvh did not give the image back");
        println!(
            "build {}: {} sites check",
            hook.build.label,
            hook.sites.len()
        );
    }
}
