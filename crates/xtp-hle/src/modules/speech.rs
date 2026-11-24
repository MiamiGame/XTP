//! Accessibility: closed captions, high contrast, and speech synthesis.
//!
//! Closed-caption properties are read from Windows via the WinRT
//! `Windows.Media.ClosedCaptioning.ClosedCaptionProperties` API (cached on
//! first query, with fallback defaults when unavailable). High-contrast mode
//! is queried through `SystemParametersInfoW(SPI_GETHIGHCONTRAST)`. Speech
//! synthesis currently emits valid empty WAVE audio so narration is silent.
// shortcut: silent speech. The host's engine (SAPI through COM, or WinRT
// speech synthesis) needs COM bindings this crate has only in diagnostics
// builds, plus voice enumeration and SSML the title sends (slot 0xa8); the
// title then plays the empty stream and goes on.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cmp::Ordering as CmpOrdering;
use std::ffi::{c_char, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, Ordering};

use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

/// Closed captions, high contrast, speech to text and synthesis.
pub const PROVIDER: Guid = Guid::new(
    0x3e24_1977,
    0x6237,
    0x41e9,
    [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1],
);
/// Unchanged since 2020.
pub const IID: Guid = Guid::new(
    0xd722_b373,
    0x8c4d,
    0x4692,
    [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d],
);

pub static OBJECT: Object = Object::new("accessibility", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT)];

/// ARGB colour (`XColor` in `XGameRuntimeTypes.h`).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XColor {
    pub a: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Closed-caption settings (`XClosedCaptionProperties` in `XAccessibility.h`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XClosedCaptionProperties {
    pub background_color: XColor,
    pub font_color: XColor,
    pub window_color: XColor,
    pub font_edge_attribute: u32,
    pub font_style: u32,
    pub font_scale: f32,
    pub enabled: bool,
}

impl XClosedCaptionProperties {
    /// Fallback properties when Windows exposes no explicit caption setting:
    /// opaque black background, opaque white text, transparent window, default
    /// edge and font style, 100% scale, disabled.
    pub const FALLBACK: Self = Self {
        background_color: XColor {
            a: 0xff,
            r: 0x00,
            g: 0x00,
            b: 0x00,
        },
        font_color: XColor {
            a: 0xff,
            r: 0xff,
            g: 0xff,
            b: 0xff,
        },
        window_color: XColor {
            a: 0x00,
            r: 0x00,
            g: 0x00,
            b: 0x00,
        },
        font_edge_attribute: 0,
        font_style: 0,
        font_scale: 1.0,
        enabled: false,
    };
}

/// `XHighContrastMode` values in `XAccessibility.h`.
pub const HIGH_CONTRAST_OFF: u32 = 0;
pub const HIGH_CONTRAST_DARK: u32 = 1;
pub const HIGH_CONTRAST_LIGHT: u32 = 2;
pub const HIGH_CONTRAST_OTHER: u32 = 3;

/// Cached host closed-caption snapshot so per-frame polling stays cheap.
static HOST_CAPTIONS: OnceLock<XClosedCaptionProperties> = OnceLock::new();

/// Title override for closed-caption enabled state: `0` = unset, `1` = false,
/// `2` = true.
static CAPTION_ENABLED_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Intermediate snapshot of WinRT `ClosedCaptionProperties` enum values.
#[derive(Default)]
struct WinrtCaptionSnapshot {
    background_color: Option<i32>,
    background_opacity: Option<i32>,
    font_color: Option<i32>,
    font_opacity: Option<i32>,
    region_color: Option<i32>,
    region_opacity: Option<i32>,
    font_effect: Option<i32>,
    font_style: Option<i32>,
    font_size: Option<i32>,
}

fn resolve_color(color: Option<i32>, opacity: Option<i32>, fallback: XColor) -> XColor {
    // Windows.Media.ClosedCaptioning.ClosedCaptionColor enum order:
    // 0=Default, 1=White, 2=Black, 3=Red, 4=Green, 5=Blue, 6=Yellow, 7=Magenta, 8=Cyan.
    let (r, g, b) = match color {
        Some(1) => (0xff, 0xff, 0xff),
        Some(2) => (0x00, 0x00, 0x00),
        Some(3) => (0xff, 0x00, 0x00),
        Some(4) => (0x00, 0xff, 0x00),
        Some(5) => (0x00, 0x00, 0xff),
        Some(6) => (0xff, 0xff, 0x00),
        Some(7) => (0xff, 0x00, 0xff),
        Some(8) => (0x00, 0xff, 0xff),
        _ => (fallback.r, fallback.g, fallback.b),
    };
    // Windows.Media.ClosedCaptioning.ClosedCaptionOpacity enum order:
    // 0=Default, 1=OneHundredPercent, 2=SeventyFivePercent, 3=TwentyFivePercent, 4=ZeroPercent.
    let a = match opacity {
        Some(1) => 0xff,
        Some(2) => 0xbf,
        Some(3) => 0x40,
        Some(4) => 0x00,
        _ => fallback.a,
    };
    XColor { a, r, g, b }
}

fn resolve_scale(size: Option<i32>) -> f32 {
    // Windows.Media.ClosedCaptioning.ClosedCaptionSize:
    // 0=Default, 1=FiftyPercent, 2=OneHundredPercent, 3=OneHundredFiftyPercent, 4=TwoHundredPercent.
    match size {
        Some(1) => 0.5,
        Some(2) => 1.0,
        Some(3) => 1.5,
        Some(4) => 2.0,
        _ => XClosedCaptionProperties::FALLBACK.font_scale,
    }
}

fn build_caption_properties(
    snap: &WinrtCaptionSnapshot,
    enabled_override: Option<bool>,
) -> XClosedCaptionProperties {
    let base = XClosedCaptionProperties::FALLBACK;
    XClosedCaptionProperties {
        background_color: resolve_color(
            snap.background_color,
            snap.background_opacity,
            base.background_color,
        ),
        font_color: resolve_color(snap.font_color, snap.font_opacity, base.font_color),
        window_color: resolve_color(snap.region_color, snap.region_opacity, base.window_color),
        font_edge_attribute: snap
            .font_effect
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v <= 5)
            .unwrap_or(base.font_edge_attribute),
        font_style: snap
            .font_style
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v <= 7)
            .unwrap_or(base.font_style),
        font_scale: resolve_scale(snap.font_size),
        enabled: enabled_override.unwrap_or(base.enabled),
    }
}

/// Distinguish dark vs light high-contrast themes by comparing window
/// background brightness against foreground text brightness.
fn high_contrast_from_pair(active: bool, bg_bgr: u32, fg_bgr: u32) -> u32 {
    if !active {
        return HIGH_CONTRAST_OFF;
    }
    let brightness = |bgr: u32| (bgr & 0xff) + ((bgr >> 8) & 0xff) + ((bgr >> 16) & 0xff);
    match brightness(bg_bgr).cmp(&brightness(fg_bgr)) {
        CmpOrdering::Less => HIGH_CONTRAST_DARK,
        CmpOrdering::Greater => HIGH_CONTRAST_LIGHT,
        CmpOrdering::Equal => HIGH_CONTRAST_OTHER,
    }
}

#[cfg(windows)]
fn query_winrt_captions() -> WinrtCaptionSnapshot {
    use windows::Media::ClosedCaptioning::ClosedCaptionProperties;

    WinrtCaptionSnapshot {
        background_color: ClosedCaptionProperties::BackgroundColor().ok().map(|v| v.0),
        background_opacity: ClosedCaptionProperties::BackgroundOpacity()
            .ok()
            .map(|v| v.0),
        font_color: ClosedCaptionProperties::FontColor().ok().map(|v| v.0),
        font_opacity: ClosedCaptionProperties::FontOpacity().ok().map(|v| v.0),
        region_color: ClosedCaptionProperties::RegionColor().ok().map(|v| v.0),
        region_opacity: ClosedCaptionProperties::RegionOpacity().ok().map(|v| v.0),
        font_effect: ClosedCaptionProperties::FontEffect().ok().map(|v| v.0),
        font_style: ClosedCaptionProperties::FontStyle().ok().map(|v| v.0),
        font_size: ClosedCaptionProperties::FontSize().ok().map(|v| v.0),
    }
}

#[cfg(not(windows))]
fn query_winrt_captions() -> WinrtCaptionSnapshot {
    WinrtCaptionSnapshot::default()
}

#[cfg(windows)]
fn query_host_high_contrast() -> u32 {
    use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, COLOR_WINDOWTEXT, GetSysColor};
    use windows_sys::Win32::UI::Accessibility::{HCF_HIGHCONTRASTON, HIGHCONTRASTW};
    use windows_sys::Win32::UI::WindowsAndMessaging::{SPI_GETHIGHCONTRAST, SystemParametersInfoW};

    let mut info = HIGHCONTRASTW {
        cbSize: std::mem::size_of::<HIGHCONTRASTW>() as u32,
        dwFlags: 0,
        lpszDefaultScheme: std::ptr::null_mut(),
    };
    // SAFETY: `info` has its exact byte size in `cbSize` and valid storage.
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETHIGHCONTRAST,
            info.cbSize,
            (&raw mut info).cast::<c_void>(),
            0,
        )
    };
    if ok == 0 {
        return HIGH_CONTRAST_OFF;
    }
    let active = (info.dwFlags & HCF_HIGHCONTRASTON) != 0;
    // SAFETY: `GetSysColor` takes standard `COLOR_*` indices with no preconditions.
    let (bg, fg) = unsafe { (GetSysColor(COLOR_WINDOW), GetSysColor(COLOR_WINDOWTEXT)) };
    high_contrast_from_pair(active, bg, fg)
}

#[cfg(not(windows))]
fn query_host_high_contrast() -> u32 {
    HIGH_CONTRAST_OFF
}

fn enabled_override() -> Option<bool> {
    match CAPTION_ENABLED_OVERRIDE.load(Ordering::Relaxed) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

extern "system" fn closed_caption_get_properties(
    _: This,
    out: *mut XClosedCaptionProperties,
) -> HRESULT {
    let mut props = *HOST_CAPTIONS.get_or_init(|| {
        let snap = query_winrt_captions();
        build_caption_properties(&snap, None)
    });
    if let Some(overridden) = enabled_override() {
        props.enabled = overridden;
    }
    put(out, props)
}

/// Takes `u8` rather than `bool` so any non-zero C++ byte is accepted without UB.
extern "system" fn closed_caption_set_enabled(_: This, enabled: u8) -> HRESULT {
    CAPTION_ENABLED_OVERRIDE.store(if enabled != 0 { 2 } else { 1 }, Ordering::Relaxed);
    S_OK
}

extern "system" fn high_contrast_get_mode(_: This, out: *mut u32) -> HRESULT {
    put(out, query_host_high_contrast())
}

/// A WAVE file of 16 kHz mono 16-bit PCM with no samples: RIFF size 36,
/// a 16-byte format chunk (PCM, 1 channel, 16,000 Hz, 32,000 bytes/s,
/// block 2, 16 bits), an empty data chunk.
const SILENCE: [u8; 44] =
    *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";

/// Synthesizers and streams hold nothing, so their handles are fixed
/// addresses and closing them is free.
static SYNTHESIZER: u8 = 0;
static STREAM: u8 = 0;

type This = *const Object;

/// No voice is installed, so the callback never runs.
extern "system" fn enumerate_voices(
    _: This,
    _context: *mut c_void,
    _callback: *const c_void,
) -> HRESULT {
    S_OK
}
extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    put(out, &raw const SYNTHESIZER)
}
extern "system" fn close(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_default_voice(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_custom_voice(_: This, _: *const u8, _voice: *const c_char) -> HRESULT {
    S_OK
}
extern "system" fn create_stream(
    _: This,
    _: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    tracing::debug!(text = text(words).unwrap_or_default(), "speech (silent)");
    put(out, &raw const STREAM)
}
extern "system" fn stream_size(_: This, _: *const u8, out: *mut usize) -> HRESULT {
    put(out, SILENCE.len())
}
extern "system" fn stream_data(
    _: This,
    _: *const u8,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    if buffer.is_null() || size < SILENCE.len() {
        return super::xasync::E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe { buffer.copy_from_nonoverlapping(SILENCE.as_ptr(), SILENCE.len()) };
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(SILENCE.len()) };
    }
    S_OK
}

static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    closed_caption_get_properties as *const (), // 0x18 XClosedCaptionGetProperties
    closed_caption_set_enabled as *const (),    // 0x20 XClosedCaptionSetEnabled
    high_contrast_get_mode as *const (),        // 0x28 XHighContrastGetMode
    missing::<6> as *const (),                  // 0x30 speech to text
    missing::<7> as *const (),                  // 0x38
    enumerate_voices as *const (),              // 0x40
    create as *const (),                        // 0x48
    close as *const (),                         // 0x50
    set_default_voice as *const (),             // 0x58
    set_custom_voice as *const (),              // 0x60
    create_stream as *const (),                 // 0x68 from text
    close as *const (),                         // 0x70 a stream
    stream_size as *const (),                   // 0x78
    stream_data as *const (),                   // 0x80
    missing::<17> as *const (),                 // 0x88 speech to text
    missing::<18> as *const (),                 // 0x90
    missing::<19> as *const (),                 // 0x98
    missing::<20> as *const (),                 // 0xa0
    create_stream as *const (),                 // 0xa8 from SSML
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn streams_are_empty_wave_files() {
        let this: This = &OBJECT;
        type Create = extern "system" fn(This, *mut *const u8) -> HRESULT;
        type Stream = extern "system" fn(This, *const u8, *const c_char, *mut *const u8) -> HRESULT;
        type Size = extern "system" fn(This, *const u8, *mut usize) -> HRESULT;
        type Data = extern "system" fn(This, *const u8, usize, *mut u8, *mut usize) -> HRESULT;
        let (mut synth, mut stream, mut size) = (std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let say = method::<Stream>(&OBJECT, 0x68);
        assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
        assert_eq!(method::<Size>(&OBJECT, 0x78)(this, stream, &mut size), S_OK);
        let mut wav = vec![0u8; size];
        let mut used = 0;
        let data = method::<Data>(&OBJECT, 0x80);
        assert_eq!(data(this, stream, size, wav.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(
            (&wav[..4], &wav[8..16], &wav[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(used, 44);
    }

    #[test]
    fn empty_winrt_snapshot_uses_fallback_properties() {
        let props = build_caption_properties(&WinrtCaptionSnapshot::default(), None);
        assert_eq!(props, XClosedCaptionProperties::FALLBACK);
        assert_eq!(std::mem::size_of_val(&props), 28);
    }

    #[test]
    fn winrt_snapshot_values_translate_to_caption_properties() {
        let snap = WinrtCaptionSnapshot {
            background_color: Some(5),   // Blue
            background_opacity: Some(2), // 75%
            font_color: Some(6),         // Yellow
            font_opacity: Some(1),       // 100%
            region_color: Some(2),       // Black
            region_opacity: Some(3),     // 25%
            font_effect: Some(4),        // UniformedEdges
            font_style: Some(3),         // MonospacedWithoutSerifs
            font_size: Some(4),          // 200%
        };
        let props = build_caption_properties(&snap, Some(true));
        assert_eq!(
            props.background_color,
            XColor {
                a: 0xbf,
                r: 0,
                g: 0,
                b: 0xff
            }
        );
        assert_eq!(
            props.font_color,
            XColor {
                a: 0xff,
                r: 0xff,
                g: 0xff,
                b: 0
            }
        );
        assert_eq!(
            props.window_color,
            XColor {
                a: 0x40,
                r: 0,
                g: 0,
                b: 0
            }
        );
        assert_eq!(props.font_edge_attribute, 4);
        assert_eq!(props.font_style, 3);
        assert_eq!(props.font_scale, 2.0);
        assert!(props.enabled);
    }

    #[test]
    fn set_enabled_round_trips_and_accepts_any_nonzero_byte() {
        let this: This = &OBJECT;
        type GetProps = extern "system" fn(This, *mut XClosedCaptionProperties) -> HRESULT;
        type SetEnabled = extern "system" fn(This, u8) -> HRESULT;
        let get = method::<GetProps>(&OBJECT, 0x18);
        let set = method::<SetEnabled>(&OBJECT, 0x20);

        assert!(get(this, std::ptr::null_mut()) < 0);

        assert_eq!(set(this, 0xff), S_OK);
        let mut props = XClosedCaptionProperties::FALLBACK;
        assert_eq!(get(this, &mut props), S_OK);
        assert!(props.enabled);

        assert_eq!(set(this, 0), S_OK);
        assert_eq!(get(this, &mut props), S_OK);
        assert!(!props.enabled);
    }

    #[test]
    fn high_contrast_pair_distinguishes_off_dark_light_and_other() {
        assert_eq!(
            high_contrast_from_pair(false, 0x0010_1010, 0x00f0_f0f0),
            HIGH_CONTRAST_OFF
        );
        assert_eq!(
            high_contrast_from_pair(true, 0x0010_1010, 0x00f0_f0f0),
            HIGH_CONTRAST_DARK
        );
        assert_eq!(
            high_contrast_from_pair(true, 0x00f0_f0f0, 0x0010_1010),
            HIGH_CONTRAST_LIGHT
        );
        assert_eq!(
            high_contrast_from_pair(true, 0x0080_8080, 0x0080_8080),
            HIGH_CONTRAST_OTHER
        );

        let this: This = &OBJECT;
        type GetMode = extern "system" fn(This, *mut u32) -> HRESULT;
        let get = method::<GetMode>(&OBJECT, 0x28);
        assert!(get(this, std::ptr::null_mut()) < 0);
        let mut mode = 99;
        assert_eq!(get(this, &mut mode), S_OK);
        assert!(mode <= HIGH_CONTRAST_OTHER);
    }
}
