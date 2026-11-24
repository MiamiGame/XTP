//! Accessibility: closed captions, high contrast, and speech synthesis.
//!
//! Closed-caption properties and high-contrast mode are read from the host
//! Windows accessibility settings (with documented defaults when Windows has
//! no explicit setting). Speech synthesis currently emits valid empty WAVE
//! audio so narration is silent.
// shortcut: silent speech. The host's engine (SAPI through COM, or WinRT
// speech synthesis) needs COM bindings this crate has only in diagnostics
// builds, plus voice enumeration and SSML the title sends (slot 0xa8); the
// title then plays the empty stream and goes on.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
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

/// ARGB colour (`XColor` in the public GDK accessibility API).
#[repr(C, align(4))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XColor {
    pub a: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Closed-caption configuration (`XClosedCaptionProperties` in `XAccessibility.h`).
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
    /// Documented defaults used when the host has no explicit caption settings:
    /// opaque black background, opaque white text, transparent window, default
    /// edge and font style, 100% scale, disabled.
    pub const DEFAULT: Self = Self {
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

/// Title override for closed-caption enabled state: `0` = unset, `1` = false,
/// `2` = true.
static CAPTION_ENABLED_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Raw values read from `HKCU\Software\Microsoft\Windows\CurrentVersion\ClosedCaptioning`.
#[derive(Default)]
struct HostCaptionSettings {
    background_color: Option<u32>,
    background_opacity: Option<u32>,
    font_color: Option<u32>,
    font_opacity: Option<u32>,
    region_color: Option<u32>,
    region_opacity: Option<u32>,
    font_effect: Option<u32>,
    font_style: Option<u32>,
    font_size: Option<u32>,
}

fn map_caption_color(code: Option<u32>, opacity: Option<u32>, fallback: XColor) -> XColor {
    let (r, g, b) = match code {
        Some(1) => (0xff, 0xff, 0xff), // White
        Some(2) => (0x00, 0x00, 0x00), // Black
        Some(3) => (0xff, 0x00, 0x00), // Red
        Some(4) => (0x00, 0xff, 0x00), // Green
        Some(5) => (0x00, 0x00, 0xff), // Blue
        Some(6) => (0xff, 0xff, 0x00), // Yellow
        Some(7) => (0xff, 0x00, 0xff), // Magenta
        Some(8) => (0x00, 0xff, 0xff), // Cyan
        _ => (fallback.r, fallback.g, fallback.b),
    };
    let a = match opacity {
        Some(1) => 0xff, // 100%
        Some(2) => 0xbf, // 75%
        Some(3) => 0x40, // 25%
        Some(4) => 0x00, // 0%
        _ => fallback.a,
    };
    XColor { a, r, g, b }
}

fn map_font_scale(code: Option<u32>) -> f32 {
    match code {
        Some(1) => 0.5,
        Some(2) => 1.0,
        Some(3) => 1.5,
        Some(4) => 2.0,
        _ => XClosedCaptionProperties::DEFAULT.font_scale,
    }
}

fn properties_from_host(
    host: &HostCaptionSettings,
    enabled_override: Option<bool>,
) -> XClosedCaptionProperties {
    let def = XClosedCaptionProperties::DEFAULT;
    XClosedCaptionProperties {
        background_color: map_caption_color(
            host.background_color,
            host.background_opacity,
            def.background_color,
        ),
        font_color: map_caption_color(host.font_color, host.font_opacity, def.font_color),
        window_color: map_caption_color(host.region_color, host.region_opacity, def.window_color),
        font_edge_attribute: host
            .font_effect
            .filter(|&v| v <= 5)
            .unwrap_or(def.font_edge_attribute),
        font_style: host
            .font_style
            .filter(|&v| v <= 7)
            .unwrap_or(def.font_style),
        font_scale: map_font_scale(host.font_size),
        enabled: enabled_override.unwrap_or(def.enabled),
    }
}

fn classify_high_contrast(enabled: bool, window_rgb: u32) -> u32 {
    if !enabled {
        return HIGH_CONTRAST_OFF;
    }
    let r = window_rgb & 0xff;
    let g = (window_rgb >> 8) & 0xff;
    let b = (window_rgb >> 16) & 0xff;
    // Perceived luminance (Rec. 601) to distinguish dark vs light themes.
    let luma = (299 * r + 587 * g + 114 * b) / 1000;
    if luma < 128 {
        HIGH_CONTRAST_DARK
    } else {
        HIGH_CONTRAST_LIGHT
    }
}

#[cfg(windows)]
fn read_host_caption_settings() -> HostCaptionSettings {
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_CURRENT_USER, KEY_READ, REG_DWORD, RegCloseKey, RegOpenKeyExW, RegQueryValueExW,
    };

    let subkey: Vec<u16> = "Software\\Microsoft\\Windows\\CurrentVersion\\ClosedCaptioning\0"
        .encode_utf16()
        .collect();
    let mut key: HKEY = std::ptr::null_mut();
    // SAFETY: `subkey` is NUL-terminated UTF-16 and `key` is a valid out-pointer.
    let status =
        unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, subkey.as_ptr(), 0, KEY_READ, &mut key) };
    if status != ERROR_SUCCESS {
        return HostCaptionSettings::default();
    }

    let query_u32 = |name: &str| -> Option<u32> {
        let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
        let mut kind = 0u32;
        let mut value = 0u32;
        let mut len = std::mem::size_of::<u32>() as u32;
        // SAFETY: `key` is an open HKEY, `wide` is NUL-terminated, and `value`
        // holds `len` bytes.
        let rc = unsafe {
            RegQueryValueExW(
                key,
                wide.as_ptr(),
                std::ptr::null(),
                &mut kind,
                (&raw mut value).cast::<u8>(),
                &mut len,
            )
        };
        (rc == ERROR_SUCCESS && kind == REG_DWORD && len == 4).then_some(value)
    };

    let settings = HostCaptionSettings {
        background_color: query_u32("BackgroundColor"),
        background_opacity: query_u32("BackgroundOpacity"),
        font_color: query_u32("FontColor"),
        font_opacity: query_u32("FontOpacity"),
        region_color: query_u32("RegionColor"),
        region_opacity: query_u32("RegionOpacity"),
        font_effect: query_u32("FontEffect"),
        font_style: query_u32("FontStyle"),
        font_size: query_u32("FontSize"),
    };

    // SAFETY: `key` was opened by `RegOpenKeyExW` above.
    unsafe { RegCloseKey(key) };
    settings
}

#[cfg(not(windows))]
fn read_host_caption_settings() -> HostCaptionSettings {
    HostCaptionSettings::default()
}

#[cfg(windows)]
fn read_host_high_contrast_mode() -> u32 {
    use windows_sys::Win32::Graphics::Gdi::{COLOR_WINDOW, GetSysColor};
    use windows_sys::Win32::UI::Accessibility::{HCF_HIGHCONTRASTON, HIGHCONTRASTW};
    use windows_sys::Win32::UI::WindowsAndMessaging::{SPI_GETHIGHCONTRAST, SystemParametersInfoW};

    let mut hc = HIGHCONTRASTW {
        cbSize: std::mem::size_of::<HIGHCONTRASTW>() as u32,
        dwFlags: 0,
        lpszDefaultScheme: std::ptr::null_mut(),
    };
    // SAFETY: `hc` is zero-initialised with its exact `cbSize`.
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETHIGHCONTRAST,
            hc.cbSize,
            (&raw mut hc).cast::<c_void>(),
            0,
        )
    };
    if ok == 0 {
        return HIGH_CONTRAST_OFF;
    }
    let enabled = (hc.dwFlags & HCF_HIGHCONTRASTON) != 0;
    // SAFETY: `GetSysColor` takes a standard `COLOR_*` index and has no preconditions.
    let window_rgb = unsafe { GetSysColor(COLOR_WINDOW) };
    classify_high_contrast(enabled, window_rgb)
}

#[cfg(not(windows))]
fn read_host_high_contrast_mode() -> u32 {
    HIGH_CONTRAST_OFF
}

fn caption_enabled_override() -> Option<bool> {
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
    let host = read_host_caption_settings();
    let props = properties_from_host(&host, caption_enabled_override());
    put(out, props)
}

extern "system" fn closed_caption_set_enabled(_: This, enabled: bool) -> HRESULT {
    CAPTION_ENABLED_OVERRIDE.store(if enabled { 2 } else { 1 }, Ordering::Relaxed);
    S_OK
}

extern "system" fn high_contrast_get_mode(_: This, out: *mut u32) -> HRESULT {
    put(out, read_host_high_contrast_mode())
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
    fn closed_caption_defaults_fill_entire_structure() {
        assert_eq!(std::mem::size_of::<XColor>(), 4);
        assert_eq!(std::mem::align_of::<XColor>(), 4);
        assert_eq!(std::mem::size_of::<XClosedCaptionProperties>(), 28);

        let props = properties_from_host(&HostCaptionSettings::default(), None);
        assert_eq!(props, XClosedCaptionProperties::DEFAULT);
        assert_eq!(
            props.background_color,
            XColor {
                a: 0xff,
                r: 0,
                g: 0,
                b: 0
            }
        );
        assert_eq!(
            props.font_color,
            XColor {
                a: 0xff,
                r: 0xff,
                g: 0xff,
                b: 0xff
            }
        );
        assert_eq!(
            props.window_color,
            XColor {
                a: 0,
                r: 0,
                g: 0,
                b: 0
            }
        );
        assert_eq!(props.font_edge_attribute, 0);
        assert_eq!(props.font_style, 0);
        assert_eq!(props.font_scale, 1.0);
        assert!(!props.enabled);
    }

    #[test]
    fn closed_caption_maps_host_registry_values() {
        let host = HostCaptionSettings {
            background_color: Some(5),   // Blue
            background_opacity: Some(2), // 75%
            font_color: Some(6),         // Yellow
            font_opacity: Some(1),       // 100%
            region_color: Some(2),       // Black
            region_opacity: Some(3),     // 25%
            font_effect: Some(5),        // DropShadowedEdges
            font_style: Some(4),         // ProportionalWithoutSerifs
            font_size: Some(3),          // 150%
        };
        let props = properties_from_host(&host, Some(true));
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
        assert_eq!(props.font_edge_attribute, 5);
        assert_eq!(props.font_style, 4);
        assert_eq!(props.font_scale, 1.5);
        assert!(props.enabled);
    }

    #[test]
    fn closed_caption_set_enabled_is_read_back() {
        let this: This = &OBJECT;
        type GetProps = extern "system" fn(This, *mut XClosedCaptionProperties) -> HRESULT;
        type SetEnabled = extern "system" fn(This, bool) -> HRESULT;
        let get = method::<GetProps>(&OBJECT, 0x18);
        let set = method::<SetEnabled>(&OBJECT, 0x20);

        assert_ne!(get(this, std::ptr::null_mut()), S_OK);

        assert_eq!(set(this, true), S_OK);
        let mut props = XClosedCaptionProperties {
            background_color: XColor {
                a: 1,
                r: 2,
                g: 3,
                b: 4,
            },
            font_color: XColor {
                a: 5,
                r: 6,
                g: 7,
                b: 8,
            },
            window_color: XColor {
                a: 9,
                r: 10,
                g: 11,
                b: 12,
            },
            font_edge_attribute: 99,
            font_style: 99,
            font_scale: -1.0,
            enabled: false,
        };
        assert_eq!(get(this, &mut props), S_OK);
        assert!(props.enabled);
        assert!(props.font_scale > 0.0);

        assert_eq!(set(this, false), S_OK);
        assert_eq!(get(this, &mut props), S_OK);
        assert!(!props.enabled);
    }

    #[test]
    fn high_contrast_mode_classification_and_slot() {
        assert_eq!(
            classify_high_contrast(false, 0x0000_0000),
            HIGH_CONTRAST_OFF
        );
        assert_eq!(
            classify_high_contrast(true, 0x0000_0000),
            HIGH_CONTRAST_DARK
        );
        assert_eq!(
            classify_high_contrast(true, 0x00ff_ffff),
            HIGH_CONTRAST_LIGHT
        );

        let this: This = &OBJECT;
        type GetMode = extern "system" fn(This, *mut u32) -> HRESULT;
        let get = method::<GetMode>(&OBJECT, 0x28);
        assert_ne!(get(this, std::ptr::null_mut()), S_OK);
        let mut mode = u32::MAX;
        assert_eq!(get(this, &mut mode), S_OK);
        assert!(mode <= HIGH_CONTRAST_LIGHT);
    }
}
