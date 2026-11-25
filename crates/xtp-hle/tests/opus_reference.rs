//! Reference comparison of `opus-pure` against Windows' built-in Opus MFT
//! (loaded dynamically from `mfplat.dll`, test-only).
//!
//! - `opus_pure_matches_windows_mft_on_synthetic_streams`: generates mono and
//!   stereo 48 kHz Opus streams with `opus_pure::OpusEncoder`, decodes them
//!   with both `opus_pure::OpusDecoder` and Windows' Opus MFT, and verifies
//!   SNR >= 60 dB.
//! - `opus_pure_matches_windows_mft_on_sample_captures`: opt-in test reading
//!   `.opuscap` captures produced by `XTP_APU_DUMP` from `XTP_SAMPLE_OPUS=<dir>`.
//!   Skips cleanly via `xtp_testkit::skip` when unset.
#![cfg(windows)]

use std::ffi::c_void;
use std::path::PathBuf;

use opus_pure::{Application, OpusDecoder, OpusEncoder};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA,
};
use xtp_hle::modules::apu;

const MIN_SNR_DB: f64 = 60.0;
const SAMPLE_RATE: u32 = 48_000;
const MAX_FRAMES: usize = 5760;

#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    d1: u32,
    d2: u16,
    d3: u16,
    d4: [u8; 8],
}

impl Guid {
    const fn new(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> Self {
        Self { d1, d2, d3, d4 }
    }
}

#[repr(C)]
struct MftRegisterTypeInfo {
    major_type: Guid,
    subtype: Guid,
}

#[repr(C)]
struct MftOutputDataBuffer {
    stream_id: u32,
    sample: *mut c_void,
    status: u32,
    events: *mut c_void,
}

const MFT_CATEGORY_AUDIO_DECODER: Guid = Guid::new(
    0x9ea7_3fb4,
    0xef7a,
    0x4559,
    [0x8d, 0x5d, 0x71, 0x9d, 0x8f, 0x04, 0x26, 0xc7],
);
const MF_MEDIA_TYPE_AUDIO: Guid = Guid::new(
    0x7364_7561,
    0x0000,
    0x0010,
    [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71],
);
const MF_AUDIO_FORMAT_OPUS: Guid = Guid::new(
    0x0000_704f,
    0x0000,
    0x0010,
    [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71],
);
const MF_AUDIO_FORMAT_FLOAT: Guid = Guid::new(
    0x0000_0003,
    0x0000,
    0x0010,
    [0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71],
);
const MF_MT_MAJOR_TYPE: Guid = Guid::new(
    0x48eb_a18e,
    0xf8c9,
    0x4687,
    [0xbf, 0x11, 0x0a, 0x74, 0xc9, 0xf9, 0x6a, 0x8f],
);
const MF_MT_SUBTYPE: Guid = Guid::new(
    0xf7e3_4c9a,
    0x42e8,
    0x4714,
    [0xb7, 0x4b, 0xcb, 0x29, 0xd7, 0x2c, 0x35, 0xe5],
);
const MF_MT_AUDIO_NUM_CHANNELS: Guid = Guid::new(
    0x37e4_8bf5,
    0x645e,
    0x4c5b,
    [0x89, 0xde, 0xad, 0xa9, 0xe2, 0x9b, 0x69, 0x6a],
);
const MF_MT_AUDIO_SAMPLES_PER_SECOND: Guid = Guid::new(
    0x5fae_eae7,
    0x0290,
    0x4c31,
    [0x9e, 0x8a, 0xc5, 0x34, 0xf6, 0x8d, 0x9d, 0xba],
);
const MF_MT_AUDIO_BITS_PER_SAMPLE: Guid = Guid::new(
    0xf2de_b57f,
    0x40fa,
    0x4764,
    [0xaa, 0x33, 0xed, 0x4f, 0x2d, 0x1f, 0xf6, 0x69],
);
const MF_MT_AUDIO_BLOCK_ALIGNMENT: Guid = Guid::new(
    0x322d_e230,
    0x9eeb,
    0x43bd,
    [0xab, 0x7a, 0xff, 0x41, 0x22, 0x51, 0x54, 0x1d],
);
const MF_MT_AUDIO_AVG_BYTES_PER_SECOND: Guid = Guid::new(
    0x1aab_75c8,
    0xcfef,
    0x451c,
    [0xab, 0x95, 0xac, 0x03, 0x4b, 0x8e, 0x17, 0x31],
);
const MF_MT_USER_DATA: Guid = Guid::new(
    0xb6bc_765f,
    0x4c3b,
    0x40a4,
    [0xbd, 0x51, 0x25, 0x35, 0xb6, 0x6f, 0xe0, 0x9d],
);
const IID_IMF_TRANSFORM: Guid = Guid::new(
    0xbf94_c121,
    0x5b05,
    0x4e6f,
    [0x80, 0x00, 0xba, 0x59, 0x89, 0x61, 0x41, 0x4d],
);

const MF_VERSION: u32 = (0x0002 << 16) | 0x0070;
const MFT_ENUM_FLAG_SYNCMFT: u32 = 0x0000_0001;
const MFT_ENUM_FLAG_LOCALMFT: u32 = 0x0000_0010;
const MFT_ENUM_FLAG_SORTANDFILTER: u32 = 0x0000_0040;
const MFT_MESSAGE_COMMAND_DRAIN: u32 = 0x0000_0001;
const MFT_MESSAGE_NOTIFY_BEGIN_STREAMING: u32 = 0x1000_0000;
const MFT_MESSAGE_NOTIFY_START_OF_STREAM: u32 = 0x1000_0003;
const MF_E_TRANSFORM_STREAM_CHANGE: i32 = 0xC00D_6D61_u32 as i32;
const MF_E_TRANSFORM_NEED_MORE_INPUT: i32 = 0xC00D_6D72_u32 as i32;

type FnMfStartup = unsafe extern "system" fn(u32, u32) -> i32;
type FnMfShutdown = unsafe extern "system" fn() -> i32;
type FnMftEnumEx = unsafe extern "system" fn(
    Guid,
    u32,
    *const MftRegisterTypeInfo,
    *const MftRegisterTypeInfo,
    *mut *mut *mut c_void,
    *mut u32,
) -> i32;
type FnMfCreateMediaType = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type FnMfCreateSample = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type FnMfCreateMemoryBuffer = unsafe extern "system" fn(u32, *mut *mut c_void) -> i32;
type FnCoTaskMemFree = unsafe extern "system" fn(*mut c_void);
type FnCoInitializeEx = unsafe extern "system" fn(*mut c_void, u32) -> i32;

struct DllGuard(HMODULE);

impl DllGuard {
    fn load_system(name: &std::ffi::CStr) -> Option<Self> {
        // SAFETY: `name` is a NUL-terminated system DLL filename; restricting
        // search to System32 avoids loading untrusted DLLs from CWD.
        let handle = unsafe {
            LoadLibraryExA(
                name.as_ptr().cast(),
                std::ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        };
        if handle.is_null() {
            None
        } else {
            Some(Self(handle))
        }
    }
}

impl Drop for DllGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid non-null module handle from `LoadLibraryExA`.
        unsafe { FreeLibrary(self.0) };
    }
}

struct MfApi {
    _ole32: DllGuard,
    _mfplat: DllGuard,
    mf_shutdown: FnMfShutdown,
    mft_enum_ex: FnMftEnumEx,
    create_media_type: FnMfCreateMediaType,
    create_sample: FnMfCreateSample,
    create_memory_buffer: FnMfCreateMemoryBuffer,
    co_task_mem_free: FnCoTaskMemFree,
}

impl MfApi {
    fn load() -> Option<Self> {
        unsafe fn sym<T: Copy>(module: HMODULE, name: &[u8]) -> Option<T> {
            // SAFETY: `module` is a valid loaded DLL and `name` is NUL-terminated.
            let p = unsafe { GetProcAddress(module, name.as_ptr().cast::<u8>()) }?;
            // SAFETY: caller supplies the exact Win32 function pointer type `T`.
            Some(unsafe { std::mem::transmute_copy(&p) })
        }

        let mfplat = DllGuard::load_system(c"mfplat.dll")?;
        let ole32 = DllGuard::load_system(c"ole32.dll")?;
        // SAFETY: loading standard Media Foundation / COM entry points; `DllGuard`
        // frees both module handles on any early return.
        let loaded = unsafe {
            let startup: FnMfStartup = sym(mfplat.0, b"MFStartup\0")?;
            let mf_shutdown: FnMfShutdown = sym(mfplat.0, b"MFShutdown\0")?;
            let mft_enum_ex: FnMftEnumEx = sym(mfplat.0, b"MFTEnumEx\0")?;
            let create_media_type: FnMfCreateMediaType = sym(mfplat.0, b"MFCreateMediaType\0")?;
            let create_sample: FnMfCreateSample = sym(mfplat.0, b"MFCreateSample\0")?;
            let create_memory_buffer: FnMfCreateMemoryBuffer =
                sym(mfplat.0, b"MFCreateMemoryBuffer\0")?;
            let co_task_mem_free: FnCoTaskMemFree = sym(ole32.0, b"CoTaskMemFree\0")?;
            let co_init: FnCoInitializeEx = sym(ole32.0, b"CoInitializeEx\0")?;
            let _ = co_init(std::ptr::null_mut(), 0); // COINIT_MULTITHREADED
            if startup(MF_VERSION, 0) < 0 {
                return None;
            }
            Self {
                _ole32: ole32,
                _mfplat: mfplat,
                mf_shutdown,
                mft_enum_ex,
                create_media_type,
                create_sample,
                create_memory_buffer,
                co_task_mem_free,
            }
        };
        Some(loaded)
    }
}

impl Drop for MfApi {
    fn drop(&mut self) {
        // SAFETY: paired with `MFStartup` in `load()`; runs before `_ole32` and
        // `_mfplat` fields are dropped.
        unsafe {
            (self.mf_shutdown)();
        }
    }
}

/// Call COM vtable slot `index` on `obj`.
unsafe fn vtable_slot<T: Copy>(obj: *mut c_void, index: usize) -> T {
    // SAFETY: `obj` is a valid COM interface pointer whose first word is its vtable.
    unsafe {
        let vt = *(obj as *const *const *const ());
        std::mem::transmute_copy(&*vt.add(index))
    }
}

unsafe fn com_release(obj: *mut c_void) {
    if !obj.is_null() {
        // SAFETY: `IUnknown::Release` is always slot 2.
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = unsafe { vtable_slot(obj, 2) };
        // SAFETY: releasing a valid COM reference.
        unsafe { release(obj) };
    }
}

/// Wraps a Windows Media Foundation Opus decoder `IMFTransform`.
struct WindowsOpusMft {
    mf: MfApi,
    transform: *mut c_void,
    channels: usize,
}

impl WindowsOpusMft {
    fn new(channels: usize) -> Option<Self> {
        let mf = MfApi::load()?;
        let input_filter = MftRegisterTypeInfo {
            major_type: MF_MEDIA_TYPE_AUDIO,
            subtype: MF_AUDIO_FORMAT_OPUS,
        };
        let mut activates: *mut *mut c_void = std::ptr::null_mut();
        let mut count = 0u32;
        // SAFETY: `MFTEnumEx` allocates `activates` of length `count`.
        let hr = unsafe {
            (mf.mft_enum_ex)(
                MFT_CATEGORY_AUDIO_DECODER,
                MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_LOCALMFT | MFT_ENUM_FLAG_SORTANDFILTER,
                &input_filter,
                std::ptr::null(),
                &mut activates,
                &mut count,
            )
        };
        if hr < 0 || count == 0 || activates.is_null() {
            return None;
        }

        let mut transform: *mut c_void = std::ptr::null_mut();
        // SAFETY: `activates[0]` is an `IMFActivate`; slot 33 is `ActivateObject`.
        unsafe {
            let first = *activates;
            type ActivateObject =
                unsafe extern "system" fn(*mut c_void, *const Guid, *mut *mut c_void) -> i32;
            let activate: ActivateObject = vtable_slot(first, 33);
            let _ = activate(first, &IID_IMF_TRANSFORM, &mut transform);
            for i in 0..count as usize {
                com_release(*activates.add(i));
            }
            (mf.co_task_mem_free)(activates.cast());
        }
        if transform.is_null() {
            return None;
        }

        let decoder = Self {
            mf,
            transform,
            channels,
        };
        if !decoder.configure() {
            return None;
        }
        Some(decoder)
    }

    fn set_attr_guid(&self, mt: *mut c_void, key: &Guid, val: &Guid) -> bool {
        // SAFETY: `IMFAttributes::SetGUID` is slot 24.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const Guid, *const Guid) -> i32 =
                vtable_slot(mt, 24);
            f(mt, key, val) >= 0
        }
    }

    fn set_attr_u32(&self, mt: *mut c_void, key: &Guid, val: u32) -> bool {
        // SAFETY: `IMFAttributes::SetUINT32` is slot 21.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const Guid, u32) -> i32 =
                vtable_slot(mt, 21);
            f(mt, key, val) >= 0
        }
    }

    fn set_attr_blob(&self, mt: *mut c_void, key: &Guid, data: &[u8]) -> bool {
        // SAFETY: `IMFAttributes::SetBlob` is slot 26.
        unsafe {
            let f: unsafe extern "system" fn(*mut c_void, *const Guid, *const u8, u32) -> i32 =
                vtable_slot(mt, 26);
            f(mt, key, data.as_ptr(), data.len() as u32) >= 0
        }
    }

    fn configure(&self) -> bool {
        // Input type: 48 kHz Opus with `self.channels` and an RFC 7845 OpusHead.
        let mut in_mt = std::ptr::null_mut();
        // SAFETY: Media Foundation factory call.
        if unsafe { (self.mf.create_media_type)(&mut in_mt) } < 0 {
            return false;
        }
        let mut opus_head = [0u8; 19];
        opus_head[..8].copy_from_slice(b"OpusHead");
        opus_head[8] = 1; // version
        opus_head[9] = self.channels as u8;
        opus_head[12..16].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
        let ok_in = self.set_attr_guid(in_mt, &MF_MT_MAJOR_TYPE, &MF_MEDIA_TYPE_AUDIO)
            && self.set_attr_guid(in_mt, &MF_MT_SUBTYPE, &MF_AUDIO_FORMAT_OPUS)
            && self.set_attr_u32(in_mt, &MF_MT_AUDIO_NUM_CHANNELS, self.channels as u32)
            && self.set_attr_u32(in_mt, &MF_MT_AUDIO_SAMPLES_PER_SECOND, SAMPLE_RATE)
            && self.set_attr_blob(in_mt, &MF_MT_USER_DATA, &opus_head);
        let hr_in = if ok_in {
            // SAFETY: `IMFTransform::SetInputType` is slot 15.
            unsafe {
                let set_in: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32 =
                    vtable_slot(self.transform, 15);
                set_in(self.transform, 0, in_mt, 0)
            }
        } else {
            -1
        };
        // SAFETY: releasing temporary media type.
        unsafe { com_release(in_mt) };
        if hr_in < 0 {
            return false;
        }
        if !self.select_output_type() {
            return false;
        }

        // SAFETY: `IMFTransform::ProcessMessage` is slot 23.
        unsafe {
            let msg: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                vtable_slot(self.transform, 23);
            let _ = msg(self.transform, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
            let _ = msg(self.transform, MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        }
        true
    }

    fn select_output_type(&self) -> bool {
        // Output type: query available types for IEEE float PCM, else construct one.
        let mut selected_out = std::ptr::null_mut();
        for idx in 0..16u32 {
            let mut avail = std::ptr::null_mut();
            // SAFETY: `IMFTransform::GetOutputAvailableType` is slot 14.
            let hr = unsafe {
                let get_out: unsafe extern "system" fn(
                    *mut c_void,
                    u32,
                    u32,
                    *mut *mut c_void,
                ) -> i32 = vtable_slot(self.transform, 14);
                get_out(self.transform, 0, idx, &mut avail)
            };
            if hr < 0 || avail.is_null() {
                break;
            }
            let mut subtype = Guid::new(0, 0, 0, [0; 8]);
            // SAFETY: `IMFAttributes::GetGUID` is slot 10.
            let got = unsafe {
                let get_guid: unsafe extern "system" fn(
                    *mut c_void,
                    *const Guid,
                    *mut Guid,
                ) -> i32 = vtable_slot(avail, 10);
                get_guid(avail, &MF_MT_SUBTYPE, &mut subtype) >= 0
            };
            if got
                && subtype.d1 == MF_AUDIO_FORMAT_FLOAT.d1
                && subtype.d2 == MF_AUDIO_FORMAT_FLOAT.d2
                && subtype.d3 == MF_AUDIO_FORMAT_FLOAT.d3
                && subtype.d4 == MF_AUDIO_FORMAT_FLOAT.d4
            {
                selected_out = avail;
                break;
            }
            // SAFETY: releasing non-matching available output type.
            unsafe { com_release(avail) };
        }
        if selected_out.is_null() {
            // SAFETY: Media Foundation factory call.
            if unsafe { (self.mf.create_media_type)(&mut selected_out) } < 0 {
                return false;
            }
            let block_align = (self.channels * 4) as u32;
            let _ = self.set_attr_guid(selected_out, &MF_MT_MAJOR_TYPE, &MF_MEDIA_TYPE_AUDIO)
                && self.set_attr_guid(selected_out, &MF_MT_SUBTYPE, &MF_AUDIO_FORMAT_FLOAT)
                && self.set_attr_u32(
                    selected_out,
                    &MF_MT_AUDIO_NUM_CHANNELS,
                    self.channels as u32,
                )
                && self.set_attr_u32(selected_out, &MF_MT_AUDIO_SAMPLES_PER_SECOND, SAMPLE_RATE)
                && self.set_attr_u32(selected_out, &MF_MT_AUDIO_BITS_PER_SAMPLE, 32)
                && self.set_attr_u32(selected_out, &MF_MT_AUDIO_BLOCK_ALIGNMENT, block_align)
                && self.set_attr_u32(
                    selected_out,
                    &MF_MT_AUDIO_AVG_BYTES_PER_SECOND,
                    SAMPLE_RATE * block_align,
                );
        }
        // SAFETY: `IMFTransform::SetOutputType` is slot 16.
        let hr_out = unsafe {
            let set_out: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32 =
                vtable_slot(self.transform, 16);
            set_out(self.transform, 0, selected_out, 0)
        };
        // SAFETY: releasing temporary media type.
        unsafe { com_release(selected_out) };
        hr_out >= 0
    }

    fn decode_packet(&self, packet: &[u8]) -> Result<Vec<f32>, i32> {
        let mut in_buf = std::ptr::null_mut();
        let mut in_sample = std::ptr::null_mut();
        // SAFETY: creating MF memory buffer & sample for the input packet.
        unsafe {
            let hr = (self.mf.create_memory_buffer)(packet.len() as u32, &mut in_buf);
            if hr < 0 {
                return Err(hr);
            }
            let lock: unsafe extern "system" fn(
                *mut c_void,
                *mut *mut u8,
                *mut u32,
                *mut u32,
            ) -> i32 = vtable_slot(in_buf, 3);
            let unlock: unsafe extern "system" fn(*mut c_void) -> i32 = vtable_slot(in_buf, 4);
            let set_len: unsafe extern "system" fn(*mut c_void, u32) -> i32 =
                vtable_slot(in_buf, 6);
            let mut ptr = std::ptr::null_mut();
            let hr = lock(in_buf, &mut ptr, std::ptr::null_mut(), std::ptr::null_mut());
            if hr < 0 {
                com_release(in_buf);
                return Err(hr);
            }
            std::ptr::copy_nonoverlapping(packet.as_ptr(), ptr, packet.len());
            let _ = unlock(in_buf);
            let _ = set_len(in_buf, packet.len() as u32);

            let hr = (self.mf.create_sample)(&mut in_sample);
            if hr < 0 {
                com_release(in_buf);
                return Err(hr);
            }
            // `IMFSample::AddBuffer` is slot 42.
            let add_buf: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32 =
                vtable_slot(in_sample, 42);
            let _ = add_buf(in_sample, in_buf);
            com_release(in_buf);

            // `IMFTransform::ProcessInput` is slot 24.
            let process_in: unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32 =
                vtable_slot(self.transform, 24);
            let hr = process_in(self.transform, 0, in_sample, 0);
            com_release(in_sample);
            if hr < 0 {
                return Err(hr);
            }
        }

        let mut samples = Vec::new();
        self.drain_output(&mut samples)?;
        Ok(samples)
    }

    fn drain(&self) -> Result<Vec<f32>, i32> {
        // SAFETY: `IMFTransform::ProcessMessage` is slot 23.
        unsafe {
            let msg: unsafe extern "system" fn(*mut c_void, u32, usize) -> i32 =
                vtable_slot(self.transform, 23);
            let _ = msg(self.transform, MFT_MESSAGE_COMMAND_DRAIN, 0);
        }
        let mut samples = Vec::new();
        self.drain_output(&mut samples)?;
        Ok(samples)
    }

    fn drain_output(&self, samples: &mut Vec<f32>) -> Result<(), i32> {
        // Query whether the MFT allocates its own output samples.
        #[repr(C)]
        struct MftOutputStreamInfo {
            flags: u32,
            cb_size: u32,
            cb_alignment: u32,
        }
        let mut osi = MftOutputStreamInfo {
            flags: 0,
            cb_size: 0,
            cb_alignment: 0,
        };
        // SAFETY: `IMFTransform::GetOutputStreamInfo` is slot 7.
        let mft_provides_samples = unsafe {
            let get_osi: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *mut MftOutputStreamInfo,
            ) -> i32 = vtable_slot(self.transform, 7);
            get_osi(self.transform, 0, &mut osi) >= 0 && (osi.flags & 0x300) != 0
        };

        loop {
            let mut caller_buf = std::ptr::null_mut();
            let mut caller_sample = std::ptr::null_mut();
            if !mft_provides_samples {
                let out_bytes = osi.cb_size.max((MAX_FRAMES * self.channels * 4) as u32);
                // SAFETY: allocating output buffer/sample when the MFT expects caller-owned storage.
                unsafe {
                    let hr = (self.mf.create_memory_buffer)(out_bytes, &mut caller_buf);
                    if hr < 0 {
                        return Err(hr);
                    }
                    let hr = (self.mf.create_sample)(&mut caller_sample);
                    if hr < 0 {
                        com_release(caller_buf);
                        return Err(hr);
                    }
                    let add_buf: unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32 =
                        vtable_slot(caller_sample, 42);
                    let _ = add_buf(caller_sample, caller_buf);
                }
            }

            let mut odb = MftOutputDataBuffer {
                stream_id: 0,
                sample: caller_sample,
                status: 0,
                events: std::ptr::null_mut(),
            };
            let mut status = 0u32;
            // SAFETY: `IMFTransform::ProcessOutput` is slot 25.
            let hr = unsafe {
                let process_out: unsafe extern "system" fn(
                    *mut c_void,
                    u32,
                    u32,
                    *mut MftOutputDataBuffer,
                    *mut u32,
                ) -> i32 = vtable_slot(self.transform, 25);
                process_out(self.transform, 0, 1, &mut odb, &mut status)
            };
            // SAFETY: releasing any events / samples returned or allocated above.
            unsafe {
                if !odb.events.is_null() {
                    com_release(odb.events);
                }
                if hr == MF_E_TRANSFORM_STREAM_CHANGE {
                    com_release(caller_sample);
                    com_release(caller_buf);
                    if !self.select_output_type() {
                        return Err(hr);
                    }
                    continue;
                }
                if hr == MF_E_TRANSFORM_NEED_MORE_INPUT {
                    com_release(caller_sample);
                    com_release(caller_buf);
                    break;
                }
                if hr < 0 {
                    com_release(caller_sample);
                    com_release(caller_buf);
                    return Err(hr);
                }
                let out_sample = odb.sample;
                if !out_sample.is_null() {
                    let mut cont_buf = std::ptr::null_mut();
                    // `IMFSample::ConvertToContiguousBuffer` is slot 41.
                    let to_buf: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32 =
                        vtable_slot(out_sample, 41);
                    if to_buf(out_sample, &mut cont_buf) >= 0 && !cont_buf.is_null() {
                        let lock: unsafe extern "system" fn(
                            *mut c_void,
                            *mut *mut u8,
                            *mut u32,
                            *mut u32,
                        ) -> i32 = vtable_slot(cont_buf, 3);
                        let unlock: unsafe extern "system" fn(*mut c_void) -> i32 =
                            vtable_slot(cont_buf, 4);
                        let mut ptr = std::ptr::null_mut();
                        let mut cur_len = 0u32;
                        if lock(cont_buf, &mut ptr, std::ptr::null_mut(), &mut cur_len) >= 0 {
                            if cur_len >= 4 {
                                let bytes = std::slice::from_raw_parts(ptr, cur_len as usize);
                                for &chunk in bytes.as_chunks::<4>().0 {
                                    samples.push(f32::from_le_bytes(chunk));
                                }
                            }
                            let _ = unlock(cont_buf);
                        }
                        com_release(cont_buf);
                    }
                    com_release(out_sample);
                }
                com_release(caller_buf);
            }
        }
        Ok(())
    }
}

impl Drop for WindowsOpusMft {
    fn drop(&mut self) {
        // SAFETY: releasing the activated `IMFTransform`.
        unsafe { com_release(self.transform) };
    }
}

/// Signal-to-noise ratio in dB between `reference` and `candidate`.
fn snr_db(reference: &[f32], candidate: &[f32]) -> f64 {
    let n = reference.len().min(candidate.len());
    assert!(n > 0, "empty PCM comparison");
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&r, &c) in reference[..n].iter().zip(&candidate[..n]) {
        let rf = f64::from(r);
        let df = rf - f64::from(c);
        sig += rf * rf;
        err += df * df;
    }
    if err == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / err).log10()
}

fn compare_stream(channels: usize, packets: &[Vec<u8>], label: &str) {
    let Some(mft) = WindowsOpusMft::new(channels) else {
        xtp_testkit::skip("no-opus-mft");
        return;
    };
    let mut pure = OpusDecoder::new(SAMPLE_RATE as i32, channels).expect("opus-pure decoder");
    let mut pure_pcm = Vec::new();
    let mut mft_pcm = Vec::new();
    let mut scratch = vec![0f32; MAX_FRAMES * channels];

    for pkt in packets {
        let frames = pure
            .decode(pkt, MAX_FRAMES, &mut scratch)
            .expect("opus-pure decode");
        pure_pcm.extend_from_slice(&scratch[..frames * channels]);
        let decoded_mft = mft.decode_packet(pkt).expect("Windows Opus MFT decode");
        mft_pcm.extend_from_slice(&decoded_mft);
    }
    let drained = mft.drain().expect("Windows Opus MFT drain");
    mft_pcm.extend_from_slice(&drained);

    assert_eq!(
        pure_pcm.len(),
        mft_pcm.len(),
        "{label}: sample count mismatch (opus-pure={} vs MFT={})",
        pure_pcm.len(),
        mft_pcm.len()
    );
    let snr = snr_db(&mft_pcm, &pure_pcm);
    assert!(
        snr >= MIN_SNR_DB,
        "{label}: SNR {snr:.2} dB < {MIN_SNR_DB} dB"
    );
}

fn synth_packets(channels: usize, count: usize) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(SAMPLE_RATE as i32, channels, Application::Audio).unwrap();
    let frame_samples = 960; // 20 ms at 48 kHz
    let mut packets = Vec::with_capacity(count);
    let mut buf = [0u8; 1500];
    for p in 0..count {
        let mut pcm = vec![0f32; frame_samples * channels];
        for i in 0..frame_samples {
            let t = (p * frame_samples + i) as f32 / SAMPLE_RATE as f32;
            let left = (t * 440.0 * std::f32::consts::TAU).sin() * 0.4;
            if channels == 1 {
                pcm[i] = left;
            } else {
                let right = (t * 660.0 * std::f32::consts::TAU).sin() * 0.4;
                pcm[i * 2] = left;
                pcm[i * 2 + 1] = right;
            }
        }
        let n = enc.encode(&pcm, frame_samples, &mut buf).unwrap();
        packets.push(buf[..n].to_vec());
    }
    packets
}

#[test]
fn opus_pure_matches_windows_mft_on_synthetic_streams() {
    // Also exercise the `.opuscap` capture round-trip on synthetic packets.
    let dir = tempfile::tempdir().unwrap();
    for channels in [1usize, 2] {
        let packets = synth_packets(channels, 10);
        let path = dir.path().join(format!("synth-{channels}ch.opuscap"));
        // Seed a stale capture with a different channel count to verify
        // `create_capture_file` truncates instead of concatenating runs.
        apu::create_capture_file(&path, if channels == 1 { 2 } else { 1 }).unwrap();
        apu::append_capture_packet(&path, if channels == 1 { 2 } else { 1 }, &[0xAA, 0xBB])
            .unwrap();
        apu::create_capture_file(&path, channels as u16).unwrap();
        for pkt in &packets {
            apu::append_capture_packet(&path, channels as u16, pkt).unwrap();
        }
        let (read_ch, read_pkts) = apu::read_capture_file(&path).unwrap();
        assert_eq!(read_ch as usize, channels);
        assert_eq!(read_pkts, packets);
        compare_stream(channels, &read_pkts, &format!("synthetic {channels}ch"));
    }
}

/// Opt-in parity test on real captures produced by `XTP_APU_DUMP=<dir>`.
/// Set `XTP_SAMPLE_OPUS=<dir>` to a folder of `.opuscap` files outside the
/// repository.
#[test]
fn opus_pure_matches_windows_mft_on_sample_captures() {
    let Some(dir) = std::env::var_os("XTP_SAMPLE_OPUS") else {
        xtp_testkit::skip("no-sample XTP_SAMPLE_OPUS");
        return;
    };
    let dir = PathBuf::from(dir);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("XTP_SAMPLE_OPUS={} unreadable: {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "opuscap"))
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "XTP_SAMPLE_OPUS={} contains no .opuscap files",
        dir.display()
    );
    for file in files {
        let (channels, packets) = apu::read_capture_file(&file)
            .unwrap_or_else(|e| panic!("reading {}: {e}", file.display()));
        if packets.is_empty() {
            continue;
        }
        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("capture");
        compare_stream(channels as usize, &packets, name);
    }
}
