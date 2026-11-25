//! Indirect rectangle-list draws (the console's topology 17,
//! `ExecuteIndirect` kind 0), expanded on the GPU. A direct rectangle draw
//! expands its vertices on the CPU ([`crate::primitive::expand_rects`]);
//! an indirect one knows its vertex counts only on the GPU. So after
//! [`crate::indirect::clamp_uav`] bounds each record, two compute passes
//! of ours run before `ExecuteIndirect`:
//!
//! 1. `rect_rewrite` reads each live record (records at or past the
//!    title's count, records with no instance and records whose source
//!    vertices `start..start + 3 * rects` reach past a bound per-vertex
//!    vertex buffer draw nothing), allocates `rects * 6` vertices from the
//!    list's scratch with an atomic counter, and writes a rewritten record
//!    into the list's own arguments buffer. Its thread 0 writes the record
//!    count there too: the title's, at most `max`. The title's arguments
//!    are left as the clamp made them.
//! 2. `rect_expand` runs once per per-vertex input slot and writes
//!    `(v0, v1, v2, v0, v2, v3)` into that slot's scratch region, with
//!    `v3 = v0 + v2 - v1` for the formats and in-stride components that
//!    [`crate::primitive::expand_rects`] computes, the same way (formats,
//!    conversions, rounding: a test compares both byte for byte).
//!
//! `ExecuteIndirect` then reads only the list's own buffers: our arguments,
//! with our record count as its count buffer (so records past the title's
//! count do not run, and no title buffer changes state), and the expanded
//! vertices. A list's scratch ([`Scratch`]) is reused by each of its
//! indirect rectangle draws, and by later lists once its recording has
//! gone.
//!
//! Both passes are HLSL compiled at run time by the system's `D3DCompile`.
//! They read buffers as 32-bit words and take their root signatures from
//! code, which Wine's compiler (vkd3d-shader) accepts too.
//!
//! Ported from Marvisak/VIctory#346 (head 8ef6bfe) by BastienRodz, with
//! fuzeofficlal's half rounding (`primitive.rs`). Changes in the port:
//! - the list's memory is draw state's ([`crate::draw`]'s list memory),
//!   not the host list's: one scratch per recording, pooled, its buffers
//!   made as draws need them (the vertex scratch for the first draw with a
//!   per-vertex slot);
//! - the allocation counter is the arguments buffer's first word, zeroed
//!   in the list before each draw's rewrite pass (`WriteBufferImmediate`),
//!   not by the CPU at submission: two rectangle draws of one list each
//!   start from zero, whichever order the GPU sees the CPU's writes in;
//! - the record count is the arguments buffer's second word, so
//!   `ExecuteIndirect` runs only the live records;
//! - a source vertex buffer is read up to 4 bytes short of its host copy's
//!   end (a margin: a root view has no bounds);
//! - the fourth corner is computed `precise` (no reassociation), as the
//!   CPU does it.
//!
//! The vertex scratch grows from 8 MiB to fit the bounded draw. For indexed
//! records the expansion gathers three R16 or R32 indices and applies the
//! signed base vertex before computing the fourth corner.
//!
//! shortcut: as in [`crate::primitive`], the fourth corner of a pipeline with
//! per-vertex inputs is computed from the vertex inputs, not the vertex
//! shader's outputs, and v1 is taken for the corner whatever the order:
//! `v3 = v0 + v2 - v1` is right for a corner at v1 (top right, top left,
//! bottom left), and makes a parallelogram for a corner at v0 or v2 (for
//! the tiled lighting's top left, bottom left, top right, (0, 0), (0, 1),
//! (1, 0) with y down, it gives (1, -1), above the top right, where the far
//! corner is (1, 1); those rectangles have no vertex input, see below).
//! Formats unknown to [`crate::primitive::components`] and
//! components past the vertex stride keep the first vertex's bytes.
//!
//! A pipeline without per-vertex inputs (one that makes its corners from
//! the vertex id) draws the record as the title wrote it, with `3 * rects`
//! vertices, as a direct rectangle draw of such a pipeline does: each
//! rectangle a triangle, the half on its corner's side. Its geometry stage
//! ([`geometry_stage`], made with the pipeline) adds the other half, with
//! the fourth corner from the vertex shader's outputs whichever vertex the
//! corner is; a pipeline made without it (refused by the driver with it,
//! or a pixel shader that reads the primitive id) keeps one triangle. Its
//! shader sees vertex ids that count from 0 in every record, as on the
//! console: D3D12's `SV_VertexID` leaves out the start vertex (a test draws
//! a record from vertex 6 with ids 0 to 2), and so does the console's v0,
//! the start reaching a program only through a user SGPR that none of the
//! title's vertex programs read (docs/analysis/drops.md §5.1.1). With inputs,
//! the vertex id is the expanded one, from an allocated start whose order
//! between records may change from run to run.
//! Vertex strides not a multiple of 4 are padded to 4 bytes in the scratch
//! (and bound with that stride), so that no two threads share a word.
//!
//! The rewrite pass bounds counts and instances itself, as the clamp does,
//! so it stays safe whether or not the clamp wrote its results in place.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition, uav_barrier};
use crate::pipeline::host::Input;
use crate::primitive;
use crate::shadow::Shadow;

mod geometry;

pub(crate) use geometry::geometry_stage;

/// A list's vertex scratch, shared by the records of one draw.
pub(crate) const SCRATCH_BYTES: u64 = 8 << 20;

/// The records one 64-thread-group pass takes at most.
const MAX_RECORDS: u32 = 65535 * 64;

/// Thread groups per dispatch dimension (D3D12's limit).
const MAX_GROUPS_PER_DIM: u32 = 65535;

/// The arguments buffer's first bytes: the allocation counter (a word), the
/// record count `ExecuteIndirect` reads (a word at [`COUNT_OFFSET`]),
/// padding; the rewritten records follow.
const COUNTER_BYTES: u64 = 16;

/// Where the record count is in the arguments buffer.
const COUNT_OFFSET: u64 = 4;

/// The arguments buffer's smallest size (255 records), and the step it
/// grows by.
const ARGS_MIN: u64 = 4 << 10;

/// The arguments buffer's largest size: [`MAX_RECORDS`] records.
const ARGS_MAX: u64 = COUNTER_BYTES + MAX_RECORDS as u64 * 16;

/// Arguments buffers larger than this are not pooled: they serve the list
/// that needed them only.
const ARGS_POOLED: u64 = 1 << 20;

/// Scratches kept for later lists when no list holds them.
const POOLED: usize = 16;

/// One per-vertex input element, as `rect_expand` reads it (8 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Element {
    /// Byte offset in the source (and destination) vertex.
    pub(crate) offset: u32,
    /// Low byte: [`primitive::Kind::code`]; bits 8 on: the components
    /// inside the vertex stride (1 to 4).
    pub(crate) packed: u32,
}

/// One per-vertex slot's region in the vertex scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotLayout {
    pub(crate) slot: u32,
    /// From the scratch's start, 16-byte aligned.
    pub(crate) dst_offset: u64,
    /// The source stride rounded up to 4 bytes.
    pub(crate) dst_stride: u32,
    /// The region's bytes: the vertex capacity times `dst_stride`.
    pub(crate) dst_size: u32,
}

/// The element table of `slot` with vertices of `stride` bytes, as
/// [`primitive::expand_rects`] reads them, and whether the pipeline has
/// any per-vertex input in that slot (even of formats it does not know: the
/// slot is expanded then, with `v3 = v0`).
pub(crate) fn slot_elements(inputs: &[Input], slot: u32, stride: u32) -> (bool, Vec<Element>) {
    let mut per_vertex = false;
    let mut out = Vec::new();
    let stride = stride as usize;
    for e in inputs {
        if e.slot != slot || e.per_instance {
            continue;
        }
        per_vertex = true;
        let Some((n, kind)) = primitive::components(e.format) else {
            continue;
        };
        let inside = stride.saturating_sub(e.offset as usize) / kind.bytes();
        let count = n.min(inside) as u32;
        if count == 0 {
            continue;
        }
        out.push(Element {
            offset: e.offset,
            packed: kind.code() | (count << 8),
        });
    }
    (per_vertex, out)
}

/// Disjoint 16-byte aligned regions in `capacity_bytes` for every
/// per-vertex `(slot, source stride)`, all of one vertex capacity: the
/// capacity (whole rectangles) and the regions. `None` if a stride is zero
/// or not one rectangle fits.
pub(crate) fn scratch_layout(
    slots: &[(u32, u32)],
    capacity_bytes: u64,
) -> Option<(u32, Vec<SlotLayout>)> {
    if slots.is_empty() {
        return Some((0, Vec::new()));
    }
    let mut dst_strides = Vec::with_capacity(slots.len());
    let mut total_stride: u64 = 0;
    for &(_, src_stride) in slots {
        if src_stride == 0 {
            return None;
        }
        let dst_stride = src_stride.checked_next_multiple_of(4)?;
        dst_strides.push(dst_stride);
        total_stride = total_stride.checked_add(u64::from(dst_stride))?;
    }
    // Without the padding between regions at most this many fit; the
    // padding (under 16 bytes a region) costs a few rectangles at most, so
    // step down until the padded layout fits.
    let mut capacity = ((capacity_bytes / total_stride / 6) * 6).min(u64::from(u32::MAX) / 6 * 6);
    let layouts = loop {
        if capacity < 6 {
            return None;
        }
        if let Some(layouts) = place(slots, &dst_strides, capacity, capacity_bytes) {
            break layouts;
        }
        capacity -= 6;
    };
    Some((capacity as u32, layouts))
}

/// The slots' regions of `verts` vertices each, 16-byte aligned one after
/// the other, if they end within `capacity_bytes`.
fn place(
    slots: &[(u32, u32)],
    dst_strides: &[u32],
    verts: u64,
    capacity_bytes: u64,
) -> Option<Vec<SlotLayout>> {
    let mut cursor = 0u64;
    let mut layouts = Vec::with_capacity(slots.len());
    for (&(slot, _), &dst_stride) in slots.iter().zip(dst_strides) {
        let dst_offset = cursor.next_multiple_of(16);
        let dst_size = verts.checked_mul(u64::from(dst_stride))?;
        let end = dst_offset.checked_add(dst_size)?;
        if end > capacity_bytes {
            return None;
        }
        layouts.push(SlotLayout {
            slot,
            dst_offset,
            dst_stride,
            dst_size: u32::try_from(dst_size).ok()?,
        });
        cursor = end;
    }
    Some(layouts)
}

/// The vertex bound all per-vertex slots share: a record's `start..start +
/// 3 * rects` lies in every slot when it lies in `0..` this. `bounds` are
/// (bytes, stride); none: no bound.
pub(crate) fn max_source_vertices(bounds: &[(u64, u32)]) -> u32 {
    bounds
        .iter()
        .map(|&(size, stride)| {
            if stride == 0 {
                0
            } else {
                (size / u64::from(stride)).min(u64::from(u32::MAX)) as u32
            }
        })
        .min()
        .unwrap_or(u32::MAX)
}

/// The bytes of a source vertex buffer the expansion reads at most from
/// `offset` in its host copy of `width` bytes: 4 short of the copy's end.
pub(crate) fn readable_in(width: u64, offset: u64) -> u64 {
    width.saturating_sub(offset).saturating_sub(4)
}

/// A dispatch grid `(x, y)` of at least `records` groups, each dimension
/// within D3D12's limit.
pub(crate) fn dispatch_grid(records: u32) -> (u32, u32) {
    if records == 0 {
        return (0, 0);
    }
    let x = records.min(MAX_GROUPS_PER_DIM);
    (x, records.div_ceil(x))
}

const REWRITE_SOURCE: &str = r#"
cbuffer Params : register(b0) {
    uint records;
    uint src_stride;
    uint has_count;
    uint max_src_verts;
    uint vert_capacity;
    uint need_verts;
    uint indexed;
    uint first_record;
};
// Buffers as 32-bit words, and no root signature attribute: Wine's system
// compiler (vkd3d-shader) has no Load on RWByteAddressBuffer and ignores the
// attribute, so the root signatures are made in code.
RWStructuredBuffer<uint> src_args : register(u0);
RWStructuredBuffer<uint> count_buf : register(u1);
// The arguments buffer's first words: the allocation counter, then the
// record count ExecuteIndirect reads.
RWStructuredBuffer<uint> alloc : register(u2);
RWStructuredBuffer<uint> dst_args : register(u3);

void store4(uint at, uint4 v) {
    dst_args[at] = v.x;
    dst_args[at + 1u] = v.y;
    dst_args[at + 2u] = v.z;
    dst_args[at + 3u] = v.w;
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint r = id.x;
    uint live = records;
    if (has_count != 0u) {
        uint count = count_buf[0];
        live = min(count - min(count, first_record), records);
    }
    if (r == 0u) {
        alloc[1] = live;
    }
    if (r >= records) return;
    uint dst_at = r * 4u;
    if (r >= live) {
        store4(dst_at, uint4(0u, 0u, 0u, 0u));
        return;
    }
    uint at = r * (src_stride / 4u);
    uint4 rec = uint4(src_args[at], src_args[at + 1u], src_args[at + 2u], src_args[at + 3u]);
    // The clamp's bounds again: 2^24 vertices, 2^26 vertex instances.
    rec.x = min(rec.x, 1u << 24);
    rec.y = min(rec.y, (1u << 26) / max(rec.x, 1u));
    if (indexed != 0u) rec.w = src_args[at + 4u];
    uint rects = rec.x / 3u;
    uint src_verts = rects * 3u;
    uint out_verts = rects * 6u;
    // Empty unless it draws.
    uint4 res = uint4(0u, rec.y, 0u, rec.w);
    if (rects == 0u || rec.y == 0u || rec.z > max_src_verts || src_verts > max_src_verts - rec.z) {
        // Nothing, or vertices past a bound vertex buffer.
    } else if (need_verts == 0u) {
        // No per-vertex input, so the corners come from the vertex id: each
        // rectangle's first triangle, from the title's start.
        res.x = src_verts;
        res.z = rec.z;
    } else if (out_verts <= vert_capacity) {
        uint before;
        InterlockedAdd(alloc[0], out_verts, before);
        if (before < vert_capacity && out_verts <= vert_capacity - before) {
            res.x = out_verts;
            res.z = before;
        }
    }
    store4(dst_at, res);
}
"#;

const EXPAND_SOURCE: &str = r#"
cbuffer Params : register(b0) {
    uint records;
    uint src_stride;
    uint grid_x;
    uint vb_src_bias;
    uint vb_src_stride;
    uint vb_dst_stride;
    uint elem_count;
    uint index_bytes;
    uint index_bias;
    uint vertex_limit;
};
// Buffers as 32-bit words (see the rewrite pass); byte positions are
// turned into word indices here.
StructuredBuffer<uint> elems : register(t0);
RWStructuredBuffer<uint> src_args : register(u0);
RWStructuredBuffer<uint> dst_args : register(u1);
RWStructuredBuffer<uint> src_vb : register(u2);
RWStructuredBuffer<uint> dst_vb : register(u3);
RWStructuredBuffer<uint> indices : register(u4);

uint vertex_id(uint at, uint base) {
    if (index_bytes == 0u) return at;
    uint pos = index_bias + at * index_bytes;
    uint word = indices[pos >> 2u];
    uint index = index_bytes == 4u ? word : (word >> ((pos & 2u) * 8u)) & 0xffffu;
    // Les listes de rectangles du titre utilisent la coupure désactivée.
    // Tous les indices participent au triplet, y compris ceux à bits tous à un.
    uint id = index + base;
    if (asint(base) >= 0 && id < index) return 0xffffffffu;
    return id;
}

uint load_u8(uint pos) {
    uint w = src_vb[(pos & ~3u) >> 2u];
    return (w >> ((pos & 3u) * 8u)) & 0xffu;
}

uint load_u16(uint pos) {
    uint rem = pos & 3u;
    uint w0 = src_vb[(pos & ~3u) >> 2u];
    if (rem <= 2u) {
        return (w0 >> (rem * 8u)) & 0xffffu;
    }
    uint w1 = src_vb[((pos & ~3u) + 4u) >> 2u];
    return ((w0 >> 24u) | (w1 << 8u)) & 0xffffu;
}

uint load_u32(uint pos) {
    uint rem = pos & 3u;
    uint w0 = src_vb[(pos & ~3u) >> 2u];
    if (rem == 0u) {
        return w0;
    }
    uint w1 = src_vb[((pos & ~3u) + 4u) >> 2u];
    return (w0 >> (rem * 8u)) | (w1 << ((4u - rem) * 8u));
}

void store_u8(uint pos, uint val) {
    uint aligned = pos & ~3u;
    uint shift = (pos & 3u) * 8u;
    uint w = dst_vb[aligned >> 2u];
    w = (w & ~(0xffu << shift)) | ((val & 0xffu) << shift);
    dst_vb[aligned >> 2u] = w;
}

void store_u16(uint pos, uint val) {
    uint rem = pos & 3u;
    if (rem <= 2u) {
        uint aligned = pos & ~3u;
        uint shift = rem * 8u;
        uint w = dst_vb[aligned >> 2u];
        w = (w & ~(0xffffu << shift)) | ((val & 0xffffu) << shift);
        dst_vb[aligned >> 2u] = w;
    } else {
        store_u8(pos, val & 0xffu);
        store_u8(pos + 1u, (val >> 8u) & 0xffu);
    }
}

void store_u32(uint pos, uint val) {
    uint rem = pos & 3u;
    if (rem == 0u) {
        dst_vb[pos >> 2u] = val;
    } else {
        uint aligned = pos & ~3u;
        uint shift = rem * 8u;
        uint inv = (4u - rem) * 8u;
        uint mask = (1u << shift) - 1u;
        uint w0 = dst_vb[aligned >> 2u];
        w0 = (w0 & mask) | (val << shift);
        dst_vb[aligned >> 2u] = w0;
        uint w1 = dst_vb[(aligned + 4u) >> 2u];
        w1 = (w1 & ~mask) | (val >> inv);
        dst_vb[(aligned + 4u) >> 2u] = w1;
    }
}

void copy_vertex(uint src_off, uint dst_off) {
    uint full = vb_src_stride & ~3u;
    for (uint b = 0u; b < full; b += 4u) {
        dst_vb[(dst_off + b) >> 2u] = load_u32(src_off + b);
    }
    uint rem = vb_src_stride & 3u;
    if (rem != 0u) {
        uint w = 0u;
        for (uint k = 0u; k < rem; k++) {
            w |= load_u8(src_off + full + k) << (k * 8u);
        }
        dst_vb[(dst_off + full) >> 2u] = w;
    }
}

float half_to_f32(uint h) {
    uint sgn = (h >> 15u) << 31u;
    uint ex = (h >> 10u) & 0x1fu;
    uint man = h & 0x3ffu;
    if (ex == 0u) {
        if (man == 0u) return asfloat(sgn);
        float v = (float)man * (1.0f / 16777216.0f);
        return sgn != 0u ? -v : v;
    }
    if (ex == 31u) {
        return asfloat(sgn | 0x7f800000u | (man << 13u));
    }
    return asfloat(sgn | ((ex + 112u) << 23u) | (man << 13u));
}

uint f32_to_half(float v) {
    uint bits = asuint(v);
    uint sgn = (bits >> 16u) & 0x8000u;
    int ex = (int)((bits >> 23u) & 0xffu) - 127 + 15;
    uint man = bits & 0x7fffffu;
    if (ex >= 31) return sgn | 0x7c00u;
    if (ex <= 0) {
        // Half subnormals, rounded to even, as primitive.rs makes them.
        if (ex < -10) return sgn;
        uint sig = (1u << 23) | man;
        uint shift = (uint)(14 - ex);
        uint halfway = 1u << (shift - 1u);
        uint q = sig >> shift;
        uint r = sig & ((1u << shift) - 1u);
        bool up = r > halfway || (r == halfway && (q & 1u) != 0u);
        return sgn | (q + (up ? 1u : 0u));
    }
    // To nearest, halfway cases to even; added, not or-ed: a rounding carry
    // out of the mantissa raises the exponent, as in primitive.rs.
    uint kept = (man >> 13u) & 1u;
    return sgn | (((uint)ex << 10u) + ((man + 0x0fffu + kept) >> 13u));
}

float read_comp(uint kind, uint pos) {
    if (kind == 0u) return asfloat(load_u32(pos));
    if (kind == 1u) return half_to_f32(load_u16(pos));
    if (kind == 2u) return (float)load_u16(pos) / 65535.0f;
    return (float)load_u8(pos) / 255.0f;
}

void write_comp(uint kind, uint pos, float v) {
    if (kind == 0u) {
        store_u32(pos, asuint(v));
    } else if (kind == 1u) {
        store_u16(pos, f32_to_half(v));
    } else if (kind == 2u) {
        store_u16(pos, (uint)(saturate(v) * 65535.0f + 0.5f));
    } else {
        store_u8(pos, (uint)(saturate(v) * 255.0f + 0.5f));
    }
}

[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint r = gid.y * grid_x + gid.x;
    if (r >= records) return;
    uint out_verts = dst_args[r * 4u];
    uint rects = out_verts / 6u;
    if (rects == 0u) return;
    uint dst_start = dst_args[r * 4u + 2u];
    uint src_start = src_args[r * (src_stride / 4u) + 2u];

    for (uint rect = gtid.x; rect < rects; rect += 64u) {
        uint at = src_start + rect * 3u;
        uint base = 0;
        if (index_bytes != 0u) base = src_args[r * (src_stride / 4u) + 3u];
        uint3 ids = uint3(vertex_id(at, base), vertex_id(at + 1u, base), vertex_id(at + 2u, base));
        if (any(ids >= vertex_limit)) {
            // A malformed triplet becomes a degenerate triangle. Never read outside a root view.
            uint begin = (dst_start + rect * 6u) * vb_dst_stride / 4u;
            for (uint i = 0; i < 6u * (vb_dst_stride / 4u); i++) dst_vb[begin + i] = 0;
            continue;
        }
        uint v0_src = vb_src_bias + ids.x * vb_src_stride;
        uint v1_src = vb_src_bias + ids.y * vb_src_stride;
        uint v2_src = vb_src_bias + ids.z * vb_src_stride;
        uint d = (dst_start + rect * 6u) * vb_dst_stride;

        copy_vertex(v0_src, d);
        copy_vertex(v1_src, d + vb_dst_stride);
        copy_vertex(v2_src, d + vb_dst_stride * 2u);
        copy_vertex(v0_src, d + vb_dst_stride * 3u);
        copy_vertex(v2_src, d + vb_dst_stride * 4u);
        uint v3_dst = d + vb_dst_stride * 5u;
        copy_vertex(v0_src, v3_dst);

        for (uint e = 0u; e < elem_count; e++) {
            uint2 desc = uint2(elems[e * 2u], elems[e * 2u + 1u]);
            uint offset = desc.x;
            uint kind = desc.y & 0xffu;
            uint comp_count = desc.y >> 8u;
            uint comp_bytes = kind == 0u ? 4u : (kind == 3u ? 1u : 2u);
            for (uint c = 0u; c < comp_count; c++) {
                uint at = offset + c * comp_bytes;
                float a = read_comp(kind, v0_src + at);
                float b = read_comp(kind, v1_src + at);
                float f = read_comp(kind, v2_src + at);
                // In the CPU's order, not reassociated.
                precise float v3 = a + f - b;
                write_comp(kind, v3_dst + at, v3);
            }
        }
    }
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

struct Shaders {
    rewrite: Pass,
    expand: Pass,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Shaders {}
// SAFETY: as above.
unsafe impl Sync for Shaders {}

fn shaders() -> Option<&'static Shaders> {
    static SHADERS: std::sync::OnceLock<Option<Shaders>> = std::sync::OnceLock::new();
    SHADERS
        .get_or_init(|| match make() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, "no rectangle expander: indirect rectangles dropped");
                None
            }
        })
        .as_ref()
}

fn compile_blob(
    name: windows::core::PCSTR,
    source: &str,
) -> Result<ID3DBlob, (windows::core::Error, String)> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        let mut text = String::new();
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            text = String::from_utf8_lossy(bytes).into_owned();
            tracing::warn!("rectangle expander: {text}");
        }
        return Err((e, text));
    }
    code.ok_or_else(|| (windows::core::Error::empty(), String::new()))
}

/// A pass's root signature: eight root constants at `b0`, then one root
/// descriptor per entry of `views`, registers counted per kind from 0.
fn root_signature(
    device: &ID3D12Device,
    views: &[D3D12_ROOT_PARAMETER_TYPE],
) -> windows::core::Result<ID3D12RootSignature> {
    let mut params = vec![D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: 10,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }];
    let (mut srv, mut uav) = (0, 0);
    for &kind in views {
        let register = if kind == D3D12_ROOT_PARAMETER_TYPE_SRV {
            &mut srv
        } else {
            &mut uav
        };
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: kind,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: *register,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
        *register += 1;
    }
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: a description of root constants and root descriptors that
    // lives through the call; the out pointer is ours.
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}

fn compile_pass(
    device: &ID3D12Device,
    name: windows::core::PCSTR,
    source: &str,
    views: &[D3D12_ROOT_PARAMETER_TYPE],
) -> windows::core::Result<Pass> {
    let code = compile_blob(name, source).map_err(|(e, _)| e)?;
    let root = root_signature(device, views)?;
    // SAFETY: the blob's own bytes, and a root signature of ours.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Pass { root, pso })
    }
}

fn make() -> windows::core::Result<Shaders> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    const UAV: D3D12_ROOT_PARAMETER_TYPE = D3D12_ROOT_PARAMETER_TYPE_UAV;
    const SRV: D3D12_ROOT_PARAMETER_TYPE = D3D12_ROOT_PARAMETER_TYPE_SRV;
    let rewrite = compile_pass(device, s!("rect_rewrite"), REWRITE_SOURCE, &[UAV; 4])?;
    let expand = compile_pass(
        device,
        s!("rect_expand"),
        EXPAND_SOURCE,
        &[SRV, UAV, UAV, UAV, UAV, UAV],
    )?;
    Ok(Shaders { rewrite, expand })
}

/// What one recording's indirect rectangle draws are expanded into: the
/// vertex scratch, and the arguments buffer (the allocation counter and
/// the record count, then the rewritten records). Taken by a list's draw
/// state on its first such draw, given back ([`give_back`]) once the
/// recording has gone. Its buffers are made as draws need them.
pub(crate) struct Scratch {
    /// Created or grown on demand for a draw with per-vertex inputs.
    vertices: Option<Buffer>,
    /// Made larger when a draw has more records.
    args: Option<Buffer>,
    direct: Option<DirectPage>,
}

// Une page mappée de commandes directes. Les anciennes pages restent dans
// la liste si celle-ci remplit sa page avant la fin de l’enregistrement.
struct DirectPage {
    buffer: ID3D12Resource,
    data: usize,
    used: usize,
}

impl DirectPage {
    fn append(page: &mut Option<Self>, list: &HostList, words: [u32; 5]) -> Option<u64> {
        const BYTES: usize = 64 << 10;
        const STEP: usize = 32;
        if page.as_ref().is_none_or(|p| p.used == BYTES) {
            let (buffer, data) = host()?.shared_buffer(BYTES as u64)?;
            *page = Some(Self {
                buffer,
                data: data as usize,
                used: 0,
            });
        }
        let p = page.as_mut()?;
        if p.used == 0 {
            list.keep(&p.buffer);
        }
        let at = p.used;
        p.used += STEP;
        // SAFETY: cette tranche de 32 octets reste réservée jusqu’à la retraite de la liste.
        unsafe {
            ((p.data + at) as *mut [u32; 5]).write_unaligned(words);
            Some(p.buffer.GetGPUVirtualAddress() + at as u64)
        }
    }
}

/// One buffer of a [`Scratch`].
struct Buffer {
    resource: ID3D12Resource,
    len: u64,
    /// The state the list's commands so far leave it in: common when a
    /// recording starts (buffers decay to it when a submission ends).
    state: D3D12_RESOURCE_STATES,
    /// Whether the host list keeps it (until its reset).
    kept: bool,
}

impl Buffer {
    fn new(len: u64, what: &str) -> Option<Buffer> {
        Some(Buffer {
            resource: uav_buffer(len, what)?,
            len,
            state: D3D12_RESOURCE_STATE_COMMON,
            kept: false,
        })
    }

    /// Kept by `list` until its reset, once.
    fn keep(&mut self, list: &HostList) {
        if !self.kept {
            list.keep(&self.resource);
            self.kept = true;
        }
    }

    /// Moves it to `to` (nothing if it is there).
    fn to(&mut self, barriers: &mut Vec<D3D12_RESOURCE_BARRIER>, to: D3D12_RESOURCE_STATES) {
        to_state(barriers, &self.resource, &mut self.state, to);
    }

    /// Its GPU address.
    fn gpu(&self) -> u64 {
        // SAFETY: a live buffer.
        unsafe { self.resource.GetGPUVirtualAddress() }
    }
}

// SAFETY: D3D12 resources are free-threaded.
unsafe impl Send for Scratch {}

/// Scratches no list holds.
static POOL: Mutex<Vec<Scratch>> = Mutex::new(Vec::new());

/// Serializes the tests that look into [`POOL`] or take from it.
#[cfg(test)]
pub(crate) static POOL_TESTS: Mutex<()> = Mutex::new(());

/// Whether `resource` is a buffer of a pooled scratch (tests).
#[cfg(test)]
pub(crate) fn pooled(resource: &ID3D12Resource) -> bool {
    let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.iter()
        .flat_map(|s| [&s.vertices, &s.args])
        .flatten()
        .any(|b| b.resource == *resource)
}

#[cfg(test)]
impl Scratch {
    /// Its arguments buffer, once made.
    pub(crate) fn arguments(&self) -> Option<&ID3D12Resource> {
        self.args.as_ref().map(|b| &b.resource)
    }
}

/// Gives a recording's scratch back for later lists: its submissions have
/// run, so its buffers are back in the common state. A large arguments
/// buffer goes.
pub(crate) fn give_back(mut scratch: Scratch) {
    if let Some(page) = &mut scratch.direct {
        page.used = 0;
    }
    for b in [&mut scratch.vertices, &mut scratch.args]
        .into_iter()
        .flatten()
    {
        b.state = D3D12_RESOURCE_STATE_COMMON;
        b.kept = false;
    }
    if scratch.args.as_ref().is_some_and(|a| a.len > ARGS_POOLED) {
        scratch.args = None;
    }
    if scratch.vertices.is_none() && scratch.args.is_none() && scratch.direct.is_none() {
        return;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    // Une seule grande réserve inactive, même si plusieurs listes ont grandi.
    let size = |s: &Scratch| s.vertices.as_ref().map_or(0, |v| v.len);
    if size(&scratch) > SCRATCH_BYTES {
        if let Some(other) = pool.iter_mut().find(|s| size(s) > SCRATCH_BYTES) {
            if size(other) < size(&scratch) {
                std::mem::swap(&mut other.vertices, &mut scratch.vertices);
            }
            scratch.vertices = None;
        } else if pool.len() == POOLED {
            pool.pop();
        }
    }
    if pool.len() < POOLED {
        pool.push(scratch);
    }
}

/// A buffer of `size` bytes in GPU memory that shaders can write, in the
/// common state.
fn uav_buffer(size: u64, what: &str) -> Option<ID3D12Resource> {
    let host = host()?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, size, "no {what} for indirect rectangles");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("rectangle {what} {size:#x}"));
    Some(buffer)
}

/// The size an arguments buffer of `len` bytes grows to for `needed`:
/// twice as large at least (few remakes), in steps of [`ARGS_MIN`], at most
/// [`ARGS_MAX`] (rounded up to a step).
fn args_growth(len: u64, needed: u64) -> u64 {
    needed
        .max(len.saturating_mul(2))
        .clamp(ARGS_MIN, ARGS_MAX.max(needed))
        .next_multiple_of(ARGS_MIN)
}

/// The list's scratch, with room for `args_len` bytes of arguments and, if
/// `vertices`, the vertex scratch: the one it holds, one from [`POOL`], or
/// a new one.
fn take_scratch<'a>(
    list: &HostList,
    held: &'a mut Option<Scratch>,
    args_len: u64,
    vertices_len: u64,
) -> Option<&'a mut Scratch> {
    let s = match held {
        Some(s) => s,
        None => {
            let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
            let best = pool
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| {
                    let bytes = s.vertices.as_ref().map_or(0, |v| v.len);
                    (bytes < vertices_len, bytes.abs_diff(vertices_len))
                })
                .map(|(i, _)| i);
            let pooled = best.map(|i| pool.swap_remove(i));
            held.insert(pooled.unwrap_or(Scratch {
                vertices: None,
                args: None,
                direct: None,
            }))
        }
    };
    let have = s.args.as_ref().map_or(0, |a| a.len);
    if have < args_len {
        // The list keeps a smaller one it used until its reset.
        s.args = Some(Buffer::new(args_growth(have, args_len), "arguments")?);
    }
    if vertices_len > s.vertices.as_ref().map_or(0, |v| v.len) {
        if let Some(larger) = Buffer::new(vertices_len, "vertices") {
            s.vertices = Some(larger);
            #[cfg(feature = "diagnostics")]
            if vertices_len > SCRATCH_BYTES {
                crate::list::cuts::GROWTH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else if s.vertices.is_none() {
            // The upper bound includes overlap between records. A much smaller
            // draw can still fit the normal scratch under memory pressure.
            if vertices_len <= SCRATCH_BYTES {
                return None;
            }
            s.vertices = Some(Buffer::new(SCRATCH_BYTES, "vertices")?);
        }
    }
    for b in [&mut s.vertices, &mut s.args].into_iter().flatten() {
        b.keep(list);
    }
    Some(s)
}

/// Moves `buffer` from `*state` to `to` (nothing if it is there).
fn to_state(
    barriers: &mut Vec<D3D12_RESOURCE_BARRIER>,
    buffer: &ID3D12Resource,
    state: &mut D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) {
    if *state != to {
        barriers.push(transition(buffer, *state, to));
        *state = to;
    }
}

/// An expanded indirect rectangle draw, ready for `ExecuteIndirect`.
pub(crate) struct Prepared {
    /// The rewritten records (stride 16), at most `max`, and where they
    /// start; the record count is in the same buffer, at `count_offset`.
    pub(crate) args: ID3D12Resource,
    pub(crate) offset: u64,
    pub(crate) count_offset: u64,
    pub(crate) max: u32,
    /// The expanded per-vertex slots.
    views: Vec<(u32, D3D12_VERTEX_BUFFER_VIEW)>,
}

impl Prepared {
    /// Binds the expanded per-vertex slots, over what the draw's own
    /// preparation bound there.
    pub(crate) fn bind_vertices(&self, list: &HostList) {
        for (slot, view) in &self.views {
            // SAFETY: an open list; the view names the list's scratch.
            unsafe { list.list.IASetVertexBuffers(*slot, Some(&[*view])) };
        }
    }
}

/// A source vertex slot of the draw.
struct Source {
    slot: u32,
    shadow: Arc<Shadow>,
    address: u64,
    size: u32,
    stride: u32,
    elements: Vec<Element>,
}

#[derive(Clone, Copy)]
pub(crate) enum Records<'a> {
    Title(&'a Arc<Shadow>, u64),
    Direct([u32; 5]),
}

/// État partagé par les lots d’un seul dessin, dont la liste conserve les commandes bornées.
#[derive(Default)]
pub(crate) struct Batches {
    writes: crate::indirect::Writes,
    bounded_gpu: Option<u64>,
}

impl Batches {
    pub(crate) fn new(writes: crate::indirect::Writes) -> Self {
        Self {
            writes,
            bounded_gpu: None,
        }
    }
}

/// Clamps an indirect rectangle draw's `max` records of `stride` bytes at
/// title address `arguments` (host copy `args`), the record count at
/// `count` if the title gives one, and records the passes that rewrite
/// them into `held` (the list's scratch) with the vertices of the
/// per-vertex slots of `inputs` (`vertices`: the bound views, (address,
/// bytes, stride) by slot) expanded. On success the compute root signature
/// and pipeline are ours: the caller binds its draw, then
/// [`Prepared::bind_vertices`], then runs `ExecuteIndirect` on
/// [`Prepared::args`] with stride 16, the same buffer as its count buffer.
/// The list's recording must hold `held` until it has gone. On error
/// nothing was recorded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn expand(
    list: &HostList,
    held: &mut Option<Scratch>,
    inputs: &[Input],
    vertices: &[(u64, u32, u32); 32],
    (records, stride, range): (Records, u32, std::ops::Range<u32>),
    index: Option<crate::draw::IndexView>,
    count: Option<(&Arc<Shadow>, u64)>,
    stats: Option<(u64, u32)>,
    batches: &mut Batches,
) -> Result<Prepared, &'static str> {
    if stride < if index.is_some() { 20 } else { 16 }
        || !stride.is_multiple_of(4)
        || matches!(records, Records::Title(s, at) if !s.gpu(at).is_multiple_of(4))
    {
        return Err("rectangle records short or not 4-byte aligned");
    }
    let total = range.end;
    let first = range.start;
    if first > total || total > MAX_RECORDS {
        return Err("too many indirect rectangle records");
    }
    if let Some((s, at)) = count
        && !s.gpu(at).is_multiple_of(4)
    {
        return Err("rectangle count not 4-byte aligned");
    }
    let mut sources = Vec::new();
    let mut bounds = Vec::new();
    for slot in 0..32u32 {
        let (address, size, slot_stride) = vertices[slot as usize];
        let (per_vertex, elements) = slot_elements(inputs, slot, slot_stride);
        if !per_vertex {
            continue;
        }
        if address == 0 || slot_stride == 0 || size == 0 {
            return Err("rectangle vertex buffer not bound");
        }
        let Some(shadow) = crate::shadow::find(address, u64::from(size)) else {
            return Err("rectangle vertex buffer outside title buffers");
        };
        // Within the view, and in the host copy, which is larger than the
        // title buffer (rounded up to 64 KiB): every word the expansion
        // reads holds a byte of the vertices it needs, so the copy's end
        // would do; 4 bytes short of it are a margin, as a root view has
        // no bounds.
        // SAFETY: a live buffer.
        let width = unsafe { shadow.buffer.GetDesc() }.Width;
        let readable = u64::from(size)
            .min(readable_in(width, shadow.offset(address)))
            .min(u64::from(u32::MAX - 3));
        bounds.push((readable, slot_stride));
        sources.push(Source {
            slot,
            shadow,
            address,
            size,
            stride: slot_stride,
            elements,
        });
    }
    let strides: Vec<(u32, u32)> = sources.iter().map(|s| (s.slot, s.stride)).collect();
    let max_src_verts = max_source_vertices(&bounds);
    let indices = index
        .map(|v| {
            let bytes = match v.format {
                57 => 2,
                42 => 4,
                _ => return Err("rectangle index format is not R16_UINT or R32_UINT"),
            };
            if !v.address.is_multiple_of(bytes) || v.size == 0 {
                return Err("invalid rectangle index view");
            }
            let shadow = crate::shadow::find(v.address, v.size.into())
                .ok_or("rectangle index buffer outside title buffers")?;
            Ok((v, bytes as u32, shadow))
        })
        .transpose()?;
    let source_limit = indices
        .as_ref()
        .map_or(max_src_verts, |(v, bytes, _)| v.size / bytes);
    // Every active record may reuse the entire source. The clamp also limits
    // a record to 2^24 vertices and a list to 2^27 vertex instances.
    let max_per_record = match records {
        Records::Direct(words) => words[0],
        Records::Title(..) => 1 << 24,
    };
    // Le compteur côté titre guide la taille des lots. Le GPU relit le vrai
    // compteur pour chacun, même si un dispatch précédent vient de le modifier.
    let per_record = u64::from(source_limit.min(max_per_record) / 3) * 6;
    let stride_sum: u64 = strides
        .iter()
        .map(|&(_, n)| u64::from(n).next_multiple_of(4))
        .sum();
    let bytes_per_record = per_record.saturating_mul(stride_sum).max(1);
    let max = if bytes_per_record.saturating_mul(u64::from(total)) <= SCRATCH_BYTES {
        total - first
    } else {
        let mut word = [0; 4];
        let hint = count
            .and_then(|(_, at)| {
                crate::host::read_into(at as *const u8, &mut word).then(|| u32::from_le_bytes(word))
            })
            .unwrap_or(total);
        // Borne le nombre de passages enregistrés et réutilise les mêmes
        // sommets entre lots, y compris si le compteur CPU est périmé.
        let budget = (4 * SCRATCH_BYTES / bytes_per_record).clamp(1, u64::from(u32::MAX)) as u32;
        hint.min(budget)
            .max(budget / 4)
            .max(total.div_ceil(64))
            .min(total - first)
    };
    let vertices_needed =
        (u64::from(source_limit.min(max_per_record) / 3) * u64::from(max)).min((1 << 27) / 3) * 6;
    let mut needed = 0u64;
    for &(_, stride) in &strides {
        let stride = stride
            .checked_next_multiple_of(4)
            .ok_or("rectangle stride overflow")?;
        needed = (needed.next_multiple_of(16) + vertices_needed * u64::from(stride))
            .min(u64::from(u32::MAX));
    }
    // A vertex view and the shader's byte offsets are 32-bit.
    let bytes = if strides.is_empty() {
        0
    } else {
        needed.clamp(SCRATCH_BYTES, u64::from(u32::MAX - 15))
    };
    let Some(passes) = shaders() else {
        return Err("no rectangle expander");
    };
    let records_len = u64::from(max.max(1)) * 16;
    let need_vertices = !sources.is_empty();
    let Some(scratch) = take_scratch(list, held, COUNTER_BYTES + records_len, bytes) else {
        return Err("no scratch for rectangles");
    };
    let available = scratch.vertices.as_ref().map_or(0, |v| v.len).min(bytes);
    let (capacity, layouts) =
        scratch_layout(&strides, available).ok_or("rectangle vertices larger than the scratch")?;
    let (Some(args_buffer), vertices_buffer) = (scratch.args.as_mut(), scratch.vertices.as_mut())
    else {
        return Err("no scratch for rectangles");
    };
    // The element tables, written now; the GPU reads them when it runs.
    let mut tables = Vec::with_capacity(sources.len());
    for s in &sources {
        let bytes = (s.elements.len().max(1) * size_of::<Element>()) as u64;
        let Some((gpu, data)) = list.scratch(bytes) else {
            return Err("no upload memory for rectangle elements");
        };
        // SAFETY: fresh upload memory of the list, `bytes` long.
        unsafe {
            std::ptr::copy_nonoverlapping(
                s.elements.as_ptr().cast::<u8>(),
                data,
                s.elements.len() * size_of::<Element>(),
            );
        }
        tables.push(gpu);
    }
    let l2 = list
        .list
        .cast::<ID3D12GraphicsCommandList2>()
        .map_err(|_| "no immediate writes on the host list")?;

    // The title's buffers, as the title left them at this place.
    let (src_args, count_gpu, stride) = match records {
        Records::Title(args, at) => {
            let gpu = if let Some(gpu) = batches.bounded_gpu {
                gpu
            } else {
                args.refresh(list, at, u64::from(stride) * u64::from(total));
                if let Some((s, at)) = count {
                    s.refresh(list, at, 4);
                }
                let bounded = crate::indirect::bound(
                    list,
                    (&args.buffer, args.offset(at)),
                    (u32::from(index.is_some()), stride, total),
                    count.map(|(s, at)| (&s.buffer, s.offset(at))),
                    stats,
                    batches.writes,
                    crate::indirect::Output::Uav,
                )
                .ok_or("not bounded (no clamp)")?;
                // SAFETY: la liste conserve le tampon du compteur et des commandes bornées.
                let gpu = unsafe { bounded.buffer.GetGPUVirtualAddress() } + bounded.offset;
                batches.bounded_gpu = Some(gpu);
                gpu
            };
            (gpu + 4, gpu, 24)
        }
        Records::Direct(words) => {
            let gpu = DirectPage::append(&mut scratch.direct, list, words)
                .ok_or("no memory for direct rectangle arguments")?;
            (gpu, gpu, stride)
        }
    };
    if first == 0 {
        if let Some((v, _, shadow)) = &indices {
            shadow.refresh(list, v.address, v.size.into());
        }
        for s in &sources {
            s.shadow.refresh(list, s.address, u64::from(s.size));
        }
    }
    let _expanding =
        crate::timestamps::command(list, crate::timestamps::What::RectangleExpansion, None);

    let src_args = src_args + u64::from(first) * u64::from(stride);
    let direct = matches!(records, Records::Direct(_));
    let args_gpu = args_buffer.gpu();
    let (counter, records) = (args_gpu, args_gpu + COUNTER_BYTES);
    let rewrite = [
        max,
        stride,
        u32::from(!direct),
        source_limit,
        capacity,
        u32::from(need_vertices),
        u32::from(index.is_some()),
        first,
    ];
    let mut barriers = Vec::new();
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    // A direct indexed draw can follow a dispatch which wrote its indices.
    if direct {
        barriers.push(uav_barrier());
    }
    args_buffer.to(&mut barriers, D3D12_RESOURCE_STATE_COPY_DEST);
    let mut vertices = None;
    if need_vertices && let Some(v) = vertices_buffer {
        v.to(&mut barriers, uav);
        vertices = Some(v);
    }
    let l = &list.list;
    // SAFETY: an open list; our passes and buffers the list keeps (the
    // title's host copies through their refresh).
    unsafe {
        if !barriers.is_empty() {
            l.ResourceBarrier(&barriers);
        }
        // This draw's allocations start from zero.
        l2.WriteBufferImmediate(
            1,
            &D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                Dest: counter,
                Value: 0,
            },
            None,
        );
        barriers.clear();
        args_buffer.to(&mut barriers, uav);
        l.ResourceBarrier(&barriers);
        l.SetComputeRootSignature(&passes.rewrite.root);
        l.SetPipelineState(&passes.rewrite.pso);
        l.SetComputeRoot32BitConstants(0, 8, rewrite.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, src_args);
        l.SetComputeRootUnorderedAccessView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, counter);
        l.SetComputeRootUnorderedAccessView(4, records);
        l.Dispatch(max.div_ceil(64), 1, 1);
    }
    let mut views = Vec::with_capacity(sources.len());
    if let Some(v) = &vertices {
        let vertices_gpu = v.gpu();
        let (grid_x, grid_y) = dispatch_grid(max);
        // SAFETY: as above.
        unsafe {
            // The rewritten records, for the expansion.
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(&passes.expand.root);
            l.SetPipelineState(&passes.expand.pso);
            l.SetComputeRootUnorderedAccessView(2, src_args);
            l.SetComputeRootUnorderedAccessView(3, records);
            let (index_gpu, index_bytes) = indices
                .as_ref()
                .map_or((src_args, 0), |(v, bytes, shadow)| {
                    (shadow.gpu(v.address), *bytes)
                });
            l.SetComputeRootUnorderedAccessView(6, index_gpu & !3);
            for ((s, layout), table) in sources.iter().zip(&layouts).zip(tables) {
                let src = s.shadow.gpu(s.address);
                let to = vertices_gpu + layout.dst_offset;
                let expand = [
                    max,
                    stride,
                    grid_x,
                    (src & 3) as u32,
                    s.stride,
                    layout.dst_stride,
                    s.elements.len() as u32,
                    index_bytes,
                    (index_gpu & 3) as u32,
                    max_src_verts,
                ];
                l.SetComputeRoot32BitConstants(0, 10, expand.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, table);
                l.SetComputeRootUnorderedAccessView(4, src & !3);
                l.SetComputeRootUnorderedAccessView(5, to);
                l.Dispatch(grid_x, grid_y, 1);
                views.push((
                    layout.slot,
                    D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: to,
                        SizeInBytes: layout.dst_size,
                        StrideInBytes: layout.dst_stride,
                    },
                ));
            }
        }
    }
    barriers.clear();
    // The records and their count, for `ExecuteIndirect`.
    args_buffer.to(&mut barriers, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    if let Some(v) = vertices {
        v.to(
            &mut barriers,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
        );
    }
    // SAFETY: an open list and buffers it keeps.
    unsafe { l.ResourceBarrier(&barriers) };
    Ok(Prepared {
        args: args_buffer.resource.clone(),
        offset: COUNTER_BYTES,
        count_offset: COUNT_OFFSET,
        max,
        views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgpu::{TitleBuffer, TitleList};

    fn input(slot: u32, offset: u32, format: u32, per_instance: bool) -> Input {
        Input {
            slot,
            offset,
            format,
            per_instance,
        }
    }

    #[test]
    fn slot_elements_filter_and_stop_at_the_stride() {
        let inputs = [
            input(0, 0, 16, false),  // R32G32_FLOAT, 8 bytes
            input(0, 8, 28, false),  // R8G8B8A8_UNORM, 4 bytes
            input(0, 4, 2, false),   // R32G32B32A32_FLOAT, 2 components inside 12 bytes
            input(0, 12, 28, false), // past the stride: none
            input(0, 0, 42, false),  // R32_UINT, unknown: none
            input(1, 0, 16, true),   // per instance: ignored
            input(2, 0, 42, false),  // unknown only: per vertex, no element
        ];
        let (per_vertex, elements) = slot_elements(&inputs, 0, 12);
        assert!(per_vertex);
        let element = |offset, kind: primitive::Kind, n: u32| Element {
            offset,
            packed: kind.code() | n << 8,
        };
        assert_eq!(
            elements,
            [
                element(0, primitive::Kind::F32, 2),
                element(8, primitive::Kind::Unorm8, 4),
                element(4, primitive::Kind::F32, 2),
            ]
        );
        assert_eq!(slot_elements(&inputs, 1, 16), (false, Vec::new()));
        assert_eq!(slot_elements(&inputs, 2, 4), (true, Vec::new()));
    }

    #[test]
    fn scratch_layouts_align_and_fit() {
        assert_eq!(scratch_layout(&[], 1024), Some((0, Vec::new())));
        assert_eq!(scratch_layout(&[(0, 0)], 1024), None);
        assert_eq!(scratch_layout(&[(0, 16)], 64), None, "6 * 16 > 64");

        // Strides 6 (padded to 8) and 12.
        let (cap, layouts) = scratch_layout(&[(0, 6), (3, 12)], 1024).unwrap();
        assert_eq!(cap % 6, 0);
        assert!(cap >= 6);
        assert_eq!(layouts.len(), 2);
        assert_eq!((layouts[0].slot, layouts[0].dst_offset), (0, 0));
        assert_eq!(layouts[0].dst_stride, 8);
        assert_eq!(layouts[0].dst_size, cap * 8);
        assert_eq!(layouts[1].slot, 3);
        assert_eq!(layouts[1].dst_offset % 16, 0);
        assert!(layouts[1].dst_offset >= u64::from(layouts[0].dst_size));
        assert_eq!(layouts[1].dst_stride, 12);
        assert_eq!(layouts[1].dst_size, cap * 12);
        assert!(layouts[1].dst_offset + u64::from(layouts[1].dst_size) <= 1024);

        // An exact fit, with no padding needed: 6 vertices of 8 bytes, then
        // 6 of 12 (48 is 16-byte aligned).
        let (cap, layouts) = scratch_layout(&[(0, 8), (1, 12)], 120).unwrap();
        assert_eq!((cap, layouts[1].dst_offset), (6, 48));
        // Padding counted where needed: 24 bytes, aligned to 32, 24 more.
        assert_eq!(scratch_layout(&[(0, 4), (1, 4)], 55), None);
        let (cap, layouts) = scratch_layout(&[(0, 4), (1, 4)], 56).unwrap();
        assert_eq!((cap, layouts[1].dst_offset), (6, 32));
    }

    #[test]
    fn the_source_bound_is_the_smallest_slots() {
        assert_eq!(max_source_vertices(&[]), u32::MAX);
        assert_eq!(max_source_vertices(&[(120, 12), (80, 16)]), 5);
        assert_eq!(max_source_vertices(&[(100, 0), (80, 16)]), 0);
        // A view is read up to 4 bytes short of its host copy's end, not of
        // its title buffer's: 72 bytes of vertices in a 64 KiB copy are all
        // readable, from its start or from 8 bytes short of its end.
        let width = 64 << 10;
        assert_eq!(readable_in(width, 0).min(72), 72);
        assert_eq!(readable_in(width, width - 8), 4);
        assert_eq!(readable_in(width, width - 2), 0);
    }

    /// The arguments buffer starts small, doubles at least when it grows,
    /// and stops at what the most records take.
    #[test]
    fn arguments_buffers_grow_by_doubling_up_to_the_most_records() {
        assert_eq!(args_growth(0, COUNTER_BYTES + 16), ARGS_MIN);
        assert_eq!(args_growth(ARGS_MIN, ARGS_MIN + 16), 2 * ARGS_MIN);
        assert_eq!(args_growth(ARGS_MIN, 5 * ARGS_MIN + 1), 6 * ARGS_MIN);
        assert_eq!(args_growth(48 << 20, 48 << 20 | 16), 64 << 20, "capped");
        assert_eq!(args_growth(0, ARGS_MAX), 64 << 20);
    }

    #[test]
    fn dispatch_grids_stay_within_the_limit() {
        assert_eq!(dispatch_grid(0), (0, 0));
        assert_eq!(dispatch_grid(1), (1, 1));
        assert_eq!(dispatch_grid(65535), (65535, 1));
        let (x, y) = dispatch_grid(65536);
        assert!(x <= 65535 && y <= 65535);
        assert!(u64::from(x) * u64::from(y) >= 65536);
    }

    /// Both passes compile with the system's compiler and the host takes
    /// them.
    #[test]
    fn the_rectangle_expander_builds() {
        let test = "rectangles::tests::the_rectangle_expander_builds";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        assert!(shaders().is_some());
    }

    /// One vertex of the byte-for-byte test (24 bytes): R32G32_FLOAT at 0,
    /// R16G16_FLOAT at 8, R16G16_UNORM at 12, R8G8B8A8_UNORM at 16, a word
    /// of an unknown format at 20 (keeps v0's bytes); an element at 24 is
    /// past the stride.
    fn vertex(x: f32, y: f32, u: u16, v: u16, n: [u16; 2], rgba: [u8; 4], tag: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(STRIDE);
        out.extend(x.to_le_bytes());
        out.extend(y.to_le_bytes());
        out.extend(u.to_le_bytes());
        out.extend(v.to_le_bytes());
        out.extend(n[0].to_le_bytes());
        out.extend(n[1].to_le_bytes());
        out.extend(rgba);
        out.extend(tag.to_le_bytes());
        out
    }
    const STRIDE: usize = 24;

    /// Three rectangles: a full-screen one; one with clamped colours,
    /// negative halves and a half subnormal fourth corner (2^-16); one
    /// whose u fourth corner is 1 + 2^-11, halfway between two halves (the
    /// even 0x3c00).
    fn rectangle_vertices() -> Vec<u8> {
        [
            vertex(-1.0, 1.0, 0, 0, [0, 10000], [0, 50, 200, 255], 0x1122_3344),
            vertex(-1.0, -1.0, 0, 0x3c00, [10000, 30000], [100; 4], 0x5566_7788),
            vertex(
                1.0,
                -1.0,
                0x3c00,
                0x3c00,
                [40000, 50000],
                [150, 200, 50, 255],
                0x99aa_bbcc,
            ),
            vertex(
                0.25,
                0.5,
                0xb800,
                0x0400,
                [50000, 1000],
                [220, 10, 128, 64],
                0xdead_beef,
            ),
            vertex(
                0.25,
                -0.5,
                0xbc00,
                0x0600,
                [10000, 20000],
                [50, 200, 250, 200],
                0xcafe_babe,
            ),
            vertex(
                2.5,
                -0.5,
                0xc000,
                0x0300,
                [60000, 5000],
                [200, 20, 10, 100],
                0x0102_0304,
            ),
            vertex(
                10.0,
                20.0,
                0x3c00,
                0x4200,
                [12345, 54321],
                [10, 20, 30, 40],
                0xaabb_ccdd,
            ),
            vertex(
                10.0,
                30.0,
                0,
                0x4400,
                [22222, 11111],
                [50, 60, 70, 80],
                0x1234_5678,
            ),
            vertex(
                40.0,
                30.0,
                0x1000,
                0x4000,
                [33333, 22222],
                [90, 80, 70, 60],
                0x8765_4321,
            ),
        ]
        .concat()
    }

    fn rectangle_inputs() -> Vec<Input> {
        vec![
            input(0, 0, 16, false),
            input(0, 8, 34, false),
            input(0, 12, 35, false),
            input(0, 16, 28, false),
            input(0, 20, 42, false),
            input(0, 24, 28, false),
        ]
    }

    /// The word `i` of `bytes` (zeros past them).
    fn word(bytes: &[u8], i: usize) -> u32 {
        let mut w = [0u8; 4];
        for (k, b) in w.iter_mut().enumerate() {
            *b = bytes.get(i * 4 + k).copied().unwrap_or(0);
        }
        u32::from_le_bytes(w)
    }

    /// A title buffer of `len` bytes starting with `bytes`, and its host
    /// copy.
    fn title_bytes(len: usize, bytes: &[u8]) -> (TitleBuffer, Arc<Shadow>) {
        let buffer = TitleBuffer::new(len);
        buffer.set_words(|i| word(bytes, i));
        let copy = buffer.shadow();
        (buffer, copy)
    }

    /// Title arguments holding `records`, and their host copy.
    fn title_records(records: &[[u32; 4]]) -> (TitleBuffer, Arc<Shadow>) {
        let bytes: Vec<u8> = records
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        title_bytes(bytes.len().max(256), &bytes)
    }

    /// What a list's indirect rectangle draws left in its scratch, read
    /// back once the list has run.
    struct Left {
        /// The allocation counter and the record count `ExecuteIndirect`
        /// reads.
        counter: u32,
        count: u32,
        /// The first rewritten records.
        records: Vec<[u32; 4]>,
        /// The vertex scratch's bytes in the ranges asked for; none if it
        /// was not made.
        vertices: Option<Vec<Vec<u8>>>,
    }

    /// Runs `list` with copies of its scratch `held`: the first `records`
    /// rewritten records and the vertex scratch's `ranges` (offset, bytes).
    fn run_and_read(
        list: &TitleList,
        held: &mut Option<Scratch>,
        records: usize,
        ranges: &[(u64, usize)],
    ) -> Left {
        let host = host().expect("a device");
        let l = list.host();
        let scratch = held.as_mut().expect("the list's scratch");
        let args_bytes = COUNTER_BYTES + 16 * records as u64;
        let (out_args, out_args_data) = host.shared_buffer(args_bytes).expect("readback");
        let mut barriers = Vec::new();
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let args = scratch.args.as_mut().expect("an arguments buffer");
        args.to(&mut barriers, source);
        let vertices = scratch.vertices.as_mut().map(|v| {
            v.to(&mut barriers, source);
            v.resource.clone()
        });
        let outs: Vec<(ID3D12Resource, *mut u8)> = match &vertices {
            Some(_) => ranges
                .iter()
                .map(|&(_, len)| host.shared_buffer(len as u64).expect("readback"))
                .collect(),
            None => Vec::new(),
        };
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            l.list.ResourceBarrier(&barriers);
            l.list
                .CopyBufferRegion(&out_args, 0, &args.resource, 0, args_bytes);
            if let Some(v) = &vertices {
                for (&(at, len), (out, _)) in ranges.iter().zip(&outs) {
                    l.list.CopyBufferRegion(out, 0, v, at, len as u64);
                }
            }
        }
        list.run();
        // SAFETY: the list has run; mapped buffers of these sizes.
        unsafe {
            let words = std::slice::from_raw_parts(out_args_data.cast::<u32>(), 4 + 4 * records);
            Left {
                counter: words[0],
                count: words[1],
                records: words[4..]
                    .chunks(4)
                    .map(|c| c.try_into().unwrap())
                    .collect(),
                vertices: vertices.map(|_| {
                    ranges
                        .iter()
                        .zip(&outs)
                        .map(|(&(_, len), &(_, data))| {
                            std::slice::from_raw_parts(data, len).to_vec()
                        })
                        .collect()
                }),
            }
        }
    }

    /// Gives the test's scratch back, as its recording's going would.
    fn done(held: &mut Option<Scratch>) {
        if let Some(s) = held.take() {
            give_back(s);
        }
    }

    /// Checks the six vertices a rectangle each of `rects` rectangles from
    /// source vertex `from` became, starting at expanded vertex `at` of a
    /// region with `dst_stride`-byte vertices, against
    /// [`primitive::expand_rects`] (the first `stride` bytes of each).
    fn check_expanded(
        region: &[u8],
        dst_stride: usize,
        at: usize,
        (source, stride, elements): (&[u8], usize, &[(u32, u32)]),
        (from, rects): (usize, usize),
        what: &str,
    ) {
        let mut expected = vec![0u8; rects * 6 * stride];
        let src = &source[from * stride..(from + rects * 3) * stride];
        primitive::expand_rects(src, &mut expected, stride, elements);
        for (k, want) in expected.chunks(stride).enumerate() {
            let got = &region[(at + k) * dst_stride..][..stride];
            assert_eq!(got, want, "{what}: expanded vertex {k}");
        }
    }

    /// The whole expansion as a draw records it, on title buffers: clamp,
    /// counter, rewrite and expansion, twice in one list (the second draw's
    /// allocations start from zero again), compared with
    /// [`primitive::expand_rects`] byte for byte. Records: two rectangles
    /// from vertex 0; one from vertex 6 with 3 instances from 7; one past
    /// the vertex buffer; one without instances; one past the title's
    /// count (4), which is the count `ExecuteIndirect` reads.
    #[test]
    fn expansion_on_the_gpu_matches_the_cpus() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::tests::expansion_on_the_gpu_matches_the_cpus";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let source = rectangle_vertices();
        let (args, args_copy) = title_records(&[
            [6, 1, 0, 0],
            [3, 3, 6, 7],
            [3, 1, 8, 0],
            [3, 0, 0, 0],
            [3, 1, 0, 0],
        ]);
        let (count, count_copy) = title_bytes(256, &4u32.to_le_bytes());
        let (vertices, _vertex_copy) = title_bytes(4096, &source);
        let inputs = rectangle_inputs();
        let mut bound = [(0u64, 0u32, 0u32); 32];
        bound[0] = (vertices.address(), source.len() as u32, STRIDE as u32);

        let list = TitleList::new();
        let mut held = None;
        let mut draw = || {
            expand(
                list.host(),
                &mut held,
                &inputs,
                &bound,
                (Records::Title(&args_copy, args.address()), 16, 0..5),
                None,
                Some((&count_copy, count.address())),
                None,
                &mut Default::default(),
            )
            .expect("expanded")
        };
        let first = draw();
        let second = draw();
        assert_eq!(
            (first.offset, first.count_offset, first.max),
            (COUNTER_BYTES, COUNT_OFFSET, 5)
        );
        assert_eq!(first.views.len(), 1);
        assert_eq!(first.views[0].0, 0);
        assert_eq!(first.views[0].1.StrideInBytes, STRIDE as u32);
        drop((first, second));
        // What the second draw left.
        let left = run_and_read(&list, &mut held, 5, &[(0, 18 * STRIDE)]);
        assert_eq!(
            left.counter, 18,
            "12 + 6 vertices, from zero for the second draw"
        );
        assert_eq!(left.count, 4, "the title's count");
        let rewritten = &left.records;
        assert_eq!(rewritten[0][0], 12);
        assert_eq!(rewritten[0][1], 1);
        assert_eq!(rewritten[0][3], 0);
        assert_eq!(rewritten[1][0], 6);
        assert_eq!(rewritten[1][1], 3);
        assert_eq!(rewritten[1][3], 7);
        assert_eq!(rewritten[2][0], 0, "past the vertex buffer");
        assert_eq!(rewritten[3][0], 0, "no instance");
        assert_eq!(rewritten[4], [0; 4], "past the count");
        let expanded = &left.vertices.expect("a vertex scratch")[0];
        let elements: Vec<(u32, u32)> = inputs.iter().map(|i| (i.offset, i.format)).collect();
        for (record, from, rects) in [(0, 0, 2), (1, 6, 1)] {
            check_expanded(
                expanded,
                STRIDE,
                rewritten[record][2] as usize,
                (&source, STRIDE, &elements),
                (from, rects),
                &format!("record {record}"),
            );
        }
        done(&mut held);
    }

    #[test]
    fn rectangle_records_fill_the_grown_vertex_scratch() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::tests::rectangle_records_fill_the_grown_vertex_scratch";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        // 1025 records, each covering 64 rectangles of 24-byte vertices.
        let max = 1025;
        let count = 3 * 64;
        let stride = 24;
        let (vb, _) = title_bytes((count * stride) as usize, &[]);
        let (args, shadow) = title_records(&vec![[count, 1, 0, 0]; max as usize]);
        let mut bound = [(0, 0, 0); 32];
        bound[0] = (vb.address(), count * stride, stride);
        let list = TitleList::new();
        let mut held = None;
        expand(
            list.host(),
            &mut held,
            &[input(0, 0, 41, false)],
            &bound,
            (Records::Title(&shadow, args.address()), 16, 0..max),
            None,
            None,
            None,
            &mut Default::default(),
        )
        .unwrap();
        let left = run_and_read(&list, &mut held, max as usize, &[]);
        assert_eq!(left.counter, max * count * 2);
        assert!(u64::from(left.counter * stride) > SCRATCH_BYTES);
        assert!(left.records.iter().all(|row| row[0] == count * 2));
        host.drain();
        let grown = held
            .as_ref()
            .unwrap()
            .vertices
            .as_ref()
            .unwrap()
            .resource
            .clone();
        done(&mut held);
        let next = TitleList::new();
        let mut next_scratch = None;
        let s = take_scratch(
            next.host(),
            &mut next_scratch,
            ARGS_MIN,
            u64::from(left.counter * stride),
        )
        .unwrap();
        assert_eq!(s.vertices.as_ref().unwrap().resource, grown);
        next.run();
        done(&mut next_scratch);
    }

    #[test]
    fn rectangle_batches_use_the_gpu_count_after_a_copy() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(_gpu) = crate::testgpu::device(
            "rectangles::tests::rectangle_batches_use_the_gpu_count_after_a_copy",
        ) else {
            return;
        };
        let max = 64u32;
        let (vb, _) = title_bytes(1 << 20, &[]);
        let (args, shadow) = title_records(&vec![[3, 1, 0, 7]; max as usize]);
        let (count, count_shadow) = title_bytes(4, &1u32.to_le_bytes());
        let (actual, _) = title_bytes(4, &(max - 1).to_le_bytes());
        let list = TitleList::new();
        assert!(crate::shadow::copy(
            list.host(),
            count.address(),
            actual.address(),
            4
        ));
        let mut bound = [(0, 0, 0); 32];
        bound[0] = (vb.address(), 1 << 20, 4);
        let mut held = None;
        let mut first = 0;
        let mut batch;
        let mut batches = Batches::default();
        loop {
            batch = expand(
                list.host(),
                &mut held,
                &[input(0, 0, 41, false)],
                &bound,
                (Records::Title(&shadow, args.address()), 16, first..max),
                None,
                Some((&count_shadow, count.address())),
                None,
                &mut batches,
            )
            .unwrap()
            .max;
            assert!(batch < max);
            first += batch;
            if first == max {
                break;
            }
        }
        let left = run_and_read(&list, &mut held, batch as usize, &[]);
        assert_eq!(left.count, batch - 1);
        assert_eq!(left.counter, (batch - 1) * 6);
        assert_eq!(left.records.last().unwrap()[0], 0);
        done(&mut held);
    }

    #[test]
    fn direct_records_keep_the_first_page_alive_after_rollover() {
        let Some(_gpu) = crate::testgpu::device(
            "rectangles::tests::direct_records_keep_the_first_page_alive_after_rollover",
        ) else {
            return;
        };
        let list = TitleList::new();
        let mut page = None;
        let first = DirectPage::append(&mut page, list.host(), [17; 5]).unwrap();
        let data = page.as_ref().unwrap().data as *const [u32; 5];
        let second = DirectPage::append(&mut page, list.host(), [23; 5]).unwrap();
        assert_eq!(second - first, 32);
        for _ in 2..2048 {
            DirectPage::append(&mut page, list.host(), [0; 5]).unwrap();
        }
        let next = DirectPage::append(&mut page, list.host(), [42; 5]).unwrap();
        assert_ne!(next, first);
        // SAFETY: la liste conserve la première page mappée.
        assert_eq!(unsafe { data.read_unaligned() }, [17; 5]);
        list.run();
    }

    #[test]
    fn indexed_rectangles_gather_gpu_indices_and_keep_the_instance_start() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test =
            "rectangles::tests::indexed_rectangles_gather_gpu_indices_and_keep_the_instance_start";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let source = rectangle_vertices();
        let (vertices, _copy) = title_bytes(source.len(), &source);
        let mut bound = [(0, 0, 0); 32];
        bound[0] = (vertices.address(), source.len() as u32, STRIDE as u32);
        let inputs = rectangle_inputs();
        let elements: Vec<_> = inputs.iter().map(|i| (i.offset, i.format)).collect();
        let mut expected = vec![0; 6 * STRIDE];
        let triplet: Vec<u8> = [2, 0, 1]
            .iter()
            .flat_map(|&i| source[i * STRIDE..(i + 1) * STRIDE].iter().copied())
            .collect();
        primitive::expand_rects(&triplet, &mut expected, STRIDE, &elements);
        for (format, bytes) in [(57, 2usize), (42, 4)] {
            // R16 : 0xffff devient le sommet 2 après application de la base.
            // R32 : le triplet hors limites reste dégénéré, le suivant reste aligné.
            let (raw_indices, base) = if bytes == 2 {
                ([99u32, 99, 9, 0, 0, 0xffff, 0xfffd, 0xfffe], -65533i32)
            } else {
                ([99u32, 99, 1, u32::MAX, 2, 3, 1, 2], -1i32)
            };
            let raw: Vec<u8> = raw_indices
                .into_iter()
                .flat_map(|i| i.to_le_bytes().into_iter().take(bytes))
                .collect();
            let (ib, _) = title_bytes(raw.len().next_multiple_of(4), &raw);
            let index = crate::draw::IndexView {
                address: ib.address() + bytes as u64,
                size: (raw.len() - bytes) as u32,
                format,
            };
            let record = [6u32, 2, 1, base as u32, 7];
            let words: Vec<u8> = record.iter().flat_map(|w| w.to_le_bytes()).collect();
            let (args, shadow) = title_bytes(256, &words);
            let records = if bytes == 2 {
                Records::Title(&shadow, args.address())
            } else {
                Records::Direct(record)
            };
            let list = TitleList::new();
            let mut held = None;
            expand(
                list.host(),
                &mut held,
                &inputs,
                &bound,
                (records, 20, 0..1),
                Some(index),
                None,
                None,
                &mut Default::default(),
            )
            .expect("expanded");
            let left = run_and_read(&list, &mut held, 1, &[(0, 12 * STRIDE)]);
            assert_eq!(left.records, [[12, 2, 0, 7]]);
            let expanded = left.vertices.unwrap();
            assert_eq!(expanded[0][..6 * STRIDE], vec![0; 6 * STRIDE]);
            assert_eq!(expanded[0][6 * STRIDE..], expected);
            done(&mut held);
        }
        host.drain();
    }

    /// A vertex buffer of exactly three vertices draws its rectangle: the
    /// source is bounded by its host copy (rounded up to 64 KiB), not 4
    /// bytes short of the title buffer's end, which dropped the record.
    #[test]
    fn a_buffer_of_three_vertices_draws_its_rectangle() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::tests::a_buffer_of_three_vertices_draws_its_rectangle";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let source = rectangle_vertices()[..3 * STRIDE].to_vec();
        let (vertices, copy) = title_bytes(3 * STRIDE, &source);
        // SAFETY: a live buffer.
        let width = unsafe { copy.buffer.GetDesc() }.Width;
        assert!(
            width >= 3 * STRIDE as u64 + 4,
            "a host copy of {width} bytes"
        );
        let (args, args_copy) = title_records(&[[3, 1, 0, 0]]);
        let inputs = rectangle_inputs();
        let mut bound = [(0u64, 0u32, 0u32); 32];
        bound[0] = (vertices.address(), source.len() as u32, STRIDE as u32);
        let list = TitleList::new();
        let mut held = None;
        expand(
            list.host(),
            &mut held,
            &inputs,
            &bound,
            (Records::Title(&args_copy, args.address()), 16, 0..1),
            None,
            None,
            None,
            &mut Default::default(),
        )
        .expect("expanded");
        let left = run_and_read(&list, &mut held, 1, &[(0, 6 * STRIDE)]);
        assert_eq!(left.records[0], [6, 1, 0, 0], "drawn");
        assert_eq!(left.count, 1, "no title count: every record");
        let elements: Vec<(u32, u32)> = inputs.iter().map(|i| (i.offset, i.format)).collect();
        check_expanded(
            &left.vertices.expect("a vertex scratch")[0],
            STRIDE,
            0,
            (&source, STRIDE, &elements),
            (0, 1),
            "the rectangle",
        );
        done(&mut held);
    }

    /// A pipeline without per-vertex inputs (corners from the vertex id):
    /// each live record draws its rectangles' first triangles, the record
    /// as the title wrote it with `3 * rects` vertices (it was `6 * rects`
    /// from vertex 0), no vertex scratch is made (a fresh scratch here, not
    /// a pooled one that may have one), and the record count is the
    /// title's, at most `max`.
    #[test]
    fn records_without_vertex_inputs_keep_their_start() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::tests::records_without_vertex_inputs_keep_their_start";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let (args, args_copy) = title_records(&[
            [6, 2, 9, 1],
            [4, 1, 30, 0],
            [2, 1, 0, 0],
            [3, 0, 4, 0],
            [3, 1, 0, 0],
        ]);
        let (count, count_copy) = title_bytes(256, &9u32.to_le_bytes());
        // Per instance only: the slot is not expanded, nor bounded.
        let inputs = [input(1, 0, 16, true)];
        let list = TitleList::new();
        let mut held = Some(Scratch {
            vertices: None,
            args: None,
            direct: None,
        });
        let prepared = expand(
            list.host(),
            &mut held,
            &inputs,
            &[(0, 0, 0); 32],
            (Records::Title(&args_copy, args.address()), 16, 0..4),
            None,
            Some((&count_copy, count.address())),
            None,
            &mut Default::default(),
        )
        .expect("expanded");
        assert!(prepared.views.is_empty());
        assert!(
            held.as_ref().is_some_and(|s| s.vertices.is_none()),
            "no vertex scratch"
        );
        let left = run_and_read(&list, &mut held, 4, &[]);
        assert!(left.vertices.is_none());
        assert_eq!(
            left.records,
            [[6, 2, 9, 1], [3, 1, 30, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        );
        assert_eq!((left.count, left.counter), (4, 0), "9 at most 4");
        done(&mut held);
    }

    /// Bytes of `n` vertices of one slot of the several-slot test, the
    /// formats of [`several_slot_inputs`]: halves, unorm16 and unorm8 in
    /// slot 0 (10-byte vertices), a half and unorm8 in slot 2 (6 bytes).
    fn slot_vertices(slot: u32, n: usize) -> Vec<u8> {
        const HALVES: [u16; 12] = [
            0x3c00, 0x4000, 0x3800, 0xb800, 0x4200, 0x3555, 0x4b00, 0x2e66, 0xc100, 0x3e00, 0x0400,
            0x5640,
        ];
        let half = |i: usize, j: usize| HALVES[(i * 5 + j * 7) % HALVES.len()].to_le_bytes();
        let mut out = Vec::new();
        for i in 0..n {
            if slot == 0 {
                out.extend(half(i, 0));
                out.extend(half(i, 1));
                out.extend(((i * 9973 + 1234) as u16).to_le_bytes());
                out.extend(((i * 31337 + 4321) as u16).to_le_bytes());
                out.extend([(i * 37 + 11) as u8, (i * 91 + 7) as u8]);
            } else {
                out.extend(half(i, 2));
                out.extend([(i * 53) as u8, (i * 17 + 200) as u8, 255, (i * 3) as u8]);
            }
        }
        out
    }

    /// Slot 0: R16G16_FLOAT at 0, R16G16_UNORM at 4, R8G8B8A8_UNORM at 8
    /// (two components inside 10 bytes); slot 1 per instance; slot 2:
    /// R16_FLOAT at 0, R8G8B8A8_UNORM at 2.
    fn several_slot_inputs() -> Vec<Input> {
        vec![
            input(0, 0, 34, false),
            input(0, 4, 35, false),
            input(0, 8, 28, false),
            input(1, 0, 16, true),
            input(2, 0, 54, false),
            input(2, 2, 28, false),
        ]
    }

    /// Two per-vertex slots of strides 10 and 6 (padded to 12 and 8 in the
    /// scratch), whose views start 2 and 1 bytes past a word, and a per
    /// instance slot between them: each slot is expanded into its own
    /// region, as [`primitive::expand_rects`] makes it.
    #[test]
    fn unaligned_slots_with_padded_strides_expand_as_on_the_cpu() {
        let _pool = POOL_TESTS.lock().unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::tests::unaligned_slots_with_padded_strides_expand_as_on_the_cpu";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let (a, b) = (slot_vertices(0, 6), slot_vertices(2, 6));
        let mut bytes = vec![0u8; 256];
        bytes[2..2 + a.len()].copy_from_slice(&a);
        bytes[97..97 + b.len()].copy_from_slice(&b);
        let (vertices, _copy) = title_bytes(256, &bytes);
        let (args, args_copy) = title_records(&[[3, 1, 3, 0], [6, 2, 0, 1]]);
        let inputs = several_slot_inputs();
        let mut bound = [(0u64, 0u32, 0u32); 32];
        bound[0] = (vertices.address() + 2, a.len() as u32, 10);
        bound[2] = (vertices.address() + 97, b.len() as u32, 6);
        let list = TitleList::new();
        let mut held = None;
        let prepared = expand(
            list.host(),
            &mut held,
            &inputs,
            &bound,
            (Records::Title(&args_copy, args.address()), 16, 0..2),
            None,
            None,
            None,
            &mut Default::default(),
        )
        .expect("expanded");
        let (_, layouts) = scratch_layout(&[(0, 10), (2, 6)], SCRATCH_BYTES).expect("a layout");
        let slots: Vec<(u32, u32)> = prepared
            .views
            .iter()
            .map(|(slot, v)| (*slot, v.StrideInBytes))
            .collect();
        assert_eq!(slots, [(0, 12), (2, 8)]);
        let ranges: Vec<(u64, usize)> = layouts
            .iter()
            .map(|l| (l.dst_offset, 18 * l.dst_stride as usize))
            .collect();
        let left = run_and_read(&list, &mut held, 2, &ranges);
        assert_eq!(left.count, 2, "no title count: every record");
        assert_eq!(left.counter, 18);
        assert_eq!([left.records[0][0], left.records[1][0]], [6, 12]);
        assert_eq!([left.records[0][1], left.records[1][1]], [1, 2]);
        assert_eq!([left.records[0][3], left.records[1][3]], [0, 1]);
        let regions = left.vertices.expect("a vertex scratch");
        let elements = |slot: u32| -> Vec<(u32, u32)> {
            inputs
                .iter()
                .filter(|i| i.slot == slot && !i.per_instance)
                .map(|i| (i.offset, i.format))
                .collect()
        };
        for (region, (source, stride, slot, dst_stride)) in
            regions.iter().zip([(&a, 10, 0, 12), (&b, 6, 2, 8)])
        {
            for (record, from, rects) in [(0, 3, 1), (1, 0, 2)] {
                check_expanded(
                    region,
                    dst_stride,
                    left.records[record][2] as usize,
                    (source, stride, &elements(slot)),
                    (from, rects),
                    &format!("slot {slot}, record {record}"),
                );
            }
        }
        done(&mut held);
    }
}
