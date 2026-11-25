//! Vector memory: buffer loads, stores and atomics (MUBUF, MTBUF) with
//! the GFX10 buffer formats, and global loads (`global_load_*`). Backlog
//! R3 (structured buffers of unknown stride: through the host's runtime
//! stride table, [`Lower::runtime_access`]).

use super::descriptor::{Entry, ODD_STRIDE, SMALL_STRIDE, UNKNOWN_STRIDE};
use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Reg};
use crate::native::{ResourceType, UsedResource};
use crate::resource::{CODE, Source};

/// GFX10 buffer resource control fields (RDNA2 ISA, table 37).
/// SWIZZLE_ENABLE and STRIDE are in word 1, the other fields in word 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BufferDescriptor {
    pub stride: u32,
    pub swizzle_enable: bool,
    pub add_tid_enable: bool,
    pub dst_sel: [u8; 4],
    pub format: u8,
    /// Number of contiguous indices (8, 16, 32 or 64).
    pub index_stride: u32,
    pub oob_select: BufferOob,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferOob {
    Structured,
    Index,
    Nonempty,
    Raw,
}

impl BufferDescriptor {
    pub(super) fn decode(word1: u32, word3: u32) -> Self {
        Self {
            stride: (word1 >> 16) & 0x3fff,
            swizzle_enable: word1 & (1 << 31) != 0,
            add_tid_enable: word3 & (1 << 23) != 0,
            dst_sel: std::array::from_fn(|k| ((word3 >> (3 * k)) & 7) as u8),
            format: ((word3 >> 12) & 0x7f) as u8,
            index_stride: 8 << ((word3 >> 21) & 3),
            oob_select: match (word3 >> 28) & 3 {
                0 => BufferOob::Structured,
                1 => BufferOob::Index,
                2 => BufferOob::Nonempty,
                _ => BufferOob::Raw,
            },
        }
    }
}

/// A raw view with the range information available at the access site.
#[derive(Clone, Copy)]
enum RawView {
    Table(Entry),
    Structured {
        handle: V,
        stride: u32,
    },
    Built {
        handle: V,
        records: u32,
        desc: BufferDescriptor,
    },
}

impl RawView {
    /// Distance physique entre deux dwords logiquement consécutifs.
    fn word_stride(self) -> u32 {
        match self {
            Self::Built { desc, .. } if desc.swizzle_enable => 4 * desc.index_stride,
            _ => 4,
        }
    }
}

/// The scalar offset moves the base. Only the VGPR and instruction
/// offsets (and the component being transferred) are within the element.
#[derive(Clone, Copy)]
struct BufferAddress {
    valid: V,
    index: Option<V>,
    within: V,
    scalar: Option<V>,
    base: u32,
    dynamic_base: Option<V>,
}

/// Plage commune aux composantes du transfert. `remaining` compte les octets
/// logiques avant la fin de l'élément ou du tampon.
#[derive(Clone, Copy)]
struct RawRange {
    byte: V,
    valid: V,
    remaining: V,
    word_stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
pub(super) fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
pub(super) fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

impl<'a> Lower<'a> {
    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    pub(super) fn format_load(
        &mut self,
        h: V,
        off: V,
        format: u8,
        channels: u32,
        word_stride: u32,
    ) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = self.raw_words(h, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else if word_stride == 4 || total == 32 {
            self.raw_words(h, off, total / 32)
        } else {
            (0..total / 32)
                .map(|k| {
                    let delta = self.s.m.i32(k * word_stride);
                    let at = self.s.m.bin(c, Bin::Add, off, delta);
                    self.raw_words(h, at, 1)[0]
                })
                .collect()
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    pub(super) fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.effect_guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        let halfword = matches!(name, "buffer_load_ushort" | "buffer_store_short");
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() || halfword => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let built = match src {
            Source::Built { bits, words, .. } => {
                Some((BufferDescriptor::decode(*bits, words[1]), words[0]))
            }
            _ => None,
        };
        if halfword
            && !(built.is_some()
                && used.is_some_and(|u| matches!(u.kind, ResourceType::Srv | ResourceType::Uav)))
        {
            return Err(name);
        }
        let add_tid = built.is_some_and(|(d, _)| d.add_tid_enable);
        let mut valid = self.s.m.bool(true);
        let index = if add_tid {
            let lane = self.buffer_lane();
            Some(index.map_or(lane, |i| self.raw_add(i, lane, &mut valid)))
        } else {
            index
        };
        let built_format =
            format.is_some() && built.is_some_and(|(d, _)| index.is_some() || d.swizzle_enable);
        let raw = !typed
            && !used.is_some_and(|r| r.kind == ResourceType::Cbv)
            && (halfword
                || built.is_some_and(|(d, _)| {
                    crate::emit::raw_structured()
                        || add_tid
                        || d.swizzle_enable
                        || used.is_some_and(|u| u.kind == ResourceType::Uav)
                })
                || self.through_table(index, used, src)?
                || (index.is_some() && crate::emit::raw_structured()));
        let mut within = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            within = if raw || built_format {
                self.raw_add(within, o, &mut valid)
            } else {
                self.s.m.bin(c, Bin::Add, within, o)
            };
        }
        let scalar = match soffset {
            Operand::Scalar { code: NULL, .. } | Operand::Const(128) => None,
            _ => Some(self.operand(inst, soffset)?),
        };
        let (base, dynamic_base) = self.built_offset(src)?;
        let address = BufferAddress {
            valid,
            index,
            within,
            scalar,
            base,
            dynamic_base,
        };
        let mut off = within;
        if !raw && !built_format {
            if let Some(so) = scalar {
                off = self.s.m.bin(c, Bin::Add, off, so);
            }
            if base != 0 {
                let b = self.s.m.i32(base);
                off = self.s.m.bin(c, Bin::Add, off, b);
            }
            if let Some(x) = dynamic_base {
                off = self.s.m.bin(c, Bin::Add, off, x);
            }
        }
        // Lanes outside EXEC run a load too where nothing branches them away
        // (every lane in the console flow; under a guard in the per-lane
        // form). With a heap index per lane, theirs may be no index at all
        // (past the heap: a fault, the device removed), so they skip the
        // load and keep their registers.
        let lane_guard = guard.filter(|_| self.lane_descriptor(src));
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let xs = self.guarded_load(lane_guard, |l| {
                let h = l.handle(site, src, false, (13, 0x1_0000))?;
                let c = l.cur;
                let byte = match index {
                    Some(i) => {
                        let stride = l.s.m.i32(
                            built
                                .filter(|(d, _)| d.add_tid_enable)
                                .map_or(16, |(d, _)| d.stride),
                        );
                        let rows = l.s.m.bin(c, Bin::Mul, i, stride);
                        l.s.m.bin(c, Bin::Add, rows, off)
                    }
                    None => off,
                };
                let mut xs = Vec::new();
                for k in 0..count {
                    let four = l.s.m.i32(4 * k);
                    let a = l.s.m.bin(c, Bin::Add, byte, four);
                    xs.push(l.cbuffer_dword(h, a));
                }
                Ok(xs)
            })?;
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || used.is_some_and(|u| u.kind == ResourceType::Uav);
        let u = if uav { 0x1000 } else { 0 };
        let iu = self.s.m.undef(self.s.t.i32);
        let raw_view = |l: &mut Self| -> Result<RawView, Why> {
            Ok(match built {
                Some((desc, records)) => {
                    let handle = l.handle(site, src, false, (11 | u, 0))?;
                    RawView::Built {
                        handle,
                        records,
                        desc,
                    }
                }
                _ if l.raw_through_table(src) => {
                    l.note_run_time(src, 11 | u);
                    RawView::Table(l.stride_entry(src)?)
                }
                _ => RawView::Structured {
                    handle: l.handle(site, src, false, (11 | u, 0))?,
                    stride: used.map_or(0, |r| r.stride),
                },
            })
        };
        if typed
            && let Some((desc, _)) = built
            // Pour un TBUFFER sur SRV, le format vient de l'instruction.
            // Le chemin de lecture brut existant le décode déjà.
            && used.is_some_and(|u| u.kind == ResourceType::Uav
                || u.kind == ResourceType::Srv && format.is_none())
        {
            // La vue hôte reste brute. Les canaux de 32 bits conservent
            // leur représentation pour FLOAT, UINT et SINT.
            let format = format.unwrap_or(desc.format);
            let (bits, _) = buffer_format(format).ok_or("a root buffer format")?;
            if d16 || desc.dst_sel != [4, 5, 6, 7] {
                return Err("a packed or swizzled typed root view");
            }
            let Some(d @ Operand::Vector { reg, .. }) = data else {
                return Err(name);
            };
            if store && (count as usize != bits.len() || bits.iter().any(|&n| n != 32)) {
                return Err("a partial or packed typed root store");
            }
            let view = raw_view(self)?;
            let RawView::Built { handle, .. } = view else {
                unreachable!()
            };
            let width = bits.iter().sum::<u32>() / 8;
            let range = self.raw_range(view, address);
            let byte = range.byte;
            let valid = self.raw_fits(range, width);
            let enabled = guard.map_or(valid, |g| self.s.m.bin(self.cur, Bin::And, g, valid));
            let values = if store {
                self.vgprs(inst, d, count as u8)?
            } else {
                Vec::new()
            };
            let xs = self.guarded_values(Some(enabled), |l| {
                if !store {
                    return l.format_load(handle, byte, format, count, range.word_stride);
                }
                for (k, &value) in values.iter().enumerate() {
                    let delta = l.s.m.i32(k as u32 * range.word_stride);
                    let at = l.s.m.bin(l.cur, Bin::Add, byte, delta);
                    l.raw_store_dword(handle, at, value);
                }
                Ok(Vec::new())
            })?;
            for (k, x) in xs.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if halfword {
            let Some(d @ Operand::Vector { reg, .. }) = data else {
                return Err(name);
            };
            let source = if store {
                Some(self.vgprs(inst, d, 1)?[0])
            } else {
                None
            };
            let view = raw_view(self)?;
            let RawView::Built { handle, .. } = view else {
                unreachable!()
            };
            if view.word_stride() != 4 {
                return Err("a swizzled halfword access");
            }
            let range = self.raw_range(view, address);
            let byte = range.byte;
            let valid = self.raw_fits(range, 2);
            let enabled = guard.map_or(valid, |g| self.s.m.bin(self.cur, Bin::And, g, valid));
            let xs =
                self.guarded_values(Some(enabled), |l| Ok(l.raw_halfword(handle, byte, source)))?;
            if !store {
                self.write(inst, Reg::V(*reg), xs[0], guard);
            }
            return Ok(());
        }
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = matches!(inst.fields, Fields::Buffer { cache, .. } if cache.glc);
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let i32t = self.s.t.i32;
            let atomic_op = |l: &mut Self, h: V, coords: [V; 3]| -> V {
                if op == u32::MAX {
                    // cmpswap: data is (new, compare).
                    let mut args = vec![h];
                    args.extend(coords);
                    args.extend([vals[1], vals[0]]);
                    l.s.op(
                        l.cur,
                        "atomicCompareExchange.i32",
                        Attr::NoUnwind,
                        i32t,
                        79,
                        &args,
                    )
                } else {
                    let mut args = vec![h, l.s.m.i32(op)];
                    args.extend(coords);
                    args.push(vals[0]);
                    l.s.op(l.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
                }
            };
            let at = self.begin_guarded(guard);
            // Out of range the console returns 0.
            let done = if raw {
                raw_view(self).and_then(|view| {
                    self.raw_components(view, address, 11 | u, 1, |l, h, byte, _| {
                        Ok(vec![atomic_op(l, h, [byte, iu, iu])])
                    })
                })
            } else {
                let coords = index.map_or([off, iu, iu], |index| [index, off, iu]);
                let props = index.map_or((11 | u, 0), |_| (12 | u, used.map_or(0, |r| r.stride)));
                self.handle(site, src, false, props)
                    .map(|h| vec![atomic_op(self, h, coords)])
            };
            let r = match done {
                Ok(r) => r[0],
                Err(why) => {
                    self.end_guarded(at, &[]);
                    return Err(why);
                }
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    _ if raw => {
                        let at = self.begin_guarded(guard);
                        let done = raw_view(self).and_then(|view| {
                            self.raw_components(view, address, 11 | u, count, |l, h, byte, k| {
                                l.raw_store_dword(h, byte, vals[k]);
                                Ok(Vec::new())
                            })
                        });
                        self.end_guarded(at, &[]);
                        return done.map(|_| ());
                    }
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        args.push(self.handle(site, src, false, (12 | u, stride))?);
                        args.extend([index, off]);
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        // (What can stop the load is checked before its guarded block.)
        let stride = used.map_or(0, |u| u.stride);
        match *format {
            Some(_) if index.is_some() && built.is_none() => {
                return Err("a typed-format load by index");
            }
            Some(_) => {}
            None if typed => {
                if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                    return Err("a typed buffer load at an offset");
                }
                if uav {
                    self.s.flags |= flags::TYPED_UAV_LOADS;
                }
            }
            None => {}
        }
        let xs = self.guarded_load(lane_guard, |l| {
            if let Some(f) = *format {
                let (bits, _) = buffer_format(f).ok_or("a buffer format")?;
                let width = bits.iter().sum::<u32>() / 8;
                if built_format {
                    let view = raw_view(l)?;
                    let range = l.raw_range(view, address);
                    let valid = l.raw_fits(range, width);
                    return l.guarded_values(Some(valid), |l| {
                        let h = l.raw_handle(view, 11 | u);
                        l.format_load(h, range.byte, f, count, range.word_stride)
                    });
                }
                if l.raw_through_table(src) {
                    // The format's bytes (a byte or a half is read in its
                    // dword, which the view's bytes hold whole).
                    return l.runtime_access(src, None, off, width, 11 | u, |l, h, byte| {
                        l.format_load(h, byte, f, count, 4)
                    });
                }
                let h = l.handle(site, src, false, (11 | u, 0))?;
                return l.format_load(h, off, f, count, 4);
            }
            let c = l.cur;
            let r = if typed {
                let h = l.handle(site, src, false, (10 | u, 0x409))?;
                let index = index.unwrap_or(l.s.m.i32(0));
                l.s.op(
                    c,
                    "bufferLoad.f32",
                    Attr::ReadOnly,
                    l.s.t.res_ret,
                    68,
                    &[h, index, iu],
                )
            } else {
                let mask = l.s.m.i8((1u8 << count) - 1);
                let align = l.s.m.i32(4);
                let (h, coords) = match index {
                    _ if raw => {
                        let view = raw_view(l)?;
                        return l.raw_load(view, address, 11 | u, count);
                    }
                    Some(index) => (l.handle(site, src, false, (12 | u, stride))?, [index, off]),
                    None => (l.handle(site, src, false, (11 | u, 0))?, [off, iu]),
                };
                l.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    l.s.t.res_ret,
                    139,
                    &[h, coords[0], coords[1], mask, align],
                )
            };
            let mut xs = Vec::new();
            if d16 {
                // Two f16 channels per VGPR (an odd last high half 0).
                let k16 = l.s.m.i32(16);
                let i32t = l.s.t.i32;
                for k in 0..count.div_ceil(2) {
                    let mut x = None;
                    for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                        let f = l.s.m.extract(c, r, ch);
                        let h = l.s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                        x = Some(match x {
                            Some(hi) => {
                                let hi = l.s.m.bin(c, Bin::Shl, hi, k16);
                                l.s.m.bin(c, Bin::Or, h, hi)
                            }
                            None => h,
                        });
                    }
                    xs.push(x.expect("a channel"));
                }
                return Ok(xs);
            }
            for k in 0..count {
                let x = l.s.m.extract(c, r, k);
                xs.push(l.as_int(x));
            }
            Ok(xs)
        })?;
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Whether an access by element of a buffer whose used-resource entry
    /// is `used` takes its stride from the host's runtime stride table: one
    /// whose stride is not known here (0), on a target that has the table
    /// ([`crate::emit::Target::stride_table`]). A known stride below 4 or
    /// off a multiple of 4, or an unknown one without the table, is refused.
    ///
    /// With `XTP_RAW_STRUCTURED=1` ([`crate::emit::raw_structured`]) a known
    /// stride goes through the table too, on a target that has it: the
    /// host makes structured views raw then, and a structured handle over
    /// a raw view reads zeros on NVIDIA (watch.md item 160); the table's
    /// raw views are raw over raw whatever the slot holds.
    ///
    /// The refusals say which stride they saw (watch.md item 318): none
    /// ([`UNKNOWN_STRIDE`], what the table translates), 1 to 3 bytes
    /// ([`SMALL_STRIDE`]), or another off a multiple of 4 ([`ODD_STRIDE`]).
    fn runtime_stride(&self, used: Option<&UsedResource>) -> Result<bool, Why> {
        match used.map_or(0, |u| u.stride) {
            s if s >= 4 && s.is_multiple_of(4) => {
                Ok(self.x.target.stride_table && crate::emit::raw_structured())
            }
            0 if self.x.target.stride_table => Ok(true),
            0 => Err(UNKNOWN_STRIDE),
            1..=3 => Err(SMALL_STRIDE),
            _ => Err(ODD_STRIDE),
        }
    }

    /// GFX10 ADD_TID_ENABLE uses the lane within the console wave. For
    /// compute, derive it from the same group numbering as the thread ID,
    /// including both halves of a wave64 on a narrower host.
    fn buffer_lane(&mut self) -> V {
        let mask = self.s.m.i32(self.wave() - 1);
        let flat = if self.s.mesh.is_some() {
            self.mesh_thread()
        } else if self.s.threads.is_some() {
            if self.console_lanes {
                self.console_flat()
            } else {
                self.s.op(
                    self.s.entry,
                    "flattenedThreadIdInGroup.i32",
                    Attr::ReadNone,
                    self.s.t.i32,
                    96,
                    &[],
                )
            }
        } else {
            self.s.flags |= flags::WAVE_OPS;
            self.lane_index()
        };
        self.s.m.bin(self.cur, Bin::And, flat, mask)
    }

    fn raw_handle(&mut self, view: RawView, props: u32) -> V {
        match view {
            RawView::Table(e) => {
                self.s
                    .heap_handle(self.cur, e.raw, false, e.nonuniform, (props, 0))
            }
            RawView::Structured { handle, .. } | RawView::Built { handle, .. } => handle,
        }
    }

    /// Une vue D3D du tas renvoie zéro pour chaque dword hors de sa plage.
    /// Le masque supplémentaire reproduit la limite logique de l'élément GFX10.
    fn raw_load(
        &mut self,
        view: RawView,
        address: BufferAddress,
        props: u32,
        count: u32,
    ) -> Result<Vec<V>, Why> {
        let range = self.raw_range(view, address);
        let byte = range.byte;
        let first = self.raw_fits(range, 4);
        // Une entrée de table vide peut désigner un slot de texture. Créer le
        // handle et lire uniquement si le premier dword est accessible.
        self.guarded_values(Some(first), |l| {
            let h = l.raw_handle(view, props);
            if count == 1 {
                return Ok(l.raw_words(h, byte, 1));
            }
            let zero = l.s.m.i32(0);
            let mut valid = vec![l.s.m.bool(true)];
            for n in 2..=count {
                valid.push(l.raw_fits(range, n * 4));
            }
            if view.word_stride() != 4 {
                // Le swizzle entrelace les dwords de plusieurs indices.
                let mut words = Vec::new();
                for (k, &ok) in valid.iter().enumerate() {
                    let delta = l.s.m.i32(k as u32 * view.word_stride());
                    let at = l.s.m.bin(l.cur, Bin::Add, byte, delta);
                    let safe = l.s.m.select(l.cur, ok, at, byte);
                    let word = l.raw_words(h, safe, 1)[0];
                    words.push(l.s.m.select(l.cur, ok, word, zero));
                }
                return Ok(words);
            }
            if matches!(view, RawView::Built { .. }) {
                // Un SRV racine n'a pas de borne D3D. Choisir la largeur
                // exacte du préfixe valide avant la lecture physique.
                let mut words = vec![zero; count as usize];
                for n in (1..=count as usize).rev() {
                    let ok = if n == count as usize {
                        valid[n - 1]
                    } else {
                        let no = l.s.m.bool(false);
                        l.s.m.select(l.cur, valid[n], no, valid[n - 1])
                    };
                    let at = l.begin_guarded(Some(ok));
                    let prefix = l.raw_words(h, byte, n as u32);
                    let pairs: Vec<_> = prefix.into_iter().zip(words.iter().copied()).collect();
                    words[..n].copy_from_slice(&l.end_guarded(at, &pairs));
                }
                return Ok(words);
            }
            let words = l.raw_words(h, byte, count);
            Ok(words
                .into_iter()
                .zip(valid)
                .map(|(w, ok)| l.s.m.select(l.cur, ok, w, zero))
                .collect())
        })
    }

    /// MUBUF checks DWORDX2/3/4 one component at a time (ISA section 8.1.5).
    fn raw_components(
        &mut self,
        view: RawView,
        address: BufferAddress,
        props: u32,
        count: u32,
        mut access: impl FnMut(&mut Self, V, V, usize) -> Result<Vec<V>, Why>,
    ) -> Result<Vec<V>, Why> {
        let range = self.raw_range(view, address);
        let mut out = Vec::new();
        for k in 0..count {
            let valid = self.raw_fits(range, 4 * (k + 1));
            let delta = self.s.m.i32(range.word_stride * k);
            let byte = self.s.m.bin(self.cur, Bin::Add, range.byte, delta);
            let xs = self.guarded_values(Some(valid), |l| {
                let h = l.raw_handle(view, props);
                access(l, h, byte, k as usize)
            })?;
            out.extend(xs);
        }
        Ok(out)
    }

    /// Conserve la retenue dans le prédicat pour qu'une adresse au-delà
    /// de 4 Gio ne revienne jamais au début de la vue D3D.
    pub(super) fn raw_add(&mut self, a: V, b: V, valid: &mut V) -> V {
        let zero = self.s.m.i32(0);
        if a == zero {
            return b;
        }
        if b == zero {
            return a;
        }
        let sum = self.s.m.bin(self.cur, Bin::Add, a, b);
        let fits = self.s.m.cmp(self.cur, pred::IUGE, sum, a);
        *valid = self.s.m.bin(self.cur, Bin::And, *valid, fits);
        sum
    }

    fn raw_scale(&mut self, value: V, factor: u32, valid: &mut V) -> V {
        let limit = self
            .s
            .m
            .i32(u32::MAX.checked_div(factor).unwrap_or(u32::MAX));
        let fits = self.s.m.cmp(self.cur, pred::IULE, value, limit);
        *valid = self.s.m.bin(self.cur, Bin::And, *valid, fits);
        let factor = self.s.m.i32(factor);
        self.s.m.bin(self.cur, Bin::Mul, value, factor)
    }

    fn raw_fits(&mut self, r: RawRange, width: u32) -> V {
        let span = (width - 1) / 4 * r.word_stride + (width - 1) % 4;
        let last = self.s.m.i32(u32::MAX - span);
        let fits = self.s.m.cmp(self.cur, pred::IULE, r.byte, last);
        let width = self.s.m.i32(width);
        let inside = self.s.m.cmp(self.cur, pred::IUGE, r.remaining, width);
        let fits = self.s.m.bin(self.cur, Bin::And, fits, inside);
        self.s.m.bin(self.cur, Bin::And, r.valid, fits)
    }

    /// RDNA2 §8.1.5. Chaque groupe de INDEX_STRIDE indices entrelace ses
    /// dwords. Les octets sont contigus à l'intérieur de chaque dword.
    fn swizzled_offset(&mut self, index: V, within: V, desc: BufferDescriptor, valid: &mut V) -> V {
        let c = self.cur;
        let mask = self.s.m.i32(desc.index_stride - 1);
        let group_mask = self.s.m.i32(!(desc.index_stride - 1));
        let group = self.s.m.bin(c, Bin::And, index, group_mask);
        let lane = self.s.m.bin(c, Bin::And, index, mask);
        let four = self.s.m.i32(4);
        let lane = self.s.m.bin(c, Bin::Mul, lane, four);
        let not3 = self.s.m.i32(!3);
        let three = self.s.m.i32(3);
        let row = self.s.m.bin(c, Bin::And, within, not3);
        let sub = self.s.m.bin(c, Bin::And, within, three);
        let row = self.raw_scale(row, desc.index_stride, valid);
        let group = self.raw_scale(group, desc.stride, valid);
        let start = self.raw_add(group, row, valid);
        let in_row = self.s.m.bin(c, Bin::Add, lane, sub);
        self.raw_add(start, in_row, valid)
    }

    fn raw_range(&mut self, view: RawView, a: BufferAddress) -> RawRange {
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let max = self.s.m.i32(u32::MAX);
        let index = a.index.unwrap_or(zero);
        let mut valid = a.valid;
        let mut checked_offset = a.within;
        let (mut byte, limit) = match view {
            RawView::Table(e) => {
                let raw = self.s.m.cmp(c, pred::EQ, e.stride, zero);
                if let Some(so) = a.scalar {
                    let so = self.s.m.select(c, raw, so, zero);
                    checked_offset = self.raw_add(a.within, so, &mut valid);
                }
                let four = self.s.m.i32(4);
                let (at, limit) = if a.index.is_some() {
                    let in_count = self.s.m.cmp(c, pred::IULT, index, e.count);
                    let indexed = self.s.m.bin(c, Bin::Or, raw, in_count);
                    valid = self.s.m.bin(c, Bin::And, valid, indexed);
                    // Le nombre d'éléments de la table est borné par la vue
                    // D3D. Pour un index valide, ce produit tient sur 32 bits.
                    let at = self.s.m.bin(c, Bin::Mul, index, e.stride);
                    let bytes = self.s.m.bin(c, Bin::Mul, e.count, four);
                    (at, self.s.m.select(c, raw, bytes, e.stride))
                } else {
                    let element = self.s.m.select(c, raw, four, e.stride);
                    (zero, self.s.m.bin(c, Bin::Mul, e.count, element))
                };
                let byte = self.raw_add(at, a.within, &mut valid);
                (self.raw_add(byte, e.bias, &mut valid), Some(limit))
            }
            RawView::Structured { stride, .. } => {
                let at = self.raw_scale(index, stride, &mut valid);
                let byte = self.raw_add(at, a.within, &mut valid);
                (byte, Some(self.s.m.i32(stride)))
            }
            RawView::Built { records, desc, .. } => {
                let byte = if desc.swizzle_enable {
                    self.swizzled_offset(index, a.within, desc, &mut valid)
                } else {
                    let at = self.raw_scale(index, desc.stride, &mut valid);
                    self.raw_add(at, a.within, &mut valid)
                };
                let mode =
                    if desc.oob_select == BufferOob::Raw && desc.swizzle_enable && desc.stride != 0
                    {
                        BufferOob::Structured
                    } else {
                        desc.oob_select
                    };
                let count = self.s.m.i32(records);
                if matches!(mode, BufferOob::Index | BufferOob::Structured) {
                    let in_count = self.s.m.cmp(c, pred::IULT, index, count);
                    valid = self.s.m.bin(c, Bin::And, valid, in_count);
                }
                let limit = match mode {
                    BufferOob::Structured => Some(self.s.m.i32(desc.stride)),
                    BufferOob::Index => None,
                    BufferOob::Nonempty if records == 0 => Some(zero),
                    BufferOob::Nonempty => None,
                    BufferOob::Raw => {
                        // Mesures sur gfx1036 et gfx1011 (VIctory #2084) : la borne
                        // inclut SOFFSET. Index × stride déplace l'adresse.
                        if let Some(so) = a.scalar {
                            checked_offset = self.raw_add(a.within, so, &mut valid);
                        }
                        Some(count)
                    }
                };
                (byte, limit)
            }
        };
        let remaining = limit.map_or(max, |size| {
            let inside = self.s.m.cmp(c, pred::IULE, checked_offset, size);
            let left = self.s.m.bin(c, Bin::Sub, size, checked_offset);
            self.s.m.select(c, inside, left, zero)
        });
        let base = self.s.m.i32(a.base);
        for offset in [Some(base), a.dynamic_base, a.scalar].into_iter().flatten() {
            byte = self.raw_add(byte, offset, &mut valid);
        }
        RawRange {
            byte,
            valid,
            remaining,
            word_stride: view.word_stride(),
        }
    }

    /// Demi-mot du constructeur, y compris à cheval sur deux dwords.
    /// D'après Marvisak/VIctory#1572 (head 0c16661) by iurysf et
    /// fxceless's VIctory fork (4e17478) pour les écritures par masque.
    fn raw_halfword(&mut self, h: V, byte: V, source: Option<V>) -> Vec<V> {
        let c = self.cur;
        let mask = self.s.m.i32(0xff);
        let three = self.s.m.i32(3);
        let aligned = self.s.m.i32(!3);
        let mut result = self.s.m.i32(0);
        for k in 0..2 {
            let delta = self.s.m.i32(k);
            let addr = self.s.m.bin(c, Bin::Add, byte, delta);
            let word = self.s.m.bin(c, Bin::And, addr, aligned);
            let within = self.s.m.bin(c, Bin::And, addr, three);
            let shift = self.s.m.bin(c, Bin::Shl, within, three);
            let field = self.s.m.i32(k * 8);
            if let Some(value) = source {
                let bits = self.s.m.bin(c, Bin::LShr, value, field);
                let bits = self.s.m.bin(c, Bin::And, bits, mask);
                let set = self.s.m.bin(c, Bin::Shl, bits, shift);
                let changed = self.s.m.bin(c, Bin::Shl, mask, shift);
                let ones = self.s.m.i32(u32::MAX);
                let keep = self.s.m.bin(c, Bin::Xor, changed, ones);
                let u = self.s.m.undef(self.s.t.i32);
                for (op, bits) in [(1, keep), (2, set)] {
                    let op = self.s.m.i32(op);
                    self.s.op(
                        c,
                        "atomicBinOp.i32",
                        Attr::NoUnwind,
                        self.s.t.i32,
                        78,
                        &[h, op, word, u, u, bits],
                    );
                }
            } else {
                let bits = self.raw_words(h, word, 1)[0];
                let bits = self.s.m.bin(c, Bin::LShr, bits, shift);
                let bits = self.s.m.bin(c, Bin::And, bits, mask);
                let bits = self.s.m.bin(c, Bin::Shl, bits, field);
                result = self.s.m.bin(c, Bin::Or, result, bits);
            }
        }
        if source.is_some() {
            Vec::new()
        } else {
            vec![result]
        }
    }

    fn raw_store_dword(&mut self, h: V, byte: V, value: V) {
        let undef = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(1);
        let align = self.s.m.i32(4);
        self.s.op(
            self.cur,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            self.s.t.void,
            140,
            &[h, byte, undef, value, undef, undef, undef, mask, align],
        );
    }

    pub(super) fn constant_read_handle(
        &mut self,
        read: crate::resource::ConstantRead,
        scalar: bool,
    ) -> V {
        let kind = match (read.packed, scalar) {
            (true, _) => crate::emit::PACKED_ADDRESS_SRV,
            (false, true) => crate::emit::SCALAR_ADDRESS_SRV,
            (false, false) => crate::emit::VECTOR_ADDRESS_SRV,
        };
        let index = self.host_word(kind, read.slot, read.dword);
        self.s.heap_handle(self.cur, index, false, false, (11, 0))
    }

    pub(super) fn raw_words(&mut self, h: V, byte: V, count: u32) -> Vec<V> {
        let undef = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8((1 << count) - 1);
        let align = self.s.m.i32(4);
        let r = self.s.op(
            self.cur,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[h, byte, undef, mask, align],
        );
        (0..count)
            .map(|k| {
                let x = self.s.m.extract(self.cur, r, k);
                self.as_int(x)
            })
            .collect()
    }

    /// Whether a raw access of `src`'s buffer (an untyped load, store or
    /// atomic, a format load) goes through the runtime stride table: by
    /// element (`index`) when its stride is not known here
    /// ([`Self::runtime_stride`]), by offset alone for a view-heap
    /// descriptor on a target with the table ([`Self::raw_through_table`]).
    fn through_table(
        &self,
        index: Option<V>,
        used: Option<&UsedResource>,
        src: &Source,
    ) -> Result<bool, Why> {
        match index {
            Some(_) => self.runtime_stride(used),
            None => Ok(self.raw_through_table(src)),
        }
    }

    /// Runs `access` (with the raw handle and the byte in its view) on the
    /// view the runtime stride table names for `src`'s slot, in the lanes
    /// where an access of `width` bytes at byte `off` (of element `index`,
    /// or by offset alone) passes the table's range checks
    /// ([`Self::raw_range`]); what it gives is 0 in the others. `props`
    /// are the raw handle's (11, with the UAV bit for unordered access). The
    /// caller's guard, if any, is around it: nothing is read for a lane
    /// that must not access.
    pub(super) fn runtime_access(
        &mut self,
        src: &Source,
        index: Option<V>,
        off: V,
        width: u32,
        props: u32,
        access: impl FnOnce(&mut Self, V, V) -> Result<Vec<V>, Why>,
    ) -> Result<Vec<V>, Why> {
        let e = self.stride_entry(src)?;
        let address = BufferAddress {
            valid: self.s.m.bool(true),
            index,
            within: off,
            scalar: None,
            base: 0,
            dynamic_base: None,
        };
        let range = self.raw_range(RawView::Table(e), address);
        let valid = self.raw_fits(range, width);
        self.note_run_time(src, props);
        self.guarded_values(Some(valid), |l| {
            let h =
                l.s.heap_handle(l.cur, e.raw, false, e.nonuniform, (props, 0));
            access(l, h, range.byte)
        })
    }

    /// The dwords `load` gives, computed only in lanes where `guard` holds
    /// (none: all); 0 in the others, whose registers the write's own guard
    /// keeps. `load` runs inside the guarded block: a failure there still
    /// closes it.
    pub(super) fn guarded_load(
        &mut self,
        guard: Option<V>,
        load: impl FnOnce(&mut Self) -> Result<Vec<V>, Why>,
    ) -> Result<Vec<V>, Why> {
        #[cfg(test)]
        if guard.is_some() {
            GUARDED_LOADS.with(|n| n.set(n.get() + 1));
        }
        self.guarded_values(guard, load)
    }

    /// Return zero for each value in lanes where the operation was skipped.
    fn guarded_values(
        &mut self,
        guard: Option<V>,
        load: impl FnOnce(&mut Self) -> Result<Vec<V>, Why>,
    ) -> Result<Vec<V>, Why> {
        let at = self.begin_guarded(guard);
        match load(self) {
            Ok(xs) => {
                let zero = self.s.m.i32(0);
                let pairs: Vec<(V, V)> = xs.iter().map(|&x| (x, zero)).collect();
                Ok(self.end_guarded(at, &pairs))
            }
            Err(why) => {
                self.end_guarded(at, &[]);
                Err(why)
            }
        }
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    pub(super) fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let guard = self.effect_guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        if let Some(&read) = self.x.res.constant_reads.get(&(b, i)) {
            let h = self.constant_read_handle(read, false);
            for k in 0..n {
                let at = self.s.m.i32(read.offset + 4 * k);
                let x = self.raw_words(h, at, 1)[0];
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let c = self.cur;
        let src = Source::Built {
            address: a,
            bits: 0,
            words: [0, 0],
        };
        let (base, extra) = self.built_offset(&src)?;
        let mut off = self.s.m.i32(base);
        if let Some(d) = extra {
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let h = if a.param == CODE {
            None
        } else {
            let uav = self
                .used_kind(&src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
            Some(self.handle((b, i), &src, false, (if uav { 11 | 0x1000 } else { 11 }, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// Loads [`Lower::guarded_load`] put under an EXEC guard on this thread.
    static GUARDED_LOADS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::super::{Parts, lower};
    use super::GUARDED_LOADS;
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    /// Lowers a wave32 compute program of 32 threads with a table of
    /// structured dword buffers at s0 (a raw UAV at dword 16) and 512
    /// bytes of LDS: whether it uses a non-uniform handle, and how many
    /// loads the lowering put under a guard (summed over its attempts).
    fn guarded_loads(words: &[u32]) -> (u32, u32) {
        let range = |kind, table_offset_dwords, stride| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 0,
            slot: 0,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            float_mode: 0,
            vgprs: 32,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![
                range(ResourceType::Srv, 0, 4),
                range(ResourceType::Uav, 16, 0),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 1 << 1,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, false, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        GUARDED_LOADS.with(|n| n.set(0));
        let l = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap();
        (
            l.nonuniform_handles,
            GUARDED_LOADS.with(std::cell::Cell::get),
        )
    }

    /// A waterfall loop's buffer load reads each lane's own descriptor; a
    /// lane outside EXEC (from 24 on, or one whose descriptor an earlier
    /// round took) may hold no descriptor to load through, so the load is
    /// guarded. A load through a descriptor of the whole wave under the
    /// same kind of EXEC keeps no guard of its own (its write still has
    /// one). (No device needed: `waterfall_tests` runs the first program.)
    #[test]
    fn loads_through_a_lanes_own_descriptor_skip_lanes_outside_exec() {
        use crate::resource::tests::WATERFALL_JOINED_BUFFER_WAVE32;
        let (nonuniform, guarded) = guarded_loads(WATERFALL_JOINED_BUFFER_WAVE32);
        assert_eq!(nonuniform, 1);
        assert!(guarded > 0);
        let uniform = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(guarded_loads(&uniform), (0, 0));
    }
}

#[cfg(test)]
mod descriptor_tests {
    use super::{BufferDescriptor, BufferOob};

    #[test]
    fn raw_buffer_descriptor_fields_use_their_isa_bits() {
        for (k, oob_select) in [
            BufferOob::Structured,
            BufferOob::Index,
            BufferOob::Nonempty,
            BufferOob::Raw,
        ]
        .into_iter()
        .enumerate()
        {
            let k = k as u32;
            let word3 = (k << 28) | (k << 21) | (0x52 << 12) | 0xfac;
            assert_eq!(
                BufferDescriptor::decode(0xc00c_ffff, word3 | 0xc980_0000),
                BufferDescriptor {
                    stride: 12,
                    swizzle_enable: true,
                    add_tid_enable: true,
                    dst_sel: [4, 5, 6, 7],
                    format: 0x52,
                    index_stride: 8 << k,
                    oob_select,
                }
            );
        }
        assert!(!BufferDescriptor::decode(1 << 30, u32::MAX).swizzle_enable);
        assert!(!BufferDescriptor::decode(u32::MAX, !(1 << 23)).add_tid_enable);
    }
}
