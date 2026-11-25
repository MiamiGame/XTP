//! LDS (`ds_*`): group-shared memory, swizzles and permutes. Backlog S3
//! (byte and half accesses), G2 (the ESGS ring).

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Rmw, V, pred};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg};

/// A byte or half LDS access (`ds_read_{u,i}{8,16}`, `ds_write_b{8,16}`
/// and their `_d16`, `_d16_hi` forms), as [`Lower::ds`] decoded it.
struct Narrow<'n> {
    /// The name without `_d16` or `_d16_hi`.
    base: &'n str,
    /// 8 or 16.
    bits: u32,
    /// The instruction's byte offset.
    offset: u32,
    dst: Option<u8>,
    guard: Option<V>,
}

impl<'a> Lower<'a> {
    /// A byte or half LDS access. LDS is an array of dwords here: the
    /// access works on the dword holding it (`p`, for the address `a` plus
    /// the offset), at bit 8 × (address & 3). A half at byte 3 straddles
    /// into the next dword, as RDNA2 accesses an unaligned half (unaligned
    /// LDS mode; a tester's readback on hardware, VIctory #571); bytes never
    /// do. Past the last LDS dword that part reads 0 and is not written, as
    /// out-of-bounds LDS accesses. `_d16` reads fill the register's low half
    /// (`_d16_hi` its high half) and keep the other; `_d16_hi` writes take
    /// the data's high half.
    // shortcut: two lanes writing different values to the same byte or
    // half in one instruction leave the bits of both (each write is a clear
    // then a set); the console keeps one lane's.
    fn ds_narrow(&mut self, inst: &Inst, n: &Narrow, a: V, p: V) -> Result<(), Why> {
        let name = inst.name();
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let o = k(self, n.offset);
        let at = self.s.m.bin(c, Bin::Add, a, o);
        let three = k(self, 3);
        let byte = self.s.m.bin(c, Bin::And, at, three);
        let sh = self.s.m.bin(c, Bin::Shl, byte, three);
        let mask = k(self, (1u32 << n.bits) - 1);
        // A half at byte 3: the next dword (clamped to the last one) and
        // whether it lies past the end of LDS.
        let straddle = if n.bits == 16 {
            let two = k(self, 2);
            let index = self.s.m.bin(c, Bin::LShr, at, two);
            let one = k(self, 1);
            let next = self.s.m.bin(c, Bin::Add, index, one);
            let last = k(self, self.s.lds_dwords - 1);
            let past = self.s.m.cmp(c, pred::IUGT, next, last);
            let next = self.s.m.select(c, past, last, next);
            let lds = self.s.lds();
            let q = self.s.m.element(c, lds, next);
            let at3 = self.s.m.cmp(c, pred::EQ, byte, three);
            Some((q, at3, past))
        } else {
            None
        };
        if name.starts_with("ds_read") {
            let reg = n.dst.ok_or("an LDS read result")?;
            let w = self.s.m.load(c, p);
            let mut x = self.s.m.bin(c, Bin::LShr, w, sh);
            if let Some((q, at3, past)) = straddle {
                // At byte 3: the dword's top byte, then the next one's low
                // byte (0 past the end).
                let hi = self.s.m.load(c, q);
                let zero = k(self, 0);
                let hi = self.s.m.select(c, past, zero, hi);
                let k8 = k(self, 8);
                let hi = self.s.m.bin(c, Bin::Shl, hi, k8);
                let both = self.s.m.bin(c, Bin::Or, x, hi);
                x = self.s.m.select(c, at3, both, x);
            }
            let x = if n.base.contains("_i") {
                let s = k(self, 32 - n.bits);
                let x = self.s.m.bin(c, Bin::Shl, x, s);
                self.s.m.bin(c, Bin::AShr, x, s)
            } else {
                self.s.m.bin(c, Bin::And, x, mask)
            };
            let x = if n.base == name {
                x
            } else {
                let old = match inst.read(Reg::V(reg)) {
                    Some(v) => self.i32_of(v)?,
                    None => k(self, 0),
                };
                let half = k(self, 0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, half);
                let (x, keep) = if name.ends_with("_hi") {
                    let k16 = k(self, 16);
                    (self.s.m.bin(c, Bin::Shl, x, k16), half)
                } else {
                    (x, k(self, 0xFFFF_0000))
                };
                let old = self.s.m.bin(c, Bin::And, old, keep);
                self.s.m.bin(c, Bin::Or, old, x)
            };
            self.write(inst, Reg::V(reg), x, n.guard);
            return Ok(());
        }
        let Fields::Ds { data, .. } = &inst.fields else {
            unreachable!()
        };
        let mut x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
        if name != n.base {
            // The data's high half.
            let k16 = k(self, 16);
            x = self.s.m.bin(c, Bin::LShr, x, k16);
        }
        let x = self.s.m.bin(c, Bin::And, x, mask);
        let ones = k(self, u32::MAX);
        // In the dword: clear the bits, then set them, so that lanes writing
        // other bytes of it keep theirs (32-bit shifts drop what straddles).
        let set = self.s.m.bin(c, Bin::Shl, x, sh);
        let m = self.s.m.bin(c, Bin::Shl, mask, sh);
        let keep = self.s.m.bin(c, Bin::Xor, m, ones);
        let guarded = self.begin_guarded(n.guard);
        self.s.m.atomic(self.cur, Rmw::And, p, keep);
        self.s.m.atomic(self.cur, Rmw::Or, p, set);
        if let Some((q, at3, past)) = straddle {
            // The high byte of a half at byte 3 goes to the next dword's
            // low byte, in the lanes whose half straddles (in bounds).
            let t = self.s.m.bool(true);
            let inside = self.s.m.bin(self.cur, Bin::Xor, past, t);
            let both = self.s.m.bin(self.cur, Bin::And, at3, inside);
            let inner = self.begin_guarded(Some(both));
            let k8 = k(self, 8);
            let hi = self.s.m.bin(self.cur, Bin::LShr, x, k8);
            let keep = k(self, !0xFF);
            self.s.m.atomic(self.cur, Rmw::And, q, keep);
            self.s.m.atomic(self.cur, Rmw::Or, q, hi);
            self.end_guarded(inner, &[]);
        }
        self.end_guarded(guarded, &[]);
        Ok(())
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    pub(super) fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        let name = inst.name();
        let guard = self.effect_guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_bpermute_b32" {
            // RDNA2 gathers separately within each 32-lane half. Disabled
            // source lanes supply zero and disabled destinations are preserved.
            self.cross_lane(false)?;
            let address =
                self.vgprs(inst, addr.as_ref().ok_or("a backward permute address")?, 1)?[0];
            let source = self.vgprs(inst, data.first().ok_or("a backward permute source")?, 1)?[0];
            let offset = self.s.m.i32(u32::from(*offset0));
            let byte_address = self.s.m.bin(c, Bin::Add, address, offset);
            let two = self.s.m.i32(2);
            let word = self.s.m.bin(c, Bin::LShr, byte_address, two);
            let mask = self.s.m.i32(31);
            let within_half = self.s.m.bin(c, Bin::And, word, mask);
            let lane = self.s.op(
                c,
                "waveGetLaneIndex",
                Attr::ReadOnly,
                self.s.t.i32,
                111,
                &[],
            );
            let high_mask = self.s.m.i32(!31);
            let base = self.s.m.bin(c, Bin::And, lane, high_mask);
            let index = self.s.m.bin(c, Bin::Or, base, within_half);

            let active = guard.unwrap_or_else(|| self.s.m.bool(true));
            let ty = self
                .s
                .m
                .named_struct("dx.types.fouri32", vec![self.s.t.i32; 4]);
            let ballot = self
                .s
                .op(c, "waveActiveBallot", Attr::NoUnwind, ty, 116, &[active]);
            let lo = self.s.m.extract(c, ballot, 0);
            let hi = self.s.m.extract(c, ballot, 1);
            let zero = self.s.m.i32(0);
            let lower_half = self.s.m.cmp(c, pred::EQ, base, zero);
            let bits = self.s.m.select(c, lower_half, lo, hi);
            let shifted = self.s.m.bin(c, Bin::LShr, bits, within_half);
            let one = self.s.m.i32(1);
            let bit = self.s.m.bin(c, Bin::And, shifted, one);
            let enabled = self.s.m.cmp(c, pred::NE, bit, zero);
            let gathered = self.read_lane(source, index);
            let value = self.s.m.select(c, enabled, gathered, zero);
            self.write(
                inst,
                Reg::V(dst_reg.ok_or("a backward permute result")?),
                value,
                guard,
            );
            return Ok(());
        }
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        if self.console_cf
            && self.split
            && self
                .s
                .threads
                .is_some_and(|t| t.iter().product::<u32>() > 32)
        {
            // The console runs an instruction in all 64 lanes before the
            // next one. The halves are two host waves, both here (console
            // flow): a group barrier before each LDS access puts it after
            // every earlier one, whichever half made it.
            // Paired lanes run in one host wave. Fence their shared
            // accesses without synchronising independent console waves.
            let sync = u32::from(!self.s.m.paired_lanes());
            let (mode, void) = (self.s.m.i32(sync | 8), self.s.t.void);
            self.s
                .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let bits = match base {
            "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(8u32),
            "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(16),
            _ => None,
        }
        .filter(|_| {
            base == name
                || name.ends_with("_d16") && base.starts_with("ds_read")
                || name.ends_with("_d16_hi")
        });
        if let Some(bits) = bits {
            let narrow = Narrow {
                base,
                bits,
                offset,
                dst: dst_reg,
                guard,
            };
            let p = dword(self, offset);
            return self.ds_narrow(inst, &narrow, a, p);
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use crate::emit::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let old_reg = dst_reg.and_then(|r| inst.read(Reg::V(r)).map(|v| (r, v)));
        let outside = match old_reg {
            Some((_, v)) => self.i32_of(v).unwrap_or(self.s.m.i32(0)),
            None => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }
}

/// Byte and half LDS accesses: hand-assembled GFX10 compute programs (our
/// own code), translated, validated where `dxil.dll` is found and run on
/// the test GPU (WARP here), against an LDS of bytes worked out on the CPU.
#[cfg(all(test, windows))]
mod tests {
    use crate::dxbc::{self, Container};
    use crate::emit::host::{Dispatch, Host, Root, Stride, View};
    use crate::emit::validator::Validator;
    use crate::emit::{HostShader, Target, emit_for};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    /// Two console waves (a group of one would declare its wave size,
    /// which WARP does not run).
    const LANES: usize = 64;
    /// LDS bytes the programs allocate (`LDS_SIZE` 2): 16 per lane.
    const LDS: usize = 1024;
    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_WAITCNT_0: u32 = 0xBF8C_0000;
    const S_BARRIER: u32 = 0xBF8A_0000;
    const ZERO: u32 = 0x80;

    fn s_mov(d: u8, src: u32) -> u32 {
        0xBE80_0300 | u32::from(d) << 16 | src
    }

    /// `s_load_dwordx4 s[d:d+3], s[base:base+1], 0`.
    fn s_load4(d: u8, base: u8) -> [u32; 2] {
        [
            0xF400_0000 | 2 << 18 | u32::from(d) << 6 | u32::from(base / 2),
            0xFA00_0000,
        ]
    }

    fn v_mov(d: u8, src: u32) -> u32 {
        0x7E00_0000 | u32::from(d) << 17 | 1 << 9 | src
    }

    /// `v_lshlrev_b32 vd, by, vs`.
    fn v_lshl(d: u8, by: u32, v: u8) -> u32 {
        0x1A << 25 | u32::from(d) << 17 | u32::from(v) << 9 | (0x80 + by)
    }

    /// MUBUF `op` with offen: v[data..] at s[rsrc..]'s base + v[addr] + offset.
    fn mubuf(op: u32, data: u8, addr: u8, rsrc: u8, offset: u32) -> [u32; 2] {
        [
            0xE000_0000 | op << 18 | 1 << 12 | offset,
            u32::from(addr) | u32::from(data) << 8 | u32::from(rsrc / 4) << 16 | ZERO << 24,
        ]
    }

    /// DS `op` with its two offsets, address, data and destination VGPRs.
    fn ds(op: u32, offsets: [u8; 2], addr: u8, data: [u8; 2], dst: u8) -> [u32; 2] {
        [
            0xD800_0000 | op << 18 | u32::from(offsets[1]) << 8 | u32::from(offsets[0]),
            u32::from(addr)
                | u32::from(data[0]) << 8
                | u32::from(data[1]) << 16
                | u32::from(dst) << 24,
        ]
    }

    /// The DS opcode the decoder names `name`.
    fn op(name: &str) -> u32 {
        (0..256)
            .find(|&op| {
                let code: Vec<u8> = ds(op, [0, 0], 0, [0, 0], 0)
                    .iter()
                    .chain(&[S_ENDPGM])
                    .flat_map(|w| w.to_le_bytes())
                    .collect();
                crate::gfx10::decode(&code).is_ok_and(|d| d[0].name() == Some(name))
            })
            .unwrap_or_else(|| panic!("{name}"))
    }

    /// Each lane loads (init, address, data, old) into v2..v5, fills LDS
    /// dwords 4 × lane..+3 with init, then (after a barrier) runs `write`
    /// of v4 at address v3, and (after another) `read` at v3 into v7, which
    /// first holds old. It stores v7 and its four LDS dwords at 32 × lane.
    fn program(write: Option<&str>, read: Option<&str>) -> Vec<u32> {
        let mut w = vec![s_mov(4, 0), s_mov(5, ZERO), s_mov(6, 1), s_mov(7, ZERO)];
        w.extend(s_load4(8, 4));
        w.extend(s_load4(12, 6));
        w.push(S_WAITCNT_0);
        w.push(v_lshl(1, 4, 0));
        w.extend(mubuf(0x0E, 2, 1, 8, 0));
        w.push(S_WAITCNT_0);
        w.push(v_lshl(6, 4, 0));
        let write2 = op("ds_write2_b32");
        w.extend(ds(write2, [0, 1], 6, [2, 2], 0));
        w.extend(ds(write2, [2, 3], 6, [2, 2], 0));
        w.extend([S_WAITCNT_0, S_BARRIER]);
        if let Some(name) = write {
            w.extend(ds(op(name), [0, 0], 3, [4, 0], 0));
        }
        w.extend([S_WAITCNT_0, S_BARRIER]);
        w.push(v_mov(7, 0x100 + 5));
        if let Some(name) = read {
            w.extend(ds(op(name), [0, 0], 3, [0, 0], 7));
        }
        w.extend(ds(op("ds_read_b128"), [0, 0], 6, [0, 0], 8));
        w.push(S_WAITCNT_0);
        w.extend([v_mov(12, ZERO), v_mov(13, ZERO), v_mov(14, ZERO)]);
        w.push(v_lshl(15, 5, 0));
        w.extend(mubuf(0x1E, 7, 15, 12, 0));
        w.extend(mubuf(0x1E, 11, 15, 12, 16));
        w.extend([S_WAITCNT_0, S_ENDPGM]);
        w
    }

    /// `words` translated for `target`; the translation for any host is
    /// validated where `dxil.dll` is found.
    fn translate(words: &[u32], target: Target) -> Result<HostShader, String> {
        let table = |user_data, slot| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot,
        };
        let used = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data,
            slot,
        };
        let n = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            float_mode: 0,
            vgprs: 16,
            layout: vec![table(0, 1), table(1, 2)],
            used: vec![used(ResourceType::Srv, 0, 1), used(ResourceType::Uav, 1, 2)],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 15 | 2 << 1,
                threads: [LANES as u32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&n.code, n.wave64, n.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &n.layout, n.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &n).unwrap();
        let empty = dxbc::build(&[]);
        let c = Container::parse(&empty).unwrap();
        let any = emit_for(&c, &n, &p, &st, &io, &res, Target::ANY).unwrap();
        if let Ok(mut v) = Validator::load(Validator::default_path().as_deref()) {
            v.validate(&mut any.dxil.clone())
                .unwrap_or_else(|e| panic!("validation: {e}"));
        }
        emit_for(&c, &n, &p, &st, &io, &res, target).map_err(|e| e.to_string())
    }

    /// A lane's inputs: its address, data and the old value of the read's
    /// destination (each lane's init is [`init`]).
    #[derive(Clone, Copy)]
    struct Lane {
        addr: u32,
        data: u32,
        old: u32,
    }

    /// What lane `l` fills its LDS dwords with: every byte value, signs
    /// set and clear.
    fn init(l: usize) -> u32 {
        0x9E37_79B9u32.wrapping_mul(l as u32 + 1) ^ 0x8040_2010
    }

    /// The CPU's answer: each lane's read result, then its LDS dwords.
    fn expected(write: Option<&str>, read: Option<&str>, lanes: &[Lane]) -> Vec<[u32; 5]> {
        let mut lds = [0u8; LDS];
        for (l, d) in lds.as_chunks_mut::<16>().0.iter_mut().enumerate() {
            for w in d.as_chunks_mut::<4>().0 {
                *w = init(l).to_le_bytes();
            }
        }
        if let Some(name) = write {
            for l in lanes {
                let data = if name.ends_with("_d16_hi") {
                    l.data >> 16
                } else {
                    l.data
                };
                let n = if name.starts_with("ds_write_b8") {
                    1
                } else {
                    2
                };
                for k in 0..n {
                    // Past the end: dropped.
                    if let Some(b) = lds.get_mut(l.addr as usize + k) {
                        *b = (data >> (8 * k)) as u8;
                    }
                }
            }
        }
        // Past the end: 0.
        let byte = |a: usize| u32::from(lds.get(a).copied().unwrap_or(0));
        lanes
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let a = l.addr as usize;
                let x = match read {
                    None => l.old,
                    Some(name) => {
                        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
                        let half = byte(a) | byte(a + 1) << 8;
                        let x = match base {
                            "ds_read_u8" => byte(a),
                            "ds_read_i8" => byte(a) as u8 as i8 as i32 as u32,
                            "ds_read_u16" => half,
                            "ds_read_i16" => half as u16 as i16 as i32 as u32,
                            _ => unreachable!("{name}"),
                        };
                        if name.ends_with("_d16_hi") {
                            l.old & 0xFFFF | x << 16
                        } else if name.ends_with("_d16") {
                            l.old & 0xFFFF_0000 | x & 0xFFFF
                        } else {
                            x
                        }
                    }
                };
                let dword =
                    |k: usize| u32::from_le_bytes(lds[16 * i + 4 * k..][..4].try_into().unwrap());
                [x, dword(0), dword(1), dword(2), dword(3)]
            })
            .collect()
    }

    /// Runs the program of `write` and `read` over `lanes` on the test GPU,
    /// against [`expected`] (translated and validated only, without a
    /// device).
    fn check(write: Option<&str>, read: Option<&str>, lanes: &[Lane]) {
        let words = program(write, read);
        // One GPU test at a time (two at once on WARP can end the process).
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let host = match Host::new(false) {
            Ok(h) => h,
            Err(e) => {
                translate(&words, Target::ANY).unwrap();
                eprintln!("no SM 6.6 device ({e}); skipped");
                return;
            }
        };
        // A group of one console wave using LDS declares its wave size.
        let s = match translate(&words, host.target()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("not for this device ({e}); skipped");
                return;
            }
        };
        let (root, pso) = host.compute_pipeline(&s).unwrap();
        let mut data = vec![0; s.user_data_dwords().max(1) as usize];
        for u in &s.user_data {
            data[u.index as usize] = match u.slot {
                1 => 0,
                2 => 32,
                n => panic!("user data of slot {n}"),
            };
        }
        let input: Vec<u32> = lanes
            .iter()
            .enumerate()
            .flat_map(|(i, l)| [init(i), l.addr, l.data, l.old])
            .collect();
        let views = [
            View::Buffer {
                data: &input,
                stride: Stride::Raw,
            },
            View::RawUav(8 * LANES as u32),
        ];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let want = expected(write, read, lanes);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                out[0][8 * i..8 * i + 5],
                w[..],
                "{write:?} {read:?}, lane {i} (address {})",
                lanes[i].addr
            );
        }
    }

    /// Bytes or halves at `addr(lane)` (the last lane at the last LDS
    /// byte), with data whose unused bits are set.
    fn lanes(addr: impl Fn(u32) -> u32) -> Vec<Lane> {
        (0..LANES as u32)
            .map(|l| Lane {
                addr: if l == LANES as u32 - 1 {
                    LDS as u32 - 1
                } else {
                    addr(l)
                },
                data: 0xDEAD_0000 ^ (0x9A5C + 0x0101 * l) ^ (l << 20),
                old: 0x1357_9BDF ^ l,
            })
            .collect()
    }

    #[test]
    fn lds_byte_writes_keep_the_other_bytes() {
        // Three bytes apart: neighbouring lanes share dwords.
        for name in ["ds_write_b8", "ds_write_b8_d16_hi"] {
            check(Some(name), Some("ds_read_u8"), &lanes(|l| 3 * l + 1));
        }
    }

    #[test]
    fn lds_half_writes_straddle_dwords_at_byte_3() {
        // At bytes 1 and 3 of a dword in turn: every other half straddles
        // into the next dword; the last lane's past the end of LDS.
        for name in ["ds_write_b16", "ds_write_b16_d16_hi"] {
            check(Some(name), Some("ds_read_u16"), &lanes(|l| 6 * l + 1));
        }
    }

    #[test]
    fn lds_half_writes_keep_the_halves_other_lanes_write() {
        // Side by side: two lanes write each dword, aligned (2l), or two
        // and three in turn with every other half straddling at byte 3
        // (2l + 1, where lanes 31 and 32, of the two console waves, share
        // a dword).
        for name in ["ds_write_b16", "ds_write_b16_d16_hi"] {
            check(Some(name), Some("ds_read_u16"), &lanes(|l| 2 * l));
            check(Some(name), Some("ds_read_u16"), &lanes(|l| 2 * l + 1));
        }
    }

    #[test]
    fn lds_byte_and_half_reads_extend_and_place_their_value() {
        for name in [
            "ds_read_u8",
            "ds_read_i8",
            "ds_read_u16",
            "ds_read_i16",
            "ds_read_u8_d16",
            "ds_read_u8_d16_hi",
            "ds_read_i8_d16",
            "ds_read_i8_d16_hi",
            "ds_read_u16_d16",
            "ds_read_u16_d16_hi",
        ] {
            // Every byte position: halves at byte 3 straddle.
            check(None, Some(name), &lanes(|l| 5 * l));
        }
    }
}
