//! Cross-lane operations and the wave layout: whole-wave sections
//! (`emit/wave.rs`), DPP, lane and quad reads, thread ids and console
//! lanes. Backlog L1-L3 (wave64 on 32-lane GPUs), M2, and S3 for
//! `v_permlanex16_b32`.

use std::collections::HashMap;

use super::mask::MaskData;
use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Dpp, Fields, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::structure::Bool;

/// Le sélecteur utilise ses six bits bas. Toutes les waves du groupe
/// doivent contenir cette lane, y compris la dernière.
pub(super) fn present_lane_read(i: &Inst, threads: u32) -> bool {
    let Fields::Valu(v) = &i.fields else {
        return false;
    };
    let lane = match v.src.get(1) {
        Some(Operand::Literal(x)) => Some(*x),
        Some(Operand::Const(c)) => crate::resource::inline(*c),
        _ => None,
    };
    i.name() == "v_readlane_b32" && lane.is_some_and(|x| (x & 63) < threads % 64)
}

/// On ne publie que dans un bloc linéaire, avec EXEC intact : tous les
/// threads atteignent chaque barrière et fournissent leur VGPR. Entre
/// deux barrières, plusieurs lectures LDS sont permises. Une écriture ou
/// une atomique doit être le seul accès, car les demi-waves sont indépendantes.
pub(super) fn group_reads_meet(p: &Program) -> bool {
    let [block] = p.blocks.as_slice() else {
        return false;
    };
    if !matches!(block.exit, crate::ir::Exit::End) {
        return false;
    }
    let mut accessed = false;
    let mut wrote = false;
    for i in &block.insts {
        if [crate::ir::EXEC, crate::ir::EXEC + 1]
            .iter()
            .any(|&r| i.write(Reg::S(r)).is_some())
        {
            return false;
        }
        let name = i.name();
        if name == "s_barrier" {
            accessed = false;
            wrote = false;
        } else if matches!(i.fields, Fields::Ds { .. })
            && !name.starts_with("ds_swizzle")
            && !name.contains("permute")
        {
            let writes = !name.starts_with("ds_read");
            if wrote || writes && accessed {
                return false;
            }
            accessed = true;
            wrote = writes;
        }
    }
    true
}

impl<'a> Lower<'a> {
    /// Finds the whole-wave sections and the values they leave.
    pub(super) fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = crate::emit::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    pub(super) fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => Val::I32(x),
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    pub(super) fn shape(&mut self, shape: crate::emit::wave::Shape) -> Result<V, Why> {
        use crate::emit::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::CountHi(mask) => {
                if self.mask_data != MaskData::Console || self.split && !self.console_cf {
                    return Err("a count without the console wave's lanes");
                }
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let c = self.cur;
                let on = self.active_mask(mask, m, c)?;
                self.s.flags |= flags::WAVE_OPS;
                let bit = self.s.m.cast(c, Cast::ZExt, on, self.s.t.i32);
                let sum = self.s.m.i8(0);
                let unsigned = self.s.m.i8(1);
                let mut count = self.s.op(
                    c,
                    "waveActiveOp.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    119,
                    &[bit, sum, unsigned],
                );
                if self.split {
                    let (lo, hi) = self.exchange(c, count);
                    count = self.s.m.bin(c, Bin::Add, lo, hi);
                }
                // Pour 1 à 64, la fraction du f32 tient dans le mot haut du f64.
                let f = self.s.m.cast(c, Cast::UiToFp, count, self.s.t.f32);
                let bits = self.as_int(f);
                let three = self.s.m.i32(3);
                let upper = self.s.m.bin(c, Bin::LShr, bits, three);
                let bias = self.s.m.i32((1023 - 127) << 20);
                let upper = self.s.m.bin(c, Bin::Add, upper, bias);
                let z = self.s.m.i32(0);
                let empty = self.s.m.cmp(c, pred::EQ, count, z);
                self.emitted_double_count = true;
                return Ok(self.s.m.select(c, empty, z, upper));
            }
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.active_mask(mask, m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        if self.split && !self.console_cf {
            return Err("a wave reduction in a wave64 program split in halves");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.active_mask(mask, m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        if self.console_cf && self.split {
            // The halves exchange dwords: a float goes as its bits. (The
            // console adds a wave's floats in its own order, the host's
            // wave operations in theirs, and the halves' sums are added
            // last: sums may differ in the last bits, as they already do
            // in one host wave.)
            let bits = |l: &mut Self, v: V| if float { l.as_int(v) } else { v };
            let value = |l: &mut Self, v: V| if float { l.as_float(v) } else { v };
            // The console-flow path keeps both halves converged. The exclusive
            // prefix for the upper half includes the complete lower half's sum.
            if prefix.is_some() {
                let add = self.s.m.i8(0);
                let unsigned = self.s.m.i8(if float { 0 } else { 1 });
                let total = self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, add, unsigned],
                );
                let total = bits(self, total);
                let (lo, _) = self.exchange(c, total);
                let lo = value(self, lo);
                let (_, half) = self.console_lane();
                let zero = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, half, zero);
                // (A select of the sums, not of the carry: adding 0.0 would
                // turn a lower half's -0.0 into 0.0.)
                let carried = self.s.m.bin(c, Bin::Add, r, lo);
                let r = self.s.m.select(c, upper, carried, r);
                return Ok(bits(self, r));
            }
            let r = bits(self, r);
            let (lo, hi) = self.exchange(c, r);
            if float {
                let (lo, hi) = (self.as_float(lo), self.as_float(hi));
                let r = match op {
                    Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
                    // FMin and FMax, as the per-lane `v_min_f32` / `v_max_f32`.
                    Op::MinF => self.binary(36, lo, hi),
                    _ => self.binary(35, lo, hi),
                };
                return Ok(self.as_int(r));
            }
            return Ok(match op {
                Op::AddU => self.s.m.bin(c, Bin::Add, lo, hi),
                Op::And => self.s.m.bin(c, Bin::And, lo, hi),
                Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
                Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
                Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                    let pred = match op {
                        Op::MinU => pred::IULT,
                        Op::MaxU => pred::IUGT,
                        Op::MinI => pred::SLT,
                        _ => pred::SGT,
                    };
                    let choose = self.s.m.cmp(c, pred, lo, hi);
                    self.s.m.select(c, choose, lo, hi)
                }
                _ => return Err("a whole-wave reduction across 32-lane halves"),
            });
        }
        Ok(if float { self.as_int(r) } else { r })
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    pub(super) fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    pub(super) fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let f = self.console_flat();
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// The console wave size.
    pub(super) fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    pub(super) fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    pub(super) fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    /// Console flow: `x` of console lane `lane` (uniform), from whichever
    /// half holds it.
    pub(super) fn console_read_lane(&mut self, x: V, lane: V) -> V {
        if !self.split {
            // One host wave is the console wave, all of its lanes here.
            return self.read_lane(x, lane);
        }
        let c = self.cur;
        let (k31, k5, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(5),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let mine = self.read_lane_in_wave(x, at);
        let (x0, x1) = self.exchange(c, mine);
        let h = self.s.m.bin(c, Bin::LShr, lane, k5);
        let h = self.s.m.bin(c, Bin::And, h, one);
        let low = self.s.m.cmp(c, pred::EQ, h, z);
        self.s.m.select(c, low, x0, x1)
    }

    /// `x` of lane `lane`. In compute, lanes are the console's: the
    /// console takes a lane operand's low six bits (five in wave32), and a
    /// host lane past the wave reads nothing defined, so the lane is
    /// reduced to them (lane 70 of a wave64 is lane 6).
    pub(super) fn read_lane(&mut self, x: V, lane: V) -> V {
        let lane = if self.mask_data == MaskData::Console {
            let wave = self.wave();
            match self.s.m.int_value(lane) {
                Some(k) if (k as u32) < wave => lane,
                Some(k) => self.s.m.i32(k as u32 & (wave - 1)),
                None => {
                    let m = self.s.m.i32(wave - 1);
                    self.s.m.bin(self.cur, Bin::And, lane, m)
                }
            }
        } else {
            lane
        };
        self.read_lane_in_wave(x, lane)
    }

    /// `x` of host lane `lane`, which is in the host wave.
    fn read_lane_in_wave(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    pub(super) fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad {
            self.quad_pick(x, src)
        } else {
            self.read_lane_in_wave(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks. Rows
            // are the console's: split in halves, the upper half's host rows
            // 0 and 1 are console rows 2 and 3.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let row_lane = if self.split {
                self.console_lane().0
            } else {
                lane
            };
            let ri = self.s.m.bin(c, Bin::LShr, row_lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    /// Console flow: whether `m` holds in some lane of the console wave.
    pub(super) fn any_lane(&mut self, m: Bool) -> Result<V, Why> {
        let c = self.cur;
        let bit = self.boolean(m, c)?;
        let bit = self.with_partial(c, bit);
        let (lo, hi) = self.console_ballot(c, bit);
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }

    /// Console flow: the console wave's ballot of `bit`, low and high word.
    pub(super) fn console_ballot(&mut self, at: usize, bit: V) -> (V, V) {
        if self.split {
            let word = self.ballot_word(at, bit);
            return self.exchange(at, word);
        }
        // One host wave is the console wave (the high word is 0 in waves
        // of 32).
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1))
    }

    /// The host wave's ballot of `bit` (32 lanes: the first word).
    pub(super) fn ballot_word(&mut self, at: usize, bit: V) -> V {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        self.s.m.extract(at, r, 0)
    }

    /// The console thread's flat id in the group (see [`Self::thread_id`]).
    pub(super) fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let shape = self.s.threads.expect("compute");
        if shape[1..] != [1, 1] {
            // Sur NVIDIA, deux waves d'une tuile 2D peuvent avoir le même
            // premier indice arrondi (0xkn0wh3r3, VIctory #2126). Déclarer
            // le groupe sur une ligne donne à chaque wave sa plage alignée.
            // Les coordonnées console se déduisent de la forme d'origine,
            // comme pour les dérivées implicites et les lanes appariées.
            self.flat_group = Some(shape);
            self.s.threads = Some([shape.iter().product(), 1, 1]);
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.s.op(
            e,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let f = if self.s.m.paired_lanes() {
            self.s.m.paired_flat(e, base, lane)
        } else {
            self.s.m.bin(e, Bin::Add, base, lane)
        };
        self.s.flags |= flags::WAVE_OPS;
        self.console_flat = Some(f);
        f
    }

    /// This lane's console lane (0..63, 0..31 in waves of 32) and its
    /// half of a wave64 (0, 1).
    pub(super) fn console_lane(&mut self) -> (V, V) {
        let f = self.console_flat();
        let e = self.s.entry;
        let (mask, k5) = (self.s.m.i32(self.wave() - 1), self.s.m.i32(5));
        let lane = self.s.m.bin(e, Bin::And, f, mask);
        let half = self.s.m.bin(e, Bin::LShr, lane, k5);
        (lane, half)
    }

    /// Whether the console flow can run this compute program on the host:
    /// used when a per-lane mask read needs all 64 console lanes.
    /// Complete waves can run paired. A partial single wave can use the
    /// original group exchange when it has no whole-wave section.
    pub(super) fn console_flow_runs(&self) -> bool {
        if self.mask_data != MaskData::Console {
            return false;
        }
        let Some(t) = self.s.threads else {
            return false;
        };
        let threads: u32 = t.iter().product();
        let p = self.x.p;
        let permute = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name().contains("permute") && i.name() != "ds_bpermute_b32");
        if self.x.native.wave64 && !self.x.target.runs(64) {
            let sections =
                (0..p.blocks.len()).any(|b| !crate::emit::wave::sections(p, b).is_empty());
            self.x.target.runs(32)
                && !permute
                && (threads.is_multiple_of(64) || threads < 64 && !sections)
        } else {
            self.x.target.runs(self.wave()) && !permute
        }
    }

    /// Console flow in halves: whether the two group-shared dwords the
    /// halves exchange through ([`Self::exchange`]) fit next to the
    /// program's own LDS in DXIL's 32 KiB of group-shared memory. (A group
    /// of 32 threads or fewer exchanges nothing.)
    pub(super) fn exchange_fits(&self) -> Result<(), Why> {
        let threads: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        if self.console_cf
            && self.split
            && !self.s.m.paired_lanes()
            && threads > 32
            && self.s.lds_dwords + 2 > 8192
        {
            return Err("a wave64 program whose LDS leaves no room for its halves' exchange");
        }
        if self.group_reads && self.s.lds_dwords + threads > 8192 {
            return Err("a wave64 program whose LDS leaves no room for its lane reads");
        }
        Ok(())
    }

    /// `v_readlane_b32` in halves of several console waves
    /// ([`Self::group_read_lane`]). It ignores EXEC, as on the console.
    pub(super) fn group_lane_read(&mut self, inst: &Inst) -> Result<(), Why> {
        let (x, lane) = (self.src(inst, 0)?, self.src(inst, 1)?);
        self.cross_lane(false)?;
        let r = self.group_read_lane(x, lane)?;
        self.write_i(inst, r, None)
    }

    /// Halves of several console waves, each lane on its own path
    /// ([`Self::group_reads`]): `x` of console lane `lane`, fixed in the code,
    /// of this thread's console wave. Every thread that is here writes its
    /// `x` at its console thread's place in group-shared memory, the group
    /// meets, each thread reads its wave's lane, and the group meets again
    /// before the next read writes. L’admission garantit que la lane
    /// existe dans la dernière wave et que chaque thread publie sa valeur.
    pub(super) fn group_read_lane(&mut self, x: V, lane: V) -> Result<V, Why> {
        let k = self
            .s
            .m
            .int_value(lane)
            .ok_or("a lane read at a computed lane across 32-lane halves")?;
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        let slots = match self.lane_slots {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(threads, self.s.t.i32);
                let g = self.s.m.global("lanes", arr, 3);
                self.lane_slots = Some(g);
                g
            }
        };
        let c = self.cur;
        let f = self.console_flat();
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(c, slots, f);
        self.s.m.store(c, mine, x);
        self.s
            .op(c, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let (wave, at) = (self.s.m.i32(!63), self.s.m.i32(k as u32 & 63));
        let first = self.s.m.bin(c, Bin::And, f, wave);
        let index = self.s.m.bin(c, Bin::Add, first, at);
        let theirs = self.s.m.element(c, slots, index);
        let r = self.s.m.load(c, theirs);
        // Every thread read before the next lane read writes.
        self.s
            .op(c, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        Ok(r)
    }

    /// Console flow: each half gives `x` (the same in all its lanes) and
    /// both get (the lower half's, the upper half's): through two group-
    /// shared dwords between group barriers, which every thread of the
    /// group reaches (see [`CONSOLE_FLOW`]). A group of 32 threads or
    /// fewer has no upper half: its lanes count as inactive there (0).
    pub(super) fn exchange(&mut self, at: usize, x: V) -> (V, V) {
        let threads: u32 = self.s.threads.expect("compute").iter().product();
        if self.s.m.paired_lanes() {
            return self.s.m.paired_words(at, x);
        }
        let z = self.s.m.i32(0);
        if threads <= 32 {
            return (x, z);
        }
        let (_, half) = self.console_lane();
        let slots = match self.xchg {
            Some(g) => g,
            None => {
                let arr = self.s.m.array(2, self.s.t.i32);
                let g = self.s.m.global("xchg", arr, 3);
                self.xchg = Some(g);
                g
            }
        };
        let void = self.s.t.void;
        // Thread-group sync with a group-shared memory fence.
        let mode = self.s.m.i32(1 | 8);
        let mine = self.s.m.element(at, slots, half);
        self.s.m.store(at, mine, x);
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        let one = self.s.m.i32(1);
        let p0 = self.s.m.element(at, slots, z);
        let p1 = self.s.m.element(at, slots, one);
        let lo = self.s.m.load(at, p0);
        let hi = self.s.m.load(at, p1);
        // Both read before the next exchange writes.
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::group_reads_meet;

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_BARRIER: u32 = 0xBF8A_0000;
    /// v_readlane_b32 s4, v0, 40
    const READ_LANE_40: [u32; 2] = [0xD760_0004, 0x0001_5100];
    /// ds_write_b32 v1, v0
    const DS_WRITE: [u32; 2] = [0xD834_0000, 0x0000_0001];
    /// ds_read_b32 v2, v1
    const DS_READ: [u32; 2] = [0xD8D8_0000, 0x0200_0001];

    fn meet(words: &[u32]) -> bool {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        group_reads_meet(&crate::ir::build(&code, true, 256).unwrap())
    }

    #[test]
    fn a_lane_read_after_a_partial_early_exit_does_not_meet() {
        let mut words = vec![
            0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
            0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
            0xBF88_0007, // s_cbranch_execz to the end: lanes 40 and up end
        ];
        words.extend(DS_WRITE);
        words.push(S_BARRIER);
        words.extend(DS_READ);
        words.extend(READ_LANE_40);
        words.push(S_ENDPGM);
        assert!(!meet(&words));
    }

    #[test]
    fn a_lane_read_in_an_arm_some_lanes_skip_does_not_meet() {
        let mut words = vec![
            0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
            0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
            0xBF88_0002, // s_cbranch_execz over the lane read
        ];
        words.extend(READ_LANE_40);
        words.extend([
            0xBEFE_0402, // s_mov_b64 exec, s[2:3]: lanes 5 and up go on
            S_BARRIER,
            S_ENDPGM,
        ]);
        assert!(!meet(&words));
    }

    #[test]
    fn a_lane_read_in_a_loop_does_not_meet() {
        let mut words = vec![
            0xBE80_0384, // s_mov_b32 s0, 4
            0x8080_8100, // 0x04: s_sub_u32 s0, s0, 1
        ];
        words.extend(READ_LANE_40);
        words.extend([
            0xBF07_8000, // s_cmp_lg_u32 s0, 0
            0xBF85_FFFB, // s_cbranch_scc1 0x04
            S_ENDPGM,
        ]);
        assert!(!meet(&words));
    }

    /// The console runs one wave's lanes in step, so its compiler may leave
    /// out the barrier between an LDS write and a read of another lane of
    /// the wave: in two host waves the read may come first.
    #[test]
    fn lds_read_after_a_write_without_a_barrier_does_not_meet() {
        let mut words = DS_WRITE.to_vec();
        words.extend(DS_READ);
        words.extend(READ_LANE_40);
        words.push(S_ENDPGM);
        assert!(!meet(&words));
        // La barrière doit aussi séparer les deux écritures.
        let mut words = DS_WRITE.to_vec();
        words.extend(DS_WRITE);
        words.push(S_BARRIER);
        words.extend(DS_READ);
        words.push(S_ENDPGM);
        assert!(!meet(&words));
        words.insert(2, S_BARRIER);
        assert!(meet(&words));
        // A read, then a write without a barrier.
        let mut words = DS_READ.to_vec();
        words.extend(DS_WRITE);
        words.push(S_ENDPGM);
        assert!(!meet(&words));
    }
}
