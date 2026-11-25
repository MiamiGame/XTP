//! Lane masks: structurisation's per-lane booleans as DXIL values, and
//! masks read as numbers (ballots, see [`MaskData`]). Backlog M1 (masks
//! the boolean tracker loses, kept as numbers) and M2 (masks holding
//! lanes elsewhere) start here.

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Reg, Value};
use crate::structure::{Atom, Bool, Structured};

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
pub(super) const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
pub(super) fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

impl<'a> Lower<'a> {
    /// Le masque initial d'EXEC dépend de la taille de la queue. Un mot de
    /// masque construit comme donnée peut aussi contenir des lanes hors EXEC.
    pub(super) fn active_mask(&mut self, v: Value, mask: Bool, at: usize) -> Result<V, Why> {
        let bit = self.boolean(mask, at)?;
        Ok(if self.x.st.active.contains(&v) {
            self.with_partial(at, bit)
        } else {
            bit
        })
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    pub(super) fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        // Per lane, a mask from a join resolved to a constant is that
        // constant only in the lanes here: a loop's exit mask is "no lane"
        // in every lane still in the loop, and holds the lanes that left.
        let resolved = !self.console_cf && self.x.st.resolved.contains(&lo);
        if m == Bool::FALSE && !resolved {
            // No lane anywhere: 0 whoever holds the mask.
            return Ok(self.s.m.i32(0));
        }
        if self.console_cf {
            // Every lane runs every block: the mask's bits are the lanes'
            // own, both halves' words exchanged.
            if let Some(&xy) = self.ballots.get(&lo) {
                return Ok(if high { xy.1 } else { xy.0 });
            }
            let at = match self.def_block.get(&lo) {
                None => self.s.entry,
                Some(&b) if b == self.cur_ir => self.cur,
                Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
            };
            let bit = self.active_mask(lo, m, at)?;
            let xy = self.console_ballot(at, bit);
            self.ballots.insert(lo, xy);
            return Ok(if high { xy.1 } else { xy.0 });
        }
        if self.split {
            return Err("a lane mask read whole in a wave64 program split in halves");
        }
        if resolved || !self.x.st.active.contains(&lo) {
            // Lanes of the mask are in other blocks (lanes that left a
            // loop, say), so no ballot here holds them. In compute the
            // console flow has the mask: every lane runs every block there.
            return Err(if self.console_flow_runs() && !self.x.per_lane_only {
                super::CONSOLE_FLOW
            } else {
                "a lane mask holding lanes elsewhere read as data"
            });
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => xy,
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let bit = self.with_partial(at, bit?);
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    pub(super) fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            return Ok(v);
        }
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        self.bools.insert((at, b), v);
        Ok(v)
    }

    pub(super) fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => self.lane_bits.get(&v).copied().ok_or_else(|| {
                self.lane_fail
                    .get(&v)
                    .copied()
                    .unwrap_or("a lane mask from a carry")
            }),
            Atom::Uniform(v) => match self.get(v) {
                Ok(Val::Bit(x)) => Ok(x),
                Ok(Val::I32(x)) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Ok(Val::Missing(w)) => Err(w),
                // Mesh programs count no mask: the SCC of a count, "some
                // lane is set", is each lane's own bit, as a mask
                // operation's is in a branch (structurisation's rule).
                Err(w) => match self.count_scc(v) {
                    Some(m) if self.mask_data == MaskData::None => self.boolean(m, at),
                    _ => Err(w),
                },
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // Split in halves, a host lane is not its console lane (the
            // upper half's lanes 0-31 are console lanes 32-63), so the
            // per-lane form has no bit of a mask held as a number to test
            // (a select or carry on one, which the split test does not
            // see): the console flow has.
            Atom::Opaque(_) | Atom::Constant(_) if self.split && !self.console_cf => {
                Err(if self.x.per_lane_only || !self.console_flow_runs() {
                    "a lane mask held as a number in a wave64 program split in halves"
                } else {
                    super::CONSOLE_FLOW
                })
            }
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                // The number may itself be known only as this mask's dword
                // (a mask copied dword by dword): no number to test then.
                if !self.opaque_open.insert(v) {
                    return Err("an opaque lane mask");
                }
                let words = (|| {
                    let hi = match self.pair_high(v) {
                        Some(h) => Some(self.get(h)?),
                        None => None,
                    };
                    Ok((self.get(v)?, hi))
                })();
                self.opaque_open.remove(&v);
                let (lo, hi) = words?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64; host lane = console
    /// lane, so not in a program split in halves: [`Self::atom`] sends
    /// those to the console flow first).
    pub(super) fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        debug_assert!(!self.split || self.console_cf);
        if self.console_cf {
            // The console lane's bit: its half's word, its host lane's bit.
            let (lane, half) = self.console_lane();
            let (k31, one, z) = (self.s.m.i32(31), self.s.m.i32(1), self.s.m.i32(0));
            let word = match hi {
                Some(hi) => {
                    let low = self.s.m.cmp(at, pred::EQ, half, z);
                    self.s.m.select(at, low, lo, hi)
                }
                None => lo,
            };
            let sh = self.s.m.bin(at, Bin::And, lane, k31);
            let t = self.s.m.bin(at, Bin::LShr, word, sh);
            let t = self.s.m.bin(at, Bin::And, t, one);
            return self.s.m.cmp(at, pred::NE, t, z);
        }
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    pub(super) fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        // Written with it by one instruction; else (two instructions, one
        // for each dword) the dword the mask's reader takes with it.
        let together = || {
            let b = *self.def_block.get(&lo)?;
            let inst = p.blocks[b]
                .insts
                .iter()
                .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
            let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
            inst.write(pick(reg)?)
        };
        together().or_else(|| self.x.st.pairs.get(&lo).copied())
    }

    pub(super) fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    pub(super) fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    pub(super) fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    /// The mask `s_bcnt1_i32_b*` counts where `scc` is that count's SCC
    /// ("the count is not zero": some lane of the mask is set), as lane bits.
    fn count_scc(&self, scc: Value) -> Option<Bool> {
        let p = self.x.p;
        let &b = self.def_block.get(&scc)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.write(Reg::Scc) == Some(scc))?;
        let width = if self.x.native.wave64 { "64" } else { "32" };
        if inst.name().strip_prefix("s_bcnt1_i32_b") != Some(width) {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        let &Operand::Scalar { code, .. } = src.first()? else {
            return None;
        };
        let lo = inst.read(Reg::S(code))?;
        let hi = if self.x.native.wave64 {
            Some(inst.read(Reg::S(code + 1))?)
        } else {
            None
        };
        self.x.st.mask_bits(lo, hi)
    }

    /// Whether `v_readfirstlane_b32` `inst`, in block `b`, is the first-lane
    /// read of the console compiler's test for a value the whole wave holds:
    ///
    /// ```text
    /// v_readfirstlane_b32 s, v
    /// v_cmp_eq_u32 m, s, v           (the lanes that hold the first lane's)
    /// s_xor_b64 d, m, exec           (those that do not)
    /// [s_bcnt1_i32_b64 n, d]
    /// s_cbranch_scc1 general         (some lane differs: a waterfall loop)
    /// ...                            (else the scalar path, which uses s)
    /// ```
    ///
    /// Each lane may take its own value as the first lane's: its compare
    /// then holds, its bit of `d` is clear, and it takes the scalar path with
    /// its own value, as the console's wave does when every lane holds it;
    /// the general path gives each lane what the scalar path gives it with
    /// its own value (it exists for the waves whose lanes differ). The
    /// branch is the lane's own bit (the SCC of a count in a mesh program,
    /// [`Self::atom`]). No read of `s` but the compare may lie where the
    /// scalar path does not decide it: every other reader is in a block the
    /// scalar path's first block dominates.
    pub(super) fn uniform_test_first_lane(&self, inst: &Inst, b: usize) -> bool {
        let p = self.x.p;
        let block = &p.blocks[b];
        let Fields::Valu(rf) = &inst.fields else {
            return false;
        };
        let (Some(&(Reg::S(_), s)), Some(&Operand::Vector { reg, .. })) =
            (inst.writes.first(), rf.src.first())
        else {
            return false;
        };
        let Some(v) = inst.read(Reg::V(reg)) else {
            return false;
        };
        let Some(at) = block
            .insts
            .iter()
            .position(|i| i.machine.offset == inst.machine.offset)
        else {
            return false;
        };
        let rest = &block.insts[at + 1..];
        let reads = |i: &Inst, x: Value| i.reads.iter().any(|r| r.1 == x);
        // The compare of the value with its first lane's.
        let Some(cmp) = rest.iter().find(|i| {
            matches!(i.name(), "v_cmp_eq_u32" | "v_cmp_eq_i32")
                && matches!(&i.fields, Fields::Valu(f) if f.sdwa.is_none() && f.dpp.is_none())
                && reads(i, s)
                && reads(i, v)
        }) else {
            return false;
        };
        let same: Vec<Value> = cmp
            .writes
            .iter()
            .filter(|w| matches!(w.0, Reg::S(_)))
            .map(|w| w.1)
            .collect();
        // The lanes that differ: the compare's mask against EXEC.
        let xor = if self.x.native.wave64 {
            "s_xor_b64"
        } else {
            "s_xor_b32"
        };
        let Some(differ) = rest.iter().find(|i| {
            i.name() == xor && i.read(Reg::S(EXEC)).is_some() && same.iter().any(|&m| reads(i, m))
        }) else {
            return false;
        };
        // The block's branch on "some lane differs".
        let Some(last) = block.insts.last() else {
            return false;
        };
        let taken_differ = match last.name() {
            "s_cbranch_scc1" => true,
            "s_cbranch_scc0" => false,
            _ => return false,
        };
        let crate::ir::Exit::Branch { taken, not_taken } = block.exit else {
            return false;
        };
        let Some(scc) = last.read(Reg::Scc) else {
            return false;
        };
        let counted = rest.iter().any(|i| {
            i.name().starts_with("s_bcnt1_i32_")
                && i.write(Reg::Scc) == Some(scc)
                && differ.writes.iter().any(|w| reads(i, w.1))
        });
        if differ.write(Reg::Scc) != Some(scc) && !counted {
            return false;
        }
        let scalar = if taken_differ { not_taken } else { taken };
        // Every other read of the value where the scalar path decides it.
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut index = vec![usize::MAX; p.blocks.len()];
        for (n, &q) in order.iter().enumerate() {
            index[q] = n;
        }
        let idom = crate::ir::idoms(&p.blocks, &order, &index);
        let dominated = |mut q: usize| loop {
            if q == scalar {
                break true;
            }
            if idom[q] == q || idom[q] == usize::MAX {
                break false;
            }
            q = idom[q];
        };
        p.blocks.iter().enumerate().all(|(q, blk)| {
            let insts = blk
                .insts
                .iter()
                .all(|i| std::ptr::eq(i, cmp) || !reads(i, s) || dominated(q));
            let phis = blk.phis.iter().all(|phi| {
                phi.args
                    .iter()
                    .zip(&blk.preds)
                    .all(|(&a, &pred)| a != s || dominated(pred))
            });
            insts && phis
        })
    }
}
