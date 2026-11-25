//! Descriptor loads whose registers the console compiler gives to numbers
//! after the descriptor's last access (R1 phase 2,
//! docs/analysis/r1-scalar-data.md, sections 4 and 5): their dwords reach
//! joins with those numbers, and [`Lower::descriptor_data`] took a join's
//! operand for a read of the descriptor as data.

use std::collections::HashSet;

use super::Lower;
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg, Value};
use crate::resource::{Site, Source};

/// How many joins [`Lower::takes_data`] visits from one join before it
/// gives up (the load keeps its refusal): bounds the walk in programs whose
/// joins chain over thousands of blocks.
const JOIN_WALK: usize = 4096;

impl Lower<'_> {
    /// Whether the dwords `dwords` of the descriptor load at `site` are
    /// read only as the descriptor of accesses through the load itself,
    /// and by joins each of which also takes a value no descriptor load
    /// wrote: the registers hold the descriptor until its last access, then
    /// a number, and where the paths meet the program reads the number.
    /// Such dwords have no host value ([`super::smem::DESCRIPTOR_WORD`]), as
    /// a load's that nothing reads, and `flow.rs` takes them at those joins
    /// as undefined on the load's edge.
    ///
    /// A number among a join's values shows the registers were reused, not
    /// that the program reads the joined value only on the number's edge
    /// (Codex review 3983f844: a read of the join on every edge stores the
    /// descriptor's dword on the load's). Per lane, `flow.rs` proves that
    /// edge unread (`unread_on_edge`), and a read there stops the program
    /// as a direct read does. The console flow takes a descriptor's dword
    /// at a join without that proof where it fails (its shortcut, counted
    /// in `emit::UNPROVEN_DESCRIPTOR_JOINS`), so there the load keeps its
    /// refusal: the program then takes the per-lane form where that
    /// translates.
    ///
    /// Not where any access takes the descriptor through a join or a choice
    /// (a descriptor chosen by control flow keeps its own path), nor where a
    /// join of the dwords takes only descriptor loads' dwords (two
    /// descriptors in one register, which a number read of the join reads
    /// as data), nor where no access uses the load itself (`flow.rs` takes
    /// only a descriptor's dwords as undefined at a join).
    pub(super) fn reused_after_access(&self, site: Site, dwords: &[Option<Value>]) -> bool {
        if self.console_cf {
            return false;
        }
        let ours: HashSet<Value> = dwords.iter().flatten().copied().collect();
        let mut direct = HashSet::new();
        // Every scalar load a descriptor comes from (`descriptor_dword`'s).
        let mut loads = HashSet::new();
        for d in &self.x.res.descriptors {
            let this = |s: &Source| matches!(*s, Source::Loaded { load, .. } if load == site);
            let leaves = d.source.leaves();
            loads.extend(leaves.iter().filter_map(|l| match **l {
                Source::Loaded { load, .. } => Some(load),
                _ => None,
            }));
            if !leaves.into_iter().any(this) {
                continue;
            }
            if !this(&d.source) {
                return false;
            }
            direct.insert(d.site);
        }
        if direct.is_empty() {
            return false;
        }
        let access: HashSet<Site> = self.x.res.descriptors.iter().map(|d| d.site).collect();
        let mut joins = HashSet::new();
        for (b, block) in self.x.p.blocks.iter().enumerate() {
            for phi in &block.phis {
                if phi.args.iter().any(|a| ours.contains(a)) {
                    joins.insert(phi.value);
                }
            }
            for (i, inst) in block.insts.iter().enumerate() {
                let own = direct.contains(&(b, i));
                for &(reg, v) in &inst.reads {
                    if ours.contains(&v) && !(own && descriptor_operand(inst, reg)) {
                        return false;
                    }
                }
            }
        }
        // A join an access reads (as its descriptor or otherwise).
        for &(b, i) in &access {
            let inst: &Inst = &self.x.p.blocks[b].insts[i];
            if inst.reads.iter().any(|r| joins.contains(&r.1)) {
                return false;
            }
        }
        joins.into_iter().all(|j| self.takes_data(j, &loads))
    }

    /// Whether the join whose value is `join` takes, directly or through
    /// other joins, a value none of the descriptor loads `loads` wrote: a
    /// number.
    fn takes_data(&self, join: Value, loads: &HashSet<Site>) -> bool {
        let descriptor = |v: Value| {
            let Some(&b) = self.def_block.get(&v) else {
                return false;
            };
            let insts = &self.x.p.blocks[b].insts;
            insts
                .iter()
                .position(|i| i.writes.iter().any(|w| w.1 == v))
                .is_some_and(|i| {
                    insts[i].name().starts_with("s_load_dword") && loads.contains(&(b, i))
                })
        };
        let mut seen = HashSet::from([join]);
        let mut work = vec![join];
        while let Some(v) = work.pop() {
            let Some(&(b, i)) = self.phi_at.get(&v) else {
                continue;
            };
            for &a in &self.x.p.blocks[b].phis[i].args {
                if descriptor(a) {
                    continue;
                }
                if !self.phi_at.contains_key(&a) {
                    return true;
                }
                if seen.len() >= JOIN_WALK {
                    return false;
                }
                if seen.insert(a) {
                    work.push(a);
                }
            }
        }
        false
    }
}

/// Whether `inst` reads `reg` as a descriptor: the resource of an
/// `s_buffer_*` instruction, a buffer or image access, or an image's
/// sampler.
fn descriptor_operand(inst: &Inst, reg: Reg) -> bool {
    let within = |o: &Operand| match (*o, reg) {
        (Operand::Scalar { code, dwords }, Reg::S(r)) => (code..code + dwords).contains(&r),
        _ => false,
    };
    match &inst.fields {
        Fields::Smem { base, .. } => inst.name().starts_with("s_buffer_") && within(base),
        Fields::Buffer { rsrc, .. } => within(rsrc),
        Fields::Image { rsrc, sampler, .. } => within(rsrc) || sampler.as_ref().is_some_and(within),
        _ => false,
    }
}
