//! Compute root arguments kept behind the 40-bit pointer in entry s0:s1.
//! Ported from Marvisak/VIctory (commit 27c3dc9d5) by Marvisak and Claude Fable 5.2.
//! Disposition et masque du mot haut d'après fxceless's VIctory fork (commit 578d459).

use std::collections::{HashMap, HashSet};

use super::{Fields, HwStage, Operand, Reg, Site, Term, Tracer, Value};
use crate::ShaderError;
use crate::native::{NativeProgram, StageIo};

const FIRST: u32 = 16;

#[derive(Default)]
pub(super) struct Spill {
    pub(super) active: bool,
    pub(super) loads: Vec<Site>,
    /// SSA result to layout entry and dword. Reloads share one origin.
    pub(super) bindings: HashMap<Value, (usize, u32)>,
    pub(super) first: HashMap<(usize, u32), Value>,
}

pub(super) fn has_table(native: &NativeProgram) -> bool {
    let StageIo::Cs(cs) = &native.io else {
        return false;
    };
    native.stage == HwStage::Cs
        && (2..=16).contains(&cs.user_sgprs())
        && native
            .layout
            .iter()
            .filter(|p| p.sgprs != 0)
            .map(|p| u32::from(p.user_data))
            .min()
            == Some(FIRST)
}

impl Tracer<'_> {
    /// Seules les adresses complètes, immuables à l'échelle du dispatch, peuvent
    /// être résolues en vue D3D12 pendant l'enregistrement de la commande.
    pub(super) fn constant_read(
        &self,
        inst: &crate::ir::Inst,
        site: Site,
    ) -> Option<super::ConstantRead> {
        let (lo, hi, offset) = match &inst.fields {
            Fields::Smem {
                base: Operand::Scalar { code, .. },
                offset,
                soffset,
                ..
            } if matches!(
                inst.name(),
                "s_load_dword" | "s_load_dwordx2" | "s_load_dwordx4"
            ) =>
            {
                let Term::Const(extra) = self.term(inst, soffset)? else {
                    return None;
                };
                (
                    self.origin(inst.read(Reg::S(*code))?),
                    self.origin(inst.read(Reg::S(code + 1))?),
                    extra.checked_add_signed(*offset)?,
                )
            }
            Fields::Flat {
                addr: Some(Operand::Vector { reg, .. }),
                saddr,
                offset,
                lds: false,
                ..
            } if matches!(
                inst.name(),
                "global_load_dword"
                    | "global_load_dwordx2"
                    | "global_load_dwordx3"
                    | "global_load_dwordx4"
            ) && matches!(
                saddr,
                None | Some(Operand::Scalar {
                    code: crate::gfx10::NULL,
                    ..
                })
            ) =>
            {
                // Les deux copies doivent avoir été exécutées par toutes les lanes
                // qui lisent la vue. Dans ce bloc, aucune écriture d'EXEC entre les deux.
                let scalar_copy = |r| {
                    let v = inst.read(Reg::V(r))?;
                    let (b, i) = self.defs[v.0 as usize]?;
                    if b != site.0
                        || i >= site.1
                        || self.p.blocks[b].insts[i..site.1].iter().any(|ins| {
                            ins.writes
                                .iter()
                                .any(|(r, _)| matches!(r, Reg::S(126 | 127)))
                        })
                    {
                        return None;
                    }
                    let copy = self.inst((b, i));
                    let Fields::Valu(f) = &copy.fields else {
                        return None;
                    };
                    if copy.name() != "v_mov_b32"
                        || f.abs != 0
                        || f.neg != 0
                        || f.sdwa.is_some()
                        || f.dpp.is_some()
                    {
                        return None;
                    }
                    self.term(copy, f.src.first()?)
                };
                (
                    scalar_copy(*reg)?,
                    scalar_copy(reg + 1)?,
                    u32::try_from(*offset).ok()?,
                )
            }
            _ => return None,
        };
        let (Term::Value(low), Term::Value(high)) = (lo, hi) else {
            return None;
        };
        let (param, dword, packed, field) = if let Some((p, w)) = self.param_value(low) {
            if self.param_value(high) != Some((p, w + 1)) {
                return None;
            }
            (p, w, false, 0)
        } else {
            let (low, field) = match self.sop2(low, &["s_or_b32"]) {
                Some((_, [Term::Const(k), Term::Value(v)] | [Term::Value(v), Term::Const(k)]))
                    if k < 256 =>
                {
                    (v, k)
                }
                None => (low, 0),
                _ => return None,
            };
            let (shift, [Term::Value(word), Term::Const(8)]) = self.sop2(low, &["s_lshl_b64"])?
            else {
                return None;
            };
            let Fields::Sop { src, .. } = &shift.fields else {
                return None;
            };
            let Operand::Scalar { code, .. } = src.first()? else {
                return None;
            };
            if shift.writes.get(1)?.1 != high
                || self.origin(shift.read(Reg::S(code + 1))?) != Term::Const(0)
            {
                return None;
            }
            let (p, w) = self.param_value(word)?;
            (p, w, true, field)
        };
        let root = self.layout.get(param)?;
        let offset = offset.checked_add(field)?;
        let end = offset.checked_add(4 * inst.writes.len() as u32)?;
        if end > super::MISALIGNED_SRV_BYTES {
            return None;
        }
        (root.root == crate::native::RootType::Constants && offset.is_multiple_of(4)).then_some(
            super::ConstantRead {
                slot: root.slot,
                dword,
                packed,
                offset,
            },
        )
    }

    pub(super) fn param_value(&self, v: Value) -> Option<(usize, u32)> {
        self.spill
            .bindings
            .get(&v)
            .copied()
            .or_else(|| self.register_param(self.inputs[v.0 as usize]?))
    }

    pub(super) fn param_high(&self, param: usize) -> Option<Value> {
        if self.spill.active {
            return self.spill.first.get(&(param, 1)).copied();
        }
        let high = self
            .stage
            .user_sgpr(self.layout[param].user_data)
            .checked_add(1)?;
        self.p
            .inputs
            .iter()
            .find(|i| i.0 == Reg::S(high))
            .map(|i| i.1)
    }

    /// Only the low eight bits of s1 are address bits. Trace the two SSA
    /// origins independently so later writes to either SGPR invalidate it.
    fn spill_offset(&self, inst: &crate::ir::Inst, low: Value, high: Value) -> Option<u32> {
        let Fields::Smem {
            base: Operand::Scalar { code, .. },
            offset,
            soffset,
            ..
        } = inst.fields
        else {
            return None;
        };
        if !inst.name().starts_with("s_load_dword")
            || self.origin(inst.read(Reg::S(code))?) != Term::Value(low)
        {
            return None;
        }
        let Term::Value(masked) = self.origin(inst.read(Reg::S(code + 1))?) else {
            return None;
        };
        let (_, halves) = self.sop2(masked, &["s_and_b32"])?;
        if halves != [Term::Value(high), Term::Const(0xff)]
            && halves != [Term::Const(0xff), Term::Value(high)]
        {
            return None;
        }
        let Term::Const(c) = self.term(inst, &soffset)? else {
            return None;
        };
        let byte = c.checked_add_signed(offset)?;
        byte.is_multiple_of(4).then_some(FIRST + byte / 4)
    }

    pub(super) fn read_spill_table(&mut self) -> Result<(), ShaderError> {
        self.spill.active = true;
        let input = |r| self.p.inputs.iter().find(|i| i.0 == Reg::S(r)).map(|i| i.1);
        let (Some(low), Some(high)) = (input(0), input(1)) else {
            return Ok(());
        };
        let used: HashSet<Value> = self
            .p
            .blocks
            .iter()
            .flat_map(|b| {
                b.phis
                    .iter()
                    .flat_map(|p| p.args.iter().copied())
                    .chain(b.insts.iter().flat_map(|i| i.reads.iter().map(|r| r.1)))
            })
            .collect();
        for (b, block) in self.p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                let Some(first) = self.spill_offset(inst, low, high) else {
                    continue;
                };
                for (k, &(_, v)) in inst.writes.iter().enumerate() {
                    let index = first + k as u32;
                    let mut entries = self.layout.iter().enumerate().filter_map(|(param, p)| {
                        let dword = index.checked_sub(u32::from(p.user_data))?;
                        (dword < p.sgprs).then_some((param, dword))
                    });
                    let binding = match (entries.next(), entries.next()) {
                        (Some(pd), None) => pd,
                        (None, None) if k != 0 && !used.contains(&v) => continue, // unused padding
                        _ => {
                            return Err(ShaderError::Unsupported {
                                offset: inst.machine.offset,
                                what: "a spill-table dword without a unique root parameter",
                            });
                        }
                    };
                    self.spill.bindings.insert(v, binding);
                    self.spill.first.entry(binding).or_insert(v);
                    // This word is immutable user data. Descriptor provenance
                    // must never classify it as a fetched descriptor word.
                    self.defs[v.0 as usize] = None;
                }
                self.spill.loads.push((b, i));
            }
        }
        if !self.spill.loads.is_empty() {
            self.origin = self.propagate_copies(self.defs.len())?;
        }
        Ok(())
    }
}
