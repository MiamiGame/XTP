//! Descriptors: the host handle of each resource access (heap index from
//! the console address, root descriptors, per-lane and waterfall forms)
//! and the used-resource entry that says its kind. Backlog R4 (per-lane
//! descriptor long tail), R3 (strides from `used_kind`, else from the
//! host's runtime stride table: [`Lower::stride_entry`]).

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::{REMAPPED_SLOT, STRIDE_HEADER, StrideEntry};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Reg};
use crate::native::{ResourceType, RootType, UsedResource};
use crate::resource::{Site, Source};

/// What the refusal of a buffer read by element without a stride says.
pub(super) const UNKNOWN_STRIDE: Why = "a structured buffer of unknown stride";

/// What the refusal of a buffer read by element with a known stride of 1
/// to 3 bytes says (an R8 or R16 structured buffer, watch.md item 318):
/// counted apart from unknown strides, which the runtime stride table
/// translates.
pub(super) const SMALL_STRIDE: Why = "a structured buffer of stride 1 to 3";

/// What the refusal of a buffer read by element with a known stride past 4
/// but off a multiple of 4 says (watch.md item 318).
pub(super) const ODD_STRIDE: Why = "a structured buffer of stride off a multiple of 4";

/// A runtime stride table entry ([`StrideEntry`]) as a program reads it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Entry {
    /// The view-heap slot of the raw view to access (only where `count`
    /// is above 0).
    pub raw: V,
    /// Bytes per element; 0 for a raw view.
    pub stride: V,
    /// Bytes from the raw view's start to the view's.
    pub bias: V,
    /// Elements of the view (dwords of a raw one); 0: no buffer view.
    pub count: V,
    /// The slot may differ between lanes (so may `raw`).
    pub nonuniform: bool,
}

impl<'a> Lower<'a> {
    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    pub(super) fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                let x = match address.dynamic {
                    Some(v) => Some(self.i32_of(v)?),
                    None => None,
                };
                let x = if self
                    .used_kind(src)
                    .is_some_and(|u| u.kind == ResourceType::Uav)
                {
                    let bias = self.host_word(
                        crate::emit::ROOT_UAV_BIAS,
                        self.x.native.layout[address.param].slot,
                        0,
                    );
                    Some(x.map_or(bias, |v| {
                        let mut valid = self.s.m.bool(true);
                        let sum = self.raw_add(v, bias, &mut valid);
                        let max = self.s.m.i32(u32::MAX);
                        self.s.m.select(self.cur, valid, sum, max)
                    }))
                } else {
                    x
                };
                Ok((address.offset, x))
            }
            _ => Ok((0, None)),
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    pub(super) fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// Whether [`Self::handle`] gives each lane a heap index of its own for
    /// `src` (a waterfall loop's load, a descriptor fetched or chosen per
    /// lane). What a lane outside EXEC holds there may be no index at all:
    /// it must not reach an access through such a handle.
    pub(super) fn lane_descriptor(&self, src: &Source) -> bool {
        match src {
            Source::Loaded { load, .. } => self.x.res.waterfall.loads.contains(load),
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => true,
            Source::Built { .. }
            | Source::Constant(_)
            | Source::Phi { .. }
            | Source::Select { .. } => false,
        }
    }

    /// Whether `src` is a descriptor of a view-heap slot the program picks
    /// when it runs: loaded at an address with a part known only then (an
    /// index read from a buffer, say), from the resource heap itself, or
    /// per lane. No used-resource entry names such a slot.
    pub(super) fn picked_at_run_time(&self, src: &Source) -> bool {
        match src {
            Source::Loaded { load, .. } => self.scalar.get(load).is_some_and(|a| {
                a.dynamic.is_some()
                    || self
                        .x
                        .native
                        .layout
                        .get(a.param)
                        .is_some_and(|p| p.root == RootType::ResourceHeap)
            }),
            Source::Phi { args, .. } | Source::Select { args, .. } => {
                args.iter().any(|a| self.picked_at_run_time(a))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => true,
            Source::Constant(_) | Source::Built { .. } => false,
        }
    }

    /// Notes a buffer handle (`props`: its kind, with the unordered access
    /// bit) made of `src`'s slot, for [`super::Lowered::run_time_buffers`].
    pub(super) fn note_run_time(&mut self, src: &Source, props: u32) {
        let buffer = matches!(props & 0xFF, 10..=12);
        if buffer && self.picked_at_run_time(src) {
            self.run_time_buffers |= crate::emit::RUN_TIME_READS;
            if props & 0x1000 != 0 {
                self.run_time_buffers |= crate::emit::RUN_TIME_WRITES;
            }
        }
    }

    /// A handle for the descriptor `src` read at `site`.
    pub(super) fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        if !sampler {
            self.note_run_time(src, props.0);
        }
        match src {
            Source::Loaded { load, first } => {
                let key = (self.cur, *load, *first, sampler, props.0);
                if let Some(&h) = self.handles.get(&key) {
                    return Ok(h);
                }
                let index = self.heap_index(*load, *first, sampler)?;
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words: [u32; 4] = words
                    .get(..4)
                    .and_then(|w| w.try_into().ok())
                    .ok_or("a sampler constant")?;
                // Portage de VIctory #1477, commit 2286e038, par geiyiliyabumo.
                // Le filtre de chaque registre statique est fixé dès le premier accès.
                let comparison = props.0 & 0x8000 != 0;
                let slot = self.samplers.iter().enumerate().position(|(n, w)| {
                    *w == words && (self.s.cmp_samplers >> n & 1 != 0) == comparison
                });
                let n = match slot {
                    Some(n) => n,
                    None => {
                        let n = self.samplers.len();
                        if n >= u64::BITS as usize {
                            return Err("more than 64 immediate sampler slots");
                        }
                        self.samplers.push(words);
                        self.s.cmp_samplers |= u64::from(comparison) << n;
                        n
                    }
                };
                Ok(self.s.sampler_handle(self.cur, n as u32))
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self.used_kind(src).map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_raw_handle(rp.slot, false))
                    }
                    (RootType::Descriptor, Some(ResourceType::Uav), 0x100b) => {
                        Ok(self.s.root_raw_handle(rp.slot, true))
                    }
                    _ => Err("a typed or incompatible root view"),
                }
            }
            Source::Phi { block, args } => {
                let _ = site;
                let phi = self.phi_heap_index(*block, args, sampler, false)?;
                Ok(self.s.heap_handle(self.cur, phi, sampler, false, props))
            }
            Source::Select { scc, args } => {
                let index = self.select_heap_index(*scc, args, sampler, false)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index of a descriptor chosen by control flow: each
    /// predecessor of `block` computes its own at its end, the join picks
    /// one with a phi. `marked`: with [`REMAPPED_SLOT`] where the user data
    /// has it ([`Self::marked_heap_index`]).
    fn phi_heap_index(
        &mut self,
        block: usize,
        args: &[Source],
        sampler: bool,
        marked: bool,
    ) -> Result<V, Why> {
        let mut incoming = Vec::new();
        for (&q, arg) in self.x.p.blocks[block].preds.iter().zip(args) {
            if self.block[q].is_none() {
                continue;
            }
            let (Some(tail), Source::Loaded { load, first }) = (self.tail[q], arg) else {
                return Err("a descriptor chosen by control flow");
            };
            let at = self.edges.get(&(q, block)).copied().unwrap_or(tail);
            let here = std::mem::replace(&mut self.cur, at);
            let index = self.index_of(*load, *first, sampler, marked);
            self.cur = here;
            incoming.push((index?, at));
        }
        let join = self.block[block].ok_or("a descriptor chosen by control flow")?;
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.s.m.set_phi(phi, incoming);
        Ok(phi)
    }

    /// The heap index of a descriptor of two picked by SCC (uniform);
    /// `marked` as for [`Self::phi_heap_index`].
    fn select_heap_index(
        &mut self,
        scc: crate::ir::Value,
        args: &[Source],
        sampler: bool,
        marked: bool,
    ) -> Result<V, Why> {
        let [
            Source::Loaded { load: a, first: fa },
            Source::Loaded { load: b, first: fb },
        ] = args[..]
        else {
            return Err("a descriptor chosen by a condition");
        };
        let cond = match self.get(scc)? {
            Val::Bit(x) => x,
            _ => return Err("SCC without a value"),
        };
        let set = self.index_of(a, fa, sampler, marked)?;
        let clear = self.index_of(b, fb, sampler, marked)?;
        Ok(self.s.m.select(self.cur, cond, set, clear))
    }

    /// The view-heap slot of `src`'s descriptor as [`Self::handle`] computes
    /// it, in the current block, with [`REMAPPED_SLOT`] where the user data
    /// has it, and whether it may differ between lanes.
    fn heap_slot(&mut self, src: &Source) -> Result<(V, bool), Why> {
        let waterfall = |l: &Self, s: &Source| matches!(s, Source::Loaded { load, .. } if l.x.res.waterfall.loads.contains(load));
        match src {
            Source::Loaded { load, first } => Ok((
                self.marked_heap_index(*load, *first, false)?,
                waterfall(self, src),
            )),
            Source::Phi { block, args } => {
                // A waterfall load picks a slot per lane, so does a phi of one.
                let lanes = args.iter().any(|a| waterfall(self, a));
                Ok((self.phi_heap_index(*block, args, false, true)?, lanes))
            }
            Source::Select { scc, args } => {
                let lanes = args.iter().any(|a| waterfall(self, a));
                Ok((self.select_heap_index(*scc, args, false, true)?, lanes))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                Ok((self.lanes_heap_index(src, false, true)?, true))
            }
            Source::Constant(_) | Source::Built { .. } => Err(UNKNOWN_STRIDE),
        }
    }

    /// The host's runtime stride table entry ([`StrideEntry`], `t0` in
    /// [`crate::emit::STRIDE_SPACE`]) of the view-heap slot of `src`, read
    /// in the current block: one load of its two dwords, which the host
    /// writes with one store.
    ///
    /// The table is a root SRV, which bounds no read, so the entry is read
    /// only inside it: past its header ([`STRIDE_HEADER`]), for a slot the
    /// host remapped ([`REMAPPED_SLOT`]) below the entries the header
    /// counts, for a title slot below the title slots the host mirrors.
    /// Any other slot reads the header's first entry, which holds no view.
    pub(super) fn stride_entry(&mut self, src: &Source) -> Result<Entry, Why> {
        let key = match src {
            Source::Loaded { load, first } => Some((self.cur, *load, *first)),
            _ => None,
        };
        if let Some(e) = key.and_then(|k| self.entries.get(&k)) {
            return Ok(*e);
        }
        let (slot, nonuniform) = self.heap_slot(src)?;
        let c = self.cur;
        let table = self.s.stride_table_handle();
        // The header: [0, title slots mirrored, 0, entries].
        let (zero, iu, all, align) = (
            self.s.m.i32(0),
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8(0xF),
            self.s.m.i32(4),
        );
        let header = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, zero, iu, all, align],
        );
        let mirrored = self.s.m.extract(c, header, 1);
        let mirrored = self.as_int(mirrored);
        let entries = self.s.m.extract(c, header, 3);
        let entries = self.as_int(entries);
        let mark = self.s.m.i32(REMAPPED_SLOT);
        let marked = self.s.m.bin(c, Bin::And, slot, mark);
        let remapped = self.s.m.cmp(c, pred::NE, marked, zero);
        let keep = self.s.m.i32(!REMAPPED_SLOT);
        let slot = self.s.m.bin(c, Bin::And, slot, keep);
        let limit = self.s.m.select(c, remapped, entries, mirrored);
        let inside = self.s.m.cmp(c, pred::IULT, slot, limit);
        let skip = self.s.m.i32(STRIDE_HEADER);
        let past = self.s.m.bin(c, Bin::Add, slot, skip);
        let entry = self.s.m.select(c, inside, past, zero);
        let bytes = self.s.m.i32(StrideEntry::BYTES);
        let at = self.s.m.bin(c, Bin::Mul, entry, bytes);
        let (iu, mask, align) = (
            self.s.m.undef(self.s.t.i32),
            self.s.m.i8(3),
            self.s.m.i32(4),
        );
        let r = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, at, iu, mask, align],
        );
        let low = self.s.m.extract(c, r, 0);
        let low = self.as_int(low);
        let high = self.s.m.extract(c, r, 1);
        let high = self.as_int(high);
        // The low dword: the count, the bias over 4 in its top two bits;
        // the high one: the stride over 4 in ten bits, the raw slot above.
        let (k2, k10, k30) = (self.s.m.i32(2), self.s.m.i32(10), self.s.m.i32(30));
        let counts = self.s.m.i32(StrideEntry::MAX_COUNT);
        let count = self.s.m.bin(c, Bin::And, low, counts);
        let bias = self.s.m.bin(c, Bin::LShr, low, k30);
        let bias = self.s.m.bin(c, Bin::Shl, bias, k2);
        let strides = self.s.m.i32(0x3FF);
        let stride = self.s.m.bin(c, Bin::And, high, strides);
        let stride = self.s.m.bin(c, Bin::Shl, stride, k2);
        let raw = self.s.m.bin(c, Bin::LShr, high, k10);
        let e = Entry {
            raw,
            stride,
            bias,
            count,
            nonuniform,
        };
        if let Some(k) = key {
            self.entries.insert(k, e);
        }
        Ok(e)
    }

    /// Whether accesses of `src`'s buffer by offset alone go through the
    /// runtime stride table's raw view: on a target that has the table,
    /// for a descriptor of the view heap. The handle is then raw over a
    /// raw view whatever view the title made; read raw, a structured view
    /// returns other dwords or zeros on NVIDIA (watch.md items 148 and 160),
    /// and a texture or null slot is not accessed at all.
    pub(super) fn raw_through_table(&self, src: &Source) -> bool {
        self.x.target.stride_table
            && matches!(
                src,
                Source::Loaded { .. }
                    | Source::Phi { .. }
                    | Source::Select { .. }
                    | Source::Fetched { .. }
                    | Source::Lanes { .. }
                    | Source::LanePhi { .. }
            )
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    pub(super) fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        self.lanes_heap_index(src, sampler, false)
    }

    /// [`Self::lane_heap_index`], `marked` as for [`Self::phi_heap_index`].
    fn lanes_heap_index(&mut self, src: &Source, sampler: bool, marked: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.index_of(*load, *first, sampler, marked),
            Source::Fetched { load, first } => {
                let index = self.marked_fetched_index(*load, *first, sampler)?;
                Ok(if marked {
                    index
                } else {
                    self.unmarked(index, sampler)
                })
            }
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lanes_heap_index(a, sampler, marked)?;
                let clear = self.lanes_heap_index(b, sampler, marked)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let index = self.lanes_heap_index(arg, sampler, marked);
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size, with [`REMAPPED_SLOT`] where the user data
    /// has it.
    fn marked_fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block: the host slot a
    /// handle is made at ([`REMAPPED_SLOT`] cleared).
    pub(super) fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        self.index_of(load, first, sampler, false)
    }

    /// [`Self::heap_index`], or with `marked` [`Self::marked_heap_index`].
    fn index_of(&mut self, load: Site, first: u8, sampler: bool, marked: bool) -> Result<V, Why> {
        let index = self.marked_heap_index(load, first, sampler)?;
        Ok(if marked {
            index
        } else {
            self.unmarked(index, sampler)
        })
    }

    /// `index` without [`REMAPPED_SLOT`], in the current block, on a target
    /// with the stride table (the host marks no index without one, and
    /// sampler tables are never remapped).
    fn unmarked(&mut self, index: V, sampler: bool) -> V {
        if sampler || !self.x.target.stride_table {
            return index;
        }
        let keep = self.s.m.i32(!REMAPPED_SLOT);
        self.s.m.bin(self.cur, Bin::And, index, keep)
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, as the user data gives it: with
    /// [`REMAPPED_SLOT`] for a table the host remapped.
    fn marked_heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }
}
