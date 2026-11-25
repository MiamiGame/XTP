//! Scalar memory (SMEM): constant-buffer reads, scalar loads read as
//! descriptor words or as data (`descriptor_data`), the program's own
//! constant table. Backlog R1 (data reads through root descriptors: the
//! loads `smem` hands to `descriptor_data` today) and R2 (descriptor
//! memory as data).

use std::collections::HashMap;

use super::{Lower, Val, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Reg, Value};
use crate::native::{ResourceType, UsedResource};
use crate::resource::{Address, CODE, Role, Site, Source};

/// What a descriptor dword with no host value (an address, a format,
/// tiling) holds as data: [`Lower::descriptor_data`] rebuilds only sizes.
pub(super) const DESCRIPTOR_WORD: Why = "a descriptor's address or format read as data";

pub(super) const ONLY_DATA: Why = "a descriptor read only as data";

/// What [`Lower::unused_descriptor`] finds a descriptor load to be.
enum Unused<'a> {
    /// No dword has a host value ([`DESCRIPTOR_WORD`]): a join takes it as
    /// undefined on the load's edge (`flow.rs`), a direct read stops.
    Dead,
    /// An image the program measures: a 2D texture, an unordered one where
    /// the entry says so.
    Image { uav: bool },
    /// A buffer the program tests by its record count, viewed as its entry
    /// says.
    Buffer { used: &'a UsedResource },
}

/// How many scalar ALU operations back [`Lower::same_scalar`] compares two
/// computed numbers.
const SAME_SCALAR_DEPTH: u32 = 8;

impl<'a> Lower<'a> {
    pub(super) fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        // Each result already has its own root-parameter binding in params.
        // get() materializes it on first use, including after an SGPR overwrite.
        if self.x.res.spill_loads.binary_search(&site).is_ok() {
            return Ok(());
        }
        if let Some(&read) = self.x.res.constant_reads.get(&site) {
            let h = self.constant_read_handle(read, true);
            for (k, &(reg, _)) in inst.writes.iter().enumerate() {
                let at = self.s.m.i32(read.offset + 4 * k as u32);
                let x = self.raw_words(h, at, 1)[0];
                self.write(inst, reg, x, None);
            }
            return Ok(());
        }
        let Fields::Smem {
            data,
            offset,
            soffset,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        // Through a descriptor that differs between lanes, lanes outside
        // EXEC skip the load (see `scalar_guard`) and read 0.
        let guard = if self.lane_descriptor(src) {
            self.scalar_guard(site.0, site.1)?
        } else {
            None
        };
        // The console reads the bytes the descriptor names (its base and
        // record count), whatever view the title made there. On the host a
        // slot holds one view, and a handle of another kind than the view's
        // is undefined: through a constant-buffer handle over a raw view,
        // d3e44e754b3eb6aa's loads lost the device (docs/analysis/hang.md),
        // and the meshlet culling (three programs, 3b471c122be79b3e among
        // them) read its records as nothing, so no meshlet passed (watch
        // item 147). So
        // a constant-buffer handle only where the slot is known to hold a
        // constant buffer: a CBV entry of the table, or a root descriptor
        // (the host binds what the translation declares there). Anything
        // else is read raw: an SRV or UAV entry (VIctory #1015 and #1055),
        // or no entry at all (a descriptor the program loads from the heap
        // at an index it computed, or from a table past its entries), where
        // the console compiler reads a raw or structured buffer at uniform
        // addresses this way. Unordered access where an access of the same
        // slot writes it (`slot_raw_kind`), as `buffer.rs` names it there.
        // shortcut: with no entry, a slot that holds a constant buffer view
        // is read raw too (VIctory #1409's rule), and WARP removes its
        // device on that, as on a constant-buffer handle over a raw or
        // structured view (`emit::smem_tests`). On a target with the
        // runtime stride table the raw read goes through the table's raw
        // view: such a slot holds no buffer view there and reads 0, which
        // the device survives, but not the constants. Reading them needs
        // the slot's kind at run time (a constant-buffer entry in the
        // table), or host slots of the title's constant buffers made as
        // raw views.
        let raw = match kind {
            Some(u) if u.kind == ResourceType::Srv => Some(11),
            Some(u) if u.kind == ResourceType::Uav => Some(11 | 0x1000),
            Some(_) => None,
            None if matches!(src, Source::Built { .. } | Source::Constant(_)) => None,
            None => Some(self.slot_raw_kind(src)),
        };
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let xs = self.guarded_load(guard, |l| {
            let c = l.cur;
            let mut xs = Vec::new();
            if let Some(kind) = raw {
                // The console ignores the two low bits of the address.
                let addr = match constant {
                    Some(byte) if byte & 3 == 0 => addr,
                    Some(byte) => l.s.m.i32(byte & !3),
                    None => {
                        let dword = l.s.m.i32(!3);
                        l.s.m.bin(c, Bin::And, addr, dword)
                    }
                };
                if l.raw_through_table(src) {
                    // Through the runtime stride table's raw view, as the
                    // slot's vector accesses by offset read it (`buffer.rs`):
                    // the slot's own view for a raw one, a raw view of the
                    // same bytes for a structured or typed one, whose slot
                    // may hold no view the host could make (the Codex review
                    // of rawfix, P1). Each dword past the view's bytes, or
                    // where the table names no buffer view, reads 0.
                    for k in 0..count {
                        let off = l.s.m.i32(4 * k);
                        let a = l.s.m.bin(l.cur, Bin::Add, addr, off);
                        let x = l.runtime_access(src, None, a, 4, kind, |l, h, byte| {
                            let c = l.cur;
                            let mask = l.s.m.i8(1);
                            let (u, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i32(4));
                            let r = l.s.op(
                                c,
                                "rawBufferLoad.f32",
                                Attr::ReadOnly,
                                l.s.t.res_ret,
                                139,
                                &[h, byte, u, mask, al],
                            );
                            let x = l.s.m.extract(c, r, 0);
                            Ok(vec![l.as_int(x)])
                        })?;
                        xs.push(x[0]);
                    }
                    return Ok(xs);
                }
                let h = l.handle(site, src, false, (kind, 0))?;
                // Past the view's end each dword reads 0 (D3D12's bounds
                // check of a buffer view, one dword per load), as on the
                // console past the record count: a host view made from the
                // title's covers the same bytes (`descriptor/host.rs`).
                for k in 0..count {
                    let off = l.s.m.i32(4 * k);
                    let a = l.s.m.bin(c, Bin::Add, addr, off);
                    let mask = l.s.m.i8(1);
                    let (u, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, a, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    xs.push(l.as_int(x));
                }
                return Ok(xs);
            }
            let h = l.handle(site, src, false, (13, 0x1_0000))?;
            let mut rows: HashMap<u32, V> = HashMap::new();
            for k in 0..count {
                let x = match constant {
                    Some(base) => {
                        let byte = base + 4 * k;
                        let row = match rows.get(&(byte >> 4)) {
                            Some(&r) => r,
                            None => {
                                let ri = l.s.m.i32(byte >> 4);
                                let r = l.s.op(
                                    c,
                                    "cbufferLoadLegacy.i32",
                                    Attr::ReadOnly,
                                    l.s.t.cbuf_ret,
                                    59,
                                    &[h, ri],
                                );
                                rows.insert(byte >> 4, r);
                                r
                            }
                        };
                        l.s.m.extract(c, row, (byte >> 2) & 3)
                    }
                    None => {
                        let off = l.s.m.i32(4 * k);
                        let a = l.s.m.bin(c, Bin::Add, addr, off);
                        l.cbuffer_dword(h, a)
                    }
                };
                xs.push(x);
            }
            Ok(xs)
        })?;
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count:
    /// elements of a structured or typed view, bytes of a raw one, as
    /// [`Self::record_count_view`] asks for them, or as the runtime stride
    /// table holds them where the buffer's accesses go through it).
    ///
    /// The view's kind comes from how the program uses the descriptor
    /// (its own access, or a twin load's), else from what the load is:
    /// [`Self::unused_descriptor`]. A constant buffer's size has no query:
    /// its dwords are data only where something other than its
    /// `s_buffer_load`s and joins with numbers reads them
    /// ([`Self::reused_after_access`]), and then the program stops.
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    pub(super) fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = DESCRIPTOR_WORD;
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // Dwords only accesses through the descriptor read: nothing to
        // rebuild, no query (through a per-lane index it would need a guard).
        let dwords: Vec<Option<Value>> = (0..count)
            .map(|k| inst.write(Reg::S(code + k as u8)))
            .collect();
        if !dwords.iter().flatten().any(|v| self.data_reads.contains(v)) {
            for v in dwords.into_iter().flatten() {
                self.vals.insert(v, Val::Missing(OTHER));
            }
            return Ok(());
        }
        // How the program uses the descriptor tells the view's kind: its
        // own uses, else those of a twin load of the same dwords (a
        // non-uniform index reloads the descriptor in each waterfall loop,
        // one load for the size bits, another for the access).
        let uses = |twin: &dyn Fn(Site) -> bool| {
            self.x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .find_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if twin(load) => Some((d.site, first)),
                        _ => None,
                    })
                })
        };
        let use_at =
            uses(&|load| load == site).or_else(|| uses(&|load| self.twin_loads(site, load)));
        // `per`: the stride the size is divided by (see `record_count_view`);
        // `table`: the record count is the runtime stride table's.
        let (first, src, props, image, uav, per, table) = match use_at {
            Some((at, first)) => {
                let src = Source::Loaded { load: site, first };
                let used = self.used_kind(&src);
                let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
                let un = ui.name();
                let uav = un.contains("store")
                    || un.contains("atomic")
                    || used.is_some_and(|u| u.kind == ResourceType::Uav);
                let u = if uav { 0x1000 } else { 0 };
                let ((props, per), image, table) = match &ui.fields {
                    Fields::Image { dim, .. } => {
                        let kind = match dim {
                            0 => 1,
                            1 => 2,
                            2 => 4,
                            4 => 6,
                            5 => 7,
                            _ => return Err("an image dimension (cube or MSAA)"),
                        };
                        (((kind | u, 0x409), None), Some(kind), false)
                    }
                    Fields::Buffer { format, idxen, .. } => {
                        if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                            return Err("a constant buffer's descriptor read as data");
                        }
                        if un.starts_with("buffer_load_format")
                            || un.starts_with("buffer_store_format")
                        {
                            // Typed: through the slot's own typed view.
                            (((10 | u, 0x409), None), None, false)
                        } else {
                            // Untyped, and typed by the instruction's format
                            // (`tbuffer_load_format_*`, which `buffer.rs`
                            // reads raw by offset): the count is the
                            // descriptor's, in elements where the entry says
                            // the view is structured, whatever instruction
                            // reads it (the Codex review of rawfix, P2: in
                            // bytes for MTBUF before).
                            let by_index = *idxen && format.is_none();
                            let table = self.counted_through_table(&src, by_index, used);
                            (Self::record_count_view(used, 11 | u), None, table)
                        }
                    }
                    Fields::Smem { .. } => {
                        // An `s_buffer_load`: the slot's kind as `smem` reads
                        // it (raw, unordered where the slot is written).
                        let raw = match used.map(|u| u.kind) {
                            Some(ResourceType::Srv) => 11,
                            Some(ResourceType::Uav) => 11 | 0x1000,
                            // A constant buffer whose registers the program
                            // gives to numbers after its last access, which
                            // joins take (r1smem's group B: a CBV table
                            // entry the console compiler reads uniformly):
                            // no dword is read as data (`smem_data.rs`).
                            Some(_) if self.reused_after_access(site, &dwords) => {
                                for v in dwords.into_iter().flatten() {
                                    self.vals.insert(v, Val::Missing(OTHER));
                                }
                                return Ok(());
                            }
                            Some(_) => return Err("a constant buffer's descriptor read as data"),
                            None => self.slot_raw_kind(&src),
                        };
                        let table = self.raw_through_table(&src);
                        (Self::record_count_view(used, raw), None, table)
                    }
                    _ => return Err("a scalar memory read as data"),
                };
                let uav = uav || props.0 & 0x1000 != 0;
                (first, src, props, image, uav, per, table)
            }
            None => match self.unused_descriptor(site, count, &dwords)? {
                Unused::Dead => {
                    for v in dwords.into_iter().flatten() {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                    return Ok(());
                }
                Unused::Image { uav } => {
                    let src = Source::Loaded {
                        load: site,
                        first: 0,
                    };
                    let u = if uav { 0x1000 } else { 0 };
                    (0, src, (2 | u, 0x409), Some(2), uav, None, false)
                }
                Unused::Buffer { used } => {
                    let src = Source::Loaded {
                        load: site,
                        first: 0,
                    };
                    let uav = used.kind == ResourceType::Uav;
                    let u = if uav { 0x1000 } else { 0 };
                    let ((props, per), table) = (
                        Self::record_count_view(Some(used), 11 | u),
                        self.raw_through_table(&src),
                    );
                    (0, src, props, None, uav, per, table)
                }
            },
        };
        // A heap index per lane where the load's offset is a waterfall
        // loop's first-lane read (each lane's own value) or the load is
        // the loop's own: lanes outside EXEC skip the query then (see
        // `scalar_guard`).
        let own = !self.x.res.waterfall.loads.contains(&site) && self.reads_lane_own(site);
        let guard = if own || self.lane_descriptor(&src) {
            self.scalar_guard(site.0, site.1)?
        } else {
            None
        };
        let rels: &[u32] = if image.is_some() { &[1, 2, 3, 4] } else { &[2] };
        debug_assert!(!table || image.is_none(), "a table count is a buffer's");
        let xs = self.guarded_load(guard, |l| {
            if table {
                // The count the table's range checks use (`runtime_view`),
                // for the view its raw view holds, whatever the slot's own
                // view is or whether the host could make one there (the
                // Codex review of rawfix, P1: a slot holding no view read 0
                // records, a stride known only at run time gave bytes):
                // elements of a structured or typed view, dwords of a raw
                // one, which the console counts in bytes.
                // shortcut: a raw view of a size off a multiple of 4 counts
                // its whole dwords only (the table's count).
                let e = l.stride_entry(&src)?;
                let c = l.cur;
                let (zero, two) = (l.s.m.i32(0), l.s.m.i32(2));
                let raw = l.s.m.cmp(c, pred::EQ, e.stride, zero);
                let bytes = l.s.m.bin(c, Bin::Shl, e.count, two);
                return Ok(vec![l.s.m.select(c, raw, bytes, e.count)]);
            }
            let h = if own {
                let index = l.heap_index(site, first, false)?;
                l.s.heap_handle(l.cur, index, false, true, props)
            } else {
                l.handle(site, &src, false, props)?
            };
            let c = l.cur;
            let i32t = l.s.t.i32;
            let dims = l.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let mip = if image.is_some() && !uav {
                l.s.m.i32(0)
            } else {
                l.s.m.undef(i32t)
            };
            let d =
                l.s.op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
            let Some(kind) = image else {
                let size = field(l, 0);
                let Some(stride) = per else {
                    return Ok(vec![size]);
                };
                // The raw view's bytes over the stride: whole elements, the
                // host's raw view holding the structured one's bytes only.
                let stride = l.s.m.i32(stride);
                return Ok(vec![l.s.m.bin(c, Bin::UDiv, size, stride)]);
            };
            let one = l.s.m.i32(1);
            let w = field(l, 0);
            let (height, depth) = match kind {
                1 => (one, one),
                2 => (field(l, 1), one),
                6 => (one, field(l, 1)),
                _ => (field(l, 1), field(l, 2)),
            };
            let levels = if uav { one } else { field(l, 3) };
            let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
            let (w, height, depth, levels) =
                (less(l, w), less(l, height), less(l, depth), less(l, levels));
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let (k30, k2, k14, k16, mask) = (k(l, 30), k(l, 2), k(l, 14), k(l, 16), k(l, 0x3FFF));
            let d1 = l.s.m.bin(c, Bin::Shl, w, k30);
            let hi = l.s.m.bin(c, Bin::LShr, w, k2);
            let hi = l.s.m.bin(c, Bin::And, hi, mask);
            let ht = l.s.m.bin(c, Bin::Shl, height, k14);
            let d2 = l.s.m.bin(c, Bin::Or, hi, ht);
            let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
            let base = k(l, 0xFAC | ty << 28);
            let last = l.s.m.bin(c, Bin::Shl, levels, k16);
            let d3 = l.s.m.bin(c, Bin::Or, base, last);
            Ok(vec![d1, d2, d3, depth])
        })?;
        let words: Vec<(u32, V)> = rels.iter().copied().zip(xs).collect();
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// The handle a buffer's record count (descriptor dword 2) is asked
    /// through, for an untyped access, an MTBUF one or none (`raw`: the raw
    /// kind, 11, with the unordered-access bit where the access has it),
    /// and the stride the size it gives is divided by. The console's count
    /// is in elements for a structured view (what a used-resource entry of
    /// a stride of 4 bytes or more says the slot holds), whatever the
    /// access: by index, by byte offset (`offen`), in a format the
    /// instruction names or by a scalar load; in bytes for a raw one.
    ///
    /// The host's view of a structured slot is structured, where a
    /// structured handle gives the elements; with `XTP_RAW_STRUCTURED=1`
    /// ([`crate::emit::raw_structured`]) it is a raw view of the same bytes,
    /// which cannot keep the title's count (its elements are dwords), and a
    /// structured handle over it gives other numbers on NVIDIA (VIctory
    /// #1762, an RTX 5080: 64 for 16 elements of 16 bytes through a shader
    /// view, 256 through an unordered one). So the size is asked through a
    /// raw handle, in bytes, and divided by the entry's stride.
    // shortcut: with `XTP_RAW_STRUCTURED=1`, a structured view the host
    // cannot make raw (bytes off 16 in its copy, `descriptor/host.rs`)
    // stays structured, and a raw handle's size of it is undefined, as are
    // the raw reads of it.
    // Ported from marquezinii/VIctory@f6bb5d5 by marquezinii (the raw
    // handle and the division) and AlanQiang0418/VIctory@b86d853 by
    // AlanQiang0418 (the stride for accesses by byte offset too).
    fn record_count_view(used: Option<&UsedResource>, raw: u32) -> ((u32, u32), Option<u32>) {
        match used.map_or(0, |u| u.stride) {
            s if s >= 4 && s.is_multiple_of(4) => {
                if crate::emit::raw_structured() {
                    ((raw, 0), Some(s))
                } else {
                    (((raw & !0xFF) | 12, s), None)
                }
            }
            _ => ((raw, 0), None),
        }
    }

    /// Whether the accesses of `src`'s buffer that a record count sizes go
    /// through the runtime stride table, as `buffer.rs` routes them: by
    /// offset alone (`by_index` false: an untyped or MTBUF access by
    /// offset, a scalar load) for a view-heap descriptor on a target with
    /// the table; by element where the stride is not known here, or, with
    /// `XTP_RAW_STRUCTURED=1`, a known one (`buffer.rs`'s
    /// `runtime_stride`; a stride it refuses gives no access to size).
    fn counted_through_table(
        &self,
        src: &Source,
        by_index: bool,
        used: Option<&UsedResource>,
    ) -> bool {
        if !by_index {
            return self.raw_through_table(src);
        }
        self.x.target.stride_table
            && match used.map_or(0, |u| u.stride) {
                s if s >= 4 && s.is_multiple_of(4) => crate::emit::raw_structured(),
                0 => true,
                _ => false,
            }
    }

    /// What a descriptor load is that no access uses as a descriptor (nor
    /// a twin load's), whose dwords `dwords` the code reads all the same,
    /// told by the load itself and the used-resource entry it falls in:
    /// - a sampler's descriptor (its load, or a twin's, gives an access its
    ///   sampler): no program reads its dwords as numbers, so where one
    ///   meets data at a join the register is dead on that edge, as a
    ///   resource descriptor's address or format is ([`Unused::Dead`]).
    ///   The material programs `f23ac5187c49056a`, `f6e7139e15a906d8` and
    ///   `578e2ba641dbe64e` join a sampler's dword 1 with a constant at a
    ///   branch whose outcome does not depend on it on that edge;
    /// - eight dwords whose width, height or levels (dwords 1 to 3) the
    ///   code reads, and not depth or layers (dword 4), with no entry or an
    ///   eight-dword SRV or UAV one: an image the program measures, taken
    ///   as a 2D texture. The puddle detection (`6b6db9f1a526daeb`, 35
    ///   programs of the pack) turns the width, height and base level of a
    ///   mask it picks from the heap by an index in a constant buffer into
    ///   a texel size, and samples the masks it picks beside it as 2D ones;
    /// - four dwords of a four-dword SRV or UAV entry: a buffer, its record
    ///   count from the entry's view (`ceb5eb68cd99cb07` skips its pass
    ///   when the buffer is empty), its other dwords dead as above. With
    ///   no entry it stops: nothing says the slot holds a buffer view.
    ///
    /// Anything else stops the program ("a descriptor read only as data"):
    /// a handle would declare a view of a kind nothing says the slot holds.
    // shortcut: a 2D texture's handle over a heap slot holding an image of
    // another dimension (an array, 3D, a cube) is undefined in D3D12.
    // Ported from fxceless's VIctory fork #203 (0e49a91) by triippywhale
    // (the image and buffer of a descriptor nothing uses); the sampler's
    // dwords dead at a join reimplemented from fxceless's VIctory fork
    // (578d459, `descriptor_data`) by fxceless.
    fn unused_descriptor(
        &self,
        site: Site,
        count: u32,
        dwords: &[Option<Value>],
    ) -> Result<Unused<'a>, Why> {
        let sampler = self.x.res.descriptors.iter().any(|d| {
            d.role == Role::Sampler
                && d.source.leaves().into_iter().any(|l| {
                    matches!(*l, Source::Loaded { load, .. }
                        if load == site || self.twin_loads(site, load))
                })
        });
        if sampler {
            return Ok(Unused::Dead);
        }
        let read = |k: usize| {
            dwords
                .get(k)
                .copied()
                .flatten()
                .is_some_and(|v| self.data_reads.contains(&v))
        };
        let used = self.used_kind(&Source::Loaded {
            load: site,
            first: 0,
        });
        let view = |u: &UsedResource, dwords: u8| {
            matches!(u.kind, ResourceType::Srv | ResourceType::Uav) && u.dwords == dwords
        };
        match (count, used) {
            (8, _) if !used.is_none_or(|u| view(u, 8)) => Err(ONLY_DATA),
            (8, _) if (1..4).any(read) && !read(4) => Ok(Unused::Image {
                uav: used.is_some_and(|u| u.kind == ResourceType::Uav),
            }),
            // An image's address and tiling (dwords 0, 5 to 7) alone.
            (8, Some(_)) if !(1..5).any(read) => Ok(Unused::Dead),
            (4, Some(u)) if view(u, 4) && read(2) => Ok(Unused::Buffer { used: u }),
            (4, Some(u)) if view(u, 4) => Ok(Unused::Dead),
            _ => Err(ONLY_DATA),
        }
    }

    /// Two scalar loads of the same dwords: same opcode and immediate
    /// offset, base read from the same values, and the same or an
    /// equivalent register offset.
    fn twin_loads(&self, a: Site, b: Site) -> bool {
        let (ia, ib): (&Inst, &Inst) = (
            &self.x.p.blocks[a.0].insts[a.1],
            &self.x.p.blocks[b.0].insts[b.1],
        );
        let (
            Fields::Smem {
                base: Operand::Scalar { code: ba, .. },
                offset: oa,
                soffset: sa,
                ..
            },
            Fields::Smem {
                base: Operand::Scalar { code: bb, .. },
                offset: ob,
                soffset: sb,
                ..
            },
        ) = (&ia.fields, &ib.fields)
        else {
            return false;
        };
        let same_base = (0..2).all(|k| {
            let va = ia.read(Reg::S(ba + k));
            va.is_some() && va == ib.read(Reg::S(bb + k))
        });
        let same_soffset = match (sa, sb) {
            (Operand::Scalar { code: NULL, .. }, Operand::Scalar { code: NULL, .. }) => true,
            (Operand::Scalar { code: ca, .. }, Operand::Scalar { code: cb, .. })
                if *ca != NULL && *cb != NULL =>
            {
                match (ia.read(Reg::S(*ca)), ib.read(Reg::S(*cb))) {
                    (Some(x), Some(y)) => self.same_scalar(x, y),
                    _ => false,
                }
            }
            _ => false,
        };
        ia.name() == ib.name() && oa == ob && same_base && same_soffset
    }

    /// The raw view kind of a slot no entry describes, read by a scalar
    /// load through descriptor `src`: 11, with 0x1000 (unordered access)
    /// where a vector access of the same slot writes it (a store or an
    /// atomic, which `buffer.rs` gives an unordered-access handle).
    fn slot_raw_kind(&self, src: &Source) -> u32 {
        let writes = self.x.res.descriptors.iter().any(|d| {
            let inst: &Inst = &self.x.p.blocks[d.site.0].insts[d.site.1];
            let name = inst.name();
            d.role == Role::Resource
                && matches!(inst.fields, Fields::Buffer { .. })
                && (name.contains("store") || name.contains("atomic"))
                && self.same_slot(&d.source, src)
        });
        if writes { 11 | 0x1000 } else { 11 }
    }

    /// Whether descriptors `a` and `b` can be the same slot: a source one
    /// of them picks from (a phi or a choice: any of them, since the
    /// slot's view is what the handles must agree with) is the other's,
    /// or is loaded from the same address, whatever the width of the loads
    /// (`same_address`).
    fn same_slot(&self, a: &Source, b: &Source) -> bool {
        let others = b.leaves();
        a.leaves()
            .into_iter()
            .any(|x| others.iter().any(|&y| x == y || self.same_address(x, y)))
    }

    /// Whether two descriptor loads read the same dwords: the same root
    /// parameter, the same byte offset of their first dword (the load's
    /// immediate and constant terms, plus 4 per dword the descriptor skips)
    /// and the same computed term (`same_scalar`).
    fn same_address(&self, a: &Source, b: &Source) -> bool {
        let (
            Source::Loaded {
                load: la,
                first: fa,
            },
            Source::Loaded {
                load: lb,
                first: fb,
            },
        ) = (a, b)
        else {
            return false;
        };
        let (Some(x), Some(y)) = (self.scalar.get(la), self.scalar.get(lb)) else {
            return false;
        };
        let start = |a: &Address, first: u8| a.offset.wrapping_add(4 * u32::from(first));
        x.param == y.param
            && x.param != CODE
            && start(x, *fa) == start(y, *fb)
            && match (x.dynamic, y.dynamic) {
                (None, None) => true,
                (Some(p), Some(q)) => self.same_scalar(p, q),
                _ => false,
            }
    }

    /// The instruction that writes `v`.
    fn def_of(&self, v: Value) -> Option<&'a Inst> {
        let p = self.x.p;
        let &b = self.def_block.get(&v)?;
        p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
    }

    /// Whether `x` and `y` hold the same number: the same value, two
    /// `v_readfirstlane_b32` of the same VGPR value (each waterfall loop
    /// reads the index anew), or the same dword of two scalar ALU results
    /// of one operation on such numbers (a heap offset the program shifts
    /// again before each load of the slot).
    fn same_scalar(&self, x: Value, y: Value) -> bool {
        self.same_scalar_within(x, y, SAME_SCALAR_DEPTH)
    }

    fn same_scalar_within(&self, x: Value, y: Value, depth: u32) -> bool {
        if x == y {
            return true;
        }
        let (Some(a), Some(b)) = (self.def_of(x), self.def_of(y)) else {
            return false;
        };
        if a.name() != b.name() {
            return false;
        }
        if a.name() == "v_readfirstlane_b32" {
            let lane = |i: &Inst| {
                i.reads
                    .iter()
                    .find(|r| matches!(r.0, Reg::V(_)))
                    .map(|r| r.1)
            };
            return matches!((lane(a), lane(b)), (Some(p), Some(q)) if p == q);
        }
        let (
            Fields::Sop {
                src: sa, imm: ia, ..
            },
            Fields::Sop {
                src: sb, imm: ib, ..
            },
        ) = (&a.fields, &b.fields)
        else {
            return false;
        };
        let at = |i: &Inst, v: Value| i.writes.iter().position(|w| w.1 == v);
        if depth == 0 || ia != ib || sa.len() != sb.len() || at(a, x) != at(b, y) {
            return false;
        }
        let same = |p: Option<Value>, q: Option<Value>| match (p, q) {
            (Some(p), Some(q)) => self.same_scalar_within(p, q, depth - 1),
            (p, q) => p.is_none() && q.is_none(),
        };
        let operands = sa.iter().zip(sb).all(|(p, q)| match (*p, *q) {
            (Operand::Scalar { code: c, dwords: n }, Operand::Scalar { code: d, dwords: m })
                if n == m =>
            {
                (0..n).all(|k| {
                    let (u, w) = (a.read(Reg::S(c + k)), b.read(Reg::S(d + k)));
                    // `null`, which no value stands for, only matches itself.
                    same(u, w) && (u.is_some() || c == d)
                })
            }
            (Operand::Scalar { .. }, _) | (_, Operand::Scalar { .. }) => false,
            _ => p == q,
        });
        operands && same(a.read(Reg::Scc), b.read(Reg::Scc))
    }

    /// Whether the scalar load at `site` reads its base or offset from a
    /// waterfall loop's first-lane read, which is each lane's own value.
    fn reads_lane_own(&self, site: Site) -> bool {
        let inst = &self.x.p.blocks[site.0].insts[site.1];
        inst.reads
            .iter()
            .any(|r| self.x.res.waterfall.reads.contains(&r.1))
    }

    /// Whether `v` is a dword a scalar load writes whose dwords the program
    /// uses as a resource or sampler descriptor.
    pub(super) fn descriptor_dword(&self, v: Value) -> bool {
        let Some(&b) = self.def_block.get(&v) else {
            return false;
        };
        let Some(i) = self.x.p.blocks[b]
            .insts
            .iter()
            .position(|i| i.writes.iter().any(|w| w.1 == v))
        else {
            return false;
        };
        self.x.p.blocks[b].insts[i]
            .name()
            .starts_with("s_load_dword")
            && self.x.res.descriptors.iter().any(|d| {
                d.source
                    .leaves()
                    .iter()
                    .any(|l| matches!(l, Source::Loaded { load, .. } if *load == (b, i)))
            })
    }

    /// `s_load_dword*` from the program's constant table.
    pub(super) fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        let c = self.cur;
        let mut addr = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            addr = self.s.m.bin(c, Bin::Add, addr, d);
        }
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    pub(super) fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// The dword at a computed byte address of a constant buffer.
    pub(super) fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Lowered, Parts, lower};
    use crate::emit::tests::{compute, emit_for};
    use crate::native::{ComputeIo, ResourceType, RootParam, RootType, StageIo, UsedResource};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn entry(
        kind: ResourceType,
        table_offset_dwords: u32,
        dwords: u8,
        stride: u32,
    ) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords,
            user_data: 0,
            slot: 0,
        }
    }

    /// A wave32 compute program with a table at s0 holding the entries
    /// `used`, and 512 bytes of LDS.
    fn program(words: &[u32], used: Vec<UsedResource>) -> crate::NativeProgram {
        let mut p = compute(words, false);
        p.layout = vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        p.used = used;
        p.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 1 << 1,
            threads: [64, 1, 1],
        });
        p
    }

    /// [`program`] with one entry: `kind` descriptors of `dwords` dwords
    /// (`stride` for buffers).
    fn translate(
        words: &[u32],
        kind: ResourceType,
        dwords: u8,
        stride: u32,
    ) -> Result<(), crate::ShaderError> {
        let p = program(words, vec![entry(kind, 0, dwords, stride)]);
        emit_for(&p, crate::emit::Target::ANY).map(|_| ())
    }

    /// The lowering (per lane, for any host) of [`program`] in groups of 32.
    fn lowered(words: &[u32], used: Vec<UsedResource>) -> Result<Lowered, crate::ShaderError> {
        let mut native = program(words, used);
        native.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        });
        let p = ir::build(&native.code, false, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        lower(&Parts {
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
    }

    fn only_data(r: Result<(), crate::ShaderError>) -> bool {
        matches!(
            r,
            Err(crate::ShaderError::Unsupported {
                what: "a descriptor read only as data",
                ..
            })
        )
    }

    /// An image the program only measures: its height bits from an eight-
    /// dword load, no access. Taken as a 2D texture where the entry is an
    /// eight-dword SRV or UAV, or where no entry says what it is (fork
    /// #203's rule, `unused_descriptor`); not where the entry says it is a
    /// four-dword descriptor (two buffers loaded at once). The values on
    /// the host are `emit::desc_data_tests`'.
    #[test]
    fn an_image_only_measured_is_taken_as_a_2d_texture() {
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x9010_8E0A, // s_lshr_b32 s16, s10, 14 (height - 1)
            0x7E02_0210, // v_mov_b32 v1, s16
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        for kind in [ResourceType::Srv, ResourceType::Uav] {
            assert_eq!(translate(&words, kind, 8, 0), Ok(()), "{kind:?}");
        }
        assert!(only_data(translate(&words, ResourceType::Srv, 4, 0)));
        let p = program(&words, vec![]);
        assert_eq!(emit_for(&p, crate::emit::Target::ANY).map(|_| ()), Ok(()));
    }

    /// The console compiler's waterfall over a buffer descriptor (as
    /// `resource::tests::WATERFALL_INDEXED_WAVE32` over an image): each
    /// lane's entry at byte (lane & 1) * 32 of the table at s0, loaded into
    /// s[12:15] at the first active lane's offset (s20) for a structured
    /// load, and again into s[28:31] at `twin_offset` for its record count
    /// (s30, dword 2), which each lane adds to what it loaded; the result
    /// goes to the buffer at dword 16 of the table. `first_lane_again`
    /// reads the offset into s22 a second time first.
    fn waterfall_twin(first_lane_again: bool) -> Vec<u32> {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x360E_0081, // v_and_b32 v7, 1, v0
            0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE97_037E, // s_mov_b32 s23, exec_lo
            0xBE95_037E, // s_mov_b32 s21, exec_lo
            0x7E28_0507, // 0x28: v_readfirstlane_b32 s20, v7
            0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        ];
        if first_lane_again {
            words.push(0x7E2C_0507); // v_readfirstlane_b32 s22, v7
        }
        let twin = if first_lane_again {
            0x2C00_0000 // s22
        } else {
            0x2800_0000 // s20
        };
        words.extend([
            0xF408_0302,
            0x2800_0000, // s_load_dwordx4 s[12:15], s[4:5], s20
            0xF408_0702,
            twin,        // s_load_dwordx4 s[28:31], s[4:5], s20 or s22
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_2000,
            0x8003_0C00, // buffer_load_dword v12, v0, s[12:15], 0 idxen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A18_181E, // v_add_nc_u32 v12, s30, v12
            0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
            if first_lane_again {
                0xBF89_FFF2 // s_cbranch_execnz -> 0x28
            } else {
                0xBF89_FFF3 // s_cbranch_execnz -> 0x28
            },
            0xBEFE_0317, // s_mov_b32 exec_lo, s23
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ]);
        words
    }

    fn waterfall_entries() -> Vec<UsedResource> {
        vec![
            entry(ResourceType::Srv, 0, 4, 4),
            entry(ResourceType::Uav, 16, 4, 0),
        ]
    }

    /// The twin load reads its offset from the loop's first-lane read, so
    /// each lane has its own descriptor there: its record count comes from
    /// a heap index per lane, and the lanes outside EXEC (those whose
    /// descriptor another round takes) skip that query, as they skip the
    /// load through the loop's own descriptor.
    #[test]
    fn a_twin_with_a_lanes_own_offset_is_queried_only_in_exec() {
        let l = lowered(&waterfall_twin(false), waterfall_entries()).unwrap();
        // The structured load and the twin's query (the loop's own load is
        // read by the access only: no query).
        assert_eq!(l.nonuniform_handles, 2);
        assert_eq!(l.scalar_guards, 1);
    }

    /// The twin reads the offset with a second `v_readfirstlane_b32` of the
    /// same VGPR (each waterfall loop reads the index anew): it is still
    /// the loop's load, and takes its view. Its own offset is no compared
    /// first-lane read, so its query is the whole wave's.
    #[test]
    fn a_twin_reading_the_first_lane_again_takes_the_loads_view() {
        let l = lowered(&waterfall_twin(true), waterfall_entries()).unwrap();
        assert_eq!(l.nonuniform_handles, 1);
        assert_eq!(l.scalar_guards, 0);
    }

    /// A buffer descriptor loaded twice, the first load used by a
    /// structured load, the second read for its record count (dword 2):
    /// the second takes the first's view. A load of another descriptor of
    /// the range is no twin: its count comes from the view its entry says
    /// the range holds (`unused_descriptor`), and with no entry it stops
    /// (`emit::desc_data_tests`).
    #[test]
    fn a_descriptor_read_as_data_takes_the_view_of_its_twin_load() {
        let twin = |offset: u32| {
            [
                0xBE84_0300, // s_mov_b32 s4, s0
                0xBE85_0380, // s_mov_b32 s5, 0
                0xF408_0202,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
                0xF408_0302,
                0xFA00_0000 | offset, // s_load_dwordx4 s[12:15], s[4:5], offset
                0xBF8C_C07F,          // s_waitcnt lgkmcnt(0)
                0x7E02_020E,          // v_mov_b32 v1, s14 (record count)
                0xE030_2000,
                0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen
                0xBF8C_3F70, // s_waitcnt vmcnt(0)
                0x4A04_0501, // v_add_nc_u32 v2, v1, v2
                0xD834_0000,
                0x0000_0200, // ds_write_b32 v0, v2
                S_ENDPGM,
            ]
        };
        assert_eq!(translate(&twin(0), ResourceType::Srv, 4, 4), Ok(()));
        assert_eq!(translate(&twin(0x10), ResourceType::Srv, 4, 4), Ok(()));
        // An entry of another kind than a buffer's (eight dwords: images).
        assert!(only_data(translate(&twin(0x10), ResourceType::Srv, 8, 4)));
    }

    /// d3e44e754b3eb6aa's shape (docs/analysis/hang.md), in groups of 32:
    /// the table at s0 holds a raw buffer whose dword 0 is a heap index, and
    /// the program loads the descriptor at that index of the resource heap
    /// (s1). It reads that buffer with a scalar load at byte 4 and with a
    /// vector load at byte 4 x lane, and lane t stores the sum at dword t
    /// of the UAV at table entry 8.
    fn heap_slot_read_two_ways() -> crate::NativeProgram {
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0304,
            0xFA00_0000, // s_buffer_load_dword s12, s[8:11], null (the index)
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x8F0C_850C, // s_lshl_b32 s12, s12, 5
            0xBE82_0301, // s_mov_b32 s2, s1
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0401,
            0x1800_0000, // s_load_dwordx4 s[16:19], s[2:3], s12
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0508,
            0xFA00_0004, // s_buffer_load_dword s20, s[16:19], 0x4
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xE030_1000,
            0x8004_0201, // buffer_load_dword v2, v1, s[16:19], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A04_0414, // v_add_nc_u32 v2, s20, v2
            0xF408_0602,
            0xFA00_0020, // s_load_dwordx4 s[24:27], s[4:5], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_1000,
            0x8006_0201, // buffer_store_dword v2, v1, s[24:27], 0 offen
            S_ENDPGM,
        ];
        let mut p = program(
            &words,
            vec![
                entry(ResourceType::Srv, 0, 4, 1),
                entry(ResourceType::Uav, 8, 4, 1),
            ],
        );
        p.layout.push(RootParam {
            root: RootType::ResourceHeap,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 1,
            slot: 0,
        });
        p.io = StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [32, 1, 1],
        });
        p
    }

    /// A heap slot reached by an index the program reads is read raw by
    /// its scalar loads too: one kind of view for the slot, the raw buffer
    /// the title made there. Through the constant-buffer handle they took
    /// before, WARP removes its device ("suspended"); on the host GPU it is
    /// undefined (the Series S world hang).
    #[test]
    fn a_heap_slot_read_by_scalar_and_vector_loads_is_read_raw_both_ways() {
        let cs = emit_for(&heap_slot_read_two_ways(), crate::emit::Target::ANY).unwrap();
        #[cfg(windows)]
        {
            use crate::emit::host::{Dispatch, Host, Root, Stride, View};
            // One GPU test at a time (see `valu_tests::ONE_AT_A_TIME`):
            // several WARP devices at once end the test process.
            let _one = crate::emit::valu_tests::ONE_AT_A_TIME
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let Ok(host) = Host::new(true) else {
                eprintln!("no WARP device; skipped");
                return;
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            // Heap slot 0: the table's raw buffer, holding the index 2.
            let index = [2u32, 0, 0, 0];
            let data: Vec<u32> = (0..64).map(|t| 1000 + t).collect();
            let views = [
                View::Buffer {
                    data: &index,
                    stride: Stride::Raw,
                },
                View::RawUav(32),
                View::Buffer {
                    data: &data,
                    stride: Stride::Raw,
                },
            ];
            let user = vec![0; (cs.user_data_dwords() as usize).max(2)];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let want: Vec<u32> = (0..32).map(|t| data[t] + data[1]).collect();
            assert_eq!(out[0], want);
        }
    }

    /// A UAV entry of the table read by a scalar load goes through a raw
    /// unordered-access handle, not a constant-buffer one: every lane
    /// stores 7 at dword 1 of the UAV, reads dword 1 back with a scalar
    /// load, and stores it plus its lane number at dword 2 + lane.
    #[test]
    fn a_uav_entry_read_by_a_scalar_load_is_read_raw() {
        let words = [
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE84_0300, // s_mov_b32 s4, s0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0x7E02_0287, // v_mov_b32 v1, 7
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_0004,
            0x8002_0100, // buffer_store_dword v1, off, s[8:11], 0 offset:4
            0xF420_0304,
            0xFA00_0004, // s_buffer_load_dword s12, s[8:11], 0x4
            0x3404_0082, // v_lshlrev_b32 v2, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x4A06_000C, // v_add_nc_u32 v3, s12, v0
            0xE070_1008,
            0x8002_0302, // buffer_store_dword v3, v2, s[8:11], 0 offen offset:8
            S_ENDPGM,
        ];
        let mut p = program(&words, vec![entry(ResourceType::Uav, 0, 4, 1)]);
        p.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        });
        let cs = emit_for(&p, crate::emit::Target::ANY).unwrap();
        #[cfg(windows)]
        {
            use crate::emit::host::{Dispatch, Host, Root, View};
            // One GPU test at a time (see `valu_tests::ONE_AT_A_TIME`):
            // several WARP devices at once end the test process.
            let _one = crate::emit::valu_tests::ONE_AT_A_TIME
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let Ok(host) = Host::new(true) else {
                eprintln!("no WARP device; skipped");
                return;
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let user = vec![0; (cs.user_data_dwords() as usize).max(1)];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user))],
                        views: &[View::RawUav(34)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let want: Vec<u32> = [0, 7].into_iter().chain(7..39).collect();
            assert_eq!(out[0], want);
        }
    }
}
