//! Title query heaps and their host occlusion and pipeline-statistics queries.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;
#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;
#[cfg(windows)]
use windows::core::Interface;

#[cfg(windows)]
#[derive(Default)]
pub(crate) struct Active(Vec<(usize, u32, bool)>);

#[cfg(windows)]
impl Active {
    pub(crate) fn dropped_draw(&mut self) {
        for (_, _, dropped) in &mut self.0 {
            *dropped = true;
        }
    }
}

#[cfg(windows)]
pub(crate) struct HostHeap {
    heap: ID3D12QueryHeap,
    // Le GPU écrit le résultat à EndQuery. Deux listes peuvent être enregistrées
    // dans un ordre différent de celui de leur soumission.
    occlusion: Option<(ID3D12Resource, ID3D12Resource)>,
}

#[cfg(windows)]
impl HostHeap {
    pub(crate) fn begin(
        &self,
        list: &crate::host::HostList,
        active: &mut Active,
        kind: u32,
        index: u32,
    ) {
        if self.occlusion.is_some() {
            active.0.push((self.heap.as_raw() as usize, index, false));
        }
        list.keep(&self.heap);
        // SAFETY: le tas et la plage ont été validés.
        unsafe {
            list.list
                .BeginQuery(&self.heap, D3D12_QUERY_TYPE(kind as i32), index)
        };
    }

    pub(crate) fn end(
        &self,
        list: &crate::host::HostList,
        active: &mut Active,
        kind: u32,
        index: u32,
    ) {
        let ty = D3D12_QUERY_TYPE(kind as i32);
        list.keep(&self.heap);
        // SAFETY: même tas et même indice qu’à l’ouverture.
        unsafe { list.list.EndQuery(&self.heap, ty, index) };
        let Some((results, visible)) = &self.occlusion else {
            return;
        };
        let dropped = active
            .0
            .iter()
            .position(|&(h, i, _)| h == self.heap.as_raw() as usize && i == index)
            .is_some_and(|pos| active.0.swap_remove(pos).2);
        list.keep(results);
        let at = u64::from(index) * 8;
        // SAFETY: results est privé au tas et promu en COPY_DEST à la première écriture.
        unsafe {
            if dropped {
                #[cfg(feature = "diagnostics")]
                crate::list::cuts::VISIBLE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                list.keep(visible);
                list.list.CopyBufferRegion(results, at, visible, 0, 8);
            } else {
                list.list
                    .ResolveQueryData(&self.heap, ty, index, 1, results, at);
            }
        }
    }
}

#[cfg(windows)]
#[derive(PartialEq, Eq)]
enum Source {
    Occlusion(ID3D12Resource),
    Statistics(ID3D12QueryHeap),
}

#[cfg(windows)]
struct ResolveBatch {
    source: Source,
    dst: ID3D12Resource,
    ranges: Vec<(u32, u32, u64)>,
}

#[cfg(windows)]
impl ResolveBatch {
    fn record(&self, list: &crate::host::HostList) {
        use crate::host::{transition, uav_barrier};
        let common = D3D12_RESOURCE_STATE_COMMON;
        let write = D3D12_RESOURCE_STATE_COPY_DEST;
        let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let mut before = vec![uav_barrier(), transition(&self.dst, common, write)];
        let mut after = vec![transition(&self.dst, write, common)];
        match &self.source {
            Source::Occlusion(results) => {
                list.keep(results);
                before.push(transition(results, write, read));
                after.push(transition(results, read, write));
            }
            Source::Statistics(heap) => list.keep(heap),
        }
        // SAFETY: chaque plage a été validée et la liste retient les ressources.
        unsafe {
            list.list.ResourceBarrier(&before);
            for &(first, count, at) in &self.ranges {
                match &self.source {
                    Source::Occlusion(results) => list.list.CopyBufferRegion(
                        &self.dst,
                        at,
                        results,
                        u64::from(first) * 8,
                        u64::from(count) * 8,
                    ),
                    Source::Statistics(heap) => list.list.ResolveQueryData(
                        heap,
                        D3D12_QUERY_TYPE_PIPELINE_STATISTICS,
                        first,
                        count,
                        &self.dst,
                        at,
                    ),
                }
            }
            list.list.ResourceBarrier(&after);
        }
    }
}

#[cfg(windows)]
#[derive(Default)]
pub(crate) struct Resolves {
    pending: std::sync::atomic::AtomicBool,
    batch: std::sync::Mutex<Option<ResolveBatch>>,
}

#[cfg(windows)]
impl Resolves {
    pub(crate) fn push(
        &self,
        list: &crate::host::HostList,
        heap: &HostHeap,
        first: u32,
        count: u32,
        dst: &ID3D12Resource,
        at: u64,
    ) {
        use std::sync::atomic::Ordering;
        let (source, stride) = match &heap.occlusion {
            Some((results, _)) => (Source::Occlusion(results.clone()), 8),
            None => (Source::Statistics(heap.heap.clone()), 88),
        };
        let mut slot = self.batch.lock().unwrap_or_else(|e| e.into_inner());
        if slot
            .as_ref()
            .is_some_and(|b| b.source != source || b.dst != *dst)
            && let Some(batch) = slot.take()
        {
            batch.record(list);
        }
        let batch = slot.get_or_insert_with(|| ResolveBatch {
            source,
            dst: dst.clone(),
            ranges: Vec::new(),
        });
        if let Some((start, n, offset)) = batch.ranges.last_mut()
            && *start + *n == first
            && *offset + u64::from(*n) * stride == at
        {
            *n += count;
        } else {
            batch.ranges.push((first, count, at));
        }
        self.pending.store(true, Ordering::Release);
    }

    pub(crate) fn flush(&self, list: &crate::host::HostList) {
        if self.pending.load(std::sync::atomic::Ordering::Acquire)
            && let Some(batch) = self.batch.lock().unwrap_or_else(|e| e.into_inner()).take()
        {
            self.pending
                .store(false, std::sync::atomic::Ordering::Release);
            batch.record(list);
        }
    }
}

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    #[cfg(windows)]
    host: Option<HostHeap>,
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host: make_host(desc),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

#[cfg(windows)]
fn make_host(desc: Desc) -> Option<HostHeap> {
    let kind = match desc.kind {
        0 => D3D12_QUERY_HEAP_TYPE_OCCLUSION,
        2 => D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
        _ => return None,
    };
    let mut heap = None;
    // SAFETY: the output belongs to us and the descriptor lives through the call.
    unsafe {
        crate::host::host()?.device.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: kind,
                Count: desc.count,
                NodeMask: 0,
            },
            &mut heap,
        )
    }
    .ok()?;
    let heap = heap?;
    let occlusion = if desc.kind == 0 {
        let host = crate::host::host()?;
        let results = crate::host::buffers::card_buffer(&host.device, u64::from(desc.count) * 8)?;
        let (visible, data) = host.upload_buffer(8)?;
        // SAFETY: les huit octets mappés restent constants pendant la vie du tas.
        unsafe { data.cast::<u64>().write_unaligned(1) };
        Some((results, visible))
    } else {
        None
    };
    Some(HostHeap { heap, occlusion })
}

#[cfg(windows)]
pub(crate) fn host_heap<'a>(this: This, kind: u32, first: u32, count: u32) -> Option<&'a HostHeap> {
    // SAFETY: a title query heap passed to one of its command-list methods.
    let heap = unsafe { this.cast::<Heap>().as_ref()? };
    if !matches!((heap.desc.kind, kind), (0, 0 | 1) | (2, 3))
        || count == 0
        || first.checked_add(count)? > heap.desc.count
    {
        return None;
    }
    heap.host.as_ref()
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::object::slot;
    use crate::testgpu::{Kernel, Pass, TitleBuffer, TitleList};

    #[test]
    fn queries_count_only_work_between_the_marks() {
        let test = "query::tests::queries_count_only_work_between_the_marks";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let kernel = Kernel::new(host);
        let scratch = TitleBuffer::new(256);
        let triangle = crate::draw::tests::Painter::new(host, false);
        for (heap_kind, kind, words) in [(0, 0, 2), (0, 1, 2), (2, 3, 22)] {
            let mut heap = std::ptr::null();
            let desc = Desc {
                kind: heap_kind,
                count: 2,
                nodes: 0,
            };
            assert_eq!(
                create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
                crate::object::S_OK
            );
            let list = TitleList::new();
            let out = TitleBuffer::new(256);
            out.set_words(|_| 0xface);
            let mark = |offset, index| {
                slot::<extern "system" fn(This, This, u32, u32)>(list.this(), offset)(
                    list.this(),
                    heap,
                    kind,
                    index,
                )
            };
            mark(0xe0, 0);
            mark(0xe8, 0);
            mark(0xe0, 1);
            if kind == 3 {
                kernel.dispatch(
                    list.host(),
                    Pass::Fill,
                    scratch.address(),
                    scratch.address(),
                );
            } else {
                triangle.bind(list.host());
                // SAFETY: le triangle couvre la moitié d’une cellule de 4 × 16 pixels.
                unsafe { list.host().list.DrawInstanced(3, 1, 0, 0) };
            }
            mark(0xe8, 1);
            let host_list = list.host();
            for index in 0..2 {
                slot::<extern "system" fn(This, This, u32, u32, u32, This, u64)>(list.this(), 0xf0)(
                    list.this(),
                    heap,
                    kind,
                    index,
                    1,
                    out.resource(),
                    8 + u64::from(index) * words as u64 * 4,
                );
            }
            assert_eq!(
                host_list
                    .query_resolves
                    .batch
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .ranges,
                [(0, 2, 8)]
            );
            release(heap);
            list.run();
            let result = out.words();
            let mut expected = vec![0; 2 * words];
            if kind == 3 {
                expected[words + 20] = 64;
            } else {
                expected[words] = if kind == 0 { 32 } else { 1 };
            }
            assert_eq!(&result[2..2 + expected.len()], expected);
            assert_eq!((result[1], result[2 + expected.len()]), (0xface, 0xface));
        }
    }

    #[test]
    fn a_dropped_draw_is_visible_until_the_slot_is_reused() {
        let Some(_gpu) = crate::testgpu::device(
            "query::tests::a_dropped_draw_is_visible_until_the_slot_is_reused",
        ) else {
            return;
        };
        for kind in [0, 1] {
            let desc = Desc {
                kind: 0,
                count: 2,
                nodes: 0,
            };
            let mut heap = std::ptr::null();
            assert_eq!(
                create(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
                crate::object::S_OK
            );
            let out = TitleBuffer::new(256);
            out.set_words(|_| 0xface);
            let writer = TitleList::new();
            let reader = TitleList::new();
            let mark = |offset, index| {
                slot::<extern "system" fn(This, This, u32, u32)>(writer.this(), offset)(
                    writer.this(),
                    heap,
                    kind,
                    index,
                )
            };
            let resolve =
                |list: &TitleList, offset| {
                    slot::<extern "system" fn(This, This, u32, u32, u32, This, u64)>(
                        list.this(),
                        0xf0,
                    )(list.this(), heap, kind, 0, 2, out.resource(), offset)
                };
            // La liste de lecture est enregistrée avant celle qui produit les résultats.
            resolve(&reader, 24);
            mark(0xe0, 0);
            slot::<extern "system" fn(This, u32, u32, u32, u32)>(writer.this(), 0x60)(
                writer.this(),
                3,
                1,
                0,
                0,
            );
            mark(0xe8, 0);
            mark(0xe0, 1);
            mark(0xe8, 1);
            resolve(&writer, 8);
            mark(0xe0, 0);
            mark(0xe8, 0);
            mark(0xe0, 1);
            slot::<extern "system" fn(This, u32, u32, u32, i32, u32)>(writer.this(), 0x68)(
                writer.this(),
                3,
                1,
                0,
                0,
                0,
            );
            mark(0xe8, 1);
            release(heap);
            writer.run();
            reader.run();
            assert_eq!(
                &out.words()[1..11],
                &[0xface, 1, 0, 0, 0, 0, 0, 1, 0, 0xface]
            );
        }
    }
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};
