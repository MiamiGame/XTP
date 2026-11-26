//! Measures Windows commit and DXGI usage for synthetic DEFAULT textures.

use std::{thread::sleep, time::Duration};

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;
use windows_sys::Win32::System::ProcessStatus::*;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const COUNT: u32 = 32;
const TILE_BYTES: u64 = 64 * 1024;

fn sample(adapter: &IDXGIAdapter3) -> [i64; 5] {
    sleep(Duration::from_millis(500));
    // SAFETY: the process and adapter are live and the outputs have their API sizes.
    unsafe {
        let mut process = PROCESS_MEMORY_COUNTERS_EX {
            cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        assert!(
            K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut process).cast(), process.cb)
                != 0
        );
        let mut system = PERFORMANCE_INFORMATION::default();
        assert!(
            K32GetPerformanceInfo(&mut system, size_of::<PERFORMANCE_INFORMATION>() as u32) != 0
        );
        let [local, non_local] = [
            DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
            DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
        ]
        .map(|segment| {
            let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter.QueryVideoMemoryInfo(0, segment, &mut info).unwrap();
            info.CurrentUsage as i64
        });
        [
            process.PrivateUsage as i64,
            process.PagefileUsage as i64,
            (system.CommitTotal * system.PageSize) as i64,
            local,
            non_local,
        ]
    }
}

/// Run alone with `XTP_TEST_ADAPTER=hardware` or `XTP_TEST_ADAPTER=warp`
/// and `XTP_BUFFER_POOL=0` to keep unrelated heaps out of the measurement.
#[test]
#[ignore = "measures driver memory and allocates about 2 GiB at a time"]
fn default_texture_commit_probe() {
    let Some(gpu) = crate::testgpu::device("texture_commit_probe::default_texture_commit_probe")
    else {
        return;
    };
    let device = &gpu.device;
    // SAFETY: these calls use the live test device, valid descriptions and
    // owned outputs. Placed textures are dropped before their heap.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        let info = adapter.GetDesc2().unwrap();
        let end = info
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(info.Description.len());
        let name = String::from_utf16_lossy(&info.Description[..end]);
        let software = info.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        let driver = match adapter.CheckInterfaceSupport(&IDXGIDevice::IID) {
            Ok(version) => {
                let [a, b, c, d] = [48, 32, 16, 0].map(|shift| (version >> shift) as u16);
                format!("{a}.{b}.{c}.{d}")
            }
            Err(error) => format!("unavailable ({:#010x})", error.code().0 as u32),
        };
        let delta = |case: &str, before: [i64; 5]| {
            let after = sample(&adapter);
            let [private, pagefile, system, local, non_local] =
                std::array::from_fn(|i| (after[i] - before[i]) as f64 / 1048576.0);
            println!(
                "{case}: {private:+.2}, {pagefile:+.2}, {system:+.2}, {local:+.2}, {non_local:+.2} \
                 ({name}, driver {driver}, software={software})"
            );
        };
        let released = |case: &str, before| {
            // Give the driver time to retire allocations before the next baseline.
            sleep(Duration::from_secs(2));
            delta(case, before);
        };
        println!("Delta MiB: PrivateUsage, PagefileUsage, CommitTotal, DXGI local, DXGI non-local");

        let properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4096,
            Height: 4096,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let state = D3D12_RESOURCE_STATE_COMMON;
        let bytes = device.GetResourceAllocationInfo(0, &[desc]).SizeInBytes;
        println!(
            "{COUNT} textures, {} MiB each before tiling",
            bytes as f64 / 1048576.0
        );

        let committed = |case: &str, flags| {
            let before = sample(&adapter);
            let textures: Vec<ID3D12Resource> = (0..COUNT)
                .map(|_| {
                    let mut texture = None;
                    device
                        .CreateCommittedResource(
                            &properties,
                            flags,
                            &desc,
                            state,
                            None,
                            &mut texture,
                        )
                        .unwrap();
                    texture.unwrap()
                })
                .collect();
            delta(case, before);
            if flags == D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT {
                let pageables: Vec<_> = textures.iter().map(|t| Some(t.cast().unwrap())).collect();
                device.MakeResident(&pageables).unwrap();
                delta("committed after MakeResident", before);
                device.Evict(&pageables).unwrap();
                delta("committed after Evict", before);
            }
            drop(textures);
            released(&format!("{case} after release"), before);
        };
        committed(
            "committed not resident",
            D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT,
        );

        let make_heap = |size| {
            let mut heap: Option<ID3D12Heap> = None;
            device
                .CreateHeap(
                    &D3D12_HEAP_DESC {
                        SizeInBytes: size,
                        Properties: properties,
                        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                        ..Default::default()
                    },
                    &mut heap,
                )
                .unwrap();
            heap.unwrap()
        };
        let before = sample(&adapter);
        let heap = make_heap(u64::from(COUNT) * bytes);
        delta("placed heap alone", before);
        let textures: Vec<ID3D12Resource> = (0..COUNT)
            .map(|i| {
                let mut texture = None;
                device
                    .CreatePlacedResource(
                        &heap,
                        u64::from(i) * bytes,
                        &desc,
                        state,
                        None,
                        &mut texture,
                    )
                    .unwrap();
                texture.unwrap()
            })
            .collect();
        delta("placed heap and textures", before);
        drop((textures, heap));
        released("placed after release", before);

        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
            )
            .unwrap();
        if options.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED {
            println!(
                "Reserved texture cases unavailable ({name}, driver {driver}, software={software}): no tiled-resource support"
            );
        } else {
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let queue = gpu.queue();
            let wait = |value| {
                queue.Signal(&fence, value).unwrap();
                fence
                    .SetEventOnCompletion(value, HANDLE::default())
                    .unwrap();
                device.GetDeviceRemovedReason().unwrap();
            };
            let before = sample(&adapter);
            let tiled = D3D12_RESOURCE_DESC {
                Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
                ..desc
            };
            let textures: Vec<ID3D12Resource> = (0..COUNT)
                .map(|_| {
                    let mut texture = None;
                    device
                        .CreateReservedResource(&tiled, state, None, &mut texture)
                        .unwrap();
                    texture.unwrap()
                })
                .collect();
            delta("reserved before mapping", before);

            let mut tiles = 0;
            device.GetResourceTiling(
                &textures[0],
                Some(&mut tiles),
                None,
                None,
                None,
                0,
                std::ptr::null_mut(),
            );
            assert_ne!(tiles, 0);
            let heap = make_heap(u64::from(COUNT) * u64::from(tiles) * TILE_BYTES);
            // A single region with no start or size covers the whole texture.
            // Map each texture to a different range in the same heap.
            for (i, texture) in (0..COUNT).zip(&textures) {
                let start = i * tiles;
                queue.UpdateTileMappings(
                    texture,
                    1,
                    None,
                    None,
                    &heap,
                    1,
                    None,
                    Some(&start),
                    None,
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            wait(1);
            delta("reserved fully mapped to one heap", before);
            for texture in &textures {
                let null = D3D12_TILE_RANGE_FLAG_NULL;
                queue.UpdateTileMappings(
                    texture,
                    1,
                    None,
                    None,
                    None::<&ID3D12Heap>,
                    1,
                    Some(&null),
                    None,
                    None,
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            wait(2);
            delta("reserved after unmapping (heap still live)", before);
            drop((textures, heap));
            released("reserved after release", before);
        }

        // Leave the resident case until last: the process charge can outlive release.
        committed("committed resident", D3D12_HEAP_FLAG_NONE);
    }
}
