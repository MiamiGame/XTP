//! The geometry stage of the console's rectangle pipelines whose vertex
//! shader has no per-vertex input (it makes the corners from the vertex
//! id): [`xtp_shader::emit::rect_gs`] behind the translated vertex shader.
//! Both draw paths run such a rectangle as the triangle of its three
//! vertices (direct: `draw::rectangle_vertices`; indirect: the rewrite pass
//! of [`crate::rectangles`]), and the stage adds the triangle with the
//! fourth corner. A pipeline of the console's rectangle type gets it at its
//! creation ([`crate::pipeline::host`]); one the driver refuses with it is
//! made without it, as before (logged there).

use std::sync::Mutex;

use crate::pipeline::host::Input;

/// The geometry shader (a DXIL container) that a pipeline of the console's
/// rectangle type with input elements `inputs` runs behind its translated
/// vertex shader `vs`, with its pixel shader `ps` (none: no pixel shader),
/// to make each rectangle's fourth corner. None when it has a per-vertex
/// input (its rectangles are expanded from their vertices instead:
/// [`crate::primitive::expand_rects`], [`crate::rectangles`]), or when the
/// stage cannot serve its shaders (logged once per reason).
pub(crate) fn geometry_stage(inputs: &[Input], vs: &[u8], ps: Option<&[u8]>) -> Option<Vec<u8>> {
    if inputs.iter().any(|e| !e.per_instance) {
        return None;
    }
    match xtp_shader::emit::rect_gs::rectangle_gs(vs, ps) {
        Ok(gs) => Some(gs),
        Err(e) => {
            static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
            let why = e.to_string();
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&why) {
                tracing::warn!(
                    reason = %why,
                    "rectangles without vertex inputs: no geometry stage, so no fourth corner \
                     (logged once per reason)"
                );
                seen.push(why);
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::{PCSTR, s};
    use xtp_shader::emit::rect_gs::probe::{self, PrimitiveId};

    use super::*;
    use crate::host::{Host, HostList, transition};
    use crate::shadow::Shadow;
    use crate::testgpu::{TitleBuffer, TitleList};

    /// The target: one cell of `CELL` × `CELL` pixels a case, side by
    /// side, four floats a pixel.
    const CELL: u32 = 16;
    const CELLS: u32 = 5;
    const W: u32 = CELL * CELLS;
    const H: u32 = CELL;
    /// A target row in the readback buffer (`W` × 16 bytes, 256-aligned).
    const PITCH: u32 = (W * 16).next_multiple_of(256);
    const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R32G32B32A32_FLOAT;
    /// The quality of the standard multisample pattern
    /// (`D3D12_STANDARD_MULTISAMPLE_PATTERN`): its sample positions average
    /// to the pixel's centre.
    const STANDARD_PATTERN: u32 = 0xffff_ffff;

    /// What a case's float output (`TEXCOORD0`) is, and what its pixels
    /// should show of it.
    #[derive(Debug, Clone, Copy)]
    enum Values {
        /// [`value`] at each vertex: its pixels show [`value`] at their
        /// centre.
        Affine,
        /// [`value`] at each vertex, not checked (not affine once drawn).
        Unchecked,
        /// These at v0, v1 and v2 (each in all four components), read flat:
        /// its pixels show v0's.
        Flat([f32; 3]),
    }

    /// A triangle the test draws: its vertices in pixels (x, y) and their
    /// w, what it should cover, and what its outputs are.
    struct Case {
        what: &'static str,
        corners: [(f32, f32); 3],
        /// w of each vertex: position (x w, y w) in clip space for pixel
        /// (x, y) at w 1.
        w: [f32; 3],
        /// Pixel x, y of what it covers: four corners for a rectangle, three
        /// for a triangle.
        covers: Vec<(f32, f32)>,
        values: Values,
        /// The clip or cull distance of v0, v1 and v2, for a draw that
        /// writes one.
        distances: [f32; 3],
        /// The plane `a x + b y + c` (pixels) its clip distances follow:
        /// only its positive side of the cover is drawn.
        plane: Option<[f32; 3]>,
    }

    /// Clip space of pixel (`x`, `y`) at w `w` (that pixel once divided).
    fn clip(x: f32, y: f32, w: f32) -> [f32; 4] {
        let (cx, cy) = (x / W as f32 * 2.0 - 1.0, 1.0 - y / H as f32 * 2.0);
        [cx * w, cy * w, 0.0, w]
    }

    /// The float output the probe carries, at pixel (`x`, `y`): affine, so
    /// the fourth corner's (`a + b − corner`) is its value there.
    fn value(x: f32, y: f32) -> [f32; 4] {
        [x / W as f32, y / H as f32, (x + 2.0 * y) / 128.0, 1.0]
    }

    /// How much each of [`value`]'s first three components changes for a
    /// pixel in x, and for one in y.
    const SLOPE: [[f32; 2]; 3] = [
        [1.0 / W as f32, 0.0],
        [0.0, 1.0 / H as f32],
        [1.0 / 128.0, 2.0 / 128.0],
    ];

    /// The flat output of vertex `k` of case `n`.
    fn flat(n: usize, k: usize) -> u32 {
        (100 * n + k + 1) as u32
    }

    /// Top left, bottom left, top right and bottom right of cell `n`.
    fn quad(n: u32) -> [(f32, f32); 4] {
        let (x0, x1, y1) = ((n * CELL) as f32, ((n + 1) * CELL) as f32, H as f32);
        [(x0, 0.0), (x0, y1), (x1, 0.0), (x1, y1)]
    }

    /// The rectangle filling cell `n`, its vertices `order` of [`quad`]'s
    /// first three.
    fn rectangle(n: u32, what: &'static str, order: [usize; 3]) -> Case {
        let at = quad(n);
        Case {
            what,
            corners: order.map(|k| at[k]),
            w: [1.0; 3],
            covers: vec![at[0], at[1], at[3], at[2]],
            values: Values::Affine,
            distances: [1.0; 3],
            plane: None,
        }
    }

    /// Three rectangles, cell by cell: one whose corner is v0 (the tiled
    /// lighting's order: top left, bottom left, top right), one whose
    /// corner is v1, one whose corner is v2.
    fn rectangles() -> Vec<Case> {
        vec![
            rectangle(0, "corner v0 (top left, bottom left, top right)", [0, 1, 2]),
            rectangle(1, "corner v1 (top right, top left, bottom left)", [2, 0, 1]),
            rectangle(2, "corner v2 (bottom left, top right, top left)", [1, 2, 0]),
        ]
    }

    /// The five cases, cell by cell: [`rectangles`], three vertices with no
    /// corner, and a corner whose third vertex has another w (its clip x
    /// and y are a corner's; once divided it is not one).
    fn cases() -> Vec<Case> {
        let triangle = |what: &'static str, corners: [(f32, f32); 3], w: [f32; 3], values| Case {
            what,
            corners,
            w,
            covers: corners.to_vec(),
            values,
            distances: [1.0; 3],
            plane: None,
        };
        let ([(x3, _), (_, y3), ..], [(x4, _), (_, y4), ..]) = (quad(3), quad(4));
        // At w 2, pixel y 4 has clip y 1, v0's: v0 shares x with v1 and
        // clip y with v2, a corner in clip space, but not once divided.
        let far = (x4 + 12.0, 4.0);
        assert_eq!(clip(far.0, far.1, 2.0)[1], clip(x4, 0.0, 1.0)[1]);
        let mut cases = rectangles();
        cases.push(triangle(
            "no corner",
            [(x3, 0.0), (x3, y3), (x3 + CELL as f32, 4.0)],
            [1.0; 3],
            Values::Affine,
        ));
        cases.push(triangle(
            "a corner whose w differ",
            [(x4, 0.0), (x4, y4), far],
            [1.0, 1.0, 2.0],
            Values::Unchecked,
        ));
        cases
    }

    /// The probe vertex shader for `cases`: three vertex ids a case, and
    /// their distances as clip (`distance` 6) or cull (7) distances, or
    /// none.
    fn vertex_shader(cases: &[Case], distance: Option<u8>) -> Vec<u8> {
        let (mut positions, mut values, mut flats, mut distances) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (n, c) in cases.iter().enumerate() {
            for k in 0..3 {
                let ((x, y), w) = (c.corners[k], c.w[k]);
                positions.push(clip(x, y, w));
                values.push(match c.values {
                    Values::Flat(v) => [v[k]; 4],
                    Values::Affine | Values::Unchecked => value(x, y),
                });
                flats.push(flat(n, k));
                distances.push(c.distances[k]);
            }
        }
        match distance {
            Some(kind) => {
                probe::vertex_with_distance(&positions, &values, &flats, kind, &distances)
            }
            None => probe::vertex(&positions, &values, &flats),
        }
    }

    /// The stage made for `vs` and `ps`, as a pipeline of no vertex input
    /// gets it.
    fn stage(vs: &[u8], ps: &[u8]) -> Vec<u8> {
        geometry_stage(&[], vs, Some(ps)).expect("a geometry stage")
    }

    /// Whether pixel point `p` is inside the convex polygon `poly` by more
    /// than `margin` pixels (`Some(true)`), outside by more (`Some(false)`),
    /// or on its edge (`None`).
    fn inside(poly: &[(f32, f32)], p: (f32, f32), margin: f32) -> Option<bool> {
        let mut sign = 0.0f32;
        let mut near = false;
        for (i, &a) in poly.iter().enumerate() {
            let b = poly[(i + 1) % poly.len()];
            let (ex, ey) = (b.0 - a.0, b.1 - a.1);
            let d = (ex * (p.1 - a.1) - ey * (p.0 - a.0)) / (ex * ex + ey * ey).sqrt();
            if d.abs() <= margin {
                near = true;
                continue;
            }
            if sign == 0.0 {
                sign = d.signum();
            } else if d.signum() != sign {
                return Some(false);
            }
        }
        if near { None } else { Some(true) }
    }

    /// A root signature with nothing in it but `flags`.
    fn empty_root(device: &ID3D12Device, flags: D3D12_ROOT_SIGNATURE_FLAGS) -> ID3D12RootSignature {
        let mut blob: Option<ID3DBlob> = None;
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            Flags: flags,
            ..Default::default()
        };
        // SAFETY: a description of nothing; the out pointer is ours.
        unsafe {
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
                .expect("serialized");
            let blob = blob.expect("a blob");
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device
                .CreateRootSignature(0, bytes)
                .expect("a root signature")
        }
    }

    fn bytecode(code: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: code.as_ptr().cast(),
            BytecodeLength: code.len(),
        }
    }

    /// `samples` a pixel, in the standard pattern when more than one.
    fn sampling(samples: u32) -> DXGI_SAMPLE_DESC {
        DXGI_SAMPLE_DESC {
            Count: samples,
            Quality: if samples > 1 { STANDARD_PATTERN } else { 0 },
        }
    }

    /// A pipeline of `vs`, `gs` and `ps` drawing triangles into a
    /// [`FORMAT`] target of `samples` a pixel, culling `cull`.
    fn pipeline(
        device: &ID3D12Device,
        root: &ID3D12RootSignature,
        (vs, gs, ps): (&[u8], Option<&[u8]>, &[u8]),
        cull: D3D12_CULL_MODE,
        samples: u32,
    ) -> ID3D12PipelineState {
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = FORMAT;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            VS: bytecode(vs),
            GS: gs.map_or_else(D3D12_SHADER_BYTECODE::default, bytecode),
            PS: bytecode(ps),
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: cull,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: formats,
            SampleDesc: sampling(samples),
            ..Default::default()
        };
        // SAFETY: the description and the code it points at live through
        // the call.
        unsafe { device.CreateGraphicsPipelineState(&desc) }.expect("the driver takes it")
    }

    /// A [`FORMAT`] texture of the target's size, `samples` a pixel, a
    /// render target or not, made in `state`.
    fn texture(
        device: &ID3D12Device,
        samples: u32,
        target: bool,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain creation from descriptions that live through the
        // call.
        unsafe {
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(W),
                    Height: H,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: FORMAT,
                    SampleDesc: sampling(samples),
                    Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                    Flags: if target {
                        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                    } else {
                        D3D12_RESOURCE_FLAG_NONE
                    },
                    ..Default::default()
                },
                state,
                None,
                &mut resource,
            )
        }
        .expect("a texture");
        resource.expect("a texture")
    }

    /// A [`FORMAT`] render target and its view, with the single-sample
    /// texture it resolves to when it has more samples a pixel.
    struct Target {
        resource: ID3D12Resource,
        resolved: Option<ID3D12Resource>,
        views: ID3D12DescriptorHeap,
    }

    impl Target {
        fn new(device: &ID3D12Device, samples: u32) -> Target {
            let resource = texture(device, samples, true, D3D12_RESOURCE_STATE_RENDER_TARGET);
            let resolved =
                (samples > 1).then(|| texture(device, 1, false, D3D12_RESOURCE_STATE_RESOLVE_DEST));
            // SAFETY: plain creation from a description that lives through
            // the call; a view of our live target.
            unsafe {
                let views: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    })
                    .expect("a view heap");
                device.CreateRenderTargetView(
                    &resource,
                    None,
                    views.GetCPUDescriptorHandleForHeapStart(),
                );
                Target {
                    resource,
                    resolved,
                    views,
                }
            }
        }

        /// Clears it and binds it with `pso` and `root`, triangles listed.
        fn bind(&self, list: &HostList, root: &ID3D12RootSignature, pso: &ID3D12PipelineState) {
            // SAFETY: an open list; our live objects, alive until it has run.
            unsafe {
                let rtv = self.views.GetCPUDescriptorHandleForHeapStart();
                let l = &list.list;
                l.ClearRenderTargetView(rtv, &[0.0f32; 4], None);
                l.SetGraphicsRootSignature(root);
                l.SetPipelineState(pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: W as f32,
                    Height: H as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: W as i32,
                    bottom: H as i32,
                }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
        }

        /// Runs `list` and reads the target back (resolved, with more
        /// samples a pixel): four floats a pixel, rows top first.
        fn read(&self, host: &Host, list: &TitleList) -> Vec<[f32; 4]> {
            let (buffer, data) = host.shared_buffer(u64::from(PITCH * H)).expect("readback");
            let l = list.host();
            // SAFETY: an open list; our live textures and buffer, the
            // footprint inside the buffer.
            unsafe {
                let source = match &self.resolved {
                    Some(resolved) => {
                        l.list.ResourceBarrier(&[transition(
                            &self.resource,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                        )]);
                        l.list
                            .ResolveSubresource(resolved, 0, &self.resource, 0, FORMAT);
                        l.list.ResourceBarrier(&[transition(
                            resolved,
                            D3D12_RESOURCE_STATE_RESOLVE_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        resolved
                    }
                    None => {
                        l.list.ResourceBarrier(&[transition(
                            &self.resource,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        &self.resource
                    }
                };
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&buffer),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: FORMAT,
                                Width: W,
                                Height: H,
                                Depth: 1,
                                RowPitch: PITCH,
                            },
                        },
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(source),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                l.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
            list.run();
            (0..W * H)
                .map(|i| {
                    let at = (i / W * PITCH + i % W * 16) as usize;
                    // SAFETY: the list has run; inside the mapped buffer.
                    unsafe { data.add(at).cast::<[f32; 4]>().read_unaligned() }
                })
                .collect()
        }
    }

    /// Draws `vertices` vertex ids with `vs`, `gs` (none: no stage) and
    /// `ps`, culling `cull`, `samples` a pixel: the target, read back.
    fn draw(
        host: &Host,
        shaders: (&[u8], Option<&[u8]>, &[u8]),
        cull: D3D12_CULL_MODE,
        samples: u32,
        vertices: u32,
    ) -> Vec<[f32; 4]> {
        let root = empty_root(&host.device, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let pso = pipeline(&host.device, &root, shaders, cull, samples);
        let target = Target::new(&host.device, samples);
        let list = TitleList::new();
        target.bind(list.host(), &root, &pso);
        // SAFETY: an open list with everything bound.
        unsafe { list.host().list.DrawInstanced(vertices, 1, 0, 0) };
        target.read(host, &list)
    }

    /// Checks `pixels` against `cases` (cell by cell; a cell past them has
    /// nothing): each pixel well inside a case's cover (and on its plane's
    /// positive side) is drawn, with the case's v0 flat value and its float
    /// outputs (an affine one as at the pixel's centre, give or take
    /// `slack` pixels); each pixel well outside is not. Returns, by case,
    /// the pixels drawn.
    fn check(pixels: &[[f32; 4]], cases: &[Case], what: &str, slack: f32) -> Vec<usize> {
        let mut drawn = vec![0; cases.len()];
        for y in 0..H {
            for x in 0..W {
                let px = pixels[(y * W + x) as usize];
                let n = (x / CELL) as usize;
                let Some(c) = cases.get(n) else {
                    assert_eq!(px, [0.0; 4], "{what}: pixel ({x}, {y}) of an empty cell");
                    continue;
                };
                let centre = (x as f32 + 0.5, y as f32 + 0.5);
                let at = format!("{what}, {}: pixel ({x}, {y})", c.what);
                let side = c.plane.map(|[a, b, k]| a * centre.0 + b * centre.1 + k);
                let state = match (inside(&c.covers, centre, 0.01), side) {
                    (Some(true), Some(s)) if s.abs() <= 0.05 => None,
                    (Some(true), Some(s)) => Some(s > 0.0),
                    (state, _) => state,
                };
                match state {
                    Some(true) => {
                        assert_eq!(px[3], flat(n, 0) as f32, "{at}: v0's flat value");
                        match c.values {
                            Values::Affine => {
                                let want = value(centre.0, centre.1);
                                for k in 0..3 {
                                    let near = 1e-4 + slack * (SLOPE[k][0] + SLOPE[k][1]);
                                    assert!(
                                        (px[k] - want[k]).abs() < near,
                                        "{at}: output {k} {} for {}",
                                        px[k],
                                        want[k]
                                    );
                                }
                            }
                            Values::Flat(v) => {
                                assert_eq!(px[..3], [v[0]; 3], "{at}: v0's flat floats")
                            }
                            Values::Unchecked => {}
                        }
                    }
                    Some(false) => assert_eq!(px, [0.0; 4], "{at}: not drawn"),
                    None => {}
                }
                if px[3] != 0.0 {
                    drawn[n] += 1;
                }
            }
        }
        drawn
    }

    /// Asserts that each of `cases`' rectangles (the first three) is drawn
    /// whole.
    fn whole(drawn: &[usize], cases: &[Case]) {
        for (n, c) in cases.iter().take(3).enumerate() {
            assert_eq!(drawn[n], (CELL * CELL) as usize, "{}: whole", c.what);
        }
    }

    /// Asserts that each of `cases`' rectangles (the first three) is drawn
    /// half (a right triangle of the cell: half of it, give or take the
    /// pixels its diagonal goes through).
    fn half(drawn: &[usize], cases: &[Case]) {
        for (n, c) in cases.iter().take(3).enumerate() {
            let half = (CELL * CELL / 2) as usize;
            assert!(
                drawn[n].abs_diff(half) <= CELL as usize,
                "{}: one half ({} pixels)",
                c.what,
                drawn[n]
            );
        }
    }

    /// The stage is made for a pipeline without per-vertex inputs (per
    /// instance ones are no vertex input), not for one with them, nor for a
    /// pixel shader that reads the primitive id (one that only declares it
    /// gets it).
    #[test]
    fn the_stage_is_for_pipelines_without_vertex_inputs() {
        let input = |per_instance| Input {
            slot: 0,
            offset: 0,
            format: 2,
            per_instance,
        };
        let cases = cases();
        let (vs, ps) = (vertex_shader(&cases, None), probe::pixel());
        assert!(geometry_stage(&[], &vs, Some(&ps)).is_some());
        assert!(geometry_stage(&[input(true)], &vs, None).is_some());
        assert!(geometry_stage(&[input(true), input(false)], &vs, Some(&ps)).is_none());
        let declared = probe::pixel_with(2, PrimitiveId::Declared);
        assert!(
            geometry_stage(&[], &vs, Some(&declared)).is_some(),
            "a primitive id declared, never read"
        );
        let read = probe::pixel_with(2, PrimitiveId::Read);
        assert!(
            geometry_stage(&[], &vs, Some(&read)).is_none(),
            "a primitive id read"
        );
    }

    fn depth_bounds_state(
        topology: i32,
        with_depth: bool,
        depth_test: bool,
    ) -> crate::pipeline::host::GraphicsState {
        use crate::pipeline::host::GraphicsState;
        #[repr(C)]
        struct Sub<T>(u32, T);
        let mut state = GraphicsState::default();
        state.targets.NumRenderTargets = 1;
        state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(topology);
        state.targets.RTFormats[0] = FORMAT;
        let bounds = Sub(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: depth_test.into(),
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        let raster = Sub(
            10,
            D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: each scalar subobject has its D3D12 type followed by its descriptor.
        unsafe {
            state.read(21, (&raw const bounds).cast());
            state.read(10, (&raw const raster).cast());
            if with_depth {
                let format = Sub(16, DXGI_FORMAT_D32_FLOAT);
                state.read(16, (&raw const format).cast());
            }
        }
        state
    }

    #[test]
    fn rectangle_geometry_obeys_depth_bounds() {
        check_rectangle_bounds("rectangle_geometry_obeys_depth_bounds", true);
    }

    #[test]
    #[ignore = "WARP ignores depth bounds with a bound DSV and DepthEnable=FALSE (Hangar 5baa47be)"]
    fn rectangle_geometry_obeys_bounds_with_depth_testing_off() {
        check_rectangle_bounds(
            "rectangle_geometry_obeys_bounds_with_depth_testing_off",
            false,
        );
    }

    fn rectangle_shaders(cases: &[Case]) -> [Arc<crate::pipeline::Shader>; 2] {
        use crate::pipeline::test_shader;
        use xtp_shader::{HwStage, emit::HostShader};
        let stage = |stage, dxil| HostShader {
            stage,
            dxil,
            group_base: None,
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            stride_table: false,
            run_time_buffers: 0,
        };
        [
            test_shader("vertex", stage(HwStage::Vs, vertex_shader(cases, None))),
            test_shader("pixel", stage(HwStage::Ps, probe::pixel())),
        ]
    }

    #[test]
    fn indexed_vertex_ids_select_the_rectangle_and_its_fourth_corner() {
        use crate::object::{This, slot};
        use crate::pipeline::{host_pipeline, test_graphics_pipeline};
        let Some(gpu) = crate::testgpu::device(
            "rectangles::geometry::tests::indexed_vertex_ids_select_the_rectangle_and_its_fourth_corner",
        ) else {
            return;
        };
        let cases = rectangles();
        let mut state = crate::pipeline::host::GraphicsState::default();
        state.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(19);
        state.targets.NumRenderTargets = 1;
        state.targets.RTFormats[0] = FORMAT;
        // Même rasterisation sans élimination de faces que les autres sondes.
        #[repr(C)]
        struct Raster(u32, D3D12_RASTERIZER_DESC);
        let raster = Raster(
            10,
            D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: sous-objet de rasterisation complet.
        unsafe { state.read(10, (&raw const raster).cast()) };
        let title = test_graphics_pipeline(rectangle_shaders(&cases).into(), state, &[0]);
        let pso = host_pipeline(title).expect("pipeline de rectangles");
        let target = Target::new(&gpu.host().device, 1);
        let list = TitleList::new();
        crate::testgpu::bind_indexed_heaps(gpu.host(), list.host());
        target.bind(list.host(), &pso.root, &pso.pso);
        let indices = TitleBuffer::new(16);
        indices.set_words(|i| [0, 6, 7, 8][i]);
        let view = crate::draw::IndexView {
            address: indices.address(),
            size: 16,
            format: 42,
        };
        let t = list.this();
        slot::<extern "system" fn(This, This)>(t, 0xc8)(t, title);
        slot::<extern "system" fn(This, u32)>(t, 0xa0)(t, 17);
        slot::<extern "system" fn(This, *const crate::draw::IndexView)>(t, 0x178)(t, &view);
        slot::<extern "system" fn(This, u32, u32, u32, i32, u32)>(t, 0x68)(t, 3, 1, 1, 0, 0);
        let pixels = target.read(gpu.host(), &list);
        for (i, pixel) in pixels.iter().enumerate() {
            let cell = (i as u32 % W) / CELL;
            assert_eq!(pixel[3], if cell == 2 { flat(2, 0) as f32 } else { 0.0 });
        }
        slot::<extern "system" fn(This) -> u32>(title, 0x10)(title);
    }

    fn check_rectangle_bounds(test: &str, depth_test: bool) {
        use crate::object::{This, slot};
        use crate::pipeline::host::HostPipeline;
        use crate::resource::tests_support::{committed, depth_unordered};
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !host.depth_bounds {
            crate::testgpu::skip(test, "depth bounds unsupported");
            return;
        }
        let cases = rectangles();
        let shaders = rectangle_shaders(&cases);
        for stored in [None, Some(0.125), Some(0.5)] {
            let state = depth_bounds_state(19, stored.is_some(), stored.is_some() && depth_test);
            let pso = HostPipeline::new(&shaders, Some(&state))
                .expect("rectangle pipeline with depth bounds");
            let list = TitleList::new();
            let target = Target::new(&host.device, 1);
            crate::testgpu::bind_indexed_heaps(host, list.host());
            target.bind(list.host(), &pso.root, &pso.pso);
            let depth = stored.map(|z| {
                let title = committed(&depth_unordered(W.into(), H, 40));
                let tex = crate::resource::host_resource(title).expect("depth target");
                let dsv = host.dsv(
                    tex,
                    Some(&D3D12_DEPTH_STENCIL_VIEW_DESC {
                        Format: DXGI_FORMAT_D32_FLOAT,
                        ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                        ..Default::default()
                    }),
                );
                crate::list::ensure_state(
                    list.host(),
                    title,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32,
                );
                // SAFETY: both descriptors and textures live through the draw.
                unsafe {
                    list.host()
                        .list
                        .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, z, 0, None);
                    list.host().list.OMSetRenderTargets(
                        1,
                        Some(&target.views.GetCPUDescriptorHandleForHeapStart()),
                        false,
                        Some(&dsv),
                    );
                }
                title
            });
            let offset = if crate::device::newer() { 0x368 } else { 0x358 };
            slot::<extern "system" fn(This, f32, f32)>(list.this(), offset)(
                list.this(),
                0.25,
                0.75,
            );
            // SAFETY: an open list with the test pipeline and its targets.
            unsafe {
                list.host()
                    .list
                    .DrawInstanced(cases.len() as u32 * 3, 1, 0, 0)
            };
            let pixels = target.read(host, &list);
            if let Some(title) = depth {
                slot::<extern "system" fn(This) -> u32>(title, 0x10)(title);
            }
            if stored == Some(0.125) {
                assert!(pixels.iter().all(|p| *p == [0.; 4]));
            } else {
                whole(&check(&pixels, &cases, "bounds passed", 0.0), &cases);
            }
        }
    }

    #[test]
    fn mesh_pipeline_accepts_bounds_with_depth_testing_off() {
        use crate::pipeline::{host::HostPipeline, test_shader};
        use xtp_shader::{
            HwStage, NativeProgram,
            native::{MeshIo, StageIo},
        };
        let test =
            "rectangles::geometry::tests::mesh_pipeline_accepts_bounds_with_depth_testing_off";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !host.depth_bounds || !host.mesh_shaders() {
            crate::testgpu::skip(test, "mesh shaders and depth bounds required");
            return;
        }
        // One wave exports position (0,0,0,1) and a degenerate primitive.
        let words: [u32; 7] = [
            0x7e00_0280,
            0x7e02_02f2,
            0xf800_08cf,
            0x0100_0000,
            0xf800_0941,
            0,
            0xbf81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            float_mode: 0,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.into_iter().flat_map(u32::to_le_bytes).collect(),
        };
        let vs = vertex_shader(&rectangles(), None);
        let mut container = xtp_shader::Container::parse(&vs).expect("output signature");
        container
            .parts
            .retain(|p| p.fourcc == xtp_shader::FourCC::OSG1);
        let mesh = xtp_shader::emit::translate_program(&container, &native).expect("mesh fixture");
        let mut state = depth_bounds_state(D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0, true, false);
        state.targets.NumRenderTargets = 0;
        state.targets.RTFormats[0] = DXGI_FORMAT_UNKNOWN;
        assert!(
            HostPipeline::new(&[test_shader("mesh", mesh)], Some(&state)).is_some(),
            "{:?}",
            crate::pipeline::host::refusals()
        );
    }

    /// A rectangle list without vertex inputs, drawn as the direct path
    /// draws it (three vertices a rectangle): each rectangle is whole, with
    /// the fourth corner whichever vertex is the corner (the float output
    /// is the affine function's value at every pixel of both triangles),
    /// and v0's flat value over all of it; three vertices with no corner,
    /// or with another w, stay one triangle. Drawn culling back faces, then
    /// front faces: each case is drawn whole by exactly one of them (both
    /// triangles turn the same way).
    #[test]
    fn rectangles_without_vertex_inputs_get_their_fourth_corner() {
        let test =
            "rectangles::geometry::tests::rectangles_without_vertex_inputs_get_their_fourth_corner";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let cases = cases();
        let (vs, ps) = (vertex_shader(&cases, None), probe::pixel());
        let gs = stage(&vs, &ps);
        let vertices = 3 * cases.len() as u32;
        let passes: Vec<Vec<[f32; 4]>> = [D3D12_CULL_MODE_BACK, D3D12_CULL_MODE_FRONT]
            .into_iter()
            .map(|cull| draw(host, (&vs, Some(&gs), &ps), cull, 1, vertices))
            .collect();
        let mut both = passes[0].clone();
        for (b, f) in both.iter_mut().zip(&passes[1]) {
            if f[3] != 0.0 {
                assert_eq!(*b, [0.0; 4], "a pixel drawn by both cull modes");
                *b = *f;
            }
        }
        let total = check(&both, &cases, "both cull modes", 0.0);
        for (n, c) in cases.iter().enumerate() {
            let count = |pixels: &[[f32; 4]]| {
                (0..W * H)
                    .filter(|&i| (i % W) / CELL == n as u32 && pixels[i as usize][3] != 0.0)
                    .count()
            };
            let (b, f) = (count(&passes[0]), count(&passes[1]));
            assert!(
                (b == 0) != (f == 0),
                "{}: drawn whole by one cull mode ({b} and {f} pixels)",
                c.what
            );
            assert_eq!(b + f, total[n]);
        }
        whole(&total, &cases);
    }

    /// Without the stage the same draw lights each rectangle's first
    /// triangle only, with its values: what the test above sees is the
    /// stage's doing.
    #[test]
    fn without_the_stage_rectangles_keep_one_triangle() {
        let test = "rectangles::geometry::tests::without_the_stage_rectangles_keep_one_triangle";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut cases = cases();
        let (vs, ps) = (vertex_shader(&cases, None), probe::pixel());
        let vertices = 3 * cases.len() as u32;
        let pixels = draw(host, (&vs, None, &ps), D3D12_CULL_MODE_NONE, 1, vertices);
        for c in &mut cases {
            c.covers = c.corners.to_vec();
        }
        let drawn = check(&pixels, &cases, "without the stage", 0.0);
        half(&drawn, &cases);
    }

    /// A float that the pixel shader reads flat (constant) while the vertex
    /// shader declares it linear, 1, 2 and 4 at v0, v1 and v2: both halves
    /// of each rectangle show v0's 1, whichever vertex is the corner (a half
    /// that the fourth corner led would show its `a + b − corner`: 5 with
    /// corner v0).
    #[test]
    fn flat_floats_keep_the_first_vertex_on_both_halves() {
        let test = "rectangles::geometry::tests::flat_floats_keep_the_first_vertex_on_both_halves";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut cases = rectangles();
        for c in &mut cases {
            c.values = Values::Flat([1.0, 2.0, 4.0]);
        }
        let vs = vertex_shader(&cases, None);
        let ps = probe::pixel_with(1, PrimitiveId::None);
        let gs = stage(&vs, &ps);
        let pixels = draw(host, (&vs, Some(&gs), &ps), D3D12_CULL_MODE_NONE, 1, 9);
        whole(&check(&pixels, &cases, "flat floats", 0.0), &cases);
    }

    /// The three w equal and not 1 (2): x and y divide alike, so the corner
    /// is one once divided, and interpolation with perspective is affine.
    /// The picture of w 1, interpolated with perspective (linear) and
    /// without (noperspective).
    #[test]
    fn a_common_w_and_noperspective_draw_as_at_w_1() {
        let test = "rectangles::geometry::tests::a_common_w_and_noperspective_draw_as_at_w_1";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut cases = rectangles();
        for c in &mut cases {
            c.w = [2.0; 3];
        }
        let vs = vertex_shader(&cases, None);
        for mode in [2, 4] {
            let ps = probe::pixel_with(mode, PrimitiveId::None);
            let gs = stage(&vs, &ps);
            let pixels = draw(host, (&vs, Some(&gs), &ps), D3D12_CULL_MODE_NONE, 1, 9);
            let drawn = check(&pixels, &cases, &format!("w 2, mode {mode}"), 0.0);
            whole(&drawn, &cases);
        }
    }

    /// Whether `device` renders [`FORMAT`] with 4 samples a pixel and
    /// resolves it.
    fn four_samples(device: &ID3D12Device) -> bool {
        let mut levels = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: FORMAT,
            SampleCount: 4,
            ..Default::default()
        };
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: FORMAT,
            ..Default::default()
        };
        // SAFETY: each query writes its own structure of the given size.
        let asked = unsafe {
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                    (&raw mut levels).cast(),
                    size_of_val(&levels) as u32,
                )
                .is_ok()
                && device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_FORMAT_SUPPORT,
                        (&raw mut support).cast(),
                        size_of_val(&support) as u32,
                    )
                    .is_ok()
        };
        let wanted = D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET.0
            | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE.0;
        asked && levels.NumQualityLevels > 0 && support.Support1.0 & wanted == wanted
    }

    /// 4 samples a pixel (the standard pattern), and the pixels that the
    /// diagonal between the halves goes through: each sample is drawn by
    /// one half (the flat value resolves whole), and the float output
    /// interpolated per sample resolves to its value at the centre (the
    /// samples average there); at the centroid, to a value within half a
    /// pixel of it (each half evaluates it where it covers the pixel).
    #[test]
    fn centroid_and_sample_interpolation_hold_at_the_diagonal_under_msaa() {
        let test = "rectangles::geometry::tests::centroid_and_sample_interpolation_hold_at_the_diagonal_under_msaa";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !four_samples(&host.device) {
            let why = "the device has no 4 samples a pixel to render and resolve four floats";
            crate::testgpu::skip(test, why);
            return;
        }
        let cases = rectangles();
        let vs = vertex_shader(&cases, None);
        for (mode, slack) in [(3, 0.5), (6, 0.0)] {
            let ps = probe::pixel_with(mode, PrimitiveId::None);
            let gs = stage(&vs, &ps);
            let pixels = draw(host, (&vs, Some(&gs), &ps), D3D12_CULL_MODE_NONE, 4, 9);
            let drawn = check(&pixels, &cases, &format!("4 samples, mode {mode}"), slack);
            whole(&drawn, &cases);
        }
    }

    /// Clip distances through both halves: the oblique plane x + y/2 = 12
    /// (from the cell's left side) crosses the diagonal of either split.
    /// Only its positive side is drawn, on the half with the fourth corner
    /// (whose distance, `a + b − corner`, is the plane's there) as on the
    /// other.
    #[test]
    fn clip_distances_clip_both_halves_along_their_plane() {
        let test = "rectangles::geometry::tests::clip_distances_clip_both_halves_along_their_plane";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut cases = rectangles();
        for (n, c) in cases.iter_mut().enumerate() {
            let plane = [1.0, 0.5, -((n as u32 * CELL) as f32 + 12.0)];
            c.distances = c
                .corners
                .map(|(x, y)| plane[0] * x + plane[1] * y + plane[2]);
            c.plane = Some(plane);
        }
        let (vs, ps) = (vertex_shader(&cases, Some(6)), probe::pixel());
        let gs = stage(&vs, &ps);
        let pixels = draw(host, (&vs, Some(&gs), &ps), D3D12_CULL_MODE_NONE, 1, 9);
        // Half of each cell is on the positive side.
        half(&check(&pixels, &cases, "clip distances", 0.0), &cases);
    }

    /// Cull distances: each half is culled by its own three, the fourth
    /// corner's being `a + b − corner`. With corner v0 (split from the
    /// corner to `d`), v0, v1, v2 at −1, 1, −3 put `d` at −1 and cull the
    /// half `v0, d, v2`; with corner v1 or v2 (split from `a` to `b`), the
    /// corner at 1 and the others at −1 put `d` at −3 and cull the half
    /// without the corner.
    #[test]
    fn cull_distances_cull_the_half_whose_three_are_negative() {
        let test =
            "rectangles::geometry::tests::cull_distances_cull_the_half_whose_three_are_negative";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut cases = rectangles();
        let culls = [[-1.0, 1.0, -3.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
        for (n, c) in cases.iter_mut().enumerate() {
            let [top_left, bottom_left, top_right, bottom_right] = quad(n as u32);
            c.distances = culls[n];
            c.covers = if n == 0 {
                vec![top_left, bottom_left, bottom_right]
            } else {
                vec![top_left, bottom_left, top_right]
            };
        }
        let (vs, ps) = (vertex_shader(&cases, Some(7)), probe::pixel());
        let gs = stage(&vs, &ps);
        let pixels = draw(host, (&vs, Some(&gs), &ps), D3D12_CULL_MODE_NONE, 1, 9);
        half(&check(&pixels, &cases, "cull distances", 0.0), &cases);
    }

    /// A pixel shader that declares `SV_PrimitiveID` and never reads it
    /// gets the stage, and its pipeline draws whole rectangles (the stage
    /// writes no primitive id, which this pixel shader does not miss).
    #[test]
    fn an_unread_primitive_id_keeps_the_stage() {
        let test = "rectangles::geometry::tests::an_unread_primitive_id_keeps_the_stage";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let cases = cases();
        let vs = vertex_shader(&cases, None);
        let ps = probe::pixel_with(2, PrimitiveId::Declared);
        let gs = stage(&vs, &ps);
        let vertices = 3 * cases.len() as u32;
        let pixels = draw(
            host,
            (&vs, Some(&gs), &ps),
            D3D12_CULL_MODE_NONE,
            1,
            vertices,
        );
        whole(
            &check(&pixels, &cases, "a primitive id declared", 0.0),
            &cases,
        );
    }

    /// One vertex the stage emitted, as stream output writes it: the words
    /// of `SV_Position`, `TEXCOORD0` and `TEXCOORD1`.
    type Streamed = [u32; 9];

    /// Runs `vs` and its stage `gs` on `vertices` vertex ids, nothing
    /// rasterized: the vertices the stage emitted, three a triangle, in
    /// order.
    fn stream_out(host: &Host, vs: &[u8], gs: &[u8], vertices: u32) -> Vec<Streamed> {
        let device = &host.device;
        let root = empty_root(device, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT);
        let entry = |name: PCSTR, index: u32, count: u8| D3D12_SO_DECLARATION_ENTRY {
            Stream: 0,
            SemanticName: name,
            SemanticIndex: index,
            StartComponent: 0,
            ComponentCount: count,
            OutputSlot: 0,
        };
        let entries = [
            entry(s!("SV_Position"), 0, 4),
            entry(s!("TEXCOORD"), 0, 4),
            entry(s!("TEXCOORD"), 1, 1),
        ];
        let stride = size_of::<Streamed>() as u32;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: bytecode(vs),
            GS: bytecode(gs),
            StreamOutput: D3D12_STREAM_OUTPUT_DESC {
                pSODeclaration: entries.as_ptr(),
                NumEntries: entries.len() as u32,
                pBufferStrides: &stride,
                NumStrides: 1,
                RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM,
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            SampleDesc: sampling(1),
            ..Default::default()
        };
        // SAFETY: the description and what it points at live through the
        // call.
        let pso: ID3D12PipelineState =
            unsafe { device.CreateGraphicsPipelineState(&desc) }.expect("the driver takes it");
        // The filled size first (8 bytes), the vertices from byte 256: two
        // triangles at most for each three vertex ids.
        let room = u64::from(2 * vertices * stride);
        let (buffer, data) = host.shared_buffer(256 + room).expect("a buffer");
        // SAFETY: inside the mapped buffer, which no GPU work uses yet.
        unsafe { data.cast::<u64>().write_unaligned(0) };
        let list = TitleList::new();
        let l = &list.host().list;
        // SAFETY: an open list; our live buffer, pipeline and root
        // signature, alive until it has run; the views inside the buffer.
        unsafe {
            let at = buffer.GetGPUVirtualAddress();
            l.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_STREAM_OUT,
            )]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.SOSetTargets(
                0,
                Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW {
                    BufferLocation: at + 256,
                    SizeInBytes: room,
                    BufferFilledSizeLocation: at,
                }]),
            );
            l.DrawInstanced(vertices, 1, 0, 0);
            l.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_STREAM_OUT,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        list.run();
        let stride = stride as usize;
        // SAFETY: the list has run; inside the mapped buffer.
        let filled = unsafe { data.cast::<u32>().read_unaligned() } as usize;
        assert_eq!(filled % (3 * stride), 0, "whole triangles ({filled} bytes)");
        (0..filled / stride)
            .map(|i| {
                // SAFETY: as above, within what the stage filled.
                unsafe {
                    data.add(256 + i * stride)
                        .cast::<Streamed>()
                        .read_unaligned()
                }
            })
            .collect()
    }

    /// `TEXCOORD0` of vertex `k` of input `i` of the streaming tests:
    /// small values, exact in `a + b − corner`.
    fn input_values(i: usize, k: usize) -> [f32; 4] {
        let base = (10 * i + k + 1) as f32;
        [base, -base, 0.5 * base, 4.0 * base]
    }

    /// Three vertices' x, y and w, in clip space.
    type Vertices = [[f32; 3]; 3];

    /// The words of a vertex: position x, y and w (z 0), `values` and
    /// `flat`.
    fn words([x, y, w]: [f32; 3], values: [f32; 4], flat: u32) -> Streamed {
        let mut out = [flat; 9];
        for (o, v) in out.iter_mut().zip([x, y, 0.0, w].into_iter().chain(values)) {
            *o = v.to_bits();
        }
        out
    }

    /// The probe vertex shader for `inputs` (three positions each: x, y and
    /// w in clip space; [`input_values`] and [`flat`]), and its stage.
    fn streaming_shaders(inputs: &[Vertices]) -> (Vec<u8>, Vec<u8>) {
        let (mut positions, mut values, mut flats) = (Vec::new(), Vec::new(), Vec::new());
        for (i, input) in inputs.iter().enumerate() {
            for (k, &[x, y, w]) in input.iter().enumerate() {
                positions.push([x, y, 0.0, w]);
                values.push(input_values(i, k));
                flats.push(flat(i, k));
            }
        }
        let vs = probe::vertex(&positions, &values, &flats);
        let gs = geometry_stage(&[], &vs, None).expect("a geometry stage");
        (vs, gs)
    }

    /// `streamed`'s triangles by the input they come from, told by their
    /// first vertex's flat value: each starts at its input's v0.
    fn by_input(streamed: &[Streamed], inputs: usize) -> Vec<Vec<[Streamed; 3]>> {
        let mut by = vec![Vec::new(); inputs];
        for t in streamed.chunks(3) {
            let first = t[0][8] as usize;
            assert_eq!(first % 100, 1, "a triangle that starts at v0 ({first})");
            by[first / 100].push([t[0], t[1], t[2]]);
        }
        by
    }

    /// The stage's triangles, counted by stream output: a finite rectangle
    /// with an area gives two (a corner at −0 beside +0 is one: the
    /// rasterizer makes them one place). A NaN or an infinity in an x, y or
    /// w, two vertices at one place, three on one line, or w that differ,
    /// give the triangle as it is: one, `v0, v1, v2`.
    #[test]
    fn only_finite_rectangles_with_an_area_get_a_second_triangle() {
        let test = "rectangles::geometry::tests::only_finite_rectangles_with_an_area_get_a_second_triangle";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (nan, inf) = (f32::NAN, f32::INFINITY);
        let inputs: [(&str, Vertices, usize); 14] = [
            (
                "a rectangle",
                [[0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0]],
                2,
            ),
            (
                "a corner at x -0 beside +0",
                [[-0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0]],
                2,
            ),
            (
                "a corner at y -0 beside +0",
                [[0.0, -0.0, 1.0], [0.0, 0.5, 1.0], [0.5, 0.0, 1.0]],
                2,
            ),
            (
                "a NaN x where the corner's equalities hold",
                [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [nan, 0.0, 1.0]],
                1,
            ),
            (
                "a NaN y",
                [[0.0, 0.0, 1.0], [0.0, nan, 1.0], [1.0, 0.0, 1.0]],
                1,
            ),
            (
                "an infinite x",
                [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [inf, 0.0, 1.0]],
                1,
            ),
            (
                "an infinite corner",
                [[inf, 0.0, 1.0], [inf, 1.0, 1.0], [0.0, 0.0, 1.0]],
                1,
            ),
            (
                "w NaN",
                [[0.0, 0.0, nan], [0.0, 1.0, nan], [1.0, 0.0, nan]],
                1,
            ),
            (
                "w infinite",
                [[0.0, 0.0, inf], [0.0, 1.0, inf], [1.0, 0.0, inf]],
                1,
            ),
            (
                "two vertices at one place",
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                1,
            ),
            (
                "two vertices at one place, x +0 and -0",
                [[0.0, 0.0, 1.0], [-0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                1,
            ),
            ("three vertices at one place", [[0.5, 0.5, 1.0]; 3], 1),
            (
                "three vertices on one line",
                [[0.0, 0.0, 1.0], [0.0, 0.5, 1.0], [0.0, 1.0, 1.0]],
                1,
            ),
            (
                "w that differ",
                [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0]],
                1,
            ),
        ];
        let positions: Vec<Vertices> = inputs.iter().map(|i| i.1).collect();
        let (vs, gs) = streaming_shaders(&positions);
        let streamed = stream_out(host, &vs, &gs, 3 * inputs.len() as u32);
        let by = by_input(&streamed, inputs.len());
        for (i, &(what, input, want)) in inputs.iter().enumerate() {
            assert_eq!(by[i].len(), want, "{what}: triangles");
            if want == 1 {
                let as_is = [0, 1, 2].map(|k| words(input[k], input_values(i, k), flat(i, k)));
                assert_eq!(by[i][0], as_is, "{what}: the triangle as it is");
            }
        }
    }

    /// The fourth corner's position is its neighbours' own x and y and
    /// their w: 2^-25 beside 1 (lost in 1 + 2^-25 − 1) and w 2^127 (w + w
    /// overflows) come out exact, for each vertex as the corner and either
    /// neighbour sharing its x. Its other float outputs are
    /// `a + b − corner`, its flat one v0's; both triangles start at v0 and
    /// turn as `v0, v1, v2` (the module's order).
    #[test]
    fn the_fourth_corner_takes_x_y_and_w_from_its_neighbours() {
        let test =
            "rectangles::geometry::tests::the_fourth_corner_takes_x_y_and_w_from_its_neighbours";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (tiny, big, huge) = (2f32.powi(-25), 2f32.powi(126), 2f32.powi(127));
        // The three vertices (x, y, w), which is the corner, and the fourth
        // corner's x, y and w.
        let inputs: [(&str, Vertices, usize, [f32; 3]); 6] = [
            (
                "corner v0, its x shared with v1",
                [[1.0, 0.0, 1.0], [1.0, 0.5, 1.0], [tiny, 0.0, 1.0]],
                0,
                [tiny, 0.5, 1.0],
            ),
            (
                "corner v0, its x shared with v2",
                [[1.0, 0.0, 1.0], [tiny, 0.0, 1.0], [1.0, 0.5, 1.0]],
                0,
                [tiny, 0.5, 1.0],
            ),
            (
                "corner v1, its x shared with v2",
                [[0.75, 0.25, 1.0], [0.25, 0.25, 1.0], [0.25, -0.5, 1.0]],
                1,
                [0.75, -0.5, 1.0],
            ),
            (
                "corner v2, its x shared with v1",
                [[0.3, -0.5, 1.0], [-0.5, 0.75, 1.0], [-0.5, -0.5, 1.0]],
                2,
                [0.3, 0.75, 1.0],
            ),
            (
                "w 2^127, corner v0",
                [[0.0, 0.0, huge], [0.0, big, huge], [big, 0.0, huge]],
                0,
                [big, big, huge],
            ),
            (
                "w 2^127, corner v2",
                [[0.0, big, huge], [big, 0.0, huge], [0.0, 0.0, huge]],
                2,
                [big, big, huge],
            ),
        ];
        let positions: Vec<Vertices> = inputs.iter().map(|i| i.1).collect();
        let (vs, gs) = streaming_shaders(&positions);
        let streamed = stream_out(host, &vs, &gs, 3 * inputs.len() as u32);
        let by = by_input(&streamed, inputs.len());
        // The triangles, by the corner: 0, 1 and 2 are v0, v1 and v2, 3 is
        // the fourth corner.
        let orders = [
            [[0, 1, 3], [0, 3, 2]],
            [[0, 1, 2], [0, 2, 3]],
            [[0, 1, 2], [0, 3, 1]],
        ];
        for (i, &(what, input, corner, d)) in inputs.iter().enumerate() {
            let (a, b) = ((corner + 1) % 3, (corner + 2) % 3);
            let (va, vb, vc) = (
                input_values(i, a),
                input_values(i, b),
                input_values(i, corner),
            );
            let dv = [0, 1, 2, 3].map(|c| (va[c] + vb[c]) - vc[c]);
            let vertex = |label: usize| {
                if label == 3 {
                    words(d, dv, flat(i, 0))
                } else {
                    words(input[label], input_values(i, label), flat(i, label))
                }
            };
            let want: Vec<[Streamed; 3]> = orders[corner].iter().map(|t| t.map(&vertex)).collect();
            assert_eq!(by[i], want, "{what}");
        }
    }

    /// The indirect path: one record of the five cases (3 vertices each),
    /// rewritten by [`crate::rectangles::expand`] for a pipeline without
    /// per-vertex inputs (the record as the title wrote it), then run by
    /// `ExecuteIndirect` with the stage: the same picture as the direct
    /// draw.
    #[test]
    fn indirect_rectangles_without_vertex_inputs_get_their_fourth_corner() {
        let _pool = crate::rectangles::POOL_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let test = "rectangles::geometry::tests::indirect_rectangles_without_vertex_inputs_get_their_fourth_corner";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let cases = cases();
        let (vs, ps) = (vertex_shader(&cases, None), probe::pixel());
        let gs = stage(&vs, &ps);
        let root = empty_root(&host.device, D3D12_ROOT_SIGNATURE_FLAG_NONE);
        let pso = pipeline(
            &host.device,
            &root,
            (&vs, Some(&gs), &ps),
            D3D12_CULL_MODE_NONE,
            1,
        );
        let record = [3 * cases.len() as u32, 1, 0, 0];
        let args = TitleBuffer::new(256);
        args.set_words(|i| record.get(i).copied().unwrap_or(0));
        let args_copy: Arc<Shadow> = args.shadow();
        let signature = draw_signature(&host.device);
        let target = Target::new(&host.device, 1);
        let list = TitleList::new();
        let mut held = None;
        let prepared = crate::rectangles::expand(
            list.host(),
            &mut held,
            &[],
            &[(0, 0, 0); 32],
            (
                crate::rectangles::Records::Title(&args_copy, args.address()),
                16,
                0..1,
            ),
            None,
            None,
            None,
            &mut Default::default(),
        )
        .expect("expanded");
        target.bind(list.host(), &root, &pso);
        prepared.bind_vertices(list.host());
        // SAFETY: an open list with everything bound; the arguments and
        // their count are the list's own, in the indirect argument state.
        unsafe {
            list.host().list.ExecuteIndirect(
                &signature,
                prepared.max,
                &prepared.args,
                prepared.offset,
                &prepared.args,
                prepared.count_offset,
            );
        }
        let pixels = target.read(host, &list);
        if let Some(s) = held.take() {
            crate::rectangles::give_back(s);
        }
        whole(&check(&pixels, &cases, "indirect", 0.0), &cases);
    }

    /// A command signature of draw arguments 16 bytes apart.
    fn draw_signature(device: &ID3D12Device) -> ID3D12CommandSignature {
        let argument = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 16,
            NumArgumentDescs: 1,
            pArgumentDescs: &argument,
            NodeMask: 0,
        };
        let mut signature: Option<ID3D12CommandSignature> = None;
        // SAFETY: a description that lives through the call; no root
        // arguments, so no root signature.
        unsafe {
            device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
        }
        .expect("a command signature");
        signature.expect("a command signature")
    }
}
