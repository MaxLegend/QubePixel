// =============================================================================
// QubePixel — HandBlockRenderer  (block-in-hand, attached to player's right hand)
// =============================================================================
//
// Renders the currently-held block as a small cube anchored to the *right
// forearm bone* of the player skeleton (rather than floating in front of the
// camera). This makes the block:
//   - Visible in both first-person and third-person views.
//   - Animated naturally by walk/swing animations (the bone already swings).
//   - Lit by the same PBR + VCT GI pipeline as the player entity, so voxel
//     shadows and tinted-glass GI affect it correctly.
//
// Pipeline: re-uses `entity_pbr.wgsl` (same group(1) VCT bind group as the
// player) so the block participates in coloured shadows and entity AABB casts
// out of the box. The atlas texture is bound as `albedo_tex` at group(0)#1
// and the cube's vertex UVs reference per-face atlas tiles.
// =============================================================================

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::player_model::PlayerVertex;
use crate::core::lighting::DayNightCycle;
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;
use crate::{debug_log, ext_debug_log};

const SHADER_SRC: &str = include_str!("../shaders/entity_pbr.wgsl");

/// EntityUniforms in entity_pbr.wgsl is 240 bytes — match exactly.
const UNIFORM_SIZE: u64 = 240;

/// Held block scale relative to a full world cube. Small enough to fit in
/// the player's hand silhouette.
const HAND_BLOCK_SCALE: f32 = 0.4;

/// Local-space offset of the cube centre from the right-hand grip pivot,
/// in *model space* (i.e. before the bone world transform). Tuned so the
/// cube sits at the fist and forward of the wrist.
/// x=-0.3125: arm centre pivot is at x=-5 model pixels = -5/16 blocks
/// y=0.75:    wrist bottom = y=12 model pixels = 12/16 blocks from foot
/// z=0.15:    slight forward offset so the block is in front of the arm
const GRIP_OFFSET: Vec3 = Vec3::new(-0.3125, 0.72, 0.15);

// ---------------------------------------------------------------------------
// HandBlockRenderer
// ---------------------------------------------------------------------------

pub struct HandBlockRenderer {
    pipeline:    wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group:  wgpu::BindGroup,
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
    /// Block registry key currently uploaded (to detect changes).
    current_block_id: String,
    atlas_layout: TextureAtlasLayout,
}

impl HandBlockRenderer {
    pub fn new(
        device:       &wgpu::Device,
        format:       wgpu::TextureFormat,
        atlas_view:   &wgpu::TextureView,
        atlas_layout: TextureAtlasLayout,
        vct_frag_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        debug_log!("HandBlockRenderer", "new",
            "Creating GI-lit hand block pipeline (entity_pbr shader)");

        // ── Shader ──────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("HandBlock Shader (entity_pbr)"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        // ── Bind group 0 layout (uniform + albedo + sampler) ─────────────────
        // Matches entity_pbr.wgsl group(0): EntityUniforms + albedo_tex + albedo_samp.
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HandBlock BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled:   false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("HandBlock Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl0), Some(vct_frag_bgl)],
            immediate_size:     0,
        });

        // ── Vertex layout (PlayerVertex: position, uv, normal) ───────────────
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PlayerVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 20, shader_location: 2 },
            ],
        };

        // ── Render pipeline ──────────────────────────────────────────────────
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("HandBlock Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers:     &[vb_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare:       Option::from(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // ── Uniform buffer (240 bytes = EntityUniforms) ──────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("HandBlock UB"),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Nearest-neighbour sampler for crisp pixel-art look.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("HandBlock Sampler"),
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:  Some("HandBlock BG0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(atlas_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        // Dummy GPU buffers (replaced on first `update_block`).
        let dummy_vert = PlayerVertex { position: [0.0; 3], uv: [0.0; 2], normal: [0.0; 3] };
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("HandBlock VB (dummy)"),
            contents: bytemuck::cast_slice(&[dummy_vert]),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("HandBlock IB (dummy)"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage:    wgpu::BufferUsages::INDEX,
        });

        Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            index_buf,
            index_count: 0,
            current_block_id: String::new(),
            atlas_layout,
        }
    }

    // -----------------------------------------------------------------------
    // Block mesh update
    // -----------------------------------------------------------------------

    /// Rebuild the cube mesh when the active block changes.
    pub fn update_block(
        &mut self,
        device:       &wgpu::Device,
        block_id:     Option<&str>,
        top_texture:  Option<&str>,
        side_texture: Option<&str>,
    ) {
        let key = block_id.unwrap_or("").to_string();
        if key == self.current_block_id { return; }
        self.current_block_id = key;

        if block_id.is_none() || (top_texture.is_none() && side_texture.is_none()) {
            self.index_count = 0;
            return;
        }

        let top_name  = top_texture.or(side_texture).unwrap_or("");
        let side_name = side_texture.or(top_texture).unwrap_or("");
        let top_uv  = self.atlas_layout.uv_for(top_name);
        let side_uv = self.atlas_layout.uv_for(side_name);

        let (vertices, indices) = build_cube_mesh(top_uv, side_uv);

        self.vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("HandBlock VB"),
            contents: bytemuck::cast_slice(&vertices),
            usage:    wgpu::BufferUsages::VERTEX,
        });
        self.index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("HandBlock IB"),
            contents: bytemuck::cast_slice(&indices),
            usage:    wgpu::BufferUsages::INDEX,
        });
        self.index_count = indices.len() as u32;
        ext_debug_log!("HandBlockRenderer", "update_block",
            "Rebuilt cube mesh for block '{}': verts={}, indices={}",
            self.current_block_id, vertices.len(), indices.len());
    }

    // -----------------------------------------------------------------------
    // Render — attached to right forearm bone transform
    // -----------------------------------------------------------------------

    /// Render the held block. `right_hand_xform` must be the right-forearm
    /// bone's world matrix (from `PlayerRenderer::bone_transforms`). The
    /// cube is positioned at `right_hand_xform * Translate(GRIP_OFFSET)` and
    /// scaled by `HAND_BLOCK_SCALE`.
    pub fn render(
        &self,
        encoder:           &mut wgpu::CommandEncoder,
        color_view:        &wgpu::TextureView,
        depth_view:        &wgpu::TextureView,
        queue:             &wgpu::Queue,
        view_proj:         Mat4,
        right_hand_xform:  Mat4,
        camera_pos:        Vec3,
        day_night:         &DayNightCycle,
        ambient_min:       f32,
        shadow_sun_enabled: bool,
        vct_bind_group:    &wgpu::BindGroup,
    ) {
        if self.index_count == 0 { return; }

        // ── Compose model matrix from the bone transform ─────────────────────
        // Convert local grip offset → world position via the bone xform, then
        // scale around that point so rotation/translation come from the bone.
        let model = right_hand_xform
            * Mat4::from_translation(GRIP_OFFSET)
            * Mat4::from_scale(Vec3::splat(HAND_BLOCK_SCALE));

        // ── Build EntityUniforms (matches entity_pbr.wgsl byte-for-byte) ─────
        let sun_dir       = day_night.sun_direction();
        let sun_color     = day_night.sun_color();
        let sun_intensity = day_night.sun_intensity();
        let moon_dir      = day_night.moon_direction();
        let moon_color    = day_night.moon_color();
        let moon_intens   = day_night.moon_intensity();
        let ambient       = day_night.ambient_color();

        let mut ub = [0f32; 60];
        ub[0..16].copy_from_slice(&view_proj.to_cols_array());
        ub[16..32].copy_from_slice(&model.to_cols_array());
        ub[32] = sun_dir.x;   ub[33] = sun_dir.y;   ub[34] = sun_dir.z;   ub[35] = sun_intensity;
        ub[36] = sun_color.x; ub[37] = sun_color.y; ub[38] = sun_color.z; ub[39] = 0.0;
        ub[40] = moon_dir.x;  ub[41] = moon_dir.y;  ub[42] = moon_dir.z;  ub[43] = moon_intens;
        ub[44] = moon_color.x;ub[45] = moon_color.y;ub[46] = moon_color.z;ub[47] = 0.0;
        ub[48] = ambient.x;   ub[49] = ambient.y;   ub[50] = ambient.z;   ub[51] = ambient_min;
        ub[52] = camera_pos.x;ub[53] = camera_pos.y;ub[54] = camera_pos.z;ub[55] = 0.0;
        ub[56] = if shadow_sun_enabled { 1.0 } else { 0.0 };
        ub[57] = 0.05; ub[58] = 0.0; ub[59] = 0.0; // shadow normal-offset bias
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&ub));

        // ── Render pass ──────────────────────────────────────────────────────
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("HandBlock Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            multiview_mask: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_bind_group(1, vct_bind_group, &[]);
        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

// ---------------------------------------------------------------------------
// Cube mesh generation
// ---------------------------------------------------------------------------

/// Build a unit cube (±0.5) with atlas UVs for top and side faces.
/// Returns 24 vertices and 36 indices (6 faces × 4 verts × 6 indices).
fn build_cube_mesh(
    top_uv:  (f32, f32, f32, f32),
    side_uv: (f32, f32, f32, f32),
) -> (Vec<PlayerVertex>, Vec<u32>) {
    let mut v = Vec::with_capacity(24);
    let mut idx = Vec::with_capacity(36);
    let s = 0.5f32;

    let mut quad = |p0:[f32;3], p1:[f32;3], p2:[f32;3], p3:[f32;3],
                    n:[f32;3], uv:(f32,f32,f32,f32)| {
        let base = v.len() as u32;
        v.push(PlayerVertex { position: p0, uv: [uv.0, uv.1], normal: n });
        v.push(PlayerVertex { position: p1, uv: [uv.2, uv.1], normal: n });
        v.push(PlayerVertex { position: p2, uv: [uv.2, uv.3], normal: n });
        v.push(PlayerVertex { position: p3, uv: [uv.0, uv.3], normal: n });
        idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +Y (top)   — CCW from above; geometric normal = +Y ✓
    quad([ s, s, s], [ s, s,-s], [-s, s,-s], [-s, s, s], [ 0.0, 1.0, 0.0], top_uv);
    // -Y (bottom) — CCW from below; geometric normal = -Y ✓
    quad([-s,-s,-s], [ s,-s,-s], [ s,-s, s], [-s,-s, s], [ 0.0,-1.0, 0.0], side_uv);
    // +Z (south / front) — CCW from front; geometric normal = +Z ✓
    quad([ s, s, s], [-s, s, s], [-s,-s, s], [ s,-s, s], [ 0.0, 0.0, 1.0], side_uv);
    // -Z (north / back)  — CCW from back; geometric normal = -Z ✓
    quad([-s, s,-s], [ s, s,-s], [ s,-s,-s], [-s,-s,-s], [ 0.0, 0.0,-1.0], side_uv);
    // +X (east / right)  — CCW from right; geometric normal = +X ✓
    quad([ s, s,-s], [ s, s, s], [ s,-s, s], [ s,-s,-s], [ 1.0, 0.0, 0.0], side_uv);
    // -X (west / left)   — CCW from left; geometric normal = -X ✓
    quad([-s, s, s], [-s, s,-s], [-s,-s,-s], [-s,-s, s], [-1.0, 0.0, 0.0], side_uv);

    (v, idx)
}
