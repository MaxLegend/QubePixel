// =============================================================================
// QubePixel — SkyRenderer  (billboard sun/moon sprites)
// =============================================================================
//
// Renders sun and moon as textured billboards in the sky.
// - Loads textures from assets/textures/sky/{sun,moon}.png
// - Falls back to procedural quads (yellow circle for sun, blue-white for moon)
// - Billboards always face the camera
// =============================================================================

use glam::{Mat4, Vec3};
use crate::debug_log;
use crate::core::lighting::DayNightCycle;
use crate::screens::game_3d_pipeline::Camera;

// ---------------------------------------------------------------------------
// Billboard vertex — position + texcoord (20 bytes)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
struct BillboardVertex {
    position: [f32; 3],
    texcoord: [f32; 2],
}

// A simple quad (two triangles)
const BILLBOARD_VERTICES: [BillboardVertex; 4] = [
    BillboardVertex { position: [-0.5, -0.5, 0.0], texcoord: [0.0, 1.0] },
    BillboardVertex { position: [ 0.5, -0.5, 0.0], texcoord: [1.0, 1.0] },
    BillboardVertex { position: [ 0.5,  0.5, 0.0], texcoord: [1.0, 0.0] },
    BillboardVertex { position: [-0.5,  0.5, 0.0], texcoord: [0.0, 0.0] },
];

const BILLBOARD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

// ---------------------------------------------------------------------------
// Billboard WGSL shader
// ---------------------------------------------------------------------------
const BILLBOARD_SHADER: &str = r"
struct BillboardUniforms {
    mvp:   mat4x4<f32>,    // model-view-projection for the billboard
    color: vec4<f32>,      // tint colour + alpha
};

@group(0) @binding(0) var<uniform> u: BillboardUniforms;
@group(0) @binding(1) var bb_tex:     texture_2d<f32>;
@group(0) @binding(2) var bb_sampler: sampler;

struct VOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       v_texcoord: vec2<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) texcoord: vec2<f32>,
) -> VOut {
    var out: VOut;
    out.clip_pos   = u.mvp * vec4<f32>(position, 1.0);
    out.v_texcoord = texcoord;
    return out;
}

@fragment
fn fs_main(input: VOut) -> @location(0) vec4<f32> {
    let tex = textureSample(bb_tex, bb_sampler, input.v_texcoord);
    let final_color = tex * u.color;
    if (final_color.a < 0.01) {
        discard;
    }
    return final_color;
}
";

// ---------------------------------------------------------------------------
// SkyRenderer
// ---------------------------------------------------------------------------
pub struct SkyRenderer {
    pipeline:       Option<wgpu::RenderPipeline>,
    bind_layout:    Option<wgpu::BindGroupLayout>,
    vertex_buffer:  Option<wgpu::Buffer>,
    index_buffer:   Option<wgpu::Buffer>,
    uniform_buffer: Option<wgpu::Buffer>,
    sampler:        Option<wgpu::Sampler>,

    sun_texture:    Option<wgpu::Texture>,
    sun_view:       Option<wgpu::TextureView>,
    moon_texture:   Option<wgpu::Texture>,
    moon_view:      Option<wgpu::TextureView>,

    initialised: bool,
}

const BB_UNIFORM_SIZE: u64 = 80; // mat4(64) + vec4(16)

impl SkyRenderer {
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_layout: None,
            vertex_buffer: None,
            index_buffer: None,
            uniform_buffer: None,
            sampler: None,
            sun_texture: None,
            sun_view: None,
            moon_texture: None,
            moon_view: None,
            initialised: false,
        }
    }

    // -----------------------------------------------------------------------
    // Lazy initialisation
    // -----------------------------------------------------------------------
    fn ensure_init(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) {
        if self.initialised { return; }
        self.initialised = true;

        debug_log!("SkyRenderer", "ensure_init", "Initialising billboard pipeline");

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Billboard Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BILLBOARD_SHADER)),
        });

        // Bind group layout
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Billboard BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
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
            label:              Some("Billboard Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size:     0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<BillboardVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x3,  // position
                1 => Float32x2,  // texcoord
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Billboard Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  None, // Billboards should be visible from both sides
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None, // No depth test — always draw on top (rendered before terrain)
            multisample:   wgpu::MultisampleState::default(),
            cache:         None,
            multiview_mask: None,
        });

        // Buffers
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Billboard VB"),
            size:  std::mem::size_of_val(&BILLBOARD_VERTICES) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, unsafe {
            std::slice::from_raw_parts(
                BILLBOARD_VERTICES.as_ptr() as *const u8,
                std::mem::size_of_val(&BILLBOARD_VERTICES),
            )
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Billboard IB"),
            size:  std::mem::size_of_val(&BILLBOARD_INDICES) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&index_buffer, 0, unsafe {
            std::slice::from_raw_parts(
                BILLBOARD_INDICES.as_ptr() as *const u8,
                std::mem::size_of_val(&BILLBOARD_INDICES),
            )
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Billboard Uniforms"),
            size:  BB_UNIFORM_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("Billboard Sampler"),
            mag_filter:     wgpu::FilterMode::Linear,
            min_filter:     wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Load textures (or fallback)
        let sun_tex  = Self::load_or_fallback(device, queue, "assets/textures/sky/sun.png",  [255, 230, 80, 255]);
        let moon_tex = Self::load_or_fallback(device, queue, "assets/textures/sky/moon.png", [200, 210, 240, 255]);

        let sun_view  = sun_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let moon_view = moon_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.pipeline       = Some(pipeline);
        self.bind_layout    = Some(bind_layout);
        self.vertex_buffer  = Some(vertex_buffer);
        self.index_buffer   = Some(index_buffer);
        self.uniform_buffer = Some(uniform_buffer);
        self.sampler        = Some(sampler);
        self.sun_texture    = Some(sun_tex);
        self.sun_view       = Some(sun_view);
        self.moon_texture   = Some(moon_tex);
        self.moon_view      = Some(moon_view);
    }

    /// Load texture from file; on failure, generate a procedural circle.
    fn load_or_fallback(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        path:   &str,
        fallback_color: [u8; 4],
    ) -> wgpu::Texture {
        // Try loading from file
        if let Ok(data) = std::fs::read(path) {
            if let Ok(img) = image::load_from_memory(&data) {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label:           Some(path),
                    size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count:    1,
                    dimension:       wgpu::TextureDimension::D2,
                    format:          wgpu::TextureFormat::Rgba8UnormSrgb,
                    usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats:    &[],
                });
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture:   &texture,
                        mip_level: 0,
                        origin:    wgpu::Origin3d::ZERO,
                        aspect:    wgpu::TextureAspect::All,
                    },
                    &rgba,
                    wgpu::TexelCopyBufferLayout {
                        offset:         0,
                        bytes_per_row:  Some(4 * w),
                        rows_per_image: Some(h),
                    },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );
                debug_log!("SkyRenderer", "load_or_fallback", "Loaded texture from {}", path);
                return texture;
            }
        }

        // Fallback: generate a 32×32 circle
        debug_log!("SkyRenderer", "load_or_fallback",
            "Generating fallback texture for {}", path);
        Self::generate_circle_texture(device, queue, 32, fallback_color)
    }

    /// Generate a procedural circle texture for fallback.
    fn generate_circle_texture(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        size:   u32,
        color:  [u8; 4],
    ) -> wgpu::Texture {
        let mut pixels = vec![0u8; (size * size * 4) as usize];
        let center = size as f32 / 2.0;
        let radius = center * 0.85;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - center + 0.5;
                let dy = y as f32 - center + 0.5;
                let dist = (dx * dx + dy * dy).sqrt();
                let idx = ((y * size + x) * 4) as usize;

                if dist < radius {
                    // Soft edge
                    let alpha = if dist > radius - 2.0 {
                        ((radius - dist) / 2.0 * 255.0) as u8
                    } else {
                        color[3]
                    };
                    // Slight radial gradient for a glow effect
                    let brightness = 1.0 - (dist / radius) * 0.3;
                    pixels[idx]     = (color[0] as f32 * brightness) as u8;
                    pixels[idx + 1] = (color[1] as f32 * brightness) as u8;
                    pixels[idx + 2] = (color[2] as f32 * brightness) as u8;
                    pixels[idx + 3] = alpha;
                }
                // else: transparent (already 0)
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Fallback Circle"),
            size:            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(4 * size),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );

        texture
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------
    pub fn render(
        &mut self,
        encoder:    &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        format:     wgpu::TextureFormat,
        camera:     &Camera,
        cycle:      &DayNightCycle,
    ) {
        self.ensure_init(device, queue, format);

        let pipeline       = self.pipeline.as_ref().unwrap();
        let bind_layout    = self.bind_layout.as_ref().unwrap();
        let vertex_buffer  = self.vertex_buffer.as_ref().unwrap();
        let index_buffer   = self.index_buffer.as_ref().unwrap();
        let uniform_buffer = self.uniform_buffer.as_ref().unwrap();
        let sampler        = self.sampler.as_ref().unwrap();

        let vp = camera.view_projection_matrix();

        // Draw sun if above horizon
        let sun_i = cycle.sun_intensity();
        if sun_i > 0.01 {
            if let Some(sun_view) = &self.sun_view {
                let sun_col = cycle.sun_color();
                let alpha = (sun_i / 1.5).min(1.0);
                let color = [sun_col.x, sun_col.y, sun_col.z, alpha];
                let sun_offset = cycle.sun_billboard_offset();
                let model = Self::billboard_model(camera, sun_offset, 15.0);
                let mvp = vp * model;

                self.upload_and_draw(
                    encoder, color_view, device, queue,
                    pipeline, bind_layout, uniform_buffer,
                    vertex_buffer, index_buffer, sampler,
                    sun_view, &mvp, color,
                );
            }
        }

        // Draw moon if above horizon
        let moon_i = cycle.moon_intensity();
        if moon_i > 0.01 {
            if let Some(moon_view) = &self.moon_view {
                let moon_col = cycle.moon_color();
                let alpha = (moon_i / 0.15).min(1.0);
                let color = [moon_col.x, moon_col.y, moon_col.z, alpha];
                let moon_offset = cycle.moon_billboard_offset();
                let model = Self::billboard_model(camera, moon_offset, 10.0);
                let mvp = vp * model;

                self.upload_and_draw(
                    encoder, color_view, device, queue,
                    pipeline, bind_layout, uniform_buffer,
                    vertex_buffer, index_buffer, sampler,
                    moon_view, &mvp, color,
                );
            }
        }
    }

    /// Compute a billboard model matrix that faces the camera.
    fn billboard_model(camera: &Camera, offset: Vec3, scale: f32) -> Mat4 {
        let world_pos = camera.position + offset;

        // Camera's view vectors
        let forward = camera.forward();
        let right   = forward.cross(Vec3::Y).normalize();
        let up      = right.cross(forward).normalize();

        // Build a model matrix that orients the quad to face the camera
        Mat4::from_cols(
            (right * scale).extend(0.0),
            (up    * scale).extend(0.0),
            (-forward * scale).extend(0.0),
            world_pos.extend(1.0),
        )
    }

    fn upload_and_draw(
        &self,
        encoder:        &mut wgpu::CommandEncoder,
        color_view:     &wgpu::TextureView,
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        pipeline:       &wgpu::RenderPipeline,
        bind_layout:    &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        vertex_buffer:  &wgpu::Buffer,
        index_buffer:   &wgpu::Buffer,
        sampler:        &wgpu::Sampler,
        texture_view:   &wgpu::TextureView,
        mvp:            &Mat4,
        color:          [f32; 4],
    ) {
        // Upload uniforms
        let mut data = [0.0f32; 20];
        data[0..16].copy_from_slice(&mvp.to_cols_array());
        data[16..20].copy_from_slice(&color);
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, 80)
        };
        queue.write_buffer(uniform_buffer, 0, bytes);

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Billboard BG"),
            layout:  bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        // Render pass
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Billboard Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,  // Preserve existing sky colour
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes:        None,
            occlusion_query_set:     None,
            multiview_mask:          None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..6, 0, 0..1);
    }
}
