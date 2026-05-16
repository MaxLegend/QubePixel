// =============================================================================
// QubePixel — VolumetricRenderer
// =============================================================================
// Renders halos (soft Gaussian billboard) and god rays for analytical lights.
// Uses additive blending with depth testing (no depth write) so effects are
// occluded by solid geometry but do not write to the depth buffer.
//
// Halos: one billboard quad per light source.
// Rays:  one quad per ray; used for spot/directional lights.
//
// Shader source: src/core/radiance_cascades/shaders/volumetric_lights.wgsl

use bytemuck;
use glam::{Mat4, Vec3};
use wgpu;

use super::dynamic_lights::{PointLightGPU, SpotLightGPU};

// ---------------------------------------------------------------------------
// GPU-side structs (must match WGSL layout byte-for-byte)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VolumetricUniforms {
    view_proj:   [f32; 16],  // mat4 — 64 bytes
    cam_right:   [f32; 4],   // vec4 — 16 bytes (.xyz used)
    cam_up:      [f32; 4],   // vec4 — 16 bytes (.xyz used)
    time_params: [f32; 4],   // vec4 — 16 bytes (.x = elapsed secs)
    _pad:        [f32; 4],   // vec4 — 16 bytes (alignment)
}
const _: () = assert!(std::mem::size_of::<VolumetricUniforms>() == 128);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HaloInstance {
    world_pos_radius: [f32; 4],   // xyz = world centre, w = halo_radius
    color_intensity:  [f32; 4],   // rgb = colour, w = halo_intensity (0..1)
}
const _: () = assert!(std::mem::size_of::<HaloInstance>() == 32);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RayInstance {
    origin_len:   [f32; 4],   // xyz = ray origin, w = ray length
    dir_width:    [f32; 4],   // xyz = direction (normalised), w = width
    color_int:    [f32; 4],   // rgb = colour, w = intensity (0..1)
    falloff_time: [f32; 4],   // x = falloff exp, y = per-ray phase offset
}
const _: () = assert!(std::mem::size_of::<RayInstance>() == 64);

// ---------------------------------------------------------------------------
// VolumetricRenderer
// ---------------------------------------------------------------------------

pub struct VolumetricRenderer {
    halo_pipeline: wgpu::RenderPipeline,
    ray_pipeline:  wgpu::RenderPipeline,

    uniform_buf: wgpu::Buffer,
    halo_buf:    wgpu::Buffer,
    ray_buf:     wgpu::Buffer,

    shared_bg:        wgpu::BindGroup,
    halo_instance_bg:  wgpu::BindGroup,
    ray_instance_bg:   wgpu::BindGroup,

    max_halos: usize,
    max_rays:  usize,
}

impl VolumetricRenderer {
    const MAX_HALOS: usize = 128;
    const MAX_RAYS:  usize = 512;

    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("volumetric_lights"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../core/radiance_cascades/shaders/volumetric_lights.wgsl").into()
            ),
        });

        // -- Bind group layout 0: shared uniforms --
        let shared_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("vol_shared_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        // -- Bind group layout 1: storage instance buffer (same layout, reused for both passes) --
        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("vol_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty:         wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let shared_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("vol_layout"),
            bind_group_layouts: &[Some(&shared_bgl), Some(&instance_bgl)],
            immediate_size:     0,
        });

        let additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
        };

        let depth_ro = wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Option::from(false),
            depth_compare:       Option::from(wgpu::CompareFunction::Less),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        };

        let halo_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("halo_pipeline"),
            layout: Some(&shared_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_halo"),
                buffers:     &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_halo"),
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     color_format,
                    blend:      Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil:  Some(depth_ro.clone()),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        let ray_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("ray_pipeline"),
            layout: Some(&shared_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_ray"),
                buffers:     &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_ray"),
                targets:     &[Some(wgpu::ColorTargetState {
                    format:     color_format,
                    blend:      Some(additive_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil:  Some(depth_ro),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        // -- Buffers --
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_uniform_buf"),
            size:               std::mem::size_of::<VolumetricUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let halo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_halo_buf"),
            size:               (Self::MAX_HALOS * std::mem::size_of::<HaloInstance>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vol_ray_buf"),
            size:               (Self::MAX_RAYS * std::mem::size_of::<RayInstance>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // -- Bind groups --
        let shared_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("vol_shared_bg"),
            layout:  &shared_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let halo_instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("vol_halo_instance_bg"),
            layout:  &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: halo_buf.as_entire_binding(),
            }],
        });

        let ray_instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("vol_ray_instance_bg"),
            layout:  &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: ray_buf.as_entire_binding(),
            }],
        });

        Self {
            halo_pipeline,
            ray_pipeline,
            uniform_buf,
            halo_buf,
            ray_buf,
            shared_bg,
            halo_instance_bg,
            ray_instance_bg,
            max_halos: Self::MAX_HALOS,
            max_rays:  Self::MAX_RAYS,
        }
    }

    /// Render halos + rays for all analytical lights this frame.
    pub fn render(
        &self,
        encoder:      &mut wgpu::CommandEncoder,
        view:         &wgpu::TextureView,
        depth_view:   &wgpu::TextureView,
        queue:        &wgpu::Queue,
        view_proj:    Mat4,
        cam_right:    Vec3,
        cam_up:       Vec3,
        time_secs:    f32,
        point_lights: &[PointLightGPU],
        spot_lights:  &[SpotLightGPU],
    ) {
        let halo_on = crate::core::config::halo_enabled();
        let rays_on = crate::core::config::volumetric_rays_enabled();

        // -- Build halo instance list --
        let mut halos: Vec<HaloInstance> = Vec::new();

        if halo_on {
            for pl in point_lights {
                if halos.len() >= self.max_halos { break; }
                let range     = pl.pos_range[3];
                let intensity = pl.color_intensity[3];
                let halo_r    = (range * 0.18).clamp(0.5, 6.0);
                let halo_i    = (intensity * 0.05).clamp(0.0, 0.75);
                halos.push(HaloInstance {
                    world_pos_radius: [pl.pos_range[0], pl.pos_range[1], pl.pos_range[2], halo_r],
                    color_intensity:  [pl.color_intensity[0], pl.color_intensity[1], pl.color_intensity[2], halo_i],
                });
            }

            for sl in spot_lights {
                if halos.len() >= self.max_halos { break; }
                let range     = sl.pos_range[3];
                let intensity = sl.color_intensity[3];
                let halo_r    = (range * 0.12).clamp(0.4, 4.0);
                let halo_i    = (intensity * 0.04).clamp(0.0, 0.6);
                halos.push(HaloInstance {
                    world_pos_radius: [sl.pos_range[0], sl.pos_range[1], sl.pos_range[2], halo_r],
                    color_intensity:  [sl.color_intensity[0], sl.color_intensity[1], sl.color_intensity[2], halo_i],
                });
            }
        }

        // -- Build ray instance list (spot lights → directional god rays) --
        let mut rays: Vec<RayInstance> = Vec::new();

        if rays_on {
            for (i, sl) in spot_lights.iter().enumerate() {
                if rays.len() >= self.max_rays { break; }
                let dir       = Vec3::new(sl.dir_inner[0], sl.dir_inner[1], sl.dir_inner[2]);
                let range     = sl.pos_range[3];
                let intensity = sl.color_intensity[3];
                let ray_len   = range.min(12.0);
                let ray_width = (range * 0.04).clamp(0.2, 1.5);
                let ray_i     = (intensity * 0.03).clamp(0.0, 0.5);
                rays.push(RayInstance {
                    origin_len:   [sl.pos_range[0], sl.pos_range[1], sl.pos_range[2], ray_len],
                    dir_width:    [dir.x, dir.y, dir.z, ray_width],
                    color_int:    [sl.color_intensity[0], sl.color_intensity[1], sl.color_intensity[2], ray_i],
                    falloff_time: [2.0, i as f32 * 1.3, 0.0, 0.0],
                });
            }
        }

        let halo_count = halos.len() as u32;
        let ray_count  = rays.len() as u32;

        if halo_count == 0 && ray_count == 0 { return; }

        // -- Upload uniform buffer --
        let uniforms = VolumetricUniforms {
            view_proj:   view_proj.to_cols_array(),
            cam_right:   [cam_right.x, cam_right.y, cam_right.z, 0.0],
            cam_up:      [cam_up.x,    cam_up.y,    cam_up.z,    0.0],
            time_params: [time_secs, 0.0, 0.0, 0.0],
            _pad:        [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        if halo_count > 0 {
            queue.write_buffer(&self.halo_buf, 0, bytemuck::cast_slice(&halos));
        }
        if ray_count > 0 {
            queue.write_buffer(&self.ray_buf, 0, bytemuck::cast_slice(&rays));
        }

        // -- Render pass: composite on top, depth test enabled, no depth write --
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:      Some("volumetric_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           view,
                resolve_target: None,
                ops:            wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:        depth_view,
                depth_ops:   Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        if halo_count > 0 {
            pass.set_pipeline(&self.halo_pipeline);
            pass.set_bind_group(0, &self.shared_bg, &[]);
            pass.set_bind_group(1, &self.halo_instance_bg, &[]);
            pass.draw(0..4, 0..halo_count);
        }

        if ray_count > 0 {
            pass.set_pipeline(&self.ray_pipeline);
            pass.set_bind_group(0, &self.shared_bg, &[]);
            pass.set_bind_group(1, &self.ray_instance_bg, &[]);
            pass.draw(0..4, 0..ray_count);
        }
    }
}
