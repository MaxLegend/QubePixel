// =============================================================================
// QubePixel — VolumetricLightPass
// =============================================================================
//
// Renders additive-blended halos and god-ray quads for emissive blocks.
//
// Pipeline overview (runs AFTER the main 3D pass):
//   1. CPU gathers visible emissive blocks → builds HaloInstance + RayInstance lists.
//   2. GPU buffers are updated with the new instance data.
//   3. Two draw calls:
//      a. Halo pass  — camera-facing billboard quad per light.
//      b. Ray pass   — N thin quads per light, radiating outward.
//   Both use additive blending + depth-test-read (no depth write).
//
// WGSL struct alignment (matches volumetric_lights.wgsl exactly):
//   VolumetricUniforms : 128 bytes
//   HaloInstance       :  32 bytes (2 × vec4)
//   RayInstance        :  64 bytes (4 × vec4)

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;
use std::f32::consts::TAU;

// ---------------------------------------------------------------------------
// GPU structs — must match volumetric_lights.wgsl byte-for-byte
// ---------------------------------------------------------------------------

/// Shared per-frame uniforms (128 bytes, binding 0).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumetricUniforms {
    pub view_proj:   [f32; 16],  //  0..64
    pub cam_right:   [f32;  4],  // 64..80
    pub cam_up:      [f32;  4],  // 80..96
    pub time_params: [f32;  4],  // 96..112  .x = elapsed seconds
    pub _pad:        [f32;  4],  // 112..128
}

/// One halo billboard (32 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HaloInstance {
    /// .xyz = world-space block centre, .w = halo radius
    pub world_pos_radius: [f32; 4],
    /// .rgb = emission colour, .w = intensity
    pub color_intensity:  [f32; 4],
}

/// One god-ray quad (64 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayInstance {
    /// .xyz = ray origin, .w = ray length
    pub origin_len:   [f32; 4],
    /// .xyz = direction (normalised), .w = ray width
    pub dir_width:    [f32; 4],
    /// .rgb = colour, .w = intensity
    pub color_int:    [f32; 4],
    /// .x = falloff exponent, .y = per-ray phase offset for wobble
    pub falloff_time: [f32; 4],
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SHADER_SOURCE: &str = include_str!(
    "radiance_cascades/shaders/volumetric_lights.wgsl"
);

/// Maximum number of halo instances buffered on the GPU.
const MAX_HALOS: usize = 256;
/// Maximum number of ray instances buffered on the GPU.
const MAX_RAYS: usize = 2048;
/// Radius (blocks) within which emissive blocks generate volumetric effects.
pub const VOLUMETRIC_RADIUS: i32 = 48;
/// Maximum separate lights included in a single frame (performance cap).
const MAX_LIGHTS: usize = 128;

// ---------------------------------------------------------------------------
// VolumetricLightData — fully resolved per-light data passed to update()
// ---------------------------------------------------------------------------

/// Resolved volumetric effect parameters for one emissive block.
/// Built by game_screen.rs from `EmissiveBlockPos` + `BlockRegistry`.
pub struct VolumetricLightData {
    pub position:      [f32; 3],  // world-space block centre
    pub color:         [f32; 3],  // emission colour (from emission.light_color)
    pub halo_enabled:  bool,
    pub halo_radius:   f32,
    pub halo_intensity: f32,
    pub ray_enabled:   bool,
    pub ray_count:     u32,
    pub ray_length:    f32,
    pub ray_width:     f32,
    pub ray_intensity: f32,
    pub ray_falloff:   f32,
}

// ---------------------------------------------------------------------------
// VolumetricLightPass
// ---------------------------------------------------------------------------

pub struct VolumetricLightPass {
    // Halo pipeline
    halo_pipeline:  wgpu::RenderPipeline,
    halo_instances: Vec<HaloInstance>,
    halo_buf:       wgpu::Buffer,

    // Ray pipeline
    ray_pipeline:   wgpu::RenderPipeline,
    ray_instances:  Vec<RayInstance>,
    ray_buf:        wgpu::Buffer,

    // Static index buffer (quad: [0,1,2, 0,2,3]) shared by both passes
    index_buf:      wgpu::Buffer,

    // Shared uniform (view_proj + camera vectors + time)
    uniform_buf:    wgpu::Buffer,
    uniform_bgl:    wgpu::BindGroupLayout,
    uniform_bg:     wgpu::BindGroup,

    // Per-pass storage buffers (halos / rays)
    halo_bgl:       wgpu::BindGroupLayout,
    halo_bg:        Option<wgpu::BindGroup>,
    ray_bgl:        wgpu::BindGroupLayout,
    ray_bg:         Option<wgpu::BindGroup>,
}

impl VolumetricLightPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        // ---- Shared uniform buffer ----
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("VolumetricUniforms"),
            size:               std::mem::size_of::<VolumetricUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Volumetric Uniform BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Volumetric Uniform BG"),
            layout:  &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ---- Storage BGL (shared layout for halos and rays) ----
        let make_storage_bgl = |device: &wgpu::Device, label: &str| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label:   Some(label),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                }],
            })
        };

        let halo_bgl = make_storage_bgl(device, "Halo Instance BGL");
        let ray_bgl  = make_storage_bgl(device, "Ray Instance BGL");

        // ---- Halo GPU buffer (pre-allocated, overwritten each frame) ----
        let halo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Halo Instance Buf"),
            size:               (MAX_HALOS * std::mem::size_of::<HaloInstance>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Ray GPU buffer ----
        let ray_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Ray Instance Buf"),
            size:               (MAX_RAYS * std::mem::size_of::<RayInstance>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Static quad index buffer [0,1,2, 0,2,3] ----
        let quad_indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Volumetric Quad IB"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage:    wgpu::BufferUsages::INDEX,
        });

        // ---- Compile shader ----
        // The shader file uses `@group(1) @binding(0)` for the instance storage buffer.
        // We create two separate render pipelines — one with vs_halo/fs_halo entry
        // points and one with vs_ray/fs_ray entry points.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Volumetric Lights Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Volumetric Pipeline Layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&halo_bgl)],
            immediate_size:     0,
        });

        let blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,   // additive
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
        };

        let make_depth_state = || wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Option::from(false),  // read depth, don't write
            depth_compare:       Option::from(wgpu::CompareFunction::Less),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        };

        let make_pipeline = |device: &wgpu::Device,
                             vs_entry: &str,
                             fs_entry: &str,
                             layout:   &wgpu::PipelineLayout|
                             -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label:  Some("Volumetric Pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module:              &shader,
                    entry_point:         Some(vs_entry),
                    buffers:             &[],  // fully procedural
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module:              &shader,
                    entry_point:         Some(fs_entry),
                    targets:             &[Some(wgpu::ColorTargetState {
                        format:     surface_format,
                        blend:      Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology:  wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil:  Some(make_depth_state()),
                multisample:    wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache:          None,
            })
        };

        let halo_pipeline = make_pipeline(device, "vs_halo", "fs_halo", &pipeline_layout);

        // Ray pipeline uses same layout but different BGL for instances
        let ray_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Volumetric Ray Pipeline Layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&ray_bgl)],
            immediate_size:     0,
        });
        let ray_pipeline = make_pipeline(device, "vs_ray", "fs_ray", &ray_layout);

        Self {
            halo_pipeline,
            halo_instances: Vec::new(),
            halo_buf,
            ray_pipeline,
            ray_instances: Vec::new(),
            ray_buf,
            index_buf,
            uniform_buf,
            uniform_bgl,
            uniform_bg,
            halo_bgl,
            halo_bg: None,
            ray_bgl,
            ray_bg: None,
        }
    }

    // -----------------------------------------------------------------------
    // update — called each frame from game_screen before render()
    // -----------------------------------------------------------------------

    /// Rebuild the instance lists from pre-resolved volumetric light data.
    ///
    /// * `lights`      — resolved lights (built by game_screen from emissive block list).
    /// * `view_proj`   — current frame's view-projection matrix.
    /// * `cam_right`   — world-space camera right vector.
    /// * `cam_up`      — world-space camera up vector.
    /// * `elapsed_s`   — seconds since game start (for ray wobble animation).
    pub fn update(
        &mut self,
        device:    &wgpu::Device,
        queue:     &wgpu::Queue,
        lights:    &[VolumetricLightData],
        view_proj: &Mat4,
        cam_right: Vec3,
        cam_up:    Vec3,
        elapsed_s: f32,
    ) {
        self.halo_instances.clear();
        self.ray_instances.clear();

        for light in lights.iter().take(MAX_LIGHTS) {
            let [wx, wy, wz] = light.position;
            let [cr, cg, cb] = light.color;

            // ---- Halo ----
            if light.halo_enabled && self.halo_instances.len() < MAX_HALOS {
                self.halo_instances.push(HaloInstance {
                    world_pos_radius: [wx, wy, wz, light.halo_radius],
                    color_intensity:  [cr, cg, cb, light.halo_intensity],
                });
            }

            // ---- Rays ----
            if light.ray_enabled && self.ray_instances.len() + light.ray_count as usize <= MAX_RAYS {
                let n = light.ray_count.max(1).min(32);
                for i in 0..n {
                    let theta = (i as f32 / n as f32) * TAU;
                    // Spread rays across upper hemisphere: fixed elevation of 30°
                    // plus alternating ±15° to break symmetry
                    let elev = std::f32::consts::FRAC_PI_6
                        + if i % 2 == 0 { 0.26 } else { -0.26 };
                    let dir = Vec3::new(
                        theta.cos() * elev.cos(),
                        elev.sin(),
                        theta.sin() * elev.cos(),
                    ).normalize();

                    let phase = theta + (i as f32) * 0.73;
                    self.ray_instances.push(RayInstance {
                        origin_len:   [wx, wy, wz, light.ray_length],
                        dir_width:    [dir.x, dir.y, dir.z, light.ray_width],
                        color_int:    [cr, cg, cb, light.ray_intensity],
                        falloff_time: [light.ray_falloff, phase, 0.0, 0.0],
                    });
                }
            }
        }

        // ---- Upload uniform ----
        let vp = view_proj.to_cols_array();
        let uniforms = VolumetricUniforms {
            view_proj:   vp,
            cam_right:   [cam_right.x, cam_right.y, cam_right.z, 0.0],
            cam_up:      [cam_up.x,    cam_up.y,    cam_up.z,    0.0],
            time_params: [elapsed_s, 0.0, 0.0, 0.0],
            _pad:        [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // ---- Upload halos ----
        if !self.halo_instances.is_empty() {
            let bytes = bytemuck::cast_slice(&self.halo_instances);
            queue.write_buffer(&self.halo_buf, 0, bytes);

            self.halo_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("Halo Instance BG"),
                layout:  &self.halo_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.halo_buf,
                        offset: 0,
                        size:   wgpu::BufferSize::new(bytes.len() as u64),
                    }),
                }],
            }));
        } else {
            self.halo_bg = None;
        }

        // ---- Upload rays ----
        if !self.ray_instances.is_empty() {
            let bytes = bytemuck::cast_slice(&self.ray_instances);
            queue.write_buffer(&self.ray_buf, 0, bytes);

            self.ray_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some("Ray Instance BG"),
                layout:  &self.ray_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.ray_buf,
                        offset: 0,
                        size:   wgpu::BufferSize::new(bytes.len() as u64),
                    }),
                }],
            }));
        } else {
            self.ray_bg = None;
        }
    }

    // -----------------------------------------------------------------------
    // render — called each frame AFTER the main 3D pass
    // -----------------------------------------------------------------------

    /// Draw all volumetric effects into `color_view`, reading (not writing) `depth_view`.
    pub fn render(
        &self,
        encoder:    &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        if self.halo_instances.is_empty() && self.ray_instances.is_empty() {
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label:             Some("Volumetric Lights Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,   // composite on top of 3D scene
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:        depth_view,
                depth_ops:   Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store, // preserve depth
                }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,
        });

        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(0, &self.uniform_bg, &[]);

        // ---- Halo pass ----
        if let Some(ref bg) = self.halo_bg {
            pass.set_pipeline(&self.halo_pipeline);
            pass.set_bind_group(1, bg, &[]);
            // 4 vertices per quad (index buffer = [0,1,2, 0,2,3] = 6 indices)
            pass.draw_indexed(0..6, 0, 0..self.halo_instances.len() as u32);
        }

        // ---- Ray pass ----
        if let Some(ref bg) = self.ray_bg {
            pass.set_pipeline(&self.ray_pipeline);
            pass.set_bind_group(1, bg, &[]);
            pass.draw_indexed(0..6, 0, 0..self.ray_instances.len() as u32);
        }
    }
}
