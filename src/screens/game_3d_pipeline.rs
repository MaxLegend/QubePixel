// =============================================================================
// QubePixel — Game3DPipeline  (camera, depth buffer, world chunk meshes)
// =============================================================================

use crate::debug_log;
use glam::{Mat4, Vec3};

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------
pub struct Camera {
    pub position: Vec3,
    yaw:          f32,
    pitch:        f32,
    aspect:       f32,
    fov:          f32,
    near:         f32,
    far:          f32,
    pub speed:    f32,
    /// Radians per raw pixel (~0.003 gives ~0.17°/px).
    sensitivity:  f32,
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        debug_log!("Camera", "new", "Creating camera {}x{}", width, height);
        Self {
            position:    Vec3::new(8.0, 20.0, 30.0),
            yaw:         -std::f32::consts::FRAC_PI_2,
            pitch:       -0.30,
            aspect:      width as f32 / height.max(1) as f32,
            fov:         70.0_f32.to_radians(),
            near:        0.1,
            far:         500.0,
            speed:       10.0,
            sensitivity: 0.003,
        }
    }

    pub fn update_aspect(&mut self, width: u32, height: u32) {
        if height > 0 {
            self.aspect = width as f32 / height as f32;
        }
    }

    // ---- Direction vectors -------------------------------------------------

    /// Full look direction including pitch — used for W/S movement.
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    /// Horizontal right vector (always on the XZ plane) — used for A/D strafing.
    pub fn right(&self) -> Vec3 {
        // 90° to the right of yaw, Y = 0
        let angle = self.yaw + std::f32::consts::FRAC_PI_2;
        Vec3::new(angle.sin(), 0.0, angle.cos()).normalize()
    }

    // ---- Movement ----------------------------------------------------------

    /// Move along the full local forward vector (includes pitch).
    /// Pressing W when looking up → actually moves up+forward.
    pub fn move_forward(&mut self, amount: f32) {
        self.position += self.forward() * amount;
    }

    /// Strafe horizontally (A/D) — always perpendicular to forward on the XZ plane.
    pub fn move_right(&mut self, amount: f32) {
        self.position += self.right() * amount;
    }

    /// Move straight up/down along world Y (Space/Shift).
    pub fn move_up(&mut self, amount: f32) {
        self.position.y += amount;
    }

    /// Apply raw mouse delta (pixels).
    ///
    /// `dx > 0` → look right (yaw increases).
    /// `dy > 0` → cursor moved down → look down (pitch decreases).
    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw  += dx as f32 * self.sensitivity;
        self.pitch = (self.pitch - dy as f32 * self.sensitivity)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

    // ---- Matrices ----------------------------------------------------------

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward(), Vec3::Y)
    }
    pub fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
}

// ---------------------------------------------------------------------------
// Vertex3D — position + normal + colour  (36 bytes)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub color:    [f32; 3],
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------
const SHADER_SOURCE: &str = r"
struct Uniforms {
    view_proj:  mat4x4<f32>,   // 64 B  @ offset  0
    light_dir:  vec4<f32>,     // 16 B  @ offset 64  (xyz=dir, w=intensity)
    camera_pos: vec4<f32>,     // 16 B  @ offset 80  (xyz=pos, w unused)
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
};
struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       v_normal:    vec3<f32>,
    @location(1)       v_color:     vec3<f32>,
    @location(2)       v_world_pos: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos    = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal    = input.normal;
    out.v_color     = input.color;
    out.v_world_pos = input.position;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(u.light_dir.xyz);
    let ambient   = 0.35;
    let diffuse   = max(dot(normalize(input.v_normal), light_dir), 0.0);
    let light     = ambient + diffuse * 0.65 * u.light_dir.w;
    let lit_color = input.v_color * light;

    let dist       = length(input.v_world_pos - u.camera_pos.xyz);
    let fog_start  = 32.0;
    let fog_end    = 56.0;
    let fog_factor = clamp((dist - fog_start) / (fog_end - fog_start), 0.0, 1.0);
    let fog_color  = vec3<f32>(0.55, 0.65, 0.85);

    return vec4<f32>(mix(lit_color, fog_color, fog_factor), 1.0);
}
";

// ---------------------------------------------------------------------------
// ChunkMesh — GPU buffers for one chunk draw call
// ---------------------------------------------------------------------------
struct ChunkMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,
    index_count:   u32,
    vram_bytes:    u64,
}

// ---------------------------------------------------------------------------
// Game3DPipeline
// ---------------------------------------------------------------------------
pub struct Game3DPipeline {
    pipeline:          wgpu::RenderPipeline,
    uniform_buffer:    wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    depth_texture:     Option<wgpu::Texture>,
    depth_view:        Option<wgpu::TextureView>,
    depth_size:        (u32, u32),
    chunk_meshes:      Vec<ChunkMesh>,
    vram_usage:        u64,
}

const UNIFORM_SIZE: u64 = 96; // 24 × f32

impl Game3DPipeline {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
    ) -> Self {
        debug_log!("Game3DPipeline", "new", "Creating 3D pipeline, format={:?}", format);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Game3D Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Game3D BGL"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   std::num::NonZeroU64::new(UNIFORM_SIZE),
                    },
                    count: None,
                }],
            },
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:                Some("Game3D Pipeline Layout"),
                bind_group_layouts:   &[&bind_group_layout],
                push_constant_ranges: &[],
            },
        );

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
                2 => Float32x3,
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Game3D Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: "vs_main",
                buffers:     &[vertex_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: "fs_main",
                targets:     &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare:       wgpu::CompareFunction::Less,
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Game3D Uniform Buffer"),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        debug_log!("Game3DPipeline", "new", "Pipeline created");

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            depth_texture: None,
            depth_view:    None,
            depth_size:    (0, 0),
            chunk_meshes:  Vec::new(),
            vram_usage:    UNIFORM_SIZE,
        }
    }

    // -----------------------------------------------------------------------
    // World mesh upload
    // -----------------------------------------------------------------------

    /// Replace the GPU chunk mesh set with new CPU-side geometry.
    pub fn set_world_meshes(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        meshes: Vec<(Vec<Vertex3D>, Vec<u32>)>,
    ) {
        let old_vram: u64 = self.chunk_meshes.iter().map(|m| m.vram_bytes).sum();
        self.vram_usage = self.vram_usage.saturating_sub(old_vram);
        self.chunk_meshes.clear();

        let mut new_vram: u64 = 0;

        for (vertices, indices) in meshes {
            if vertices.is_empty() || indices.is_empty() { continue; }

            let vert_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8,
                    vertices.len() * std::mem::size_of::<Vertex3D>(),
                )
            };
            let idx_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    indices.as_ptr() as *const u8,
                    indices.len() * std::mem::size_of::<u32>(),
                )
            };

            let vb = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Chunk VB"),
                size:               vert_bytes.len() as u64,
                usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&vb, 0, vert_bytes);

            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some("Chunk IB"),
                size:               idx_bytes.len() as u64,
                usage:              wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ib, 0, idx_bytes);

            let bytes = vert_bytes.len() as u64 + idx_bytes.len() as u64;
            new_vram += bytes;

            self.chunk_meshes.push(ChunkMesh {
                vertex_buffer: vb,
                index_buffer:  ib,
                index_count:   indices.len() as u32,
                vram_bytes:    bytes,
            });
        }

        self.vram_usage += new_vram;

        debug_log!(
            "Game3DPipeline", "set_world_meshes",
            "Uploaded {} chunk meshes, VRAM={:.2} MB",
            self.chunk_meshes.len(),
            self.vram_usage as f64 / (1024.0 * 1024.0)
        );
    }

    // -----------------------------------------------------------------------
    // Depth texture management
    // -----------------------------------------------------------------------

    fn ensure_depth(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.depth_size == (width, height) { return; }

        debug_log!(
            "Game3DPipeline", "ensure_depth",
            "Recreating depth texture {}x{}", width, height
        );

        if self.depth_texture.is_some() {
            let old = (self.depth_size.0 as u64) * (self.depth_size.1 as u64) * 4;
            self.vram_usage = self.vram_usage.saturating_sub(old);
        }

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Game3D Depth"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Depth32Float,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.vram_usage += (width as u64) * (height as u64) * 4;
        self.depth_texture = Some(tex);
        self.depth_view    = Some(view);
        self.depth_size    = (width, height);
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
        camera:     &Camera,
        width:      u32,
        height:     u32,
    ) {
        self.ensure_depth(device, width, height);

        // Uniforms
        let vp = camera.view_projection_matrix();
        let mut data = [0.0f32; 24];
        data[0..16].copy_from_slice(&vp.to_cols_array());
        data[16..20].copy_from_slice(&[0.45, 0.80, 0.55, 1.0]);
        data[20..24].copy_from_slice(&[
            camera.position.x, camera.position.y, camera.position.z, 0.0,
        ]);
        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 24]>(),
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Game3D BG"),
            layout:  &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });

        // Depth clear
        {
            let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:                    Some("Depth Clear"),
                color_attachments:        &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:        self.depth_view.as_ref().unwrap(),
                    depth_ops:   Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }

        // Main 3D pass
        if !self.chunk_meshes.is_empty() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Game3D Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:        self.depth_view.as_ref().unwrap(),
                    depth_ops:   Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            for mesh in &self.chunk_meshes {
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
    }

    pub fn vram_usage(&self) -> u64 {
        self.vram_usage
    }
}