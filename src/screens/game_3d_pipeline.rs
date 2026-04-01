// =============================================================================
// QubePixel — Game3DPipeline  (camera, depth buffer, world chunk meshes)
// =============================================================================

use std::collections::HashMap;
use std::time::Instant;
use crate::{debug_log, ext_debug_log, flow_debug_log};
use glam::{Mat4, Vec3};

// ---------------------------------------------------------------------------
// FrustumPlanes
// ---------------------------------------------------------------------------
pub struct FrustumPlanes {
    planes: [[f32; 4]; 6],
}

impl FrustumPlanes {
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let c   = vp.to_cols_array_2d();
        let row = |i: usize| -> [f32; 4] { [c[0][i], c[1][i], c[2][i], c[3][i]] };
        let r0  = row(0);
        let r1  = row(1);
        let r2  = row(2);
        let r3  = row(3);

        let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
        };
        let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
            [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]
        };

        Self {
            planes: [
                add(r3, r0), sub(r3, r0),
                add(r3, r1), sub(r3, r1),
                add(r3, r2), sub(r3, r2),
            ],
        }
    }

    pub fn intersects_aabb(&self, min: [f32; 3], max: [f32; 3]) -> bool {
        for plane in &self.planes {
            let [a, b, c, d] = *plane;
            let px = if a >= 0.0 { max[0] } else { min[0] };
            let py = if b >= 0.0 { max[1] } else { min[1] };
            let pz = if c >= 0.0 { max[2] } else { min[2] };
            if a * px + b * py + c * pz + d < 0.0 {
                return false;
            }
        }
        true
    }
}

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
        if height > 0 { self.aspect = width as f32 / height as f32; }
    }
    pub fn fov(&self) -> f32 { self.fov }
    pub fn aspect(&self) -> f32 { self.aspect }
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    }

    pub fn right(&self) -> Vec3 { self.forward().cross(Vec3::Y).normalize() }

    pub fn move_forward(&mut self, amount: f32) { self.position += self.forward() * amount; }
    pub fn move_right(&mut self, amount: f32)   { self.position += self.right()   * amount; }
    pub fn move_up(&mut self, amount: f32)       { self.position.y += amount; }

    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.yaw  += dx as f32 * self.sensitivity;
        self.pitch = (self.pitch - dy as f32 * self.sensitivity)
            .clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }

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
// Vertex3D — position + normal + colour + texcoord (44 bytes)
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub color:    [f32; 3],
    pub texcoord: [f32; 2],
}

// ---------------------------------------------------------------------------
// WGSL shaders
// ---------------------------------------------------------------------------
const SHADER_SOURCE: &str = r"
struct Uniforms {
    view_proj:  mat4x4<f32>,
    light_dir:  vec4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex:     texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
    @location(3) texcoord: vec2<f32>,
};
struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       v_normal:    vec3<f32>,
    @location(1)       v_color:     vec3<f32>,
    @location(2)       v_world_pos: vec3<f32>,
    @location(3)       v_texcoord:  vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos    = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal    = input.normal;
    out.v_color     = input.color;
    out.v_world_pos = input.position;
    out.v_texcoord  = input.texcoord;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color  = textureSample(atlas_tex, atlas_sampler, input.v_texcoord).rgb;
    let base_color = input.v_color * tex_color;
    let light_dir  = normalize(u.light_dir.xyz);
    let ambient    = 0.35;
    let diffuse    = max(dot(normalize(input.v_normal), light_dir), 0.0);
    let light      = ambient + diffuse * 0.65 * u.light_dir.w;
    return vec4<f32>(base_color * light, 1.0);
}
";

const OUTLINE_SHADER_SOURCE: &str = r"
struct OutlineUniforms {
    view_proj: mat4x4<f32>,
    block_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: OutlineUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    let world_pos = position + u.block_pos.xyz;
    return u.view_proj * vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
";

const OUTLINE_UNIFORM_SIZE: u64 = 80;

// ---------------------------------------------------------------------------
// ChunkMesh
// ---------------------------------------------------------------------------
struct ChunkMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer:  wgpu::Buffer,
    index_count:   u32,
    vram_bytes:    u64,
    aabb_min:      [f32; 3],
    aabb_max:      [f32; 3],
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
    chunk_meshes:      HashMap<(i32, i32, i32), ChunkMesh>,
    vram_usage:        u64,
    pub culled_last_frame: u32,
    atlas_sampler:     Option<wgpu::Sampler>,
    outline_pipeline:           Option<wgpu::RenderPipeline>,
    outline_bind_group_layout:  Option<wgpu::BindGroupLayout>,
    outline_bind_group:         Option<wgpu::BindGroup>,
    outline_uniform_buffer:     Option<wgpu::Buffer>,
    outline_vb:                 Option<wgpu::Buffer>,
    surface_format:             wgpu::TextureFormat,
    cached_bind_group:          Option<wgpu::BindGroup>,
}

const UNIFORM_SIZE: u64 = 96;

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
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   std::num::NonZeroU64::new(UNIFORM_SIZE),
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
            },
        );

        // wgpu 22: PipelineLayoutDescriptor has no push_constant_ranges field
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Game3D Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            },
        );

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &wgpu::vertex_attr_array![
                0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2,
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Game3D Pipeline"),
            layout: Some(&pipeline_layout),
            // wgpu 22: entry_point is Option<&str>, needs compilation_options
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                buffers:             &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                cull_mode:  Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare: Option::from(wgpu::CompareFunction::Less),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            // wgpu 22: no multiview field; cache field added
            cache: None,
            multiview_mask: None,
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
            chunk_meshes:  HashMap::new(),
            vram_usage:    UNIFORM_SIZE,
            culled_last_frame: 0,
            atlas_sampler: None,
            outline_pipeline:          None,
            outline_bind_group_layout: None,
            outline_bind_group:        None,
            outline_uniform_buffer:    None,
            outline_vb:                None,
            surface_format:            format,
            cached_bind_group:         None,
        }
    }

    // -----------------------------------------------------------------------
    // Incremental chunk mesh management
    // -----------------------------------------------------------------------

    pub fn update_chunk_meshes(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    ) -> u128 {
        let t0    = Instant::now();
        let count = meshes.len();
        let mut new_vram: u64 = 0;

        for (key, vertices, indices, aabb_min, aabb_max) in meshes {
            if let Some(old) = self.chunk_meshes.remove(&key) {
                self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
            }
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
                label: Some("Chunk VB"), size: vert_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&vb, 0, vert_bytes);

            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Chunk IB"), size: idx_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ib, 0, idx_bytes);

            let bytes = vert_bytes.len() as u64 + idx_bytes.len() as u64;
            new_vram += bytes;

            self.chunk_meshes.insert(key, ChunkMesh {
                vertex_buffer: vb, index_buffer: ib,
                index_count: indices.len() as u32,
                vram_bytes: bytes, aabb_min, aabb_max,
            });
        }

        self.vram_usage += new_vram;

        let upload_us = t0.elapsed().as_micros();
        ext_debug_log!(
            "Game3DPipeline", "update_chunk_meshes",
            "[PERF] upload={:.2}ms chunks={} total={} VRAM={:.2}MB",
            upload_us as f64 / 1000.0, count,
            self.chunk_meshes.len(),
            self.vram_usage as f64 / (1024.0 * 1024.0)
        );
        upload_us
    }
    /// Insert a pre-created chunk mesh into the render map.
    /// Used by UploadWorker to deliver asynchronously created GPU buffers.
    /// Replaces any existing mesh for the same key and returns the old
    /// mesh's VRAM size (already subtracted from self.vram_usage).
    /// Insert a pre-created chunk mesh into the render map.
    /// Used by the async upload pipeline to deliver GPU buffers
    /// that were created on the main thread from packed staging data.
    pub fn insert_chunk_mesh(
        &mut self,
        key:           (i32, i32, i32),
        vertex_buffer: wgpu::Buffer,
        index_buffer:  wgpu::Buffer,
        index_count:   u32,
        vram_bytes:    u64,
        aabb_min:      [f32; 3],
        aabb_max:      [f32; 3],
    ) {
        // Remove old mesh if present → track freed VRAM
        if let Some(old) = self.chunk_meshes.remove(&key) {
            self.vram_usage = self.vram_usage.saturating_sub(old.vram_bytes);
        }

        self.vram_usage += vram_bytes;
        self.chunk_meshes.insert(key, ChunkMesh {
            vertex_buffer,
            index_buffer,
            index_count,
            vram_bytes,
            aabb_min,
            aabb_max,
        });
    }
    pub fn remove_chunk_meshes(&mut self, keys: &[(i32, i32, i32)]) {
        let mut freed = 0u64;
        let mut removed = 0u32;
        for key in keys {
            if let Some(mesh) = self.chunk_meshes.remove(key) {
                freed   += mesh.vram_bytes;
                removed += 1;
            }
        }
        if removed > 0 {
            self.vram_usage = self.vram_usage.saturating_sub(freed);
            debug_log!(
                "Game3DPipeline", "remove_chunk_meshes",
                "Freed {} chunks, recovered {:.2}MB",
                removed, freed as f64 / (1024.0 * 1024.0)
            );
        }
    }

    #[allow(dead_code)]
    pub fn set_world_meshes(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        meshes: Vec<(Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    ) {
        let old: u64 = self.chunk_meshes.values().map(|m| m.vram_bytes).sum();
        self.vram_usage = self.vram_usage.saturating_sub(old);
        self.chunk_meshes.clear();
        let mut new_vram: u64 = 0;

        for (vertices, indices, aabb_min, aabb_max) in meshes {
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
                label: Some("Chunk VB"), size: vert_bytes.len() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&vb, 0, vert_bytes);
            let ib = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Chunk IB"), size: idx_bytes.len() as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&ib, 0, idx_bytes);
            let bytes = vert_bytes.len() as u64 + idx_bytes.len() as u64;
            new_vram += bytes;
            self.chunk_meshes.insert((0, 0, 0), ChunkMesh {
                vertex_buffer: vb, index_buffer: ib,
                index_count: indices.len() as u32,
                vram_bytes: bytes, aabb_min, aabb_max,
            });
        }

        self.vram_usage += new_vram;
        debug_log!(
            "Game3DPipeline", "set_world_meshes",
            "Full rebuild: {} chunks, VRAM={:.2}MB",
            self.chunk_meshes.len(),
            self.vram_usage as f64 / (1024.0 * 1024.0)
        );
    }

    // -----------------------------------------------------------------------
    // Depth texture
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
            mip_level_count: 1, sample_count: 1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Depth32Float,
            usage:           wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats:    &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.vram_usage   += (width as u64) * (height as u64) * 4;
        self.depth_texture = Some(tex);
        self.depth_view    = Some(view);
        self.depth_size    = (width, height);
    }

    // -----------------------------------------------------------------------
    // Render — frustum culling
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
        atlas_view: &wgpu::TextureView,
    ) -> u128 {
        let t0 = Instant::now();
        self.ensure_depth(device, width, height);

        if self.atlas_sampler.is_none() {
            self.atlas_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
                label:         Some("Atlas Mipmap Sampler"),
                mag_filter:    wgpu::FilterMode::Nearest,
                min_filter:    wgpu::FilterMode::Nearest,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }));
            debug_log!("Game3DPipeline", "render", "Atlas sampler created");
        }

        let vp      = camera.view_projection_matrix();
        let frustum = FrustumPlanes::from_view_projection(&vp);

        // Upload uniforms
        let mut data = [0.0f32; 24];
        data[0..16].copy_from_slice(&vp.to_cols_array());
        data[16..20].copy_from_slice(&[0.45, 0.80, 0.55, 1.0]);
        data[20..24].copy_from_slice(&[
            camera.position.x, camera.position.y, camera.position.z, 0.0,
        ]);
        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of::<[f32; 24]>())
        };
        queue.write_buffer(&self.uniform_buffer, 0, uniform_bytes);

        // Cache bind group — resources (uniform buffer, atlas view, sampler)
        // never change after first frame, so we create it once and reuse.
        if self.cached_bind_group.is_none() {
            self.cached_bind_group = Some(device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label:   Some("Game3D BG"),
                    layout:  &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding:  0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding:  1,
                            resource: wgpu::BindingResource::TextureView(atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding:  2,
                            resource: wgpu::BindingResource::Sampler(
                                self.atlas_sampler.as_ref().unwrap()
                            ),
                        },
                    ],
                },
            ));
            debug_log!("Game3DPipeline", "render", "Bind group cached");
        }
        let bind_group = self.cached_bind_group.as_ref().unwrap();

        // Depth clear pass
        {
            // wgpu 22: RenderPassDescriptor needs multiview_mask
            let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:             Some("Depth Clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view:      self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes:    None,
                occlusion_query_set: None,
                multiview_mask:      None,  // wgpu 22
            });
        }

        // Frustum culling counts
        let mut visible = 0u32;
        let mut culled  = 0u32;
        for mesh in self.chunk_meshes.values() {
            if frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { visible += 1; }
            else { culled += 1; }
        }
        self.culled_last_frame = culled;

        if visible == 0 { return 0; }

        // Main 3D pass

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Game3D Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,  // wgpu 22
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:     None,  // wgpu 22
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);

        for mesh in self.chunk_meshes.values() {
            if !frustum.intersects_aabb(mesh.aabb_min, mesh.aabb_max) { continue; }
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }

        let render_us = t0.elapsed().as_micros();
        flow_debug_log!(
            "Game3DPipeline", "render",
            "[PERF] render={:.2}ms visible={} culled={} gpu_chunks={}",
            render_us as f64 / 1000.0, visible, culled, self.chunk_meshes.len()
        );
        render_us
    }

    pub fn vram_usage(&self) -> u64 { self.vram_usage }
    pub fn gpu_chunk_count(&self) -> usize { self.chunk_meshes.len() }
    // -----------------------------------------------------------------------
    // Outline pipeline
    // -----------------------------------------------------------------------

    fn init_outline(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        debug_log!("Game3DPipeline", "init_outline", "Creating outline pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Outline Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(OUTLINE_SHADER_SOURCE)),
        });

        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label:   Some("Outline BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty:                 wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size:   std::num::NonZeroU64::new(OUTLINE_UNIFORM_SIZE),
                        },
                        count: None,
                    },
                ],
            },
        );

        // wgpu 22: no push_constant_ranges
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label:              Some("Outline Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size:     0,
            },
        );

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Outline Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes:   &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:      &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:      self.surface_format,
                    blend:       None,
                    write_mask:  wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(false),
                depth_compare: Option::from(wgpu::CompareFunction::LessEqual),
                stencil:             wgpu::StencilState::default(),
                bias:                wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            cache: None,  // wgpu 22
            multiview_mask: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Outline Uniform Buffer"),
            size:               OUTLINE_UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        const E: f32 = 0.005;
        let vertices: [[f32; 3]; 24] = [
            [-E, -E, -E], [ 1.0+E, -E, -E],
            [ 1.0+E, -E, -E], [ 1.0+E, -E,  1.0+E],
            [ 1.0+E, -E,  1.0+E], [-E, -E,  1.0+E],
            [-E, -E,  1.0+E], [-E, -E, -E],
            [-E,  1.0+E, -E], [ 1.0+E,  1.0+E, -E],
            [ 1.0+E,  1.0+E, -E], [ 1.0+E,  1.0+E,  1.0+E],
            [ 1.0+E,  1.0+E,  1.0+E], [-E,  1.0+E,  1.0+E],
            [-E,  1.0+E,  1.0+E], [-E,  1.0+E, -E],
            [-E, -E, -E], [-E,  1.0+E, -E],
            [ 1.0+E, -E, -E], [ 1.0+E,  1.0+E, -E],
            [ 1.0+E, -E,  1.0+E], [ 1.0+E,  1.0+E,  1.0+E],
            [-E, -E,  1.0+E], [-E,  1.0+E,  1.0+E],
        ];
        let vb_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<[f32; 3]>(),
            )
        };
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline VB"), size: vb_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vb, 0, vb_bytes);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Outline BG"),
            layout:  &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        self.vram_usage += vb_bytes.len() as u64 + OUTLINE_UNIFORM_SIZE;

        self.outline_pipeline          = Some(pipeline);
        self.outline_bind_group_layout = Some(bind_group_layout);
        self.outline_bind_group        = Some(bind_group);
        self.outline_uniform_buffer    = Some(uniform_buffer);
        self.outline_vb                = Some(vb);

        debug_log!("Game3DPipeline", "init_outline", "Outline pipeline ready");
    }

    pub fn render_outline(
        &mut self,
        encoder:    &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        camera:     &Camera,
        width:      u32,
        height:     u32,
        block_pos:  Option<glam::IVec3>,
    ) {
        let block_pos = match block_pos { Some(p) => p, None => return };
        if self.depth_view.is_none() { return; }
        if self.outline_pipeline.is_none() { self.init_outline(device, queue); }

        self.ensure_depth(device, width, height);

        let vp = camera.view_projection_matrix();
        let mut data = [0.0f32; 20];
        data[0..16].copy_from_slice(&vp.to_cols_array());
        data[16] = block_pos.x as f32;
        data[17] = block_pos.y as f32;
        data[18] = block_pos.z as f32;

        let uniform_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, OUTLINE_UNIFORM_SIZE as usize)
        };
        queue.write_buffer(self.outline_uniform_buffer.as_ref().unwrap(), 0, uniform_bytes);

        // wgpu 22: depth_slice + multiview_mask
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Outline Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,  // wgpu 22
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view:      self.depth_view.as_ref().unwrap(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes:    None,
            occlusion_query_set: None,
            multiview_mask:      None,  // wgpu 22
        });

        pass.set_pipeline(self.outline_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, self.outline_bind_group.as_ref().unwrap(), &[]);
        pass.set_vertex_buffer(0, self.outline_vb.as_ref().unwrap().slice(..));
        pass.draw(0..24, 0..1);

        flow_debug_log!(
            "Game3DPipeline", "render_outline",
            "Drawing outline at ({}, {}, {})", block_pos.x, block_pos.y, block_pos.z
        );
    }
}