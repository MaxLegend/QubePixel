// =============================================================================
// QubePixel — EntityRenderer
// =============================================================================
// Generic GPU renderer for skeletal entities (players, NPCs, etc.).
//
// Design:
//   - Supports multiple named model types (registered once via `register_model`).
//   - Supports multiple live instances of each model (spawned via `spawn_instance`).
//   - Uses entity_pbr.wgsl: full Cook-Torrance PBR + VCT GI + DDA voxel shadows.
//   - Group 0: per-bone uniform (view_proj + bone world matrix + lighting) + skin texture.
//   - Group 1: VCT bind group shared from VCTSystem::prepare_fragment each frame.
//
// Usage per frame:
//   1. Call `render(encoder, color_view, depth_view, queue, vct_bg, &calls)`.
//      `calls` is a slice of EntityRenderCall — one per visible entity.
//   2. The renderer writes bone uniforms before the render pass, then issues
//      one draw call per visible bone in each entity.

use std::collections::HashMap;
use std::mem;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::core::entity::EntityId;
use crate::core::player_model::{PlayerModel, PlayerVertex};

const SHADER_SRC: &str = include_str!("../shaders/entity_pbr.wgsl");

// ---------------------------------------------------------------------------
// Compile-time size check
// ---------------------------------------------------------------------------

/// Per-bone uniform buffer layout: must match EntityUniforms in entity_pbr.wgsl.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct EntityUniforms {
    view_proj:      [f32; 16],  // 64 bytes
    model:          [f32; 16],  // 64 bytes
    sun_direction:  [f32;  4],  // 16 bytes  (xyz = dir, w = intensity)
    sun_color:      [f32;  4],  // 16 bytes
    moon_direction: [f32;  4],  // 16 bytes  (xyz = dir, w = intensity)
    moon_color:     [f32;  4],  // 16 bytes
    ambient_color:  [f32;  4],  // 16 bytes  (xyz = colour, w = min_level)
    camera_pos:     [f32;  4],  // 16 bytes
    shadow_params:  [f32;  4],  // 16 bytes  x=shadow_on, y=normal_offset
}
const _: () = assert!(mem::size_of::<EntityUniforms>() == 240);

const UNIFORM_SIZE: u64 = mem::size_of::<EntityUniforms>() as u64;

// ---------------------------------------------------------------------------
// Per-model GPU geometry (shared across all instances of the same model type)
// ---------------------------------------------------------------------------

struct BoneBuffers {
    vertex_buf:  wgpu::Buffer,
    index_buf:   wgpu::Buffer,
    index_count: u32,
}

struct EntityModelGpu {
    bones:      HashMap<String, BoneBuffers>,
    bone_order: Vec<String>,  // draw order (back-to-front for correct alpha)
    skin_tex:   wgpu::Texture,
    skin_view:  wgpu::TextureView,
    skin_samp:  wgpu::Sampler,
}

// ---------------------------------------------------------------------------
// Per-instance GPU resources (one set per spawned entity)
// ---------------------------------------------------------------------------

struct EntityInstanceGpu {
    model_id: String,
    /// Per-bone: (uniform_buf, bind_group_0).
    /// bind_group_0 references the model's skin_view/skin_samp.
    bone_gpu: HashMap<String, (wgpu::Buffer, wgpu::BindGroup)>,
}

// ---------------------------------------------------------------------------
// Public render-call types
// ---------------------------------------------------------------------------

/// World-space transform for one bone in a single render call.
pub struct BoneDrawData {
    pub name:            String,
    pub world_transform: Mat4,
}

/// Everything needed to render one entity for one frame.
pub struct EntityRenderCall {
    pub entity_id:        EntityId,
    pub model_id:         String,
    pub bones:            Vec<BoneDrawData>,
    pub view_proj:        Mat4,
    pub sun_dir:          Vec3,
    pub sun_color:        Vec3,
    pub sun_intensity:    f32,
    pub moon_dir:         Vec3,
    pub moon_color:       Vec3,
    pub moon_intensity:   f32,
    pub ambient:          Vec3,
    pub ambient_min:      f32,
    pub camera_pos:       Vec3,
    pub shadow_sun_enabled: bool,
    pub shadow_offset:      f32,
    pub skip_bones:       Vec<String>,
}

// ---------------------------------------------------------------------------
// EntityRenderer
// ---------------------------------------------------------------------------

pub struct EntityRenderer {
    pipeline:  wgpu::RenderPipeline,
    bgl_0:     wgpu::BindGroupLayout,
    models:    HashMap<String, EntityModelGpu>,
    instances: HashMap<EntityId, EntityInstanceGpu>,
}

impl EntityRenderer {
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    /// `vct_frag_bgl` is `VCTSystem::frag_bgl` — the Group 1 layout shared
    /// with the block renderer.
    pub fn new(
        device:       &wgpu::Device,
        format:       wgpu::TextureFormat,
        vct_frag_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Entity PBR Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        // Group 0: per-bone uniform + skin texture + sampler
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Entity BGL 0"),
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
            label:              Some("Entity Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl_0), Some(vct_frag_bgl)],
            immediate_size:     0,
        });

        // Vertex layout matches PlayerVertex (position, uv, normal)
        let vb_layout = wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<PlayerVertex>() as u64,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format:           wgpu::VertexFormat::Float32x3,
                    offset:           0,
                    shader_location:  0,
                },
                wgpu::VertexAttribute {
                    format:           wgpu::VertexFormat::Float32x2,
                    offset:           12,
                    shader_location:  1,
                },
                wgpu::VertexAttribute {
                    format:           wgpu::VertexFormat::Float32x3,
                    offset:           20,
                    shader_location:  2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Entity Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers:             &[vb_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend:      Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:   wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Cw,
                cull_mode:  Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Option::from(true),
                depth_compare:       Some(wgpu::CompareFunction::Less),
                stencil:             Default::default(),
                bias:                Default::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:          None,
        });

        Self {
            pipeline,
            bgl_0,
            models:    HashMap::new(),
            instances: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Model registration
    // -----------------------------------------------------------------------

    /// Register a model type.  Call once per model type; geometry is shared
    /// across all instances.  `bone_order` controls back-to-front draw order.
    /// `skin_path` is an optional PNG path; falls back to a checkerboard.
    pub fn register_model(
        &mut self,
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        model_id:   &str,
        model:      &PlayerModel,
        bone_order: &[&str],
        skin_path:  Option<&str>,
    ) {
        // --- Upload geometry ---
        let mut bones = HashMap::new();
        for (name, mesh) in &model.bone_meshes {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("Entity VB: {}/{}", model_id, name)),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("Entity IB: {}/{}", model_id, name)),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage:    wgpu::BufferUsages::INDEX,
            });
            bones.insert(name.clone(), BoneBuffers {
                vertex_buf:  vb,
                index_buf:   ib,
                index_count: mesh.indices.len() as u32,
            });
        }

        let order: Vec<String> = bone_order.iter()
            .map(|s| s.to_string())
            .filter(|s| bones.contains_key(s))
            .collect();

        // --- Upload skin texture ---
        let (skin_tex, skin_view) = Self::load_skin(device, queue, skin_path);
        let skin_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some(&format!("Entity Skin Sampler: {}", model_id)),
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        self.models.insert(model_id.to_string(), EntityModelGpu {
            bones,
            bone_order: order,
            skin_tex,
            skin_view,
            skin_samp,
        });
    }

    /// Register a model type with a pre-built GPU texture (e.g. per-model atlas).
    /// Use this for block models that have their own packed texture atlas.
    pub fn register_model_with_texture(
        &mut self,
        device:    &wgpu::Device,
        model_id:  &str,
        model:     &PlayerModel,
        bone_order: &[&str],
        tex:       wgpu::Texture,
        view:      wgpu::TextureView,
    ) {
        // --- Upload geometry ---
        let mut bones = HashMap::new();
        for (name, mesh) in &model.bone_meshes {
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("Entity VB: {}/{}", model_id, name)),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage:    wgpu::BufferUsages::VERTEX,
            });
            let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:    Some(&format!("Entity IB: {}/{}", model_id, name)),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage:    wgpu::BufferUsages::INDEX,
            });
            bones.insert(name.clone(), BoneBuffers {
                vertex_buf:  vb,
                index_buf:   ib,
                index_count: mesh.indices.len() as u32,
            });
        }

        let order: Vec<String> = bone_order.iter()
            .map(|s| s.to_string())
            .filter(|s| bones.contains_key(s))
            .collect();

        let skin_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some(&format!("Entity Skin Sampler: {}", model_id)),
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        self.models.insert(model_id.to_string(), EntityModelGpu {
            bones,
            bone_order: order,
            skin_tex:  tex,
            skin_view: view,
            skin_samp,
        });
    }

    /// Spawn a GPU instance for one entity.  The skin texture is taken from
    /// the model's registered skin (model-level skin shared across instances).
    /// Call `despawn_instance` when the entity is removed from the world.
    pub fn spawn_instance(
        &mut self,
        device:   &wgpu::Device,
        id:       EntityId,
        model_id: &str,
    ) {
        let Some(model) = self.models.get(model_id) else {
            eprintln!("[EntityRenderer] spawn_instance: model '{}' not registered", model_id);
            return;
        };

        let mut bone_gpu = HashMap::new();
        for bone_name in model.bones.keys() {
            let ub = device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&format!("Entity UB {:?}/{}", id, bone_name)),
                size:               UNIFORM_SIZE,
                usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some(&format!("Entity BG0 {:?}/{}", id, bone_name)),
                layout: &self.bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: wgpu::BindingResource::TextureView(&model.skin_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: wgpu::BindingResource::Sampler(&model.skin_samp),
                    },
                ],
            });
            bone_gpu.insert(bone_name.clone(), (ub, bg));
        }

        self.instances.insert(id, EntityInstanceGpu {
            model_id: model_id.to_string(),
            bone_gpu,
        });
    }

    /// Free GPU resources for a despawned entity.
    pub fn despawn_instance(&mut self, id: EntityId) {
        self.instances.remove(&id);
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    /// Render all entity draw calls for this frame.
    /// Must be called after VCT dispatch; `vct_bg` must be ready.
    pub fn render(
        &self,
        encoder:    &mut wgpu::CommandEncoder,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        queue:      &wgpu::Queue,
        vct_bg:     &wgpu::BindGroup,
        calls:      &[EntityRenderCall],
    ) {
        if calls.is_empty() { return; }

        // --- Pre-write all bone uniform buffers (before render pass opens) ---
        for call in calls {
            let Some(instance) = self.instances.get(&call.entity_id) else { continue };
            let vp = call.view_proj.to_cols_array();

            for bd in &call.bones {
                let Some((ub, _)) = instance.bone_gpu.get(&bd.name) else { continue };
     let uniforms = EntityUniforms {
                    view_proj:      vp,
                    model:          bd.world_transform.to_cols_array(),
                    sun_direction:  [call.sun_dir.x,   call.sun_dir.y,   call.sun_dir.z,   call.sun_intensity],
                    sun_color:      [call.sun_color.x,  call.sun_color.y,  call.sun_color.z,  0.0],
                    moon_direction: [call.moon_dir.x,  call.moon_dir.y,  call.moon_dir.z,  call.moon_intensity],
                    moon_color:     [call.moon_color.x, call.moon_color.y, call.moon_color.z, 0.0],
                    ambient_color:  [call.ambient.x,   call.ambient.y,   call.ambient.z,   call.ambient_min],
                    camera_pos:     [call.camera_pos.x, call.camera_pos.y, call.camera_pos.z, 0.0],
                    shadow_params:  [
                        if call.shadow_sun_enabled { 1.0 } else { 0.0 },
                        call.shadow_offset,
                        0.0,
                        0.0,
                    ],
                };
                queue.write_buffer(ub, 0, bytemuck::bytes_of(&uniforms));
            }
        }

        // --- Single render pass for all entities ---
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Entity Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           color_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            multiview_mask: None,
            ..Default::default()
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(1, vct_bg, &[]);

        for call in calls {
            let Some(instance) = self.instances.get(&call.entity_id) else { continue };
            let Some(model)    = self.models.get(&call.model_id)     else { continue };

            for bone_name in &model.bone_order {
                if call.skip_bones.iter().any(|s| s == bone_name) { continue }
                let Some((_, bg0))   = instance.bone_gpu.get(bone_name) else { continue };
                let Some(bone_bufs) = model.bones.get(bone_name)         else { continue };

                rpass.set_bind_group(0, bg0, &[]);
                rpass.set_vertex_buffer(0, bone_bufs.vertex_buf.slice(..));
                rpass.set_index_buffer(bone_bufs.index_buf.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..bone_bufs.index_count, 0, 0..1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal: skin texture loading
    // -----------------------------------------------------------------------

    fn load_skin(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        path:   Option<&str>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        if let Some(p) = path {
            if let Ok(bytes) = std::fs::read(p) {
                if let Ok(img) = image::load_from_memory(&bytes) {
                    let rgba     = img.to_rgba8();
                    let (w, h)   = rgba.dimensions();
                    return Self::upload_rgba(device, queue, rgba.as_raw(), w, h);
                }
            }
        }
        // Fallback: 64×64 magenta/black checkerboard
        let mut data = vec![0u8; 64 * 64 * 4];
        for y in 0..64u32 {
            for x in 0..64u32 {
                let i   = ((y * 64 + x) * 4) as usize;
                let odd = (x / 8 + y / 8) % 2 == 0;
                data[i]   = if odd { 255 } else { 80 };
                data[i+1] = 0;
                data[i+2] = if odd { 255 } else { 80 };
                data[i+3] = 255;
            }
        }
        Self::upload_rgba(device, queue, &data, 64, 64)
    }

    fn upload_rgba(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        data:   &[u8],
        w: u32, h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Entity Skin"),
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
                texture:   &tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        (tex, view)
    }
}
