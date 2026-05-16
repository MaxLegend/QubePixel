// =============================================================================
// QubePixel — Radiance Cascades: GPU System (Step 4)
// =============================================================================
//
// Orchestrator that ties together all GPU compute pipelines:
//   1. Ray March pipeline (per cascade) — DDA voxel tracing
//   2. Merge pipeline (cascade chain)   — Trilinear interpolation
//   3. GI Sampling (fragment shader)    — C0 probe lookup
//
// Per-frame workflow (called from GameScreen::render):
//   rc.recenter(camera_pos);
//   rc.update_voxel_texture(device, queue);  // filled from World chunks
//   rc.update_block_lut(device, queue);      // filled from BlockRegistry
//   rc.dispatch(&mut encoder, queue, sky_brightness);
//   // rc.gi_bg passed into render pass for fragment shader sampling

use glam::Vec3;

use wgpu::util::DeviceExt;

use crate::{debug_log, flow_debug_log};

use super::types::*;
use super::sampling::*;
use super::voxel_tex::*;

// ---------------------------------------------------------------------------
// Compile-time size assertions (must match WGSL structs exactly)
// ---------------------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<CascadeDispatchParams>() == 64);
const _: () = assert!(std::mem::size_of::<MergeParams>() == 96);
const _: () = assert!(std::mem::size_of::<GiCascadeEntry>() == 48);
const _: () = assert!(std::mem::size_of::<GiSampleParams>() == 256);
const _: () = assert!(std::mem::size_of::<RadianceInterval>() == 48);

// ---------------------------------------------------------------------------
// CascadeDispatchParams — 64 bytes
// ---------------------------------------------------------------------------
// Matches CascadeParams uniform in ray_march.wgsl.
// All f32 to avoid WGSL vec3 alignment headaches.

// #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// pub struct CascadeDispatchParams {
//     pub grid_origin_x: f32,    // [  0] world origin of probe grid
//     pub grid_origin_y: f32,    // [  4]
//     pub grid_origin_z: f32,    // [  8]
//     pub grid_spacing: f32,     // [ 12] distance between adjacent probes
//     pub ray_count: f32,        // [ 16] number of rays per probe
//     pub grid_size: f32,        // [ 20] grid resolution per axis
//     pub max_steps: f32,        // [ 24] max DDA steps per ray
//     pub max_dist: f32,         // [ 28] max ray travel distance
//     pub ray_start: f32,        // [ 32] min distance from probe center
//     pub opaque_threshold: f32, // [ 36] tau above which medium is opaque
//     pub sky_brightness: f32,   // [ 40] current sky brightness factor
//     pub _pad0: f32,            // [ 44]
//     pub _pad1: f32,            // [ 48]
//     pub _pad2: f32,            // [ 52]
//     pub _pad3: f32,            // [ 56]
//     pub _pad4: f32,            // [ 60]
// }
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CascadeDispatchParams {
    // Порядок полей строго совпадает с CascadeParams в ray_march.wgsl
    pub grid_spacing:      f32,   // [  0]
    pub ray_length:        f32,   // [  4]  макс. дистанция луча
    pub ray_start:         f32,   // [  8]
    pub grid_half_extent:  f32,   // [ 12]
    pub grid_size:         u32,   // [ 16]  WAS f32 — WGSL использует u32
    pub ray_count:         u32,   // [ 20]  WAS f32 — WGSL использует u32
    pub max_steps:         u32,   // [ 24]  WAS f32 — WGSL использует u32
    pub voxel_origin_x:    f32,   // [ 28]  НОВОЕ — мировое начало воксел-текстуры
    pub voxel_origin_y:    f32,   // [ 32]  НОВОЕ
    pub voxel_origin_z:    f32,   // [ 36]  НОВОЕ
    pub grid_origin_x:     f32,   // [ 40]
    pub grid_origin_y:     f32,   // [ 44]
    pub grid_origin_z:     f32,   // [ 48]
    pub opaque_threshold:  f32,   // [ 52]
    pub sky_brightness:    f32,   // [ 56]
    pub cascade_level:     u32,   // [ 60]  0=C0 finest, 4=C4 coarsest
}
// ---------------------------------------------------------------------------
// MergeParams — 96 bytes
// ---------------------------------------------------------------------------
// Matches MergeUniform in merge.wgsl.

// #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// #[repr(C)]
// pub struct MergeParams {
//     pub child_origin_x: f32,   // [  0]
//     pub child_origin_y: f32,   // [  4]
//     pub child_origin_z: f32,   // [  8]
//     pub child_spacing: f32,    // [ 12]
//     pub parent_origin_x: f32,  // [ 16]
//     pub parent_origin_y: f32,  // [ 20]
//     pub parent_origin_z: f32,  // [ 24]
//     pub parent_spacing: f32,   // [ 28]
//     pub child_ray_count: f32,  // [ 32]
//     pub parent_ray_count: f32, // [ 36]
//     pub child_grid_size: f32,  // [ 40]
//     pub parent_grid_size: f32, // [ 44]
//     pub tau_threshold: f32,    // [ 48] opacity clamp for merge
//     pub damping_factor: f32,   // [ 52] perpendicular displacement damping
//     pub child_level: f32,      // [ 56]
//     pub parent_level: f32,     // [ 60]
//     pub _pad0: f32,            // [ 64]
//     pub _pad1: f32,            // [ 68]
//     pub _pad2: f32,            // [ 72]
//     pub _pad3: f32,            // [ 76]
//     pub _pad4: f32,            // [ 80]
//     pub _pad5: f32,            // [ 84]
//     pub _pad6: f32,            // [ 88]
//     pub _pad7: f32,            // [ 92]
// }
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MergeParams {
    // -- Child cascade (WGSL строки 41-48) --
    pub child_grid_spacing:      f32,   // [  0]
    pub child_ray_start:         f32,   // [  4]
    pub child_ray_length:        f32,   // [  8]
    pub child_grid_half_extent:  f32,   // [ 12]
    pub child_grid_size:         u32,   // [ 16]
    pub child_ray_count:         u32,   // [ 20]
    pub _pad_c0:                 u32,   // [ 24]
    pub _pad_c1:                 u32,   // [ 28]
    // -- Parent cascade (WGSL строки 50-57) --
    pub parent_grid_spacing:     f32,   // [ 32]
    pub parent_ray_start:        f32,   // [ 36]
    pub parent_ray_length:       f32,   // [ 40]
    pub parent_grid_half_extent: f32,   // [ 44]
    pub parent_grid_size:        u32,   // [ 48]
    pub parent_ray_count:        u32,   // [ 52]
    pub _pad_p0:                 u32,   // [ 56]
    pub _pad_p1:                 u32,   // [ 60]
    // -- Grid origins (WGSL строки 59-66) --
    pub child_origin_x:          f32,   // [ 64]
    pub child_origin_y:          f32,   // [ 68]
    pub child_origin_z:          f32,   // [ 72]
    pub _pad_o0:                 u32,   // [ 76]
    pub parent_origin_x:         f32,   // [ 80]
    pub parent_origin_y:         f32,   // [ 84]
    pub parent_origin_z:         f32,   // [ 88]
    pub opaque_threshold:        f32,   // [ 92]
}
// ---------------------------------------------------------------------------
// GiCascadeEntry — 32 bytes (one entry per cascade level in GiSampleParams)
// ---------------------------------------------------------------------------
// Matches GiCascadeEntry in pbr_lighting.wgsl.
// Uses vec4 layout so AlignOf = 16, which satisfies WGSL uniform array rules
// (array element must have AlignOf >= 16 in uniform address space).

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GiCascadeEntry {
    /// [0..16] .xyz = grid world origin, .w = probe grid spacing
    pub origin_and_spacing: [f32; 4],
    /// [16..24] [0] = ray_count per probe, [1] = grid_size per axis
    pub counts: [u32; 2],
    pub _pad: [u32; 2],                    // [24..32] padding
    /// [32..44] Sub-block offset: compensates for snapped grid vs actual camera position.
    /// In the shader: `rel = world_pos - grid_origin + fractional_offset`
    pub fractional_offset: [f32; 3],       // [32..44]
    pub _pad2: u32,                         // [44..48] padding
}

// ---------------------------------------------------------------------------
// GiSampleParams — 256 bytes (fragment shader uniform)
// ---------------------------------------------------------------------------
// Matches GiSampleParams in pbr_lighting.wgsl.
// Contains one GiCascadeEntry per cascade level so the fragment shader can
// select the finest cascade that spatially covers each fragment.

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GiSampleParams {
    pub cascades: [GiCascadeEntry; NUM_CASCADES], // [0..240]  5 × 48B
    pub gi_intensity: f32,                        // [240]
    pub bounce_intensity: f32,                    // [244]
    pub _pad0: f32,                               // [248]
    pub _pad1: f32,                               // [252]
}

// ---------------------------------------------------------------------------
// EmissiveBlockGpu — 32 bytes (matches EmissiveBlock in emission_inject.wgsl)
// ---------------------------------------------------------------------------

/// GPU-side emissive block entry for point-light injection.
///
/// Layout: two `vec4<f32>` = 32 bytes. Matches `EmissiveBlock` in
/// `emission_inject.wgsl`.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct EmissiveBlockGpu {
    /// World-space position of block center (.xyz), .w = influence radius in blocks.
    pub pos_radius: [f32; 4],
    /// Emission color × intensity (.rgb), .a = attenuation factor k (inverse-square).
    pub color_atten: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<EmissiveBlockGpu>() == 32);

// ---------------------------------------------------------------------------
// Helper: ceil division for dispatch workgroup count
// ---------------------------------------------------------------------------

#[inline]
fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// RadianceCascadeSystemGpu
// ---------------------------------------------------------------------------

/// GPU-side orchestrator for the full Radiance Cascades pipeline.
///
/// Owns all wgpu resources: compute pipelines, textures, buffers, bind groups.
/// Created once at startup, used every frame.
pub struct RadianceCascadeSystemGpu {
    // ---- Configuration ----
    configs: [CascadeConfig; NUM_CASCADES],
    /// World-space origin of each cascade's probe grid (updated by recenter).
    grid_origins: [Vec3; NUM_CASCADES],
    /// Per-cascade fractional offset: compensates for snapped grid vs actual camera position.
    /// In the shader: `rel = world_pos - grid_origin + fractional_offset`
    fractional_offsets: [Vec3; NUM_CASCADES],

    // ---- Voxel texture (128^3 R16Uint) ----
    voxel_texture: wgpu::Texture,
    voxel_view: wgpu::TextureView,

    // ---- Block LUT (256x1 Rgba32Float) ----
    block_lut_texture: wgpu::Texture,
    block_lut_view: wgpu::TextureView,

    // ---- Per-cascade output buffers (RadianceInterval arrays) ----
    output_buffers: Vec<wgpu::Buffer>,
    /// Per-cascade ray direction buffers (vec4<f32> per direction, immutable).
    ray_dir_buffers: Vec<wgpu::Buffer>,
    /// Per-cascade uniform buffers for ray march (64B each).
    rm_uniform_buffers: Vec<wgpu::Buffer>,

    // ---- Shared merge uniform buffer (96B, rewritten per merge step) ----
    merge_uniform_buffers: Vec<wgpu::Buffer>,

    // ---- Ray March pipeline ----
    ray_march_pipeline: wgpu::ComputePipeline,
    /// Pre-created bind groups, one per cascade.
    rm_bind_groups: Vec<wgpu::BindGroup>,

    // ---- Merge pipeline ----
    merge_pipeline: wgpu::ComputePipeline,
    /// Pre-created bind groups for merge chain:
    ///   merge_bind_groups[0] -> C3<-C4  (parent=C4, child=C3)
    ///   merge_bind_groups[1] -> C2<-C3  (parent=C3, child=C2)
    ///   merge_bind_groups[2] -> C1<-C2  (parent=C2, child=C1)
    ///   merge_bind_groups[3] -> C0<-C1  (parent=C1, child=C0)
    merge_bind_groups: Vec<wgpu::BindGroup>,

    // ---- GI sampling (fragment shader) ----
    gi_params_buffer: wgpu::Buffer,
    /// Bind group layout for the fragment shader GI bind group.
    /// Game3DPipeline should include this layout when creating its pipeline.
    pub gi_bgl: wgpu::BindGroupLayout,
    /// Bind group for GI sampling: gi_params + c0 output buffer.
    pub gi_bg: wgpu::BindGroup,

    // ---- Emission Injection (Point Light) pipeline ----
    emission_inject_pipeline: wgpu::ComputePipeline,
    emission_inject_bgl: wgpu::BindGroupLayout,
    /// Bind group for emission injection into C0 (recreated when blocks change).
    emission_inject_bg: Option<wgpu::BindGroup>,
    /// Bind group for emission injection into C1 (recreated when blocks change).
    emission_inject_bg_c1: Option<wgpu::BindGroup>,
    /// Storage buffer for emissive block data (recreated when blocks change).
    emissive_blocks_buffer: Option<wgpu::Buffer>,
    /// Uniform buffer for emission injection into C0 (64B).
    emission_inject_uniform: wgpu::Buffer,
    /// Uniform buffer for emission injection into C1 (64B).
    emission_inject_uniform_c1: wgpu::Buffer,
    /// Number of emissive blocks currently in the buffer.
    emissive_block_count: u32,

    // ---- CPU-side builders ----
    voxel_builder: VoxelTextureBuilder,
    lut_builder: BlockLUTBuilder,

    // ---- Staggered dispatch ----
    /// Counts how many times dispatch() has been called.
    /// Used to stagger coarse cascade updates: C2-C4 only every 4th dispatch.
    dispatch_counter: u32,
}

/// Key RC parameters exposed to the debug overlay (CPU-side snapshot).
#[derive(Debug, Clone, Copy)]
pub struct RcDebugInfo {
    pub c0_grid_origin: Vec3,
    pub voxel_origin:   Vec3,
    pub c0_grid_size:   u32,
    pub c0_spacing:     f32,
    pub c0_rays:        u32,
    pub c0_ray_length:  f32,
}

impl RadianceCascadeSystemGpu {
    // ===================================================================
    // Construction
    // ===================================================================

    /// Create and initialize all GPU resources.
    ///
    /// Must be called once after the wgpu device is available.
    /// The shader files `shaders/ray_march.wgsl` and `shaders/merge.wgsl`
    /// must exist relative to this file.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        debug_log!(
            "RCSystemGpu", "new",
            "Initializing Radiance Cascade GPU system ({} cascades)",
            NUM_CASCADES
        );

        let configs = default_cascade_configs();

        // ----------------------------------------------------------------
        // 1. Voxel texture: 128^3 R16Uint
        // ----------------------------------------------------------------
        let voxel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rc_voxel_texture"),
            size: wgpu::Extent3d {
                width: VOXEL_TEX_SIZE,
                height: VOXEL_TEX_SIZE,
                depth_or_array_layers: VOXEL_TEX_SIZE,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let voxel_view = voxel_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rc_voxel_view"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        // Fill with sentinel (0xFFFF = unloaded)
        let sentinel: Vec<u16> = vec![0xFFFF; (VOXEL_TEX_SIZE as usize).pow(3)];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &voxel_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&sentinel),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VOXEL_TEX_SIZE * 2),
                rows_per_image: Some(VOXEL_TEX_SIZE),
            },
            wgpu::Extent3d {
                width: VOXEL_TEX_SIZE,
                height: VOXEL_TEX_SIZE,
                depth_or_array_layers: VOXEL_TEX_SIZE,
            },
        );
        drop(sentinel);

        // ----------------------------------------------------------------
        // 2. Block LUT texture: 256x1 Rgba32Float
        // ----------------------------------------------------------------
        let block_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("rc_block_lut"),
            size: wgpu::Extent3d {
                width: MAX_BLOCK_TYPES,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let block_lut_view = block_lut_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("rc_block_lut_view"),
            ..Default::default()
        });
        // Initialize with all-air LUT
        let lut_builder = BlockLUTBuilder::new();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &block_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(lut_builder.data()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MAX_BLOCK_TYPES * 16),
                rows_per_image: Some(1u32),
            },
            wgpu::Extent3d {
                width: MAX_BLOCK_TYPES,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        // ----------------------------------------------------------------
        // 3. Per-cascade resources
        // ----------------------------------------------------------------
        let mut output_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(NUM_CASCADES);
        let mut ray_dir_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(NUM_CASCADES);
        let mut rm_uniform_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(NUM_CASCADES);

        for i in 0..NUM_CASCADES {
            let cfg = &configs[i];
            let total_intervals = cfg.total_probes() * cfg.ray_count as usize;
            let output_bytes = total_intervals * RadianceInterval::SIZE as usize;

            // Output buffer (written by ray march / merge compute)
            output_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("rc_c{}_output", i)),
                size: output_bytes as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));

            // Ray direction buffer (immutable, uploaded once).
            // derive_subdiv maps ray_count → subdiv so cube_face_directions
            // returns exactly cfg.ray_count uniformly-spaced directions.
            let subdiv = derive_subdiv(cfg.ray_count);
            let directions = cube_face_directions(subdiv);
            let dir_data: Vec<f32> = directions
                .iter()
                .flat_map(|d| [d.x, d.y, d.z, 0.0f32])
                .collect();
            ray_dir_buffers.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("rc_c{}_ray_dirs", i)),
                contents: bytemuck::cast_slice(&dir_data),
                usage: wgpu::BufferUsages::STORAGE,
            }));

            // Ray march uniform buffer (64B, updated each frame)
            rm_uniform_buffers.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("rc_c{}_rm_uniform", i)),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            debug_log!(
                "RCSystemGpu", "new",
                "  C{}: grid={}^3, rays={}, intervals={}, output={:.1} KB",
                i, cfg.grid_size(), cfg.ray_count,
                total_intervals,
                output_bytes as f64 / 1024.0
            );
        }

        // ----------------------------------------------------------------
        // 4. Merge uniform buffer (96B, reused for each merge step)
        // ----------------------------------------------------------------
        let merge_uniform_buffers: Vec<wgpu::Buffer> = (0..NUM_CASCADES - 1)
            .map(|k| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rc_merge_uniform_{}", k)),
                    size: std::mem::size_of::<MergeParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // ----------------------------------------------------------------
        // 5. Ray March compute pipeline + bind groups
        // ----------------------------------------------------------------
        let rm_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_ray_march_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/ray_march.wgsl").into(),
            ),
        });

        let ray_march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_ray_march_bgl"),
            entries: &[
                // @binding(0) uniform CascadeParams (64B)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                    },
                    count: None,
                },
                // @binding(1) texture_3d<r16uint> voxel_tex
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) texture_2d<rgba32float> block_lut
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3) storage<read> ray_directions
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) storage<read_write> output_intervals
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let rm_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rc_ray_march_layout"),
            bind_group_layouts: &[Some(&ray_march_bgl)],
            immediate_size: 0,
        });

        let ray_march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_ray_march_pipeline"),
            layout: Some(&rm_layout),
            module: &rm_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind groups for each cascade
        let rm_bind_groups: Vec<wgpu::BindGroup> = (0..NUM_CASCADES)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("rc_rm_bg_{}", i)),
                    layout: &ray_march_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: rm_uniform_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&voxel_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&block_lut_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: ray_dir_buffers[i].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: output_buffers[i].as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        // ----------------------------------------------------------------
        // 6. Merge compute pipeline + bind groups
        // ----------------------------------------------------------------
        let merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_merge_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/merge.wgsl").into(),
            ),
        });

        let merge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_merge_bgl"),
            entries: &[
                // @binding(0) uniform MergeParams (96B)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(96).unwrap()),
                    },
                    count: None,
                },
                // @binding(1) storage<read> parent_intervals
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) storage<read_write> child_intervals
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let merge_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rc_merge_layout"),
            bind_group_layouts: &[Some(&merge_bgl)],
            immediate_size: 0,
        });

        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_merge_pipeline"),
            layout: Some(&merge_layout),
            module: &merge_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Create merge bind groups: merge_bind_groups[k] for step
        //   child = configs[k+1], parent = configs[k],   k = 0..NUM_CASCADES-1
        //   k=0: C3<-C4,  k=1: C2<-C3,  k=2: C1<-C2,  k=3: C0<-C1
        // Merge chain (coarse→fine):
        //   k=0: parent=C4 (idx=4), child=C3 (idx=3)
        //   k=1: parent=C3 (idx=3), child=C2 (idx=2)
        //   k=2: parent=C2 (idx=2), child=C1 (idx=1)
        //   k=3: parent=C1 (idx=1), child=C0 (idx=0)
        let merge_bind_groups: Vec<wgpu::BindGroup> = (0..NUM_CASCADES - 1)
            .map(|k| {
                let parent_idx = NUM_CASCADES - 1 - k; // 4, 3, 2, 1
                let child_idx  = parent_idx - 1;        // 3, 2, 1, 0
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("rc_merge_bg_c{}fromc{}", child_idx, parent_idx)),
                    layout: &merge_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: merge_uniform_buffers[k].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: output_buffers[parent_idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buffers[child_idx].as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();

        // ----------------------------------------------------------------
        // 7. GI sampling resources (for fragment shader)
        // ----------------------------------------------------------------
        // Bind group layout:
        //   @binding(0) uniform GiSampleParams (256B) — params for all 5 cascades
        //   @binding(1..5) storage<read> C0..C4 output interval buffers
        // Fragment shader picks the finest cascade whose grid covers the fragment.
        let gi_storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let gi_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_gi_sample_bgl"),
            entries: &[
                // @binding(0) uniform GiSampleParams (256B)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(256).unwrap()),
                    },
                    count: None,
                },
                gi_storage_entry(1), // C0 intervals
                gi_storage_entry(2), // C1 intervals
                gi_storage_entry(3), // C2 intervals
                gi_storage_entry(4), // C3 intervals
                gi_storage_entry(5), // C4 intervals
            ],
        });

        let gi_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_gi_params"),
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gi_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rc_gi_bg"),
            layout: &gi_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gi_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffers[0].as_entire_binding() }, // C0
                wgpu::BindGroupEntry { binding: 2, resource: output_buffers[1].as_entire_binding() }, // C1
                wgpu::BindGroupEntry { binding: 3, resource: output_buffers[2].as_entire_binding() }, // C2
                wgpu::BindGroupEntry { binding: 4, resource: output_buffers[3].as_entire_binding() }, // C3
                wgpu::BindGroupEntry { binding: 5, resource: output_buffers[4].as_entire_binding() }, // C4
            ],
        });

        // ----------------------------------------------------------------
        // 8. Emission Injection pipeline (point light into C0)
        // ----------------------------------------------------------------
        let emission_inject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rc_emission_inject_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/emission_inject.wgsl").into(),
            ),
        });

        let emission_inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rc_emission_inject_bgl"),
            entries: &[
                // @binding(0) uniform CascadeParams (64B) — C0 params
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                    },
                    count: None,
                },
                // @binding(1) storage<read> emissive_blocks
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) storage<read> ray_directions (C0)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3) storage<read_write> intervals (C0 output)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4) voxel texture (R16Uint 3D) — for occlusion ray march
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let emission_inject_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rc_emission_inject_layout"),
            bind_group_layouts: &[Some(&emission_inject_bgl)],
            immediate_size: 0,
        });

        let emission_inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rc_emission_inject_pipeline"),
            layout: Some(&emission_inject_layout),
            module: &emission_inject_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Emission inject uniform buffer for C0 (64B, same layout as CascadeDispatchParams)
        let emission_inject_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_emission_inject_uniform"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Emission inject uniform buffer for C1 (64B)
        let emission_inject_uniform_c1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rc_emission_inject_uniform_c1"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        debug_log!(
            "RCSystemGpu", "new",
            "GPU system initialized: {} RM bind groups, {} merge bind groups, emission inject ready",
            rm_bind_groups.len(),
            merge_bind_groups.len()
        );

        Self {
            configs,
            grid_origins: [Vec3::ZERO; NUM_CASCADES],
            fractional_offsets: [Vec3::ZERO; NUM_CASCADES],
            voxel_texture,
            voxel_view,
            block_lut_texture,
            block_lut_view,
            output_buffers,
            ray_dir_buffers,
            rm_uniform_buffers,
            merge_uniform_buffers,
            ray_march_pipeline,
            rm_bind_groups,
            merge_pipeline,
            merge_bind_groups,
            gi_params_buffer,
            gi_bgl,
            gi_bg,
            emission_inject_pipeline,
            emission_inject_bgl,
            emission_inject_bg: None,
            emission_inject_bg_c1: None,
            emissive_blocks_buffer: None,
            emission_inject_uniform,
            emission_inject_uniform_c1,
            emissive_block_count: 0,
            voxel_builder: VoxelTextureBuilder::new(),
            lut_builder,
            dispatch_counter: 0,
        }
    }

    // ===================================================================
    // Per-frame: recenter
    // ===================================================================

    /// Recenter all cascade probe grids around a new camera position.
    ///
    /// NOTE: The voxel texture builder is NOT recentered here. Its origin must
    /// stay in sync with the actual GPU texture data, which is only updated
    /// when the world worker sends a VoxelUpdate. If we recentered the voxel
    /// builder here, the shader would use a new origin while the GPU texture
    /// still contains data positioned for the old origin — causing all ray
    /// marches to read voxels at wrong positions (sky visibility breaks,
    /// lighting flickers with camera movement).
    ///
    /// The voxel builder origin is updated exclusively via `set_data_raw()`
    /// when a `VoxelUpdate::Full` arrives from the world worker.
    pub fn recenter(&mut self, camera_pos: Vec3) {
        for i in 0..NUM_CASCADES {
            let cfg = &self.configs[i];
            let spacing = cfg.grid_spacing;

            let half_extent = cfg.grid_half_extent as f32;

            // World-anchored probe grid.  Probe[k] world pos:
            //   grid_origin + (k + 0.5) * spacing
            // grid_origin snaps in steps of `spacing` (= 1 block for C0).
            // Between snaps the grid is 100 % static, so emissive-block GI
            // never drifts.  The 1-block step is visually invisible because
            // trilinear interpolation between adjacent probes (1 block apart)
            // is perfectly smooth.
            let snapped = (camera_pos / spacing).floor() * spacing;
            self.grid_origins[i] = snapped
                - Vec3::splat(half_extent * spacing)
                + Vec3::splat(spacing * 0.5);
            self.fractional_offsets[i] = Vec3::ZERO;
        }

        flow_debug_log!(
            "RCSystemGpu", "recenter",
            "Recentered to ({:.1}, {:.1}, {:.1})",
            camera_pos.x, camera_pos.y, camera_pos.z
        );
    }

    // ===================================================================
    // Per-frame: texture uploads
    // ===================================================================

    /// Upload the voxel texture to the GPU from the internal builder.
    ///
    /// The caller should fill `self.voxel_builder` with block data from
    /// World/Chunk before calling this (e.g. iterate loaded chunks, call
    /// set_block for each voxel).
    pub fn upload_voxel_texture(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.voxel_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(self.voxel_builder.data()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VOXEL_TEX_SIZE * 2),
                rows_per_image: Some(VOXEL_TEX_SIZE),
            },
            wgpu::Extent3d {
                width: VOXEL_TEX_SIZE,
                height: VOXEL_TEX_SIZE,
                depth_or_array_layers: VOXEL_TEX_SIZE,
            },
        );
    }

    /// Upload a single 16³ voxel-texture sub-region (Partial update path).
    ///
    /// `region.data` must be exactly 16³ = 4096 u16 elements.
    /// Layout: `data[dx + dy*16 + dz*16*16]` — matches write_texture row layout.
    pub fn upload_voxel_region(
        &self,
        queue: &wgpu::Queue,
        region: &crate::core::world_gen::world::VoxelChunkRegion,
    ) {
        let cs = 16u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &self.voxel_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d { x: region.tx, y: region.ty, z: region.tz },
                aspect:    wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&region.data),
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(cs * 2),   // 16 * 2 = 32 bytes per x-row
                rows_per_image: Some(cs),        // 16 y-rows per z-slice
            },
            wgpu::Extent3d {
                width:                 cs,
                height:                cs,
                depth_or_array_layers: cs,
            },
        );
    }

    /// Upload the block LUT texture to the GPU from the internal builder.
    ///
    /// The caller should fill `self.lut_builder` with BlockProperties
    /// from BlockRegistry before calling this.
    pub fn upload_block_lut(&self, queue: &wgpu::Queue) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.block_lut_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(self.lut_builder.data()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(MAX_BLOCK_TYPES * 16),
                rows_per_image: Some(1u32),
            },
            wgpu::Extent3d {
                width: MAX_BLOCK_TYPES,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
    }

    // ===================================================================
    // Emissive block buffer management
    // ===================================================================

    /// Update the GPU buffer of emissive blocks for point-light injection.
    ///
    /// Call this whenever the world worker reports new `emissive_blocks`.
    /// The method creates (or recreates) the storage buffer + bind group
    /// only when the data actually changes, so it is safe to call every
    /// frame with the same slice.
    ///
    /// `blocks` — slice of `EmissiveBlockGpu` entries (CPU-side).
    pub fn update_emissive_blocks(
        &mut self,
        device:  &wgpu::Device,
        _queue:  &wgpu::Queue,
        blocks: &[EmissiveBlockGpu],
    ) {
        let count = blocks.len() as u32;

        // Fast path: same count → just rewrite the buffer contents.
        if count > 0 && count == self.emissive_block_count {
            if let Some(ref buf) = self.emissive_blocks_buffer {
                _queue.write_buffer(buf, 0, bytemuck::cast_slice(blocks));
                return;
            }
        }

        // Count changed or first call → recreate buffer + bind groups.
        if count == 0 {
            // No emissive blocks — clear everything.
            self.emissive_blocks_buffer = None;
            self.emission_inject_bg = None;
            self.emission_inject_bg_c1 = None;
            self.emissive_block_count = 0;
            return;
        }

        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rc_emissive_blocks"),
            contents: bytemuck::cast_slice(blocks),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // C0 bind group
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rc_emission_inject_bg"),
            layout: &self.emission_inject_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.emission_inject_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_dir_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.voxel_view),
                },
            ],
        });

        // C1 bind group — same emissive blocks, but C1's uniform/ray_dirs/output
        let bg_c1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rc_emission_inject_bg_c1"),
            layout: &self.emission_inject_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.emission_inject_uniform_c1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.ray_dir_buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.voxel_view),
                },
            ],
        });

        self.emissive_blocks_buffer = Some(buf);
        self.emission_inject_bg = Some(bg);
        self.emission_inject_bg_c1 = Some(bg_c1);
        self.emissive_block_count = count;

        debug_log!(
            "RCSystemGpu", "update_emissive_blocks",
            "Updated emissive block buffer: {} entries ({} bytes)",
            count, count * 32
        );
    }

    // ===================================================================
    // Per-frame: dispatch full cascade pipeline
    // ===================================================================

    /// Dispatch the complete Radiance Cascades pipeline for one frame.
    ///
    /// Encodes into the given command encoder (no submit — caller submits).
    ///
    /// Steps:
    ///   1. For each cascade C4..C0: dispatch ray march compute
    ///   2. Merge chain: C3<-C4, C2<-C3, C1<-C2, C0<-C1
    ///   3. Update GI params uniform for fragment shader
    ///
    /// `sky_brightness`: 0.0 (night) .. 1.0 (day), modulates emission.
    pub fn dispatch(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        sky_brightness: f32,
    ) {
        self.dispatch_counter = self.dispatch_counter.wrapping_add(1);

        // Staggered dispatch: coarse cascades (C2-C4) are only re-marched
        // every 4th dispatch because they change slowly (large spacing, far
        // from camera). C0-C1 are always marched for responsive near-field GI.
        // The output buffers persist in VRAM, so stale coarse data is reused.
        let full_dispatch = self.dispatch_counter % 4 == 0;

        flow_debug_log!(
            "RCSystemGpu", "dispatch",
            "Dispatching RC pipeline (sky_brightness={:.2}, full={})",
            sky_brightness, full_dispatch
        );

        // ----------------------------------------------------------------
        // Pass 1: Ray march — C0-C1 always, C2-C4 only on full dispatch
        // ----------------------------------------------------------------

        let march_upper = if full_dispatch { NUM_CASCADES } else { 2 };
        for i in 0..march_upper {
            let cfg = &self.configs[i];
            let origin = &self.grid_origins[i];

            // Update uniform buffer
            // let params = CascadeDispatchParams {
            //     grid_origin_x: origin.x,
            //     grid_origin_y: origin.y,
            //     grid_origin_z: origin.z,
            //     grid_spacing: cfg.grid_spacing,
            //     ray_count: cfg.ray_count as f32,
            //     grid_size: cfg.grid_size() as f32,
            //     max_steps: MAX_RAY_STEPS as f32,
            //     max_dist: cfg.ray_length,
            //     ray_start: cfg.ray_start,
            //     opaque_threshold: OPAQUE_THRESHOLD,
            //     sky_brightness,
            //     _pad0: 0.0,
            //     _pad1: 0.0,
            //     _pad2: 0.0,
            //     _pad3: 0.0,
            //     _pad4: 0.0,
            // };
            // Берём мировое начало воксел-текстуры из VoxelTextureBuilder
            let vox_origin = self.voxel_builder.world_origin();

            let params = CascadeDispatchParams {
                grid_spacing:      cfg.grid_spacing,
                ray_length:        cfg.ray_length,
                ray_start:         cfg.ray_start,
                grid_half_extent:  cfg.grid_half_extent as f32,
                grid_size:         cfg.grid_size(),
                ray_count:         cfg.ray_count,
                max_steps:         MAX_RAY_STEPS,
                voxel_origin_x:    vox_origin.x as f32,
                voxel_origin_y:    vox_origin.y as f32,
                voxel_origin_z:    vox_origin.z as f32,
                grid_origin_x:     origin.x,
                grid_origin_y:     origin.y,
                grid_origin_z:     origin.z,
                opaque_threshold:  OPAQUE_THRESHOLD,
                sky_brightness,
                cascade_level:     i as u32,
            };
            queue.write_buffer(&self.rm_uniform_buffers[i], 0, bytemuck::cast_slice(&[params]));

            // Dispatch
            let total_probes = cfg.total_probes();
            let workgroup_size = 64usize;
            let invocations = total_probes * cfg.ray_count as usize;
            let dispatch_x = ceil_div(invocations, workgroup_size);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("rc_ray_march_c{}", i)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.ray_march_pipeline);
                pass.set_bind_group(0, &self.rm_bind_groups[i], &[]);
                pass.dispatch_workgroups(dispatch_x as u32, 1, 1);
            }

            flow_debug_log!(
                "RCSystemGpu", "dispatch",
                "  C{}: {} probes x {} rays = {} invocations ({} wg)",
                i, total_probes, cfg.ray_count, invocations, dispatch_x
            );
        }

        // ----------------------------------------------------------------
        // Pass 1.5: Emission Injection (point light into C0)
        // ----------------------------------------------------------------
        // Injects point light contributions from emissive blocks into C0
        // probe intervals. Runs AFTER ray marching, BEFORE merging, so
        // injected emission propagates through the entire merge chain.
        if self.emissive_block_count > 0 {
            if let Some(ref ei_bg) = self.emission_inject_bg {
                // Write C0 cascade params to the emission inject uniform
                let c0_cfg = &self.configs[0];
                let c0_origin = self.grid_origins[0];
                let vox_origin = self.voxel_builder.world_origin();
                let ei_params = CascadeDispatchParams {
                    grid_spacing:      c0_cfg.grid_spacing,
                    ray_length:        c0_cfg.ray_length,
                    ray_start:         c0_cfg.ray_start,
                    grid_half_extent:  c0_cfg.grid_half_extent as f32,
                    grid_size:         c0_cfg.grid_size(),
                    ray_count:         c0_cfg.ray_count,
                    max_steps:         MAX_RAY_STEPS,
                    voxel_origin_x:    vox_origin.x as f32,
                    voxel_origin_y:    vox_origin.y as f32,
                    voxel_origin_z:    vox_origin.z as f32,
                    grid_origin_x:     c0_origin.x,
                    grid_origin_y:     c0_origin.y,
                    grid_origin_z:     c0_origin.z,
                    opaque_threshold:  OPAQUE_THRESHOLD,
                    sky_brightness,
                    cascade_level:     0,
                };
                queue.write_buffer(&self.emission_inject_uniform, 0, bytemuck::cast_slice(&[ei_params]));

                let total_probes = c0_cfg.total_probes();
                let invocations = total_probes * c0_cfg.ray_count as usize;
                let dispatch_x = ceil_div(invocations, 64);

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rc_emission_inject"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.emission_inject_pipeline);
                    pass.set_bind_group(0, ei_bg, &[]);
                    pass.dispatch_workgroups(dispatch_x as u32, 1, 1);
                }

                flow_debug_log!(
                    "RCSystemGpu", "dispatch",
                    "  Emission inject C0: {} blocks, {} invocations ({} wg)",
                    self.emissive_block_count, invocations, dispatch_x
                );
            }

            // --- C1 emission injection ---
            // Emissive blocks use world-space positions (correct), but C0's probe
            // grid moves with the camera. Injecting into C1 as well stabilises
            // point-light contributions across grid movement.
            if let Some(ref ei_bg_c1) = self.emission_inject_bg_c1 {
                let c1_cfg = &self.configs[1];
                let c1_origin = self.grid_origins[1];
                let vox_origin = self.voxel_builder.world_origin();
                let ei_params_c1 = CascadeDispatchParams {
                    grid_spacing:      c1_cfg.grid_spacing,
                    ray_length:        c1_cfg.ray_length,
                    ray_start:         c1_cfg.ray_start,
                    grid_half_extent:  c1_cfg.grid_half_extent as f32,
                    grid_size:         c1_cfg.grid_size(),
                    ray_count:         c1_cfg.ray_count,
                    max_steps:         MAX_RAY_STEPS,
                    voxel_origin_x:    vox_origin.x as f32,
                    voxel_origin_y:    vox_origin.y as f32,
                    voxel_origin_z:    vox_origin.z as f32,
                    grid_origin_x:     c1_origin.x,
                    grid_origin_y:     c1_origin.y,
                    grid_origin_z:     c1_origin.z,
                    opaque_threshold:  OPAQUE_THRESHOLD,
                    sky_brightness,
                    cascade_level:     1,
                };
                queue.write_buffer(&self.emission_inject_uniform_c1, 0, bytemuck::cast_slice(&[ei_params_c1]));

                let total_probes_c1 = c1_cfg.total_probes();
                let invocations_c1 = total_probes_c1 * c1_cfg.ray_count as usize;
                let dispatch_x_c1 = ceil_div(invocations_c1, 64);

                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("rc_emission_inject_c1"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.emission_inject_pipeline);
                    pass.set_bind_group(0, ei_bg_c1, &[]);
                    pass.dispatch_workgroups(dispatch_x_c1 as u32, 1, 1);
                }

                flow_debug_log!(
                    "RCSystemGpu", "dispatch",
                    "  Emission inject C1: {} blocks, {} invocations ({} wg)",
                    self.emissive_block_count, invocations_c1, dispatch_x_c1
                );
            }
        }

        // ----------------------------------------------------------------
        // Pass 2: Merge chain  C3<-C4 -> C2<-C3 -> C1<-C2 -> C0<-C1
        // ----------------------------------------------------------------
        // merge_bind_groups[k]:
        //   k=0: parent=C4 (idx=4), child=C3 (idx=3)
        //   k=1: parent=C3 (idx=3), child=C2 (idx=2)
        //   k=2: parent=C2 (idx=2), child=C1 (idx=1)
        //   k=3: parent=C1 (idx=1), child=C0 (idx=0)
        //
        // On non-full dispatch, skip coarse merges (C3←C4, C2←C3) since
        // neither parent nor child was re-marched — results would be identical.
        // Only merge C1←C2 and C0←C1 (k=2,3) to propagate stale coarse data
        // into the freshly marched C0-C1.
        let merge_start = if full_dispatch { 0 } else { NUM_CASCADES - 3 }; // skip first 2 steps

        for k in merge_start..NUM_CASCADES - 1 {
            // Coarse → fine: k=0 → parent=C4,child=C3 … k=3 → parent=C1,child=C0
            let parent_idx = NUM_CASCADES - 1 - k; // 4, 3, 2, 1
            let child_idx  = parent_idx - 1;        // 3, 2, 1, 0

            let parent_cfg = &self.configs[parent_idx];
            let child_cfg  = &self.configs[child_idx];
            let parent_origin = self.grid_origins[parent_idx];
            let child_origin  = self.grid_origins[child_idx];

            let mp = MergeParams {
                // -- Child cascade --
                child_grid_spacing:      child_cfg.grid_spacing,
                child_ray_start:         child_cfg.ray_start,
                child_ray_length:        child_cfg.ray_length,
                child_grid_half_extent:  child_cfg.grid_half_extent as f32,
                child_grid_size:         child_cfg.grid_size(),
                child_ray_count:         child_cfg.ray_count,
                _pad_c0:                 0,
                _pad_c1:                 0,
                // -- Parent cascade --
                parent_grid_spacing:     parent_cfg.grid_spacing,
                parent_ray_start:        parent_cfg.ray_start,
                parent_ray_length:       parent_cfg.ray_length,
                parent_grid_half_extent: parent_cfg.grid_half_extent as f32,
                parent_grid_size:        parent_cfg.grid_size(),
                parent_ray_count:        parent_cfg.ray_count,
                _pad_p0:                 0,
                _pad_p1:                 0,
                // -- Grid origins --
                child_origin_x:          child_origin.x,
                child_origin_y:          child_origin.y,
                child_origin_z:          child_origin.z,
                _pad_o0:                 0,
                parent_origin_x:         parent_origin.x,
                parent_origin_y:         parent_origin.y,
                parent_origin_z:         parent_origin.z,
                opaque_threshold:        OPAQUE_THRESHOLD,
            };
            // Write to per-pass buffer (no aliasing between passes)
            queue.write_buffer(&self.merge_uniform_buffers[k], 0, bytemuck::cast_slice(&[mp]));

            // Dispatch: 1 invocation = 1 (child_probe, child_ray)
            let child_probes = child_cfg.total_probes();
            let child_rays = child_cfg.ray_count as usize;
            let invocations = child_probes * child_rays;
            let dispatch_x = ceil_div(invocations, 64usize);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("rc_merge_c{}fromc{}", child_idx, parent_idx)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.merge_pipeline);
                pass.set_bind_group(0, &self.merge_bind_groups[k], &[]);
                pass.dispatch_workgroups(dispatch_x as u32, 1, 1);
            }

            flow_debug_log!(
                "RCSystemGpu", "dispatch",
                "  Merge C{}<-C{}: {} child probes x {} rays = {} invocations",
                child_idx, parent_idx, child_probes, child_rays, invocations
            );
        }

        // ----------------------------------------------------------------
        // Pass 3: Update GI params for fragment shader (all 5 cascades)
        // ----------------------------------------------------------------
        // Each GiCascadeEntry carries origin+spacing+ray_count+grid_size for
        // one cascade level. The fragment shader picks the finest cascade whose
        // probe grid spatially covers the fragment, giving full-scene GI coverage.
        let mut cascade_entries = [GiCascadeEntry {
            origin_and_spacing: [0.0; 4],
            counts: [0; 2],
            _pad: [0; 2],
            fractional_offset: [0.0; 3],
            _pad2: 0,
        }; NUM_CASCADES];
        for i in 0..NUM_CASCADES {
            let cfg    = &self.configs[i];
            let origin = self.grid_origins[i];
            let fo     = self.fractional_offsets[i];
            cascade_entries[i] = GiCascadeEntry {
                origin_and_spacing: [origin.x, origin.y, origin.z, cfg.grid_spacing],
                counts: [cfg.ray_count, cfg.grid_size()],
                _pad: [0; 2],
                fractional_offset: [fo.x, fo.y, fo.z],
                _pad2: 0,
            };
        }
        let gi_params = GiSampleParams {
            cascades: cascade_entries,
            gi_intensity: 0.08,      // Full GI intensity — emissive blocks must illuminate surroundings.
                                    // Was 0.08 (too dim — emission injection was invisible).
                                    // The merge chain + trilinear fix keep sky GI stable.
            bounce_intensity: 1.0,  // Block emission bounce multiplier
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.gi_params_buffer, 0, bytemuck::cast_slice(&[gi_params]));

        flow_debug_log!(
            "RCSystemGpu", "dispatch",
            "GI params updated: C0 origin=({:.1},{:.1},{:.1}) spacing={:.1}, C4 spacing={:.1}",
            self.grid_origins[0].x, self.grid_origins[0].y, self.grid_origins[0].z,
            self.configs[0].grid_spacing, self.configs[NUM_CASCADES - 1].grid_spacing
        );
    }

    // ===================================================================
    // Accessors for integration
    // ===================================================================

    /// Mutable reference to the voxel builder (for filling from World chunks).
    pub fn voxel_builder_mut(&mut self) -> &mut VoxelTextureBuilder {
        &mut self.voxel_builder
    }

    /// Mutable reference to the LUT builder (for filling from BlockRegistry).
    pub fn lut_builder_mut(&mut self) -> &mut BlockLUTBuilder {
        &mut self.lut_builder
    }

    /// Number of emissive blocks currently in the GPU buffer.
    pub fn emissive_block_count(&self) -> u32 {
        self.emissive_block_count
    }

    /// Reference to cascade configs (read-only).
    pub fn configs(&self) -> &[CascadeConfig; NUM_CASCADES] {
        &self.configs
    }

    /// World-space origin of a cascade's probe grid.
    pub fn grid_origin(&self, cascade_idx: usize) -> Vec3 {
        self.grid_origins[cascade_idx]
    }

    /// Reference to the voxel texture view (for voxel ray-traced shadows).
    pub fn voxel_view(&self) -> &wgpu::TextureView {
        &self.voxel_view
    }

    /// Reference to the block LUT texture view (for voxel ray-traced shadows).
    pub fn block_lut_view(&self) -> &wgpu::TextureView {
        &self.block_lut_view
    }

    /// World-space origin of the voxel texture (block coordinates of texel 0,0,0).
    /// Used by the fragment shader to convert world positions to voxel texture coordinates.
    pub fn voxel_world_origin(&self) -> glam::IVec3 {
        self.voxel_builder.world_origin()
    }

    /// Size of the voxel texture per axis (256).
    pub fn voxel_tex_size(&self) -> u32 {
        super::voxel_tex::VOXEL_TEX_SIZE
    }

    /// Snapshot of key RC parameters for the debug overlay.
    pub fn debug_info(&self) -> RcDebugInfo {
        let c0 = &self.configs[0];
        RcDebugInfo {
            c0_grid_origin:  self.grid_origins[0],
            voxel_origin:    self.voxel_builder.world_origin().as_vec3(),
            c0_grid_size:    c0.grid_size(),
            c0_spacing:      c0.grid_spacing,
            c0_rays:         c0.ray_count,
            c0_ray_length:   c0.ray_length,
        }
    }

    /// Estimated total GPU memory usage in bytes.
    pub fn estimated_gpu_memory(&self) -> u64 {
        let mut total = 0u64;

        // Voxel texture
        total += (VOXEL_TEX_SIZE as u64).pow(3) * 2; // R16Uint

        // Block LUT
        total += MAX_BLOCK_TYPES as u64 * 16; // Rgba32Float

        // Per-cascade output buffers + ray dir buffers + uniform buffers
        for i in 0..NUM_CASCADES {
            let cfg = &self.configs[i];
            let total_intervals = cfg.total_probes() * cfg.ray_count as usize;
            total += total_intervals as u64 * RadianceInterval::SIZE;
            total += cfg.ray_count as u64 * 16; // ray dirs (vec4 per direction)
            total += 64; // rm uniform
        }

        // Merge uniform
        total += 96;

        // GI params
        total += 256;

        total
    }
}
