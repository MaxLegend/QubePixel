// =============================================================================
// QubePixel — Radiance Cascades: GPU Dispatch
// =============================================================================
//
// Creates and manages the wgpu compute pipeline for DDA ray marching.
// Responsibilities:
//   1. Build the WGSL compute shader module.
//   2. Create bind group layout (uniform, voxel tex, block LUT, directions, output).
//   3. Pre-upload Fibonacci sphere directions to GPU storage buffer.
//   4. Per-frame: write cascade params, create bind group, encode dispatch.
//
// Usage:
//   let pipeline = RayMarchPipeline::new(device, queue, 96);
//   let output_buf = create_output_buffer(device, &config);
//   let bg = pipeline.create_bind_group(device, &voxel_view, &lut_view, &output_buf);
//   pipeline.dispatch(encoder, queue, &bg, &params);
//   // ... later: read output_buf in merge shader or map for CPU readback

use std::num::NonZeroU64;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor,
    Queue, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    TextureSampleType, TextureView, TextureViewDimension,
};

use crate::debug_log;

use super::types::{
    CascadeConfig, MAX_RAY_STEPS, NUM_CASCADES, OPAQUE_THRESHOLD,
    RadianceInterval,
};

use super::sampling::fibonacci_sphere_directions;

// ---------------------------------------------------------------------------
// WGSL shader source (embedded at compile time)
// ---------------------------------------------------------------------------

const RAY_MARCH_SHADER: &str = include_str!("shaders/ray_march.wgsl");

// ---------------------------------------------------------------------------
// CascadeDispatchParams (64 bytes, matches WGSL CascadeParams exactly)
// ---------------------------------------------------------------------------

/// GPU-safe dispatch parameters for a single cascade.
///
/// Sent as a uniform buffer (64 bytes) to the ray march compute shader.
/// Field order and types must match the WGSL `struct CascadeParams` exactly.
///
/// Layout: 16 × f32/u32 × 4 bytes = 64 bytes.
/// Uniform buffer min alignment = 16 bytes (64 is already aligned).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CascadeDispatchParams {
    pub grid_spacing: f32,
    pub ray_length: f32,
    pub ray_start: f32,
    pub grid_half_extent: f32,
    pub grid_size: u32,
    pub ray_count: u32,
    pub max_steps: u32,
    pub voxel_origin_x: f32,
    pub voxel_origin_y: f32,
    pub voxel_origin_z: f32,
    pub grid_origin_x: f32,
    pub grid_origin_y: f32,
    pub grid_origin_z: f32,
    pub opaque_threshold: f32,
    pub sky_brightness: f32,
    pub _pad: f32,
}

impl CascadeDispatchParams {
    /// Byte size of this struct (must be 64).
    pub const SIZE: u64 = 64;

    /// Build dispatch params from a cascade config, voxel origin, and grid origin.
    pub fn from_cascade(
        config: &CascadeConfig,
        voxel_origin: glam::IVec3,
        grid_origin: glam::Vec3,
        sky_brightness: f32,
    ) -> Self {
        debug_log!(
            "CascadeDispatchParams", "from_cascade",
            "cascade={}, spacing={:.1}, grid_size={}, ray_count={}, sky={:.3}",
            config.level, config.grid_spacing, config.grid_size(),
            config.ray_count, sky_brightness
        );

        Self {
            grid_spacing: config.grid_spacing,
            ray_length: config.ray_length,
            ray_start: config.ray_start,
            grid_half_extent: config.grid_half_extent as f32,
            grid_size: config.grid_size(),
            ray_count: config.ray_count,
            max_steps: MAX_RAY_STEPS,
            voxel_origin_x: voxel_origin.x as f32,
            voxel_origin_y: voxel_origin.y as f32,
            voxel_origin_z: voxel_origin.z as f32,
            grid_origin_x: grid_origin.x,
            grid_origin_y: grid_origin.y,
            grid_origin_z: grid_origin.z,
            opaque_threshold: OPAQUE_THRESHOLD,
            sky_brightness,
            _pad: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// RayMarchPipeline
// ---------------------------------------------------------------------------

/// Owns the compute pipeline, bind group layout, uniform buffer, and
/// the pre-computed Fibonacci sphere direction buffer.
///
/// One instance per RadianceCascadeSystem. Shared across all cascade levels —
/// only the bind group and uniform contents change per cascade.
pub struct RayMarchPipeline {
    pipeline: ComputePipeline,
    bgl: BindGroupLayout,
    /// Uniform buffer holding CascadeDispatchParams (64 bytes).
    uniform_buffer: Buffer,
    /// Storage buffer of ray directions: array<vec4<f32>>, max_ray_count entries.
    ray_dir_buffer: Buffer,
    /// Maximum number of ray directions this pipeline was built for.
    max_ray_count: u32,
}

impl RayMarchPipeline {
    /// Create the ray march pipeline and associated GPU resources.
    ///
    /// `max_ray_count`: must be >= the largest ray_count across all cascades
    /// (C4 has 96 rays, so pass 96 or higher).
    pub fn new(device: &Device, queue: &Queue, max_ray_count: u32) -> Self {
        debug_log!(
            "RayMarchPipeline", "new",
            "Creating ray march pipeline, max_ray_count={}", max_ray_count
        );

        let bgl = Self::create_bgl(device);
        let pipeline = Self::create_pipeline(device, &bgl);
        let uniform_buffer = Self::create_uniform_buffer(device);
        let ray_dir_buffer = Self::create_ray_dir_buffer(device, queue, max_ray_count);

        debug_log!(
            "RayMarchPipeline", "new",
            "Pipeline created: uniform={}B, ray_dirs={}×16B",
            CascadeDispatchParams::SIZE,
            max_ray_count
        );

        Self {
            pipeline,
            bgl,
            uniform_buffer,
            ray_dir_buffer,
            max_ray_count,
        }
    }

    // -- Internal builders --------------------------------------------------

    fn create_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rc_ray_march_bgl"),
            entries: &[
                // @binding(0): CascadeParams uniform buffer
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(CascadeDispatchParams::SIZE).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1): Voxel texture (R16Uint, texture_3d<u32>)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Uint,
                        view_dimension: TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2): Block LUT (Rgba32Float, texture_2d<f32>)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: false },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(3): Ray directions (storage buffer, read)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(4): Output intervals (storage buffer, read_write)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_pipeline(device: &Device, bgl: &BindGroupLayout) -> ComputePipeline {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rc_ray_march_shader"),
            source: ShaderSource::Wgsl(RAY_MARCH_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rc_ray_march_layout"),
            bind_group_layouts: &[Option::from(bgl)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("rc_ray_march_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("ray_march_main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn create_uniform_buffer(device: &Device) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("rc_ray_march_uniform"),
            size: CascadeDispatchParams::SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Pre-compute Fibonacci sphere directions and upload to GPU storage buffer.
    fn create_ray_dir_buffer(device: &Device, queue: &Queue, max_ray_count: u32) -> Buffer {
        let dirs = fibonacci_sphere_directions(max_ray_count);
        // Pack as [f32; 4] for GPU: .xyz = direction, .w = 0
        let dir_data: Vec<[f32; 4]> = dirs
            .iter()
            .map(|d| [d.x, d.y, d.z, 0.0])
            .collect();

        let buffer_size = (max_ray_count as u64) * 16; // vec4<f32> = 16 bytes

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("rc_ray_directions"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&dir_data));

        debug_log!(
            "RayMarchPipeline", "create_ray_dir_buffer",
            "Uploaded {} Fibonacci directions ({} bytes)",
            max_ray_count, buffer_size
        );

        buffer
    }

    // -- Public API ---------------------------------------------------------

    /// Create a bind group for a specific cascade's dispatch.
    ///
    /// The `output_buffer` must be large enough for
    /// `config.total_probes() * config.ray_count` RadianceIntervals.
    pub fn create_bind_group(
        &self,
        device: &Device,
        voxel_view: &TextureView,
        block_lut_view: &TextureView,
        output_buffer: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("rc_ray_march_bg"),
            layout: &self.bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(voxel_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(block_lut_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.ray_dir_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode a ray march compute dispatch for one cascade level.
    ///
    /// 1. Writes `params` to the uniform buffer.
    /// 2. Begins a compute pass and dispatches workgroups.
    ///
    /// The caller is responsible for:
    ///   - Ensuring `bind_group` was created with matching views/buffers.
    ///   - Ensuring `output_buffer` is large enough.
    ///   - Submitting the encoder after all passes are recorded.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        bind_group: &BindGroup,
        params: &CascadeDispatchParams,
    ) {
        // Upload params to uniform buffer
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[*params]),
        );

        // Compute total invocations and workgroup count
        let gs = params.grid_size as u64;
        let rays = params.ray_count as u64;
        let total_invocations = gs * gs * gs * rays;

        let workgroup_size: u64 = 64;
        let dispatch_x = (total_invocations + workgroup_size - 1) / workgroup_size;

        debug_log!(
            "RayMarchPipeline", "dispatch",
            "grid={}³={}, rays={}, total={}, workgroups={}",
            params.grid_size, gs * gs * gs, params.ray_count,
            total_invocations, dispatch_x
        );

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("rc_ray_march_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x as u32, 1, 1);

        // pass is dropped here, ending the compute pass
    }

    /// Maximum ray count this pipeline was built for.
    #[inline]
    pub fn max_ray_count(&self) -> u32 {
        self.max_ray_count
    }

    /// Reference to the bind group layout (for pipeline compatibility checks).
    #[inline]
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bgl
    }
}

// ---------------------------------------------------------------------------
// Output buffer helpers
// ---------------------------------------------------------------------------

/// Create a storage buffer large enough for all probe intervals of one cascade.
///
/// Capacity = `total_probes × ray_count` RadianceIntervals (48 bytes each).
/// The buffer has `STORAGE | COPY_SRC` usage so it can be read by the merge
/// shader (Step 3) and optionally mapped for CPU readback (debugging).
pub fn create_output_buffer(device: &Device, config: &CascadeConfig) -> Buffer {
    let total_probes = config.total_probes();
    let total_intervals = total_probes * config.ray_count as usize;
    let byte_size = (total_intervals * RadianceInterval::SIZE as usize) as u64;

    debug_log!(
        "RayMarchPipeline", "create_output_buffer",
        "cascade={}, probes={}, rays={}, intervals={}, size={:.2} MB",
        config.level, total_probes, config.ray_count,
        total_intervals, byte_size as f64 / (1024.0 * 1024.0)
    );

    device.create_buffer(&BufferDescriptor {
        label: Some(&format!("rc_ray_march_output_c{}", config.level)),
        size: byte_size.max(16), // wgpu requires min 16 bytes for storage buffers
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

/// Create an output buffer with explicit size (for combined or custom layouts).
pub fn create_output_buffer_sized(device: &Device, num_intervals: usize) -> Buffer {
    let byte_size = (num_intervals * RadianceInterval::SIZE as usize) as u64;
    device.create_buffer(&BufferDescriptor {
        label: Some("rc_ray_march_output_custom"),
        size: byte_size.max(16),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use super::*;

    #[test]
    fn test_cascade_dispatch_params_size() {
        assert_eq!(
            std::mem::size_of::<CascadeDispatchParams>(),
            64,
            "CascadeDispatchParams must be exactly 64 bytes"
        );
        assert_eq!(CascadeDispatchParams::SIZE, 64);
    }

    #[test]
    fn test_cascade_dispatch_params_layout() {
        let params = CascadeDispatchParams {
            grid_spacing: 4.0,
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 16.0,
            grid_size: 33,
            ray_count: 6,
            max_steps: 256,
            voxel_origin_x: -64.0,
            voxel_origin_y: -64.0,
            voxel_origin_z: -64.0,
            grid_origin_x: -64.0,
            grid_origin_y: -64.0,
            grid_origin_z: -64.0,
            opaque_threshold: 500.0,
            sky_brightness: 0.3,
            _pad: 0.0,
        };

        // Verify field accessibility
        assert!((params.grid_spacing - 4.0).abs() < 1e-6);
        assert_eq!(params.grid_size, 33);
        assert_eq!(params.ray_count, 6);
    }

    #[test]
    fn test_from_cascade_c0() {
        let config = CascadeConfig {
            level: 0,
            grid_spacing: 4.0,
            ray_count: 6,
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 16,
        };
        let voxel_origin = glam::IVec3::new(-64, -64, -64);
        let grid_origin = glam::Vec3::new(-64.0, -64.0, -64.0);

        let params = CascadeDispatchParams::from_cascade(
            &config, voxel_origin, grid_origin, 0.3,
        );

        assert!((params.grid_spacing - 4.0).abs() < 1e-6);
        assert_eq!(params.grid_size, 33);
        assert_eq!(params.ray_count, 6);
        assert_eq!(params.max_steps, MAX_RAY_STEPS);
        assert!((params.sky_brightness - 0.3).abs() < 1e-6);
        assert!((params.voxel_origin_x - (-64.0)).abs() < 1e-6);
    }

    #[test]
    fn test_from_cascade_c4() {
        let config = CascadeConfig {
            level: 4,
            grid_spacing: 64.0,
            ray_count: 96,
            ray_length: 96.0,
            ray_start: 114.0, // cumulative from C0..C3
            grid_half_extent: 1,
        };
        let voxel_origin = glam::IVec3::ZERO;
        let grid_origin = glam::Vec3::ZERO;

        let params = CascadeDispatchParams::from_cascade(
            &config, voxel_origin, grid_origin, 0.1,
        );

        assert!((params.grid_spacing - 64.0).abs() < 1e-6);
        assert_eq!(params.grid_size, 3);
        assert_eq!(params.ray_count, 96);
        assert!((params.ray_start - 114.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_buffer_size() {
        // C0: 33³ × 6 × 48 = 10,349,856 bytes
        let config = CascadeConfig {
            level: 0,
            grid_spacing: 4.0,
            ray_count: 6,
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 16,
        };
        let total = config.total_probes() * config.ray_count as usize;
        let expected_bytes = total * 48;
        assert_eq!(expected_bytes, 10_349_856);
    }

    #[test]
    fn test_pod_zeroable() {
        // Verify CascadeDispatchParams is Pod + Zeroable
        let zero = CascadeDispatchParams::zeroed();
        assert_eq!(zero.grid_spacing, 0.0);
        assert_eq!(zero.grid_size, 0);

        // Verify it's Copy
        let copy = zero;
        assert_eq!(copy.grid_spacing, zero.grid_spacing);
    }
}