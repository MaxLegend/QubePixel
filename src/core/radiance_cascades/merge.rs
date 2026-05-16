// =============================================================================
// QubePixel — Radiance Cascades: Merge Pipeline & Dispatch
// =============================================================================
//
// Creates and manages the wgpu compute pipeline for cascading interval merge
// with Trilinear Fix.
//
// Responsibilities:
//   1. Build the WGSL merge shader module.
//   2. Create bind group layout (uniform, parent storage, child storage).
//   3. Per-cascade-pair: write MergeParams, create bind group, encode dispatch.
//   4. Orchestrate the full cascade chain: C4→C3→C2→C1→C0.
//
// Usage:
//   let pipeline = MergePipeline::new(device);
//   dispatch_all_merges(device, &pipeline, encoder, queue,
//       &configs, &grid_origins, &output_buffers, OPAQUE_THRESHOLD);
//   // output_buffers[0] now contains the final merged C0 intervals.

use std::num::NonZeroU64;

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayoutDescriptor,
    Queue, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::debug_log;

use super::types::{CascadeConfig, NUM_CASCADES, OPAQUE_THRESHOLD};

// ---------------------------------------------------------------------------
// WGSL shader source (embedded at compile time)
// ---------------------------------------------------------------------------

const MERGE_SHADER: &str = include_str!("shaders/merge.wgsl");

// ---------------------------------------------------------------------------
// MergeParams (96 bytes, matches WGSL MergeParams exactly)
// ---------------------------------------------------------------------------

/// GPU-safe merge parameters for a single (child, parent) cascade pair.
///
/// Sent as a uniform buffer (96 bytes) to the merge compute shader.
/// Field order and types must match the WGSL `struct MergeParams` exactly.
///
/// Layout: 24 × 4 bytes = 96 bytes.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MergeParams {
    // -- Child cascade (offset 0..31) --
    pub child_grid_spacing: f32,
    pub child_ray_start: f32,
    pub child_ray_length: f32,
    pub child_grid_half_extent: f32,
    pub child_grid_size: u32,
    pub child_ray_count: u32,
    pub _pad_c0: u32,
    pub _pad_c1: u32,
    // -- Parent cascade (offset 32..63) --
    pub parent_grid_spacing: f32,
    pub parent_ray_start: f32,
    pub parent_ray_length: f32,
    pub parent_grid_half_extent: f32,
    pub parent_grid_size: u32,
    pub parent_ray_count: u32,
    pub _pad_p0: u32,
    pub _pad_p1: u32,
    // -- Grid origins (offset 64..95) --
    pub child_origin_x: f32,
    pub child_origin_y: f32,
    pub child_origin_z: f32,
    pub _pad_o0: u32,
    pub parent_origin_x: f32,
    pub parent_origin_y: f32,
    pub parent_origin_z: f32,
    pub opaque_threshold: f32,
}

impl MergeParams {
    /// Byte size of this struct (must be 96).
    pub const SIZE: u64 = 96;

    /// Build merge params from child and parent cascade configs.
    ///
    /// # Panics
    /// Debug-only: asserts that parent has more rays than child.
    pub fn from_cascades(
        child_config: &CascadeConfig,
        parent_config: &CascadeConfig,
        child_origin: glam::Vec3,
        parent_origin: glam::Vec3,
        opaque_threshold: f32,
    ) -> Self {
        debug_assert!(
            parent_config.ray_count >= child_config.ray_count,
            "[MergeParams] Parent cascade C{} must have >= rays than child C{} ({} vs {})",
            parent_config.level, child_config.level,
            parent_config.ray_count, child_config.ray_count
        );

        debug_log!(
            "MergeParams", "from_cascades",
            "C{}({}rays,gs={}) ← C{}({}rays,gs={})",
            child_config.level, child_config.ray_count, child_config.grid_size(),
            parent_config.level, parent_config.ray_count, parent_config.grid_size()
        );

        Self {
            child_grid_spacing: child_config.grid_spacing,
            child_ray_start: child_config.ray_start,
            child_ray_length: child_config.ray_length,
            child_grid_half_extent: child_config.grid_half_extent as f32,
            child_grid_size: child_config.grid_size(),
            child_ray_count: child_config.ray_count,
            _pad_c0: 0,
            _pad_c1: 0,
            parent_grid_spacing: parent_config.grid_spacing,
            parent_ray_start: parent_config.ray_start,
            parent_ray_length: parent_config.ray_length,
            parent_grid_half_extent: parent_config.grid_half_extent as f32,
            parent_grid_size: parent_config.grid_size(),
            parent_ray_count: parent_config.ray_count,
            _pad_p0: 0,
            _pad_p1: 0,
            child_origin_x: child_origin.x,
            child_origin_y: child_origin.y,
            child_origin_z: child_origin.z,
            _pad_o0: 0,
            parent_origin_x: parent_origin.x,
            parent_origin_y: parent_origin.y,
            parent_origin_z: parent_origin.z,
            opaque_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// MergePipeline
// ---------------------------------------------------------------------------

/// Owns the compute pipeline, bind group layout, and uniform buffer for
/// the cascade merge pass.
///
/// One instance per RadianceCascadeSystem. Shared across all cascade pairs —
/// only the bind group and uniform contents change per merge step.
pub struct MergePipeline {
    pipeline: ComputePipeline,
    bgl: BindGroupLayout,
    /// Uniform buffer holding MergeParams (96 bytes).
    uniform_buffer: Buffer,
}

impl MergePipeline {
    /// Create the merge pipeline and associated GPU resources.
    pub fn new(device: &Device) -> Self {
        debug_log!("MergePipeline", "new", "Creating merge pipeline");

        let bgl = Self::create_bgl(device);
        let pipeline = Self::create_pipeline(device, &bgl);
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("rc_merge_uniform"),
            size: MergeParams::SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        debug_log!(
            "MergePipeline", "new",
            "Pipeline created: uniform={}B",
            MergeParams::SIZE
        );

        Self {
            pipeline,
            bgl,
            uniform_buffer,
        }
    }

    // -- Internal builders --------------------------------------------------

    fn create_bgl(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("rc_merge_bgl"),
            entries: &[
                // @binding(0): MergeParams uniform buffer (96 bytes)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZeroU64::new(MergeParams::SIZE).unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1): Parent intervals (storage buffer, read-only)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): Child intervals (storage buffer, read_write)
                BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("rc_merge_shader"),
            source: ShaderSource::Wgsl(MERGE_SHADER.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rc_merge_layout"),
            bind_group_layouts: &[Option::from(bgl)],
            immediate_size: 0,
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("rc_merge_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("merge_main"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    // -- Public API ---------------------------------------------------------

    /// Create a bind group for a specific (parent, child) cascade pair.
    ///
    /// - `parent_output`: storage buffer with parent's (already-merged) intervals.
    /// - `child_output`: storage buffer with child's intervals (will be written in-place).
    pub fn create_bind_group(
        &self,
        device: &Device,
        parent_output: &Buffer,
        child_output: &Buffer,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("rc_merge_bg"),
            layout: &self.bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: parent_output.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: child_output.as_entire_binding(),
                },
            ],
        })
    }

    /// Encode a single merge compute dispatch for one (child, parent) pair.
    ///
    /// 1. Writes `params` to the uniform buffer.
    /// 2. Begins a compute pass and dispatches workgroups.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        bind_group: &BindGroup,
        params: &MergeParams,
    ) {
        // Upload params to uniform buffer
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[*params]),
        );

        // Compute total invocations and workgroup count
        let c_gs: u64 = params.child_grid_size as u64;
        let c_rc: u64 = params.child_ray_count as u64;
        let total_invocations = c_gs * c_gs * c_gs * c_rc;

        let workgroup_size: u64 = 64;
        let dispatch_x = (total_invocations + workgroup_size - 1) / workgroup_size;

        debug_log!(
            "MergePipeline", "dispatch",
            "C{}←C{}: grid={}³×{}rays={}, workgroups={}",
            params.child_grid_size, // child level encoded indirectly via grid_size
            0u32, // placeholder; caller knows the cascade indices
            params.child_grid_size,
            params.child_ray_count,
            total_invocations,
            dispatch_x
        );

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("rc_merge_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x as u32, 1, 1);

        // pass is dropped here, ending the compute pass
    }

    /// Reference to the bind group layout (for pipeline compatibility checks).
    #[inline]
    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bgl
    }
}

// ---------------------------------------------------------------------------
// Cascade chain dispatch
// ---------------------------------------------------------------------------

/// Dispatch all merge passes in cascade order: C3←C4, C2←C3, C1←C2, C0←C1.
///
/// Each pass reads from the parent's output buffer (which may have been
/// modified by a previous merge pass) and writes in-place to the child's
/// output buffer.
///
/// After this call, `output_buffers[0]` contains the final merged C0
/// intervals covering the full radial range with the finest spatial resolution.
///
/// # Arguments
/// * `device`  — wgpu device (for creating bind groups).
/// * `pipeline` — the merge pipeline (created once, reused).
/// * `encoder` — command encoder (passes are appended sequentially).
/// * `queue`   — command queue (for uniform buffer uploads).
/// * `configs` — cascade configs for all 5 levels.
/// * `grid_origins` — world-space grid origins for all 5 levels.
/// * `output_buffers` — per-cascade output buffers from the ray march step.
/// * `opaque_threshold` — tau threshold for fully opaque termination.
pub fn dispatch_all_merges(
    device: &Device,
    pipeline: &MergePipeline,
    encoder: &mut CommandEncoder,
    queue: &Queue,
    configs: &[CascadeConfig; NUM_CASCADES],
    grid_origins: &[glam::Vec3; NUM_CASCADES],
    output_buffers: &[Buffer; NUM_CASCADES],
    opaque_threshold: f32,
) {
    debug_log!(
        "dispatch_all_merges", "dispatch_all_merges",
        "Starting {} merge passes (C4→C0)",
        NUM_CASCADES - 1
    );

    // Process from coarsest-1 down to 0:
    //   i=3: merge C3 ← C4
    //   i=2: merge C2 ← C3 (now merged)
    //   i=1: merge C1 ← C2 (now merged)
    //   i=0: merge C0 ← C1 (now merged)
    for i in (0..NUM_CASCADES - 1).rev() {
        let child_idx = i;
        let parent_idx = i + 1;

        let params = MergeParams::from_cascades(
            &configs[child_idx],
            &configs[parent_idx],
            grid_origins[child_idx],
            grid_origins[parent_idx],
            opaque_threshold,
        );

        let bg = pipeline.create_bind_group(
            device,
            &output_buffers[parent_idx], // parent (read)
            &output_buffers[child_idx],  // child (read_write, in-place)
        );

        pipeline.dispatch(encoder, queue, &bg, &params);

        debug_log!(
            "dispatch_all_merges", "dispatch_all_merges",
            "Dispatched merge: C{} ← C{}", child_idx, parent_idx
        );
    }

    debug_log!(
        "dispatch_all_merges", "dispatch_all_merges",
        "All merge passes complete. Final result in output_buffers[0] (C0)"
    );
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use super::*;

    #[test]
    fn test_merge_params_size() {
        assert_eq!(
            std::mem::size_of::<MergeParams>(),
            96,
            "MergeParams must be exactly 96 bytes"
        );
        assert_eq!(MergeParams::SIZE, 96);
    }

    #[test]
    fn test_merge_params_layout() {
        let params = MergeParams {
            child_grid_spacing: 4.0,
            child_ray_start: 0.0,
            child_ray_length: 6.0,
            child_grid_half_extent: 16.0,
            child_grid_size: 33,
            child_ray_count: 6,
            _pad_c0: 0,
            _pad_c1: 0,
            parent_grid_spacing: 8.0,
            parent_ray_start: 6.0,
            parent_ray_length: 12.0,
            parent_grid_half_extent: 8.0,
            parent_grid_size: 17,
            parent_ray_count: 12,
            _pad_p0: 0,
            _pad_p1: 0,
            child_origin_x: -64.0,
            child_origin_y: -64.0,
            child_origin_z: -64.0,
            _pad_o0: 0,
            parent_origin_x: -64.0,
            parent_origin_y: -64.0,
            parent_origin_z: -64.0,
            opaque_threshold: 500.0,
        };

        assert!((params.child_grid_spacing - 4.0).abs() < 1e-6);
        assert_eq!(params.child_grid_size, 33);
        assert_eq!(params.child_ray_count, 6);
        assert_eq!(params.parent_ray_count, 12);
        assert_eq!(params.parent_grid_size, 17);
        assert!((params.opaque_threshold - 500.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_cascades_c0_c1() {
        let c0 = CascadeConfig {
            level: 0,
            grid_spacing: 4.0,
            ray_count: 6,
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 16,
        };
        let c1 = CascadeConfig {
            level: 1,
            grid_spacing: 8.0,
            ray_count: 12,
            ray_length: 12.0,
            ray_start: 6.0,
            grid_half_extent: 8,
        };

        let child_origin = glam::Vec3::new(-64.0, -64.0, -64.0);
        let parent_origin = glam::Vec3::new(-64.0, -64.0, -64.0);

        let params = MergeParams::from_cascades(&c0, &c1, child_origin, parent_origin, 500.0);

        assert!((params.child_grid_spacing - 4.0).abs() < 1e-6);
        assert_eq!(params.child_grid_size, 33);
        assert_eq!(params.child_ray_count, 6);
        assert!((params.parent_grid_spacing - 8.0).abs() < 1e-6);
        assert_eq!(params.parent_grid_size, 17);
        assert_eq!(params.parent_ray_count, 12);
        assert!((params.child_ray_start - 0.0).abs() < 1e-6);
        assert!((params.parent_ray_start - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_cascades_c3_c4() {
        let c3 = CascadeConfig {
            level: 3,
            grid_spacing: 32.0,
            ray_count: 48,
            ray_length: 48.0,
            ray_start: 30.0,
            grid_half_extent: 2,
        };
        let c4 = CascadeConfig {
            level: 4,
            grid_spacing: 64.0,
            ray_count: 96,
            ray_length: 96.0,
            ray_start: 78.0,
            grid_half_extent: 1,
        };

        let params = MergeParams::from_cascades(
            &c3, &c4,
            glam::Vec3::ZERO, glam::Vec3::ZERO,
            500.0,
        );

        assert!((params.child_grid_spacing - 32.0).abs() < 1e-6);
        assert_eq!(params.child_grid_size, 5);
        assert_eq!(params.child_ray_count, 48);
        assert!((params.parent_grid_spacing - 64.0).abs() < 1e-6);
        assert_eq!(params.parent_grid_size, 3);
        assert_eq!(params.parent_ray_count, 96);
    }

    #[test]
    fn test_pod_zeroable() {
        let zero = MergeParams::zeroed();
        assert_eq!(zero.child_grid_spacing, 0.0);
        assert_eq!(zero.child_grid_size, 0);
        assert_eq!(zero.parent_ray_count, 0);

        // Verify it's Copy
        let _copy = zero;
    }

    #[test]
    fn test_from_cascades_field_positions() {
        // Verify that fields are at the correct byte offsets by checking
        // a known property: byte offset of parent_grid_spacing must be 32.
        let params = MergeParams {
            child_grid_spacing: 1.0,  // offset 0
            child_ray_start: 2.0,     // offset 4
            child_ray_length: 3.0,    // offset 8
            child_grid_half_extent: 4.0, // offset 12
            child_grid_size: 5,       // offset 16
            child_ray_count: 6,       // offset 20
            _pad_c0: 7,               // offset 24
            _pad_c1: 8,               // offset 28
            parent_grid_spacing: 99.0, // offset 32 — KEY CHECK
            parent_ray_start: 10.0,   // offset 36
            parent_ray_length: 11.0,  // offset 40
            parent_grid_half_extent: 12.0, // offset 44
            parent_grid_size: 13,     // offset 48
            parent_ray_count: 14,     // offset 52
            _pad_p0: 15,              // offset 56
            _pad_p1: 16,              // offset 60
            child_origin_x: 17.0,     // offset 64
            child_origin_y: 18.0,     // offset 68
            child_origin_z: 19.0,     // offset 72
            _pad_o0: 20,              // offset 76
            parent_origin_x: 21.0,    // offset 80
            parent_origin_y: 22.0,    // offset 84
            parent_origin_z: 23.0,    // offset 88
            opaque_threshold: 24.0,   // offset 92
        };

        // Verify by casting to byte slice and reading the f32 at offset 32
        let bytes: &[u8] = bytemuck::bytes_of(&params);
        let at_32 = f32::from_le_bytes(bytes[32..36].try_into().unwrap());
        assert!(
            (at_32 - 99.0).abs() < 1e-6,
            "Expected 99.0 at offset 32, got {}",
            at_32
        );
    }
}