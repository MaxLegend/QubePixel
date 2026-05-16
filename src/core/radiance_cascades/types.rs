// =============================================================================
// QubePixel — Radiance Cascades: Core Types
// =============================================================================
//
// Data structures for the cascade hierarchy, radiance intervals, probe grids,
// and the top-level RadianceCascadeSystem.
//
// GPU layout conventions:
//   - All GPU-visible structs are #[repr(C)] + Pod + Zeroable (bytemuck).
//   - Vec fields are CPU-only; they are NOT sent to the GPU directly.
//   - Probe data is packed into 3D textures (Rgba32Float) for GPU consumption.

use glam::{IVec3, Vec3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of cascade levels in the system.
/// C0 = finest (near), C(N-1) = coarsest (far).
pub const NUM_CASCADES: usize = 5;

/// Base grid spacing in world units (voxels) for cascade C0.
/// Each subsequent cascade doubles this: C_i = BASE_SPACING * 2^i.
pub const BASE_SPACING: f32 = 4.0;

/// Cube-face subdivision levels per cascade. Ray count = 6 * subdiv².
///   C0: subdiv=2 →  24 rays (4 per face; fixes directional aliasing in merge)
///   C1: subdiv=2 →  24 rays
///   C2: subdiv=3 →  54 rays
///   C3: subdiv=4 →  96 rays
///   C4: subdiv=5 → 150 rays
pub const CUBE_FACE_SUBDIVS: [u32; 5] = [2, 2, 3, 4, 5];

/// Base grid half-extent for cascade C0 (used in tests and as reference).
pub const BASE_GRID_HALF_EXTENT: u32 = 16;

/// Per-cascade grid half-extents. Coverage = (2×extent+1) × spacing per axis.
///   C0:  13×1  =  13 blocks   C1:  9×2  =  18 blocks   C2:  5×4  =  20 blocks
///   C3:   5×8  =  40 blocks   C4:  3×16 =  48 blocks
/// C0: subdiv=2 → 24 rays, extent=6 → 13³×24 ≈ 52k/frame
/// C1: subdiv=2 → 24 rays, extent=4 →  9³×24 ≈ 17k/frame
/// C0+C1 marched every frame; C2-C4 every 4th frame.
pub const CASCADE_GRID_HALF_EXTENTS: [u32; 5] = [6, 4, 2, 2, 1];

/// Maximum ray marching steps per ray (prevents infinite loops).
pub const MAX_RAY_STEPS: u32 = 128;

/// Opacity threshold above which a block is considered fully opaque.
/// Used in the compute shader to terminate rays early.
pub const OPAQUE_THRESHOLD: f32 = 500.0;

/// Single-block ray step (for DDA max distance = cascade spacing).
/// Rays in cascade C_i travel at most this many voxels.
pub const RAY_MAX_DIST_FACTOR: f32 = 8.0;

// ---------------------------------------------------------------------------
// CascadeConfig
// ---------------------------------------------------------------------------

/// Parameters for a single cascade level.
///
/// GPU-safe: #[repr(C)], 32 bytes.
/// Sent to compute shaders via uniform buffer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct CascadeConfig {
    /// Cascade index: 0 = finest, N-1 = coarsest.
    pub level: u32,
    /// Grid spacing in world units (voxels) between adjacent probes.
    pub grid_spacing: f32,
    /// Number of rays per probe (angular resolution).
    pub ray_count: u32,
    /// Length of each ray in world units (voxels).
    pub ray_length: f32,
    /// Distance from probe center where rays start tracing.
    /// Ensures cascades tile the radial range without overlap.
    pub ray_start: f32,
    /// Grid half-extent: number of probes from center to edge per axis.
    /// Total grid size = 2 * grid_half_extent + 1.
    pub grid_half_extent: u32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            level: 0,
            grid_spacing: 0.0,
            ray_count: 0,
            ray_length: 0.0,
            ray_start: 0.0,
            grid_half_extent: 0,
        }
    }
}

impl CascadeConfig {
    /// Total grid size per axis: 2 * half_extent + 1.
    #[inline]
    pub fn grid_size(&self) -> u32 {
        self.grid_half_extent * 2 + 1
    }

    /// Total number of probes in this cascade's grid.
    #[inline]
    pub fn total_probes(&self) -> usize {
        let s = self.grid_size() as usize;
        s * s * s
    }
}

/// Build the default cascade configuration array.
///
/// Cascade layout (Penumbra Criterion):
///   - Spatial resolution halves per level (spacing doubles).
///   - Angular resolution doubles per level (ray count * BRANCHING_FACTOR).
///   - Radial coverage: rays tile [ray_start, ray_start + ray_length].
///
/// Returns an array of length NUM_CASCADES.
pub fn default_cascade_configs() -> [CascadeConfig; NUM_CASCADES] {
    let mut configs = [CascadeConfig::default(); NUM_CASCADES];

    let mut cumulative_start = 0.0f32;

    for i in 0..NUM_CASCADES {
        let spacing = BASE_SPACING * (1u32 << i) as f32;
        let ray_len = spacing * RAY_MAX_DIST_FACTOR;
        let subdiv = CUBE_FACE_SUBDIVS[i];
        let rays = 6 * subdiv * subdiv; // exact cube-face ray count
        let extent = CASCADE_GRID_HALF_EXTENTS[i];

        configs[i] = CascadeConfig {
            level: i as u32,
            grid_spacing: spacing,
            ray_count: rays,
            ray_length: ray_len,
            ray_start: cumulative_start,
            grid_half_extent: extent,
        };

        cumulative_start += ray_len;
    }

    debug_log!(
        "CascadeConfig", "default_cascade_configs",
        "Built {} cascade levels, total radial coverage: {:.1} voxels",
        NUM_CASCADES, cumulative_start
    );

    configs
}

// ---------------------------------------------------------------------------
// RadianceInterval
// ---------------------------------------------------------------------------

/// A single radiance interval — the fundamental data unit of the cascade system.
///
/// Represents the light transport along one ray segment between two points.
/// Stored per-probe per-ray; merged across cascade levels.
///
/// GPU layout: 3 x vec4<f32> = 48 bytes per interval.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RadianceInterval {
    /// In-scattered radiance accumulated along the ray.
    /// .rgb = RGB radiance (W/sr), .a = unused (padding).
    pub radiance_in: [f32; 4],

    /// Out-scattered radiance at the far end of the ray + optical thickness.
    /// .rgb = emission at the hit point, .w = total optical thickness (tau).
    /// tau = 0: fully transparent. tau = infinity: fully opaque.
    pub radiance_out: [f32; 4],

    /// Ray direction (normalized) and actual distance traveled.
    /// .xyz = direction, .w = distance (0 if ray was culled).
    pub direction_length: [f32; 4],
}

impl Default for RadianceInterval {
    fn default() -> Self {
        Self {
            radiance_in: [0.0; 4],
            radiance_out: [0.0; 4],
            direction_length: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

impl RadianceInterval {
    /// Create a zero-initialized interval with a given direction.
    pub fn from_direction(dir: Vec3) -> Self {
        Self {
            radiance_in: [0.0; 4],
            radiance_out: [0.0; 4],
            direction_length: [dir.x, dir.y, dir.z, 0.0],
        }
    }

    /// Optical thickness (tau) of this interval.
    #[inline]
    pub fn tau(&self) -> f32 {
        self.radiance_out[3]
    }

    /// In-scattered radiance as Vec3.
    #[inline]
    pub fn radiance_in_rgb(&self) -> Vec3 {
        Vec3::new(self.radiance_in[0], self.radiance_in[1], self.radiance_in[2])
    }

    /// Emission at far end as Vec3.
    #[inline]
    pub fn radiance_out_rgb(&self) -> Vec3 {
        Vec3::new(
            self.radiance_out[0],
            self.radiance_out[1],
            self.radiance_out[2],
        )
    }

    /// Ray direction as Vec3.
    #[inline]
    pub fn direction(&self) -> Vec3 {
        Vec3::new(
            self.direction_length[0],
            self.direction_length[1],
            self.direction_length[2],
        )
    }

    /// Distance the ray actually traveled before termination.
    #[inline]
    pub fn distance(&self) -> f32 {
        self.direction_length[3]
    }

    /// Byte size of this struct for GPU alignment.
    pub const SIZE: u64 = 48; // 3 x 16 bytes
}

/// Merge two adjacent radiance intervals using transmittance-weighted composition.
///
/// Given interval A (near segment) and interval B (far segment):
///   M.radiance_in  = A.radiance_in + exp(-A.tau) * B.radiance_in
///   M.radiance_out = B.radiance_out
///   M.tau          = A.tau + B.tau
///   M.direction    = A.direction (keep the original ray direction)
///   M.distance     = A.distance + B.distance
#[inline]
pub fn merge_intervals(a: &RadianceInterval, b: &RadianceInterval) -> RadianceInterval {
    let transmittance = (-a.radiance_out[3]).clamp(-80.0, 0.0).exp();
    // Clamp exp argument to avoid NaN with very large negative tau

    RadianceInterval {
        radiance_in: [
            a.radiance_in[0] + transmittance * b.radiance_in[0],
            a.radiance_in[1] + transmittance * b.radiance_in[1],
            a.radiance_in[2] + transmittance * b.radiance_in[2],
            0.0,
        ],
        radiance_out: b.radiance_out,
        direction_length: [
            a.direction_length[0],
            a.direction_length[1],
            a.direction_length[2],
            a.direction_length[3] + b.direction_length[3],
        ],
    }
}

// ---------------------------------------------------------------------------
// ProbeData
// ---------------------------------------------------------------------------

/// CPU-side representation of a single probe in the cascade grid.
///
/// Holds the probe world position, grid coordinates, and all radiance
/// intervals for its rays. This is the CPU mirror of the GPU texture data.
#[derive(Debug, Clone)]
pub struct ProbeData {
    /// World-space position of this probe center.
    pub position: Vec3,
    /// Grid coordinates within this cascade level.
    pub grid_coord: IVec3,
    /// Radiance intervals: one per ray direction.
    /// Length = cascade_config.ray_count.
    pub rays: Vec<RadianceInterval>,
}

impl ProbeData {
    /// Create an empty probe (all intervals zeroed).
    pub fn new(position: Vec3, grid_coord: IVec3, ray_count: usize) -> Self {
        let rays = vec![RadianceInterval::default(); ray_count];
        Self {
            position,
            grid_coord,
            rays,
        }
    }
}

// ---------------------------------------------------------------------------
// ProbeGrid
// ---------------------------------------------------------------------------

/// Container for all probes in a single cascade level.
///
/// On the CPU: sparse HashMap of probes within loaded chunks.
/// On the GPU: packed into a 3D texture (Rgba32Float) for compute shader access.
pub struct ProbeGrid {
    /// Cascade configuration for this grid.
    pub config: CascadeConfig,
    /// Sparse map: grid_coord -> ProbeData.
    /// Only contains probes whose center voxel is within loaded chunks.
    pub probes: HashMap<(i32, i32, i32), ProbeData>,
    /// GPU texture holding packed probe ray data.
    /// Created lazily on first GPU access. Used in Step 4.
    #[allow(dead_code)]
    gpu_texture: Option<wgpu::Texture>,
    /// GPU texture view for shader binding.
    gpu_view: Option<wgpu::TextureView>,
    /// Grid origin in world space (recomputed each frame to center on camera).
    grid_origin: Vec3,
    /// Whether GPU resources are dirty and need re-upload.
    gpu_dirty: bool,
}

impl ProbeGrid {
    /// Create a new empty probe grid for the given cascade config.
    pub fn new(config: CascadeConfig) -> Self {
        debug_log!(
            "ProbeGrid", "new",
            "Creating grid for cascade {}: spacing={:.1}, rays={}, extent={}",
            config.level, config.grid_spacing, config.ray_count, config.grid_half_extent
        );
        Self {
            config,
            probes: HashMap::new(),
            gpu_texture: None,
            gpu_view: None,
            grid_origin: Vec3::ZERO,
            gpu_dirty: false,
        }
    }

    /// Recenter the grid around a new camera position.
    /// Clears all existing probes and marks GPU resources as dirty.
    pub fn recenter(&mut self, camera_pos: Vec3) {
        let s = self.config.grid_spacing * self.config.grid_half_extent as f32;
        self.grid_origin = camera_pos - Vec3::new(s, s, s);
        self.probes.clear();
        self.gpu_dirty = true;
    }

    /// World-space position for a probe at grid coordinates (ix, iy, iz).
    #[inline]
    pub fn probe_world_pos(&self, ix: i32, iy: i32, iz: i32) -> Vec3 {
        self.grid_origin
            + Vec3::new(
                ix as f32 * self.config.grid_spacing,
                iy as f32 * self.config.grid_spacing,
                iz as f32 * self.config.grid_spacing,
            )
    }

    /// Number of texel layers needed in the GPU texture.
    /// Each interval = 3 x vec4. 4 intervals packed per texel.
    /// layers = ceil(ray_count / 4) * 3.
    pub fn texture_layers(&self) -> u32 {
        let packed = (self.config.ray_count + 3) / 4;
        packed * 3
    }

    /// Grid size per axis (total probes).
    #[inline]
    pub fn grid_size(&self) -> u32 {
        self.config.grid_size()
    }

    /// Grid origin in world space.
    #[inline]
    pub fn grid_origin(&self) -> Vec3 {
        self.grid_origin
    }

    /// Whether the GPU texture needs re-upload.
    #[inline]
    pub fn is_gpu_dirty(&self) -> bool {
        self.gpu_dirty
    }

    /// Mark GPU resources as clean (after successful upload).
    pub fn mark_gpu_clean(&mut self) {
        self.gpu_dirty = false;
    }
}

// ---------------------------------------------------------------------------
// BlockProperties (for GPU LUT)
// ---------------------------------------------------------------------------

/// Per-block-type optical and emission properties.
///
/// Packed into a 1D lookup texture (Rgba32Float) indexed by block ID.
/// .rgba = (opacity, emission_r, emission_g, emission_b).
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockProperties {
    /// Optical density per voxel.
    /// 0.0 = fully transparent (air), >500 = effectively opaque.
    pub opacity: f32,
    /// Pre-multiplied emission: emission_color * emission_intensity.
    pub emission_r: f32,
    pub emission_g: f32,
    pub emission_b: f32,
}

impl Default for BlockProperties {
    fn default() -> Self {
        Self {
            opacity: 0.0,
            emission_r: 0.0,
            emission_g: 0.0,
            emission_b: 0.0,
        }
    }
}

impl BlockProperties {
    /// Byte size for GPU alignment.
    pub const SIZE: u64 = 16; // 1 x vec4

    /// Create a transparent (air) block property.
    pub fn air() -> Self {
        Self::default()
    }

    /// Create a fully opaque block property.
    pub fn opaque() -> Self {
        Self {
            opacity: 1000.0,
            ..Self::default()
        }
    }

    /// Create an opaque emissive block property.
    pub fn emissive(r: f32, g: f32, b: f32, intensity: f32) -> Self {
        Self {
            opacity: 1000.0,
            emission_r: r * intensity,
            emission_g: g * intensity,
            emission_b: b * intensity,
        }
    }

    /// Emission color as Vec3.
    pub fn emission_rgb(&self) -> Vec3 {
        Vec3::new(self.emission_r, self.emission_g, self.emission_b)
    }
}

// ---------------------------------------------------------------------------
// RadianceCascadeSystem — top-level manager
// ---------------------------------------------------------------------------

/// Top-level system that owns all cascade levels and manages the GPU pipeline.
///
/// Lifecycle:
///   1. new() — create with default configs, no GPU resources yet.
///   2. init_gpu() — create textures and pipelines (called once).
///   3. Each frame: update_voxel_texture() → dispatch_cascades() → read probes in fs.
pub struct RadianceCascadeSystem {
    /// Cascade configuration for each level.
    pub configs: [CascadeConfig; NUM_CASCADES],
    /// Probe grids for each cascade level.
    pub grids: [ProbeGrid; NUM_CASCADES],
    /// 3D voxel texture (R16Uint): block IDs for ray marching.
    pub voxel_texture: Option<wgpu::Texture>,
    /// View into the voxel texture.
    pub voxel_view: Option<wgpu::TextureView>,
    /// 1D block property LUT texture (Rgba32Float).
    pub block_lut_texture: Option<wgpu::Texture>,
    /// View into the block LUT texture.
    pub block_lut_view: Option<wgpu::TextureView>,
    /// Compute pipeline for ray marching.
    pub ray_march_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for ray marching.
    pub ray_march_bgl: Option<wgpu::BindGroupLayout>,
    /// Compute pipeline for merging.
    pub merge_pipeline: Option<wgpu::ComputePipeline>,
    /// Bind group layout for merging.
    pub merge_bgl: Option<wgpu::BindGroupLayout>,
    /// Performance: last dispatch time in microseconds.
    pub last_dispatch_time_us: u64,
    /// Whether the system has been initialized with GPU resources.
    gpu_initialized: bool,
}

impl RadianceCascadeSystem {
    /// Create a new Radiance Cascade system with default configuration.
    ///
    /// GPU resources are created lazily via init_gpu().
    pub fn new() -> Self {
        debug_log!(
            "RadianceCascadeSystem", "new",
            "Creating Radiance Cascade system with {} levels", NUM_CASCADES
        );

        let configs = default_cascade_configs();
        let grids = configs.map(ProbeGrid::new);

        Self {
            configs,
            grids,
            voxel_texture: None,
            voxel_view: None,
            block_lut_texture: None,
            block_lut_view: None,
            ray_march_pipeline: None,
            ray_march_bgl: None,
            merge_pipeline: None,
            merge_bgl: None,
            last_dispatch_time_us: 0,
            gpu_initialized: false,
        }
    }

    /// Initialize GPU resources (textures, pipelines).
    /// Must be called once with a valid device before dispatch.
    pub fn init_gpu(&mut self, _device: &wgpu::Device) {
        if self.gpu_initialized {
            return;
        }
        debug_log!(
            "RadianceCascadeSystem", "init_gpu",
            "Initializing GPU resources for Radiance Cascades"
        );
        // GPU resource creation will be implemented in Step 4.
        // Here we just mark the system as initialized and log.
        self.gpu_initialized = true;
    }

    /// Recenter all cascade grids around the camera position.
    /// Called each frame before dispatch.
    pub fn recenter_all(&mut self, camera_pos: Vec3) {
        for grid in &mut self.grids {
            grid.recenter(camera_pos);
        }
    }

    /// Total estimated GPU memory usage in bytes.
    pub fn estimated_gpu_memory(&self) -> u64 {
        let probe_mem: u64 = self
            .grids
            .iter()
            .map(|g| {
                let s = g.grid_size() as u64;
                let layers = g.texture_layers() as u64;
                s * s * s * layers * 16 // Rgba32Float = 16 bytes/texel
            })
            .sum();

        let voxel_mem = 128u64 * 128 * 128 * 2; // R16Uint = 2 bytes/texel
        let lut_mem = 256u64 * 16; // 256 texels * Rgba32Float

        probe_mem + voxel_mem + lut_mem
    }

    /// Whether GPU resources have been initialized.
    pub fn is_gpu_initialized(&self) -> bool {
        self.gpu_initialized
    }

    /// View into the merged C0 probe texture (for fragment shader sampling).
    /// Returns None if not yet initialized.
    pub fn merged_probe_view(&self) -> Option<&wgpu::TextureView> {
        self.grids[0].gpu_view.as_ref()
    }
}

impl Default for RadianceCascadeSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cascade_configs() {
        let configs = default_cascade_configs();

        // Verify cascade count
        assert_eq!(configs.len(), NUM_CASCADES);

        // C0: spacing=1.0, subdiv=2 → 24 rays, extent=CASCADE_GRID_HALF_EXTENTS[0]
        assert_eq!(configs[0].level, 0);
        assert!((configs[0].grid_spacing - 1.0).abs() < 1e-6);
        assert_eq!(configs[0].ray_count, 24);  // 6*2*2
        assert_eq!(configs[0].grid_half_extent, CASCADE_GRID_HALF_EXTENTS[0]);

        // C1: spacing=2.0, subdiv=2 → 24 rays
        assert_eq!(configs[1].level, 1);
        assert!((configs[1].grid_spacing - 2.0).abs() < 1e-6);
        assert_eq!(configs[1].ray_count, 24);  // 6*2*2

        // C2: spacing=4.0, subdiv=3 → 54 rays
        assert_eq!(configs[2].level, 2);
        assert!((configs[2].grid_spacing - 4.0).abs() < 1e-6);
        assert_eq!(configs[2].ray_count, 54);  // 6*3*3

        // C3: spacing=8.0, subdiv=4 → 96 rays
        assert_eq!(configs[3].level, 3);
        assert!((configs[3].grid_spacing - 8.0).abs() < 1e-6);
        assert_eq!(configs[3].ray_count, 96);  // 6*4*4

        // C4: spacing=16.0, subdiv=5 → 150 rays
        assert_eq!(configs[4].level, 4);
        assert!((configs[4].grid_spacing - 16.0).abs() < 1e-6);
        assert_eq!(configs[4].ray_count, 150); // 6*5*5

        // Ray start increases with each cascade
        for i in 1..NUM_CASCADES {
            assert!(
                configs[i].ray_start > configs[i - 1].ray_start,
                "C{} ray_start ({}) should be > C{} ({})",
                i, configs[i].ray_start, i - 1, configs[i - 1].ray_start
            );
        }
    }

    #[test]
    fn test_grid_size() {
        let configs = default_cascade_configs();
        // Each cascade uses its per-cascade extent from CASCADE_GRID_HALF_EXTENTS
        for i in 0..NUM_CASCADES {
            let expected = CASCADE_GRID_HALF_EXTENTS[i] * 2 + 1;
            assert_eq!(configs[i].grid_size(), expected);
        }
    }

    #[test]
    fn test_radiance_interval_size() {
        assert_eq!(std::mem::size_of::<RadianceInterval>(), 48);
        assert_eq!(RadianceInterval::SIZE as usize, 48);
    }

    #[test]
    fn test_block_properties_size() {
        assert_eq!(std::mem::size_of::<BlockProperties>(), 16);
        assert_eq!(BlockProperties::SIZE as usize, 16);
    }

    #[test]
    fn test_merge_intervals_transparent() {
        let a = RadianceInterval {
            radiance_in: [1.0, 0.5, 0.2, 0.0],
            radiance_out: [0.0, 0.0, 0.0, 0.0], // tau = 0
            direction_length: [0.0, 1.0, 0.0, 2.0],
        };
        let b = RadianceInterval {
            radiance_in: [0.3, 0.1, 0.4, 0.0],
            radiance_out: [0.5, 0.0, 0.0, 1.0], // tau = 1
            direction_length: [0.0, 1.0, 0.0, 3.0],
        };

        let m = merge_intervals(&a, &b);

        // exp(-0) = 1.0, so M.radiance_in = A + B
        assert!((m.radiance_in[0] - 1.3).abs() < 1e-6);
        assert!((m.radiance_in[1] - 0.6).abs() < 1e-6);
        assert!((m.radiance_in[2] - 0.6).abs() < 1e-6);

        // radiance_out comes from B
        assert!((m.radiance_out[0] - 0.5).abs() < 1e-6);
        // tau = 0 + 1 = 1
        assert!((m.radiance_out[3] - 1.0).abs() < 1e-6);

        // distance = 2 + 3 = 5
        assert!((m.direction_length[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_merge_intervals_opaque() {
        let a = RadianceInterval {
            radiance_in: [0.5, 0.5, 0.5, 0.0],
            radiance_out: [1.0, 0.0, 0.0, 100.0], // tau = 100 (opaque)
            direction_length: [1.0, 0.0, 0.0, 1.0],
        };
        let b = RadianceInterval {
            radiance_in: [10.0, 10.0, 10.0, 0.0], // Should be fully attenuated
            radiance_out: [0.0, 0.0, 0.0, 0.0],
            direction_length: [1.0, 0.0, 0.0, 1.0],
        };

        let m = merge_intervals(&a, &b);

        // exp(-100) ≈ 0, so B's contribution is negligible
        assert!((m.radiance_in[0] - 0.5).abs() < 0.01);
        assert!((m.radiance_in[1] - 0.5).abs() < 0.01);
        assert!((m.radiance_in[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_probe_grid_recenter() {
        let cfg = CascadeConfig {
            level: 0,
            grid_spacing: 4.0,
            ray_count: 6,
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 2,
        };
        let mut grid = ProbeGrid::new(cfg);

        // Recenter at (100, 50, 30)
        grid.recenter(Vec3::new(100.0, 50.0, 30.0));

        // Grid origin should be at camera - spacing * half_extent = (92, 42, 22)
        let origin = grid.grid_origin();
        assert!((origin.x - 92.0).abs() < 1e-6);
        assert!((origin.y - 42.0).abs() < 1e-6);
        assert!((origin.z - 22.0).abs() < 1e-6);

        // Probe at grid (0,0,0) should be at origin
        let p = grid.probe_world_pos(0, 0, 0);
        assert!((p.x - 92.0).abs() < 1e-6);
        assert!((p.y - 42.0).abs() < 1e-6);
        assert!((p.z - 22.0).abs() < 1e-6);

        // Probe at grid (4,4,4) should be at origin + 4*spacing = (108, 58, 38)
        let p = grid.probe_world_pos(4, 4, 4);
        assert!((p.x - 108.0).abs() < 1e-6);
        assert!((p.y - 58.0).abs() < 1e-6);
        assert!((p.z - 38.0).abs() < 1e-6);
    }

    #[test]
    fn test_texture_layers() {
        let cfg = CascadeConfig {
            level: 0,
            grid_spacing: 4.0,
            ray_count: 6,    // ceil(6/4) * 3 = 2 * 3 = 6
            ray_length: 6.0,
            ray_start: 0.0,
            grid_half_extent: 2,
        };
        let grid = ProbeGrid::new(cfg);
        assert_eq!(grid.texture_layers(), 6);

        // 12 rays: ceil(12/4) * 3 = 3 * 3 = 9
        let cfg2 = CascadeConfig {
            level: 1,
            grid_spacing: 8.0,
            ray_count: 12,
            ray_length: 12.0,
            ray_start: 6.0,
            grid_half_extent: 1,
        };
        let grid2 = ProbeGrid::new(cfg2);
        assert_eq!(grid2.texture_layers(), 9);
    }

    #[test]
    fn test_radiance_cascade_system_new() {
        let system = RadianceCascadeSystem::new();
        assert!(!system.is_gpu_initialized());
        assert_eq!(system.configs.len(), NUM_CASCADES);
        assert_eq!(system.grids.len(), NUM_CASCADES);
        assert_eq!(system.last_dispatch_time_us, 0);
    }

    #[test]
    fn test_estimated_gpu_memory() {
        let system = RadianceCascadeSystem::new();
        let mem = system.estimated_gpu_memory();
        // Should be at least a few MB
        assert!(mem > 1_000_000, "Expected > 1MB, got {} bytes", mem);
        debug_log!(
            "RadianceCascadeSystem", "test_estimated_gpu_memory",
            "Estimated GPU memory: {:.2} MB",
            mem as f64 / (1024.0 * 1024.0)
        );
    }
}
