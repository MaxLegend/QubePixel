// =============================================================================
// QubePixel — VoxelVolume: CPU-side voxel snapshot + GPU texture packing
// =============================================================================

/// Side length of the cubic voxel volume (in blocks).
pub const VOLUME_SIZE: u32 = 128;

/// Maximum light propagation range (in blocks).  Must match the inject shader
/// multiplier so that `alpha * INJECT_MULTIPLIER` produces correct brightness.
/// Blocks with `light_range = MAX_LIGHT_RANGE` get alpha = 1.0 (full injection).
pub const MAX_LIGHT_RANGE: f32 = 64.0;

/// Total number of voxels in the volume.
pub const VOLUME_TOTAL: usize = (VOLUME_SIZE * VOLUME_SIZE * VOLUME_SIZE) as usize;

/// Flat index into the volume — matches GPU `write_texture` layout exactly.
///
/// GPU 3D texture convention: offset = z * SIZE*SIZE + y * SIZE + x
///   - x varies FASTEST  (innermost, = bytes_per_row direction)
///   - y varies middle   (= rows_per_image direction)
///   - z varies SLOWEST  (outermost, = depth direction)
///
/// This MUST match `wgpu::TexelCopyBufferLayout { bytes_per_row: 4*SIZE, rows_per_image: SIZE }`.
#[inline(always)]
pub fn vol_idx(x: u32, y: u32, z: u32) -> usize {
    (z as usize) * (VOLUME_SIZE as usize) * (VOLUME_SIZE as usize)
        + (y as usize) * (VOLUME_SIZE as usize)
        + x as usize
}

// ---------------------------------------------------------------------------
// VoxelSnapshot — block IDs extracted from the World on the worker thread
// ---------------------------------------------------------------------------

/// A cubic snapshot of block IDs centred (approximately) on the camera.
/// Sent from the world-worker thread to the main thread via `WorldResult`.
pub struct VoxelSnapshot {
    /// World-space coordinate of the volume's lowest corner (min x, min y, min z).
    pub origin: [i32; 3],
    /// Side length (always `VOLUME_SIZE`).
    pub size: u32,
    /// Flat array of block IDs.  Index = `vol_idx(x, y, z)`.
    /// 0 = air, 1..=255 = solid block type.
    pub blocks: Vec<u8>,
}

// ---------------------------------------------------------------------------
// GPU texture packing — converts block IDs → RGBA8 using BlockRegistry
// ---------------------------------------------------------------------------

use crate::core::gameobjects::block::BlockRegistry;

/// Packed per-voxel data for `voxel_data` texture (RGBA8Unorm).
///   R, G, B = albedo colour (opaque blocks) OR block_id (model blocks, R only)
///   A       = opacity marker: 0=air, ≈100=model, ≈180=glass, 255=opaque solid
pub type VoxelDataPixel = [u8; 4];

/// Packed per-voxel data for `voxel_emission` texture (RGBA8Unorm).
///   R, G, B = emission colour (0..255)
///   A       = emission intensity (0..255, where 255 = 1.0)
pub type VoxelEmissionPixel = [u8; 4];

/// Packed per-voxel data for `voxel_tint` texture (RGBA8Unorm).
/// Only meaningful for glass voxels (`voxel_data.a == VOXEL_ALPHA_GLASS`).
///   R, G, B = tint colour (0..255) — light passing through is multiplied by this
///   A       = opacity (0..255, where 255 = fully opaque)
/// All zeros for non-glass voxels.
pub type VoxelTintPixel = [u8; 4];

pub const VOXEL_ALPHA_MODEL: u8 = 100;

/// Voxel-data alpha for fluid blocks (water, lava).
/// Placed in the 0 < a < 0.20 range so shaders do NOT treat fluids as
/// opaque shadow casters, but can still distinguish them from air (a == 0)
/// to apply water light absorption.
pub const VOXEL_ALPHA_FLUID: u8 = 40; // 40/255 ≈ 0.157

/// Voxel-data alpha marker used by the shaders to identify a glass voxel.
///
/// Final alpha map (consumed by WGSL):
///   a == 0                              → air
///   0 < a < 0.20   (e.g.  40/255)      → fluid (water/lava) — light absorption
///   0.20 < a < 0.50 (e.g. 100/255)     → model block
///   0.50 < a < 0.85 (e.g. 180/255)     → glass — sample voxel_tint
///   0.85 ≤ a ≤ 1.00 (e.g. 255/255)     → fully opaque solid
pub const VOXEL_ALPHA_GLASS: u8 = 180;

/// Pack two [0,1] floats into a single byte as two 4-bit values (nibbles).
/// Upper nibble = lo, lower nibble = hi, each quantised to 0..15.
/// Decoded in the shader: lo = (byte >> 4) / 15.0, hi = (byte & 0xF) / 15.0.
#[inline(always)]
fn pack_nibble_pair(lo: f32, hi: f32) -> u8 {
    let lo4 = (lo.clamp(0.0, 1.0) * 15.0).round() as u8;
    let hi4 = (hi.clamp(0.0, 1.0) * 15.0).round() as u8;
    (lo4 << 4) | (hi4 & 0xF)
}

/// Convert a `VoxelSnapshot` into GPU-ready RGBA8 arrays using the block registry.
///
/// Returns `(voxel_data, voxel_emission, voxel_tint)` — each is `VOLUME_TOTAL` pixels.
/// Layout is Z-outer, Y-middle, X-inner to match `write_texture` convention.
///
/// `voxel_tint` is only filled for glass voxels (`def.is_glass()`); all other
/// voxels have a zeroed entry. Shaders detect glass by `voxel_data.a` falling
/// inside the glass band (`VOXEL_ALPHA_GLASS / 255 ≈ 0.706`).
pub fn pack_volume(
    snapshot: &VoxelSnapshot,
    registry: &BlockRegistry,
) -> (Vec<VoxelDataPixel>, Vec<VoxelEmissionPixel>, Vec<VoxelTintPixel>) {
    let n = VOLUME_TOTAL;
    let mut data     = vec![[0u8; 4]; n];
    let mut emission = vec![[0u8; 4]; n];
    let mut tint     = vec![[0u8; 4]; n];

    for i in 0..n {
        let block_id = snapshot.blocks[i];
        if block_id == 0 {
            continue; // air — already zeroed
        }

        if let Some(def) = registry.get(block_id) {
            let c = &def.color;
            let is_model_block = def.model.is_some();
            let is_glass_block = def.is_glass();
            let is_fluid_block = def.is_fluid();
            let mut voxel_alpha: u8 = if is_model_block { 0 } else { 255 };
            let mut r = (c[0] * 255.0).min(255.0) as u8;
            let mut g = (c[1] * 255.0).min(255.0) as u8;
            let mut b = (c[2] * 255.0).min(255.0) as u8;

            // Model blocks with known per-cube shadow data get special alpha (100).
            // R encodes the block_id so the DDA shadow ray can look up the cube list
            // from the model_shadow_header / model_shadow_cubes storage buffers.
            if is_model_block && def.model_shadow_cubes.is_some() {
                voxel_alpha = VOXEL_ALPHA_MODEL; // 100
                r = block_id;                    // block_id → decoded in shader as round(r*255)
                g = 0;
                b = 0;
            }

            // Glass blocks override the alpha marker so shaders take the
            // partial-transmission branch. RGB still carries the albedo (used
            // for face rendering — the propagation/shadow code reads the tint
            // from voxel_tint instead).
            if is_glass_block {
                voxel_alpha = VOXEL_ALPHA_GLASS; // 180
                let gp = &def.glass;
                tint[i] = [
                    (gp.tint_color[0] * 255.0).min(255.0) as u8,
                    (gp.tint_color[1] * 255.0).min(255.0) as u8,
                    (gp.tint_color[2] * 255.0).min(255.0) as u8,
                    (gp.opacity.clamp(0.0, 1.0) * 255.0) as u8,
                ];
            }

            // Fluid blocks (water, lava) use a dedicated sub-0.20 alpha so shaders
            // skip hard DDA shadows and apply gradual light absorption instead.
            if is_fluid_block {
                voxel_alpha = VOXEL_ALPHA_FLUID; // 40
            }

            data[i] = [r, g, b, voxel_alpha];

            if def.emission.emit_light {
                let ec = &def.emission.light_color;
                // Scale alpha by (range / MAX_LIGHT_RANGE) so the inject shader
                // produces brightness proportional to the block's configured range.
                // inject_multiplier (16.0) × (range/64) = 6.0 for default range=24.
                let range_scale = (def.emission.light_range / MAX_LIGHT_RANGE).clamp(0.0, 1.0);
                let ei = (def.emission.light_intensity * def.emission.light_strength * range_scale)
                    .clamp(0.0, 1.0);
                emission[i] = [
                    (ec[0] * 255.0).min(255.0) as u8,
                    (ec[1] * 255.0).min(255.0) as u8,
                    (ec[2] * 255.0).min(255.0) as u8,
                    (ei * 255.0) as u8,
                ];
            }
        } else {
            // Unknown block — treat as solid grey
            data[i] = [128, 128, 128, 255];
        }
    }

    (data, emission, tint)
}
