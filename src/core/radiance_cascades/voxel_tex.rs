// =============================================================================
// QubePixel — Radiance Cascades: Voxel Texture & Block LUT
// =============================================================================
//
// Builds GPU resources for the compute shaders:
//   1. VoxelTexture3D — 3D R16Uint texture of block IDs around the camera.
//   2. BlockLUT — 1D Rgba32Float texture of per-block-type properties.
//
// These resources are uploaded to the GPU each frame (or when chunks change)
// and bound to the compute shader for ray marching.

use glam::Vec3;

use crate::debug_log;

use super::types::BlockProperties;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of the voxel texture per axis (blocks).
/// Covers VOXEL_TEX_HALF_SIZE blocks in each direction from the camera.
pub const VOXEL_TEX_SIZE: u32 = 128;

/// Half-size: camera sits at the center, coverage = VOXEL_TEX_SIZE/2 blocks.
pub const VOXEL_TEX_HALF_SIZE: u32 = 64;

/// Maximum number of block types.
pub const MAX_BLOCK_TYPES: u32 = 256;

// ---------------------------------------------------------------------------
// VoxelTextureBuilder
// ---------------------------------------------------------------------------

/// Builder for the 3D voxel texture that holds block IDs.
///
/// The texture is R16Uint: each texel = one block ID (0-255).
/// Dimensions: VOXEL_TEX_SIZE x VOXEL_TEX_SIZE x VOXEL_TEX_SIZE.
///
/// Usage:
///   1. Create with VoxelTextureBuilder::new().
///   2. Fill block data via fill_from_world() or set_block().
///   3. Upload to GPU via build().
pub struct VoxelTextureBuilder {
    /// CPU-side buffer: flat array of block IDs.
    /// Layout: [x + y * W + z * W * H], each element = u16 block ID.
    data: Vec<u16>,
    /// World origin (block coordinates) corresponding to texel (0,0,0).
    world_origin: glam::IVec3,
    /// Camera position (world coordinates) used to center the texture.
    camera_pos: Vec3,
}

impl VoxelTextureBuilder {
    /// Create an empty voxel texture builder.
    ///
    /// The texture is initialized with 0xFFFF (unloaded sentinel).
    pub fn new() -> Self {
        let size = VOXEL_TEX_SIZE as usize;
        let total = size * size * size;
        debug_log!(
            "VoxelTextureBuilder", "new",
            "Allocating voxel texture buffer: {}x{}x{} = {} texels ({:.2} MB)",
            VOXEL_TEX_SIZE, VOXEL_TEX_SIZE, VOXEL_TEX_SIZE,
            total,
            total as f64 * 2.0 / (1024.0 * 1024.0)
        );
        Self {
            data: vec![0xFFFF; total], // Sentinel: unloaded
            world_origin: glam::IVec3::ZERO,
            camera_pos: Vec3::ZERO,
        }
    }

    /// Recenter the voxel texture around a new camera position.
    ///
    /// Sets the world_origin such that the camera sits at the center of
    /// the texture. Existing data is cleared (all set to sentinel).
    pub fn recenter(&mut self, camera_pos: Vec3) {
        self.camera_pos = camera_pos;
        let cx = camera_pos.x.round() as i32;
        let cy = camera_pos.y.round() as i32;
        let cz = camera_pos.z.round() as i32;
        self.world_origin = glam::IVec3::new(
            cx - VOXEL_TEX_HALF_SIZE as i32,
            cy - VOXEL_TEX_HALF_SIZE as i32,
            cz - VOXEL_TEX_HALF_SIZE as i32,
        );
        // Clear all data
        self.data.fill(0xFFFF);
    }

    /// Convert world block coordinates to voxel texture coordinates.
    ///
    /// Returns None if the world position is outside the texture bounds.
    pub fn world_to_texel(&self, wx: i32, wy: i32, wz: i32) -> Option<(u32, u32, u32)> {
        let tx = wx - self.world_origin.x;
        let ty = wy - self.world_origin.y;
        let tz = wz - self.world_origin.z;
        if tx < 0 || ty < 0 || tz < 0 {
            return None;
        }
        let tx = tx as u32;
        let ty = ty as u32;
        let tz = tz as u32;
        if tx >= VOXEL_TEX_SIZE || ty >= VOXEL_TEX_SIZE || tz >= VOXEL_TEX_SIZE {
            return None;
        }
        Some((tx, ty, tz))
    }

    /// Set a single block in the CPU buffer.
    ///
    /// `wx, wy, wz`: world block coordinates.
    /// `block_id`: block type (0-255).
    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block_id: u8) {
        if let Some((tx, ty, tz)) = self.world_to_texel(wx, wy, wz) {
            let size = VOXEL_TEX_SIZE as usize;
            let idx = tx as usize + ty as usize * size + tz as usize * size * size;
            self.data[idx] = block_id as u16;
        }
    }

    /// Get a block ID from the CPU buffer.
    ///
    /// Returns 0xFFFF if the position is outside bounds or unloaded.
    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> u16 {
        self.world_to_texel(wx, wy, wz)
            .map(|(tx, ty, tz)| {
                let size = VOXEL_TEX_SIZE as usize;
                self.data[tx as usize + ty as usize * size + tz as usize * size * size]
            })
            .unwrap_or(0xFFFF)
    }
    /// Напрямую задать данные воксел-буфера и его мировое начало.
    /// Используется для передачи данных из WorldWorker без копирования чанков.
    pub fn set_data_raw(&mut self, origin: glam::IVec3, data: Vec<u16>) {
        self.world_origin = origin;
        // Обновляем camera_pos как центр переданного региона
        self.camera_pos = glam::Vec3::new(
            origin.x as f32 + VOXEL_TEX_HALF_SIZE as f32,
            origin.y as f32 + VOXEL_TEX_HALF_SIZE as f32,
            origin.z as f32 + VOXEL_TEX_HALF_SIZE as f32,
        );
        if data.len() == self.data.len() {
            self.data = data;
        } else {
            debug_log!("VoxelTextureBuilder", "set_data_raw",
                "Data size mismatch: got {} expected {}", data.len(), self.data.len());
        }
    }
    /// Access the raw data buffer (for GPU upload).
    pub fn data(&self) -> &[u16] {
        &self.data
    }

    /// World origin of the texture (block coordinates of texel 0,0,0).
    pub fn world_origin(&self) -> glam::IVec3 {
        self.world_origin
    }

    /// Size of the data buffer in bytes.
    pub fn data_bytes(&self) -> usize {
        self.data.len() * 2 // u16 = 2 bytes
    }
}

impl Default for VoxelTextureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BlockLUTBuilder
// ---------------------------------------------------------------------------

/// Builder for the 1D block property lookup texture.
///
/// The texture is Rgba32Float: each texel = one BlockProperties.
/// Indexed by block ID (0 = Air, 1-255 = registered blocks).
///
/// Usage:
///   1. Create with BlockLUTBuilder::new().
///   2. Fill from BlockRegistry via fill_from_registry().
///   3. Upload to GPU via build().
pub struct BlockLUTBuilder {
    /// CPU-side buffer: 256 BlockProperties entries.
    data: Vec<BlockProperties>,
}

impl BlockLUTBuilder {
    /// Create a new BlockLUT builder with all entries initialized to air.
    pub fn new() -> Self {
        debug_log!(
            "BlockLUTBuilder", "new",
            "Allocating block LUT: {} entries ({:.1} KB)",
            MAX_BLOCK_TYPES,
            MAX_BLOCK_TYPES as f64 * BlockProperties::SIZE as f64 / 1024.0
        );
        Self {
            data: vec![BlockProperties::air(); MAX_BLOCK_TYPES as usize],
        }
    }

    /// Set the properties for a specific block ID.
    ///
    /// `block_id`: block type (0-255).
    /// `props`: optical and emission properties.
    pub fn set(&mut self, block_id: u8, props: BlockProperties) {
        if (block_id as usize) < self.data.len() {
            self.data[block_id as usize] = props;
        }
    }

    /// Get properties for a specific block ID.
    pub fn get(&self, block_id: u8) -> BlockProperties {
        self.data
            .get(block_id as usize)
            .copied()
            .unwrap_or(BlockProperties::air())
    }

    /// Access the raw data buffer (for GPU upload).
    pub fn data(&self) -> &[BlockProperties] {
        &self.data
    }

    /// Number of entries in the LUT.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Size of the data buffer in bytes.
    pub fn data_bytes(&self) -> usize {
        self.data.len() * BlockProperties::SIZE as usize
    }
}

impl Default for BlockLUTBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Fill helpers (will be connected to BlockRegistry in integration)
// ---------------------------------------------------------------------------

/// Fill the BlockLUT from a block registry.
///
/// This is a standalone function that takes a closure returning block properties
/// for each block ID. This avoids a direct dependency on BlockRegistry here.
///
/// The closure receives a block_id (1-255) and returns Option<BlockProperties>.
/// Returning None means the block type is not registered (kept as air).
pub fn fill_lut_from_closure<F>(lut: &mut BlockLUTBuilder, mut get_props: F)
where
    F: FnMut(u8) -> Option<BlockProperties>,
{
    let mut registered = 0usize;
    for id in 1u8..=255 {
        if let Some(props) = get_props(id) {
            lut.set(id, props);
            registered += 1;
        }
    }
    debug_log!(
        "BlockLUTBuilder", "fill_lut_from_closure",
        "Registered {} block types in LUT",
        registered
    );
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_texture_builder_new() {
        let vtb = VoxelTextureBuilder::new();
        assert_eq!(vtb.data().len(), 128 * 128 * 128);
        // All should be sentinel (0xFFFF)
        assert!(vtb.data().iter().all(|&b| b == 0xFFFF));
    }

    #[test]
    fn test_voxel_texture_set_get() {
        let mut vtb = VoxelTextureBuilder::new();
        // Recenter so we know the mapping
        vtb.recenter(Vec3::new(64.0, 64.0, 64.0));
        // Origin should be at (0, 0, 0)
        assert_eq!(vtb.world_origin(), glam::IVec3::new(0, 0, 0));

        // Set block at world (10, 20, 30) = texel (10, 20, 30)
        vtb.set_block(10, 20, 30, 42);
        assert_eq!(vtb.get_block(10, 20, 30), 42);

        // Unset block should still be sentinel
        assert_eq!(vtb.get_block(0, 0, 0), 0xFFFF);
    }

    #[test]
    fn test_voxel_texture_out_of_bounds() {
        let mut vtb = VoxelTextureBuilder::new();
        vtb.recenter(Vec3::new(64.0, 64.0, 64.0));

        // Before origin
        assert_eq!(vtb.get_block(-1, 0, 0), 0xFFFF);
        // After origin + size
        assert_eq!(vtb.get_block(128, 0, 0), 0xFFFF);
        // Inside
        vtb.set_block(127, 127, 127, 99);
        assert_eq!(vtb.get_block(127, 127, 127), 99);
    }

    #[test]
    fn test_voxel_texture_recenter_clears() {
        let mut vtb = VoxelTextureBuilder::new();
        vtb.recenter(Vec3::new(64.0, 64.0, 64.0));
        vtb.set_block(10, 20, 30, 42);

        // Recenter to a different position
        vtb.recenter(Vec3::new(200.0, 200.0, 200.0));
        // Previous block should be cleared
        assert_eq!(vtb.get_block(10, 20, 30), 0xFFFF);

        // New origin should be at (136, 136, 136)
        let origin = vtb.world_origin();
        assert_eq!(origin.x, 136);
        assert_eq!(origin.y, 136);
        assert_eq!(origin.z, 136);
    }

    #[test]
    fn test_voxel_texture_data_bytes() {
        let vtb = VoxelTextureBuilder::new();
        // 128^3 * 2 bytes = 4,194,304 bytes = 4 MB
        assert_eq!(vtb.data_bytes(), 4_194_304);
    }

    #[test]
    fn test_block_lut_builder_new() {
        let lut = BlockLUTBuilder::new();
        assert_eq!(lut.len(), 256);
        // Block 0 = Air (default)
        let air = lut.get(0);
        assert!((air.opacity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_lut_set_get() {
        let mut lut = BlockLUTBuilder::new();
        lut.set(1, BlockProperties::opaque());
        let b1 = lut.get(1);
        assert!((b1.opacity - 1000.0).abs() < 1e-6);

        lut.set(2, BlockProperties::emissive(1.0, 0.5, 0.0, 2.0));
        let b2 = lut.get(2);
        assert!((b2.opacity - 1000.0).abs() < 1e-6);
        assert!((b2.emission_r - 2.0).abs() < 1e-6);
        assert!((b2.emission_g - 1.0).abs() < 1e-6);
        assert!((b2.emission_b - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_block_lut_data_bytes() {
        let lut = BlockLUTBuilder::new();
        // 256 * 16 bytes = 4096 bytes = 4 KB
        assert_eq!(lut.data_bytes(), 4096);
    }

    #[test]
    fn test_fill_lut_from_closure() {
        let mut lut = BlockLUTBuilder::new();
        fill_lut_from_closure(&mut lut, |id| {
            match id {
                1 => Some(BlockProperties::opaque()),
                2 => Some(BlockProperties::emissive(1.0, 0.0, 0.0, 1.0)),
                _ => None,
            }
        });
        assert!((lut.get(1).opacity - 1000.0).abs() < 1e-6);
        assert!((lut.get(2).emission_r - 1.0).abs() < 1e-6);
        assert!((lut.get(3).opacity - 0.0).abs() < 1e-6); // Not registered
    }
}
