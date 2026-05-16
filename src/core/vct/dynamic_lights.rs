// =============================================================================
// QubePixel — Dynamic Lights (point + spot) for VCT
// =============================================================================

/// GPU-side point light (48 bytes, matches WGSL `PointLight`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLightGPU {
    /// xyz = world position, w = range (blocks)
    pub pos_range: [f32; 4],
    /// rgb = colour, w = intensity
    pub color_intensity: [f32; 4],
    /// xyz = world-space integer block coords of the emitting block, w = 1 if valid (0 for entity lights).
    /// Used in DDA shadow march to skip self-shadowing from the source block.
    pub source_voxel: [i32; 4],
}

/// GPU-side spot light (64 bytes, matches WGSL `SpotLight`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpotLightGPU {
    /// xyz = world position, w = range (blocks)
    pub pos_range: [f32; 4],
    /// xyz = direction (normalised), w = cos(inner_angle)
    pub dir_inner: [f32; 4],
    /// rgb = colour, w = intensity
    pub color_intensity: [f32; 4],
    /// x = cos(outer_angle), yzw = padding
    pub outer_pad: [f32; 4],
}

/// GPU-side entity shadow AABB (32 bytes, matches WGSL `EntityShadowAABB`).
/// Used for casting vector (DDA) shadows from non-voxel entities (player, mobs).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EntityShadowAABB {
    /// xyz = world-space min corner, w = unused (alignment).
    pub min_pad: [f32; 4],
    /// xyz = world-space max corner, w = opacity (0..1, 1.0 = fully opaque).
    pub max_opacity: [f32; 4],
}

// Compile-time size asserts (must match WGSL struct layout byte-for-byte).
const _: () = assert!(std::mem::size_of::<PointLightGPU>() == 48);
const _: () = assert!(std::mem::size_of::<SpotLightGPU>() == 64);
const _: () = assert!(std::mem::size_of::<EntityShadowAABB>() == 32);
