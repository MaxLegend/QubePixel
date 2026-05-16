// =============================================================================
// QubePixel — VCT Light Injection Compute Shader
// =============================================================================
// Injects emissive block light into the radiance volume.
// For each emissive solid voxel, writes emission colour × intensity.
// For non-emissive solid voxels: writes zero (they block light but don't emit).
// For air voxels: writes zero (will be filled by propagation).

struct InjectParams {
    volume_size: vec4<u32>,     // x = side length (128), yzw = 0
    _pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: InjectParams;
@group(0) @binding(1) var voxel_emission: texture_3d<f32>;
@group(0) @binding(2) var radiance_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = params.volume_size.x;
    if (any(gid >= vec3<u32>(size))) { return; }

    let pos = vec3<i32>(gid);

    // Read emission for this voxel
    let em = textureLoad(voxel_emission, pos, 0);
    let intensity = em.a;

    if (intensity > 0.001) {
        // Emissive block: inject its light.
        // Multiplier kept low (2.0) because max-based propagation preserves energy
        // without the 1/r² averaging penalty.  CPU-side pack_volume encodes
        // alpha = intensity * (light_range / MAX_LIGHT_RANGE), so brighter = farther range.
        textureStore(radiance_out, pos, vec4<f32>(em.rgb * intensity * 2.0, 1.0));
    } else {
        // Non-emissive or air: zero radiance
        textureStore(radiance_out, pos, vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }
}
