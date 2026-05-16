struct OutlineUniforms {
    view_proj: mat4x4<f32>,  // 64 bytes
    rotation:  mat4x4<f32>,  // 64 bytes — block rotation matrix (identity for regular blocks)
    block_pos: vec4<f32>,    // 16 bytes
    aabb_min:  vec4<f32>,    // 16 bytes
    aabb_max:  vec4<f32>,    // 16 bytes
};
// Total: 176 bytes

@group(0) @binding(0) var<uniform> u: OutlineUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    // Map unit-cube vertex [0..1] (with small epsilon expansion) to the actual AABB.
    let local_pos = mix(u.aabb_min.xyz, u.aabb_max.xyz, position);
    // Rotate around block pivot (0.5, 0, 0.5) — matches BlockModelRenderer world_pos origin.
    let pivot     = vec3<f32>(0.5, 0.0, 0.5);
    let centered  = local_pos - pivot;
    let rotated   = (u.rotation * vec4<f32>(centered, 0.0)).xyz + pivot;
    let world_pos = rotated + u.block_pos.xyz;
    return u.view_proj * vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
