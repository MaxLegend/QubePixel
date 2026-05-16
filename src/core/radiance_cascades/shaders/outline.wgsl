struct OutlineUniforms {
    view_proj: mat4x4<f32>,
    block_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: OutlineUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    let world_pos = position + u.block_pos.xyz;
    return u.view_proj * vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}