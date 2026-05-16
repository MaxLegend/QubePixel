// =============================================================================
// QubePixel — Debug Lines Shader
// Renders coloured 3D line segments with no lighting or depth test.
// Used for bounding boxes, physics capsules, light vectors, etc.
// =============================================================================

struct Uniforms {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) pos:   vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip:  vec4<f32>,
    @location(0)       color: vec4<f32>,
};

@vertex
fn vs_main(v: VsIn) -> VsOut {
    return VsOut(u.view_proj * vec4<f32>(v.pos, 1.0), v.color);
}

@fragment
fn fs_main(v: VsOut) -> @location(0) vec4<f32> {
    return v.color;
}
