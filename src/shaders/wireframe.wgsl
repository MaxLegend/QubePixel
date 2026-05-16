// =============================================================================
// QubePixel — Wireframe Shader
// Renders solid geometry as polygon-mode lines (PolygonMode::Line).
// Reads only @location(0) position from the Vertex3D layout (stride = 52 bytes).
// Requires wgpu::Features::POLYGON_MODE_LINE.
// =============================================================================

struct Uniforms {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // Semi-transparent teal — clearly visible over terrain without obscuring it.
    return vec4<f32>(0.0, 1.0, 0.85, 0.55);
}
