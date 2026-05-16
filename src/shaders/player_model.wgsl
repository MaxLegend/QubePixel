// =============================================================================
// QubePixel — player_model.wgsl
// Simple textured + lit shader for the player body.
// =============================================================================

struct Uniforms {
    /// Combined view * projection matrix.
    view_proj: mat4x4<f32>,
    /// Model matrix (world transform: translation + yaw rotation).
    model: mat4x4<f32>,
    /// Sun direction (world space, pointing toward sun).
    sun_dir: vec3<f32>,
    _pad0: f32,
    /// Ambient light colour × intensity.
    ambient: vec3<f32>,
    _pad1: f32,
    /// Sun / directional light colour × intensity.
    sun_color: vec3<f32>,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var skin_tex:  texture_2d<f32>;
@group(0) @binding(2) var skin_samp: sampler;

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
}

struct VertOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv:     vec2<f32>,
    @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
    var out: VertOut;
    let world_pos = u.model * vec4<f32>(v.position, 1.0);
    out.clip_pos  = u.view_proj * world_pos;
    out.uv        = v.uv;
    // Transform normal by the model matrix (ignoring non-uniform scale — model only rotates).
    out.normal    = normalize((u.model * vec4<f32>(v.normal, 0.0)).xyz);
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let color = textureSample(skin_tex, skin_samp, in.uv);
    // Alpha cutout (for hat / sleeve overlay layers)
    if color.a < 0.05 {
        discard;
    }

    let n       = normalize(in.normal);
    let diffuse = max(dot(n, normalize(u.sun_dir)), 0.0);
    let light   = u.ambient + u.sun_color * diffuse;

    return vec4<f32>(color.rgb * light, color.a);
}
