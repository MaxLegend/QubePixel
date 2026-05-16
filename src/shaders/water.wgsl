// =============================================================================
// QubePixel — Fluid Shader  (texture-based, diffuse lighting)
// =============================================================================
//
// Samples a 16×(16*N) sprite-sheet texture (N frames of 16×16 stacked
// vertically).  Frame advances every `FRAME_SECS` real seconds, driven by
// time_params.z.
//
// UV tiling is world-space triplanar so the texture repeats cleanly across
// every block face regardless of world position.
//
// Lighting: ambient + sun diffuse + moon diffuse.
// No specular, caustics, foam, or Fresnel — those come later.
// =============================================================================

struct Uniforms {
    view_proj:        mat4x4<f32>,   // [0..16]
    camera_pos:       vec4<f32>,     // [16..20]
    sun_direction:    vec4<f32>,     // [20..24]  xyz=dir, w=intensity
    sun_color:        vec4<f32>,     // [24..28]
    moon_direction:   vec4<f32>,     // [28..32]  xyz=dir, w=intensity
    moon_color:       vec4<f32>,     // [32..36]
    ambient_color:    vec4<f32>,     // [36..40]  w=min_level
    shadow_params:    vec4<f32>,     // [40..44]
    ssao_params:      vec4<f32>,     // [44..48]
    time_params:      vec4<f32>,     // [48..52]  z = real elapsed seconds
    shadow_view_proj: mat4x4<f32>,   // [52..68]
};

@group(0) @binding(0) var<uniform> u:              Uniforms;
@group(0) @binding(1) var atlas_tex:               texture_2d<f32>; // kept for layout compat
@group(0) @binding(2) var atlas_sampler:           sampler;
@group(0) @binding(3) var water_still_tex:         texture_2d<f32>; // 16×512, 32 frames — top face
@group(0) @binding(4) var water_still_sampler:     sampler;
@group(0) @binding(5) var water_flow_tex:          texture_2d<f32>; // 16×512, 32 frames — side faces
@group(0) @binding(6) var water_flow_sampler:      sampler;

// ---------------------------------------------------------------------------
// Vertex I/O
// ---------------------------------------------------------------------------

struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,   // Snorm8x4 → [-1,1]
    @location(2) color_ao:  vec4<f32>,   // Unorm8x4 → [0,1]; a = depth alpha
    @location(3) texcoord:  vec2<f32>,
    @location(4) material:  vec4<f32>,   // Unorm8x4; x=roughness, y=metalness
    @location(5) emission:  vec4<f32>,   // Float32x4; w<0 → emissive fluid
    @location(6) tangent:   vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos:    vec4<f32>,
    @location(0)       v_normal:    vec3<f32>,
    @location(1)       v_color:     vec4<f32>,
    @location(2)       v_world_pos: vec3<f32>,
    @location(3)       v_texcoord:  vec2<f32>,
    @location(4)       v_material:  vec2<f32>,
    @location(5)       v_emission:  vec4<f32>,
};

// ---------------------------------------------------------------------------
// Vertex shader
// ---------------------------------------------------------------------------

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos    = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal    = normalize(input.normal.xyz);
    out.v_color     = input.color_ao;
    out.v_world_pos = input.position;
    out.v_texcoord  = input.texcoord;
    out.v_material  = input.material.xy;
    out.v_emission  = input.emission;
    return out;
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {

    let elapsed = u.time_params.z;
    let wp      = input.v_world_pos;
    let N       = normalize(input.v_normal);

    // -----------------------------------------------------------------------
    // 1. Animated texture — frame selection
    // -----------------------------------------------------------------------
    // Sprite sheet: 32 frames of 16×16 stacked vertically (16×512 total).
    // Animation advances at 20 frames/sec (matching Minecraft's tick rate),
    // producing a smooth cycle with period 32/20 = 1.6 seconds.
    let num_frames = 32.0;
    let anim_fps   = 20.0;
    let frame_idx  = i32(floor(elapsed * anim_fps)) % i32(num_frames);

    // -----------------------------------------------------------------------
    // 2. World-space triplanar UV — tiles cleanly in both directions
    // -----------------------------------------------------------------------
    let abs_n = abs(N);

    var tile_u:       f32;
    var tile_v_local: f32;  // position within one 16×16 frame (0..1)
    var use_flow:     bool; // true → sample flow texture (side faces)

    if abs_n.y >= abs_n.x && abs_n.y >= abs_n.z {
        // Horizontal face (top / bottom): XZ tiling, still-water texture
        tile_u       = fract(wp.x);
        tile_v_local = fract(wp.z);
        use_flow     = false;
    } else if abs_n.x >= abs_n.z {
        // X-facing side: ZY tiling, flow texture
        tile_u       = fract(wp.z);
        tile_v_local = fract(wp.y);
        use_flow     = true;
    } else {
        // Z-facing side: XY tiling, flow texture
        tile_u       = fract(wp.x);
        tile_v_local = fract(wp.y);
        use_flow     = true;
    }

    // Map local V into the correct frame strip of the sprite sheet
    let tile_v  = (tile_v_local + f32(frame_idx)) / num_frames;
    let tile_uv = vec2<f32>(tile_u, tile_v);

    // Sample the appropriate sprite sheet
    var tex_col: vec4<f32>;
    if use_flow {
        tex_col = textureSample(water_flow_tex, water_flow_sampler, tile_uv);
    } else {
        tex_col = textureSample(water_still_tex, water_still_sampler, tile_uv);
    }

    // -----------------------------------------------------------------------
    // 3. Base colour — texture tinted by block albedo
    // -----------------------------------------------------------------------
    var base = input.v_color.rgb * tex_col.rgb;

    // -----------------------------------------------------------------------
    // 4. Diffuse lighting  (ambient + sun + moon)
    // -----------------------------------------------------------------------
    var light = u.ambient_color.rgb;

    let sun_intensity = u.sun_direction.w;
    if sun_intensity > 0.001 {
        let sun_L = normalize(u.sun_direction.xyz);
        let NdotL = max(dot(N, sun_L), 0.0);
        light += u.sun_color.rgb * sun_intensity * NdotL;
    }

    let moon_intensity = u.moon_direction.w;
    if moon_intensity > 0.001 {
        let moon_L = normalize(u.moon_direction.xyz);
        let NdotL  = max(dot(N, moon_L), 0.0);
        light += u.moon_color.rgb * moon_intensity * NdotL * 0.45;
    }

    var color = base * light;

    // -----------------------------------------------------------------------
    // 5. Emissive fluids (lava)
    // -----------------------------------------------------------------------
    // Emission intensity stored as negative value in v_emission.w
    let is_lava = input.v_emission.w < -0.01;
    if is_lava {
        let emi = abs(input.v_emission.w);
        color = color + input.v_emission.rgb * emi * 0.5;
    }

    // -----------------------------------------------------------------------
    // 6. Output — depth-dependent alpha from vertex (already in [0,1])
    // -----------------------------------------------------------------------
    return vec4<f32>(color, input.v_color.a);
}
