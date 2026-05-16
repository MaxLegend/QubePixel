// =============================================================================
// QubePixel — Simplified PBR Lighting (static directional light only)
// =============================================================================
// Uniform layout (272 bytes = 68 floats) — matches pack_lighting_uniforms()
struct Uniforms {
    view_proj:        mat4x4<f32>,   // [0..16]
    camera_pos:       vec4<f32>,     // [16..20]
    sun_direction:    vec4<f32>,     // [20..24]
    sun_color:        vec4<f32>,     // [24..28]
    moon_direction:   vec4<f32>,     // [28..32]
    moon_color:       vec4<f32>,     // [32..36]
    ambient_color:    vec4<f32>,     // [36..40]
    shadow_params:    vec4<f32>,     // [40..44]  (unused — kept for layout compat)
    ssao_params:      vec4<f32>,     // [44..48]  (unused — kept for layout compat)
    time_params:      vec4<f32>,     // [48..52]  (unused — kept for layout compat)
    shadow_view_proj: mat4x4<f32>,   // [52..68]  (unused — kept for layout compat)
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex:      texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler:  sampler;
@group(0) @binding(3) var normal_tex:     texture_2d<f32>;
@group(0) @binding(4) var normal_sampler: sampler;

// Packed vertex input — 52 bytes per vertex.
struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec4<f32>,
    @location(2) color_ao:  vec4<f32>,
    @location(3) texcoord:  vec2<f32>,
    @location(4) material:  vec4<f32>,
    @location(5) emission:  vec4<f32>,
    @location(6) tangent:   vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos:     vec4<f32>,
    @location(0)       v_normal:     vec3<f32>,
    @location(1)       v_color:      vec3<f32>,
    @location(2)       v_world_pos:  vec3<f32>,
    @location(3)       v_texcoord:   vec2<f32>,
    @location(4)       v_ao:         f32,
    @location(5)       v_material:   vec2<f32>,
    @location(6)       v_emission:   vec4<f32>,
    @location(7)       v_tangent:    vec3<f32>,
};

// ===== Group 1: VCT bindings (for shadows) =================================
struct VCTParams {
    volume_origin: vec4<f32>,    // xyz = world origin, w = volume_size (float)
    inv_size:      vec4<f32>,    // xyz = 1.0 / volume_size, w = gi_intensity
    gi_config:     vec4<f32>,    // x = max_shadow_steps, y = decay, z = ambient_min, w = 0
    light_counts:  vec4<u32>,    // x = point_count, y = spot_count
};

struct EntityShadowAABB {
    min_pad:     vec4<f32>,
    max_opacity: vec4<f32>,
};

@group(1) @binding(0) var<uniform>       vct: VCTParams;
@group(1) @binding(1) var                voxel_data:     texture_3d<f32>;
@group(1) @binding(2) var                voxel_radiance: texture_3d<f32>;
@group(1) @binding(3) var                voxel_sampler:  sampler;
@group(1) @binding(6) var<storage, read> model_shadow_header: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read> model_shadow_cubes:  array<f32>;
@group(1) @binding(8) var                voxel_tint:     texture_3d<f32>;
@group(1) @binding(9) var<storage, read> entity_shadow_aabbs: array<EntityShadowAABB>;

const FLUID_ALPHA_MAX:  f32 = 0.20;
const GLASS_ALPHA_MIN:  f32 = 0.55;
const GLASS_ALPHA_MAX:  f32 = 0.85;
const WATER_ABSORPTION: f32 = 0.65;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal     = normalize(input.normal.xyz);
    out.v_color      = input.color_ao.xyz;
    out.v_world_pos  = input.position;
    out.v_texcoord   = input.texcoord;
    out.v_ao         = input.color_ao.w;
    out.v_material   = input.material.xy;
    out.v_emission   = input.emission;
    out.v_tangent    = normalize(input.tangent.xyz);
    return out;
}

// ===== Model-block shadow helpers ==========================================

fn decode_nibble_pair(ch: f32) -> vec2<f32> {
    let byte = u32(round(ch * 255.0));
    return vec2<f32>(f32(byte >> 4u) / 15.0, f32(byte & 0xFu) / 15.0);
}

fn ray_seg_hits_aabb(
    ray_orig: vec3<f32>, ray_dir: vec3<f32>, t_seg: f32,
    box_min: vec3<f32>, box_max: vec3<f32>,
) -> bool {
    var t_near = 0.0;
    var t_far  = t_seg;

    if (abs(ray_dir.x) > 0.0001) {
        let t1 = (box_min.x - ray_orig.x) / ray_dir.x;
        let t2 = (box_max.x - ray_orig.x) / ray_dir.x;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.x < box_min.x || ray_orig.x > box_max.x) { return false; }

    if (abs(ray_dir.y) > 0.0001) {
        let t1 = (box_min.y - ray_orig.y) / ray_dir.y;
        let t2 = (box_max.y - ray_orig.y) / ray_dir.y;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.y < box_min.y || ray_orig.y > box_max.y) { return false; }

    if (abs(ray_dir.z) > 0.0001) {
        let t1 = (box_min.z - ray_orig.z) / ray_dir.z;
        let t2 = (box_max.z - ray_orig.z) / ray_dir.z;
        t_near = max(t_near, min(t1, t2));
        t_far  = min(t_far,  max(t1, t2));
        if (t_near > t_far) { return false; }
    } else if (ray_orig.z < box_min.z || ray_orig.z > box_max.z) { return false; }

    return true;
}

// ===========================================================================

const PI: f32 = 3.14159265;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness)
         * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn compute_pbr_light(
    N: vec3<f32>, V: vec3<f32>, L: vec3<f32>,
    light_color: vec3<f32>,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let H = normalize(V + L);
    let n_dot_l = max(dot(N, L), 0.0);
    let n_dot_v = max(dot(N, V), 0.001);
    let n_dot_h = max(dot(N, H), 0.0);
    let h_dot_v = max(dot(H, V), 0.0);

    let D = distribution_ggx(n_dot_h, roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);
    let F = fresnel_schlick(h_dot_v, f0);

    let specular = (D * G * F) / (4.0 * n_dot_v * n_dot_l + 0.0001);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metalness);
    let diffuse = kD * albedo / PI;

    return (diffuse + specular) * light_color * n_dot_l;
}

// ---------------------------------------------------------------------------
// Voxel Ray-Traced Shadow: DDA march through voxel texture
// ---------------------------------------------------------------------------
fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - vct.volume_origin.xyz) / vct.volume_origin.w;
}

fn entity_shadow_blocked_simple(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> f32 {
    let n = vct.light_counts.z;
    for (var i = 0u; i < n; i++) {
        let ab = entity_shadow_aabbs[i];
        let bmin = ab.min_pad.xyz;
        let bmax = ab.max_opacity.xyz;
        if (ray_seg_hits_aabb(origin, dir, max_t, bmin, bmax)) {
            return 1.0 - clamp(ab.max_opacity.w, 0.0, 1.0);
        }
    }
    return 1.0;
}

fn voxel_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    var transmission = vec3<f32>(1.0);

    let dir_norm = normalize(light_dir);
    let entity_factor = entity_shadow_blocked_simple(world_pos + dir_norm * 0.05, dir_norm, 256.0);
    if (entity_factor < 0.001) { return vec3<f32>(0.0); }
    transmission = transmission * entity_factor;

    let voxel_origin = vct.volume_origin.xyz;
    let vsize = vct.volume_origin.w; // usually 128.0
    let local_pos = world_pos - voxel_origin;

    // If fragment is outside voxel texture bounds, assume lit
    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return transmission;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    let dir = dir_norm;
    let sz = i32(vsize);

    let step_x: i32 = select(1, -1, dir.x < 0.0);
    let step_y: i32 = select(1, -1, dir.y < 0.0);
    let step_z: i32 = select(1, -1, dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1.0e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1.0e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001)
    );

    var t_max = vec3<f32>(0.0, 0.0, 0.0);
    if (abs(dir.x) > 0.0001) {
        t_max.x = select(f32(voxel.x + 1) - local_pos.x, f32(voxel.x) - local_pos.x, dir.x < 0.0) / dir.x;
    } else { t_max.x = 1.0e10; }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(f32(voxel.y + 1) - local_pos.y, f32(voxel.y) - local_pos.y, dir.y < 0.0) / dir.y;
    } else { t_max.y = 1.0e10; }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(f32(voxel.z + 1) - local_pos.z, f32(voxel.z) - local_pos.z, dir.z < 0.0) / dir.z;
    } else { t_max.z = 1.0e10; }
    t_max = max(t_max, vec3<f32>(0.0));

    // Pre-skip: starting voxel may be a model block.
    {
        let sv_uv = (vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z)) + 0.5) / vsize;
        let sv    = textureSampleLevel(voxel_data, voxel_sampler, sv_uv, 0.0);
        if (sv.a > 0.2 && sv.a < 0.5) {
            let block_lut = u32(round(sv.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_seg     = min(t_max.x, min(t_max.y, t_max.z));
            let vf        = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(local_pos - vf, dir, t_seg, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
    }

    var t_prev: f32 = 0.0;

    // Skip the first voxel (contains the surface) to avoid self-shadowing
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        t_prev = t_max.x;
        voxel.x = voxel.x + step_x;
        t_max.x = t_max.x + t_delta.x;
    } else if (t_max.y <= t_max.z) {
        t_prev = t_max.y;
        voxel.y = voxel.y + step_y;
        t_max.y = t_max.y + t_delta.y;
    } else {
        t_prev = t_max.z;
        voxel.z = voxel.z + step_z;
        t_max.z = t_max.z + t_delta.z;
    }

    let max_steps: u32 = 128u;
    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return transmission;
        }

        let uv = vec3<f32>(f32(voxel.x) + 0.5, f32(voxel.y) + 0.5, f32(voxel.z) + 0.5) / vsize;
        let data = textureSampleLevel(voxel_data, voxel_sampler, uv, 0.0);

        if (data.a > 0.0 && data.a < FLUID_ALPHA_MAX) {
            transmission = transmission * WATER_ABSORPTION;
            if (dot(transmission, transmission) < 0.001) {
                return vec3<f32>(0.0);
            }
        } else if (data.a >= GLASS_ALPHA_MIN && data.a <= GLASS_ALPHA_MAX) {
            let tint = textureSampleLevel(voxel_tint, voxel_sampler, uv, 0.0);
            let op   = tint.a;
            let mult = mix(vec3<f32>(1.0), tint.rgb, op) * (1.0 - 0.85 * op);
            transmission = transmission * mult;
            if (dot(transmission, transmission) < 0.0005) {
                return vec3<f32>(0.0);
            }
        } else if (data.a > 0.85) {
            let rad = textureSampleLevel(voxel_radiance, voxel_sampler, uv, 0.0);
            let is_emissive = rad.a > 0.5 && dot(rad.rgb, rad.rgb) > 0.1;
            if (!is_emissive) {
                return vec3<f32>(0.0);
            }
        } else if (data.a > 0.2 && data.a < 0.5) {
            // Model block: per-cube AABB lookup.
            let block_lut = u32(round(data.r * 255.0));
            let hdr       = model_shadow_header[block_lut];
            let c_start   = hdr.x;
            let c_count   = hdr.y;
            let t_next    = min(t_max.x, min(t_max.y, t_max.z));
            let voxel_f   = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
            let ray_entry = local_pos + dir * t_prev - voxel_f;
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (ray_seg_hits_aabb(ray_entry, dir, t_next - t_prev, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            t_prev = t_max.x;
            voxel.x = voxel.x + step_x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            t_prev = t_max.y;
            voxel.y = voxel.y + step_y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            t_prev = t_max.z;
            voxel.z = voxel.z + step_z;
            t_max.z = t_max.z + t_delta.z;
        }
    }

    return transmission;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(atlas_tex, atlas_sampler, input.v_texcoord).rgb;
    let albedo    = input.v_color * tex_color;

    // Normal mapping: TBN matrix
    let N_geom = normalize(input.v_normal);
    let T = normalize(input.v_tangent);
    let B = cross(N_geom, T);
    let nm_sample = textureSample(normal_tex, normal_sampler, input.v_texcoord).rgb;
    let N_t = normalize(nm_sample * 2.0 - 1.0);
    let N = normalize(T * N_t.x + B * N_t.y + N_geom * N_t.z);

    let V = normalize(u.camera_pos.xyz - input.v_world_pos);

    let roughness  = input.v_material.x;
    let metalness  = input.v_material.y;
    let vertex_ao  = input.v_ao;

    let f0 = mix(vec3<f32>(0.04), albedo, metalness);

    var lo = vec3<f32>(0.0);

    let sun_shadow_on = u.shadow_params.x > 0.5;

    // --- Sun light ---
    let sun_intensity = u.sun_direction.w;
    if (sun_intensity > 0.001) {
        let sun_L        = normalize(u.sun_direction.xyz);
        let sun_radiance = u.sun_color.rgb * sun_intensity;
        var shadow = vec3<f32>(1.0);
        if (sun_shadow_on) { shadow = voxel_shadow(input.v_world_pos + N_geom * 0.01, sun_L); }
        lo += compute_pbr_light(N, V, sun_L, sun_radiance * shadow, albedo, f0, roughness, metalness);
    }

    // --- Moon light ---
    let moon_intensity = u.moon_direction.w;
    if (moon_intensity > 0.001) {
        let moon_L        = normalize(u.moon_direction.xyz);
        let moon_radiance = u.moon_color.rgb * moon_intensity;
        var shadow = vec3<f32>(1.0);
        if (sun_shadow_on) { shadow = voxel_shadow(input.v_world_pos + N_geom * 0.01, moon_L); }
        lo += compute_pbr_light(N, V, moon_L, moon_radiance * shadow, albedo, f0, roughness, metalness);
    }

    // --- Emission ---
    // Multiplied by albedo so texture detail is preserved (same fix as pbr_vct.wgsl).
    let raw_intensity = input.v_emission.w;
    let display_intensity = min(abs(raw_intensity), 0.8);
    let emission = input.v_emission.rgb * albedo * display_intensity;

    // --- Ambient ---
    let ambient = u.ambient_color.rgb * albedo * vertex_ao * 0.15;

    var color = lo + emission + ambient;

    // Exposure
    color = color * 0.95;

    // ACES filmic tone mapping
    let aces_a = vec3<f32>(2.51);
    let aces_b = vec3<f32>(0.03);
    let aces_c = vec3<f32>(2.43);
    let aces_d = vec3<f32>(0.59);
    let aces_e = vec3<f32>(0.14);
    color = clamp(
        (color * (aces_a * color + aces_b)) / (color * (aces_c * color + aces_d) + aces_e),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );

    return vec4<f32>(color, 1.0);
}
