// =============================================================================
// QubePixel — entity_pbr.wgsl
// Full PBR + VCT Global Illumination for skeletal entities.
//
// Group 0: per-bone uniform (view_proj + bone model matrix + lighting)
//          + entity skin/albedo texture
// Group 1: VCT GI volume  — exactly the same layout as pbr_vct.wgsl group 1
//          (shared wgpu::BindGroup from VCTSystem::prepare_fragment)
// =============================================================================

// ===== Group 0: Per-bone uniforms + skin texture =============================

struct EntityUniforms {
    view_proj:      mat4x4<f32>,  // [  0.. 64]
    model:          mat4x4<f32>,  // [ 64..128]
    sun_direction:  vec4<f32>,    // [128..144]  xyz = dir, w = intensity
    sun_color:      vec4<f32>,    // [144..160]
    moon_direction: vec4<f32>,    // [160..176]  xyz = dir, w = intensity
    moon_color:     vec4<f32>,    // [176..192]
    ambient_color:  vec4<f32>,    // [192..208]  xyz = colour, w = min_level
    camera_pos:     vec4<f32>,    // [208..224]
    shadow_params:  vec4<f32>,    // [224..240]  x = shadow_on(1/0), y = normal_offset, z = pad, w = pad
}

@group(0) @binding(0) var<uniform> u:           EntityUniforms;
@group(0) @binding(1) var          albedo_tex:  texture_2d<f32>;
@group(0) @binding(2) var          albedo_samp: sampler;

// ===== Group 1: VCT (shared with block shader — same layout) =================

struct VCTParams {
    volume_origin: vec4<f32>,   // xyz = world origin, w = volume_size (blocks)
    inv_size:      vec4<f32>,   // xyz = 1/volume_size, w = gi_intensity
    gi_config:     vec4<f32>,   // x = max_shadow_steps, y = decay, z = ambient_min
    light_counts:  vec4<u32>,   // x = point_count, y = spot_count
}

struct PointLight {
    pos_range:       vec4<f32>,  // xyz = position, w = range
    color_intensity: vec4<f32>,  // rgb = colour,   w = intensity
    source_voxel:    vec4<i32>,  // xyz = world block coords of emitter; w = 1 if valid, 0 for entity lights
}

struct SpotLight {
    pos_range:       vec4<f32>,  // xyz = position, w = range
    dir_inner:       vec4<f32>,  // xyz = direction, w = cos(inner_half_angle)
    color_intensity: vec4<f32>,  // rgb = colour,    w = intensity
    outer_pad:       vec4<f32>,  // x = cos(outer_half_angle)
}

struct EntityShadowAABB {
    min_pad:     vec4<f32>,
    max_opacity: vec4<f32>,
};

@group(1) @binding(0) var<uniform>       vct:           VCTParams;
@group(1) @binding(1) var                voxel_data:    texture_3d<f32>;
@group(1) @binding(2) var                voxel_radiance:texture_3d<f32>;
@group(1) @binding(3) var                voxel_sampler: sampler;
@group(1) @binding(4) var<storage, read> point_lights:  array<PointLight>;
@group(1) @binding(5) var<storage, read> spot_lights:   array<SpotLight>;
@group(1) @binding(6) var<storage, read> model_shadow_header: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read> model_shadow_cubes:  array<f32>;
@group(1) @binding(8) var                voxel_tint:    texture_3d<f32>;
@group(1) @binding(9) var<storage, read> entity_shadow_aabbs: array<EntityShadowAABB>;

const FLUID_ALPHA_MAX:  f32 = 0.20;
const GLASS_ALPHA_MIN:  f32 = 0.55;
const GLASS_ALPHA_MAX:  f32 = 0.85;
const WATER_ABSORPTION: f32 = 0.65;

/// Test ray vs entity AABBs. Returns 1.0 if unobstructed, else (1 - opacity)
/// of the first AABB hit. Entities self-test is avoided because the rays
/// originate on the entity surface and travel outward.
fn entity_shadow_blocked(origin: vec3<f32>, dir: vec3<f32>, max_t: f32) -> f32 {
    let n = vct.light_counts.z;
    for (var i = 0u; i < n; i++) {
        let ab = entity_shadow_aabbs[i];
        let bmin = ab.min_pad.xyz;
        let bmax = ab.max_opacity.xyz;
        // Skip if origin is inside this AABB (would self-shadow the same entity).
        if (all(origin >= bmin) && all(origin <= bmax)) { continue; }
        if (ray_seg_hits_aabb(origin, dir, max_t, bmin, bmax)) {
            return 1.0 - clamp(ab.max_opacity.w, 0.0, 1.0);
        }
    }
    return 1.0;
}

// ===== Vertex I/O ============================================================

struct VertIn {
    @location(0) position: vec3<f32>,
    @location(1) uv:       vec2<f32>,
    @location(2) normal:   vec3<f32>,
}

struct VertOut {
    @builtin(position) clip_pos:  vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal:    vec3<f32>,
    @location(2)       uv:        vec2<f32>,
}

@vertex
fn vs_main(v: VertIn) -> VertOut {
    let wp = u.model * vec4<f32>(v.position, 1.0);
    var out: VertOut;
    out.world_pos = wp.xyz;
    out.clip_pos  = u.view_proj * wp;
    out.normal    = normalize((u.model * vec4<f32>(v.normal, 0.0)).xyz);
    out.uv        = v.uv;
    return out;
}

// ===== PBR core (Cook-Torrance) ==============================================

const PI: f32 = 3.14159265;

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let d  = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.0001);
}

fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(ndv: f32, ndl: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(ndv, roughness) * geometry_schlick_ggx(ndl, roughness);
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
    let H     = normalize(V + L);
    let ndl   = max(dot(N, L), 0.0);
    let ndv   = max(dot(N, V), 0.001);
    let ndh   = max(dot(N, H), 0.0);
    let hdv   = max(dot(H, V), 0.0);
    let D     = distribution_ggx(ndh, roughness);
    let G     = geometry_smith(ndv, ndl, roughness);
    let F     = fresnel_schlick(hdv, f0);
    let spec  = (D * G * F) / (4.0 * ndv * ndl + 0.0001);
    let kD    = (vec3<f32>(1.0) - F) * (1.0 - metalness);
    return (kD * albedo / PI + spec) * light_color * ndl;
}

// ===== VCT helpers ===========================================================

fn world_to_voxel_uv(world_pos: vec3<f32>) -> vec3<f32> {
    return (world_pos - vct.volume_origin.xyz) / vct.volume_origin.w;
}

fn is_in_volume(uv: vec3<f32>) -> bool {
    return all(uv >= vec3<f32>(0.0)) && all(uv < vec3<f32>(1.0));
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

// ---------------------------------------------------------------------------
// DDA voxel shadow — directional light (sun/moon).
// Same algorithm as pbr_vct.wgsl for identical shadow quality.
// ---------------------------------------------------------------------------

fn voxel_shadow_directional(world_pos: vec3<f32>, light_dir: vec3<f32>) -> vec3<f32> {
    var transmission = vec3<f32>(1.0);

    // Entity AABB pre-test (skip-self handled inside).
    let dir_norm = normalize(light_dir);
    let entity_factor = entity_shadow_blocked(world_pos + dir_norm * 0.05, dir_norm, 256.0);
    if (entity_factor < 0.001) { return vec3<f32>(0.0); }
    transmission = transmission * entity_factor;

    let voxel_origin = vct.volume_origin.xyz;
    let vsize        = vct.volume_origin.w;
    let local_pos    = world_pos - voxel_origin;

    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return transmission;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z)),
    );

    let dir    = dir_norm;
    let sz     = i32(vsize);
    let step_x = select(1, -1, dir.x < 0.0);
    let step_y = select(1, -1, dir.y < 0.0);
    let step_z = select(1, -1, dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1.0e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1.0e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001),
    );

    var t_max = vec3<f32>(1.0e10);
    if (abs(dir.x) > 0.0001) {
        t_max.x = select(f32(voxel.x + 1) - local_pos.x,
                         f32(voxel.x)     - local_pos.x, dir.x < 0.0) / dir.x;
    }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(f32(voxel.y + 1) - local_pos.y,
                         f32(voxel.y)     - local_pos.y, dir.y < 0.0) / dir.y;
    }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(f32(voxel.z + 1) - local_pos.z,
                         f32(voxel.z)     - local_pos.z, dir.z < 0.0) / dir.z;
    }
    t_max = max(t_max, vec3<f32>(0.0));

    // Pre-skip: starting voxel may be a model block.
    // NOTE: the entity fragment IS on the model surface, so its position is always
    // inside its own cube's AABB (AABB ⊇ rotated cube for any rotation).  We must
    // skip any AABB that already contains the fragment origin — otherwise t_near=0
    // trivially satisfies the slab test and every face falsely reports shadow.
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
            let ro        = local_pos - vf;
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (all(ro >= bmin) && all(ro <= bmax)) { continue; }
                if (ray_seg_hits_aabb(ro, dir, t_seg, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
    }

    var t_prev = 0.0;
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
    } else if (t_max.y <= t_max.z) {
        voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
    } else {
        voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
    }

    let max_steps = i32(vct.gi_config.x);
    for (var i = 0; i < max_steps; i++) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return transmission;
        }
        let uv   = vec3<f32>(f32(voxel.x) + 0.5, f32(voxel.y) + 0.5, f32(voxel.z) + 0.5) / vsize;
        let data = textureSampleLevel(voxel_data, voxel_sampler, uv, 0.0);
        let opacity = data.a;

        if (opacity > 0.0 && opacity < FLUID_ALPHA_MAX) {
            // Fluid voxel: absorb light gradually, no hard shadow.
            transmission = transmission * WATER_ABSORPTION;
            if (dot(transmission, transmission) < 0.001) {
                return vec3<f32>(0.0);
            }
        } else if (opacity >= GLASS_ALPHA_MIN && opacity <= GLASS_ALPHA_MAX) {
            let tint = textureSampleLevel(voxel_tint, voxel_sampler, uv, 0.0);
            let op   = tint.a;
            let mult = mix(vec3<f32>(1.0), tint.rgb, op) * (1.0 - 0.85 * op);
            transmission = transmission * mult;
            if (dot(transmission, transmission) < 0.0005) {
                return vec3<f32>(0.0);
            }
        } else if (opacity > 0.85) {
            let rad        = textureSampleLevel(voxel_radiance, voxel_sampler, uv, 0.0);
            let is_emissive = rad.a > 0.5 && dot(rad.rgb, rad.rgb) > 0.1;
            if (!is_emissive) { return vec3<f32>(0.0); }
        } else if (opacity > 0.2 && opacity < 0.5) {
            // Model block: look up per-cube AABBs from storage buffers.
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
            voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
        } else if (t_max.y <= t_max.z) {
            voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
        } else {
            voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
        }
    }
    return transmission;
}

// ---------------------------------------------------------------------------
// DDA voxel shadow — point/spot light (limited range)
// ---------------------------------------------------------------------------

// source_vox: xyz = world block coords of emitting block; w = 1 if valid (0 for entity/spot lights).
fn voxel_shadow_to_point(world_pos: vec3<f32>, light_pos: vec3<f32>, source_vox: vec4<i32>) -> vec3<f32> {
    let diff = light_pos - world_pos;
    let dist = length(diff);
    if (dist < 1.0) { return vec3<f32>(1.0); }
    let light_dir = diff / dist;

    var transmission = vec3<f32>(1.0);

    // Entity AABB pre-test
    let entity_factor = entity_shadow_blocked(world_pos + light_dir * 0.05, light_dir, dist);
    if (entity_factor < 0.001) { return vec3<f32>(0.0); }
    transmission = transmission * entity_factor;

    let voxel_origin = vct.volume_origin.xyz;
    let vsize        = vct.volume_origin.w;
    let local_pos    = world_pos - voxel_origin;

    // Precompute source block in VCT-local voxel coords for self-shadow skip.
    let vox_origin_i     = vec3<i32>(i32(voxel_origin.x), i32(voxel_origin.y), i32(voxel_origin.z));
    let source_vox_local = source_vox.xyz - vox_origin_i;

    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return transmission;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z)),
    );

    let dir    = light_dir;
    let sz     = i32(vsize);
    let step_x = select(1, -1, dir.x < 0.0);
    let step_y = select(1, -1, dir.y < 0.0);
    let step_z = select(1, -1, dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1.0e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1.0e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001),
    );

    var t_max = vec3<f32>(1.0e10);
    if (abs(dir.x) > 0.0001) {
        t_max.x = select(f32(voxel.x + 1) - local_pos.x,
                         f32(voxel.x)     - local_pos.x, dir.x < 0.0) / dir.x;
    }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(f32(voxel.y + 1) - local_pos.y,
                         f32(voxel.y)     - local_pos.y, dir.y < 0.0) / dir.y;
    }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(f32(voxel.z + 1) - local_pos.z,
                         f32(voxel.z)     - local_pos.z, dir.z < 0.0) / dir.z;
    }
    t_max = max(t_max, vec3<f32>(0.0));

    // Pre-skip: starting voxel may be a model block (same self-shadow fix as above).
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
            let ro        = local_pos - vf;
            for (var ci = 0u; ci < c_count; ci++) {
                let b    = c_start + ci * 6u;
                let bmin = vec3<f32>(model_shadow_cubes[b],     model_shadow_cubes[b+2u], model_shadow_cubes[b+4u]);
                let bmax = vec3<f32>(model_shadow_cubes[b+1u],  model_shadow_cubes[b+3u], model_shadow_cubes[b+5u]);
                if (all(ro >= bmin) && all(ro <= bmax)) { continue; }
                if (ray_seg_hits_aabb(ro, dir, t_seg, bmin, bmax)) {
                    return vec3<f32>(0.0);
                }
            }
        }
    }

    var t_prev = 0.0;
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
    } else if (t_max.y <= t_max.z) {
        voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
    } else {
        voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
    }

    let max_steps = min(i32(dist), 64);
    for (var i = 0; i < max_steps; i++) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return transmission;
        }
        // Skip the emitting block itself — prevents self-shadowing from the source block.
        if (source_vox.w != 0 && all(voxel == source_vox_local)) {
            if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
                voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
            } else if (t_max.y <= t_max.z) {
                voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
            } else {
                voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
            }
            continue;
        }

        let uv  = vec3<f32>(f32(voxel.x) + 0.5, f32(voxel.y) + 0.5, f32(voxel.z) + 0.5) / vsize;
        let rad = textureSampleLevel(voxel_radiance, voxel_sampler, uv, 0.0);
        if (rad.a > 0.5 && dot(rad.rgb, rad.rgb) > 2.0) {
            if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
                voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
            } else if (t_max.y <= t_max.z) {
                voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
            } else {
                voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
            }
            continue;
        }
        let data = textureSampleLevel(voxel_data, voxel_sampler, uv, 0.0);
        let opacity = data.a;
        if (opacity > 0.0 && opacity < FLUID_ALPHA_MAX) {
            // Fluid voxel: absorb light gradually, no hard shadow.
            transmission = transmission * WATER_ABSORPTION;
            if (dot(transmission, transmission) < 0.001) {
                return vec3<f32>(0.0);
            }
        } else if (opacity >= GLASS_ALPHA_MIN && opacity <= GLASS_ALPHA_MAX) {
            let tint = textureSampleLevel(voxel_tint, voxel_sampler, uv, 0.0);
            let op   = tint.a;
            let mult = mix(vec3<f32>(1.0), tint.rgb, op) * (1.0 - 0.85 * op);
            transmission = transmission * mult;
            if (dot(transmission, transmission) < 0.0005) {
                return vec3<f32>(0.0);
            }
        } else if (opacity > 0.85) {
            return vec3<f32>(0.0);
        } else if (opacity > 0.2 && opacity < 0.5) {
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
            voxel.x += step_x; t_prev = t_max.x; t_max.x += t_delta.x;
        } else if (t_max.y <= t_max.z) {
            voxel.y += step_y; t_prev = t_max.y; t_max.y += t_delta.y;
        } else {
            voxel.z += step_z; t_prev = t_max.z; t_max.z += t_delta.z;
        }
    }
    return transmission;
}

// ---------------------------------------------------------------------------
// GI: sample propagated block light from radiance volume
// ---------------------------------------------------------------------------

fn sample_block_light(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // Sample slightly in front of the surface to avoid fetching the solid voxel
    let uv = world_to_voxel_uv(world_pos + normal * 0.6);
    if (!is_in_volume(uv)) { return vec3<f32>(0.0); }
    return textureSampleLevel(voxel_radiance, voxel_sampler, uv, 0.0).rgb;
}

// ---------------------------------------------------------------------------
// Point / spot light PBR evaluation
// ---------------------------------------------------------------------------

fn eval_point_light(
    N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>,
    light: PointLight,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let to_light = light.pos_range.xyz - world_pos;
    let dist     = length(to_light);
    let range    = light.pos_range.w;
    if (dist > range) { return vec3<f32>(0.0); }
    let L         = to_light / max(dist, 0.001);
    let t         = clamp(dist / range, 0.0, 1.0);
    let atten     = (1.0 - t) * (1.0 - t);
    let radiance  = light.color_intensity.rgb * light.color_intensity.w * atten;
    let shadow    = voxel_shadow_to_point(world_pos + N * u.shadow_params.y, light.pos_range.xyz, light.source_voxel);
    return compute_pbr_light(N, V, L, radiance * shadow, albedo, f0, roughness, metalness);
}

fn eval_spot_light(
    N: vec3<f32>, V: vec3<f32>, world_pos: vec3<f32>,
    light: SpotLight,
    albedo: vec3<f32>, f0: vec3<f32>,
    roughness: f32, metalness: f32,
) -> vec3<f32> {
    let to_light   = light.pos_range.xyz - world_pos;
    let dist       = length(to_light);
    let range      = light.pos_range.w;
    if (dist > range) { return vec3<f32>(0.0); }
    let L          = to_light / max(dist, 0.001);
    let spot_dir   = normalize(light.dir_inner.xyz);
    let cos_angle  = dot(-L, spot_dir);
    let inner_cos  = light.dir_inner.w;
    let outer_cos  = light.outer_pad.x;
    let spot_f     = clamp((cos_angle - outer_cos) / max(inner_cos - outer_cos, 0.001), 0.0, 1.0);
    if (spot_f <= 0.0) { return vec3<f32>(0.0); }
    let t          = clamp(dist / range, 0.0, 1.0);
    let atten      = (1.0 - t) * (1.0 - t) * spot_f;
    let radiance   = light.color_intensity.rgb * light.color_intensity.w * atten;
    let shadow     = voxel_shadow_to_point(world_pos, light.pos_range.xyz, vec4<i32>(0, 0, 0, 0));
    return compute_pbr_light(N, V, L, radiance * shadow, albedo, f0, roughness, metalness);
}

// ===== Fragment shader =======================================================

// Tunable display parameters — match pbr_vct.wgsl for visual consistency
const EXPOSURE:        f32 = 1.5;
const SATURATION:      f32 = 0.9;
const CONTRAST_POWER:  f32 = 1.0;

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
    let tex = textureSample(albedo_tex, albedo_samp, in.uv);
    // Alpha cutout (skin overlay layers)
    if (tex.a < 0.05) { discard; }

    let albedo    = tex.rgb;
    let N         = normalize(in.normal);
    let V         = normalize(u.camera_pos.xyz - in.world_pos);

    // Default material: skin / cloth (non-metallic, medium roughness)
    let roughness = 0.75;
    let metalness = 0.0;
    let f0        = vec3<f32>(0.04); // non-metallic base reflectance

    var lo = vec3<f32>(0.0);
    // Shadow parameters — used by both sun and moon
    let shadow_on  = u.shadow_params.x > 0.5;
    let shadow_off = 0.05;

    // 1. Sun directional light + voxel shadow (coloured by any glass in the way)
    let sun_intensity = u.sun_direction.w;
    if (sun_intensity > 0.001) {
        let sun_L      = normalize(u.sun_direction.xyz);
        let sun_rad    = u.sun_color.rgb * sun_intensity;
        let sun_shadow = select(vec3<f32>(1.0), voxel_shadow_directional(in.world_pos + N * shadow_off, sun_L), shadow_on);
        lo += compute_pbr_light(N, V, sun_L, sun_rad * sun_shadow, albedo, f0, roughness, metalness);
    }

    // 2. Moon directional light + voxel shadow
    let moon_intensity = u.moon_direction.w;
    if (moon_intensity > 0.001) {
        let moon_L      = normalize(u.moon_direction.xyz);
        let moon_rad    = u.moon_color.rgb * moon_intensity;
        let moon_shadow = select(vec3<f32>(1.0), voxel_shadow_directional(in.world_pos + N * shadow_off, moon_L), shadow_on);
        lo += compute_pbr_light(N, V, moon_L, moon_rad * moon_shadow, albedo, f0, roughness, metalness);
    }

    // 3. Block light from VCT radiance volume (point / spot / emissive blocks)
    let block_light = sample_block_light(in.world_pos, N);
    let gi_intensity = vct.inv_size.w;
    lo += block_light * albedo * gi_intensity;

    // 4. Dynamic point lights
    let n_point = vct.light_counts.x;
    for (var i = 0u; i < n_point; i++) {
        lo += eval_point_light(N, V, in.world_pos, point_lights[i], albedo, f0, roughness, metalness);
    }

    // 5. Dynamic spot lights
    let n_spot = vct.light_counts.y;
    for (var i = 0u; i < n_spot; i++) {
        lo += eval_spot_light(N, V, in.world_pos, spot_lights[i], albedo, f0, roughness, metalness);
    }

    // 6. Ambient (minimum floor light — no vertex AO for entities)
    let ambient_min = u.ambient_color.w;
    let ambient     = max(u.ambient_color.xyz, vec3<f32>(ambient_min)) * albedo * 0.35;
    lo += ambient;

    // 7. Exposure + ACES tone mapping + saturation/contrast (matches pbr_vct.wgsl)
    var color = lo * EXPOSURE;

    let aces_a = vec3<f32>(2.51);
    let aces_b = vec3<f32>(0.03);
    let aces_c = vec3<f32>(2.43);
    let aces_d = vec3<f32>(0.59);
    let aces_e = vec3<f32>(0.14);
    color = clamp(
        (color * (aces_a * color + aces_b)) / (color * (aces_c * color + aces_d) + aces_e),
        vec3<f32>(0.0), vec3<f32>(1.0),
    );

    // Saturation
    let lum  = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    color    = mix(vec3<f32>(lum), color, SATURATION);

    // Contrast
    color = pow(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(CONTRAST_POWER));

    return vec4<f32>(color, tex.a);
}
