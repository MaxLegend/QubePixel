// ===== Uniform layout (272 bytes = 68 floats) =====
struct Uniforms {
    view_proj:        mat4x4<f32>,   // [0..16]
    camera_pos:       vec4<f32>,     // [16..20]
    sun_direction:    vec4<f32>,     // [20..24]
    sun_color:        vec4<f32>,     // [24..28]
    moon_direction:   vec4<f32>,     // [28..32]
    moon_color:       vec4<f32>,     // [32..36]
    ambient_color:    vec4<f32>,     // [36..40]
    shadow_params:    vec4<f32>,     // [40..44]
    ssao_params:      vec4<f32>,     // [44..48]
    time_params:      vec4<f32>,     // [48..52]
    shadow_view_proj: mat4x4<f32>,   // [52..68]
}; // <--- Точка с запятой обязательна!

// ===== Radiance Cascades GI structs =====
// One entry per cascade level (C0..C4). AlignOf = 16 (vec4 member) so the
// array satisfies WGSL uniform-buffer array-element alignment requirements.
struct GiCascadeEntry {
    origin_and_spacing: vec4<f32>, // .xyz = grid world origin, .w = probe spacing
    counts:             vec2<u32>, // .x = ray_count,  .y = grid_size per axis
    _pad:               vec2<u32>,
    fractional_offset:  vec3<f32>, // sub-block offset: compensates for snapped grid vs camera
    _pad2:              f32,
};

struct GiSampleParams {
    cascades:         array<GiCascadeEntry, 5>, // [0..160]  5 × 32B
    gi_intensity:     f32,                      // [160]
    bounce_intensity: f32,                      // [164]
    _pad0:            f32,                      // [168]
    _pad1:            f32,                      // [172]
};

struct RadianceInterval {
    radiance_in:      vec4<f32>,
    radiance_out:     vec4<f32>,
    direction_length: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex:      texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler:  sampler;
@group(0) @binding(3) var voxel_tex:   texture_3d<u32>;   // Voxel texture for ray-traced shadows
@group(0) @binding(4) var block_lut:   texture_2d<f32>;   // Block properties LUT (opacity/emission)
@group(0) @binding(5) var normal_tex:     texture_2d<f32>;
@group(0) @binding(6) var normal_sampler: sampler;

@group(1) @binding(0) var<uniform>       gi_params:       GiSampleParams;
@group(1) @binding(1) var<storage, read> gi_c0_intervals: array<RadianceInterval>;
@group(1) @binding(2) var<storage, read> gi_c1_intervals: array<RadianceInterval>;
@group(1) @binding(3) var<storage, read> gi_c2_intervals: array<RadianceInterval>;
@group(1) @binding(4) var<storage, read> gi_c3_intervals: array<RadianceInterval>;
@group(1) @binding(5) var<storage, read> gi_c4_intervals: array<RadianceInterval>;

// Packed vertex input — 52 bytes per vertex.
// Snorm8x4 and Unorm8x4 are decoded to vec4<f32> by the hardware before the shader sees them.
struct VertexInput {
    @location(0) position:  vec3<f32>,  // Float32x3
    @location(1) normal:    vec4<f32>,  // Snorm8x4  → .xyz = normal  [−1..1], .w = unused
    @location(2) color_ao:  vec4<f32>,  // Unorm8x4  → .xyz = albedo  [0..1],  .w = AO [0..1]
    @location(3) texcoord:  vec2<f32>,  // Float32x2
    @location(4) material:  vec4<f32>,  // Unorm8x4  → .x = roughness, .y = metalness
    @location(5) emission:  vec4<f32>,  // Float32x4 (unchanged)
    @location(6) tangent:   vec4<f32>,  // Snorm8x4  → .xyz = tangent [−1..1], .w = unused
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

// ---------------------------------------------------------------------------
// GI helpers: multi-cascade sampling
// ---------------------------------------------------------------------------

/// Fetch one RadianceInterval from the correct cascade storage buffer.
/// Returns a zeroed interval (direction_length.w == 0) on out-of-bounds.
fn fetch_gi_interval(cascade: u32, idx: u32) -> RadianceInterval {
    var iv: RadianceInterval;
    switch (cascade) {
        case 0u: {
            if (idx < arrayLength(&gi_c0_intervals)) { iv = gi_c0_intervals[idx]; }
        }
        case 1u: {
            if (idx < arrayLength(&gi_c1_intervals)) { iv = gi_c1_intervals[idx]; }
        }
        case 2u: {
            if (idx < arrayLength(&gi_c2_intervals)) { iv = gi_c2_intervals[idx]; }
        }
        case 3u: {
            if (idx < arrayLength(&gi_c3_intervals)) { iv = gi_c3_intervals[idx]; }
        }
        default: {
            if (idx < arrayLength(&gi_c4_intervals)) { iv = gi_c4_intervals[idx]; }
        }
    }
    return iv;
}

/// Sample indirect GI from one cascade level using trilinear probe interpolation.
///
/// Returns vec4: .xyz = accumulated GI radiance, .w = sky visibility (0=occluded, 1=open).
fn sample_gi_cascade(
    cascade_level: u32,
    world_pos:     vec3<f32>,
    normal:        vec3<f32>,
    entry:         GiCascadeEntry,
    no_self_gi:    bool,
    bounce_int:    f32,
) -> vec4<f32> {
    let grid_origin  = entry.origin_and_spacing.xyz;
    let spacing      = entry.origin_and_spacing.w;
    let ray_count    = entry.counts.x;
    let gs_u         = entry.counts.y;
    let gs_i         = i32(gs_u);
    let grid_size_f  = f32(gs_u);

    // Probes are world-anchored: grid_origin = floor(cam/sp)*sp - half_extent*sp + 0.5*sp
    // The lookup is purely world-relative — no camera-dependent correction.
    // fractional_offset is always zero; kept in the struct for future use.
    let rel       = world_pos - grid_origin;
    let grid_f    = rel / spacing;
    let gc_clamped = clamp(grid_f, vec3<f32>(0.0), vec3<f32>(grid_size_f - 1.0001));
    let gc_floor  = vec3<i32>(i32(gc_clamped.x), i32(gc_clamped.y), i32(gc_clamped.z));
    let frac      = gc_clamped - vec3<f32>(f32(gc_floor.x), f32(gc_floor.y), f32(gc_floor.z));

    var gi           = vec3<f32>(0.0);
    var weight_total = 0.0;
    var sky_vis_sum  = 0.0;

    for (var dz: i32 = 0; dz <= 1; dz = dz + 1) {
    for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
        let px = gc_floor.x + dx;
        let py = gc_floor.y + dy;
        let pz = gc_floor.z + dz;
        if (px < 0 || px >= gs_i || py < 0 || py >= gs_i || pz < 0 || pz >= gs_i) {
            continue;
        }

        let wx = select(1.0 - frac.x, frac.x, dx == 1);
        let wy = select(1.0 - frac.y, frac.y, dy == 1);
        let wz = select(1.0 - frac.z, frac.z, dz == 1);
        let trilinear_w = wx * wy * wz;
        if (trilinear_w < 0.0001) { continue; }

        let probe_idx   = u32(px) + u32(py) * gs_u + u32(pz) * gs_u * gs_u;
        let probe_start = probe_idx * ray_count;

        var probe_gi      = vec3<f32>(0.0);
        var probe_gi_w    = 0.0;
        var probe_sky_vis = 0.0;

        // Stride sampling: always read at most GI_MAX_SAMPLES rays per probe.
        // C0=6 rays (stride=1, all), C1=24 (stride=4, 6 samples),
        // C2=54 (stride=9), C3=96 (stride=16), C4=150 (stride=25).
        // Reduces worst-case storage-buffer reads from 8×150=1200 to 8×6=48
        // per pixel — a 25× cache-pressure reduction for distant cascades.
        let GI_MAX_SAMPLES = 6u;
        let gi_stride = max(1u, ray_count / GI_MAX_SAMPLES);
        // solid_w accounts for the full sphere even when sub-sampling.
        let solid_w_full = 4.0 * PI / f32(ray_count);
        let solid_w_stride = solid_w_full * f32(gi_stride);

        for (var r = 0u; r < ray_count; r = r + gi_stride) {
            let interval = fetch_gi_interval(cascade_level, probe_start + r);
            if (interval.direction_length.w < 0.001) { continue; }

            let ray_dir = normalize(interval.direction_length.xyz);
            let ndotl   = dot(normal, ray_dir);
            if (ndotl <= 0.0) { continue; }

            let cos_w   = ndotl;
            let solid_w = solid_w_stride;

            let tau      = clamp(interval.radiance_out.w, 0.0, 80.0);
            let trans    = exp(-tau);
            let em_tau   = clamp(interval.radiance_in.a, 0.0, 80.0);
            let em_trans = exp(-em_tau);

            // Emission from endpoint (direct hit on emissive surface)
            let out_em_contrib = interval.radiance_out.rgb * em_trans * cos_w * solid_w * bounce_int;
            // Emission propagated through merge chain (in-scattered radiance from
            // emissive blocks that this ray passed near, not just hit directly).
            // This is what makes emissive blocks act as point lights — their
            // emission spreads to neighbouring probes via the merge chain.
            let in_em_contrib = interval.radiance_in.rgb * cos_w * solid_w;
            if (!no_self_gi) {
                probe_gi = probe_gi + out_em_contrib + in_em_contrib;
            }
            probe_gi_w    = probe_gi_w    + cos_w * solid_w;
            probe_sky_vis = probe_sky_vis + trans * cos_w * solid_w;
        }

        if (probe_gi_w > 0.001) {
            gi          = gi          + trilinear_w * (probe_gi      / probe_gi_w);
            sky_vis_sum = sky_vis_sum + trilinear_w * (probe_sky_vis / probe_gi_w);
            weight_total = weight_total + trilinear_w;
        }
    }}}

    if (weight_total > 0.001) {
        gi          = gi          / weight_total;
        sky_vis_sum = sky_vis_sum / weight_total;
    }

    return vec4<f32>(gi, sky_vis_sum);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = u.view_proj * vec4<f32>(input.position, 1.0);
    // Unpack compressed fields — hardware delivers them as normalized vec4<f32>
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
const PI: f32 = 3.14159265;
// Debug modes:
// 0=off
// 1=cascade coverage (green=C0, cyan=C1, yellow=C2, orange=C3, red=C4, dark red=outside)
// 2=sky visibility (blue=occluded, green=open sky)
// 3=probe grid cell parity
// 4=pre-tonemap heatmap (luminance → blue/dark → green/bright → red/overexposed)
// 5=raw GI contribution (brightness = GI magnitude, tinted by GI color)
// 6=sun direct light only (no GI, no ambient, no emission)
// 7=component breakdown: R=sun_direct, G=GI, B=ambient
// 8=shadow value only (white=lit, black=shadow)
const GI_DEBUG_MODE: u32 = 0u;

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

/// Trace a ray from world_pos toward the sun through the voxel texture.
/// Returns 0.0 = fully in shadow, 1.0 = fully lit.
/// Uses DDA (Amanatides-Woo) for efficient voxel grid traversal.
/// shadow_params.xyz = voxel texture world origin, shadow_params.w = texture size.
fn voxel_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    let voxel_origin = u.shadow_params.xyz;
    let vsize = u.shadow_params.w; // 256.0
    let local_pos = world_pos - voxel_origin;

    // If fragment is outside voxel texture bounds, assume lit
    if (local_pos.x < 0.0 || local_pos.y < 0.0 || local_pos.z < 0.0 ||
        local_pos.x >= vsize || local_pos.y >= vsize || local_pos.z >= vsize) {
        return 1.0;
    }

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    let dir = normalize(light_dir);
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
        t_max.x = select(
            f32(voxel.x + 1) - local_pos.x,
            f32(voxel.x) - local_pos.x,
            dir.x < 0.0
        ) / dir.x;
    } else { t_max.x = 1.0e10; }
    if (abs(dir.y) > 0.0001) {
        t_max.y = select(
            f32(voxel.y + 1) - local_pos.y,
            f32(voxel.y) - local_pos.y,
            dir.y < 0.0
        ) / dir.y;
    } else { t_max.y = 1.0e10; }
    if (abs(dir.z) > 0.0001) {
        t_max.z = select(
            f32(voxel.z + 1) - local_pos.z,
            f32(voxel.z) - local_pos.z,
            dir.z < 0.0
        ) / dir.z;
    } else { t_max.z = 1.0e10; }
    t_max = max(t_max, vec3<f32>(0.0));

    // Skip the first voxel (contains the surface) to avoid self-shadowing
    if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
        voxel.x = voxel.x + step_x;
        t_max.x = t_max.x + t_delta.x;
    } else if (t_max.y <= t_max.z) {
        voxel.y = voxel.y + step_y;
        t_max.y = t_max.y + t_delta.y;
    } else {
        voxel.z = voxel.z + step_z;
        t_max.z = t_max.z + t_delta.z;
    }

    let max_steps: u32 = 256u;
    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
            voxel.x >= sz || voxel.y >= sz || voxel.z >= sz) {
            return 1.0; // Exited texture — lit by sun
        }

        let block_id = textureLoad(voxel_tex, voxel, 0).r;
        // 0 = air, 0xFFFF = unloaded → no shadow
        if (block_id != 0u && block_id != 0xFFFFu) {
            // Look up block_lut: .rgba = (opacity, emission_r, emission_g, emission_b).
            // Emissive blocks are light sources — they don't cast directional shadows.
            let lut = textureLoad(block_lut, vec2<i32>(i32(block_id), 0), 0);
            let is_emissive = (lut.g + lut.b + lut.a) > 0.001;
            if (!is_emissive) {
                return 0.0; // Hit opaque non-emissive block — in shadow
            }
        }

        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            voxel.x = voxel.x + step_x;
            t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            voxel.y = voxel.y + step_y;
            t_max.y = t_max.y + t_delta.y;
        } else {
            voxel.z = voxel.z + step_z;
            t_max.z = t_max.z + t_delta.z;
        }
    }

    return 1.0; // Didn't hit anything — lit
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(atlas_tex, atlas_sampler, input.v_texcoord).rgb;
    let albedo    = input.v_color * tex_color;

    // ---- Normal mapping: TBN matrix from tangent + geometric normal ----
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
  //  let sky_light  = input.v_sky_light;

    let f0 = mix(vec3<f32>(0.04), albedo, metalness);

    var lo = vec3<f32>(0.0);
    var debug_shadow: f32 = 1.0; // for debug overlay

    // --- Sun light ---
    let sun_intensity = u.sun_direction.w;
    if (sun_intensity > 0.001) {
        let sun_L       = normalize(u.sun_direction.xyz);
        let sun_radiance = u.sun_color.rgb * sun_intensity;

        // Voxel ray-traced shadow: DDA march through the voxel texture.
        // Offset starting position along normal to avoid self-shadowing artifacts.
        debug_shadow = voxel_shadow(input.v_world_pos + N * 0.01, sun_L);
        let pbr_light = compute_pbr_light(N, V, sun_L, sun_radiance, albedo, f0, roughness, metalness);
        lo += pbr_light * debug_shadow;
    }

    // --- Moon light ---
    let moon_intensity = u.moon_direction.w;
    if (moon_intensity > 0.001) {
        let moon_L       = normalize(u.moon_direction.xyz);
        let moon_radiance = u.moon_color.rgb * moon_intensity;
        lo += compute_pbr_light(N, V, moon_L, moon_radiance, albedo, f0, roughness, metalness);
    }

     // --- Emission ---
     // v_emission.w < 0 means no_self_gi=true: block opts out of GI self-coloring.
     // Use abs() for the actual displayed emission so the block still glows visually.
     let raw_intensity = input.v_emission.w;
     let display_intensity = min(abs(raw_intensity), 0.8);
     let no_self_gi = raw_intensity < 0.0;
     // Desaturate emission for surface display: real glowing surfaces appear whiter
     // at their face (not deeply colored). Prevents over-saturation of warm emitters.
     let em_lum = dot(input.v_emission.rgb, vec3<f32>(0.299, 0.587, 0.114));
     let emission = mix(input.v_emission.rgb, vec3<f32>(em_lum), 0.45) * display_intensity;

     // --- Base color: direct lighting + emission (ambient added after GI) ---
     var color = lo + emission;

        // --- Radiance Cascades GI — multi-cascade sampling ---
        // For each fragment we pick the finest cascade whose probe grid spatially
        // covers the fragment. Cascades are tried finest→coarsest (C0..C4).
        // At grid boundaries we blend smoothly into the next coarser cascade so
        // there are no hard seams. Together the 5 cascade grids cover the full
        // rendered scene (C4 reaches ±64 blocks from camera).
        var gi_sky_vis = 0.0;
        var gi_debug_in_grid = false;
        var gi_debug_fade    = 1.0;
        var gi_debug_level   = 5u; // 5 = outside all cascades

    // Debug: save individual lighting components for overlay modes 4-8
    var debug_sun_direct = lo;       // sun+moon direct (before GI/ambient)
    var debug_gi         = vec3<f32>(0.0);
    var debug_ambient    = vec3<f32>(0.0);

    {
        var gi_acc      = vec3<f32>(0.0);
        var sky_acc     = 0.0;
        var remaining_w = 1.0;

        for (var level = 0u; level < 5u; level = level + 1u) {
            if (remaining_w < 0.001) { break; }

            let entry       = gi_params.cascades[level];
            let grid_origin = entry.origin_and_spacing.xyz;
            let spacing     = entry.origin_and_spacing.w;
            let grid_size_f = f32(entry.counts.y);

            // Skip cascades with no probes (e.g. uninitialized)
            if (entry.counts.x == 0u || entry.counts.y == 0u || spacing < 0.001) { continue; }

            // Normalised position in probe grid [0 .. grid_size] — world-relative.
            let grid_f = (input.v_world_pos - grid_origin) / spacing;

            // Is this fragment inside the grid (with 0.5-cell margin for trilinear)?
            if (any(grid_f < vec3<f32>(-0.5)) || any(grid_f > vec3<f32>(grid_size_f - 0.5))) {
                continue; // outside — try next coarser cascade
            }

            // Boundary fade: 1.0 at grid centre, 0.0 at outermost 15% of grid.
            // Prevents hard seams when a fragment is near the edge of a cascade.
            let half        = grid_size_f * 0.5;
            let centered    = abs(grid_f - vec3<f32>(half)) / vec3<f32>(half);
            let max_centered = max(centered.x, max(centered.y, centered.z));
            let fade_inner  = 0.85;
            var fade = 1.0 - clamp((max_centered - fade_inner) / (1.0 - fade_inner), 0.0, 1.0);
            fade = fade * fade; // quadratic ease

            let w = min(fade, remaining_w);
            if (w < 0.001) { continue; }

            let result = sample_gi_cascade(level, input.v_world_pos, N, entry, no_self_gi, gi_params.bounce_intensity);
            gi_acc  = gi_acc  + result.xyz * w;
            sky_acc = sky_acc + result.w   * w;
            remaining_w = remaining_w - w;

            // Record finest cascade used (for debug overlay)
            if (gi_debug_level == 5u) {
                gi_debug_level   = level;
                gi_debug_in_grid = true;
                gi_debug_fade    = fade;
            }
        }

        let gi_raw = clamp(gi_acc * gi_params.gi_intensity, vec3<f32>(0.0), vec3<f32>(2.0));
        // GI is irradiance (W/m²). Modulate by albedo so dark/colored surfaces
        // receive proportionally less GI — prevents yellow emission from washing
        // out surface textures as a uniform opaque overlay.
        let gi = gi_raw * albedo;
        debug_gi = gi;
        gi_sky_vis = clamp(sky_acc, 0.0, 1.0);
        color = color + gi;
    }

        // --- Ambient (inversely modulated by GI sky visibility) ---
        // GI already provides sky light through the merge chain (C4 sky radiance
        // propagates inward). Using gi_sky_vis proportionally would double-count
        // sky illumination. Instead, ambient is HIGH where GI sky is LOW
        // (underground — GI can't reach) and LOW where GI sky is HIGH (surface
        // — GI already provides ample sky light).
        // Values: underground max 0.12 (dim but visible), surface 0.04 (subtle fill).
        let ambient_floor = 0.04;
        let ambient_ceiling = 0.12;
        let gi_ambient_factor = clamp(gi_sky_vis, 0.0, 1.0);
        let ambient = u.ambient_color.rgb * albedo * vertex_ao
                    * mix(ambient_ceiling, ambient_floor, gi_ambient_factor);
        debug_ambient = ambient;
        color = color + ambient;
    // Exposure
    color = color * 0.95;

    // ACES filmic tone mapping (Hill approximation)
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
// Гамма-коррекция (linear → sRGB)
//color = pow(color, vec3<f32>(1.0 / 0.99));   // ← СТАНДАРТ 2.2

// Контраст (around 0.5 midpoint)
//let contrast: f32 = 1.05;     // 1.0=нейтральный, >1=жёстче, <1=мягче
//color = (color - 0.5) * contrast + 0.5;

// Насыщенность (lerp с grayscale)
//let saturation: f32 = 1.3;   // 1.0=нейтральная, 0.0=B/W, >1=ярче
//let gray = dot(color, vec3<f32>(0.299, 0.587, 0.114));
//color = mix(vec3<f32>(gray), color, saturation);
    // --- Debug overlay ---
    if (GI_DEBUG_MODE == 1u) {
        // Cascade coverage: colour shows which cascade level covers this fragment.
        // C0=green (finest), C1=cyan, C2=yellow, C3=orange, C4=red, none=dark red.
        var dbg: vec3<f32>;
        if (!gi_debug_in_grid) {
            dbg = vec3<f32>(0.5, 0.0, 0.0);                    // dark red = outside all
        } else {
            switch (gi_debug_level) {
                case 0u: { dbg = vec3<f32>(0.0, 0.6, 0.1); }  // green  = C0 (finest)
                case 1u: { dbg = vec3<f32>(0.0, 0.5, 0.5); }  // cyan   = C1
                case 2u: { dbg = vec3<f32>(0.5, 0.5, 0.0); }  // yellow = C2
                case 3u: { dbg = vec3<f32>(0.6, 0.3, 0.0); }  // orange = C3
                default: { dbg = vec3<f32>(0.6, 0.0, 0.0); }  // red    = C4 (coarsest)
            }
            if (gi_debug_fade < 0.5) { dbg = dbg * 0.55; }    // dim at cascade boundaries
        }
        color = mix(color, dbg, 0.55);
    } else if (GI_DEBUG_MODE == 2u) {
        // Sky visibility: blue=fully occluded, green=open sky
        let sky = clamp(gi_sky_vis * 1.5, 0.0, 1.0);
        color = mix(color, vec3<f32>(0.0, sky, 1.0 - sky), 0.65);
    } else if (GI_DEBUG_MODE == 3u) {
        // Probe grid cell parity — reveals probe tiling artefacts (uses C0 grid)
        let c0          = gi_params.cascades[0u];
        let c0_origin   = c0.origin_and_spacing.xyz;
        let c0_spacing  = c0.origin_and_spacing.w;
        let rel3   = (input.v_world_pos - c0_origin) / c0_spacing;
        let parity = (i32(floor(rel3.x)) + i32(floor(rel3.z))) % 2;
        let tile_col = select(vec3<f32>(0.15, 0.15, 0.8), vec3<f32>(0.8, 0.8, 0.15), parity == 0);
        color = mix(color, tile_col, 0.55);
    } else if (GI_DEBUG_MODE == 4u) {
        // Pre-tonemap heatmap: luminance mapped to blue→green→red
        // Shows raw HDR values BEFORE tone mapping — reveals over/underexposure
        let lum = dot(color, vec3<f32>(0.299, 0.587, 0.114));
        let t = clamp(lum, 0.0, 2.0) / 2.0; // normalize to [0..1] range (2.0 = very bright)
        var heat: vec3<f32>;
        if (t < 0.5) {
            heat = mix(vec3<f32>(0.0, 0.0, 0.5), vec3<f32>(0.0, 1.0, 0.0), t * 2.0);
        } else {
            heat = mix(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
        }
        color = heat;
    } else if (GI_DEBUG_MODE == 5u) {
        // Raw GI contribution — brightness = GI magnitude, tinted by GI color
        let gi_mag = length(debug_gi);
        let gi_norm = debug_gi / max(gi_mag, 0.001);
        let t = clamp(gi_mag * 5.0, 0.0, 1.0); // scale up for visibility
        color = gi_norm * t;
    } else if (GI_DEBUG_MODE == 6u) {
        // Sun direct light only (no GI, no ambient, no emission)
        color = debug_sun_direct;
        // Apply tone mapping so it's visible
        color = clamp(
            (color * (aces_a * color + aces_b)) / (color * (aces_c * color + aces_d) + aces_e),
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
    } else if (GI_DEBUG_MODE == 7u) {
        // Component breakdown: R=sun_direct magnitude, G=GI magnitude, B=ambient magnitude
        let sun_lum = dot(debug_sun_direct, vec3<f32>(0.299, 0.587, 0.114));
        let gi_lum  = dot(debug_gi, vec3<f32>(0.299, 0.587, 0.114));
        let amb_lum = dot(debug_ambient, vec3<f32>(0.299, 0.587, 0.114));
        color = vec3<f32>(
            clamp(sun_lum * 2.0, 0.0, 1.0),
            clamp(gi_lum * 5.0, 0.0, 1.0),
            clamp(amb_lum * 10.0, 0.0, 1.0)
        );
    } else if (GI_DEBUG_MODE == 8u) {
        // Shadow value only: white=lit, black=in shadow
        color = vec3<f32>(debug_shadow);
    }

    return vec4<f32>(color, 1.0);
}
