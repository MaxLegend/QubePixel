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
struct GiSampleParams {
    grid_origin_x:    f32,
    grid_origin_y:    f32,
    grid_origin_z:    f32,
    grid_spacing:     f32,
    ray_count:        f32,
    grid_size:        f32,
    gi_intensity:     f32,
    bounce_intensity: f32,
};

struct RadianceInterval {
    radiance_in:      vec4<f32>,
    radiance_out:     vec4<f32>,
    direction_length: vec4<f32>,
};
// GI bindings: @group(1) — params + C0 cascade storage @group(1) @binding(0) var gi_params: GiSampleParams; @group(1) @binding(1) var<storage, read> gi_c0_intervals: array;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var atlas_tex:      texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler:  sampler;
@group(0) @binding(3) var shadow_map:     texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;


@group(1) @binding(0) var<uniform> gi_params:          GiSampleParams;
@group(1) @binding(1) var<storage, read> gi_c0_intervals: array<RadianceInterval>;

struct VertexInput {
    @location(0) position:  vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) color:     vec3<f32>,
    @location(3) texcoord:  vec2<f32>,
    @location(4) ao:        f32,
    @location(5) material:  vec2<f32>,
    @location(6) emission:  vec4<f32>,
    @location(7) sky_light: f32,
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
    @location(7)       v_shadow_pos: vec4<f32>,
    @location(8)       v_sky_light:  f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos     = u.view_proj * vec4<f32>(input.position, 1.0);
    out.v_normal     = input.normal;
    out.v_color      = input.color;
    out.v_world_pos  = input.position;
    out.v_texcoord   = input.texcoord;
    out.v_ao         = input.ao;
    out.v_material   = input.material;
    out.v_emission   = input.emission;
    out.v_shadow_pos = u.shadow_view_proj * vec4<f32>(input.position, 1.0);
    out.v_sky_light  = input.sky_light;
    return out;
}

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

// ===== Radiance Cascades GI sampling =====
//fn sample_radiance_cascades(
//    world_pos: vec3<f32>,
//    normal: vec3<f32>,
//) -> vec3<f32> {
//    let grid_origin = vec3<f32>(gi_params.grid_origin_x, gi_params.grid_origin_y, gi_params.grid_origin_z);
//    let ray_count = u32(gi_params.ray_count);
//    let grid_size_f = gi_params.grid_size;
//
//    if (ray_count == 0u) { return vec3<f32>(0.0); }
//
//    // Nearest probe
//    let rel = world_pos - grid_origin;
//    let grid_f = rel / gi_params.grid_spacing;
//    let clamped = clamp(grid_f, vec3<f32>(0.0), vec3<f32>(grid_size_f - 1.0));
//    let gc = vec3<u32>(u32(clamped.x), u32(clamped.y), u32(clamped.z));
//    let gs = u32(grid_size_f);
//    let probe_flat = gc.x + gc.y * gs + gc.z * gs * gs;
//    let probe_start = probe_flat * ray_count;
//
//    var gi = vec3<f32>(0.0);
//    var w_sum = 0.0;
//    let sa = 4.0 * PI / f32(ray_count);
//
//    for (var r = 0u; r < ray_count; r = r + 1u) {
//        let idx = probe_start + r;
//        if (idx >= arrayLength(&gi_c0_intervals)) { break; }
//        let iv = gi_c0_intervals[idx];
//        if (iv.direction_length.w < 0.001) { continue; }
//        let rd = normalize(iv.direction_length.xyz);
//        let ndl = dot(normal, rd);
//        if (ndl <= 0.0) { continue; }
//        let tau = clamp(iv.radiance_out.w, 0.0, 80.0);
//        let tr = exp(-tau);
//        gi = gi + (iv.radiance_in.rgb + iv.radiance_out.rgb * tr * gi_params.bounce_intensity) * ndl * sa;
//        w_sum = w_sum + ndl * sa;
//    }
//
//    if (w_sum > 0.001) { gi = gi / w_sum; }
//    gi = gi * gi_params.gi_intensity;
//    return gi;
//}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(atlas_tex, atlas_sampler, input.v_texcoord).rgb;
    let albedo    = input.v_color * tex_color;

    let N = normalize(input.v_normal);
    let V = normalize(u.camera_pos.xyz - input.v_world_pos);

    let roughness  = input.v_material.x;
    let metalness  = input.v_material.y;
    let vertex_ao  = input.v_ao;
    let sky_light  = input.v_sky_light;

    let f0 = mix(vec3<f32>(0.04), albedo, metalness);

    var lo = vec3<f32>(0.0);

    // --- Sun light ---
    let sun_intensity = u.sun_direction.w * sky_light;
    if (sun_intensity > 0.001) {
        let sun_L       = normalize(u.sun_direction.xyz);
        let sun_radiance = u.sun_color.rgb * sun_intensity;

        let shadow_pos = input.v_shadow_pos.xyz / input.v_shadow_pos.w;
        let shadow_uv  = vec2<f32>(
            shadow_pos.x * 0.5 + 0.5,
           -shadow_pos.y * 0.5 + 0.5
        );
        var shadow = 1.0;
        if (shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 &&
            shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 &&
            shadow_pos.z <= 1.0 && shadow_pos.z >= 0.0) {
            shadow = textureSampleCompareLevel(
                shadow_map, shadow_sampler,
                shadow_uv, shadow_pos.z - u.shadow_params.x
            );
        }
        let pbr_light = compute_pbr_light(N, V, sun_L, sun_radiance, albedo, f0, roughness, metalness);
        lo += pbr_light * shadow;
    }

    // --- Moon light ---
    let moon_intensity = u.moon_direction.w * sky_light;
    if (moon_intensity > 0.001) {
        let moon_L       = normalize(u.moon_direction.xyz);
        let moon_radiance = u.moon_color.rgb * moon_intensity;
        lo += compute_pbr_light(N, V, moon_L, moon_radiance, albedo, f0, roughness, metalness);
    }

     // --- Emission ---
     let emission = input.v_emission.rgb * input.v_emission.w;

     // --- Base color: direct lighting + emission (ambient added after GI) ---
     var color = lo + emission;

     // --- Radiance Cascades GI + ambient modulation ---
     // GI сэмплирует C0 probe intervals и вычисляет:
     //   gi           — баунс-освещённость (аддитивная)
     //   gi_sky_vis   — средняя пропускание лучей (proxy «видно ли небо»)
     // ambient масштабируется через gi_sky_vis:
     //   открытое небо → gi_sky_vis ≈ 1.0 → ambient ≈ ambient_sky (полный)
     //   пещера       → gi_sky_vis ≈ 0.0 → ambient ≈ 0.03 (минимальный пол)
     var gi_sky_vis = 0.0;

 {
     let gi_grid_origin = vec3<f32>(gi_params.grid_origin_x, gi_params.grid_origin_y, gi_params.grid_origin_z);
     let gi_grid_size = gi_params.grid_size;
     let gi_ray_count = u32(gi_params.ray_count);

     if (gi_ray_count > 0u) {
         let rel = input.v_world_pos - gi_grid_origin;
         let grid_f = rel / gi_params.grid_spacing;
         let gc = clamp(grid_f, vec3<f32>(0.0), vec3<f32>(gi_grid_size - 1.0));
         let gcu = vec3<u32>(u32(gc.x), u32(gc.y), u32(gc.z));
         let probe_idx = gcu.x + gcu.y * u32(gi_grid_size) + gcu.z * u32(gi_grid_size) * u32(gi_grid_size);
         let probe_start = probe_idx * gi_ray_count;

         var gi = vec3<f32>(0.0);
         var gi_weight = 0.0;

         for (var r = 0u; r < gi_ray_count; r = r + 1u) {
             let idx = probe_start + r;
             if (idx >= u32(arrayLength(&gi_c0_intervals))) { break; }
             let interval = gi_c0_intervals[idx];
             if (interval.direction_length.w < 0.001) { continue; }

             let ray_dir = normalize(interval.direction_length.xyz);
             let ndotl = dot(N, ray_dir);
             if (ndotl <= 0.0) { continue; }

             let cos_w = ndotl;
             let solid_w = 4.0 * 3.14159265 / f32(gi_ray_count);
             let tau = clamp(interval.radiance_out.w, 0.0, 80.0);
             let trans = exp(-tau);

             let bounce = interval.radiance_in.rgb * cos_w * solid_w;
             let emContrib = interval.radiance_out.rgb * trans * cos_w * solid_w * gi_params.bounce_intensity;
             gi = gi + bounce + emContrib;
             gi_weight = gi_weight + cos_w * solid_w;

             // Аккумулируем пропускание для оценки видимости неба
             gi_sky_vis = gi_sky_vis + trans * cos_w * solid_w;
         }

         if (gi_weight > 0.001) {
             gi = gi / gi_weight;
             gi_sky_vis = gi_sky_vis / gi_weight;
         }
         gi = gi * gi_params.gi_intensity;

         color = color + gi;
     }
 }

     // --- Ambient (модулируется GI sky visibility) ---
     let underground_factor = 0.15;
     let ambient_floor = 0.03;
     let ambient_sky = mix(underground_factor, 1.0, sky_light);
     let gi_ambient_factor = clamp(gi_sky_vis, 0.0, 1.0);
     let ambient = u.ambient_color.rgb * albedo * vertex_ao
                 * mix(ambient_floor, ambient_sky, gi_ambient_factor);
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

    return vec4<f32>(color, 1.0);
}