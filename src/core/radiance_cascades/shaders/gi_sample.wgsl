// =============================================================================
// QubePixel — GI Sampling Shader (Fragment Helper)
// =============================================================================
//
// This file is meant to be #include'd or embedded into the main terrain
// fragment shader (game_3d_pipeline.wgsl). It is NOT a standalone entry point.
//
// Bindings expected in the fragment shader:
//   @group(GI_GROUP) @binding(0) uniform params: GiSampleParams (32B)
//   @group(GI_GROUP) @binding(1) var<storage, read> c0_intervals: array<RadianceInterval>
//
// The caller defines GI_GROUP (e.g. `#define GI_GROUP 3`) before including.

// ---------------------------------------------------------------------------
// GI Uniform Parameters (32 bytes, matches Rust GiSampleParams)
// ---------------------------------------------------------------------------

struct GiSampleParams {
    grid_origin_x: f32,    // [  0] C0 grid world origin
    grid_origin_y: f32,    // [  4]
    grid_origin_z: f32,    // [  8]
    grid_spacing: f32,     // [ 12] C0 probe spacing
    ray_count: f32,        // [ 16] C0 rays per probe
    grid_size: f32,        // [ 20] C0 grid resolution
    gi_intensity: f32,     // [ 24] overall GI strength
    bounce_intensity: f32, // [ 28] bounce multiplier
}

// ---------------------------------------------------------------------------
// RadianceInterval (48 bytes, matches Rust RadianceInterval)
// ---------------------------------------------------------------------------

struct RadianceInterval {
    radiance_in: vec4<f32>,      // .rgb = in-scattered, .a = pad
    radiance_out: vec4<f32>,     // .rgb = emission, .w = tau
    direction_length: vec4<f32>, // .xyz = direction, .w = distance
}

// ---------------------------------------------------------------------------
// Nearest probe index helper
// ---------------------------------------------------------------------------

/// Find the integer grid index for the nearest C0 probe to a world position.
/// Returns clamped grid coordinate (0 .. grid_size-1).
fn nearest_probe_grid_coord(
    world_pos: vec3<f32>,
    grid_origin: vec3<f32>,
    grid_spacing: f32,
    grid_size_f: f32,
) -> vec3<u32> {
    let rel = world_pos - grid_origin;
    let grid_f = rel / grid_spacing;
    let clamped = clamp(grid_f, vec3<f32>(0.0), vec3<f32>(grid_size_f - 1.0));
    return vec3<u32>(u32(clamped.x), u32(clamped.y), u32(clamped.z));
}

// ---------------------------------------------------------------------------
// Main GI sampling function
// ---------------------------------------------------------------------------

/// Sample Global Illumination from the C0 cascade for a world-space position.
///
/// # Parameters
///   - `world_pos`:   Fragment position in world space.
///   - `normal`:      Fragment normal (normalized).
///   - `params`:      The GI uniform parameters.
///   - `intervals`:   The C0 merged interval storage buffer.
///
/// # Returns
///   Indirect radiance contribution (vec3<f32>).
///
/// Algorithm:
///   1. Find the nearest C0 probe to world_pos.
///   2. Iterate all rays of that probe.
///   3. For each ray with NdotL > 0 (upper hemisphere):
///      - Accumulate radiance_in weighted by NdotL and bounce_intensity.
///   4. Normalize by the number of contributing rays.
///   5. Multiply by gi_intensity.
fn sample_gi(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    params: GiSampleParams,
    intervals: array<RadianceInterval>,
) -> vec3<f32> {
    let grid_origin = vec3<f32>(params.grid_origin_x, params.grid_origin_y, params.grid_origin_z);
    let grid_size = params.grid_size;
    let ray_count = u32(params.ray_count);

    // Early out if no rays
    if (ray_count == 0u) {
        return vec3<f32>(0.0);
    }

    // Step 1: Find nearest probe grid coordinate
    let grid_coord = nearest_probe_grid_coord(world_pos, grid_origin, params.grid_spacing, grid_size);

    // Step 2: Compute linear index into the interval array
    // Layout: intervals[probe_flat_index * ray_count + ray_index]
    let probe_flat = grid_coord.x + grid_coord.y * u32(grid_size) + grid_coord.z * u32(grid_size) * u32(grid_size);
    let probe_start = probe_flat * ray_count;

    // Step 3: Accumulate GI from upper hemisphere rays
    var gi = vec3<f32>(0.0);
    var weight_sum = 0.0;

    for (var r = 0u; r < ray_count; r = r + 1u) {
        let idx = probe_start + r;
        // Bounds check
        if (idx >= u32(arrayLength(&intervals))) {
            break;
        }
        let interval = intervals[idx];

        // Skip zero-distance rays (probe outside loaded area)
        if (interval.direction_length.w < 0.001) {
            continue;
        }

        // Ray direction
        let ray_dir = normalize(interval.direction_length.xyz);

        // Hemisphere check: only accumulate rays in the same hemisphere as normal
        let ndotl = dot(normal, ray_dir);
        if (ndotl <= 0.0) {
            continue;
        }

        // Weight: cosine (Lambert) * solid angle proxy (4*pi / ray_count)
        let cos_weight = ndotl;
        let solid_angle_weight = 4.0 * 3.14159265 / f32(ray_count);

        // Transmittance-weighted in-scattered radiance
        // exp(-tau) scales how much light from far away reaches this point
         let tau = clamp(interval.radiance_out.w, 0.0, 80.0);
         let transmittance = exp(-tau);

         let em_tau = clamp(interval.radiance_in.a, 0.0, 80.0);
         let em_trans = exp(-em_tau);

         let bounce_contribution = interval.radiance_in.rgb * cos_weight * solid_angle_weight;
         let emission_contribution = interval.radiance_out.rgb * em_trans * cos_weight * solid_angle_weight * params.bounce_intensity;
        gi = gi + bounce_contribution + emission_contribution;
        weight_sum = weight_sum + cos_weight * solid_angle_weight;
    }

    // Step 4: Normalize
    if (weight_sum > 0.001) {
        gi = gi / weight_sum;
    }

    // Step 5: Apply global intensity
    gi = gi * params.gi_intensity;

    return gi;
}

// ---------------------------------------------------------------------------
// Convenience: sample GI for a fragment
// ---------------------------------------------------------------------------
// Usage in the main fragment shader (after defining bindings):
//
//   // @group(3) @binding(0) var<uniform> gi_params: GiSampleParams;
//   // @group(3) @binding(1) var<storage, read> gi_c0_intervals: array<RadianceInterval>;
//
//   let indirect = sample_gi(world_position, normalize(normal), gi_params, gi_c0_intervals);
//   final_color = final_color + indirect;
//
