// =============================================================================
// QubePixel — Radiance Cascades: Ray March Compute Shader
// =============================================================================
//
// DDA (Amanatides-Woo) ray marching through the voxel world.
// One invocation per (probe, ray) pair.
//
// Bind group layout (group 0):
//   @binding(0)  cascade_params       — uniform buffer (64 bytes)
//   @binding(1)  voxel_tex           — texture_3d<u32> (R16Uint block IDs)
//   @binding(2)  block_lut           — texture_2d<f32> (Rgba32Float properties)
//   @binding(3)  ray_directions      — storage buffer, read (array<vec4<f32>>)
//   @binding(4)  output_intervals    — storage buffer, read_write (array<RadianceInterval>)
//
// Dispatch: (ceil(total_probes * ray_count / 64), 1, 1) workgroups of 64.

// ---------------------------------------------------------------------------
// Data structures (must match Rust types exactly)
// ---------------------------------------------------------------------------

/// Radiance interval — 48 bytes, matches Rust RadianceInterval.
/// Layout: 3 x vec4<f32>.
struct RadianceInterval {
    /// .rgb = in-scattered radiance (W/sr), .a = unused.
    radiance_in:    vec4<f32>,
    /// .rgb = emission at hit point, .w = optical thickness (tau).
    radiance_out:   vec4<f32>,
    /// .xyz = ray direction, .w = distance traveled (0 if culled).
    direction_length: vec4<f32>,
};

/// Cascade dispatch parameters — 64 bytes, matches Rust CascadeDispatchParams.
/// All fields are f32 or u32 (no vec3 to avoid WGSL uniform alignment issues).
struct CascadeParams {
    grid_spacing:      f32,
    ray_length:        f32,
    ray_start:         f32,
    grid_half_extent:  f32,
    grid_size:         u32,
    ray_count:         u32,
    max_steps:         u32,
    voxel_origin_x:    f32,
    voxel_origin_y:    f32,
    voxel_origin_z:    f32,
    grid_origin_x:     f32,
    grid_origin_y:     f32,
    grid_origin_z:     f32,
    opaque_threshold:  f32,
    sky_brightness:    f32,
    _pad:              f32,
};

// ---------------------------------------------------------------------------
// Bind group bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> params:           CascadeParams;
@group(0) @binding(1) var        voxel_tex:          texture_3d<u32>;
@group(0) @binding(2) var        block_lut:          texture_2d<f32>;
@group(0) @binding(3) var<storage, read>    ray_directions:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output_intervals: array<RadianceInterval>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE:      u32 = 64u;
const VOXEL_TEX_SIZE:      u32 = 128u;
const UNLOADED_SENTINEL:   u32 = 0xFFFFu;
const GOLDEN_ANGLE:        f32 = 2.39996323;
const EPSILON:             f32 = 1.0e-8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read block optical properties from the LUT texture.
/// block_lut is Rgba32Float, 256×1 texels indexed by block ID.
/// Returns vec4<f32>(opacity, emission_r, emission_g, emission_b).
fn read_block_props(block_id: u32) -> vec4<f32> {
    if (block_id >= 256u) {
        // Unknown block type: treat as fully opaque, no emission.
        return vec4<f32>(1000.0, 0.0, 0.0, 0.0);
    }
    return textureLoad(block_lut, vec2<i32>(i32(block_id), 0), 0);
}

/// Sample block ID at a voxel coordinate within the 3D texture.
/// Returns 0 (air) for out-of-bounds or unloaded texels.
fn sample_voxel_block(vx: i32, vy: i32, vz: i32) -> u32 {
    if (vx < 0 || vy < 0 || vz < 0 ||
        vx >= 128 || vy >= 128 || vz >= 128) {
        return 0u;
    }
    let raw = textureLoad(voxel_tex, vec3<i32>(vx, vy, vz), 0).r;
    if (raw == UNLOADED_SENTINEL) {
        return 0u;
    }
    return raw;
}

/// Compute a single Fibonacci sphere direction on the GPU.
/// Used as fallback when ray_idx exceeds the pre-uploaded CPU direction buffer.
fn fibonacci_dir_gpu(idx: u32, n: u32) -> vec3<f32> {
    let inv_n = 1.0 / f32(n);
    let y = 1.0 - 2.0 * (f32(idx) + 0.5) * inv_n;
    let r = sqrt(max(0.0, 1.0 - y * y));
    let theta = GOLDEN_ANGLE * f32(idx);
    return vec3<f32>(cos(theta) * r, y, sin(theta) * r);
}

// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ---- Flat index decomposition ----
    let flat_idx: u32 = gid.x;
    let total_probes: u32 = params.grid_size * params.grid_size * params.grid_size;
    let total: u32 = total_probes * params.ray_count;

    if (flat_idx >= total) {
        return;
    }

    let rays_per_probe: u32 = params.ray_count;
    let probe_flat: u32 = flat_idx / rays_per_probe;
    let ray_idx: u32    = flat_idx - probe_flat * rays_per_probe;

    let gs: u32 = params.grid_size;
    let probe_ix: u32 = probe_flat % gs;
    let probe_iy: u32 = (probe_flat / gs) % gs;
    let probe_iz: u32 = probe_flat / (gs * gs);

    // ---- Probe world position ----
    let grid_origin = vec3<f32>(
        params.grid_origin_x,
        params.grid_origin_y,
        params.grid_origin_z
    );
    let probe_pos = grid_origin
        + vec3<f32>(f32(probe_ix), f32(probe_iy), f32(probe_iz))
        * params.grid_spacing;

    // ---- Ray direction ----
    var ray_dir: vec3<f32>;
    let num_stored: u32 = arrayLength(&ray_directions);
    if (ray_idx < num_stored && num_stored > 0u) {
        ray_dir = normalize(ray_directions[ray_idx].xyz);
    } else {
        ray_dir = fibonacci_dir_gpu(ray_idx, rays_per_probe);
    }

    // ---- Ray origin (offset from probe) ----
    let ray_origin = probe_pos + ray_dir * params.ray_start;
    let voxel_origin = vec3<f32>(
        params.voxel_origin_x,
        params.voxel_origin_y,
        params.voxel_origin_z
    );

    // ---- DDA initialization (Amanatides-Woo) ----
    let local_pos = ray_origin - voxel_origin;

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    // Step direction: +1 or -1 per axis
    let step_x: i32 = select(1, -1, ray_dir.x < 0.0);
    let step_y: i32 = select(1, -1, ray_dir.y < 0.0);
    let step_z: i32 = select(1, -1, ray_dir.z < 0.0);

    // tDelta: distance between consecutive voxel boundaries along ray per axis
    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / ray_dir.x), abs(ray_dir.x) > EPSILON),
        select(1.0e10, abs(1.0 / ray_dir.y), abs(ray_dir.y) > EPSILON),
        select(1.0e10, abs(1.0 / ray_dir.z), abs(ray_dir.z) > EPSILON)
    );

    // tMax: distance from ray_origin to next voxel boundary per axis
    var t_max = vec3<f32>(0.0, 0.0, 0.0);

    // X axis
    if (abs(ray_dir.x) > EPSILON) {
        if (ray_dir.x > 0.0) {
            t_max.x = (f32(voxel.x + 1) - local_pos.x) / ray_dir.x;
        } else {
            t_max.x = (f32(voxel.x) - local_pos.x) / ray_dir.x;
        }
    } else {
        t_max.x = 1.0e10;
    }

    // Y axis
    if (abs(ray_dir.y) > EPSILON) {
        if (ray_dir.y > 0.0) {
            t_max.y = (f32(voxel.y + 1) - local_pos.y) / ray_dir.y;
        } else {
            t_max.y = (f32(voxel.y) - local_pos.y) / ray_dir.y;
        }
    } else {
        t_max.y = 1.0e10;
    }

    // Z axis
    if (abs(ray_dir.z) > EPSILON) {
        if (ray_dir.z > 0.0) {
            t_max.z = (f32(voxel.z + 1) - local_pos.z) / ray_dir.z;
        } else {
            t_max.z = (f32(voxel.z) - local_pos.z) / ray_dir.z;
        }
    } else {
        t_max.z = 1.0e10;
    }

    // Clamp negative tMax (numerical safety near voxel boundaries)
    t_max = max(t_max, vec3<f32>(0.0, 0.0, 0.0));

    // ---- Ray march state ----
    var tau:          f32 = 0.0;
    var transmittance: f32 = 1.0;
    var radiance_in:  vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var radiance_out: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var total_distance: f32 = 0.0;
    var hit_opaque:   bool = false;
    var t_prev:       f32 = 0.0;

    // ---- DDA loop ----
    for (var step_i: u32 = 0u; step_i < params.max_steps; step_i = step_i + 1u) {
        let t_current: f32 = min(min(t_max.x, t_max.y), t_max.z);

        // Exceeded maximum ray length
        if (t_current >= params.ray_length) {
            total_distance = params.ray_length;
            break;
        }

        let step_dist: f32 = max(t_current - t_prev, 0.001);

        // ---- Sample block at current voxel ----
        let block_id: u32 = sample_voxel_block(voxel.x, voxel.y, voxel.z);

        if (block_id != 0u) {
            let props = read_block_props(block_id);
            let block_opacity:  f32 = props.x;
            let block_emission: vec3<f32> = vec3<f32>(props.y, props.z, props.w);

            if (block_opacity > 0.001) {
                // Optical thickness for this voxel traversal
                let step_tau: f32 = block_opacity * step_dist;

                // Update transmittance
                let prev_t: f32 = transmittance;
                let exp_val: f32 = exp(-min(step_tau, 80.0));
                transmittance = transmittance * exp_val;

                // In-scattered emission weighted by transmittance difference
                radiance_in = radiance_in + (prev_t - transmittance) * block_emission;

                // Accumulate optical thickness
                tau = tau + step_tau;

                // Check opaque termination
                if (tau >= params.opaque_threshold) {
                    radiance_out = block_emission;
                    hit_opaque = true;
                    total_distance = t_current;
                    break;
                }
            }
        }

        // ---- Advance DDA to next voxel ----
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

        t_prev = t_current;
    }

    // ---- Finalize ----
    // If loop ended without break, ray reached max distance
    if (!hit_opaque && total_distance < 0.001) {
        total_distance = params.ray_length;
    }

    // Clamp tau to prevent overflow
    tau = min(tau, params.opaque_threshold);

    // Sky contribution for rays that didn't hit opaque surface
    if (!hit_opaque) {
        radiance_out = vec3<f32>(params.sky_brightness);
    }

    // ---- Write output interval ----
    let output_idx: u32 = probe_flat * rays_per_probe + ray_idx;
    output_intervals[output_idx] = RadianceInterval(
        vec4<f32>(radiance_in, 0.0),
        vec4<f32>(radiance_out, tau),
        vec4<f32>(ray_dir, total_distance)
    );
}