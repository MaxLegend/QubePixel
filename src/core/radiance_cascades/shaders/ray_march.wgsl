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
    cascade_level:     u32,  // 0 = C0 (finest), NUM_CASCADES-1 = C4 (coarsest)
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
const VOXEL_TEX_SIZE:      u32 = 256u; // must match VoxelTextureBuilder::VOXEL_TEX_SIZE
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
    let sz = i32(VOXEL_TEX_SIZE);
    if (vx < 0 || vy < 0 || vz < 0 || vx >= sz || vy >= sz || vz >= sz) {
        return 0u;
    }
    let raw = textureLoad(voxel_tex, vec3<i32>(vx, vy, vz), 0).r;
    if (raw == UNLOADED_SENTINEL) {
        return 0u;
    }
    return raw;
}

/// Compute a single cube-face subdivision direction on the GPU.
/// Matches the CPU cube_face_directions(subdiv) function in sampling.rs.
///
/// Face order: 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z
/// Each face has subdiv×subdiv cells. Total directions = 6 * subdiv * subdiv.
fn cube_face_dir_gpu(idx: u32, subdiv: u32) -> vec3<f32> {
    let cells_per_face = subdiv * subdiv;
    let face_idx = idx / cells_per_face;
    let cell_idx  = idx % cells_per_face;
    let cu = cell_idx % subdiv;
    let cv = cell_idx / subdiv;
    let u = 2.0 * (f32(cu) + 0.5) / f32(subdiv) - 1.0;
    let v = 2.0 * (f32(cv) + 0.5) / f32(subdiv) - 1.0;

    // Construct point on cube face
    var pt: vec3<f32>;
    if (face_idx == 0u) {
        pt = vec3<f32>( 1.0, u, v);   // +X
    } else if (face_idx == 1u) {
        pt = vec3<f32>(-1.0, u, v);   // −X
    } else if (face_idx == 2u) {
        pt = vec3<f32>(u,  1.0, v);   // +Y
    } else if (face_idx == 3u) {
        pt = vec3<f32>(u, -1.0, v);   // −Y
    } else if (face_idx == 4u) {
        pt = vec3<f32>(u, v,  1.0);   // +Z
    } else {
        pt = vec3<f32>(u, v, -1.0);   // −Z
    };

    return normalize(pt);
}

/// Derive the subdivision level from ray count.
/// ray_count = 6 * subdiv^2  →  subdiv = sqrt(ray_count / 6)
fn derive_subdiv(ray_count: u32) -> u32 {
    // Exact for our cascade configs: 6, 24, 54, 96, 150
    var s: u32 = 1u;
    for (var i: u32 = 1u; i <= 10u; i = i + 1u) {
        if (6u * i * i >= ray_count) { s = i; break; }
    }
    return s;
}


// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)

fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let grid_origin = vec3<f32>(
        params.grid_origin_x,
        params.grid_origin_y,
        params.grid_origin_z
    );
    let probe_pos = grid_origin
        + vec3<f32>(f32(probe_ix), f32(probe_iy), f32(probe_iz))
        * params.grid_spacing;

    var ray_dir: vec3<f32>;
    let num_stored: u32 = arrayLength(&ray_directions);
    if (ray_idx < num_stored && num_stored > 0u) {
        ray_dir = normalize(ray_directions[ray_idx].xyz);
    } else {
        let subdiv = derive_subdiv(rays_per_probe);
        ray_dir = cube_face_dir_gpu(ray_idx, subdiv);
    }

    let ray_origin = probe_pos + ray_dir * params.ray_start;
    let voxel_origin = vec3<f32>(
        params.voxel_origin_x,
        params.voxel_origin_y,
        params.voxel_origin_z
    );

    let local_pos = ray_origin - voxel_origin;

    var voxel = vec3<i32>(
        i32(floor(local_pos.x)),
        i32(floor(local_pos.y)),
        i32(floor(local_pos.z))
    );

    let step_x: i32 = select(1, -1, ray_dir.x < 0.0);
    let step_y: i32 = select(1, -1, ray_dir.y < 0.0);
    let step_z: i32 = select(1, -1, ray_dir.z < 0.0);

    let t_delta = vec3<f32>(
        select(1.0e10, abs(1.0 / ray_dir.x), abs(ray_dir.x) > EPSILON),
        select(1.0e10, abs(1.0 / ray_dir.y), abs(ray_dir.y) > EPSILON),
        select(1.0e10, abs(1.0 / ray_dir.z), abs(ray_dir.z) > EPSILON)
    );

    var t_max = vec3<f32>(0.0, 0.0, 0.0);

    if (abs(ray_dir.x) > EPSILON) {
        if (ray_dir.x > 0.0) {
            t_max.x = (f32(voxel.x + 1) - local_pos.x) / ray_dir.x;
        } else {
            t_max.x = (f32(voxel.x) - local_pos.x) / ray_dir.x;
        }
    } else {
        t_max.x = 1.0e10;
    }

    if (abs(ray_dir.y) > EPSILON) {
        if (ray_dir.y > 0.0) {
            t_max.y = (f32(voxel.y + 1) - local_pos.y) / ray_dir.y;
        } else {
            t_max.y = (f32(voxel.y) - local_pos.y) / ray_dir.y;
        }
    } else {
        t_max.y = 1.0e10;
    }

    if (abs(ray_dir.z) > EPSILON) {
        if (ray_dir.z > 0.0) {
            t_max.z = (f32(voxel.z + 1) - local_pos.z) / ray_dir.z;
        } else {
            t_max.z = (f32(voxel.z) - local_pos.z) / ray_dir.z;
        }
    } else {
        t_max.z = 1.0e10;
    }

    t_max = max(t_max, vec3<f32>(0.0, 0.0, 0.0));

    var tau:          f32 = 0.0;
    var transmittance: f32 = 1.0;
    var radiance_in:  vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var radiance_out: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var total_distance: f32 = 0.0;
    var hit_opaque:   bool = false;
    var t_prev:       f32 = 0.0;

    // Pre-hit tau: tau before the opaque surface (for emission attenuation
    // in the fragment shader). Stored in radiance_in.a (the .w component).
    var pre_hit_tau:  f32 = 0.0;

    for (var step_i: u32 = 0u; step_i < params.max_steps; step_i = step_i + 1u) {
        let t_current: f32 = min(min(t_max.x, t_max.y), t_max.z);

        if (t_current >= params.ray_length) {
            total_distance = params.ray_length;
            break;
        }

        let step_dist: f32 = max(t_current - t_prev, 0.001);

        let block_id: u32 = sample_voxel_block(voxel.x, voxel.y, voxel.z);

        if (block_id != 0u) {
            let props = read_block_props(block_id);
            let block_opacity:  f32 = props.x;
            let block_emission = clamp(
                vec3<f32>(props.y, props.z, props.w),
                vec3<f32>(0.0),
                vec3<f32>(100.0)
            );

            if (block_opacity > 0.001) {
                let step_tau: f32 = block_opacity * step_dist;

                let prev_t: f32 = transmittance;
                let exp_val: f32 = exp(-min(step_tau, 80.0));
                transmittance = transmittance * exp_val;

                radiance_in = radiance_in + (prev_t - transmittance) * block_emission;

                tau = tau + step_tau;

                if (tau >= params.opaque_threshold) {
                    // Surface emission: remove volume contribution from this step,
                    // store it only as radiance_out (endpoint emission).
                    radiance_in = radiance_in - (prev_t - transmittance) * block_emission;
                    pre_hit_tau = tau - step_tau;
                    radiance_out = block_emission;
                    hit_opaque = true;
                    total_distance = t_current;
                    break;
                }
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

        t_prev = t_current;
    }

    if (!hit_opaque && total_distance < 0.001) {
        total_distance = params.ray_length;
    }

    tau = min(tau, params.opaque_threshold);

    // Sky contribution: ONLY sampled in the outermost cascade (C4, cascade_level == 4).
    // Inner cascades (C0-C3) leave radiance_in at 0 when the ray exits without a hit,
    // and the merge chain propagates C4's sky radiance inward via transmittance weighting.
    // This prevents sky from being counted NUM_CASCADES times (which causes white overexposure).
    if (!hit_opaque) {
        pre_hit_tau = tau;
        if (params.cascade_level == 4u) {
            let sky_intensity = clamp(params.sky_brightness, 0.0, 1.0);
            let sky_color = vec3<f32>(
                sky_intensity * 0.75,
                sky_intensity * 0.88,
                sky_intensity * 1.0
            );
            radiance_in = radiance_in + transmittance * sky_color;
        }
        radiance_out = vec3<f32>(0.0, 0.0, 0.0);
    }

    // radiance_in.a (.w) = pre_hit_tau — used by fragment shader for
    // emission attenuation (only medium BEFORE the opaque surface).
    let output_idx: u32 = probe_flat * rays_per_probe + ray_idx;
    output_intervals[output_idx] = RadianceInterval(
        vec4<f32>(radiance_in, pre_hit_tau),
        vec4<f32>(radiance_out, tau),
        vec4<f32>(ray_dir, total_distance)
    );
}