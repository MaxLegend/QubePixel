// =============================================================================
// QubePixel — Radiance Cascades: Emission Injection (Point Light)
// =============================================================================
//
// Compute shader that injects point light contributions from emissive blocks
// into the finest cascade (C0) probe intervals. This makes emissive blocks
// act as true point light sources — their light spreads to nearby probes
// even if the probe's rays don't directly hit the block.
//
// Algorithm (per probe):
//   1. Read probe world position from cascade params
//   2. Iterate all emissive block entries
//   3. For each block within influence radius:
//      a. Compute distance from probe to block
//      b. Compute attenuation: 1 / (1 + k * d²)
//      c. For each ray of this probe:
//         - Compute solid angle weight (NdotL * 4π / ray_count)
//         - Add emission * attenuation * solid_angle to radiance_in
//
// Dispatch: (ceil(total_probes * ray_count / 64), 1, 1)
//
// This pass runs AFTER ray marching and BEFORE merging, so injected emission
// propagates through the entire merge chain to all cascade levels.

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// One emissive block entry — 32 bytes.
/// CPU fills a buffer of these and uploads every frame.
struct EmissiveBlock {
    /// World-space position of the block center (.xyz), .w = influence radius.
    pos_radius: vec4<f32>,
    /// Emission color * intensity (.rgb), .a = attenuation factor (k).
    color_atten: vec4<f32>,
};

/// Radiance interval — 48 bytes (matches ray_march.wgsl).
struct RadianceInterval {
    radiance_in:      vec4<f32>,
    radiance_out:     vec4<f32>,
    direction_length: vec4<f32>,
};

/// Cascade parameters — minimal set needed for injection.
/// Matches the layout from ray_march.wgsl CascadeParams (64 bytes).
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
    cascade_level:     u32,
};

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>            params:          CascadeParams;
@group(0) @binding(1) var<storage, read>       emissive_blocks: array<EmissiveBlock>;
@group(0) @binding(2) var<storage, read>       ray_directions:  array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> intervals:       array<RadianceInterval>;
@group(0) @binding(4) var                      voxel_tex:       texture_3d<u32>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE: u32 = 64u;
const PI: f32 = 3.14159265;

// ---------------------------------------------------------------------------
// Cube-face direction generation (matches ray_march.wgsl)
// ---------------------------------------------------------------------------

fn cube_face_dir_gpu(idx: u32, subdiv: u32) -> vec3<f32> {
    let cells_per_face = subdiv * subdiv;
    let face_idx = idx / cells_per_face;
    let cell_idx  = idx % cells_per_face;
    let cu = cell_idx % subdiv;
    let cv = cell_idx / subdiv;
    let u = 2.0 * (f32(cu) + 0.5) / f32(subdiv) - 1.0;
    let v = 2.0 * (f32(cv) + 0.5) / f32(subdiv) - 1.0;

    var pt: vec3<f32>;
    if (face_idx == 0u) {
        pt = vec3<f32>( 1.0, u, v);
    } else if (face_idx == 1u) {
        pt = vec3<f32>(-1.0, u, v);
    } else if (face_idx == 2u) {
        pt = vec3<f32>(u,  1.0, v);
    } else if (face_idx == 3u) {
        pt = vec3<f32>(u, -1.0, v);
    } else if (face_idx == 4u) {
        pt = vec3<f32>(u, v,  1.0);
    } else {
        pt = vec3<f32>(u, v, -1.0);
    };

    return normalize(pt);
}

fn derive_subdiv(ray_count: u32) -> u32 {
    var s: u32 = 1u;
    for (var i: u32 = 1u; i <= 10u; i = i + 1u) {
        if (6u * i * i >= ray_count) { s = i; break; }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Voxel occlusion: DDA ray march from probe to emissive block
// Returns true if any opaque voxel lies between probe_pos and block_pos.
// ---------------------------------------------------------------------------

fn is_occluded(probe_pos: vec3<f32>, block_pos: vec3<f32>) -> bool {
    let vox_origin = vec3<f32>(params.voxel_origin_x, params.voxel_origin_y, params.voxel_origin_z);
    let vox_size   = vec3<i32>(textureDimensions(voxel_tex));

    let to_block = block_pos - probe_pos;
    let dist     = length(to_block);
    if (dist < 1.001) { return false; } // probe is inside or adjacent to the block

    let dir  = to_block / dist;
    let lpos = probe_pos - vox_origin; // probe in voxel-texture local space

    var vox = vec3<i32>(i32(floor(lpos.x)), i32(floor(lpos.y)), i32(floor(lpos.z)));

    // If probe is outside texture, assume unoccluded
    if (any(vox < vec3<i32>(0)) || any(vox >= vox_size)) { return false; }

    let sx = select(-1, 1, dir.x >= 0.0);
    let sy = select(-1, 1, dir.y >= 0.0);
    let sz = select(-1, 1, dir.z >= 0.0);

    let td = vec3<f32>(
        select(1e10, abs(1.0 / dir.x), abs(dir.x) > 0.0001),
        select(1e10, abs(1.0 / dir.y), abs(dir.y) > 0.0001),
        select(1e10, abs(1.0 / dir.z), abs(dir.z) > 0.0001)
    );

    var tm = max(vec3<f32>(
        select(1e10, (f32(select(vox.x, vox.x + 1, sx > 0)) - lpos.x) / dir.x, abs(dir.x) > 0.0001),
        select(1e10, (f32(select(vox.y, vox.y + 1, sy > 0)) - lpos.y) / dir.y, abs(dir.y) > 0.0001),
        select(1e10, (f32(select(vox.z, vox.z + 1, sz > 0)) - lpos.z) / dir.z, abs(dir.z) > 0.0001)
    ), vec3<f32>(0.0));

    // Skip the first voxel (probe itself)
    if (tm.x <= tm.y && tm.x <= tm.z) { vox.x += sx; tm.x += td.x; }
    else if (tm.y <= tm.z)             { vox.y += sy; tm.y += td.y; }
    else                               { vox.z += sz; tm.z += td.z; }

    let stop_t  = dist - 0.5; // stop just before the emissive block voxel
    let max_steps = min(u32(dist) + 2u, 64u);

    for (var i = 0u; i < max_steps; i = i + 1u) {
        let t_cur = min(tm.x, min(tm.y, tm.z));
        if (t_cur >= stop_t) { break; }

        if (any(vox < vec3<i32>(0)) || any(vox >= vox_size)) { return false; }

        let bid = textureLoad(voxel_tex, vox, 0).r;
        if (bid != 0u && bid != 0xFFFFu) { return true; } // hit opaque block

        if (tm.x <= tm.y && tm.x <= tm.z) { vox.x += sx; tm.x += td.x; }
        else if (tm.y <= tm.z)             { vox.y += sy; tm.y += td.y; }
        else                               { vox.z += sz; tm.z += td.z; }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx: u32 = gid.x;
    let gs: u32 = params.grid_size;
    let rays_per_probe: u32 = params.ray_count;
    let total: u32 = gs * gs * gs * rays_per_probe;

    if (flat_idx >= total) {
        return;
    }

    // Decompose flat index into (probe, ray)
    let probe_flat: u32 = flat_idx / rays_per_probe;
    let ray_idx: u32    = flat_idx - probe_flat * rays_per_probe;

    // Probe grid coordinates
    let probe_ix: u32 = probe_flat % gs;
    let probe_iy: u32 = (probe_flat / gs) % gs;
    let probe_iz: u32 = probe_flat / (gs * gs);

    // Probe world position
    let grid_origin = vec3<f32>(
        params.grid_origin_x,
        params.grid_origin_y,
        params.grid_origin_z
    );
    let probe_pos = grid_origin
        + vec3<f32>(f32(probe_ix), f32(probe_iy), f32(probe_iz))
        * params.grid_spacing;

    // Ray direction
    var ray_dir: vec3<f32>;
    let num_stored: u32 = arrayLength(&ray_directions);
    if (ray_idx < num_stored && num_stored > 0u) {
        ray_dir = normalize(ray_directions[ray_idx].xyz);
    } else {
        let subdiv = derive_subdiv(rays_per_probe);
        ray_dir = cube_face_dir_gpu(ray_idx, subdiv);
    }

    // Accumulate point light contributions from all emissive blocks.
    // NOTE: we do NOT multiply by solid_angle here. The radiance_in field
    // stores radiance (W/sr), and the fragment shader applies solid_angle
    // weighting during irradiance reconstruction. Including solid_angle here
    // would double-count it, making injected emission ~20× too dim.
    var point_light = vec3<f32>(0.0);
    let num_blocks = arrayLength(&emissive_blocks);

    for (var b: u32 = 0u; b < num_blocks; b = b + 1u) {
        let block = emissive_blocks[b];
        let block_pos = block.pos_radius.xyz;
        let influence_radius = block.pos_radius.w;
        let emission_color = block.color_atten.rgb;
        let atten_k = block.color_atten.a; // attenuation factor

        // Vector from probe to block
        let to_block = block_pos - probe_pos;
        let dist = length(to_block);

        // Skip if outside influence radius
        if (dist > influence_radius || dist < 0.001) {
            continue;
        }

        // Skip if an opaque voxel blocks the path from probe to block
        if (is_occluded(probe_pos, block_pos)) {
            continue;
        }

        // Attenuation: inverse square with linear term to avoid singularity
        // atten = 1 / (1 + k * dist^2)
        let attenuation = 1.0 / (1.0 + atten_k * dist * dist);

        // Direction from probe to block
        let dir_to_block = to_block / dist;

        // Cosine factor: how much this ray "sees" the block
        // Rays pointing toward the block get more contribution
        let ndotl = dot(ray_dir, dir_to_block);

        // Only inject for rays that face toward the block (hemisphere)
        if (ndotl > 0.0) {
            point_light = point_light + emission_color * attenuation * ndotl;
        }
    }

    // Add point light contribution to existing interval
    if (length(point_light) > 0.0001) {
        var iv = intervals[flat_idx];
        iv.radiance_in = vec4<f32>(
            iv.radiance_in.x + point_light.x,
            iv.radiance_in.y + point_light.y,
            iv.radiance_in.z + point_light.z,
            iv.radiance_in.a  // preserve pre_hit_tau in .w
        );
        intervals[flat_idx] = iv;
    }
}
