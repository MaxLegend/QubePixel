// =============================================================================
// QubePixel — Radiance Cascades: Merge + Trilinear Fix Compute Shader
// =============================================================================
//
// Merges radiance intervals from a parent (coarser) cascade into a child
// (finer) cascade using transmittance-weighted composition.
//
// Process order (called sequentially by the host):
//   C3 ← C4,  C2 ← C3,  C1 ← C2,  C0 ← C1
// Each pass reads the parent's (already-merged) output and writes the merged
// result back into the child's output buffer (in-place).
//
// For each child probe, 8 surrounding parent probes are trilinearly
// interpolated. A "Trilinear Fix" applies perpendicular-displacement damping
// to reduce GI bleeding through walls at cascade boundaries.
//
// Bind group layout (group 0):
//   @binding(0)  merge_params        — uniform buffer (96 bytes)
//   @binding(1)  parent_intervals    — storage buffer, read (array<RadianceInterval>)
//   @binding(2)  child_intervals     — storage buffer, read_write (array<RadianceInterval>)
//
// Dispatch: (ceil(child_probes * child_rays / 64), 1, 1)

// ---------------------------------------------------------------------------
// Data structures (must match Rust types exactly)
// ---------------------------------------------------------------------------

/// Radiance interval — 48 bytes, matches Rust RadianceInterval.
/// Layout: 3 x vec4<f32>.
struct RadianceInterval {
    radiance_in:      vec4<f32>,
    radiance_out:     vec4<f32>,
    direction_length: vec4<f32>,
};

/// Merge parameters — 96 bytes, matches Rust MergeParams.
/// 24 x 4-byte fields (f32 or u32).
/// No vec3 to avoid WGSL uniform alignment padding issues.
struct MergeParams {
    // -- Child cascade --
    child_grid_spacing:      f32,
    child_ray_start:         f32,
    child_ray_length:        f32,
    child_grid_half_extent:  f32,
    child_grid_size:         u32,
    child_ray_count:         u32,
    _pad_c0:                 u32,
    _pad_c1:                 u32,
    // -- Parent cascade --
    parent_grid_spacing:     f32,
    parent_ray_start:        f32,
    parent_ray_length:       f32,
    parent_grid_half_extent: f32,
    parent_grid_size:        u32,
    parent_ray_count:        u32,
    _pad_p0:                 u32,
    _pad_p1:                 u32,
    // -- Grid origins (world space) --
    child_origin_x:          f32,
    child_origin_y:          f32,
    child_origin_z:          f32,
    _pad_o0:                 u32,
    parent_origin_x:         f32,
    parent_origin_y:         f32,
    parent_origin_z:         f32,
    opaque_threshold:        f32,
};

// ---------------------------------------------------------------------------
// Bind group bindings
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>         merge_params:      MergeParams;
@group(0) @binding(1) var<storage, read>    parent_intervals:  array<RadianceInterval>;
@group(0) @binding(2) var<storage, read_write> child_intervals: array<RadianceInterval>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WORKGROUP_SIZE:          u32 = 64u;
const GOLDEN_ANGLE:            f32 = 2.39996323;
const TRILINEAR_FIX_DAMPING:   f32 = 0.25;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a single cube-face subdivision direction on the GPU.
/// Matches the CPU cube_face_directions(subdiv) function in sampling.rs.
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
    var s: u32 = 1u;
    for (var i: u32 = 1u; i <= 10u; i = i + 1u) {
        if (6u * i * i >= ray_count) { s = i; break; }
    }
    return s;
}

/// Analytically find the nearest cube-face direction index.
/// O(1) — projects target onto the dominant cube face and maps to cell.
/// Matches the CPU nearest_cube_face_dir_index() function in sampling.rs.
fn nearest_cube_face_dir(targetcuve: vec3<f32>, subdiv: u32) -> u32 {
    let t = normalize(targetcuve);
    let ax = abs(t.x);
    let ay = abs(t.y);
    let az = abs(t.z);

    var face_idx: u32 = 0u;
    var u: f32 = 0.0;
    var v: f32 = 0.0;

    if (ax >= ay && ax >= az) {
        if (t.x >= 0.0) { face_idx = 0u; } else { face_idx = 1u; }
        u = t.y;
        v = t.z;
    } else if (ay >= az) {
        if (t.y >= 0.0) { face_idx = 2u; } else { face_idx = 3u; }
        u = t.x;
        v = t.z;
    } else {
        if (t.z >= 0.0) { face_idx = 4u; } else { face_idx = 5u; }
        u = t.x;
        v = t.y;
    };

    // Map [-1, 1] → cell index [0, subdiv-1]
    var cu: u32 = u32((u + 1.0) * 0.5 * f32(subdiv));
    var cv: u32 = u32((v + 1.0) * 0.5 * f32(subdiv));
    if (cu >= subdiv) { cu = subdiv - 1u; }
    if (cv >= subdiv) { cv = subdiv - 1u; }

    return face_idx * subdiv * subdiv + cv * subdiv + cu;
}
// ---------------------------------------------------------------------------
// Compute entry point
// ---------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // ---- Flat index decomposition ----
    let flat_idx: u32 = gid.x;
    let c_gs: u32 = merge_params.child_grid_size;
    let c_rc: u32 = merge_params.child_ray_count;
    let total: u32 = c_gs * c_gs * c_gs * c_rc;

    if (flat_idx >= total) {
        return;
    }

    let probe_flat: u32 = flat_idx / c_rc;
    let ray_idx:    u32 = flat_idx - probe_flat * c_rc;

    let probe_ix: u32 = probe_flat % c_gs;
    let probe_iy: u32 = (probe_flat / c_gs) % c_gs;
    let probe_iz: u32 = probe_flat / (c_gs * c_gs);

    // ---- Child probe world position ----
    let child_pos = vec3<f32>(
        merge_params.child_origin_x + f32(probe_ix) * merge_params.child_grid_spacing,
        merge_params.child_origin_y + f32(probe_iy) * merge_params.child_grid_spacing,
        merge_params.child_origin_z + f32(probe_iz) * merge_params.child_grid_spacing
    );

    // ---- Parent grid local coordinates (fractional) ----
    let p_gs:  u32 = merge_params.parent_grid_size;
    let p_gs_f: f32 = f32(p_gs);

    let parent_local = vec3<f32>(
        (child_pos.x - merge_params.parent_origin_x) / merge_params.parent_grid_spacing,
        (child_pos.y - merge_params.parent_origin_y) / merge_params.parent_grid_spacing,
        (child_pos.z - merge_params.parent_origin_z) / merge_params.parent_grid_spacing
    );

    // Bounds check: child probe must be within parent grid (with half-texel margin)
    if (parent_local.x < -0.5 || parent_local.y < -0.5 || parent_local.z < -0.5 ||
        parent_local.x > p_gs_f - 0.5 || parent_local.y > p_gs_f - 0.5 || parent_local.z > p_gs_f - 0.5) {
        return; // Outside parent grid — no merge needed
    }

    // Clamp for trilinear interpolation
    let pl_clamped   = clamp(parent_local, vec3<f32>(0.0), vec3<f32>(p_gs_f - 1.0));
    let parent_floor  = vec3<i32>(i32(floor(pl_clamped.x)),
                                   i32(floor(pl_clamped.y)),
                                   i32(floor(pl_clamped.z)));
    let parent_frac   = pl_clamped - vec3<f32>(f32(parent_floor.x),
                                                f32(parent_floor.y),
                                                f32(parent_floor.z));

    // ---- Get child's own interval ----
    let child_iv = child_intervals[flat_idx];

    // Early exit: child already fully opaque → parent contributes nothing
    if (child_iv.radiance_out.w >= merge_params.opaque_threshold) {
        return;
    }

    // ---- Child ray direction (cube-face subdivision) ----
    let child_subdiv = derive_subdiv(c_rc);
    let child_dir = cube_face_dir_gpu(ray_idx, child_subdiv);
    let child_dir_n = normalize(child_dir);

    // ---- Find nearest parent ray direction (cube-face analytical) ----
    let p_rc: u32 = merge_params.parent_ray_count;
    let parent_subdiv = derive_subdiv(p_rc);
    let nearest_parent_ray: u32 = nearest_cube_face_dir(child_dir_n, parent_subdiv);

    // ---- Trilinear interpolation of 8 parent corners ----
    // Accumulate weighted parent interval data.
    var interp_in:     vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var interp_out:    vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
    var interp_tau:    f32 = 0.0;
    var interp_dist:   f32 = 0.0;
    var total_weight:  f32 = 0.0;
    var interp_pre_hit: f32 = 0.0;

    let fx = parent_frac.x;
    let fy = parent_frac.y;
    let fz = parent_frac.z;

    for (var dz: i32 = 0; dz <= 1; dz = dz + 1) {
        for (var dy: i32 = 0; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = 0; dx <= 1; dx = dx + 1) {
                let px = parent_floor.x + dx;
                let py = parent_floor.y + dy;
                let pz = parent_floor.z + dz;

                // Skip out-of-bounds parent probes
               if (px < 0 || px >= i32(p_gs) || py < 0 || py >= i32(p_gs) || pz < 0 || pz >= i32(p_gs)) {
                    continue;
                }

                let parent_probe_flat = u32(px) + u32(py) * p_gs + u32(pz) * p_gs * p_gs;
                let parent_idx = parent_probe_flat * p_rc + nearest_parent_ray;

                // Bounds check on parent interval array
                let parent_total = p_gs * p_gs * p_gs * p_rc;
                if (parent_idx >= parent_total) {
                    continue;
                }

                let p_iv = parent_intervals[parent_idx];

                // Skip parent rays that didn't travel (culled probes)
                if (p_iv.direction_length.w < 0.001) {
                    continue;
                }

                // Trilinear weight (standard 8-texel interpolation)
                let wx = select(1.0 - fx, fx, dx == 1);
                let wy = select(1.0 - fy, fy, dy == 1);
                let wz = select(1.0 - fz, fz, dz == 1);
                var weight = wx * wy * wz;

                // ---- Trilinear Fix: perpendicular displacement damping ----
                // Reduces GI bleeding through walls at cascade boundaries.
                // Parent probe position in world space
                let parent_pos = vec3<f32>(
                    merge_params.parent_origin_x + f32(px) * merge_params.parent_grid_spacing,
                    merge_params.parent_origin_y + f32(py) * merge_params.parent_grid_spacing,
                    merge_params.parent_origin_z + f32(pz) * merge_params.parent_grid_spacing
                );

                // Displacement from parent to child probe
                let displacement = child_pos - parent_pos;

                // Decompose into parallel and perpendicular components
                // relative to the child's ray direction
                let dot_disp = dot(displacement, child_dir_n);
                let perp = displacement - child_dir_n * dot_disp;
                let perp_len = length(perp);

                // Damping: parent probes with large perpendicular offset
                // contribute less — their ray paths diverge from the child's
                // expected path, reducing parallax/ringing artifacts.
                let damping = 1.0 / (1.0 + perp_len * TRILINEAR_FIX_DAMPING);
                weight = weight * damping;

                // Accumulate
                interp_in   = interp_in   + weight * vec3<f32>(p_iv.radiance_in.x,
                                                                 p_iv.radiance_in.y,
                                                                 p_iv.radiance_in.z);
                interp_out  = interp_out  + weight * vec3<f32>(p_iv.radiance_out.x,
                                                                 p_iv.radiance_out.y,
                                                                 p_iv.radiance_out.z);
                interp_tau  = interp_tau  + weight * p_iv.radiance_out.w;
                interp_dist = interp_dist + weight * p_iv.direction_length.w;
                total_weight = total_weight + weight;
                interp_pre_hit = interp_pre_hit + weight * p_iv.radiance_in.a;
            }
        }
    }

    // No valid parent data — keep child's interval unchanged
    if (total_weight < 0.001) {
        return;
    }

    // Normalize accumulated values
    let inv_w = 1.0 / total_weight;
    interp_in   = interp_in   * inv_w;
    interp_out  = interp_out  * inv_w;
    interp_tau  = interp_tau  * inv_w;
    interp_dist = interp_dist * inv_w;
    interp_pre_hit = interp_pre_hit * inv_w;
    // ---- Merge: child (near) + interpolated parent (far) ----
    // M.radiance_in  = child.radiance_in + exp(-child.tau) * parent_interp.radiance_in
    // M.radiance_out = parent_interp.radiance_out  (far-end emission)
    // M.tau          = min(child.tau + parent_interp.tau, opaque_threshold)
    // M.direction    = child.direction
    // M.distance     = child.distance + parent_interp.distance

    let transmittance = exp(-min(child_iv.radiance_out.w, 80.0));

    let merged_in = clamp(vec3<f32>(
        child_iv.radiance_in.x + transmittance * interp_in.x,
        child_iv.radiance_in.y + transmittance * interp_in.y,
        child_iv.radiance_in.z + transmittance * interp_in.z
        ),
        vec3<f32>(0.0),
        vec3<f32>(2.0)
    );

    let merged_tau = min(child_iv.radiance_out.w + interp_tau, merge_params.opaque_threshold);

    let merged_pre_hit = child_iv.radiance_in.a + interp_pre_hit;

    let result = RadianceInterval(
        vec4<f32>(merged_in, merged_pre_hit),
        vec4<f32>(interp_out, merged_tau),
        vec4<f32>(child_dir_n, child_iv.direction_length.w + interp_dist)
    );

    // ---- Write merged result back to child output ----
    child_intervals[flat_idx] = result;
}