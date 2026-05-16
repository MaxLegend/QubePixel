// =============================================================================
// QubePixel — Radiance Cascades: Direction Sampling
// =============================================================================
//
// Uniform distribution of ray directions on the unit sphere.
// Uses the Fibonacci (Golden Spiral) method for deterministic, well-spaced
// point sets. Consistent angular gaps are critical for the merge step where
// rays between adjacent cascades need spatial alignment.
//
// Alternative methods (Gauss-Radau quadrature, Hammersley) are considered
// for future optimization but Fibonacci is the default for its simplicity
// and quality trade-off.

use glam::Vec3;

#[allow(unused_imports)]
use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Golden angle in radians: 2 * PI * (1 - 1/phi) where phi = (1+sqrt(5))/2.
/// Used by the Fibonacci sphere algorithm.
pub const GOLDEN_ANGLE: f32 = 2.399_963_23; // PI * (1 + sqrt(5))

/// Pre-computed golden ratio.
pub const GOLDEN_RATIO: f32 = 1.618_033_99;

// ---------------------------------------------------------------------------
// Cube-Face Subdivision
// ---------------------------------------------------------------------------

/// Total number of directions for a given subdivision level.
/// Each of the 6 cube faces is divided into subdiv×subdiv cells.
/// subdiv=1 → 6, subdiv=2 → 24, subdiv=3 → 54, subdiv=4 → 96, subdiv=5 → 150.
#[inline]
pub fn cube_face_ray_count(subdiv: u32) -> u32 {
    6 * subdiv * subdiv
}

/// Derive the cube-face subdivision level from a desired ray count.
///
/// Returns the smallest `subdiv` such that `6 * subdiv² >= ray_count`.
/// This mirrors the `derive_subdiv` WGSL function in the compute shaders.
///
/// Valid exact levels: ray_count=6→1, 24→2, 54→3, 96→4, 150→5.
/// For intermediate values, rounds up to the next valid subdiv.
#[inline]
pub fn derive_subdiv(ray_count: u32) -> u32 {
    for i in 1u32..=10 {
        if 6 * i * i >= ray_count {
            return i;
        }
    }
    1
}

/// Generate directions on the unit sphere by subdividing the 6 faces of a cube.
///
/// Each face is divided into a `subdiv × subdiv` grid. The center of each cell
/// is projected from the cube face onto the unit sphere and normalized.
///
/// **Face order:** 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z
///
/// For subdiv=1 the result is exactly the 6 cardinal directions:
///   (1,0,0), (−1,0,0), (0,1,0), (0,−1,0), (0,0,1), (0,0,−1)
///
/// For subdiv=2, each face gets 4 directions (corners at ±0.5 offset),
/// giving 24 total — perfect for covering face diagonals and edge midpoints.
///
/// This is the **primary** direction generator for Radiance Cascades in
/// QubePixel because voxel worlds have axis-aligned reflective surfaces.
///
/// **Panics:** if `subdiv == 0`.
pub fn cube_face_directions(subdiv: u32) -> Vec<Vec3> {
    assert!(subdiv > 0, "cube_face_directions: subdiv must be > 0");

    let mut dirs = Vec::with_capacity((6 * subdiv * subdiv) as usize);

    for face_idx in 0u32..6 {
        for cv in 0u32..subdiv {
            for cu in 0u32..subdiv {
                // Map cell (cu, cv) on this face to [-1, 1] range
                // Use cell CENTER: (cu + 0.5) / subdiv scaled to [-1, 1]
                let u = 2.0 * (cu as f32 + 0.5) / subdiv as f32 - 1.0;
                let v = 2.0 * (cv as f32 + 0.5) / subdiv as f32 - 1.0;

                // Construct point on cube face, then normalize to unit sphere
                let pt = cube_face_point(face_idx, u, v);
                let dir = pt.normalize_or_zero();
                dirs.push(dir);
            }
        }
    }

    dirs
}

/// Construct a 3D point on a cube face from face index and (u,v) coordinates.
///
/// Face layout (u = horizontal, v = vertical on the face):
///   0: +X  → ( +1,  u,  v )
///   1: −X  → ( −1,  u,  v )
///   2: +Y  → (  u, +1,  v )
///   3: −Y  → (  u, −1,  v )
///   4: +Z  → (  u,  v, +1 )
///   5: −Z  → (  u,  v, −1 )
///
/// The caller is responsible for normalizing the result.
#[inline]
pub fn cube_face_point(face_idx: u32, u: f32, v: f32) -> Vec3 {
    match face_idx {
        0 => Vec3::new( 1.0,  u,   v  ),  // +X
        1 => Vec3::new(-1.0,  u,   v  ),  // −X
        2 => Vec3::new( u,   1.0, v  ),  // +Y
        3 => Vec3::new( u,  -1.0, v  ),  // −Y
        4 => Vec3::new( u,   v,   1.0),  // +Z
        5 => Vec3::new( u,   v,  -1.0),  // −Z
        _ => Vec3::ZERO,
    }
}

/// Find the index of the nearest cube-face direction to a target direction.
///
/// This is the O(1) analytical inverse of `cube_face_directions`:
///   1. Find the dominant axis of `target` (max absolute component).
///   2. Determine face_idx (0..5) from the dominant axis.
///   3. Project target onto that face's tangent plane → (u, v).
///   4. Convert (u, v) to cell index (cu, cv) within the subdiv×subdiv grid.
///   5. Return the flat index: face_idx * subdiv² + cv * subdiv + cu.
///
/// **Panics:** if `subdiv == 0`.
#[inline]
pub fn nearest_cube_face_dir_index(target: Vec3, subdiv: u32) -> u32 {
    assert!(subdiv > 0, "nearest_cube_face_dir_index: subdiv must be > 0");
    let t = target.normalize_or_zero();

    // 1. Dominant axis → determines which cube face the direction belongs to
    let ax = t.x.abs();
    let ay = t.y.abs();
    let az = t.z.abs();

    let (face_idx, u, v) = if ax >= ay && ax >= az {
        if t.x >= 0.0 { (0u32, t.y, t.z) } else { (1u32, t.y, t.z) }
    } else if ay >= az {
        if t.y >= 0.0 { (2u32, t.x, t.z) } else { (3u32, t.x, t.z) }
    } else {
        if t.z >= 0.0 { (4u32, t.x, t.y) } else { (5u32, t.x, t.y) }
    };

    // 2. Project (u, v) onto [-1, 1] face coordinates
    //    (u, v) are components of the normalized target on the tangent plane,
    //    already in [-1, 1] range (since target is on the unit sphere).

    // 3. Map [-1, 1] → cell index [0, subdiv-1]
    let cu = ((u + 1.0) * 0.5 * subdiv as f32) as u32;
    let cv = ((v + 1.0) * 0.5 * subdiv as f32) as u32;

    // Clamp to valid range (guard against floating point edge cases)
    let cu = cu.min(subdiv - 1);
    let cv = cv.min(subdiv - 1);

    // 4. Flat index
    face_idx * subdiv * subdiv + cv * subdiv + cu
}

// ---------------------------------------------------------------------------
// Fibonacci Sphere
// ---------------------------------------------------------------------------

/// Generate `n` approximately uniform directions on the unit sphere
/// using the Fibonacci / Golden Spiral method.
///
/// The i-th point is computed as:
///   y = 1 - 2*(i + 0.5)/n          (polar coordinate, linear in cos(theta))
///   r = sqrt(1 - y*y)              (radius at height y)
///   theta = golden_angle * i       (azimuthal angle)
///   direction = (cos(theta)*r, y, sin(theta)*r)
///
/// **Properties:**
///   - Deterministic: same `n` always produces the same set of directions.
///   - Approximately uniform: minimizes clumping (O(n^{-1/2}) discrepancy).
///   - No extra memory: computed analytically, no lookup table needed.
///
/// **Parameters:**
///   - `n`: number of directions to generate. Must be > 0.
///
/// **Returns:** Vec of `n` normalized direction vectors.
///
/// **Panics:** if `n == 0`.
pub fn fibonacci_sphere_directions(n: u32) -> Vec<Vec3> {
    assert!(n > 0, "fibonacci_sphere_directions: n must be > 0");

    let mut dirs = Vec::with_capacity(n as usize);

    let inv_n = 1.0 / n as f32;

    for i in 0..n {
        // Polar coordinate: linear in cos(theta)
        let y = 1.0 - 2.0 * (i as f32 + 0.5) * inv_n;

        // Radius on the unit sphere at height y
        let radius_sq = 1.0 - y * y;
        let radius = radius_sq.sqrt().max(0.0); // Guard against rounding

        // Azimuthal angle (golden spiral)
        let theta = GOLDEN_ANGLE * i as f32;

        dirs.push(Vec3::new(
            theta.cos() * radius,
            y,
            theta.sin() * radius,
        ));
    }

    dirs
}

/// Generate a single Fibonacci sphere direction for index `i` out of `n`.
///
/// This is the scalar (non-allocating) version used by the WGSL shader
/// logic for testing and CPU-side validation.
///
/// **Parameters:**
///   - `i`: direction index (0-based).
///   - `n`: total number of directions.
///
/// **Returns:** Normalized direction vector.
#[inline]
pub fn fibonacci_direction(i: u32, n: u32) -> Vec3 {
    debug_assert!(n > 0);
    debug_assert!(i < n);

    let inv_n = 1.0 / n as f32;
    let y = 1.0 - 2.0 * (i as f32 + 0.5) * inv_n;
    let radius = (1.0 - y * y).sqrt().max(0.0);
    let theta = GOLDEN_ANGLE * i as f32;

    Vec3::new(theta.cos() * radius, y, theta.sin() * radius)
}

// ---------------------------------------------------------------------------
// Utility: nearest direction index
// ---------------------------------------------------------------------------

/// Find the index of the direction in `directions` that is closest to `target`.
///
/// Used by the Trilinear Fix to map a ray from a coarser cascade to the
/// nearest direction in a finer cascade.
///
/// **Returns:** index into `directions` of the closest direction.
///
/// **Panics:** if `directions` is empty.
pub fn nearest_direction_index(directions: &[Vec3], target: Vec3) -> usize {
    assert!(!directions.is_empty(), "directions must not be empty");

    let target = target.normalize_or_zero();
    let mut best_idx = 0;
    let mut best_dot = directions[0].dot(target);

    for (idx, &dir) in directions.iter().enumerate().skip(1) {
        let dot = dir.dot(target);
        if dot > best_dot {
            best_dot = dot;
            best_idx = idx;
        }
    }

    best_idx
}

// ---------------------------------------------------------------------------
// Utility: hemisphere sampling
// ---------------------------------------------------------------------------

/// Filter directions to only those in the upper hemisphere defined by `normal`.
///
/// Returns indices of directions whose dot product with `normal` is > 0.
/// Used by the fragment shader to accumulate GI from the upper hemisphere.
pub fn upper_hemisphere_indices(directions: &[Vec3], normal: Vec3) -> Vec<usize> {
    let normal = normal.normalize_or_zero();
    directions
        .iter()
        .enumerate()
        .filter(|(_, dir)| dir.dot(normal) > 0.0)
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_sphere_count() {
        let dirs = fibonacci_sphere_directions(12);
        assert_eq!(dirs.len(), 12);
    }

    #[test]
    fn test_fibonacci_sphere_unit_length() {
        let dirs = fibonacci_sphere_directions(100);
        for (i, d) in dirs.iter().enumerate() {
            let len = d.length();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "Direction {} has length {}, expected 1.0",
                i,
                len
            );
        }
    }

    #[test]
    fn test_fibonacci_sphere_coverage() {
        // With enough points, we should cover all octants of the sphere.
        let dirs = fibonacci_sphere_directions(100);
        let mut octants = [false; 8];

        for d in &dirs {
            let ox = if d.x >= 0.0 { 0 } else { 4 };
            let oy = if d.y >= 0.0 { 0 } else { 2 };
            let oz = if d.z >= 0.0 { 0 } else { 1 };
            octants[ox | oy | oz] = true;
        }

        for (i, &covered) in octants.iter().enumerate() {
            assert!(covered, "Octant {} not covered by 100 Fibonacci points", i);
        }
    }

    #[test]
    fn test_fibonacci_direction_matches_sphere() {
        let n = 24u32;
        let dirs = fibonacci_sphere_directions(n);
        for i in 0..n {
            let single = fibonacci_direction(i, n);
            let diff = (dirs[i as usize] - single).length();
            assert!(
                diff < 1e-6,
                "Direction {} mismatch: sphere vs scalar, diff = {}",
                i,
                diff
            );
        }
    }

    #[test]
    fn test_fibonacci_zero_panics() {
        // This should panic
        let result = std::panic::catch_unwind(|| fibonacci_sphere_directions(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_nearest_direction_index() {
        let dirs = fibonacci_sphere_directions(24);

        // The nearest direction to itself should be the same index
        for i in 0..dirs.len() {
            let idx = nearest_direction_index(&dirs, dirs[i]);
            assert_eq!(idx, i, "Nearest to dir[{}] should be {}, got {}", i, i, idx);
        }

        // The nearest direction to +Y should have the largest Y component
        let y_idx = nearest_direction_index(&dirs, Vec3::Y);
        let y_val = dirs[y_idx].y;
        for (i, d) in dirs.iter().enumerate() {
            assert!(
                d.y <= y_val + 1e-6,
                "dir[{}] y={} should be <= max y={}",
                i,
                d.y,
                y_val
            );
        }
    }

    #[test]
    fn test_upper_hemisphere_indices() {
        let dirs = fibonacci_sphere_directions(24);
        let normal = Vec3::Y;

        let upper = upper_hemisphere_indices(&dirs, normal);

        // All returned indices should have positive Y
        for &idx in &upper {
            assert!(
                dirs[idx].y > 0.0,
                "Upper hemisphere index {} has y={}",
                idx,
                dirs[idx].y
            );
        }

        // Should be roughly half (not exact due to discrete sampling)
        assert!(
            upper.len() >= 8 && upper.len() <= 16,
            "Expected ~12 upper hemisphere indices, got {}",
            upper.len()
        );
    }

    #[test]
    fn test_fibonacci_deterministic() {
        let a = fibonacci_sphere_directions(42);
        let b = fibonacci_sphere_directions(42);
        assert_eq!(a.len(), b.len());
        for (i, (da, db)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (*da - *db).length();
            assert!(diff < 1e-7, "Direction {} differs: diff={}", i, diff);
        }
    }
}
