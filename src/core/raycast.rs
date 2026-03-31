// =============================================================================
// QubePixel — DDA Voxel Raycast (Amanatides & Woo, 1987)
// =============================================================================
//
// Casts a ray from a point through the voxel grid and returns the first
// solid block hit, along with the face normal through which the ray entered.
//
// Used by: block highlight (Phase 2), block breaking (Phase 4),
//          block placement (Phase 5).
// =============================================================================

use glam::{IVec3, Vec3};
use crate::{debug_log, ext_debug_log};

/// Maximum distance (in world units) a raycast will travel.
pub const MAX_RAYCAST_DISTANCE: f32 = 8.0;

/// Hard cap on raycast iterations to prevent infinite loops.
const MAX_RAYCAST_STEPS: u32 = 256;

/// Result of a voxel raycast — the block the player is looking at.
#[derive(Debug, Clone, Copy)]
pub struct RaycastResult {
    /// World-space voxel position of the hit block.
    pub block_pos: IVec3,
    /// Normal of the face through which the ray entered the block.
    /// Points outward (towards the ray origin).
    /// Used for block placement: new_block_pos = block_pos + normal.
    pub normal: IVec3,
    /// Block type ID at the hit position.
    pub block_id: u8,
}

/// DDA voxel traversal (Amanatides & Woo, 1987).
///
/// Walks the ray voxel-by-voxel, checking `get_block` at each step.
/// Returns the **first solid block** (block_id != 0) encountered within
/// `max_distance` of the origin, together with the entry face normal.
///
/// # Parameters
/// - `origin` — ray start (world-space, typically camera position)
/// - `direction` — ray direction (will be normalised internally)
/// - `max_distance` — maximum travel distance in world units
/// - `get_block` — closure `(wx, wy, wz) -> u8`; returns 0 for air
pub fn dda_raycast(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    mut get_block: impl FnMut(i32, i32, i32) -> u8,
) -> Option<RaycastResult> {
    if direction.length_squared() < 1e-10 {
        return None;
    }
    let dir = direction.normalize();

    // Current voxel (the one containing the origin).
    let mut ix = origin.x.floor() as i32;
    let mut iy = origin.y.floor() as i32;
    let mut iz = origin.z.floor() as i32;

    // Step direction along each axis (+1 or -1).
    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    // How far along the ray we must travel to cross one full voxel on each axis.
    let t_delta_x = if dir.x.abs() > 1e-10 { 1.0 / dir.x.abs() } else { f32::INFINITY };
    let t_delta_y = if dir.y.abs() > 1e-10 { 1.0 / dir.y.abs() } else { f32::INFINITY };
    let t_delta_z = if dir.z.abs() > 1e-10 { 1.0 / dir.z.abs() } else { f32::INFINITY };

    // Parametric distance to the *first* voxel boundary on each axis.
    let mut t_max_x = if dir.x > 0.0 {
        (ix as f32 + 1.0 - origin.x) * t_delta_x
    } else if dir.x < 0.0 {
        (origin.x - ix as f32) * t_delta_x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dir.y > 0.0 {
        (iy as f32 + 1.0 - origin.y) * t_delta_y
    } else if dir.y < 0.0 {
        (origin.y - iy as f32) * t_delta_y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if dir.z > 0.0 {
        (iz as f32 + 1.0 - origin.z) * t_delta_z
    } else if dir.z < 0.0 {
        (origin.z - iz as f32) * t_delta_z
    } else {
        f32::INFINITY
    };

    for _step in 0..MAX_RAYCAST_STEPS {
        // Determine which axis boundary is closest and step through it.
        let face = if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                if t_max_x > max_distance { break; }
                ix += step_x;
                t_max_x += t_delta_x;
                IVec3::new(-step_x, 0, 0)
            } else {
                if t_max_z > max_distance { break; }
                iz += step_z;
                t_max_z += t_delta_z;
                IVec3::new(0, 0, -step_z)
            }
        } else {
            if t_max_y < t_max_z {
                if t_max_y > max_distance { break; }
                iy += step_y;
                t_max_y += t_delta_y;
                IVec3::new(0, -step_y, 0)
            } else {
                if t_max_z > max_distance { break; }
                iz += step_z;
                t_max_z += t_delta_z;
                IVec3::new(0, 0, -step_z)
            }
        };

        // Check the voxel we just stepped into.
        let block_id = get_block(ix, iy, iz);
        if block_id != 0 {
            ext_debug_log!(
                "Raycast", "dda_raycast",
                "Hit block_id={} at ({}, {}, {}) normal=({}, {}, {})",
                block_id, ix, iy, iz, face.x, face.y, face.z
            );
            return Some(RaycastResult {
                block_pos: IVec3::new(ix, iy, iz),
                normal:    face,
                block_id,
            });
        }
    }

    debug_log!("Raycast", "dda_raycast", "No block hit within distance");
    None
}