// =============================================================================
// QubePixel — PhysicsWorld  (rapier3d 0.32, без игровой логики игрока)
// =============================================================================

use std::collections::HashMap;
use rapier3d::prelude::*;
use crate::{debug_log, ext_debug_log, flow_debug_log};
use glam::Vec3;

// ---------------------------------------------------------------------------
// Физические константы уровня мира
// ---------------------------------------------------------------------------
const GROUND_CHECK_DIST: f32 = 0.15;
/// Void boundary — must be well below the deepest possible terrain block (Y=0).
/// With columnar chunks the world bottom is Y=0, so anything below –64 is void.
pub const VOID_Y: f32    = -64.0;
const MAX_PHYSICS_DT: f32    = 1.0 / 20.0;
// -----------------------------------------------------------------------
// Управление свойствами тел
// -----------------------------------------------------------------------

/// Устанавливает масштаб гравитации для конкретного тела.
/// 0.0 = тело не подвержено гравитации (полёт), 1.0 = нормальная гравитация.

// ---------------------------------------------------------------------------
// Вспомогательные конвертеры glam ↔ nalgebra
// ---------------------------------------------------------------------------
#[inline]
pub(crate) fn to_vector(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

#[inline]
pub(crate) fn from_vector(v: &Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

// ---------------------------------------------------------------------------
// PhysicsChunkReady — pre-built collider, created off-thread
// ---------------------------------------------------------------------------

/// A chunk collider whose compound shape has already been built on a background
/// thread.  The main thread only needs to insert it into the collider set.
pub struct PhysicsChunkReady {
    pub key:      (i32, i32, i32),
    pub collider: Collider,
}

// Safety: Collider is Send because SharedShape is Arc<dyn Shape + Send + Sync>.
// rapier3d types are Send when not attached to a set.
unsafe impl Send for PhysicsChunkReady {}

// ---------------------------------------------------------------------------
// PhysicsWorld — только симуляция и коллайдеры чанков
// ---------------------------------------------------------------------------
pub struct PhysicsWorld {
    gravity:                Vector,
    integration_parameters: IntegrationParameters,
    pipeline:               PhysicsPipeline,
    island_manager:         IslandManager,
    broad_phase:            DefaultBroadPhase,
    narrow_phase:           NarrowPhase,
    impulse_joint_set:      ImpulseJointSet,
    multibody_joint_set:    MultibodyJointSet,
    ccd_solver:             CCDSolver,
    rigid_body_set:         RigidBodySet,
    collider_set:           ColliderSet,
    chunk_colliders:        HashMap<(i32, i32, i32), ColliderHandle>,
}

impl PhysicsWorld {
    pub fn new() -> Self {
        debug_log!("PhysicsWorld", "new", "Creating PhysicsWorld (player-free)");
        Self {
            gravity:                Vector::new(0.0, -20.0, 0.0),
            integration_parameters: IntegrationParameters::default(),
            pipeline:               PhysicsPipeline::new(),
            island_manager:         IslandManager::new(),
            broad_phase:            DefaultBroadPhase::new(),
            narrow_phase:           NarrowPhase::new(),
            impulse_joint_set:      ImpulseJointSet::new(),
            multibody_joint_set:    MultibodyJointSet::new(),
            ccd_solver:             CCDSolver::new(),
            rigid_body_set:         RigidBodySet::new(),
            collider_set:           ColliderSet::new(),
            chunk_colliders:        HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Управление телами
    // -----------------------------------------------------------------------

    /// Создаёт капсульный динамический коллайдер игрока.
    /// Капсула скользит по углам блоков, не застревая.
    /// friction=0 → тело скользит вдоль стен.
    pub fn create_box_body(
        &mut self,
        pos:    Vec3,
        half_w: f32,
        half_h: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let body = RigidBodyBuilder::dynamic()
            .translation(to_vector(pos))
            .lock_rotations()
            .linear_damping(0.0)
            .build();
        // Capsule: cylinder half-height = half_h - radius, so total height = 2*half_h.
        // This naturally slides over block corners instead of catching on them.
        let radius = half_w;
        let capsule_half_h = (half_h - radius).max(0.01);
        let collider = ColliderBuilder::capsule_y(capsule_half_h, radius)
            .friction(0.0)
            .restitution(0.0)
            .contact_skin(0.02)
            .build();
        let body_handle = self.rigid_body_set.insert(body);
        let collider_handle = self.collider_set.insert_with_parent(
            collider, body_handle, &mut self.rigid_body_set,
        );
        debug_log!(
            "PhysicsWorld", "create_box_body",
            "Capsule body at ({:.1},{:.1},{:.1}) hw={} hh={}", pos.x, pos.y, pos.z, half_w, half_h
        );
        (body_handle, collider_handle)
    }

    // -----------------------------------------------------------------------
    // Аксессоры тел
    // -----------------------------------------------------------------------

    pub fn get_translation(&self, handle: RigidBodyHandle) -> Vec3 {
        from_vector(&self.rigid_body_set[handle].translation())
    }

    pub fn set_translation(&mut self, handle: RigidBodyHandle, pos: Vec3) {
        self.rigid_body_set[handle].set_translation(to_vector(pos), true);
    }

    pub fn get_linvel(&self, handle: RigidBodyHandle) -> Vec3 {
        from_vector(&self.rigid_body_set[handle].linvel())
    }

    pub fn set_linvel(&mut self, handle: RigidBodyHandle, vel: Vec3) {
        self.rigid_body_set[handle].set_linvel(to_vector(vel), true);
    }

    // -----------------------------------------------------------------------
    // Проверка земли — исключает само тело из рейкаста
    // -----------------------------------------------------------------------

    /// Кастит луч вниз из-под нижней грани коллайдера.
    /// `half_h` — полувысота кубоида.
    /// Тело `handle` исключено из запроса — нет ложных попаданий в себя.
    pub fn check_ground(&self, handle: RigidBodyHandle, half_h: f32) -> bool {
        let pos = self.rigid_body_set[handle].translation();
        let ray_origin = pos - Vector::new(0.0, half_h + 0.05, 0.0);
        let ray = Ray::new(ray_origin, Vector::new(0.0, -1.0, 0.0));

        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default().exclude_rigid_body(handle),
        );
        qp.cast_ray(&ray, GROUND_CHECK_DIST, true).is_some()
    }

    // -----------------------------------------------------------------------
    // Void respawn
    // -----------------------------------------------------------------------

    pub fn void_respawn(&mut self, handle: RigidBodyHandle, spawn_pos: Vec3) {
        if self.rigid_body_set[handle].translation().y < VOID_Y {
            debug_log!("PhysicsWorld", "void_respawn", "Body below void — respawning");
            let body = &mut self.rigid_body_set[handle];
            body.set_translation(to_vector(spawn_pos), true);
            body.set_linvel(Vector::new(0.0, 0.0, 0.0), true);
        }
    }

    // -----------------------------------------------------------------------
    // Шаг симуляции
    // -----------------------------------------------------------------------

    pub fn step(&mut self, dt: f32) {
        let clamped = dt.min(MAX_PHYSICS_DT);
        self.integration_parameters.dt = clamped;
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
    pub fn set_body_gravity_scale(&mut self, handle: RigidBodyHandle, scale: f32) {
        if let Some(body) = self.rigid_body_set.get_mut(handle) {
            body.set_gravity_scale(scale, true);
        }
    }
    /// Включает/выключает коллайдер тела.
    /// При enabled=false тело не участвует в коллизиях — можно проходить сквозь блоки.
    pub fn enable_collider(&mut self, handle: ColliderHandle, enabled: bool) {
        if let Some(collider) = self.collider_set.get_mut(handle) {
            collider.set_enabled(enabled);
        }
    }

    // -----------------------------------------------------------------------
    // Stance support — replace the player collider with a different-sized one
    // and adjust body center so feet stay on the ground.
    // -----------------------------------------------------------------------

    /// Replace the player's cuboid collider with new half-sizes.
    /// Adjusts the body translation so the BOTTOM of the new box stays at the
    /// same world position as the bottom of the old box (feet don't teleport).
    ///
    /// Returns the new ColliderHandle (old one is removed).
    pub fn replace_player_collider(
        &mut self,
        body_handle:    RigidBodyHandle,
        old_collider:   ColliderHandle,
        old_half_h:     f32,
        new_half_w:     f32,
        new_half_h:     f32,
    ) -> ColliderHandle {
        // Keep feet position constant: feet = body_y - old_half_h
        // new body_y = feet + new_half_h = body_y - old_half_h + new_half_h
        let pos = self.rigid_body_set[body_handle].translation().clone();
        let new_y = pos.y - old_half_h + new_half_h;
        self.rigid_body_set[body_handle].set_translation(
            Vector::new(pos.x, new_y, pos.z), true,
        );

        self.collider_set.remove(
            old_collider,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            true,
        );

        // Capsule: cylinder half-height = half_h - radius, total height = 2*half_h.
        let radius = new_half_w;
        let capsule_half_h = (new_half_h - radius).max(0.01);
        let new_collider = ColliderBuilder::capsule_y(capsule_half_h, radius)
            .friction(0.0)
            .restitution(0.0)
            .contact_skin(0.02)
            .build();

        self.collider_set.insert_with_parent(
            new_collider, body_handle, &mut self.rigid_body_set,
        )
    }

    /// Returns true if a vertical ray of `height` upward from the body center
    /// top hits a solid block (i.e. there isn't room to grow to `target_half_h`).
    pub fn head_blocked(
        &self,
        body_handle:    RigidBodyHandle,
        current_half_h: f32,
        target_half_h:  f32,
    ) -> bool {
        let pos = self.rigid_body_set[body_handle].translation();
        // We need `target_half_h - current_half_h` more headroom above the box top.
        let extra = target_half_h - current_half_h;
        if extra <= 0.0 { return false; }
        let ray_start = pos + Vector::new(0.0, current_half_h, 0.0);
        let ray = Ray::new(ray_start, Vector::new(0.0, 1.0, 0.0));
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default().exclude_rigid_body(body_handle),
        );
        qp.cast_ray(&ray, extra + 0.05, true).is_some()
    }

    /// Проверяет, есть ли стена (твёрдый блок) в горизонтальном направлении
    /// от центра тела игрока. Используется для определения, стоит ли игрок
    /// вплотную к блоку — чтобы при прыжке обнулить горизонтальную скорость.
    ///
    /// `direction` — нормализованное горизонтальное направление.
    /// `half_w` — радиус капсулы (полуширина игрока).
    pub fn is_wall_in_direction(
        &self,
        body_handle: RigidBodyHandle,
        direction:   Vec3,
        half_w:      f32,
    ) -> bool {
        if direction.length_squared() < 1e-6 { return false; }
        let dir = direction.normalize();
        let pos = self.rigid_body_set[body_handle].translation();
        let ray = Ray::new(pos, Vector::new(dir.x, dir.y, dir.z));
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_body_set,
            &self.collider_set,
            QueryFilter::default().exclude_rigid_body(body_handle),
        );
        // Детектируем блоки, поверхность которых находится в пределах
        // half_w + WALL_DETECT_MARGIN от центра тела (т.е. вплотную к капсуле).
        const WALL_DETECT_MARGIN: f32 = 0.1;
        let max_dist = half_w + WALL_DETECT_MARGIN;
        qp.cast_ray(&ray, max_dist, true).is_some()
    }
    // -----------------------------------------------------------------------
    // Синхронизация коллайдеров чанков (insert-only: shapes pre-built off-thread)
    // -----------------------------------------------------------------------

    /// Accept pre-built colliders from the world-worker thread.
    /// Compound shape construction (the expensive O(N log N) BVH) already happened
    /// on the background thread; here we only insert / remove handles — O(1) each.
    pub fn sync_chunks_ready(
        &mut self,
        added:   Vec<PhysicsChunkReady>,
        removed: &[(i32, i32, i32)],
    ) {
        for key in removed {
            if let Some(handle) = self.chunk_colliders.remove(key) {
                self.collider_set.remove(
                    handle,
                    &mut self.island_manager,
                    &mut self.rigid_body_set,
                    false,
                );
                ext_debug_log!(
                    "PhysicsWorld", "sync_chunks_ready",
                    "Removed chunk collider ({},{},{})", key.0, key.1, key.2
                );
            }
        }

        let mut added_count = 0u32;
        for ready in added {
            if let Some(old) = self.chunk_colliders.remove(&ready.key) {
                self.collider_set.remove(
                    old, &mut self.island_manager, &mut self.rigid_body_set, false,
                );
            }

            let handle = self.collider_set.insert(ready.collider);
            self.chunk_colliders.insert(ready.key, handle);
            added_count += 1;
        }

        if added_count > 0 || !removed.is_empty() {
            flow_debug_log!(
                "PhysicsWorld", "sync_chunks_ready",
                "Synced: added={} removed={} total={}",
                added_count, removed.len(), self.chunk_colliders.len()
            );
        }
    }
}