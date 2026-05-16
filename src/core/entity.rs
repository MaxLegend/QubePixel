// =============================================================================
// QubePixel — Entity system
// =============================================================================
// Entities are world objects that exist independently of the block grid.
// Each entity has a world-space position, orientation, velocity, and references
// a named model type registered in EntityRenderer.
//
// The player is a special entity whose transform is driven by PlayerController;
// all other entities (NPCs, dropped items, etc.) live in EntityManager.

use glam::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EntityId
// ---------------------------------------------------------------------------

/// Unique identifier for a live entity.  Never reused within a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

// ---------------------------------------------------------------------------
// EntityTransform
// ---------------------------------------------------------------------------

/// World-space transform of one entity.
#[derive(Debug, Clone)]
pub struct EntityTransform {
    /// World-space position (foot / base point for humanoids).
    pub position: Vec3,
    /// Body yaw in radians (0 = +Z, increases clockwise from above).
    pub yaw: f32,
    /// Uniform scale factor (1.0 = normal size).
    pub scale: f32,
}

impl EntityTransform {
    pub fn at(position: Vec3) -> Self {
        Self { position, yaw: 0.0, scale: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// Entity
// ---------------------------------------------------------------------------

/// A live entity in the world (CPU data only — no GPU resources here).
#[derive(Debug, Clone)]
pub struct Entity {
    pub id:        EntityId,
    pub transform: EntityTransform,
    pub velocity:  Vec3,
    /// Identifies which model is registered in EntityRenderer.
    pub model_id:  String,
}

// ---------------------------------------------------------------------------
// EntityManager
// ---------------------------------------------------------------------------

/// Manages all active world entities.
pub struct EntityManager {
    entities: HashMap<EntityId, Entity>,
    next_id:  u64,
}

impl EntityManager {
    pub fn new() -> Self {
        Self { entities: HashMap::new(), next_id: 1 }
    }

    /// Spawn a new entity and return its assigned ID.
    pub fn spawn(&mut self, transform: EntityTransform, model_id: impl Into<String>) -> EntityId {
        let id = EntityId(self.next_id);
        self.next_id += 1;
        self.entities.insert(id, Entity {
            id,
            transform,
            velocity: Vec3::ZERO,
            model_id: model_id.into(),
        });
        id
    }

    /// Remove an entity.  A corresponding `despawn_instance` call on
    /// EntityRenderer is needed to free GPU resources.
    pub fn despawn(&mut self, id: EntityId) {
        self.entities.remove(&id);
    }

    pub fn get(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.values()
    }

    // -----------------------------------------------------------------------
    // Сохранение / восстановление
    // -----------------------------------------------------------------------

    /// Snapshot every managed entity into a list of serialisable DTOs.
    /// `skip_ids` — entity IDs to exclude (e.g. the player entity).
    pub fn to_save_data(
        &self,
        skip_ids: &[EntityId],
    ) -> Vec<crate::core::save_system::SavedEntity> {
        self.entities
            .values()
            .filter(|e| !skip_ids.contains(&e.id))
            .map(|e| crate::core::save_system::SavedEntity {
                model_id: e.model_id.clone(),
                position: [e.transform.position.x, e.transform.position.y, e.transform.position.z],
                yaw:      e.transform.yaw,
                scale:    e.transform.scale,
            })
            .collect()
    }

    /// Despawn all current entities (except those in `keep_ids`), then spawn
    /// every entity from `data`. Returns the new `EntityId`s in insertion order.
    pub fn restore_from_save(
        &mut self,
        data:     &[crate::core::save_system::SavedEntity],
        keep_ids: &[EntityId],
    ) -> Vec<EntityId> {
        // Despawn everything except kept entities
        let to_remove: Vec<EntityId> = self.entities
            .keys()
            .copied()
            .filter(|id| !keep_ids.contains(id))
            .collect();
        for id in to_remove {
            self.entities.remove(&id);
        }

        // Spawn saved entities
        let mut new_ids = Vec::with_capacity(data.len());
        for saved in data {
            let transform = EntityTransform {
                position: Vec3::new(saved.position[0], saved.position[1], saved.position[2]),
                yaw:      saved.yaw,
                scale:    saved.scale,
            };
            let id = self.spawn(transform, saved.model_id.clone());
            new_ids.push(id);
        }
        new_ids
    }
}
