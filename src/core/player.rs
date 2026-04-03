// =============================================================================
// QubePixel — PlayerController  (движение, прыжок, выбор и взаимодействие с блоками)
// =============================================================================

use glam::Vec3;
use rapier3d::prelude::{RigidBodyHandle, ColliderHandle};

use crate::{debug_log, ext_debug_log, flow_debug_log};
use crate::core::physics::PhysicsWorld;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::raycast::RaycastResult;
use crate::core::gameobjects::world::BlockOp;

// ---------------------------------------------------------------------------
// Константы игрока
// ---------------------------------------------------------------------------
pub const PLAYER_HALF_W: f32     = 0.3;   // полуширина кубоида (итого 0.6)
pub const PLAYER_HALF_H: f32     = 0.9;   // полувысота кубоида (итого 1.8)
pub const PLAYER_EYE_OFFSET: f32 = 1.62;  // высота глаз над центром тела
const PLAYER_JUMP_VEL: f32       = 7.5;
const PLAYER_SPEED: f32          = 5.0;
const SPAWN: Vec3 = Vec3::new(8.0, 25.0, 30.0);
const FLY_SPEED: f32 = 12.0;
// ---------------------------------------------------------------------------
// PlayerController
// ---------------------------------------------------------------------------
pub struct PlayerController {
    body_handle:   RigidBodyHandle,
    collider_handle: ColliderHandle,
    on_ground:     bool,
    fly_mode:      bool,

    // -- Хотбар ---------------------------------------------------------
    pub registry:       BlockRegistry,
    selected_slot:  usize,
    hotbar_ids:     Vec<u8>,     // numeric id из registry
    hotbar_names:   Vec<String>, // строковый id для отображения
}

impl PlayerController {
    /// Создаёт физическое тело игрока через `physics` и инициализирует хотбар.
    pub fn new(physics: &mut PhysicsWorld, registry: BlockRegistry) -> Self {
        debug_log!("PlayerController", "new", "Creating player body + hotbar");

        let (body_handle, collider_handle) = physics.create_box_body(
            SPAWN, PLAYER_HALF_W, PLAYER_HALF_H,
        );
        // Хотбар — все solid-блоки из registry
        let mut hotbar_ids   = Vec::new();
        let mut hotbar_names = Vec::new();
        for id in 1u8..=255 {
            if let Some(def) = registry.get(id) {
                if def.solid {
                    hotbar_ids.push(id);
                    hotbar_names.push(def.id.clone());
                }
            }
        }
        if hotbar_ids.is_empty() {
            hotbar_ids.push(1);
            hotbar_names.push("block".to_owned());
        }

        debug_log!(
            "PlayerController", "new",
            "Hotbar ready: {} block type(s)", hotbar_ids.len()
        );

        Self {
            body_handle,
            collider_handle,
            fly_mode: false,
            on_ground: false,
            registry,
            selected_slot: 0,
            hotbar_ids,
            hotbar_names,
        }
    }
    // -----------------------------------------------------------------------
    // Режим свободной камеры (NOC / Fly)
    // -----------------------------------------------------------------------

    /// Переключает режим свободной камеры.
    /// В режиме полёта гравитация отключается, игрок может летать в любом
    /// направлении (Space — вверх, Shift — вниз), коллизии сохраняются.
    pub fn toggle_fly(&mut self, physics: &mut PhysicsWorld) {
        self.fly_mode = !self.fly_mode;
        if self.fly_mode {
            physics.set_body_gravity_scale(self.body_handle, 0.0);
            physics.enable_collider(self.collider_handle, false);
            physics.set_linvel(self.body_handle, Vec3::ZERO);
            debug_log!("PlayerController", "toggle_fly", "Fly mode ON (no collisions)");
        } else {
            physics.set_body_gravity_scale(self.body_handle, 1.0);
            physics.enable_collider(self.collider_handle, true);
            physics.set_linvel(self.body_handle, Vec3::ZERO);
            debug_log!("PlayerController", "toggle_fly", "Fly mode OFF");
        }
    }

    pub fn is_fly_mode(&self) -> bool { self.fly_mode }
    // -----------------------------------------------------------------------
    // Движение и прыжок — вызывать ДО physics.step()
    // -----------------------------------------------------------------------

    /// Устанавливает скорость тела.
    /// Горизонтальные составляющие берутся напрямую из ввода (FPS-движение).
    /// При friction=0 на коллайдере rapier сам убирает компонент, упирающийся в стену,
    /// оставляя параллельный — это и есть скольжение по стенам.
    /// Устанавливает скорость тела.
    /// В обычном режиме — горизонтальное FPS-движение + прыжок с гравитацией.
    /// В режиме полёта — прямое управление во всех направлениях, гравитация = 0.
    pub fn apply_movement(
        &mut self,
        physics:   &mut PhysicsWorld,
        move_dir:  Vec3,
        jump:      bool,
        fly_down:  bool,
    ) {
        if self.fly_mode {
            // ── Fly mode: free movement, no gravity ──
            let mut fly_dir = move_dir;
            if jump     { fly_dir.y += 1.0; }
            if fly_down { fly_dir.y -= 1.0; }

            let speed = if fly_dir.length_squared() > 1e-6 {
                fly_dir = fly_dir.normalize();
                FLY_SPEED
            } else {
                0.0
            };

            physics.set_linvel(
                self.body_handle,
                Vec3::new(fly_dir.x * speed, fly_dir.y * speed, fly_dir.z * speed),
            );
        } else {
            // ── Normal mode: FPS movement + gravity ──
            let speed  = if move_dir.length_squared() > 1e-6 { PLAYER_SPEED } else { 0.0 };
            let cur_vy = physics.get_linvel(self.body_handle).y;

            let new_vy = if jump && self.on_ground {
                ext_debug_log!("PlayerController", "apply_movement", "Jump! vy={}", PLAYER_JUMP_VEL);
                PLAYER_JUMP_VEL
            } else if self.on_ground && cur_vy.abs() < 0.5 {
                0.0
            } else {
                cur_vy
            };

            physics.set_linvel(
                self.body_handle,
                Vec3::new(move_dir.x * speed, new_vy, move_dir.z * speed),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Пост-шаговое обновление — вызывать ПОСЛЕ physics.step()
    // -----------------------------------------------------------------------

    /// Пост-шаговое обновление — вызывать ПОСЛЕ physics.step().
    /// В режиме полёта пропускает проверку земли и void-respawn.
    pub fn update(&mut self, physics: &mut PhysicsWorld) {
        if self.fly_mode {
            flow_debug_log!(
                "PlayerController", "update",
                "FLY pos=({:.1},{:.1},{:.1})",
                self.player_position(physics).x,
                self.player_position(physics).y,
                self.player_position(physics).z,
            );
            return;
        }

        self.on_ground = physics.check_ground(self.body_handle, PLAYER_HALF_H);
        physics.void_respawn(self.body_handle, SPAWN);

        flow_debug_log!(
            "PlayerController", "update",
            "pos=({:.1},{:.1},{:.1}) on_ground={}",
            self.player_position(physics).x,
            self.player_position(physics).y,
            self.player_position(physics).z,
            self.on_ground,
        );
    }

    // -----------------------------------------------------------------------
    // Запросы позиции
    // -----------------------------------------------------------------------

    pub fn player_position(&self, physics: &PhysicsWorld) -> Vec3 {
        physics.get_translation(self.body_handle)
    }

    pub fn eye_position(&self, physics: &PhysicsWorld) -> Vec3 {
        self.player_position(physics) + Vec3::new(0.0, PLAYER_EYE_OFFSET, 0.0)
    }

    pub fn is_on_ground(&self) -> bool { self.on_ground }

    // -----------------------------------------------------------------------
    // Хотбар
    // -----------------------------------------------------------------------

    pub fn selected_block_id(&self) -> u8 {
        self.hotbar_ids.get(self.selected_slot).copied().unwrap_or(1)
    }

    pub fn selected_block_name(&self) -> &str {
        self.hotbar_names
            .get(self.selected_slot)
            .map(|s| s.as_str())
            .unwrap_or("?")
    }

    pub fn selected_slot(&self) -> usize { self.selected_slot }
    pub fn slot_count(&self) -> usize     { self.hotbar_ids.len() }

    pub fn slot_name(&self, n: usize) -> &str {
        self.hotbar_names.get(n).map(|s| s.as_str()).unwrap_or("?")
    }

    /// Scroll > 0 → следующий, < 0 → предыдущий.
    pub fn scroll_slot(&mut self, delta: f32) {
        if self.hotbar_ids.is_empty() { return; }
        let n = self.hotbar_ids.len();
        if delta > 0.0 {
            self.selected_slot = (self.selected_slot + 1) % n;
        } else if delta < 0.0 {
            self.selected_slot = (self.selected_slot + n - 1) % n;
        }
        debug_log!(
            "PlayerController", "scroll_slot",
            "slot {} = {}", self.selected_slot, self.selected_block_name()
        );
    }

    /// Прямой выбор слота (0-based), игнорирует невалидные индексы.
    pub fn select_slot(&mut self, slot: usize) {
        if slot < self.hotbar_ids.len() {
            self.selected_slot = slot;
            debug_log!(
                "PlayerController", "select_slot",
                "slot {} = {}", slot, self.selected_block_name()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Блочные операции
    // -----------------------------------------------------------------------

    /// Возвращает `BlockOp::Break` для цели (если есть).
    pub fn break_block(target: Option<&RaycastResult>) -> Option<BlockOp> {
        target.map(|t| BlockOp::Break { x: t.block_pos.x, y: t.block_pos.y, z: t.block_pos.z })
    }

    /// Возвращает `BlockOp::Place` для цели, если нет пересечения с AABB игрока.
    pub fn place_block(
        &self,
        target:  Option<&RaycastResult>,
        physics: &PhysicsWorld,
    ) -> Option<BlockOp> {
        let t = target?;
        let pp  = self.player_position(physics);
        let place = t.block_pos + t.normal;
        let px = place.x as f32;
        let py = place.y as f32;
        let pz = place.z as f32;
        let hw = PLAYER_HALF_W;
        let hh = PLAYER_HALF_H;

        // AABB overlap: кубик 1x1x1 vs AABB игрока
        let overlaps =
            px + 1.0 > pp.x - hw && px < pp.x + hw
                && py + 1.0 > pp.y - hh && py < pp.y + hh
                && pz + 1.0 > pp.z - hw && pz < pp.z + hw;

        if overlaps {
            debug_log!("PlayerController", "place_block", "Blocked: overlaps player");
            None
        } else {
            Some(BlockOp::Place {
                x: place.x, y: place.y, z: place.z,
                block_id: self.selected_block_id(),
            })
        }
    }
}