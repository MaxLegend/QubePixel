// =============================================================================
// QubePixel — GameScreen  (3D world + HUD + debug overlay)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, ext_debug_log, flow_debug_log};
use std::time::Instant;
use crate::screens::bitmap_font::BitmapFont;
use crate::screens::debug_overlay::DebugOverlay;
use crate::screens::game_3d_pipeline::{Camera, Game3DPipeline};
use crate::screens::ui_renderer::UiRenderer;
use crate::core::gameobjects::world::{WorldWorker, WorldResult, BlockOp};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::screens::settings::SettingsScreen;
use crate::core::gameobjects::texture_atlas::TextureAtlas;
use crate::core::raycast::RaycastResult;
use crate::core::physics::PhysicsWorld;
use crate::core::player::PlayerController;
use crate::core::gameobjects::block::BlockRegistry;
use glam::Vec3;
use crate::screens::button::Button;

pub struct GameScreen {
    // -- 3D ------------------------------------------------------------------
    camera:      Camera,
    pipeline_3d: Option<Game3DPipeline>,
    world_worker: WorldWorker,
    atlas: TextureAtlas,
    pending_world_result: Option<WorldResult>,
    last_upload_us: u128,

    // -- 2D ------------------------------------------------------------------
    ui:      Option<UiRenderer>,
    font:    Option<BitmapFont>,
    buttons: Vec<Button>,

    // -- Debug ---------------------------------------------------------------
    debug: DebugOverlay,

    // -- Physics + Player ----------------------------------------------------
    physics_world: PhysicsWorld,
    player:        PlayerController,

    // -- Input ---------------------------------------------------------------
    cursor_x:      f64,
    cursor_y:      f64,
    mouse_dx:      f64,
    mouse_dy:      f64,
    camera_locked: bool,
    #[allow(dead_code)]
    mouse_down:    bool,
    keys:          std::collections::HashSet<PhysicalKey>,
    target_block:  Option<RaycastResult>,
    pending_block_ops: Vec<BlockOp>,

    // -- Transition ----------------------------------------------------------
    pending_action: ScreenAction,
}

impl GameScreen {
    pub fn new() -> Self {
        debug_log!("GameScreen", "new", "Creating GameScreen");



        let atlas       = TextureAtlas::load();
        let atlas_layout = atlas.layout().clone();

        // PhysicsWorld не знает об игроке — создаём отдельно
        let mut physics_world = PhysicsWorld::new();
        // Загружаем registry для хотбара (WorldWorker загружает свой независимо)
        let registry = BlockRegistry::load();
        let player   = PlayerController::new(&mut physics_world, registry);

        Self {
            camera:      Camera::new(840, 480),
            pipeline_3d: None,
            world_worker: WorldWorker::new(atlas_layout),
            pending_world_result: None,
            last_upload_us: 0,
            atlas,

            ui:      None,
            font:    None,
            buttons: Vec::new(),

            debug: DebugOverlay::new(),

            physics_world,
            player,

            cursor_x:      0.0,
            cursor_y:      0.0,
            mouse_dx:      0.0,
            mouse_dy:      0.0,
            camera_locked: false,
            mouse_down:    false,
            keys:          std::collections::HashSet::new(),
            target_block:  None,
            pending_block_ops: Vec::new(),
            pending_action: ScreenAction::None,
        }
    }

    fn process_input(&mut self, _dt: f64) {
        // Вращение камеры — только когда заблокирован курсор
        if self.camera_locked && (self.mouse_dx != 0.0 || self.mouse_dy != 0.0) {
            self.camera.rotate(self.mouse_dx, self.mouse_dy);
        }
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

        // Горизонтальное направление движения (FPS)
        let forward_xz = {
            let f = self.camera.forward();
            let flat = Vec3::new(f.x, 0.0, f.z);
            let len = flat.length();
            if len > 1e-6 { flat / len } else { Vec3::ZERO }
        };
        let right_xz = forward_xz.cross(Vec3::Y);

        let mut move_dir = Vec3::ZERO;
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) { move_dir += forward_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) { move_dir -= forward_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) { move_dir -= right_xz; }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) { move_dir += right_xz; }

        let move_dir = if move_dir.length_squared() > 1e-6 {
            move_dir.normalize()
        } else {
            Vec3::ZERO
        };

        let jump = self.keys.contains(&PhysicalKey::Code(KeyCode::Space));
        self.player.apply_movement(&mut self.physics_world, move_dir, jump);
    }
}

impl Screen for GameScreen {
    fn name(&self) -> &str { "Game" }
    fn load(&mut self) { debug_log!("GameScreen", "load", "Loading resources"); }
    fn start(&mut self) { debug_log!("GameScreen", "start", "First-frame init"); }

    fn update(&mut self, dt: f64) {
        flow_debug_log!("GameScreen", "update", "dt={:.4}", dt);

        self.process_input(dt);

        // Шаг физики
        self.physics_world.step(dt as f32);

        // Пост-шаговое обновление игрока: земля, void-respawn
        self.player.update(&mut self.physics_world);

        // Синхронизация позиции камеры
        self.camera.position = self.player.eye_position(&self.physics_world);

        // Запрос обновления мира (фон)
        let ray_dir = if self.camera_locked { Some(self.camera.forward()) } else { None };
        // Передаём накопленные операции; WorldWorker аккумулирует их даже если занят
        self.world_worker.request_update(
            self.camera.position,
            ray_dir,
            std::mem::take(&mut self.pending_block_ops),
        );

        // Получаем результат (non-blocking)
        if let Some(result) = self.world_worker.try_recv_result() {
            self.target_block = result.raycast_result;
            self.physics_world.sync_chunks(&result.physics_chunks, &result.evicted);
            self.pending_world_result = Some(result);
        }

        let vram = self.pipeline_3d.as_ref().map_or(0, |p| p.vram_usage())
            + self.font.as_ref().map_or(0, |_| 16384 + 8 + 4096);
        self.debug.update(dt, vram);
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view:    &wgpu::TextureView,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        format:  wgpu::TextureFormat,
        width:   u32,
        height:  u32,
    ) {
        // Lazy-init
        if self.pipeline_3d.is_none() {
            self.pipeline_3d = Some(Game3DPipeline::new(device, queue, format));
        }
        if self.ui.is_none()   { self.ui   = Some(UiRenderer::new(device, format)); }
        if self.font.is_none() { self.font = Some(BitmapFont::new(device, format)); }

        self.atlas.ensure_gpu(device, queue);

        // Загрузка GPU-мешей чанков
        if let Some(result) = self.pending_world_result.take() {
            let pipeline = self.pipeline_3d.as_mut().unwrap();
            if !result.evicted.is_empty() { pipeline.remove_chunk_meshes(&result.evicted); }
            if !result.meshes.is_empty() {
                let us = pipeline.update_chunk_meshes(device, queue, result.meshes);
                self.last_upload_us = us;
                flow_debug_log!(
                    "GameScreen", "render",
                    "[PERF] upload={:.2}ms gen={:.2}ms mesh={:.2}ms",
                    us as f64 / 1000.0,
                    result.gen_time_us as f64 / 1000.0,
                    result.mesh_time_us as f64 / 1000.0,
                );
            }
        }

        let pipeline = self.pipeline_3d.as_mut().unwrap();
        let ui       = self.ui.as_ref().unwrap();
        let font     = self.font.as_ref().unwrap();

        self.camera.update_aspect(width, height);

        // 3D-рендер
        let atlas_view = self.atlas.texture_view().unwrap();
        let render_us = pipeline.render(
            encoder, view, device, queue, &self.camera, width, height, atlas_view,
        );
        // Обводка блока под прицелом
        pipeline.render_outline(
            encoder, view, device, queue, &self.camera, width, height,
            self.target_block.map(|r| r.block_pos),
        );

        flow_debug_log!(
            "GameScreen", "render",
            "[PERF] render={:.2}ms chunks={}", render_us as f64/1000.0, self.world_worker.chunk_count()
        );

        let w  = width  as f32;
        let h  = height as f32;
        let cx = w / 2.0;
        let cy = h / 2.0;

        // Прицел
        let cc     = [1.0f32, 1.0, 1.0, 0.5];
        let cross_h = (cx - 12.0, cy - 1.5, 24.0,  3.0, cc);
        let cross_v = (cx -  1.5, cy - 12.0,  3.0, 24.0, cc);



        let mut rects: Vec<(f32, f32, f32, f32, [f32; 4])> = vec![cross_h, cross_v];

        ui.draw_rects(encoder, view, device, queue, &rects, width, height);

        // -- Текстовые оверлеи -----------------------------------------------
        let ts = 2.0_f32;



        // Подсказка блокировки курсора (верх-центр)
        let lock_hint = if self.camera_locked { "J - unlock cursor" } else { "J - lock cursor / enable camera" };
        let lw = lock_hint.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(encoder, view, device, queue,
                       lock_hint, (w - lw) / 2.0, 8.0,
                       if self.camera_locked { [0.4, 1.0, 0.4, 0.9] } else { [1.0, 1.0, 0.4, 0.7] },
                       1.5, width, height);

        // Управление (низ-право)
        let hint = "WASD move  Space jump";
        let hw = hint.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(encoder, view, device, queue,
                       hint, w - hw - 10.0, h - 24.0,
                       [1.0, 1.0, 1.0, 0.4], 1.5, width, height);

        // -------- Хотбар (низ-центр) ----------------------------------------
        let n   = self.player.slot_count();
        let cur = self.player.selected_slot();
        if n > 0 {
            // Показываем: предыдущий | [ТЕКУЩИЙ] | следующий
            let prev = if cur == 0 { n - 1 } else { cur - 1 };
            let next = (cur + 1) % n;

            let hotbar = if n == 1 {
                format!("[{}]", self.player.selected_block_name().to_uppercase())
            } else {
                format!(
                    "< {} | [{}] | {} >",
                    self.player.slot_name(prev),
                    self.player.selected_block_name().to_uppercase(),
                    self.player.slot_name(next),
                )
            };
            let hotbar_scale = 1.5_f32;
            let hotbar_w = hotbar.len() as f32 * BitmapFont::char_width() * hotbar_scale;
            font.draw_text(encoder, view, device, queue,
                           &hotbar,
                           (w - hotbar_w) / 2.0,
                           h - 50.0,
                           [1.0, 1.0, 0.6, 0.9],
                           hotbar_scale, width, height);

            // Подсказка выбора (scroll/1-9)
            let sel_hint = "Scroll or 1-9 to select";
            let sh_w = sel_hint.len() as f32 * BitmapFont::char_width() * 1.2;
            font.draw_text(encoder, view, device, queue,
                           sel_hint, (w - sh_w) / 2.0, h - 32.0,
                           [0.8, 0.8, 0.8, 0.4], 1.2, width, height);
        }

        // Debug overlay
        self.debug.draw(font, encoder, view, device, queue, width, height);
    }

    fn post_process(&mut self, _dt: f64) {}

    fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx;
        self.mouse_dy += dy;
    }

    fn wants_pointer_lock(&self) -> bool { self.camera_locked }

    fn on_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
            }

            // Прокрутка — смена блока в хотбаре
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 / 30.0,
                };
                if dy != 0.0 {
                    // Scroll up → предыдущий блок; scroll down → следующий
                    self.player.scroll_slot(-dy);
                }
            }

            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.mouse_down = true;
                        if self.camera_locked {
                            if let Some(op) = PlayerController::break_block(self.target_block.as_ref()) {

                                self.pending_block_ops.push(op);
                            }
                        }
                    }
                    ElementState::Released => {
                        self.mouse_down = false;
                    }
                }
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Right,
                ..
            } => {
                if self.camera_locked {
                    if let Some(op) = self.player.place_block(
                        self.target_block.as_ref(), &self.physics_world,
                    ) {

                        self.pending_block_ops.push(op);
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => {
                        self.keys.insert(event.physical_key);
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::KeyJ => {
                                    self.camera_locked = !self.camera_locked;
                                    self.mouse_dx = 0.0;
                                    self.mouse_dy = 0.0;
                                    debug_log!("GameScreen","on_event",
                                        "Camera lock: {}", self.camera_locked);
                                }
                                KeyCode::Escape => {
                                    self.camera_locked = false;
                                    self.pending_action = ScreenAction::Push(
                                        Box::new(SettingsScreen::new())
                                    );
                                }
                                // Быстрый выбор слота 1-9
                                KeyCode::Digit1 => self.player.select_slot(0),
                                KeyCode::Digit2 => self.player.select_slot(1),
                                KeyCode::Digit3 => self.player.select_slot(2),
                                KeyCode::Digit4 => self.player.select_slot(3),
                                KeyCode::Digit5 => self.player.select_slot(4),
                                KeyCode::Digit6 => self.player.select_slot(5),
                                KeyCode::Digit7 => self.player.select_slot(6),
                                KeyCode::Digit8 => self.player.select_slot(7),
                                KeyCode::Digit9 => self.player.select_slot(8),
                                _ => {}
                            }
                        }
                    }
                    ElementState::Released => { self.keys.remove(&event.physical_key); }
                }
            }

            _ => {}
        }
    }

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.55, g: 0.65, b: 0.85, a: 1.0 }
    }
}