// =============================================================================
// QubePixel — GameScreen  (3D world + HUD + debug overlay)
// =============================================================================
//
// CHANGELOG vs original:
//   • World replaced with WorldWorker — chunk gen + mesh build runs on a
//     background thread; main thread never blocks on CPU-heavy world work.
//   • GPU upload is still on the main thread (wgpu requirement) but only for
//     dirty chunks, and only when data is ready (non-blocking poll).
//   • fog disabled in fragment shader (temporarily).
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::debug_log;
use crate::flow_debug_log;
use std::time::Instant;
use crate::screens::bitmap_font::BitmapFont;
use crate::screens::button::Button;
use crate::screens::debug_overlay::DebugOverlay;
use crate::screens::game_3d_pipeline::{Camera, Game3DPipeline};
use crate::screens::main_menu::MainMenuScreen;
use crate::screens::ui_renderer::UiRenderer;
use crate::core::gameobjects::world::{WorldWorker, WorldResult};
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use crate::screens::settings::SettingsScreen;

pub struct GameScreen {
    // -- 3D ------------------------------------------------------------------
    camera:      Camera,
    pipeline_3d: Option<Game3DPipeline>,
    /// Background thread that owns `World` and does chunk gen + mesh building.
    world_worker: WorldWorker,
    /// Completed world work waiting to be uploaded to GPU.
    pending_world_result: Option<WorldResult>,
    /// Profiling: last GPU upload time in microseconds.
    last_upload_us: u128,

    // -- 2D ------------------------------------------------------------------
    ui:      Option<UiRenderer>,
    font:    Option<BitmapFont>,
    buttons: Vec<Button>,

    // -- Debug ---------------------------------------------------------------
    debug: DebugOverlay,

    // -- Input ---------------------------------------------------------------
    cursor_x:      f64,
    cursor_y:      f64,
    /// Raw mouse delta accumulated since last `process_input`.
    /// Fed by `on_mouse_motion` (DeviceEvent) — works even when cursor is locked.
    mouse_dx:      f64,
    mouse_dy:      f64,
    /// `true` = camera rotation active (cursor locked + hidden).
    /// Toggled by pressing J.
    camera_locked: bool,
    #[allow(dead_code)]
    mouse_down:    bool,
    keys:          std::collections::HashSet<PhysicalKey>,

    // -- Transition ----------------------------------------------------------
    pending_action: ScreenAction,
}

impl GameScreen {
    pub fn new() -> Self {
        debug_log!("GameScreen", "new", "Creating GameScreen");

        let back_button = Button::new(
            0.0, 0.0, 200.0, 50.0,
            [0.6, 0.2, 0.2, 1.0],
            [0.8, 0.3, 0.3, 1.0],
            "Back to Menu",
        );

        Self {
            camera:      Camera::new(840, 480),
            pipeline_3d: None,
            world_worker: WorldWorker::new(),
            pending_world_result: None,
            last_upload_us: 0,

            ui:      None,
            font:    None,
            buttons: vec![back_button],

            debug: DebugOverlay::new(),

            cursor_x:      0.0,
            cursor_y:      0.0,
            mouse_dx:      0.0,
            mouse_dy:      0.0,
            camera_locked: false, // J to activate
            mouse_down:    false,
            keys:          std::collections::HashSet::new(),

            pending_action: ScreenAction::None,
        }
    }

    fn process_input(&mut self, dt: f64) {
        let speed = self.camera.speed * dt as f32;

        // WASD — moves in local camera space (forward includes pitch)
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyW)) {
            self.camera.move_forward(speed);
        }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyS)) {
            self.camera.move_forward(-speed);
        }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyA)) {
            self.camera.move_right(-speed);
        }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::KeyD)) {
            self.camera.move_right(speed);
        }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::Space)) {
            self.camera.move_up(speed);
        }
        if self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft))
            || self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftRight))
        {
            self.camera.move_up(-speed);
        }

        // Camera rotation — only when locked; delta came from DeviceEvent::MouseMotion
        if self.camera_locked && (self.mouse_dx != 0.0 || self.mouse_dy != 0.0) {
            self.camera.rotate(self.mouse_dx, self.mouse_dy);
        }
        // Always consume the accumulated delta
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
    }
}

impl Screen for GameScreen {
    fn name(&self) -> &str { "Game" }

    fn load(&mut self) {
        debug_log!("GameScreen", "load", "Loading GameScreen resources");
    }

    fn start(&mut self) {
        debug_log!("GameScreen", "start", "GameScreen first-frame init");
    }

    fn update(&mut self, dt: f64) {
        flow_debug_log!("GameScreen", "update", "dt={:.4}", dt);

        self.process_input(dt);

        // ★ Request background world update (no-op if previous is still running)
        self.world_worker.request_update(self.camera.position);

        // ★ Non-blocking poll — if the worker finished, stash the result
        if let Some(result) = self.world_worker.try_recv_result() {
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
            debug_log!("GameScreen", "render", "Game3DPipeline initialised lazily");
        }
        if self.ui.is_none() {
            self.ui = Some(UiRenderer::new(device, format));
            debug_log!("GameScreen", "render", "UiRenderer initialised lazily");
        }
        if self.font.is_none() {
            self.font = Some(BitmapFont::new(device, format));
            debug_log!("GameScreen", "render", "BitmapFont initialised lazily");
        }

        // ★ Upload GPU meshes when background worker delivers results
        if let Some(result) = self.pending_world_result.take() {
            let pipeline = self.pipeline_3d.as_mut().unwrap();

            if !result.evicted.is_empty() {
                pipeline.remove_chunk_meshes(&result.evicted);
            }
            if !result.meshes.is_empty() {
                let upload_us = pipeline.update_chunk_meshes(device, queue, result.meshes);
                self.last_upload_us = upload_us;

                flow_debug_log!(
                    "GameScreen", "render",
                    "[PERF] upload={:.2}ms worker_gen={:.2}ms worker_mesh={:.2}ms pending={}",
                    upload_us as f64 / 1000.0,
                    result.gen_time_us as f64 / 1000.0,
                    result.mesh_time_us as f64 / 1000.0,
                    result.pending
                );
            }
        }

        let pipeline = self.pipeline_3d.as_mut().unwrap();
        let ui       = self.ui.as_ref().unwrap();
        let font     = self.font.as_ref().unwrap();

        self.camera.update_aspect(width, height);

        // 3D pass
        let render_us = pipeline.render(encoder, view, device, queue, &self.camera, width, height);

        flow_debug_log!(
            "GameScreen", "render",
            "[PERF] frame: upload={:.2}ms render={:.2}ms chunks={}",
            self.last_upload_us as f64 / 1000.0,
            render_us as f64 / 1000.0,
            self.world_worker.chunk_count()
        );

        let w  = width  as f32;
        let h  = height as f32;
        let cx = w / 2.0;
        let cy = h / 2.0;

        // Crosshair
        let cc     = [1.0f32, 1.0, 1.0, 0.5];
        let cross_h = (cx - 12.0, cy - 1.5, 24.0,  3.0, cc);
        let cross_v = (cx -  1.5, cy - 12.0,  3.0, 24.0, cc);

        // Back button (bottom-centre)
        let btn = &mut self.buttons[0];
        btn.x = (w - btn.width)  / 2.0;
        btn.y = h - 70.0;

        let mut rects: Vec<(f32, f32, f32, f32, [f32; 4])> = vec![cross_h, cross_v];
        for b in &self.buttons {
            rects.push((b.x, b.y, b.width, b.height,
                        b.current_color(self.cursor_x, self.cursor_y)));
        }
        ui.draw_rects(encoder, view, device, queue, &rects, width, height);

        // Button labels
        let ts = 2.0_f32;
        for btn in &self.buttons {
            let tx = btn.x + (btn.width  - btn.label.len() as f32 * BitmapFont::char_width()  * ts) / 2.0;
            let ty = btn.y + (btn.height - BitmapFont::char_height() * ts) / 2.0;
            font.draw_text(encoder, view, device, queue,
                           &btn.label, tx, ty, [1.0, 1.0, 1.0, 1.0], ts, width, height);
        }

        // Camera-lock hint (shown in top centre)
        let lock_hint = if self.camera_locked {
            "J — unlock cursor"
        } else {
            "J — lock cursor / enable camera"
        };
        let lw = lock_hint.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            lock_hint, (w - lw) / 2.0, 8.0,
            if self.camera_locked { [0.4, 1.0, 0.4, 0.9] } else { [1.0, 1.0, 0.4, 0.7] },
            1.5, width, height,
        );

        // Controls hint (bottom-right)
        let hint   = "WASD move  Space/Shift up/down";
        let hw     = hint.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            hint, w - hw - 10.0, h - 24.0,
            [1.0, 1.0, 1.0, 0.4], 1.5, width, height,
        );

        // Debug overlay
        self.debug.draw(font, encoder, view, device, queue, width, height);
    }

    fn post_process(&mut self, _dt: f64) {}

    // -----------------------------------------------------------------------
    // on_mouse_motion — receives raw DeviceEvent delta (works when cursor locked)
    // -----------------------------------------------------------------------
    fn on_mouse_motion(&mut self, dx: f64, dy: f64) {
        self.mouse_dx += dx;
        self.mouse_dy += dy;
    }

    // -----------------------------------------------------------------------
    // wants_pointer_lock — tells main.rs to lock/hide the cursor
    // -----------------------------------------------------------------------
    fn wants_pointer_lock(&self) -> bool {
        self.camera_locked
    }

    fn on_event(&mut self, event: &WindowEvent) {
        match event {
            // Cursor position — needed for button hover; camera delta comes
            // from on_mouse_motion (DeviceEvent), NOT from here.
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
            }

            // Left mouse button — button hit-testing only
            WindowEvent::MouseInput { state, button: winit::event::MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed  => self.mouse_down = true,
                    ElementState::Released => {
                        self.mouse_down = false;
                        for btn in &self.buttons {
                            if btn.contains_point(self.cursor_x, self.cursor_y) {
                                debug_log!(
                                    "GameScreen", "on_event",
                                    "Button '{}' clicked", btn.label
                                );
                                if btn.label == "Back to Menu" {
                                    self.pending_action = ScreenAction::Switch(
                                        Box::new(MainMenuScreen::new())
                                    );
                                }
                            }
                        }
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => {
                        self.keys.insert(event.physical_key);

                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                // J — toggle camera lock
                                KeyCode::KeyJ => {
                                    self.camera_locked = !self.camera_locked;
                                    // Discard any stale delta accumulated while unlocked
                                    self.mouse_dx = 0.0;
                                    self.mouse_dy = 0.0;
                                    debug_log!(
                                        "GameScreen", "on_event",
                                        "Camera lock toggled: {}", self.camera_locked
                                    );
                                }
                                // Escape — back to main menu (also releases lock)
                                KeyCode::Escape => {
                                    debug_log!(
                                        "GameScreen", "on_event",
                                        "Escape — opening Settings"
                                         );
                                    self.camera_locked = false;  // отпускаем курсор до выхода
                                    self.pending_action = ScreenAction::Push(
                                        Box::new(SettingsScreen::new())
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                    ElementState::Released => {
                        self.keys.remove(&event.physical_key);
                    }
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
