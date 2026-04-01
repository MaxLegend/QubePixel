// =============================================================================
// QubePixel — GameScreen  (3D world + HUD + debug overlay + profiler)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, ext_debug_log, flow_debug_log};
use std::time::Instant;

use crate::screens::debug_overlay::DebugOverlay;
use crate::screens::profiler_overlay::{ProfilerOverlay, ProfilerFrame};
use crate::screens::game_3d_pipeline::{Camera, Game3DPipeline};
use std::collections::VecDeque;
use crate::core::upload_worker::{UploadWorker, UploadJob, UploadPacked};
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
use egui::{Color32, FontId, Pos2, Rect, Stroke, Vec2, Align2, Rounding};
use crate::core::config;


const MAX_GPU_UPLOADS_PER_FRAME: usize = 32;


pub struct GameScreen {
    // -- 3D ------------------------------------------------------------------
    camera:      Camera,
    pipeline_3d: Option<Game3DPipeline>,
    world_worker: WorldWorker,
    atlas: TextureAtlas,
    pending_world_result: Option<WorldResult>,
    last_upload_us: u128,
    upload_worker: UploadWorker,
    upload_queue: VecDeque<UploadPacked>,
    // -- Debug + Profiler ----------------------------------------------------
    debug:    DebugOverlay,
    profiler: ProfilerOverlay,

    // -- Profiler accumulators (fed from WorldResult + local measurements) ----
    last_gen_time_us:      u128,
    last_mesh_time_us:     u128,
    last_gen_count:        usize,
    last_dirty_count:      usize,
    last_upload_count:     usize,
    last_worker_total_us:  u128,
    last_lod_counts:       [usize; 3],
    last_worker_pending:   usize,

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
    paused:        bool,
    // -- Transition ----------------------------------------------------------
    pending_action: ScreenAction,
}

impl GameScreen {
    pub fn new() -> Self {
        debug_log!("GameScreen", "new", "Creating GameScreen");

        let atlas       = TextureAtlas::load();
        let atlas_layout = atlas.layout().clone();

        let mut physics_world = PhysicsWorld::new();
        let registry = BlockRegistry::load();
        let player   = PlayerController::new(&mut physics_world, registry);

        Self {
            camera:      Camera::new(840, 480),
            pipeline_3d: None,
            world_worker: WorldWorker::new(atlas_layout),
            pending_world_result: None,
            last_upload_us: 0,
            upload_worker: UploadWorker::new(),
            upload_queue: VecDeque::new(),
            atlas,

            debug:    DebugOverlay::new(),
            profiler: ProfilerOverlay::new(),

            last_gen_time_us:      0,
            last_mesh_time_us:     0,
            last_gen_count:        0,
            last_dirty_count:      0,
            last_upload_count:     0,
            last_worker_total_us:  0,
            last_lod_counts:       [0; 3],
            last_worker_pending:   0,

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
            paused:        false,
            pending_block_ops: Vec::new(),
            pending_action: ScreenAction::None,
        }
    }

    fn process_input(&mut self, _dt: f64) {
        if self.camera_locked && (self.mouse_dx != 0.0 || self.mouse_dy != 0.0) {
            self.camera.rotate(self.mouse_dx, self.mouse_dy);
        }
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;

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

        let jump     = self.keys.contains(&PhysicalKey::Code(KeyCode::Space));
        let fly_down = self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft));
        self.player.apply_movement(&mut self.physics_world, move_dir, jump, fly_down);
    }
}

impl Screen for GameScreen {
    fn name(&self) -> &str { "Game" }
    fn load(&mut self) { debug_log!("GameScreen", "load", "Loading resources"); }
    fn start(&mut self) { debug_log!("GameScreen", "start", "First-frame init"); }

    fn update(&mut self, dt: f64) {
        let t_update_start = Instant::now();

        self.process_input(dt);

        // Physics step
        self.physics_world.step(dt as f32);
        self.player.update(&mut self.physics_world);

        // Camera sync
        self.camera.position = self.player.eye_position(&self.physics_world);

        // World update request

        let cam_fwd = self.camera.forward();
        let ray_dir = if self.camera_locked { Some(cam_fwd) } else { None };
        self.world_worker.request_update(
            self.camera.position,
            cam_fwd,
            ray_dir,
            std::mem::take(&mut self.pending_block_ops),
        );

        // Non-blocking receive
        if let Some(result) = self.world_worker.try_recv_result() {
            self.target_block = result.raycast_result;
            self.physics_world.sync_chunks(&result.physics_chunks, &result.evicted);

            // Capture profiler data from WorldResult
            self.last_gen_time_us     = result.gen_time_us;
            self.last_mesh_time_us    = result.mesh_time_us;
            self.last_gen_count       = result.meshes.len();
            self.last_dirty_count     = result.dirty_count;
            self.last_worker_total_us = result.worker_total_us;
            self.last_lod_counts      = result.lod_counts;
            self.last_worker_pending  = result.pending;

            self.pending_world_result = Some(result);
        }

        let update_ms = t_update_start.elapsed().as_secs_f64() * 1000.0;

        // Update debug overlay
        let vram = self.pipeline_3d.as_ref().map_or(0, |p| p.vram_usage());
        let fwd = self.camera.forward();
        let visible = self.pipeline_3d.as_ref().map_or(0, |p| {
            let total = p.gpu_chunk_count() as u32;
            total.saturating_sub(p.culled_last_frame)
        });
        let culled = self.pipeline_3d.as_ref().map_or(0, |p| p.culled_last_frame);

        self.debug.update(
            dt,
            vram,
            update_ms,
            [self.camera.position.x, self.camera.position.y, self.camera.position.z],
            [fwd.x, fwd.y, fwd.z],
            self.world_worker.chunk_count(),
            visible,
            culled,
            self.last_lod_counts,
        );
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

        self.atlas.ensure_gpu(device, queue);

        // --- Dispatch world result: eviction + submit to packing worker ---
        if let Some(result) = self.pending_world_result.take() {
            let pipeline = self.pipeline_3d.as_mut().unwrap();

            // Eviction
            if !result.evicted.is_empty() {
                pipeline.remove_chunk_meshes(&result.evicted);
            }

            // New meshes → UploadWorker (CPU packing)
            if !result.meshes.is_empty() {
                let jobs: Vec<UploadJob> = result.meshes
                    .into_iter()
                    .map(|(key, vertices, indices, aabb_min, aabb_max)| UploadJob {
                        key, vertices, indices, aabb_min, aabb_max,
                    })
                    .collect();
                self.upload_worker.submit(jobs);
            }
        }

        // --- Poll packed results into GPU upload queue ---
        {
            let packed_results = self.upload_worker.poll();
            for r in packed_results {
                self.upload_queue.push_back(r);
            }
        }

        // --- Budgeted GPU upload: at most MAX_GPU_UPLOADS_PER_FRAME per frame ---
        // Uses mapped_at_creation to bypass internal wgpu staging overhead.
        let upload_time_us;
        let upload_count;
        {
            let pipeline = self.pipeline_3d.as_mut().unwrap();
            let budget = MAX_GPU_UPLOADS_PER_FRAME.min(self.upload_queue.len());

            if budget > 0 {
                let t0 = Instant::now();

                for _ in 0..budget {
                    let r = self.upload_queue.pop_front().unwrap();

                    let vb_size = r.vertex_data.len() as u64;
                    let ib_size = r.index_data.len() as u64;

                    // Vertex buffer — mapped_at_creation skips write_buffer's
                    // internal staging allocation; we memcpy directly.
                    let vb = device.create_buffer(&wgpu::BufferDescriptor {
                        label:              Some("Chunk VB (async)"),
                        size:               vb_size,
                        usage:              wgpu::BufferUsages::VERTEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    vb.slice(..).get_mapped_range_mut()
                        .copy_from_slice(&r.vertex_data);
                    vb.unmap();

                    // Index buffer — same pattern
                    let ib = device.create_buffer(&wgpu::BufferDescriptor {
                        label:              Some("Chunk IB (async)"),
                        size:               ib_size,
                        usage:              wgpu::BufferUsages::INDEX
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: true,
                    });
                    ib.slice(..).get_mapped_range_mut()
                        .copy_from_slice(&r.index_data);
                    ib.unmap();

                    pipeline.insert_chunk_mesh(
                        r.key, vb, ib, r.index_count,
                        vb_size + ib_size,
                        r.aabb_min, r.aabb_max,
                    );
                }

                upload_time_us = t0.elapsed().as_micros();
                upload_count = budget;
            } else {
                upload_time_us = 0;
                upload_count = 0;
            }
        }

        self.last_upload_us = upload_time_us;
        self.last_upload_count = upload_count;

        self.last_upload_us = upload_time_us;
        self.last_upload_count = upload_count;

        let pipeline = self.pipeline_3d.as_mut().unwrap();
        self.camera.update_aspect(width, height);

        // 3D render
        let atlas_view = self.atlas.texture_view().unwrap();
        let render_us = pipeline.render(
            encoder, view, device, queue, &self.camera, width, height, atlas_view,
        );

        // Block outline
        pipeline.render_outline(
            encoder, view, device, queue, &self.camera, width, height,
            self.target_block.map(|r| r.block_pos),
        );

        // Feed profiler
        let vram = pipeline.vram_usage();
        self.profiler.feed(ProfilerFrame {
            gen_time_us:     self.last_gen_time_us,
            mesh_time_us:    self.last_mesh_time_us,
            upload_time_us:  upload_time_us,
            render_time_us:  render_us,
            worker_total_us: self.last_worker_total_us,
            gen_count:       self.last_gen_count,
            dirty_count:     self.last_dirty_count,
            upload_count:    upload_count,
            total_chunks:    self.world_worker.chunk_count(),
            visible_chunks:  pipeline.gpu_chunk_count() as u32
                - pipeline.culled_last_frame,
            culled_chunks:   pipeline.culled_last_frame,
            lod0_count:      self.last_lod_counts[0],
            lod1_count:      self.last_lod_counts[1],
            lod2_count:      self.last_lod_counts[2],
            worker_pending:  self.last_worker_pending,
            upload_pending:  self.upload_worker.pending_count(),
            gpu_queue_depth: self.upload_queue.len(),
            vram_bytes:      vram,
        });
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        let screen_rect = ctx.screen_rect();
        let sw = screen_rect.width();
        let sh = screen_rect.height();

        let layer_id = egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("game_hud"),
        );
        let painter = ctx.layer_painter(layer_id);

        // =================== Crosshair ======================================
        let cx = sw / 2.0;
        let cy = sh / 2.0;
        let cross_color = Color32::from_rgba_unmultiplied(255, 255, 255, 128);
        let cross_stroke = Stroke::new(3.0, cross_color);

        painter.line_segment(
            [Pos2::new(cx - 12.0, cy), Pos2::new(cx + 12.0, cy)],
            cross_stroke,
        );
        painter.line_segment(
            [Pos2::new(cx, cy - 12.0), Pos2::new(cx, cy + 12.0)],
            cross_stroke,
        );

        // =================== Cursor lock hint (top center) ==================
        let lock_text = if self.camera_locked {
            "J - unlock cursor"
        } else {
            "J - lock cursor / enable camera"
        };
        let lock_color = if self.camera_locked {
            Color32::from_rgb(102, 255, 102)
        } else {
            Color32::from_rgb(255, 255, 102)
        };
        painter.text(
            Pos2::new(sw / 2.0, 8.0),
            Align2::CENTER_TOP,
            lock_text,
            FontId::proportional(16.0),
            lock_color,
        );

        // =================== Controls (bottom right) ========================
        painter.text(
            Pos2::new(sw - 10.0, sh - 24.0),
            Align2::RIGHT_BOTTOM,
            "WASD move  Space jump",
            FontId::proportional(14.0),
            Color32::from_rgba_unmultiplied(255, 255, 255, 102),
        );

        // =================== Hotbar (bottom center) =========================
        let n   = self.player.slot_count();
        let cur = self.player.selected_slot();
        if n > 0 {
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

            painter.text(
                Pos2::new(sw / 2.0, sh - 50.0),
                Align2::CENTER_BOTTOM,
                &hotbar,
                FontId::proportional(18.0),
                Color32::from_rgb(255, 255, 153),
            );

            painter.text(
                Pos2::new(sw / 2.0, sh - 32.0),
                Align2::CENTER_BOTTOM,
                "Scroll or 1-9 to select",
                FontId::proportional(14.0),
                Color32::from_rgba_unmultiplied(204, 204, 204, 102),
            );
        }

        // =================== Debug overlay (left) ===========================
        self.debug.draw_egui(&painter);

        // =================== Profiler overlay (right) =======================
        self.profiler.draw_egui(&painter, sw);

        // =================== Pause menu =====================================
        if self.paused {
            let painter_bg = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Middle,
                egui::Id::new("pause_dimmer"),
            ));
            painter_bg.rect_filled(
                screen_rect,
                0.0,
                Color32::from_black_alpha(160),
            );

            let mut open = true;
            egui::Window::new("Pause Menu")
                .title_bar(false)
                .resizable(false)
                .movable(false)
                .collapsible(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .default_width(320.0)
                .frame(egui::Frame {
                    fill: Color32::from_rgb(30, 30, 45),
                    corner_radius: egui::CornerRadius::same(8),
                    stroke: Stroke::new(1.5, Color32::from_rgb(80, 80, 120)),
                    inner_margin: egui::Margin::same(20),
                    ..Default::default()
                })
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {


                        ui.add_space(20.0);

                        // --- Vertical Below slider ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(320.0);

                            ui.label(
                                egui::RichText::new("Chunks Below Camera")
                                    .size(18.0)
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                            ui.add_space(8.0);

                            let mut vb = config::vertical_below() as f32;
                            let slider = egui::Slider::new(&mut vb, 1.0..=16.0)
                                .step_by(1.0)
                                .suffix(" chunks")
                                .text("Below");
                            if ui.add(slider).changed() {
                                config::set_vertical_below(vb as i32);
                                debug_log!("SettingsScreen", "build_ui",
                            "Vertical below -> {}", vb as i32);
                            }

                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(
                                    "How many chunk layers load below the camera. \
                             Lower = better underground performance."
                                )
                                    .size(12.0)
                                    .color(egui::Color32::GRAY),
                            );
                        });

                        ui.add_space(20.0);

                        // --- Vertical Above slider ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(320.0);

                            ui.label(
                                egui::RichText::new("Chunks Above Camera")
                                    .size(18.0)
                                    .color(egui::Color32::LIGHT_GRAY),
                            );
                            ui.add_space(8.0);

                            let mut va = config::vertical_above() as f32;
                            let slider = egui::Slider::new(&mut va, 1.0..=16.0)
                                .step_by(1.0)
                                .suffix(" chunks")
                                .text("Above");
                            if ui.add(slider).changed() {
                                config::set_vertical_above(va as i32);
                                debug_log!("SettingsScreen", "build_ui",
                            "Vertical above -> {}", va as i32);
                            }

                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(
                                    "How many chunk layers load above the camera. \
                             Lower = fewer sky chunks loaded."
                                )
                                    .size(12.0)
                                    .color(egui::Color32::GRAY),
                            );
                        });
                        ui.add_space(8.0);

                        ui.label(
                            egui::RichText::new("Settings")
                                .size(28.0)
                                .color(Color32::WHITE),
                        );

                        ui.add_space(20.0);
                        ui.separator();
                        ui.add_space(16.0);

                        // --- Render Distance ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.label(
                                egui::RichText::new("Render Distance")
                                    .size(16.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut rd = config::render_distance() as f32;
                            let slider = egui::Slider::new(&mut rd, 1.0..=128.0)
                                .step_by(1.0)
                                .suffix(" chunks");
                            if ui.add(slider).changed() {
                                config::set_render_distance(rd as i32);
                            }
                        });

                        ui.add_space(16.0);

                        // --- LOD Distance Multiplier ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.label(
                                egui::RichText::new("LOD Distance")
                                    .size(16.0)
                                    .color(Color32::LIGHT_GRAY),
                            );
                            ui.add_space(6.0);

                            let mut lod_m = config::lod_multiplier();
                            let slider = egui::Slider::new(&mut lod_m, 0.25..=4.0)
                                .step_by(0.25)
                                .suffix("x");
                            if ui.add(slider).changed() {
                                config::set_lod_multiplier(lod_m);
                            }

                            ui.add_space(4.0);
                            ui.label(
                                egui::RichText::new(
                                    format!(
                                        "LOD0 ≤{:.0}  LOD1 ≤{:.0}  LOD2 beyond",
                                        config::LOD_NEAR_BASE * lod_m,
                                        config::LOD_FAR_BASE * lod_m,
                                    )
                                )
                                .size(12.0)
                                .color(Color32::GRAY),
                            );
                        });

                        ui.add_space(16.0);

                        // --- Debug / Profiler toggles ---
                        ui.vertical_centered(|ui| {
                            ui.set_width(260.0);

                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.debug.visible, "Debug Overlay");
                                ui.checkbox(&mut self.profiler.visible, "Profiler");
                            });
                        });

                        ui.add_space(24.0);
                        ui.separator();
                        ui.add_space(16.0);

                        // --- Resume ---
                        let resume_btn = egui::Button::new(
                            egui::RichText::new("Resume").size(18.0),
                        )
                            .min_size(egui::vec2(200.0, 42.0))
                            .fill(Color32::from_rgb(40, 160, 60));

                        if ui.add(resume_btn).clicked() {
                            self.paused = false;
                        }

                        ui.add_space(10.0);

                        // --- Back to Menu ---
                        let back_btn = egui::Button::new(
                            egui::RichText::new("Back to Menu").size(18.0),
                        )
                            .min_size(egui::vec2(200.0, 42.0))
                            .fill(Color32::from_rgb(60, 60, 140));

                        if ui.add(back_btn).clicked() {
                            self.pending_action = ScreenAction::Switch(
                                Box::new(
                                    crate::screens::main_menu::MainMenuScreen::new()
                                ),
                            );
                        }
                    });
                });

            if !open {
                self.paused = false;
            }
        }
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

            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p)   => p.y as f32 / 30.0,
                };
                if dy != 0.0 {
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
                                KeyCode::KeyN => {
                                    self.player.toggle_fly(&mut self.physics_world);
                                }
                                KeyCode::KeyJ => {
                                    self.camera_locked = !self.camera_locked;
                                    self.mouse_dx = 0.0;
                                    self.mouse_dy = 0.0;
                                    debug_log!("GameScreen","on_event",
                                        "Camera lock: {}", self.camera_locked);
                                }
                                KeyCode::Escape => {
                                    self.paused = !self.paused;
                                    if self.paused {
                                        self.camera_locked = false;
                                        self.mouse_dx = 0.0;
                                        self.mouse_dy = 0.0;
                                    }
                                }
                                KeyCode::F3 => {
                                    self.debug.visible = !self.debug.visible;
                                }
                                KeyCode::F4 => {
                                    self.profiler.visible = !self.profiler.visible;
                                }
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
