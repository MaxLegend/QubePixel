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
use crate::core::lighting::{DayNightCycle, LightingConfig, pack_lighting_uniforms};
use crate::screens::sky_renderer::SkyRenderer;
use glam::Vec3;
use egui::{Color32, FontId, Pos2, Rect, Stroke, Vec2, Align2, Rounding};
use crate::core::config;
use crate::core::radiance_cascades::system::RadianceCascadeSystemGpu;
use crate::core::radiance_cascades::voxel_tex::fill_lut_from_closure;
use crate::core::radiance_cascades::types::BlockProperties;
use crate::core::volumetric_lights::{VolumetricLightPass, VolumetricLightData};

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
    // -- Lighting ------------------------------------------------------------
    day_night:       DayNightCycle,
    lighting_config: LightingConfig,
    sky_renderer:    SkyRenderer,
    // -- Transition ----------------------------------------------------------
    pending_action: ScreenAction,
    // В struct GameScreen:
    rc_system: Option<RadianceCascadeSystemGpu>,
    /// Whether the Block LUT has been populated (only needs to happen once).
    rc_lut_initialized: bool,
    rc_frame_counter: u32,
    last_voxel_data: Option<crate::core::gameobjects::world::VoxelUpdate>,
    // -- Volumetric lights ---------------------------------------------------
    volumetric_pass: Option<VolumetricLightPass>,
    /// Cached resolved volumetric light data (rebuilt when emissive_blocks changes).
    volumetric_lights: Vec<VolumetricLightData>,
    /// Seconds elapsed since game start (for animated god rays).
    elapsed_s: f32,
    // -- RC / pipeline profiler accumulators ---------------------------------
    /// Counts calls to rc.dispatch() — mirrors system.rs dispatch_counter
    /// to determine full vs partial dispatch without exposing RC internals.
    rc_dispatch_call_count:    u32,
    last_rc_dispatch_time_us:  u128,
    last_voxel_upload_time_us: u128,
    last_rc_dispatched:        bool,
    last_rc_was_full:          bool,
    last_sky_render_us:        u128,
    last_outline_us:           u128,
    last_frame_total_us:       u128,
    last_voxel_uploaded:       bool,
    last_upload_bytes:         u64,
    last_upload_throughput:    f64,   // MB/s (staging bandwidth)
    /// Chunks evicted from GPU VRAM by LRU budget; passed to world worker next frame.
    gpu_evicted_pending:       Vec<(i32, i32, i32)>,
}

impl GameScreen {
    pub fn new() -> Self {
        debug_log!("GameScreen", "new", "Creating GameScreen");

        let atlas       = TextureAtlas::load();
        let atlas_layout = atlas.layout().clone();

        let mut physics_world = PhysicsWorld::new();
        let registry = BlockRegistry::load();
        let player   = PlayerController::new(&mut physics_world, registry);

        let lighting_config = LightingConfig::load();
        let day_night = DayNightCycle::new(&lighting_config);

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

            day_night,
            lighting_config,
            sky_renderer: SkyRenderer::new(),
            rc_system: None,
            pending_action: ScreenAction::None,
            rc_lut_initialized: false,
            rc_frame_counter: 0,
            last_voxel_data: None,
            volumetric_pass: None,
            volumetric_lights: Vec::new(),
            elapsed_s: 0.0,
            rc_dispatch_call_count:    0,
            last_rc_dispatch_time_us:  0,
            last_voxel_upload_time_us: 0,
            last_rc_dispatched:        false,
            last_rc_was_full:          false,
            last_sky_render_us:        0,
            last_outline_us:           0,
            last_frame_total_us:       0,
            last_voxel_uploaded:       false,
            last_upload_bytes:         0,
            last_upload_throughput:    0.0,
            gpu_evicted_pending:       Vec::new(),
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

        // Advance day-night cycle
        self.day_night.update(dt);

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
            std::mem::take(&mut self.gpu_evicted_pending),
        );

        // Non-blocking receive
        if let Some(mut result) = self.world_worker.try_recv_result() {
            self.target_block = result.raycast_result;
            // Извлекаем воксел-снапшот для RC (take, чтобы не клонировать)
            self.last_voxel_data = result.voxel_data.take();
            self.physics_world.sync_chunks_ready(
                std::mem::take(&mut result.physics_ready),
                &result.evicted,
            );

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

        // Update debug overlay — minimal: FPS + GPU/CPU load + VRAM
        let vram = self.pipeline_3d.as_ref().map_or(0, |p| p.vram_usage());
        // cpu_work_ms = update + last frame encode (best CPU estimate available on CPU timeline)
        let cpu_work_ms = update_ms + self.last_frame_total_us as f64 / 1000.0;
        self.debug.update(dt, vram, cpu_work_ms);
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
        let t_frame_start = Instant::now();

        // Lazy-init 3D pipeline
        if self.pipeline_3d.is_none() {
            self.pipeline_3d = Some(Game3DPipeline::new(device, queue, format, None));
        }

        // Lazy-init Radiance Cascades GPU system
        if self.rc_system.is_none() {
            let rc = RadianceCascadeSystemGpu::new(device, queue);
            debug_log!("GameScreen", "render",
                "RC GPU system initialized ({:.1} MB estimated)",
                rc.estimated_gpu_memory() as f64 / (1024.0 * 1024.0));
            self.rc_system = Some(rc);
        }
        self.atlas.ensure_gpu(device, queue);
        // Lazy-init volumetric light pass
        if self.volumetric_pass.is_none() {
            self.volumetric_pass = Some(VolumetricLightPass::new(device, format));
        }
        // --- Dispatch world result: eviction + submit to packing worker ---
        if let Some(mut result) = self.pending_world_result.take() {
            let pipeline = self.pipeline_3d.as_mut().unwrap();

            // Eviction
            if !result.evicted.is_empty() {
                pipeline.remove_chunk_meshes(&result.evicted);
            }
            // Rebuild volumetric light list when emissive blocks changed
            if let Some(emissive) = result.emissive_blocks.take() {
                let registry = &self.player.registry;
                self.volumetric_lights = emissive
                    .into_iter()
                    .filter_map(|b| {
                        let def = registry.get(b.block_id)?;
                        if !def.volumetric.is_enabled() { return None; }
                        Some(VolumetricLightData {
                            position:       [b.wx, b.wy, b.wz],
                            color:          def.emission.light_color,
                            halo_enabled:   def.volumetric.halo_enabled,
                            halo_radius:    def.volumetric.halo_radius,
                            halo_intensity: def.volumetric.halo_intensity,
                            ray_enabled:    def.volumetric.ray_enabled,
                            ray_count:      def.volumetric.ray_count,
                            ray_length:     def.volumetric.ray_length,
                            ray_width:      def.volumetric.ray_width,
                            ray_intensity:  def.volumetric.ray_intensity,
                            ray_falloff:    def.volumetric.ray_falloff,
                        })
                    })
                    .collect();
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
                let mut total_bytes = 0u64;

                for _ in 0..budget {
                    let r = self.upload_queue.pop_front().unwrap();

                    let vb_size = r.vertex_data.len() as u64;
                    let ib_size = r.index_data.len() as u64;
                    total_bytes += vb_size + ib_size;

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

                    let evicted = pipeline.insert_chunk_mesh(
                        r.key, vb, ib, r.index_count,
                        vb_size + ib_size,
                        r.aabb_min, r.aabb_max,
                    );
                    self.gpu_evicted_pending.extend(evicted);
                }

                upload_time_us = t0.elapsed().as_micros();
                upload_count = budget;
                self.last_upload_bytes = total_bytes;
                // Staging throughput: bytes / time (MB/s)
                self.last_upload_throughput = if upload_time_us > 0 {
                    total_bytes as f64 / upload_time_us as f64 / 1.048576 // µs → MB/s
                } else {
                    0.0
                };
            } else {
                self.last_upload_bytes      = 0;
                self.last_upload_throughput = 0.0;
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

        // Sky billboard rendering (sun/moon) — before 3D terrain
        {
            let t = Instant::now();
            self.sky_renderer.render(
                encoder, view, device, queue, format,
                &self.camera, &self.day_night,
            );
            self.last_sky_render_us = t.elapsed().as_micros();
        }

        // ---- Radiance Cascades: throttled GI update ----
        // RC dispatches every RC_THROTTLE frames — GI changes slowly,
        // per-frame accuracy is not required and was the main GPU bottleneck.
        const RC_THROTTLE: u32 = 4;
        self.rc_frame_counter = self.rc_frame_counter.wrapping_add(1);

        if let Some(ref mut rc) = self.rc_system {
            rc.recenter(self.camera.position);

            // Voxel texture upload — now uses dirty-region tracking (Variant B).
            // Full = 32 MB only when camera crosses chunk boundary.
            // Partial = N × 8 KB for dirty chunks only (common case).
            if let Some(update) = self.last_voxel_data.take() {
                use crate::core::gameobjects::world::VoxelUpdate;
                let t_vox = Instant::now();
                match update {
                    VoxelUpdate::Full { origin, data } => {
                        rc.voxel_builder_mut().set_data_raw(origin, data);
                        rc.upload_voxel_texture(queue);
                        debug_log!("GameScreen", "render",
                            "Voxel FULL upload: origin=({},{},{})",
                            origin.x, origin.y, origin.z);
                    }
                    VoxelUpdate::Partial { origin: _, regions } => {
                        for region in &regions {
                            rc.upload_voxel_region(queue, region);
                        }
                        debug_log!("GameScreen", "render",
                            "Voxel PARTIAL upload: {} regions", regions.len());
                    }
                }
                self.last_voxel_upload_time_us = t_vox.elapsed().as_micros();
                self.last_voxel_uploaded = true;
            } else {
                self.last_voxel_uploaded = false;
            }

            // Block LUT — once at startup
            if !self.rc_lut_initialized {
                let lut = rc.lut_builder_mut();
                let registry = &self.player.registry;
                for id in 1u8..=255 {
                    if let Some(def) = registry.get(id) {
                        let props = if def.transparent {
                            BlockProperties::air()
                        } else if def.emission.emit_light {
                            BlockProperties::emissive(
                                def.emission.light_color[0],
                                def.emission.light_color[1],
                                def.emission.light_color[2],
                                def.emission.light_intensity,
                            )
                        } else {
                            BlockProperties::opaque()
                        };
                        lut.set(id, props);
                    }
                }
                rc.upload_block_lut(queue);
                self.rc_lut_initialized = true;
                debug_log!("GameScreen", "render", "RC block LUT initialized");
            }

            // Dispatch RC pipeline only every RC_THROTTLE frames
            if self.rc_frame_counter % RC_THROTTLE == 0 {
                self.rc_dispatch_call_count = self.rc_dispatch_call_count.wrapping_add(1);
                // Mirror system.rs dispatch_counter logic: full every 4 calls.
                self.last_rc_was_full = self.rc_dispatch_call_count % 4 == 0;

                let sky_brightness = self.day_night.sky_brightness();
                let t_rc = Instant::now();
                rc.dispatch(encoder, queue, sky_brightness);
                self.last_rc_dispatch_time_us = t_rc.elapsed().as_micros();
                self.last_rc_dispatched = true;

                flow_debug_log!("GameScreen", "render",
                    "RC dispatched at frame {} [{:.2}ms, {}]",
                    self.rc_frame_counter,
                    self.last_rc_dispatch_time_us as f64 / 1000.0,
                    if self.last_rc_was_full { "FULL" } else { "partial" });
            } else {
                self.last_rc_dispatched = false;
            }
        }

        // 3D render — pack lighting uniforms and pass to pipeline
        let atlas_view = self.atlas.texture_view().unwrap();
        let normal_atlas_view = self.atlas.normal_texture_view().unwrap();
        let vp = self.camera.view_projection_matrix();

        // let sun_dir = self.day_night.sun_direction();
        // let center = self.camera.position;
        // let dist = 100.0;
        // let up = if sun_dir.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        // let shadow_view = glam::Mat4::look_at_rh(
        //     center + sun_dir * dist,
        //     center,
        //     up,
        // );
        // let ext = 64.0;
        // let shadow_proj = glam::Mat4::orthographic_rh(-ext, ext, -ext, ext, 1.0, 200.0);
        // let shadow_view_proj = shadow_proj * shadow_view;
        let sun_dir = self.day_night.sun_direction();
        let center = self.camera.position;
        let dist = 100.0;
        let up_hint = if sun_dir.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let ext = 64.0_f32;

        // --- Texel snapping: align shadow map texels with voxel grid ---
        // Shadow map is 1024×1024 covering 128×128 world units.
        // Each texel = 0.125 world units.  MUST match shadow_size in
        // Game3DPipeline::new() where the shadow texture is created.
        let shadow_map_size = 1024.0_f32;
        let texel_size = 2.0 * ext / shadow_map_size; // 0.125

        // Light-space basis vectors (matching look_at_rh internals).
        // look_at_rh computes: fwd = normalize(target-eye), right = cross(fwd, up), up' = cross(right, fwd).
        let light_fwd = -sun_dir;
        let light_right = light_fwd.cross(up_hint).normalize();
        let light_up = light_right.cross(light_fwd).normalize();

        // Project camera center onto light-space axes.
        let c_right = center.dot(light_right);
        let c_up = center.dot(light_up);
        let c_fwd = center.dot(light_fwd);

        // Snap perpendicular axes to texel grid.
        // The +0.5 offset ensures integer world coordinates map to texel
        // CENTERS (not boundaries), so block edges align with texel edges.
        let snapped_right = ((c_right / texel_size).floor() + 0.5) * texel_size;
        let snapped_up = ((c_up / texel_size).floor() + 0.5) * texel_size;

        // Reconstruct snapped center in world space (depth/forward unchanged).
        let snapped_center = light_right * snapped_right
            + light_up * snapped_up
            + light_fwd * c_fwd;

        let shadow_view = glam::Mat4::look_at_rh(
            snapped_center + sun_dir * dist,
            snapped_center,
            light_up,
        );
        let shadow_proj = glam::Mat4::orthographic_rh(-ext, ext, -ext, ext, 1.0, 200.0);
        let shadow_view_proj = shadow_proj * shadow_view;

        flow_debug_log!(
            "GameScreen", "render",
            "[Shadow] center=({:.2},{:.2},{:.2}) snapped=({:.2},{:.2},{:.2}) texel={:.3}",
            center.x, center.y, center.z,
            snapped_center.x, snapped_center.y, snapped_center.z,
            texel_size
        );
        let lighting_data = pack_lighting_uniforms(
            &vp,
            self.camera.position,
            &self.day_night,
            &self.lighting_config,
            &shadow_view_proj,
        );

        // Создаём fallback bind group только если RC система ещё не инициализирована
        // (в норме она инициализируется в этом же кадре выше)
        let fallback_gi_bg: Option<wgpu::BindGroup> = if self.rc_system.is_none() {
            let bgl = pipeline.gi_bgl.as_ref().unwrap();
            let dummy_uniform = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gi_dummy_uniform"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });
            let dummy_storage_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gi_dummy_storage"),
                size: 48,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GI Fallback BG"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: dummy_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dummy_storage_buf.as_entire_binding() },
                ],
            }))
        } else {
            None
        };

        let gi_bg: &wgpu::BindGroup = if let Some(ref rc) = self.rc_system {
            &rc.gi_bg
        } else {
            fallback_gi_bg.as_ref().unwrap()
        };

        let render_us = pipeline.render(
            encoder, view, device, queue, &self.camera, width, height,
            atlas_view, normal_atlas_view, &lighting_data, gi_bg,
            &shadow_view_proj,
        );
        // -- Volumetric lights (halos + god rays) — additive pass after 3D --
        if let (Some(ref mut vol_pass), Some(depth_view)) = (
            self.volumetric_pass.as_mut(),
            pipeline.depth_view(),
        ) {
            let vp        = self.camera.view_projection_matrix();
            let cam_right = self.camera.right();
            let cam_fwd   = self.camera.forward();
            let cam_up    = cam_right.cross(cam_fwd).normalize();

            vol_pass.update(
                device, queue,
                &self.volumetric_lights,
                &vp,
                cam_right,
                cam_up,
                self.elapsed_s,
            );
            vol_pass.render(encoder, view, depth_view);
        }
        // Block outline
        {
            let t = Instant::now();
            pipeline.render_outline(
                encoder, view, device, queue, &self.camera, width, height,
                self.target_block.map(|r| r.block_pos),
            );
            self.last_outline_us = t.elapsed().as_micros();
        }

        self.last_frame_total_us = t_frame_start.elapsed().as_micros();

        // Feed profiler
        let vram = pipeline.vram_usage();
        let visible_chunks = pipeline.gpu_chunk_count() as u32
            - pipeline.culled_last_frame;
        let draw_calls         = pipeline.last_draw_calls;
        let shadow_draw_calls  = pipeline.last_shadow_draw_calls;
        let visible_triangles  = pipeline.last_visible_triangles;
        let visible_vram_bytes = pipeline.last_visible_vram_bytes;
        self.profiler.feed(ProfilerFrame {
            fps:                  self.debug.fps,
            frame_total_us:       self.last_frame_total_us,
            gen_time_us:          self.last_gen_time_us,
            gen_count:            self.last_gen_count,
            mesh_time_us:         self.last_mesh_time_us,
            dirty_count:          self.last_dirty_count,
            upload_time_us:       upload_time_us,
            upload_count:         upload_count,
            voxel_upload_time_us: self.last_voxel_upload_time_us,
            voxel_uploaded:       self.last_voxel_uploaded,
            sky_render_us:        self.last_sky_render_us,
            rc_dispatch_time_us:  self.last_rc_dispatch_time_us,
            rc_dispatched:        self.last_rc_dispatched,
            rc_was_full:          self.last_rc_was_full,
            render_3d_us:         render_us,
            outline_us:           self.last_outline_us,
            total_chunks:         self.world_worker.chunk_count(),
            visible_chunks,
            culled_chunks:        pipeline.culled_last_frame,
            lod0_count:           self.last_lod_counts[0],
            lod1_count:           self.last_lod_counts[1],
            lod2_count:           self.last_lod_counts[2],
            worker_pending:       self.last_worker_pending,
            upload_pending:       self.upload_worker.pending_count(),
            gpu_queue_depth:      self.upload_queue.len(),
            vram_bytes:              vram,
            draw_calls,
            shadow_draw_calls,
            visible_triangles,
            visible_vram_bytes,
            upload_bytes:            self.last_upload_bytes,
            upload_throughput_mb_s:  self.last_upload_throughput,
            cam_pos:              [
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z,
            ],
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

        // =================== Time speed HUD (top-right) ====================
        {
            let speed_label = self.day_night.time_speed.label();
            let time = self.day_night.time_of_day();
            let hours = (time * 24.0) as u32;
            let minutes = ((time * 24.0 - hours as f32) * 60.0) as u32;
            let time_text = format!(
                "{:02}:{:02}  Speed: {}  (T/Shift+T/Ctrl+T)",
                hours, minutes, speed_label
            );
            painter.text(
                Pos2::new(sw - 10.0, 8.0),
                Align2::RIGHT_TOP,
                &time_text,
                FontId::proportional(14.0),
                Color32::from_rgba_unmultiplied(255, 255, 200, 180),
            );
        }

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
                                // -- Time controls --
                                KeyCode::KeyT => {
                                    let shift = self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftLeft))
                                             || self.keys.contains(&PhysicalKey::Code(KeyCode::ShiftRight));
                                    let ctrl  = self.keys.contains(&PhysicalKey::Code(KeyCode::ControlLeft))
                                             || self.keys.contains(&PhysicalKey::Code(KeyCode::ControlRight));
                                    if shift {
                                        self.day_night.time_speed = self.day_night.time_speed.faster();
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    } else if ctrl {
                                        self.day_night.time_speed = self.day_night.time_speed.slower();
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    } else {
                                        // Toggle pause
                                        use crate::core::lighting::TimeSpeed;
                                        if self.day_night.time_speed == TimeSpeed::Paused {
                                            self.day_night.time_speed = TimeSpeed::Normal;
                                        } else {
                                            self.day_night.time_speed = TimeSpeed::Paused;
                                        }
                                        debug_log!("GameScreen", "on_event",
                                            "Time speed -> {}", self.day_night.time_speed.label());
                                    }
                                }
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
        let sky = self.day_night.sky_color();
        wgpu::Color { r: sky[0], g: sky[1], b: sky[2], a: 1.0 }
    }
}
