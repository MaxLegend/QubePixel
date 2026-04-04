// =============================================================================
// QubePixel — ProfilerOverlay  (detailed pipeline breakdown — right side)
// Shows every rendering stage with CPU timing and world stats.
// F4 to toggle.
// =============================================================================

use egui::{Color32, FontId, Pos2, Align2, Rect};

// ---------------------------------------------------------------------------
// ProfilerFrame — one snapshot per frame, fed from GameScreen::render
// ---------------------------------------------------------------------------
pub struct ProfilerFrame {
    // --- Frame summary -------------------------------------------------------
    pub fps:            f64,
    pub frame_total_us: u128,   // render() wall time (µs)

    // --- Pipeline stages (CPU encoding / staging time) ----------------------
    /// Worker: terrain generation (noise + block placement)
    pub gen_time_us:    u128,
    pub gen_count:      usize,
    /// Worker: mesh building (greedy quads → vertices/indices)
    pub mesh_time_us:   u128,
    pub dirty_count:    usize,
    /// Render: chunk VB/IB GPU upload (mapped_at_creation memcpy)
    pub upload_time_us: u128,
    pub upload_count:   usize,
    /// Render: voxel texture write_texture (128³×2B = 4MB) to RC system
    pub voxel_upload_time_us: u128,
    /// True only in the frame where an actual upload happened (throttled).
    pub voxel_uploaded: bool,
    /// Render: sky billboard pass (sun/moon quads)
    pub sky_render_us:  u128,
    /// Render: RC compute dispatch encoding (ray march + merge commands)
    pub rc_dispatch_time_us: u128,
    pub rc_dispatched:  bool,
    pub rc_was_full:    bool,
    /// Render: main 3D terrain pass (chunk draw calls + GI fragment shader)
    pub render_3d_us:   u128,
    /// Render: block selection outline pass
    pub outline_us:     u128,

    // --- World / GPU stats --------------------------------------------------
    pub total_chunks:   usize,
    pub visible_chunks: u32,
    pub culled_chunks:  u32,
    pub lod0_count:     usize,
    pub lod1_count:     usize,
    pub lod2_count:     usize,
    pub worker_pending: usize,
    pub upload_pending: usize,
    pub gpu_queue_depth: usize,
    pub vram_bytes:     u64,

    // --- Chunk render stats -------------------------------------------------
    /// Number of draw_indexed calls in the main 3D pass this frame.
    pub draw_calls:             u32,
    /// Number of draw_indexed calls in the shadow pass (all chunks).
    pub shadow_draw_calls:      u32,
    /// Total triangles drawn in the main 3D pass.
    pub visible_triangles:      u32,
    /// VRAM bytes of visible chunk meshes (subset of vram_bytes).
    pub visible_vram_bytes:     u64,
    /// Bytes transferred to GPU for new chunk VB/IB this frame.
    pub upload_bytes:           u64,
    /// Estimated CPU-side staging throughput in MB/s.
    pub upload_throughput_mb_s: f64,

    // --- Camera -------------------------------------------------------------
    pub cam_pos: [f32; 3],
}

// ---------------------------------------------------------------------------
pub struct ProfilerOverlay {
    pub visible: bool,
    last_frame:  Option<ProfilerFrame>,
}

impl ProfilerOverlay {
    pub fn new() -> Self {
        Self { visible: false, last_frame: None }
    }

    pub fn feed(&mut self, frame: ProfilerFrame) {
        self.last_frame = Some(frame);
    }

    pub fn draw_egui(&self, painter: &egui::Painter, sw: f32) {
        if !self.visible { return; }
        let Some(f) = &self.last_frame else { return };

        let padding = 10.0_f32;
        let line_h  = 18.0_f32;
        let font    = FontId::monospace(12.0);

        // ---- helper closures -----------------------------------------------
        let ms_color = |us: u128, warn: u128, crit: u128| -> Color32 {
            if us >= crit { Color32::RED }
            else if us >= warn { Color32::YELLOW }
            else { Color32::GREEN }
        };

        let ms = |us: u128| us as f64 / 1000.0;

        // ---- build lines ---------------------------------------------------
        let fps_color = if f.fps >= 60.0 { Color32::GREEN }
            else if f.fps >= 30.0 { Color32::YELLOW }
            else { Color32::RED };

        let frame_wall_ms = if f.fps > 0.0 { 1000.0 / f.fps } else { 0.0 };
        let frame_cpu_ms  = ms(f.frame_total_us);
        let gpu_gap_ms    = (frame_wall_ms - frame_cpu_ms).max(0.0);
        let frame_color   = if frame_wall_ms > 33.3 { Color32::RED }
            else if frame_wall_ms > 16.6 { Color32::YELLOW }
            else { Color32::GREEN };

        let rc_label = if !f.rc_dispatched { "skip" }
            else if f.rc_was_full { "FULL C0-C4" }
            else { "C0+C1" };

        let vox_label = if f.voxel_uploaded { " 4MB!" } else { " (skip)" };

        // Per-chunk averages
        let avg_tris = if f.draw_calls > 0 { f.visible_triangles / f.draw_calls } else { 0 };
        let avg_vram_kb = if f.draw_calls > 0 {
            f.visible_vram_bytes as f64 / f.draw_calls as f64 / 1024.0
        } else { 0.0 };

        let upload_kb = f.upload_bytes as f64 / 1024.0;

        let mut lines: Vec<(String, Color32)> = vec![
            // Header
            ("=== PROFILER ===".to_owned(),
                Color32::from_rgb(255, 180, 60)),

            // Frame summary
            (format!("FPS:  {:.1}  wall={:.2}ms  CPU={:.2}ms",
                f.fps, frame_wall_ms, frame_cpu_ms),
                fps_color),
            (format!("GPU gap: {:.2} ms  (GPU execution time)",
                gpu_gap_ms),
                if gpu_gap_ms > 16.6 { Color32::RED }
                else if gpu_gap_ms > 8.0 { Color32::YELLOW }
                else { Color32::from_rgb(180, 255, 180) }),
            (format!("Frame total CPU: {:.2} ms",
                frame_cpu_ms),
                frame_color),

            // Separator
            ("--- CPU encode pipeline ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            // Stage 1-8
            (format!("1. Gen:       {:6.2} ms  ({} chunks)",
                ms(f.gen_time_us), f.gen_count),
                ms_color(f.gen_time_us, 5_000, 15_000)),
            (format!("2. Mesh:      {:6.2} ms  ({} dirty)",
                ms(f.mesh_time_us), f.dirty_count),
                ms_color(f.mesh_time_us, 5_000, 15_000)),
            (format!("3. Ch.Upload: {:6.2} ms  {} bufs  {:.0}KB",
                ms(f.upload_time_us), f.upload_count, upload_kb),
                ms_color(f.upload_time_us, 2_000, 5_000)),
            (format!("4. Vox.Upload:{:6.2} ms{}",
                ms(f.voxel_upload_time_us), vox_label),
                ms_color(f.voxel_upload_time_us, 2_000, 4_000)),
            (format!("5. Sky:       {:6.2} ms",
                ms(f.sky_render_us)),
                ms_color(f.sky_render_us, 1_000, 3_000)),
            (format!("6. RC GI:     {:6.2} ms  ({})",
                ms(f.rc_dispatch_time_us), rc_label),
                ms_color(f.rc_dispatch_time_us, 500, 2_000)),
            (format!("7. 3D Render: {:6.2} ms",
                ms(f.render_3d_us)),
                ms_color(f.render_3d_us, 2_000, 5_000)),
            (format!("8. Outline:   {:6.2} ms",
                ms(f.outline_us)),
                ms_color(f.outline_us, 500, 2_000)),

            // Separator
            ("--- chunk draw ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Draws: {}  shadow: {}  tris: {}",
                f.draw_calls, f.shadow_draw_calls, f.visible_triangles),
                Color32::from_rgb(180, 220, 255)),
            (format!("Avg/chunk: {} tris  {:.1} KB VRAM",
                avg_tris, avg_vram_kb),
                Color32::from_rgb(160, 200, 255)),
            (format!("Staging: {:.0} KB/frame  {:.0} MB/s",
                upload_kb, f.upload_throughput_mb_s),
                ms_color(f.upload_bytes as u128 * 1000, 512_000, 2_048_000)),
            (format!("Vis.VRAM: {:.2} MB / {:.2} MB total",
                f.visible_vram_bytes as f64 / (1024.0*1024.0),
                f.vram_bytes as f64 / (1024.0*1024.0)),
                Color32::from_rgb(0, 230, 230)),

            // Separator
            ("--- world ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Chunks: {}  vis={}  cull={}",
                f.total_chunks, f.visible_chunks, f.culled_chunks),
                Color32::from_rgb(180, 220, 255)),
            (format!("LOD: 0={}  1={}  2={}",
                f.lod0_count, f.lod1_count, f.lod2_count),
                Color32::from_rgb(200, 180, 255)),
            (format!("Worker pending: {}   upload Q: {}",
                f.worker_pending, f.gpu_queue_depth),
                if f.gpu_queue_depth > 64 { Color32::YELLOW }
                else { Color32::from_rgb(180, 180, 180) }),

            // Separator
            ("--- camera ---".to_owned(),
                Color32::from_rgb(80, 80, 80)),

            (format!("Pos: ({:.1}, {:.1}, {:.1})",
                f.cam_pos[0], f.cam_pos[1], f.cam_pos[2]),
                Color32::from_rgb(220, 220, 180)),
        ];

        // Annotate the heaviest CPU stage (indices 5..12 = stages 1-8)
        let stages = [
            (f.gen_time_us,          5usize),
            (f.mesh_time_us,         6),
            (f.upload_time_us,       7),
            (f.voxel_upload_time_us, 8),
            (f.sky_render_us,        9),
            (f.rc_dispatch_time_us,  10),
            (f.render_3d_us,         11),
            (f.outline_us,           12),
        ];
        if let Some(&(worst_us, worst_idx)) = stages.iter().max_by_key(|&&(us, _)| us) {
            if worst_us > 500 {
                let entry = &mut lines[worst_idx];
                entry.0 = format!("{} ◄", entry.0);
                entry.1 = Color32::from_rgb(255, 120, 120);
            }
        }

        // ---- layout --------------------------------------------------------
        let bg_w   = 310.0_f32;
        let bg_h   = lines.len() as f32 * line_h + padding;
        let x_text = sw - padding - bg_w + 8.0;
        let y0     = padding;

        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(x_text - 8.0, y0 - 6.0),
                egui::Vec2::new(bg_w, bg_h),
            ),
            4.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 150),
        );

        let mut y = y0;
        for (text, color) in &lines {
            painter.text(
                Pos2::new(x_text, y),
                Align2::LEFT_TOP,
                text.as_str(),
                font.clone(),
                *color,
            );
            y += line_h;
        }
    }
}
