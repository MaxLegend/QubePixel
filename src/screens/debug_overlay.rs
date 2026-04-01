// =============================================================================
// QubePixel — DebugOverlay  (expanded HUD — left side of screen)
// =============================================================================
//
// Shows: FPS, CPU estimate, GPU load, VRAM, GPU name,
//        camera position/direction, chunk stats, LOD breakdown.
//
// Call `update()` every frame with all metrics,
// then `draw_egui()` from `build_ui()`.
// =============================================================================

use crate::debug_log;
use egui::{Color32, FontId, Pos2, Align2, Rect};

pub struct DebugOverlay {
    // Rolling FPS
    frame_times: Vec<f64>,
    max_samples: usize,
    fps: f64,
    frame_time_ms: f64,

    // GPU load estimate
    gpu_load: f64,
    frame_budget_ms: f64,

    // CPU estimate
    cpu_work_ms: f64,
    cpu_est_pct: f64,

    // VRAM
    vram_used_mb: f64,

    // Camera
    cam_pos: [f32; 3],
    cam_dir: [f32; 3],

    // Chunk stats
    total_chunks: usize,
    visible_chunks: u32,
    culled_chunks: u32,

    // LOD
    lod_counts: [usize; 3],

    // Visibility
    pub visible: bool,
}

impl DebugOverlay {
    pub fn new() -> Self {
        debug_log!("DebugOverlay", "new", "Creating debug overlay");
        Self {
            frame_times: Vec::with_capacity(120),
            max_samples: 120,
            fps: 0.0,
            frame_time_ms: 0.0,
            gpu_load: 0.0,
            frame_budget_ms: 1000.0 / 60.0,
            cpu_work_ms: 0.0,
            cpu_est_pct: 0.0,
            vram_used_mb: 0.0,
            cam_pos: [0.0; 3],
            cam_dir: [0.0; 3],
            total_chunks: 0,
            visible_chunks: 0,
            culled_chunks: 0,
            lod_counts: [0; 3],
            visible: true,
        }
    }

    /// Call every frame with all metrics.
    pub fn update(
        &mut self,
        dt: f64,
        vram_used_bytes: u64,
        cpu_work_ms: f64,
        cam_pos: [f32; 3],
        cam_dir: [f32; 3],
        total_chunks: usize,
        visible_chunks: u32,
        culled_chunks: u32,
        lod_counts: [usize; 3],
    ) {
        self.frame_time_ms = dt * 1000.0;
        self.cpu_work_ms = cpu_work_ms;

        // Rolling window
        self.frame_times.push(dt);
        if self.frame_times.len() > self.max_samples {
            self.frame_times.remove(0);
        }

        // Average FPS
        if !self.frame_times.is_empty() {
            let avg_dt: f64 =
                self.frame_times.iter().sum::<f64>() / self.frame_times.len() as f64;
            self.fps = if avg_dt > 0.0 { 1.0 / avg_dt } else { 0.0 };
        }

        // GPU load estimate
        if self.frame_budget_ms > 0.0 {
            self.gpu_load = (self.frame_time_ms / self.frame_budget_ms * 100.0).min(999.0);
        }

        // CPU estimate: work_time / frame_time
        if self.frame_time_ms > 0.001 {
            self.cpu_est_pct = (self.cpu_work_ms / self.frame_time_ms * 100.0).min(999.0);
        }

        self.vram_used_mb = vram_used_bytes as f64 / (1024.0 * 1024.0);
        self.cam_pos = cam_pos;
        self.cam_dir = cam_dir;
        self.total_chunks = total_chunks;
        self.visible_chunks = visible_chunks;
        self.culled_chunks = culled_chunks;
        self.lod_counts = lod_counts;
    }

    /// Draws the debug HUD using egui Painter (left side).
    pub fn draw_egui(&self, painter: &egui::Painter) {
        if !self.visible { return; }

        let padding = 10.0_f32;
        let line_h  = 20.0_f32;
        let font    = FontId::monospace(13.0);
        let x0 = padding;
        let y0 = padding;

        let lines: Vec<(String, Color32)> = vec![
            // Header
            ("=== DEBUG ===".to_string(), Color32::from_rgb(100, 200, 255)),

            // Performance
            (
                format!("FPS: {:.1}  ({:.2} ms)", self.fps, self.frame_time_ms),
                Color32::YELLOW,
            ),
            (
                format!("CPU Est: {:.1}%  ({:.2} ms work)",
                    self.cpu_est_pct, self.cpu_work_ms),
                if self.cpu_est_pct < 60.0 { Color32::GREEN }
                else if self.cpu_est_pct < 85.0 { Color32::YELLOW }
                else { Color32::RED },
            ),
            (
                format!("GPU Load: {:.1}%", self.gpu_load),
                if self.gpu_load < 60.0 { Color32::GREEN }
                else if self.gpu_load < 85.0 { Color32::YELLOW }
                else { Color32::RED },
            ),
            (
                format!("VRAM: {:.2} MB", self.vram_used_mb),
                Color32::from_rgb(0, 230, 230),
            ),
            (
                format!("GPU: {}", crate::screens::gpu_info::get()),
                Color32::from_rgba_unmultiplied(255, 255, 255, 153),
            ),

            // Separator
            ("---".to_string(), Color32::from_rgb(80, 80, 80)),

            // Camera
            (
                format!("Pos: ({:.1}, {:.1}, {:.1})",
                    self.cam_pos[0], self.cam_pos[1], self.cam_pos[2]),
                Color32::from_rgb(220, 220, 180),
            ),
            (
                format!("Dir: ({:.2}, {:.2}, {:.2})",
                    self.cam_dir[0], self.cam_dir[1], self.cam_dir[2]),
                Color32::from_rgb(200, 200, 160),
            ),

            // Separator
            ("---".to_string(), Color32::from_rgb(80, 80, 80)),

            // Chunks
            (
                format!("Chunks: {}  vis={} cull={}",
                    self.total_chunks, self.visible_chunks, self.culled_chunks),
                Color32::from_rgb(180, 220, 255),
            ),
            (
                format!("LOD: 0={} 1={} 2={}",
                    self.lod_counts[0], self.lod_counts[1], self.lod_counts[2]),
                Color32::from_rgb(200, 180, 255),
            ),
        ];

        // Background
        let bg_height = lines.len() as f32 * line_h + padding;
        let bg_width  = 300.0;
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(x0 - 6.0, y0 - 6.0),
                egui::Vec2::new(bg_width, bg_height),
            ),
            4.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 140),
        );

        let mut y = y0;
        for (text, color) in &lines {
            painter.text(
                Pos2::new(x0, y),
                Align2::LEFT_TOP,
                text.as_str(),
                font.clone(),
                *color,
            );
            y += line_h;
        }
    }
}
