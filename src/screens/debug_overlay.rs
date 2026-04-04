// =============================================================================
// QubePixel — DebugOverlay  (minimal HUD — left side)
// Shows: FPS, frame time, CPU estimate, GPU load, VRAM, GPU name.
// F3 to toggle.
// =============================================================================

use egui::{Color32, FontId, Pos2, Align2, Rect};

pub struct DebugOverlay {
    frame_times: Vec<f64>,
    max_samples: usize,

    pub fps:          f64,
    pub frame_time_ms: f64,
    pub gpu_load:     f64,
    pub cpu_work_ms:  f64,
    pub cpu_est_pct:  f64,
    pub vram_used_mb: f64,

    pub visible: bool,
}

impl DebugOverlay {
    pub fn new() -> Self {
        Self {
            frame_times:   Vec::with_capacity(120),
            max_samples:   120,
            fps:           0.0,
            frame_time_ms: 0.0,
            gpu_load:      0.0,
            cpu_work_ms:   0.0,
            cpu_est_pct:   0.0,
            vram_used_mb:  0.0,
            visible:       true,
        }
    }

    /// Call every frame.
    /// `dt`           — elapsed seconds since last frame
    /// `vram_bytes`   — VRAM used by chunk meshes (from pipeline)
    /// `cpu_work_ms`  — CPU work time this frame (update + render encoding)
    pub fn update(&mut self, dt: f64, vram_bytes: u64, cpu_work_ms: f64) {
        self.frame_time_ms = dt * 1000.0;
        self.cpu_work_ms   = cpu_work_ms;

        self.frame_times.push(dt);
        if self.frame_times.len() > self.max_samples {
            self.frame_times.remove(0);
        }

        if !self.frame_times.is_empty() {
            let avg: f64 = self.frame_times.iter().sum::<f64>()
                / self.frame_times.len() as f64;
            self.fps = if avg > 0.0 { 1.0 / avg } else { 0.0 };
        }

        let budget_ms = 1000.0 / 60.0;
        self.gpu_load = (self.frame_time_ms / budget_ms * 100.0).min(999.0);

        if self.frame_time_ms > 0.001 {
            self.cpu_est_pct =
                (self.cpu_work_ms / self.frame_time_ms * 100.0).min(999.0);
        }

        self.vram_used_mb = vram_bytes as f64 / (1024.0 * 1024.0);
    }

    pub fn draw_egui(&self, painter: &egui::Painter) {
        if !self.visible { return; }

        let padding = 10.0_f32;
        let line_h  = 20.0_f32;
        let font    = FontId::monospace(13.0);

        let fps_color = if self.fps >= 60.0 { Color32::GREEN }
            else if self.fps >= 30.0 { Color32::YELLOW }
            else { Color32::RED };

        let gpu_color = if self.gpu_load < 80.0 { Color32::GREEN }
            else if self.gpu_load < 100.0 { Color32::YELLOW }
            else { Color32::RED };

        let cpu_color = if self.cpu_est_pct < 60.0 { Color32::GREEN }
            else if self.cpu_est_pct < 85.0 { Color32::YELLOW }
            else { Color32::RED };

        let lines: &[(&str, Color32)] = &[];
        let owned: Vec<(String, Color32)> = vec![
            ("=== DEBUG ===".to_owned(),
                Color32::from_rgb(100, 200, 255)),
            (format!("FPS: {:.1}  ({:.2} ms)", self.fps, self.frame_time_ms),
                fps_color),
            (format!("CPU: {:.1}%  ({:.2} ms work)",
                self.cpu_est_pct, self.cpu_work_ms),
                cpu_color),
            (format!("GPU Load: {:.1}%", self.gpu_load),
                gpu_color),
            (format!("RAM:  {:.0} MB", crate::screens::gpu_info::get_ram_mb()),
                Color32::from_rgb(180, 230, 180)),
            (format!("VRAM: {:.2} MB", self.vram_used_mb),
                Color32::from_rgb(0, 230, 230)),
            (format!("GPU: {}", crate::screens::gpu_info::get()),
                Color32::from_rgba_unmultiplied(255, 255, 255, 153)),
        ];
        let _ = lines; // suppress warning

        let bg_w = 280.0_f32;
        let bg_h = owned.len() as f32 * line_h + padding;
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(padding - 6.0, padding - 6.0),
                egui::Vec2::new(bg_w, bg_h),
            ),
            4.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 140),
        );

        let mut y = padding;
        for (text, color) in &owned {
            painter.text(
                Pos2::new(padding, y),
                Align2::LEFT_TOP,
                text.as_str(),
                font.clone(),
                *color,
            );
            y += line_h;
        }
    }
}
