// =============================================================================
// QubePixel — DebugOverlay (FPS, VRAM, GPU load — rendered as 2D text)
// =============================================================================

use crate::debug_log;
use egui::{Color32, FontId, Pos2, Align2, Stroke, Rect};
use crate::screens::gpu_info;

/// Rolling-window FPS tracker and debug HUD renderer.
///
/// Call `update(dt, vram_used)` every frame, then `draw(...)` inside the
/// screen's render pass (after 3D, alongside the rest of the 2D UI).
pub struct DebugOverlay {
    /// Rolling history of frame deltas (seconds).
    frame_times: Vec<f64>,
    max_samples: usize,
    /// 1 / average_dt  (smoothed FPS).
    fps: f64,
    /// Last raw frame time in milliseconds.
    frame_time_ms: f64,
    /// GPU load estimate: frame_time / frame_budget * 100 (%).
    gpu_load: f64,
    /// Target frame budget in ms (e.g. 16.67 ms for 60 FPS).
    frame_budget_ms: f64,
    /// Total VRAM currently allocated (bytes) — sum of pipeline + UI + font.
    vram_used_mb: f64,


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
            frame_budget_ms: 1000.0 / 60.0, // 60 FPS target
            vram_used_mb: 0.0,

        }
    }

    /// Call every frame with the delta time and total GPU memory usage.
    pub fn update(&mut self, dt: f64, vram_used_bytes: u64) {
        self.frame_time_ms = dt * 1000.0;

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

        // GPU load estimate (% of frame budget used)
        if self.frame_budget_ms > 0.0 {
            self.gpu_load = (self.frame_time_ms / self.frame_budget_ms * 100.0).min(100.0);
        }

        // VRAM
        self.vram_used_mb = vram_used_bytes as f64 / (1024.0 * 1024.0);
    }



    /// Draws the debug HUD using egui Painter. Call from `build_ui()`.
    pub fn draw_egui(&self, painter: &egui::Painter) {
        let padding = 10.0_f32;
        let line_h  = 25.0_f32;
        let font_id = FontId::proportional(15.0);
        let x0 = padding;
        let y0 = padding;

        // Собираем строки — храним String, не &str
        let lines: Vec<(String, Color32)> = vec![
            (
                format!("FPS: {:.1} ({:.2} ms)", self.fps, self.frame_time_ms),
                Color32::YELLOW,
            ),
            (
                format!("GPU Load: {:.1}%", self.gpu_load),
                if self.gpu_load < 60.0 {
                    Color32::GREEN
                } else if self.gpu_load < 85.0 {
                    Color32::YELLOW
                } else {
                    Color32::RED
                },
            ),
            (
                format!("VRAM: {:.2} MB", self.vram_used_mb),
                Color32::from_rgb(0, 230, 230),
            ),
            (
                format!("GPU: {}", crate::screens::gpu_info::get()),
                Color32::from_rgba_unmultiplied(255, 255, 255, 153),
            ),
        ];

        // Фон (полупрозрачный тёмный)
        let bg_height = lines.len() as f32 * line_h + padding;
        let bg_width  = 280.0;
        painter.rect_filled(
            Rect::from_min_size(
                Pos2::new(x0 - 6.0, y0 - 6.0),
                egui::Vec2::new(bg_width, bg_height),
            ),
            4.0,
            Color32::from_rgba_unmultiplied(0, 0, 0, 140),
        );

        // Отрисовка строк
        let mut y = y0;
        for (text, color) in &lines {
            painter.text(
                Pos2::new(x0, y),
                Align2::LEFT_TOP,
                text.as_str(),
                font_id.clone(),
                *color,
            );
            y += line_h;
        }
    }
}
