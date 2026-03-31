// =============================================================================
// QubePixel — DebugOverlay (FPS, VRAM, GPU load — rendered as 2D text)
// =============================================================================

use crate::debug_log;
use crate::screens::bitmap_font::BitmapFont;
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
    /// Whether the overlay was initialised (first render call).
    ready: bool,
    /// Background rectangle dimensions (updated each frame).
    _bg_width: f32,
    _bg_height: f32,
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
            ready: false,
            _bg_width: 0.0,
            _bg_height: 0.0,
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

    /// Draws the debug HUD in the top-left corner of the screen.
    ///
    /// Must be called **after** the 3D pass so it renders on top.
    pub fn draw(
        &self,
        font: &BitmapFont,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) {
        let scale = 1.5_f32;
        let padding = 10.0_f32;
        let line_h = BitmapFont::char_height() * scale + 4.0;
        let x0 = padding;
        let mut y = padding;

        // -- FPS (yellow) --------------------------------------------------------
        let fps_text = format!("FPS: {:.1} ({:.2} ms)", self.fps, self.frame_time_ms);
        font.draw_text(
            encoder, view, device, queue,
            &fps_text, x0, y,
            [1.0, 1.0, 0.0, 1.0],
            scale, width, height,
        );
        y += line_h;

        // -- GPU load (green < 60%, yellow < 85%, red >= 85%) --------------------
        let (r, g, b) = if self.gpu_load < 60.0 {
            (0.0, 1.0, 0.0)
        } else if self.gpu_load < 85.0 {
            (1.0, 1.0, 0.0)
        } else {
            (1.0, 0.0, 0.0)
        };
        let load_text = format!("GPU Load: {:.1}%", self.gpu_load);
        font.draw_text(
            encoder, view, device, queue,
            &load_text, x0, y,
            [r, g, b, 1.0],
            scale, width, height,
        );
        y += line_h;

        // -- VRAM (cyan) ---------------------------------------------------------
        let vram_text = format!("VRAM: {:.2} MB", self.vram_used_mb);
        font.draw_text(
            encoder, view, device, queue,
            &vram_text, x0, y,
            [0.0, 0.9, 0.9, 1.0],
            scale, width, height,
        );
        y += line_h;

        // -- GPU name (white, dimmed) --------------------------------------------
        let gpu_text = format!("GPU: {}", gpu_info::get());
        font.draw_text(
            encoder, view, device, queue,
            &gpu_text, x0, y,
            [1.0, 1.0, 1.0, 0.6],
            scale, width, height,
        );
    }
}
