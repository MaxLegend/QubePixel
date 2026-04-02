// =============================================================================
// QubePixel — ProfilerOverlay  (stub — displays per-frame profiling data)
// =============================================================================

use egui::{Color32, FontId, Pos2, Align2};

pub struct ProfilerFrame {
    pub gen_time_us:     u128,
    pub mesh_time_us:    u128,
    pub upload_time_us:  u128,
    pub render_time_us:  u128,
    pub worker_total_us: u128,
    pub gen_count:       usize,
    pub dirty_count:     usize,
    pub upload_count:    usize,
    pub total_chunks:    usize,
    pub visible_chunks:  u32,
    pub culled_chunks:   u32,
    pub lod0_count:      usize,
    pub lod1_count:      usize,
    pub lod2_count:      usize,
    pub worker_pending:  usize,
    pub upload_pending:  usize,
    pub gpu_queue_depth: usize,
    pub vram_bytes:      u64,
}

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

        let lines = [
            format!("Gen:    {:.2}ms  ({})", f.gen_time_us as f64 / 1000.0, f.gen_count),
            format!("Mesh:   {:.2}ms  ({}d)", f.mesh_time_us as f64 / 1000.0, f.dirty_count),
            format!("Upload: {:.2}ms  ({})", f.upload_time_us as f64 / 1000.0, f.upload_count),
            format!("Render: {:.2}ms", f.render_time_us as f64 / 1000.0),
            format!("Worker: {:.2}ms  pending={}", f.worker_total_us as f64 / 1000.0, f.worker_pending),
            format!("Chunks: {} vis={} cull={}", f.total_chunks, f.visible_chunks, f.culled_chunks),
            format!("LOD: 0={} 1={} 2={}", f.lod0_count, f.lod1_count, f.lod2_count),
            format!("GPU Q: {} upload_pend={}", f.gpu_queue_depth, f.upload_pending),
            format!("VRAM: {:.2}MB", f.vram_bytes as f64 / (1024.0 * 1024.0)),
        ];

        let x = sw - 12.0;
        let mut y = 60.0;
        for line in &lines {
            painter.text(
                Pos2::new(x, y),
                Align2::RIGHT_TOP,
                line,
                FontId::monospace(12.0),
                Color32::from_rgba_unmultiplied(200, 200, 200, 200),
            );
            y += 16.0;
        }
    }
}
