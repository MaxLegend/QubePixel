// =============================================================================
// QubePixel — DebugOverlay  (minimal HUD — left side)
// Shows: FPS, frame time, CPU estimate, GPU load, VRAM, GPU name,
//        terrain info, look direction, and hovered block info.
// F3 to toggle.
// =============================================================================

use egui::{Color32, FontId, Pos2, Align2, Rect};

use crate::core::world_gen::pipeline::BiomePipeline;
use crate::core::world_gen::nature::NatureLayer;

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

    // RC / probe grid debug
    pub rc_camera_pos:      [f32; 3],
    pub rc_c0_origin:       [f32; 3],
    pub rc_c0_snapped:      [f32; 3],
    pub rc_c0_frac_offset:  [f32; 3],
    pub rc_c0_spacing:      f32,

    // Look direction
    pub yaw_deg:  f32,
    pub pitch_deg: f32,
    pub look_dir: [f32; 3],

    // Terrain info (sampled from pipeline each frame).
    pipeline: BiomePipeline,
    terrain_biome_name: String,
    terrain_climate: String,
    terrain_macro: String,
    terrain_micro: String,
    terrain_height: i32,
    terrain_continentalness: f64,
    terrain_temperature: f64,
    terrain_humidity: f64,
    terrain_is_ocean: bool,
    terrain_nature_biome: String,
    terrain_is_river: bool,

    // Hovered block info
    pub hovered_block_id: u8,
    pub hovered_block_key: String,
    pub hovered_placement_mode: String,
    pub hovered_model_type: String,
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

            pipeline: BiomePipeline::new(12345),
            terrain_biome_name: String::new(),
            terrain_climate: String::new(),
            terrain_macro: String::new(),
            terrain_micro: String::new(),
            terrain_height: 0,
            terrain_continentalness: 0.0,
            terrain_temperature: 0.0,
            terrain_humidity: 0.0,
            terrain_is_ocean: false,
            terrain_nature_biome: String::new(),
            terrain_is_river: false,

            rc_camera_pos:     [0.0; 3],
            rc_c0_origin:      [0.0; 3],
            rc_c0_snapped:     [0.0; 3],
            rc_c0_frac_offset: [0.0; 3],
            rc_c0_spacing:     1.0,

            yaw_deg:  0.0,
            pitch_deg: 0.0,
            look_dir: [0.0, 0.0, -1.0],

            hovered_block_id: 0,
            hovered_block_key: String::new(),
            hovered_placement_mode: String::new(),
            hovered_model_type: String::new(),
        }
    }

    /// Call every frame.
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

    /// Update camera yaw, pitch and forward direction.
    pub fn update_look(&mut self, yaw: f32, pitch: f32, dir: [f32; 3]) {
        self.yaw_deg   = yaw.to_degrees();
        self.pitch_deg = pitch.to_degrees();
        self.look_dir  = dir;
    }

    /// Update hovered block info. Pass id=0 to clear.
    pub fn update_hovered_block(
        &mut self,
        id: u8,
        key: &str,
        placement: &str,
        model_type: &str,
    ) {
        self.hovered_block_id        = id;
        self.hovered_block_key       = key.to_string();
        self.hovered_placement_mode  = placement.to_string();
        self.hovered_model_type      = model_type.to_string();
    }

    /// Sample terrain info at the player's world position.
    pub fn update_terrain(&mut self, wx: f32, _wy: f32, wz: f32) {
        if !self.visible { return; }

        let ix = wx.floor() as i32;
        let iz = wz.floor() as i32;

        let geo = self.pipeline.geography.sample(ix, iz);
        let climate = self.pipeline.climate.sample(ix, iz);
        let geology = self.pipeline.geology.sample(ix, iz);
        let biome_id = self.pipeline.determine_biome(&geo, &climate, &geology);

        self.terrain_biome_name = self.pipeline.biome_dict
            .get(biome_id)
            .map(|b| b.name.to_string())
            .unwrap_or_else(|| format!("#{}", biome_id));

        self.terrain_climate = climate.climate_type.name().to_string();
        self.terrain_macro = geology.macro_region.name().to_string();
        self.terrain_micro = format!("{:?}", geology.micro_region);
        self.terrain_height = geo.surface_height;
        self.terrain_continentalness = geo.continentalness;
        self.terrain_temperature = climate.temperature;
        self.terrain_humidity = climate.humidity;
        self.terrain_is_ocean = geo.is_ocean;
        self.terrain_is_river = geo.is_river;

        let nature = NatureLayer::assign_nature_biome(climate.climate_type, &geo, climate.humidity);
        self.terrain_nature_biome = format!("{:?}", nature);
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

        let header_color = Color32::from_rgb(100, 200, 255);
        let info_color   = Color32::from_rgb(220, 220, 220);
        let dim_color    = Color32::from_rgba_unmultiplied(255, 255, 255, 153);

        let look_x = self.look_dir[0];
        let look_y = self.look_dir[1];
        let look_z = self.look_dir[2];

        // Cardinal facing label from yaw
        let yaw_norm = self.yaw_deg.rem_euclid(360.0);
        let facing = match yaw_norm as u32 {
            315..=360 | 0..=44  => "N",
            45..=134            => "E",
            135..=224           => "S",
            _                   => "W",
        };

        let mut owned: Vec<(String, Color32)> = vec![
            ("=== DEBUG ===".to_owned(), header_color),
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
                dim_color),
            // -- Look direction --
            ("--- LOOK ---".to_owned(), header_color),
            (format!("Yaw: {:.1}°  Pitch: {:.1}°  ({})",
                self.yaw_deg, self.pitch_deg, facing),
                Color32::from_rgb(255, 220, 120)),
            (format!("Dir: ({:.3}, {:.3}, {:.3})",
                look_x, look_y, look_z),
                Color32::from_rgb(200, 200, 255)),
            // -- Terrain info --
            ("--- TERRAIN ---".to_owned(), header_color),
            (format!("Biome: {}", self.terrain_biome_name),
                Color32::from_rgb(120, 255, 120)),
            (format!("Nature: {}{}",
                self.terrain_nature_biome,
                if self.terrain_is_river { "  [RIVER]" } else { "" }),
                Color32::from_rgb(160, 220, 120)),
            (format!("Climate: {} (T:{:.2} H:{:.2})",
                self.terrain_climate, self.terrain_temperature, self.terrain_humidity),
                Color32::from_rgb(255, 200, 100)),
            (format!("Geology: {} / {}",
                self.terrain_macro, self.terrain_micro),
                Color32::from_rgb(200, 180, 255)),
            (format!("Height: {}  Cont: {:.2}{}",
                self.terrain_height,
                self.terrain_continentalness,
                if self.terrain_is_ocean { "  [OCEAN]" } else { "" }),
                info_color),
        ];

        // -- Hovered block info --
        if self.hovered_block_id != 0 {
            owned.push(("--- BLOCK ---".to_owned(), header_color));
            owned.push((
                format!("ID: {}  Key: {}", self.hovered_block_id, self.hovered_block_key),
                Color32::from_rgb(255, 160, 80),
            ));
            owned.push((
                format!("Placement: {}", self.hovered_placement_mode),
                Color32::from_rgb(180, 220, 255),
            ));
            owned.push((
                format!("Model: {}", self.hovered_model_type),
                Color32::from_rgb(220, 180, 255),
            ));
        }

        let bg_w = 320.0_f32;
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
