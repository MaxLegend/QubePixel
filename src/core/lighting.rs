// =============================================================================
// QubePixel — Lighting  (Day/Night cycle, lighting config, uniform data)
// =============================================================================

use glam::Vec3;
use serde::Deserialize;
use crate::debug_log;

// ---------------------------------------------------------------------------
// LightingConfig — loaded from assets/lighting_config.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DayNightConfig {
    pub cycle_duration_seconds: f32,
    pub start_time: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SunConfig {
    pub color_noon:    [f32; 3],
    pub color_sunrise: [f32; 3],
    pub max_intensity: f32,
    pub sprite_texture: String,
    pub sprite_scale: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MoonConfig {
    pub color:         [f32; 3],
    pub max_intensity: f32,
    pub sprite_texture: String,
    pub sprite_scale: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AmbientConfig {
    pub day_color:   [f32; 3],
    pub night_color: [f32; 3],
    pub min_level:   f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShadowConfig {
    pub enabled:  bool,
    pub cascades: u32,
    pub map_size: u32,
    pub bias:     f32,
    pub distance: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SsaoConfig {
    pub enabled:     bool,
    pub kernel_size: u32,
    pub radius:      f32,
    pub bias:        f32,
    pub blur_passes: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightingConfig {
    pub day_night: DayNightConfig,
    pub sun:       SunConfig,
    pub moon:      MoonConfig,
    pub ambient:   AmbientConfig,
    pub shadows:   ShadowConfig,
    pub ssao:      SsaoConfig,
}

impl LightingConfig {
    pub fn load() -> Self {
        let path = "assets/lighting_config.json";
        match std::fs::read_to_string(path) {
            Ok(s) => {
                match serde_json::from_str::<Self>(&s) {
                    Ok(cfg) => {
                        debug_log!("LightingConfig", "load", "Loaded from {}", path);
                        cfg
                    }
                    Err(e) => {
                        debug_log!("LightingConfig", "load",
                            "Failed to parse {}: {}, using defaults", path, e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                debug_log!("LightingConfig", "load",
                    "File {} not found, using defaults", path);
                Self::default()
            }
        }
    }
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            day_night: DayNightConfig {
                cycle_duration_seconds: 600.0,
                start_time: 0.35,
            },
            sun: SunConfig {
                color_noon:    [1.0, 0.98, 0.92],
                color_sunrise: [1.0, 0.6, 0.3],
                max_intensity: 1.0,
                sprite_texture: "sun".into(),
                sprite_scale: 4.0,
            },
            moon: MoonConfig {
                color:         [0.7, 0.75, 0.9],
                max_intensity: 0.15,
                sprite_texture: "moon".into(),
                sprite_scale: 3.0,
            },
            ambient: AmbientConfig {
                day_color:   [0.3, 0.35, 0.45],
                night_color: [0.05, 0.05, 0.12],
                min_level:   0.08,
            },
            shadows: ShadowConfig {
                enabled:  true,
                cascades: 8,
                map_size: 2048,
                bias:     0.0,
                distance: 64.0,
            },
            ssao: SsaoConfig {
                enabled:     true,
                kernel_size: 32,
                radius:      0.5,
                bias:        0.025,
                blur_passes: 1,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// DayNightCycle — tracks time of day, computes sun/moon positions & colours
// ---------------------------------------------------------------------------

/// Time speed multiplier options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeSpeed {
    Paused,     // 0×
    Slow,       // 0.25×
    Normal,     // 1×
    Fast,       // 4×
    VeryFast,   // 16×
}

impl TimeSpeed {
    pub fn multiplier(self) -> f32 {
        match self {
            Self::Paused   => 0.0,
            Self::Slow     => 0.25,
            Self::Normal   => 1.0,
            Self::Fast     => 4.0,
            Self::VeryFast => 16.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Paused   => "Paused",
            Self::Slow     => "0.25x",
            Self::Normal   => "1x",
            Self::Fast     => "4x",
            Self::VeryFast => "16x",
        }
    }

    /// Cycle to slower speed
    pub fn slower(self) -> Self {
        match self {
            Self::VeryFast => Self::Fast,
            Self::Fast     => Self::Normal,
            Self::Normal   => Self::Slow,
            Self::Slow     => Self::Paused,
            Self::Paused   => Self::Paused,
        }
    }

    /// Cycle to faster speed
    pub fn faster(self) -> Self {
        match self {
            Self::Paused   => Self::Slow,
            Self::Slow     => Self::Normal,
            Self::Normal   => Self::Fast,
            Self::Fast     => Self::VeryFast,
            Self::VeryFast => Self::VeryFast,
        }
    }
}

pub struct DayNightCycle {
    /// Normalised time of day: 0.0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
    time_of_day: f32,
    /// Duration of one full day cycle in seconds
    cycle_duration: f32,
    /// Current time speed
    pub time_speed: TimeSpeed,

    /// Cached config values
    sun_color_noon:    Vec3,
    sun_color_sunrise: Vec3,
    sun_max_intensity: f32,
    moon_color:        Vec3,
    moon_max_intensity: f32,
    ambient_day:       Vec3,
    ambient_night:     Vec3,
    ambient_min:       f32,
}

impl DayNightCycle {
    pub fn new(config: &LightingConfig) -> Self {
        debug_log!("DayNightCycle", "new",
            "Initialising — start_time={}, cycle={}s",
            config.day_night.start_time, config.day_night.cycle_duration_seconds);

        Self {
            time_of_day:   config.day_night.start_time,
            cycle_duration: config.day_night.cycle_duration_seconds,
            time_speed:     TimeSpeed::Normal,

            sun_color_noon:    Vec3::from(config.sun.color_noon),
            sun_color_sunrise: Vec3::from(config.sun.color_sunrise),
            sun_max_intensity: config.sun.max_intensity,
            moon_color:        Vec3::from(config.moon.color),
            moon_max_intensity: config.moon.max_intensity,
            ambient_day:       Vec3::from(config.ambient.day_color),
            ambient_night:     Vec3::from(config.ambient.night_color),
            ambient_min:       config.ambient.min_level,
        }
    }

    /// Advance time by dt seconds (real time).
    pub fn update(&mut self, dt: f64) {
        let advance = (dt as f32 / self.cycle_duration) * self.time_speed.multiplier();
        self.time_of_day = (self.time_of_day + advance) % 1.0;
    }

    pub fn time_of_day(&self) -> f32 { self.time_of_day }

    // -----------------------------------------------------------------------
    // Sun
    // -----------------------------------------------------------------------

    /// Sun direction in world space.
    /// Sun rises at time 0.25 (east), peaks at 0.5 (overhead), sets at 0.75 (west).
    /// Below horizon outside 0.25..0.75.
    pub fn sun_direction(&self) -> Vec3 {
        let t = self.time_of_day;
        // Map [0.25, 0.75] → [0, π]
        let angle = (t - 0.25) * 2.0 * std::f32::consts::PI;
        // Sun arc: rises from +X, goes up, sets at -X
        // Y = sin(angle) → up at noon, Z-tilt for slight depth
        Vec3::new(
            angle.cos(),
            angle.sin().max(0.0),
            0.3 * angle.sin(),
        ).normalize()
    }

    /// Sun intensity: 0 at night, peaks at noon (time 0.5).
    pub fn sun_intensity(&self) -> f32 {
        let t = self.time_of_day;
        if t < 0.20 || t > 0.80 {
            return 0.0; // Fully below horizon
        }
        // Smooth fade near horizon
        let horizon_factor = if t < 0.30 {
            (t - 0.20) / 0.10 // fade in at dawn
        } else if t > 0.70 {
            (0.80 - t) / 0.10 // fade out at dusk
        } else {
            1.0
        };
        // Elevation factor: sin-based for peak at noon
        let elevation = ((t - 0.25) * std::f32::consts::PI * 2.0).sin().max(0.0);
        self.sun_max_intensity * elevation * horizon_factor
    }

    /// Sun colour: warm near horizon, neutral at noon.
    pub fn sun_color(&self) -> Vec3 {
        let t = self.time_of_day;
        // Interpolation: sunrise/sunset → warm tones, noon → cool/neutral
        let noon_factor = if t > 0.25 && t < 0.75 {
            let mid_dist = (t - 0.5).abs() * 4.0; // 0 at noon, 1 at horizon
            1.0 - mid_dist.min(1.0)
        } else {
            0.0
        };
        self.sun_color_sunrise.lerp(self.sun_color_noon, noon_factor)
    }

    // -----------------------------------------------------------------------
    // Moon
    // -----------------------------------------------------------------------

    /// Moon direction — opposite phase from sun.
    /// Moon rises at time 0.75 (sunset), peaks at 0.0 (midnight), sets at 0.25 (dawn).
    pub fn moon_direction(&self) -> Vec3 {
        let t = (self.time_of_day + 0.5) % 1.0;
        let angle = (t - 0.25) * 2.0 * std::f32::consts::PI;
        Vec3::new(
            angle.cos(),
            angle.sin().max(0.0),
            -0.2 * angle.sin(),
        ).normalize()
    }

    /// Moon intensity: 0 during day, peaks at midnight.
    pub fn moon_intensity(&self) -> f32 {
        let t = self.time_of_day;
        // Moon visible when sun is down: roughly t < 0.25 or t > 0.75
        let visible = if t < 0.25 {
            1.0 - (t / 0.25) * 0.5  // slight fade toward dawn
        } else if t > 0.75 {
            0.5 + ((t - 0.75) / 0.25) * 0.5 // ramp up after sunset
        } else if t > 0.65 {
            (t - 0.65) / 0.10 * 0.3 // early twilight
        } else {
            0.0
        };
        self.moon_max_intensity * visible
    }

    /// Moon colour (constant cool blue-white).
    pub fn moon_color(&self) -> Vec3 {
        self.moon_color
    }

    // -----------------------------------------------------------------------
    // Sky & Ambient
    // -----------------------------------------------------------------------

    /// Dynamic sky colour for the clear pass.
    pub fn sky_color(&self) -> [f64; 3] {
        let t = self.time_of_day;

        // Day sky (blue)
        let day_sky   = Vec3::new(0.45, 0.65, 0.95);
        // Sunrise/sunset (orange-pink)
        let dawn_sky  = Vec3::new(0.85, 0.50, 0.30);
        // Night sky (dark blue)
        let night_sky = Vec3::new(0.02, 0.02, 0.08);

        let sky = if t > 0.30 && t < 0.70 {
            // Daytime
            day_sky
        } else if t >= 0.20 && t <= 0.30 {
            // Dawn transition
            let f = (t - 0.20) / 0.10;
            night_sky.lerp(dawn_sky, f.min(0.5) * 2.0)
                .lerp(day_sky, (f - 0.5).max(0.0) * 2.0)
        } else if t >= 0.70 && t <= 0.80 {
            // Dusk transition
            let f = (t - 0.70) / 0.10;
            day_sky.lerp(dawn_sky, f.min(0.5) * 2.0)
                .lerp(night_sky, (f - 0.5).max(0.0) * 2.0)
        } else {
            // Night
            night_sky
        };

        [sky.x as f64, sky.y as f64, sky.z as f64]
    }

    /// Ambient colour for the shader — blends between day and night ambient.
    pub fn ambient_color(&self) -> Vec3 {
        let sun_i = self.sun_intensity();
        let factor = (sun_i / self.sun_max_intensity).clamp(0.0, 1.0);
        let col = self.ambient_night.lerp(self.ambient_day, factor);
        // Enforce minimum ambient
        Vec3::new(
            col.x.max(self.ambient_min),
            col.y.max(self.ambient_min),
            col.z.max(self.ambient_min),
        )
    }

    // -----------------------------------------------------------------------
    // Position on sky sphere (for billboard placement)
    // -----------------------------------------------------------------------

    /// World-space position of the sun relative to camera (far away).
    pub fn sun_billboard_offset(&self) -> Vec3 {
        self.sun_direction() * 200.0
    }

    /// World-space position of the moon relative to camera (far away).
    pub fn moon_billboard_offset(&self) -> Vec3 {
        self.moon_direction() * 200.0
    }

    // -----------------------------------------------------------------------
    // GI helper
    // -----------------------------------------------------------------------

    /// Overall sky brightness factor (0.0 = full night, 1.0 = noon).
    /// Used by Radiance Cascades to modulate sky emission intensity.
    pub fn sky_brightness(&self) -> f32 {
        self.sun_intensity().clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// LightingUniforms — GPU uniform data written each frame
// ---------------------------------------------------------------------------

/// Size in bytes of the lighting uniform buffer (must match shader struct).
pub const LIGHTING_UNIFORM_SIZE: u64 = 272; // 17 vec4s = 272 bytes

/// Packs all lighting data into an f32 array for uploading to the GPU.
/// Layout (each row = 4 floats = 16 bytes):
///   [0..16]   view_proj       (mat4x4)
///   [16..20]  camera_pos      (vec4)
///   [20..24]  sun_direction   (vec4, w = intensity)
///   [24..28]  sun_color       (vec4)
///   [28..32]  moon_direction  (vec4, w = intensity)
///   [32..36]  moon_color      (vec4)
///   [36..40]  ambient_color   (vec4, w = min_level)
///   [40..44]  shadow_params   (vec4: bias, shadow_enabled, cascades, map_size)
///   [44..48]  ssao_params     (vec4: enabled, radius, bias, kernel_size)
///   [48..52]  time_params     (vec4: time_of_day, 0, 0, 0)
///   [52..68]  shadow_view_proj (mat4x4)
pub fn pack_lighting_uniforms(
    view_proj:   &glam::Mat4,
    camera_pos:  Vec3,
    cycle:       &DayNightCycle,
    config:      &LightingConfig,
    shadow_view_proj: &glam::Mat4,
) -> [f32; 68] {
    let mut data = [0.0f32; 68];

    // view_proj (mat4)
    data[0..16].copy_from_slice(&view_proj.to_cols_array());

    // camera_pos (vec4)
    data[16] = camera_pos.x;
    data[17] = camera_pos.y;
    data[18] = camera_pos.z;
    data[19] = 0.0;

    // sun_direction (vec4, w = intensity)
    let sun_dir = cycle.sun_direction();
    data[20] = sun_dir.x;
    data[21] = sun_dir.y;
    data[22] = sun_dir.z;
    data[23] = cycle.sun_intensity();

    // sun_color (vec4)
    let sun_col = cycle.sun_color();
    data[24] = sun_col.x;
    data[25] = sun_col.y;
    data[26] = sun_col.z;
    data[27] = 1.0;

    // moon_direction (vec4, w = intensity)
    let moon_dir = cycle.moon_direction();
    data[28] = moon_dir.x;
    data[29] = moon_dir.y;
    data[30] = moon_dir.z;
    data[31] = cycle.moon_intensity();

    // moon_color (vec4)
    let moon_col = cycle.moon_color();
    data[32] = moon_col.x;
    data[33] = moon_col.y;
    data[34] = moon_col.z;
    data[35] = 1.0;

    // ambient_color (vec4, w = min_level)
    let amb = cycle.ambient_color();
    data[36] = amb.x;
    data[37] = amb.y;
    data[38] = amb.z;
    data[39] = config.ambient.min_level;

    // shadow_params
    data[40] = config.shadows.bias;
    data[41] = if config.shadows.enabled { 1.0 } else { 0.0 };
    data[42] = config.shadows.cascades as f32;
    data[43] = config.shadows.map_size as f32;

    // ssao_params
    data[44] = if config.ssao.enabled { 1.0 } else { 0.0 };
    data[45] = config.ssao.radius;
    data[46] = config.ssao.bias;
    data[47] = config.ssao.kernel_size as f32;

    // time_params
    data[48] = cycle.time_of_day();
    data[49] = cycle.time_speed.multiplier();
    data[50] = 0.0;
    data[51] = 0.0;

    // shadow_view_proj (mat4x4)
    data[52..68].copy_from_slice(&shadow_view_proj.to_cols_array());

    data
}
