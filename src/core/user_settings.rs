// =============================================================================
// QubePixel — UserSettings  (persistent user preferences saved to JSON)
// =============================================================================

use serde::{Deserialize, Serialize};
use crate::{debug_log};
use crate::core::config;

const PATH: &str = "saves/user_settings.json";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserSettings {
    pub render_distance:           i32,
    pub lod_multiplier_x100:       u32,
    pub vertical_below:            i32,
    pub vertical_above:            i32,
    pub biome_ambient_tint_enabled: bool,
    pub shadow_sun_enabled:        bool,
    pub shadow_block_enabled:      bool,
    pub gi_enabled:                bool,
    pub shadow_quality:            u32,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            render_distance:           12,
            lod_multiplier_x100:       300,
            vertical_below:            0,
            vertical_above:            0,
            biome_ambient_tint_enabled: true,
            shadow_sun_enabled:        true,
            shadow_block_enabled:      true,
            gi_enabled:                true,
            shadow_quality:            64,
        }
    }
}

/// Read the JSON file (or defaults) and push all values into the global atomics.
pub fn load_and_apply() {
    if let Err(e) = std::fs::create_dir_all("saves") {
        debug_log!("UserSettings", "load_and_apply", "Could not create saves/ dir: {}", e);
    }

    let settings: UserSettings = std::fs::read_to_string(PATH)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    debug_log!(
        "UserSettings", "load_and_apply",
        "Loaded: rd={} lod={} vb={} va={} tint={} sun={} block={} gi={} sq={}",
        settings.render_distance,
        settings.lod_multiplier_x100,
        settings.vertical_below,
        settings.vertical_above,
        settings.biome_ambient_tint_enabled,
        settings.shadow_sun_enabled,
        settings.shadow_block_enabled,
        settings.gi_enabled,
        settings.shadow_quality,
    );

    config::set_render_distance(settings.render_distance);
    config::set_lod_multiplier(settings.lod_multiplier_x100 as f32 / 100.0);
    config::set_vertical_below(settings.vertical_below);
    config::set_vertical_above(settings.vertical_above);
    config::set_biome_ambient_tint_enabled(settings.biome_ambient_tint_enabled);
    config::set_shadow_sun_enabled(settings.shadow_sun_enabled);
    config::set_shadow_block_enabled(settings.shadow_block_enabled);
    config::set_gi_enabled(settings.gi_enabled);
    config::set_shadow_quality(settings.shadow_quality);
}

/// Read current atomics and write them to the JSON file.
pub fn save_current() {
    let settings = UserSettings {
        render_distance:           config::render_distance(),
        lod_multiplier_x100:       (config::lod_multiplier() * 100.0).round() as u32,
        vertical_below:            config::vertical_below(),
        vertical_above:            config::vertical_above(),
        biome_ambient_tint_enabled: config::biome_ambient_tint_enabled(),
        shadow_sun_enabled:        config::shadow_sun_enabled(),
        shadow_block_enabled:      config::shadow_block_enabled(),
        gi_enabled:                config::gi_enabled(),
        shadow_quality:            config::shadow_quality(),
    };

    match serde_json::to_string_pretty(&settings) {
        Ok(json) => {
            if let Err(e) = std::fs::write(PATH, json) {
                debug_log!("UserSettings", "save_current", "Write error: {}", e);
            } else {
                debug_log!("UserSettings", "save_current", "Settings saved to {}", PATH);
            }
        }
        Err(e) => {
            debug_log!("UserSettings", "save_current", "Serialize error: {}", e);
        }
    }
}
