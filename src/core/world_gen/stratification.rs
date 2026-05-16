//! # Vertical Block Stratification
//!
//! Determines which block types go at which Y-level for each world column.
//!
//! This module translates high-level geological, geographical, and climate
//! information into concrete block IDs by consulting the [`BlockRegistry`].
//! Each vertical column receives a [`ColumnStratigraphy`] describing the exact
//! block palette and layer thicknesses for that position.
//!
//! ## Layer stack (top → bottom)
//!
//! | Layer | Typical blocks | Thickness |
//! |-------|---------------|-----------|
//! | Surface | Grass / Sand / Snow / Ice | 1 block |
//! | Dirt / Soil | Climate-dependent dirt variant | 4–5 blocks |
//! | Soft Clay | Geological clay variant | 4–6 blocks |
//! | Hard Clay | Geological hard clay variant | 10–20 blocks |
//! | Stone | Base rock (geology-dependent) | Until Y = 32 |
//! | Deep Granite | Basement rock | Below Y = 32 |
//!
//! ## Noise-driven variation
//!
//! Layer thicknesses are modulated by a Perlin noise field so that adjacent
//! columns have smoothly varying depths, avoiding flat "layer-cake" terrain.

use noise::{NoiseFn, Perlin};

use crate::core::config;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::world_gen::{ClimateInfo, ClimateType, GeoInfo, GeologyInfo};
use crate::core::world_gen::geography::{ocean_max_val, sea_level_val};
use crate::{debug_log, ext_debug_log};
use crate::flow_debug_log;
// ---------------------------------------------------------------------------
// ColumnStratigraphy
// ---------------------------------------------------------------------------

/// Complete block-ID palette and layer geometry for a **single** vertical column
/// at world coordinates `(wx, wz)`.
///
/// This struct is produced by [`StratificationLayer::compute_column`] and
/// consumed by [`StratificationLayer::get_block_at_y`] to resolve the block at
/// any arbitrary Y-level.
///
/// # Thread safety
///
/// `ColumnStratigraphy` is `Clone` + `Send` + `Sync` and can be freely shared
/// across rayon workers.
#[derive(Debug, Clone)]
pub struct ColumnStratigraphy {
    // -- Surface / cover blocks ------------------------------------------

    /// Grass (or grass-variant) block ID for this column.
    ///
    /// Climate- and geology-dependent. For ocean columns this is set to
    /// `sand_id` (the ocean floor surface).
    pub grass_id: u8,

    /// Dirt / soil block ID.
    ///
    /// Typically a geology-dependent variant (e.g. `"dirt/dirt_gneiss"`).
    pub dirt_id: u8,

    /// Clay block ID.
    ///
    /// Transition zone between dirt and claystone.
    pub clay_id: u8,

    /// Claystone block ID.
    ///
    /// Hardened clay layer sitting directly above the base stone.
    pub claystone_id: u8,

    // -- Rock blocks -----------------------------------------------------

    /// Base stone block ID (determined by micro-region geology).
    ///
    /// E.g. `"rocks/gneiss"`, `"rocks/basalt"`, `"rocks/limestone"`.
    pub stone_id: u8,

    /// Deep granite block ID used below [`DEEP_GRANITE_THRESHOLD`].
    pub deep_granite_id: u8,

    // -- Fluid / special blocks ------------------------------------------

    /// Water block ID (for oceans and rivers).
    pub water_id: u8,

    /// Sand block ID (beaches, deserts, ocean floors).
    pub sand_id: u8,

    /// Snow block ID (polar / high-altitude surfaces).
    pub snow_id: u8,

    /// Ice block ID (frozen ocean surfaces).
    pub ice_id: u8,

    /// Gravel block ID (underwater deposits, cave floors).
    pub gravel_id: u8,

    // -- Layer geometry --------------------------------------------------

    /// Thickness of the dirt layer in blocks (typically 4–5).
    pub dirt_thickness: u8,

    /// Thickness of the clay layer in blocks (typically 4–6).
    pub clay_thickness: u8,

    /// Thickness of the claystone layer in blocks (typically 10–20).
    pub claystone_thickness: u8,

    /// World-space surface height (Y-coordinate of the topmost solid block).
    pub surface_height: i32,

    // -- Column flags ----------------------------------------------------

    /// `true` when this column is below sea level (ocean / river bed).
    pub is_ocean: bool,

    /// `true` when the surface water should be frozen (ice instead of water).
    pub is_frozen: bool,
}

// ---------------------------------------------------------------------------
// StratificationLayer
// ---------------------------------------------------------------------------

/// Generates [`ColumnStratigraphy`] values for arbitrary world positions.
///
/// A single `StratificationLayer` instance is shared across all rayon worker
/// threads — it is `Send + Sync` and contains no mutable state after
/// construction.
///
/// # Example
///
/// ```ignore
/// use qubepixel_terrain::stratification::StratificationLayer;
///
/// let strat = StratificationLayer::new(42);
/// let col = strat.compute_column(100, -200, &geo, &climate, &geology, &registry);
/// println!("surface block at Y={}: id={}", col.surface_height, col.grass_id);
/// ```
pub struct StratificationLayer {
    seed: u64,
    thickness_noise: Perlin,

    // Config values captured at construction.
    deep_granite_threshold: i32,
    deep_granite_transition: i32,
    thickness_noise_scale: f64,
    dirt_base: u8,
    dirt_var: u8,
    clay_base: u8,
    clay_var: u8,
    claystone_base: u8,
    claystone_var: u8,
    ocean_max: f64,
    sea_level: i32,
}

impl StratificationLayer {
    /// Create a new stratification layer.
    ///
    /// # Arguments
    ///
    /// * `seed` — world seed. A Perlin noise instance is seeded with
    ///   `seed + 500` to avoid correlation with other noise fields.
    pub fn new(seed: u64) -> Self {
        let scfg = &config::world_gen_config().stratification;
        debug_log!("StratificationLayer", "new", "seed={}", seed);
        let thickness_noise = Perlin::new((seed.wrapping_add(500)) as u32);
        Self {
            seed,
            thickness_noise,
            deep_granite_threshold: scfg.deep_granite_threshold,
            deep_granite_transition: scfg.deep_granite_transition,
            thickness_noise_scale: scfg.thickness_noise_scale,
            dirt_base: scfg.dirt_thickness_base,
            dirt_var: scfg.dirt_thickness_variation,
            clay_base: scfg.clay_thickness_base,
            clay_var: scfg.clay_thickness_variation,
            claystone_base: scfg.claystone_thickness_base,
            claystone_var: scfg.claystone_thickness_variation,
            ocean_max: ocean_max_val(),
            sea_level: sea_level_val(),
        }
    }

    /// Compute the full stratigraphy for the column at world coordinates
    /// `(wx, wz)`.
    ///
    /// This is the main entry point. It consults the block registry to map
    /// geological / climate descriptors to concrete block IDs, then uses a
    /// noise field to add spatial variation to layer thicknesses.
    ///
    /// # Arguments
    ///
    /// * `wx`, `wz` — world X and Z coordinates of the column.
    /// * `geo` — geographical info (elevation, continentalness, ocean flag).
    /// * `climate` — climate info (temperature, humidity, climate type).
    /// * `geology` — geological info (micro-region with rock/soil variants).
    /// * `registry` — block registry used to resolve string names → `u8` IDs.
    ///
    /// # Returns
    ///
    /// A [`ColumnStratigraphy`] fully describing the vertical block stack.
    pub fn compute_column(
        &self,
        wx: i32,
        wz: i32,
        geo: &GeoInfo,
        climate: &ClimateInfo,
        geology: &GeologyInfo,
        registry: &BlockRegistry,
    ) -> ColumnStratigraphy {
        flow_debug_log!(
            "StratificationLayer",
            "compute_column",
            "wx={} wz={}",
            wx,
            wz
        );

        // -- 1. Base rock (geology-dependent) ----------------------------
        let stone_name = geology.micro_region.stone_name();
        let stone_id = registry
            .id_for(stone_name)
            .unwrap_or_else(|| {
                ext_debug_log!(
                    "StratificationLayer",
                    "compute_column",
                    "unknown stone '{}', falling back to 'rocks/granite'",
                    stone_name
                );
                registry.id_for("rocks/granite").unwrap_or(1)
            });

        let deep_granite_id = registry
            .id_for("rocks/deep_granite")
            .unwrap_or(stone_id);

        // -- 2. Special blocks (with fallbacks) -------------------------
        let water_id = registry.id_for("water").unwrap_or(0);
        let sand_id = registry.id_for("sand").unwrap_or(0);
        let snow_id = registry.id_for("snow").unwrap_or(0);
        let ice_id = registry.id_for("ice").unwrap_or(0);
        let gravel_id = registry.id_for("gravel").unwrap_or(stone_id);

        // -- 3. Surface block determination -------------------------------
        let mut is_ocean = geo.is_ocean;
        let mut is_frozen = false;

        // River bed: gravel replaces any normal surface block.
        // This applies to both land rivers and rivers carved below sea level.
        let grass_id = if geo.is_river {
            gravel_id
        } else if is_ocean {
            match climate.climate_type {
                ClimateType::Arctic
                | ClimateType::Antarctic
                | ClimateType::Subantarctic => {
                    is_frozen = true;
                    sand_id
                }
                _ => sand_id,
            }
        } else if geo.continentalness < self.ocean_max + 0.05 {
            // Beach zone (land very close to sea level).
            sand_id
        } else {
            match climate.climate_type {
                ClimateType::Arctic | ClimateType::Antarctic => snow_id,
                _ => {
                    // Hot-and-dry check for desert.
                    let is_hot = matches!(
                        climate.climate_type,
                        ClimateType::Equatorial
                            | ClimateType::NorthTropical
                            | ClimateType::SouthTropical
                            | ClimateType::NorthSubequatorial
                            | ClimateType::SouthSubequatorial
                    );
                    if is_hot && climate.humidity < 0.3 {
                        sand_id
                    } else {
                        // Climate-dependent grass variant from geology.
                        let grass_name = geology.micro_region.grass_name();
                        registry.id_for(grass_name).unwrap_or_else(|| {
                            ext_debug_log!(
                                "StratificationLayer",
                                "compute_column",
                                "unknown grass '{}', falling back to 'grass/grass_granite'",
                                grass_name
                            );
                            registry.id_for("grass/grass_granite").unwrap_or(2)
                        })
                    }
                }
            }
        };

        // -- 4. Dirt layer (geology-dependent) ---------------------------
        let dirt_name = geology.micro_region.dirt_name();
        let dirt_id = registry.id_for(dirt_name).unwrap_or_else(|| {
            ext_debug_log!(
                "StratificationLayer",
                "compute_column",
                "unknown dirt '{}', falling back to 'dirt/dirt_granite'",
                dirt_name
            );
            registry.id_for("dirt/dirt_granite").unwrap_or(3)
        });

        // -- 5. Clay + Claystone layers (geology-dependent) --------------
        let clay_name = geology.micro_region.clay_name();
        let clay_id = registry.id_for(clay_name).unwrap_or_else(|| {
            ext_debug_log!(
                "StratificationLayer",
                "compute_column",
                "unknown clay '{}', falling back to 'clay/clay_granite'",
                clay_name
            );
            registry.id_for("clay/clay_granite").unwrap_or(4)
        });

        let claystone_name = geology.micro_region.claystone_name();
        let claystone_id = registry.id_for(claystone_name).unwrap_or_else(|| {
            ext_debug_log!(
                "StratificationLayer",
                "compute_column",
                "unknown claystone '{}', falling back to 'claystone/claystone_granite'",
                claystone_name
            );
            registry.id_for("claystone/claystone_granite").unwrap_or(5)
        });

        // -- 6. Layer thicknesses (noise-modulated) ----------------------
        let ns = self.thickness_noise_scale;
        let wx_f = wx as f64;
        let wz_f = wz as f64;

        let tn_dirt = self.thickness_noise.get([wx_f * ns, wz_f * ns]);
        let dirt_thickness: u8 = self.dirt_base + (tn_dirt.abs() * self.dirt_var as f64) as u8;

        let tn_clay = self.thickness_noise.get([wx_f * ns + 100.0, wz_f * ns + 100.0]);
        let clay_thickness: u8 = self.clay_base + (tn_clay.abs() * self.clay_var as f64) as u8;

        let tn_claystone = self.thickness_noise.get([wx_f * ns + 200.0, wz_f * ns + 200.0]);
        let claystone_thickness: u8 = self.claystone_base + (tn_claystone.abs() * self.claystone_var as f64) as u8;

        ext_debug_log!(
            "StratificationLayer",
            "compute_column",
            "dirt={} clay={} claystone={}",
            dirt_thickness,
            clay_thickness,
            claystone_thickness
        );

        ColumnStratigraphy {
            grass_id,
            dirt_id,
            clay_id,
            claystone_id,
            stone_id,
            deep_granite_id,
            water_id,
            sand_id,
            snow_id,
            ice_id,
            gravel_id,
            dirt_thickness,
            clay_thickness,
            claystone_thickness,
            surface_height: geo.surface_height,
            is_ocean,
            is_frozen,
        }
    }

    /// Determine the block ID at a specific world Y-level for a given column.
    ///
    /// This is a **pure function** (no `&self`) — it only reads the
    /// pre-computed [`ColumnStratigraphy`] and a noise value for the
    /// stone-to-granite transition zone.
    ///
    /// # Block selection logic
    ///
    /// ```text
    ///  Y > surface            → air (or water/ice if ocean)
    ///  Y == surface           → grass_id
    ///  surface-1 .. -dirt     → dirt_id        (dirt_thickness blocks)
    ///  below dirt .. -clay    → clay_id        (clay_thickness blocks)
    ///  below clay .. -clyst   → claystone_id   (claystone_thickness blocks)
    ///  below claystone .. 32  → stone_id
    ///  transition 17..32      → noise-blended stone / deep_granite
    ///  Y < 17                 → deep_granite_id
    /// ```
    pub fn get_block_at_y(column: &ColumnStratigraphy, world_y: i32, noise_val: f64) -> u8 {
        let cfg = &config::world_gen_config().stratification;
        let granite_threshold = cfg.deep_granite_threshold;
        let granite_transition = cfg.deep_granite_transition;
        let sl = config::world_gen_config().geography.sea_level;

        let surface = column.surface_height;
        let dirt_bottom = surface - 1 - column.dirt_thickness as i32;
        let clay_bottom = dirt_bottom - column.clay_thickness as i32;
        let claystone_bottom = clay_bottom - column.claystone_thickness as i32;

        if world_y > surface {
            if column.is_ocean {
                if world_y <= sl {
                    if column.is_frozen { column.ice_id } else { column.water_id }
                } else {
                    0
                }
            } else {
                0
            }
        } else if world_y == surface {
            column.grass_id
        } else if world_y > dirt_bottom {
            column.dirt_id
        } else if world_y > clay_bottom {
            column.clay_id
        } else if world_y > claystone_bottom {
            column.claystone_id
        } else if world_y > granite_threshold {
            column.stone_id
        } else if world_y > granite_threshold - granite_transition {
            let t = (granite_threshold - world_y) as f64
                / granite_transition as f64;
            if noise_val.abs() < t {
                column.deep_granite_id
            } else {
                column.stone_id
            }
        } else {
            column.deep_granite_id
        }
    }
}

// ---------------------------------------------------------------------------
// Send + Sync assertions
// ---------------------------------------------------------------------------

// Compile-time verification that all public types can be shared across threads.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        assert_send_sync::<ColumnStratigraphy>();
        assert_send_sync::<StratificationLayer>();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a minimal `GeoInfo` for testing.
    #[allow(dead_code)]
    fn test_geo(surface_height: i32, is_ocean: bool, continentalness: f64) -> GeoInfo {
        GeoInfo {
            surface_height,
            is_ocean,
            continentalness,
            erosion: 0.5,
            peaks_and_valleys: 0.5,
            ocean_depth: 0,
            river_strength: 0.0,
            is_river: false,
        }
    }

    /// Helper to build a minimal `ClimateInfo` for testing.
    #[allow(dead_code)]
    fn test_climate(
        temperature: f64,
        humidity: f64,
        climate_type: ClimateType,
    ) -> ClimateInfo {
        ClimateInfo {
            temperature,
            humidity,
            climate_type,
            zone_index: 6, // Equatorial
        }
    }

    #[test]
    fn test_stratification_layer_new() {
        let strat = StratificationLayer::new(42);
        assert_eq!(strat.seed, 42);
    }

    #[test]
    fn test_get_block_at_y_deep_granite() {
        // Surface at Y=64, granite below Y=32.
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        // Well below the granite threshold.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 0, 0.0), 7);
        assert_eq!(StratificationLayer::get_block_at_y(&col, -10, 0.0), 7);
    }

    #[test]
    fn test_get_block_at_y_stone() {
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        // With off-by-one fix:
        // dirt_bottom = 64 - 1 - 4 = 59
        // clay_bottom = 59 - 4 = 55
        // claystone_bottom = 55 - 10 = 45
        // Stone: Y in [33, 45]
        assert_eq!(StratificationLayer::get_block_at_y(&col, 40, 0.0), 6);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 33, 0.0), 6);
    }

    #[test]
    fn test_get_block_at_y_surface() {
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        assert_eq!(StratificationLayer::get_block_at_y(&col, 64, 0.0), 2);
    }

    #[test]
    fn test_get_block_at_y_air_above_surface() {
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        assert_eq!(StratificationLayer::get_block_at_y(&col, 65, 0.0), 0);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 100, 0.0), 0);
    }

    #[test]
    fn test_get_block_at_y_ocean_water() {
        // Sea level from config is 51.  Ocean floor at 44 (below sea level).
        let col = ColumnStratigraphy {
            grass_id: 9,  // sand (ocean floor)
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 44, // below sea level (51)
            is_ocean: true,
            is_frozen: false,
        };

        // Above ocean floor, at/below sea level (44 < y <= 51) → water.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 48, 0.0), 8);
        // Above sea level → air.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 55, 0.0), 0);
        // At ocean floor surface → sand.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 44, 0.0), 9);
    }

    #[test]
    fn test_get_block_at_y_frozen_ocean() {
        // Sea level from config is 51.  Ocean floor at 42.
        let col = ColumnStratigraphy {
            grass_id: 9,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 42,
            is_ocean: true,
            is_frozen: true,
        };

        // Between floor and sea level on a frozen ocean → ice.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 47, 0.0), 11);
    }

    #[test]
    fn test_get_block_at_y_transition_zone() {
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        // Transition zone is Y = 17..=32. Formula: t = (32-Y)/16.
        // At Y = 32 (top): t = 0.0, |noise| < 0.0 never true → always stone.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 32, 0.0), 6);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 32, 0.5), 6);

        // At Y = 16 (bottom): t = 1.0, |noise| < 1.0 almost always → granite.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 16, 0.0), 7);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 16, 0.5), 7);

        // At Y = 24 (middle): t = 0.5. |0.3| < 0.5 → granite; |0.7| >= 0.5 → stone.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 24, 0.3), 7);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 24, 0.7), 6);
    }

    #[test]
    fn test_get_block_at_y_dirt_layer() {
        let col = ColumnStratigraphy {
            grass_id: 2,
            dirt_id: 3,
            clay_id: 4,
            claystone_id: 5,
            stone_id: 6,
            deep_granite_id: 7,
            water_id: 8,
            sand_id: 9,
            snow_id: 10,
            ice_id: 11,
            gravel_id: 12,
            dirt_thickness: 4,
            clay_thickness: 4,
            claystone_thickness: 10,
            surface_height: 64,
            is_ocean: false,
            is_frozen: false,
        };

        // dirt_bottom = 64 - 1 - 4 = 59.
        // Dirt: Y in [60, 63] → 4 blocks.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 63, 0.0), 3);
        assert_eq!(StratificationLayer::get_block_at_y(&col, 60, 0.0), 3);
        // Y = 59 → clay.
        assert_eq!(StratificationLayer::get_block_at_y(&col, 59, 0.0), 4);
    }
}
