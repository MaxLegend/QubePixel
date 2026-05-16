//! # Geography Layer
//!
//! Macro-terrain generation layer that produces continental-scale features:
//! oceans, continents, islands, lowlands, highlands, and mountain peaks.
//!
//! Uses fractal Brownian motion (Fbm) noise at very large scales to create
//! realistic continental distributions, with a high-frequency Perlin noise
//! island mask for spawning archipelagos in ocean zones.
//!
//! ## Usage
//! ```ignore
//! let geo = GeographyLayer::new(42);
//! let info = geo.sample(1024, 2048);
//! println!("Height: {}, Ocean: {}", info.surface_height, info.is_ocean);
//! ```

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use rayon::prelude::*;
use crate::core::config;
use crate::debug_log;
use crate::flow_debug_log;
// ---------------------------------------------------------------------------
// Config-backed accessors — read from assets/world_gen_config.json
// ---------------------------------------------------------------------------

/// These functions read from the global config (loaded once via OnceLock).
/// They are used by other modules (stratification, pipeline) that need
/// geography thresholds.

pub fn deep_ocean_max() -> f64 { config::world_gen_config().geography.deep_ocean_max }
pub fn ocean_max_val() -> f64  { config::world_gen_config().geography.ocean_max }
pub fn sea_level_val() -> i32  { config::world_gen_config().geography.sea_level }

// Backward-compatible pub constants for use in other modules' `use` statements.
// These are the default values; for config-aware code, use the functions above.
pub const DEEP_OCEAN_MAX: f64 = 0.2;
pub const OCEAN_MAX: f64 = 0.4;
pub const LOWLAND_MAX: f64 = 0.6;
pub const MIDLAND_MAX: f64 = 0.8;
pub const SEA_LEVEL: i32 = 64;

// ---------------------------------------------------------------------------
// GeoInfo
// ---------------------------------------------------------------------------

/// Geographic information for a given `(x, z)` world position.
///
/// Encapsulates all macro-terrain data needed by higher-level terrain
/// generators (biome selection, block placement, structure spawning, etc.).
#[derive(Debug, Clone, Copy)]
pub struct GeoInfo {
    /// Continentalness: `0.0` = deep ocean, `1.0` = mountain peaks.
    ///
    /// This is the primary driver of terrain height and biome selection.
    pub continentalness: f64,

    /// Erosion factor: `0.0` = heavily eroded (flat), `1.0` = sharp peaks.
    ///
    /// Controls local height variation — high erosion produces dramatic
    /// peaks and valleys, low erosion produces rolling plains.
    pub erosion: f64,

    /// Peaks and valleys factor: combined value for extreme terrain features.
    ///
    /// Higher values correlate with more mountainous/alpine terrain.
    pub peaks_and_valleys: f64,

    /// Computed surface height in block Y coordinates.
    ///
    /// Derived from continentalness, erosion noise, and the continentalness
    /// zone (deep ocean, shallow ocean, lowland, midland, highland).
    /// For river columns this already includes the carved depth.
    pub surface_height: i32,

    /// Whether this position is under water (`surface_height < sea_level`).
    pub is_ocean: bool,

    /// Ocean depth in blocks below sea level.
    ///
    /// Only meaningful when [`is_ocean`](GeoInfo::is_ocean) is `true`.
    pub ocean_depth: i32,

    /// River channel strength at this position, in `[0.0, 1.0]`.
    ///
    /// `0.0` = no river; `1.0` = centre of a river channel.
    /// Only non-zero on land columns (never set for ocean).
    pub river_strength: f64,

    /// `true` when this column is a river channel (`river_strength > 0.3`).
    ///
    /// Used by the stratification layer to place gravel/sand as the river bed
    /// surface block instead of the biome's normal surface block.
    pub is_river: bool,
}

// ---------------------------------------------------------------------------
// GeographyLayer
// ---------------------------------------------------------------------------

/// Macro-terrain generator producing continent-scale geographic features.
///
/// Combines multiple octaves of Fbm noise for continentalness, erosion,
/// and peaks with a high-frequency Perlin island mask. The resulting
/// terrain has:
/// - Deep oceans with dramatic depth variation
/// - Shallow coastal shelves
/// - Continental lowlands and plains
/// - Midland hills and valleys
/// - Highland mountains with sharp peaks
/// - Island chains in ocean zones
///
/// # Thread Safety
/// This type is `Send + Sync` and safe to share across threads.
pub struct GeographyLayer {
    /// Low-frequency Fbm noise for continental-scale features.
    continentalness_noise: Fbm<Perlin>,
    /// Medium-frequency Fbm noise for erosion detail.
    erosion_noise: Fbm<Perlin>,
    /// Low-frequency Fbm noise for peaks and valleys.
    peaks_noise: Fbm<Perlin>,
    /// High-frequency Perlin noise for island archipelago masking.
    island_mask: Perlin,
    /// Medium-frequency Fbm noise for river network detection.
    /// Rivers form at zero-crossings of this noise field.
    river_noise: Fbm<Perlin>,
    /// World seed.
    seed: u64,

    // Config values (captured at construction for hot-path perf).
    continent_scale: f64,
    erosion_scale: f64,
    peaks_scale: f64,
    island_scale: f64,
    island_threshold: f64,
    island_boost: f64,
    cfg_deep_ocean_max: f64,
    cfg_ocean_max: f64,
    cfg_lowland_max: f64,
    cfg_midland_max: f64,
    cfg_sea_level: i32,
    terrain_base: i32,

    // River config.
    river_scale: f64,
    river_threshold: f64,
    river_carve_depth: f64,
}

impl GeographyLayer {
    /// Creates a new geography layer with the given world seed.
    ///
    /// Each noise generator receives a derived seed to ensure independent
    /// but reproducible noise patterns:
    /// - `seed + 0` → continentalness
    /// - `seed + 1` → erosion
    /// - `seed + 2` → peaks and valleys
    /// - `seed + 3` → island mask
    ///
    /// # Arguments
    /// * `seed` - World seed (any `u64` value).
    ///
    /// # Examples
    /// ```ignore
    /// let layer = GeographyLayer::new(12345);
    /// let info = layer.sample(0, 0);
    /// ```
    pub fn new(seed: u64) -> Self {
        let cfg = &config::world_gen_config().geography;

        debug_log!(
            "GeographyLayer",
            "new",
            "Creating geography layer with seed={} continent_scale={:.0}",
            seed,
            cfg.continent_scale
        );

        let continentalness_noise = Fbm::new(seed as u32)
            .set_octaves(cfg.continent_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let erosion_noise = Fbm::new((seed + 1) as u32)
            .set_octaves(cfg.erosion_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let peaks_noise = Fbm::new((seed + 2) as u32)
            .set_octaves(cfg.peaks_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let island_mask = Perlin::new((seed + 3) as u32);

        // River network: 3-octave Fbm; rivers form at zero-crossings.
        let river_noise = Fbm::new((seed + 4) as u32)
            .set_octaves(3)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        Self {
            continentalness_noise,
            erosion_noise,
            peaks_noise,
            island_mask,
            river_noise,
            seed,
            continent_scale: cfg.continent_scale,
            erosion_scale: cfg.erosion_scale,
            peaks_scale: cfg.peaks_scale,
            island_scale: cfg.island_scale,
            island_threshold: cfg.island_threshold,
            island_boost: cfg.island_boost,
            cfg_deep_ocean_max: cfg.deep_ocean_max,
            cfg_ocean_max: cfg.ocean_max,
            cfg_lowland_max: cfg.lowland_max,
            cfg_midland_max: cfg.midland_max,
            cfg_sea_level: cfg.sea_level,
            terrain_base: cfg.terrain_base,
            river_scale: cfg.river_scale,
            river_threshold: cfg.river_threshold,
            river_carve_depth: cfg.river_carve_depth,
        }
    }

    /// Constructs a geography layer using an explicit config (bypasses global).
    pub fn with_config(seed: u64, cfg: &crate::core::config::GeographyConfig) -> Self {
        let continentalness_noise = Fbm::new(seed as u32)
            .set_octaves(cfg.continent_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let erosion_noise = Fbm::new((seed + 1) as u32)
            .set_octaves(cfg.erosion_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let peaks_noise = Fbm::new((seed + 2) as u32)
            .set_octaves(cfg.peaks_octaves)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        let island_mask = Perlin::new((seed + 3) as u32);

        let river_noise = Fbm::new((seed + 4) as u32)
            .set_octaves(3)
            .set_persistence(0.5)
            .set_lacunarity(2.0)
            .set_frequency(1.0);

        Self {
            continentalness_noise,
            erosion_noise,
            peaks_noise,
            island_mask,
            river_noise,
            seed,
            continent_scale: cfg.continent_scale,
            erosion_scale: cfg.erosion_scale,
            peaks_scale: cfg.peaks_scale,
            island_scale: cfg.island_scale,
            island_threshold: cfg.island_threshold,
            island_boost: cfg.island_boost,
            cfg_deep_ocean_max: cfg.deep_ocean_max,
            cfg_ocean_max: cfg.ocean_max,
            cfg_lowland_max: cfg.lowland_max,
            cfg_midland_max: cfg.midland_max,
            cfg_sea_level: cfg.sea_level,
            terrain_base: cfg.terrain_base,
            river_scale: cfg.river_scale,
            river_threshold: cfg.river_threshold,
            river_carve_depth: cfg.river_carve_depth,
        }
    }

    /// Samples geographic information at the given world coordinates.
    ///
    /// Computes continentalness, erosion, peaks, applies the island mask,
    /// derives surface height using smooth zone interpolation, and carves
    /// river channels where the river noise crosses zero.
    ///
    /// # Arguments
    /// * `wx` - World X coordinate (blocks).
    /// * `wz` - World Z coordinate (blocks).
    ///
    /// # Returns
    /// A [`GeoInfo`] struct with all geographic data for this position.
    pub fn sample(&self, wx: i32, wz: i32) -> GeoInfo {
        let nx_cont = wx as f64 / self.continent_scale;
        let nz_cont = wz as f64 / self.continent_scale;

        let raw_cont = self.continentalness_noise.get([nx_cont, nz_cont]);
        let mut continentalness = map_to_01(raw_cont);

        let nx_eros = wx as f64 / self.erosion_scale;
        let nz_eros = wz as f64 / self.erosion_scale;
        let raw_eros = self.erosion_noise.get([nx_eros, nz_eros]);
        let erosion = map_to_01(raw_eros);

        let raw_peaks = self.peaks_noise.get([nx_cont, nz_cont]);
        let peaks_and_valleys = map_to_01(raw_peaks);

        // Island mask — in ocean zones, high-frequency Perlin raises continentalness.
        if continentalness < self.cfg_ocean_max {
            let island_val =
                self.island_mask.get([wx as f64 / self.island_scale, wz as f64 / self.island_scale]);
            if island_val > self.island_threshold {
                continentalness = (continentalness + self.island_boost).min(1.0);
            }
        }

        let noise_variation = (erosion - 0.5) * 20.0;
        // Smooth zone-transition formula — no hard steps at zone boundaries.
        let base_height = self.compute_surface_height(continentalness, noise_variation);

        // River detection: rivers form at zero-crossings of the river noise.
        // Only on land columns; ocean columns never host rivers.
        let base_is_ocean = base_height < self.cfg_sea_level;
        let (surface_height, river_strength, is_river) = if !base_is_ocean {
            let rx = wx as f64 / self.river_scale;
            let rz = wz as f64 / self.river_scale;
            let river_raw = self.river_noise.get([rx, rz]);
            let river_dist = river_raw.abs();

            if river_dist < self.river_threshold {
                let raw_strength = 1.0 - river_dist / self.river_threshold;
                // Smoothstep so river walls slope gently rather than step.
                let s = smoothstep(0.0, 1.0, raw_strength);
                // Mountains get shallower rivers/streams than lowlands.
                let carve_mult = if continentalness > self.cfg_midland_max { 0.35 } else { 1.0 };
                let carve = (s * self.river_carve_depth * carve_mult).round() as i32;
                (base_height - carve, s, s > 0.3)
            } else {
                (base_height, 0.0, false)
            }
        } else {
            (base_height, 0.0, false)
        };

        let is_ocean = surface_height < self.cfg_sea_level;
        let ocean_depth = if is_ocean { self.cfg_sea_level - surface_height } else { 0 };

        flow_debug_log!(
            "GeographyLayer",
            "sample",
            "pos=({}, {}) cont={:.3} eros={:.3} peaks={:.3} height={} ocean={} river_str={:.2}",
            wx,
            wz,
            continentalness,
            erosion,
            peaks_and_valleys,
            surface_height,
            is_ocean,
            river_strength
        );

        GeoInfo {
            continentalness,
            erosion,
            peaks_and_valleys,
            surface_height,
            is_ocean,
            ocean_depth,
            river_strength,
            is_river,
        }
    }

    /// Samples geographic information for a rectangular area (e.g., one chunk).
    ///
    /// Uses Rayon for parallel evaluation across all positions in the area.
    /// The returned vector is row-major: index `dz * size + dx`.
    ///
    /// # Arguments
    /// * `cx` - Chunk X index (world start = `cx * size`).
    /// * `cz` - Chunk Z index (world start = `cz * size`).
    /// * `size` - Side length of the square area (typically 16 for chunks).
    ///
    /// # Returns
    /// A `Vec<GeoInfo>` with `size * size` elements in row-major order.
    pub fn sample_area(&self, cx: i32, cz: i32, size: usize) -> Vec<GeoInfo> {
        flow_debug_log!(
            "GeographyLayer",
            "sample_area",
            "Sampling area chunk=({}, {}) size={}",
            cx,
            cz,
            size
        );

        let start_x = cx * size as i32;
        let start_z = cz * size as i32;
        let total = size * size;

        let results: Vec<GeoInfo> = (0..total)
            .into_par_iter()
            .map(|i| {
                let dx = (i % size) as i32;
                let dz = (i / size) as i32;
                self.sample(start_x + dx, start_z + dz)
            })
            .collect();

        flow_debug_log!(
            "GeographyLayer",
            "sample_area",
            "Sampled {} positions for chunk ({}, {})",
            results.len(),
            cx,
            cz
        );

        results
    }

    /// Returns the world seed this layer was created with.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Computes surface height using **smooth hermite interpolation** between
    /// continentalness zones so zone boundaries never produce sharp cliffs.
    ///
    /// Each boundary has a `tw`-wide transition band where the two adjacent
    /// height formulae are blended with a smoothstep curve.
    ///
    /// Zones (centre values, before noise):
    /// - Deep ocean (`< deep_ocean_max`):  `sea_level − 30`
    /// - Shallow ocean (`< ocean_max`):    `sea_level − 15`
    /// - Lowland (`< lowland_max`):        `terrain_base + noise`
    /// - Midland (`< midland_max`):        `terrain_base + 20 + noise×2`
    /// - Highland (`≥ midland_max`):       `terrain_base + 40 + noise×3`
    #[inline]
    fn compute_surface_height(&self, continentalness: f64, noise_variation: f64) -> i32 {
        let sl = self.cfg_sea_level as f64;
        let tb = self.terrain_base as f64;

        // Half-width of each zone-transition band (5% of the [0,1] range).
        const TW: f64 = 0.05;

        let h_deep   = sl - 30.0 + noise_variation * 0.3;
        let h_shallow = sl - 15.0 + noise_variation * 0.5;
        let h_low    = tb + noise_variation;
        let h_mid    = tb + 20.0 + noise_variation * 2.0;
        let h_high   = tb + 40.0 + noise_variation * 3.0;

        let c = continentalness;
        let height = if c < self.cfg_deep_ocean_max - TW {
            h_deep
        } else if c < self.cfg_deep_ocean_max + TW {
            let t = smoothstep(self.cfg_deep_ocean_max - TW, self.cfg_deep_ocean_max + TW, c);
            lerp(h_deep, h_shallow, t)
        } else if c < self.cfg_ocean_max - TW {
            h_shallow
        } else if c < self.cfg_ocean_max + TW {
            let t = smoothstep(self.cfg_ocean_max - TW, self.cfg_ocean_max + TW, c);
            lerp(h_shallow, h_low, t)
        } else if c < self.cfg_lowland_max - TW {
            h_low
        } else if c < self.cfg_lowland_max + TW {
            let t = smoothstep(self.cfg_lowland_max - TW, self.cfg_lowland_max + TW, c);
            lerp(h_low, h_mid, t)
        } else if c < self.cfg_midland_max - TW {
            h_mid
        } else if c < self.cfg_midland_max + TW {
            let t = smoothstep(self.cfg_midland_max - TW, self.cfg_midland_max + TW, c);
            lerp(h_mid, h_high, t)
        } else {
            h_high
        };

        height.round() as i32
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Maps a noise value from `[-1, 1]` to `[0, 1]`, clamping the result.
///
/// Note: Fbm noise can exceed [-1, 1], so clamping is necessary.
#[inline(always)]
fn map_to_01(val: f64) -> f64 {
    ((val + 1.0) * 0.5).clamp(0.0, 1.0)
}

/// Classic smoothstep: returns 0 at `edge0`, 1 at `edge1`, cubic easing between.
#[inline(always)]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linear interpolation.
#[inline(always)]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_info_ocean_deep() {
        let layer = GeographyLayer::new(42);
        // Sample many positions; we expect some to be ocean
        let mut found_ocean = false;
        let mut found_land = false;
        for x in 0..100 {
            for z in 0..100 {
                let info = layer.sample(x * 100, z * 100);
                if info.is_ocean {
                    found_ocean = true;
                    assert!(info.ocean_depth > 0);
                    assert!(info.surface_height < SEA_LEVEL);
                } else {
                    found_land = true;
                    assert_eq!(info.ocean_depth, 0);
                }
            }
        }
        assert!(found_ocean, "Expected some ocean positions");
        assert!(found_land, "Expected some land positions");
    }

    #[test]
    fn test_sample_area_size() {
        let layer = GeographyLayer::new(42);
        let results = layer.sample_area(0, 0, 16);
        assert_eq!(results.len(), 256);
    }

    #[test]
    fn test_continentalness_range() {
        let layer = GeographyLayer::new(42);
        for x in 0..50 {
            for z in 0..50 {
                let info = layer.sample(x * 200, z * 200);
                assert!(
                    (0.0..=1.0).contains(&info.continentalness),
                    "continentalness out of range: {}",
                    info.continentalness
                );
                assert!(
                    (0.0..=1.0).contains(&info.erosion),
                    "erosion out of range: {}",
                    info.erosion
                );
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let layer = GeographyLayer::new(999);
        let a = layer.sample(500, 500);
        let b = layer.sample(500, 500);
        assert_eq!(a.continentalness, b.continentalness);
        assert_eq!(a.surface_height, b.surface_height);
    }
}
