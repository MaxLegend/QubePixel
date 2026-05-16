//! # Geology Layer
//!
//! Voronoi-based geological region system that divides the world into
//! macro and micro rock-type regions without depending on external noise
//! for the Voronoi computation.
//!
//! Two levels of Voronoi tessellation are used:
//!
//! 1. **Macro regions** (cell size 2048 blocks): Determine broad geological
//!    category — Metamorphic, Sedimentary, Igneous, or Mantle.
//! 2. **Micro regions** (cell size 512 blocks): Determine specific rock type
//!    within the macro category (e.g., Basalt vs Granite vs Andesite for Igneous).
//!
//! Mantle regions are rare, appearing only as patchy tectonic intrusions
//! (~4% of total area) to simulate deep geological activity.
//!
//! ## Hash-based Feature Points
//! Each Voronoi cell has a jittered "feature point" derived from a fast
//! deterministic hash. The nearest feature point to any world position
//! determines its region assignment.
//!
//! ## Usage
//! ```ignore
//! let geology = GeologyLayer::new(42);
//! let info = geology.sample(1024, 2048);
//! println!("Macro: {:?}, Micro: {:?}", info.macro_region, info.micro_region);
//! println!("Stone: {}", info.micro_region.stone_name());
//! ```

use rayon::prelude::*;
use crate::core::config;
use crate::debug_log;
use crate::flow_debug_log;

const SEARCH_RADIUS: i32 = 1;

// ---------------------------------------------------------------------------
// Hash function
// ---------------------------------------------------------------------------

/// Fast deterministic hash combining a seed and two integer coordinates.
///
/// Uses multiplicative hashing with a large odd constant for good
/// avalanche properties. The result is a `u64` suitable for deriving
/// jitter offsets and region assignments.
///
/// # Algorithm
/// ```text
/// h = seed
/// h = h * 0x517cc1b727220a95 XOR x
/// h = h * 0x517cc1b727220a95 XOR z
/// h = h * 0x517cc1b727220a95
/// ```
///
/// # Arguments
/// * `seed` - Base seed value.
/// * `x` - First coordinate (typically cell X).
/// * `z` - Second coordinate (typically cell Z).
///
/// # Returns
/// A `u64` hash value.
#[inline]
fn hash_coord(seed: u64, x: i32, z: i32) -> u64 {
    const MULTIPLIER: u64 = 0x517cc1b7_27220a95;
    let mut h = seed;
    h = h.wrapping_mul(MULTIPLIER);
    h ^= x as u64;
    h = h.wrapping_mul(MULTIPLIER);
    h ^= z as u64;
    h = h.wrapping_mul(MULTIPLIER);
    h
}

// ---------------------------------------------------------------------------
// MacroRegion
// ---------------------------------------------------------------------------

/// Macro geological region — broad rock category.
///
/// Determines the general type of geology at a world position, which in
/// turn constrains the possible micro (specific) rock types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MacroRegion {
    /// Metamorphic rock: gneiss, schist, marble.
    /// Formed by heat and pressure transforming existing rock.
    Metamorphic,

    /// Sedimentary rock: sandstone, limestone, shale.
    /// Formed by accumulation and compression of mineral particles.
    Sedimentary,

    /// Igneous rock: basalt, granite, andesite.
    /// Formed by cooling of magma or lava.
    Igneous,

    /// Mantle-derived rock: lherzolite, kimberlite, ophiolite.
    /// Rare, appears as tectonic intrusions from deep geological activity.
    Mantle,
}

impl MacroRegion {
    /// Returns a human-readable name for this macro region.
    pub fn name(&self) -> &'static str {
        match self {
            MacroRegion::Metamorphic => "Metamorphic",
            MacroRegion::Sedimentary => "Sedimentary",
            MacroRegion::Igneous => "Igneous",
            MacroRegion::Mantle => "Mantle",
        }
    }

    /// Maps a hash remainder to a macro region.
    ///
    /// Distribution: 25% each for Metamorphic, Sedimentary, Igneous;
    /// ~4% Mantle (further reduced by the mantle rarity filter).
    #[inline]
    fn from_hash(hash: u64) -> Self {
        match hash % 4 {
            0 => MacroRegion::Metamorphic,
            1 => MacroRegion::Sedimentary,
            2 => MacroRegion::Igneous,
            _ => MacroRegion::Mantle,
        }
    }
}

// ---------------------------------------------------------------------------
// MicroRegion
// ---------------------------------------------------------------------------

/// Micro geological region — specific rock type within a macro category.
///
/// There are 12 micro regions total (3 per macro category). The specific
/// rock type determines block textures and material properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MicroRegion {
    // Metamorphic
    /// Banded foliated metamorphic rock.
    Gneiss,
    /// Medium-grained foliated metamorphic rock.
    Schist,
    /// Non-foliated metamorphic rock (recrystallized limestone).
    Marble,

    // Sedimentary
    /// Granular sedimentary rock from sand consolidation.
    Sandstone,
    /// Carbonate sedimentary rock from marine organisms.
    Limestone,
    /// Fine-grained fissile sedimentary rock from clay.
    Shale,

    // Igneous
    /// Fine-grained mafic extrusive rock (volcanic).
    Basalt,
    /// Coarse-grained felsic intrusive rock (plutonic).
    Granite,
    /// Intermediate extrusive rock (volcanic).
    Andesite,

    // Mantle
    /// Ultramafic peridotite rock from Earth's upper mantle.
    Lherzolite,
    /// Ultramafic igneous rock known for diamond-bearing pipes.
    Kimberlite,
    /// Section of oceanic crust and upper mantle.
    Ophiolite,
}

impl MicroRegion {
    /// Returns the base stone block name for registry lookup.
    ///
    /// Names include directory prefix matching block_registry.json.
    /// Note: Shale maps to "slate" in block files (geological naming discrepancy).
    pub fn stone_name(&self) -> &'static str {
        match self {
            MicroRegion::Gneiss => "rocks/gneiss",
            MicroRegion::Schist => "rocks/schist",
            MicroRegion::Marble => "rocks/marble",
            MicroRegion::Sandstone => "rocks/sandstone",
            MicroRegion::Limestone => "rocks/limestone",
            MicroRegion::Shale => "rocks/slate",
            MicroRegion::Basalt => "rocks/basalt",
            MicroRegion::Granite => "rocks/granite",
            MicroRegion::Andesite => "rocks/andesite",
            MicroRegion::Lherzolite => "rocks/lherzolite",
            MicroRegion::Kimberlite => "rocks/kimberlite",
            MicroRegion::Ophiolite => "rocks/ophiolite",
        }
    }

    /// Returns the grass block name for this region's surface grass.
    pub fn grass_name(&self) -> &'static str {
        match self {
            MicroRegion::Gneiss => "grass/grass_gneiss",
            MicroRegion::Schist => "grass/grass_schist",
            MicroRegion::Marble => "grass/grass_marble",
            MicroRegion::Sandstone => "grass/grass_sandstone",
            MicroRegion::Limestone => "grass/grass_limestone",
            MicroRegion::Shale => "grass/grass_slate",
            MicroRegion::Basalt => "grass/grass_basalt",
            MicroRegion::Granite => "grass/grass_granite",
            MicroRegion::Andesite => "grass/grass_andesite",
            MicroRegion::Lherzolite => "grass/grass_lherzolite",
            MicroRegion::Kimberlite => "grass/grass_kimberlite",
            MicroRegion::Ophiolite => "grass/grass_ophiolite",
        }
    }

    /// Returns the dirt block name for this region's subsurface soil.
    pub fn dirt_name(&self) -> &'static str {
        match self {
            MicroRegion::Gneiss => "dirt/dirt_gneiss",
            MicroRegion::Schist => "dirt/dirt_schist",
            MicroRegion::Marble => "dirt/dirt_marble",
            MicroRegion::Sandstone => "dirt/dirt_sandstone",
            MicroRegion::Limestone => "dirt/dirt_limestone",
            MicroRegion::Shale => "dirt/dirt_slate",
            MicroRegion::Basalt => "dirt/dirt_basalt",
            MicroRegion::Granite => "dirt/dirt_granite",
            MicroRegion::Andesite => "dirt/dirt_andesite",
            MicroRegion::Lherzolite => "dirt/dirt_lherzolite",
            MicroRegion::Kimberlite => "dirt/dirt_kimberlite",
            MicroRegion::Ophiolite => "dirt/dirt_ophiolite",
        }
    }

    /// Returns the clay block name for this region.
    ///
    /// Clay sits between dirt and claystone in the vertical stack.
    pub fn clay_name(&self) -> &'static str {
        match self {
            MicroRegion::Gneiss => "clay/clay_gneiss",
            MicroRegion::Schist => "clay/clay_schist",
            MicroRegion::Marble => "clay/clay_marble",
            MicroRegion::Sandstone => "clay/clay_sandstone",
            MicroRegion::Limestone => "clay/clay_limestone",
            MicroRegion::Shale => "clay/clay_slate",
            MicroRegion::Basalt => "clay/clay_basalt",
            MicroRegion::Granite => "clay/clay_granite",
            MicroRegion::Andesite => "clay/clay_andesite",
            MicroRegion::Lherzolite => "clay/clay_lherzolite",
            MicroRegion::Kimberlite => "clay/clay_kimberlite",
            MicroRegion::Ophiolite => "clay/clay_ophiolite",
        }
    }

    /// Returns the claystone block name for this region.
    ///
    /// Claystone is the hardened clay layer above base stone.
    pub fn claystone_name(&self) -> &'static str {
        match self {
            MicroRegion::Gneiss => "claystone/claystone_gneiss",
            MicroRegion::Schist => "claystone/claystone_schist",
            MicroRegion::Marble => "claystone/claystone_marble",
            MicroRegion::Sandstone => "claystone/claystone_sandstone",
            MicroRegion::Limestone => "claystone/claystone_limestone",
            MicroRegion::Shale => "claystone/claystone_slate",
            MicroRegion::Basalt => "claystone/claystone_basalt",
            MicroRegion::Granite => "claystone/claystone_granite",
            MicroRegion::Andesite => "claystone/claystone_andesite",
            MicroRegion::Lherzolite => "claystone/claystone_lherzolite",
            MicroRegion::Kimberlite => "claystone/claystone_kimberlite",
            MicroRegion::Ophiolite => "claystone/claystone_ophiolite",
        }
    }

    /// Returns the parent macro region for this micro region.
    pub fn macro_region(&self) -> MacroRegion {
        match self {
            MicroRegion::Gneiss | MicroRegion::Schist | MicroRegion::Marble => {
                MacroRegion::Metamorphic
            }
            MicroRegion::Sandstone | MicroRegion::Limestone | MicroRegion::Shale => {
                MacroRegion::Sedimentary
            }
            MicroRegion::Basalt | MicroRegion::Granite | MicroRegion::Andesite => {
                MacroRegion::Igneous
            }
            MicroRegion::Lherzolite | MicroRegion::Kimberlite | MicroRegion::Ophiolite => {
                MacroRegion::Mantle
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GeologyInfo
// ---------------------------------------------------------------------------

/// Geological information for a given `(x, z)` world position.
///
/// Contains both the broad macro region and specific micro region.
#[derive(Debug, Clone, Copy)]
pub struct GeologyInfo {
    /// Broad geological category.
    pub macro_region: MacroRegion,

    /// Specific rock type within the macro category.
    pub micro_region: MicroRegion,
}

// ---------------------------------------------------------------------------
// GeologyLayer
// ---------------------------------------------------------------------------

/// Geological region generator using manual Voronoi tessellation.
///
/// Two independent Voronoi grids determine the macro and micro regions
/// at any world position. All randomness comes from a deterministic
/// hash function — no external noise library is used.
///
/// # Thread Safety
/// This type is `Send + Sync` and safe to share across threads.
pub struct GeologyLayer {
    seed: u64,
    macro_cell_size: i32,
    micro_cell_size: i32,
    mantle_keep_denom: u64,
}

impl GeologyLayer {
    /// Creates a new geology layer with the given world seed.
    ///
    /// # Arguments
    /// * `seed` - World seed (any `u64` value).
    pub fn new(seed: u64) -> Self {
        let cfg = &config::world_gen_config().geology;

        debug_log!(
            "GeologyLayer",
            "new",
            "Geology layer seed={} macro_cell={} micro_cell={}",
            seed,
            cfg.macro_cell_size,
            cfg.micro_cell_size
        );

        Self {
            seed,
            macro_cell_size: cfg.macro_cell_size,
            micro_cell_size: cfg.micro_cell_size,
            mantle_keep_denom: cfg.mantle_keep_denominator,
        }
    }

    /// Constructs a geology layer using an explicit config (bypasses global).
    pub fn with_config(seed: u64, cfg: &crate::core::config::GeologyConfig) -> Self {
        Self {
            seed,
            macro_cell_size: cfg.macro_cell_size,
            micro_cell_size: cfg.micro_cell_size,
            mantle_keep_denom: cfg.mantle_keep_denominator,
        }
    }

    /// Samples geological information at the given world coordinates.
    ///
    /// Performs two Voronoi nearest-cell lookups:
    /// 1. Macro region at 2048-block cell scale.
    /// 2. Micro region at 512-block cell scale (constrained by macro).
    ///
    /// Mantle macro regions are filtered for rarity (only ~20% survive;
    /// the rest become Igneous).
    ///
    /// # Arguments
    /// * `wx` - World X coordinate (blocks).
    /// * `wz` - World Z coordinate (blocks).
    ///
    /// # Returns
    /// A [`GeologyInfo`] struct with macro and micro region data.
    pub fn sample(&self, wx: i32, wz: i32) -> GeologyInfo {
        let (macro_cx, macro_cz) = Self::nearest_cell(self.seed, wx, wz, self.macro_cell_size);

        let mut macro_region = MacroRegion::from_hash(hash_coord(self.seed, macro_cx, macro_cz));

        if macro_region == MacroRegion::Mantle {
            let mantle_hash = hash_coord(self.seed + 1, macro_cx, macro_cz);
            if mantle_hash % self.mantle_keep_denom != 0 {
                macro_region = MacroRegion::Igneous;
            }
        }

        let (micro_cx, micro_cz) = Self::nearest_cell(self.seed + 10, wx, wz, self.micro_cell_size);
        let micro_hash = hash_coord(self.seed + 10, micro_cx, micro_cz);
        let micro_region = Self::micro_from_hash(macro_region, micro_hash);

        flow_debug_log!(
            "GeologyLayer",
            "sample",
            "pos=({}, {}) macro={:?} micro={:?} (stone={})",
            wx,
            wz,
            macro_region,
            micro_region,
            micro_region.stone_name()
        );

        GeologyInfo {
            macro_region,
            micro_region,
        }
    }

    /// Samples geological information for a rectangular area (e.g., one chunk).
    ///
    /// Uses Rayon for parallel evaluation. The returned vector is row-major:
    /// index `dz * size + dx`.
    ///
    /// # Arguments
    /// * `cx` - Chunk X index (world start = `cx * size`).
    /// * `cz` - Chunk Z index (world start = `cz * size`).
    /// * `size` - Side length of the square area (typically 16).
    ///
    /// # Returns
    /// A `Vec<GeologyInfo>` with `size * size` elements.
    pub fn sample_area(&self, cx: i32, cz: i32, size: usize) -> Vec<GeologyInfo> {
        flow_debug_log!(
            "GeologyLayer",
            "sample_area",
            "Sampling area chunk=({}, {}) size={}",
            cx,
            cz,
            size
        );

        let start_x = cx * size as i32;
        let start_z = cz * size as i32;
        let total = size * size;

        let results: Vec<GeologyInfo> = (0..total)
            .into_par_iter()
            .map(|i| {
                let dx = (i % size) as i32;
                let dz = (i / size) as i32;
                self.sample(start_x + dx, start_z + dz)
            })
            .collect();

        flow_debug_log!(
            "GeologyLayer",
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

    /// Computes the jittered feature point for a Voronoi cell.
    ///
    /// The feature point is offset from the cell center by a hash-derived
    /// jitter in `[0, cell_size)` in both X and Z.
    ///
    /// # Arguments
    /// * `seed` - Seed for hash computation.
    /// * `cx` - Cell X index.
    /// * `cz` - Cell Z index.
    /// * `cell_size` - Size of the Voronoi cell in blocks.
    ///
    /// # Returns
    /// The `(x, z)` position of the jittered feature point in world coords.
    #[inline]
    fn feature_point(seed: u64, cx: i32, cz: i32, cell_size: i32) -> (f64, f64) {
        let h = hash_coord(seed, cx, cz);
        // Use the lower 16 bits for X jitter and the next 16 bits for Z jitter
        let jx = (h & 0xFFFF) as f64 / 65535.0;
        let jz = ((h >> 16) & 0xFFFF) as f64 / 65535.0;
        let cell_f = cell_size as f64;
        (
            cx as f64 * cell_f + jx * cell_f,
            cz as f64 * cell_f + jz * cell_f,
        )
    }

    /// Finds the Voronoi cell whose feature point is nearest to `(wx, wz)`.
    ///
    /// Searches a `(2 * SEARCH_RADIUS + 1)²` neighborhood around the
    /// cell containing the target point.
    ///
    /// # Arguments
    /// * `seed` - Seed for hash/feature computation.
    /// * `wx` - World X coordinate.
    /// * `wz` - World Z coordinate.
    /// * `cell_size` - Voronoi cell size in blocks.
    ///
    /// # Returns
    /// The `(cell_x, cell_z)` of the nearest feature point's cell.
    fn nearest_cell(seed: u64, wx: i32, wz: i32, cell_size: i32) -> (i32, i32) {
        let origin_cx = wx.div_euclid(cell_size);
        let origin_cz = wz.div_euclid(cell_size);
        let wx_f = wx as f64;
        let wz_f = wz as f64;

        let mut best_dist_sq = f64::MAX;
        let mut best_cx = origin_cx;
        let mut best_cz = origin_cz;

        for dcx in -SEARCH_RADIUS..=SEARCH_RADIUS {
            for dcz in -SEARCH_RADIUS..=SEARCH_RADIUS {
                let cx = origin_cx + dcx;
                let cz = origin_cz + dcz;
                let (fx, fz) = Self::feature_point(seed, cx, cz, cell_size);
                let dx = wx_f - fx;
                let dz = wz_f - fz;
                let dist_sq = dx * dx + dz * dz;
                if dist_sq < best_dist_sq {
                    best_dist_sq = dist_sq;
                    best_cx = cx;
                    best_cz = cz;
                }
            }
        }

        (best_cx, best_cz)
    }

    /// Maps a hash value to a specific micro region within the given macro region.
    ///
    /// Each macro region has exactly 3 possible micro regions, selected
    /// by `hash % 3`.
    ///
    /// # Arguments
    /// * `macro_region` - The parent macro region.
    /// * `hash` - Hash value for deterministic selection.
    ///
    /// # Returns
    /// The selected [`MicroRegion`].
    #[inline]
    fn micro_from_hash(macro_region: MacroRegion, hash: u64) -> MicroRegion {
        match macro_region {
            MacroRegion::Metamorphic => match hash % 3 {
                0 => MicroRegion::Gneiss,
                1 => MicroRegion::Schist,
                _ => MicroRegion::Marble,
            },
            MacroRegion::Sedimentary => match hash % 3 {
                0 => MicroRegion::Sandstone,
                1 => MicroRegion::Limestone,
                _ => MicroRegion::Shale,
            },
            MacroRegion::Igneous => match hash % 3 {
                0 => MicroRegion::Basalt,
                1 => MicroRegion::Granite,
                _ => MicroRegion::Andesite,
            },
            MacroRegion::Mantle => match hash % 3 {
                0 => MicroRegion::Lherzolite,
                1 => MicroRegion::Kimberlite,
                _ => MicroRegion::Ophiolite,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_deterministic() {
        let a = hash_coord(42, 100, 200);
        let b = hash_coord(42, 100, 200);
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_different_coords() {
        let a = hash_coord(42, 100, 200);
        let b = hash_coord(42, 200, 100);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_different_seeds() {
        let a = hash_coord(42, 100, 200);
        let b = hash_coord(99, 100, 200);
        assert_ne!(a, b);
    }

    #[test]
    fn test_feature_point_within_cell() {
        let (fx, fz) = GeologyLayer::feature_point(42, 5, 10, 2048);
        // Feature point should be within the cell [5*2048, 6*2048) x [10*2048, 11*2048)
        assert!(fx >= 5.0 * 2048.0);
        assert!(fx < 6.0 * 2048.0);
        assert!(fz >= 10.0 * 2048.0);
        assert!(fz < 11.0 * 2048.0);
    }

    #[test]
    fn test_sample_deterministic() {
        let layer = GeologyLayer::new(42);
        let a = layer.sample(1000, 2000);
        let b = layer.sample(1000, 2000);
        assert_eq!(a.macro_region, b.macro_region);
        assert_eq!(a.micro_region, b.micro_region);
    }

    #[test]
    fn test_sample_area_size() {
        let layer = GeologyLayer::new(42);
        let results = layer.sample_area(0, 0, 16);
        assert_eq!(results.len(), 256);
    }

    #[test]
    fn test_micro_consistent_with_macro() {
        let layer = GeologyLayer::new(42);
        // Check many positions
        for x in 0..100 {
            for z in 0..100 {
                let info = layer.sample(x * 200, z * 200);
                assert_eq!(
                    info.micro_region.macro_region(),
                    info.macro_region,
                    "Micro {:?} does not belong to macro {:?} at ({}, {})",
                    info.micro_region,
                    info.macro_region,
                    x * 200,
                    z * 200
                );
            }
        }
    }

    #[test]
    fn test_mantle_rarity() {
        let layer = GeologyLayer::new(42);
        let mut mantle_count = 0;
        let total = 50000;
        for i in 0..total {
            let info = layer.sample(i * 100, (i * 137) % 100000);
            if info.macro_region == MacroRegion::Mantle {
                mantle_count += 1;
            }
        }
        // Mantle should be rare (< 10% of positions)
        let ratio = mantle_count as f64 / total as f64;
        assert!(
            ratio < 0.10,
            "Mantle ratio too high: {:.2}% (expected < 10%)",
            ratio * 100.0
        );
    }

    #[test]
    fn test_all_macro_regions_present() {
        let layer = GeologyLayer::new(42);
        let mut found = std::collections::HashSet::new();
        // Scan a large area
        for x in 0..200 {
            for z in 0..200 {
                let info = layer.sample(x * 100, z * 100);
                found.insert(info.macro_region);
            }
        }
        assert!(
            found.contains(&MacroRegion::Metamorphic),
            "Metamorphic not found"
        );
        assert!(
            found.contains(&MacroRegion::Sedimentary),
            "Sedimentary not found"
        );
        assert!(
            found.contains(&MacroRegion::Igneous),
            "Igneous not found"
        );
    }

    #[test]
    fn test_micro_region_names() {
        assert_eq!(MicroRegion::Basalt.stone_name(), "rocks/basalt");
        assert_eq!(MicroRegion::Granite.grass_name(), "grass/grass_granite");
        assert_eq!(MicroRegion::Limestone.dirt_name(), "dirt/dirt_limestone");
        assert_eq!(
            MicroRegion::Kimberlite.clay_name(),
            "clay/clay_kimberlite"
        );
        assert_eq!(
            MicroRegion::Andesite.claystone_name(),
            "claystone/claystone_andesite"
        );
    }

    #[test]
    fn test_micro_macro_parent() {
        assert_eq!(MicroRegion::Gneiss.macro_region(), MacroRegion::Metamorphic);
        assert_eq!(MicroRegion::Shale.macro_region(), MacroRegion::Sedimentary);
        assert_eq!(MicroRegion::Basalt.macro_region(), MacroRegion::Igneous);
        assert_eq!(MicroRegion::Ophiolite.macro_region(), MacroRegion::Mantle);
    }

    #[test]
    fn test_voronoi_cell_boundaries() {
        let layer = GeologyLayer::new(42);
        // At exact cell boundaries, we should still get a valid assignment
        for z in 0..20 {
            let wx = layer.macro_cell_size;
            let info = layer.sample(wx, z * 1000);
            // Just ensure it doesn't panic and returns something valid
            let _ = info.macro_region.name();
        }
    }
}
