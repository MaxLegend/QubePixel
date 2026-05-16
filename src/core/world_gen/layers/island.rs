//! # Root island layers
//!
//! These layers sit at the **bottom** of the generation pipeline — they have no
//! parent and generate their values purely from the world seed and coordinates.
//!
//! | Layer | Purpose |
//! |-------|---------|
//! | [`LayerIsland`] | Generates a coarse binary land/ocean map |
//! | [`LayerRiverInit`] | Generates random integer values that seed river networks |

use crate::core::world_gen::biome_layer::{BiomeLayer, GenContext, check_output_size, coord_hash_to_f64};
use crate::{debug_log, ext_debug_log, flow_debug_log};
// ---------------------------------------------------------------------------
// LayerIsland
// ---------------------------------------------------------------------------

/// Root layer that produces a coarse binary map: **0** = ocean, **1** = land.
///
/// Each cell is assigned land with an internal probability of approximately
/// **55 %**.  The decision is fully deterministic, driven by
/// [`coord_hash_to_f64`].
///
/// This layer has **no parent** — it is typically used as the first element in
/// a layer stack, followed by zoom and modifier layers that refine the crude
/// landmass shapes.
///
/// # Example usage
///
/// ```ignore
/// let mut ctx = GenContext::new(12345);
/// let island = LayerIsland::new(12345);
/// let mut out = vec![0u32; 64 * 64];
/// island.generate(0, 0, 64, 64, &mut ctx, &mut out);
/// ```
pub struct LayerIsland {
    seed: u64,
}

impl LayerIsland {
    /// Internal land probability — roughly 55 % of cells will be land.
    const LAND_CHANCE: f64 = 0.55;

    /// Create a new island layer with the given seed.
    pub fn new(seed: u64) -> Self {
        debug_log!("LayerIsland", "new", "seed={}", seed);
        Self { seed }
    }
}

impl BiomeLayer for LayerIsland {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        _ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerIsland",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        for dy in 0..height {
            for dx in 0..width {
                let wx = x + dx as i32;
                let wy = y + dy as i32;
                let chance = coord_hash_to_f64(self.seed, wx, wy);
                output[dy * width + dx] = if chance < Self::LAND_CHANCE { 1 } else { 0 };
            }
        }

        ext_debug_log!(
            "LayerIsland",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ---------------------------------------------------------------------------
// LayerRiverInit
// ---------------------------------------------------------------------------

/// Root layer that generates random integer values for river-network seeding.
///
/// Each cell receives a value in the inclusive range **[1, 10]**, chosen
/// deterministically via [`coord_hash_to_f64`].  Downstream river-generation
/// layers interpret these values as flow directions, drainage basins, or other
/// river-related parameters.
///
/// Like [`LayerIsland`], this layer has **no parent** and is typically placed
/// at the base of a secondary pipeline dedicated to hydrology.
pub struct LayerRiverInit {
    seed: u64,
}

impl LayerRiverInit {
    /// Minimum value produced by this layer.
    const MIN_VALUE: u32 = 1;
    /// Maximum value produced by this layer.
    const MAX_VALUE: u32 = 10;
    /// Exclusive upper bound used in `f64 → u32` conversion.
    const RANGE: f64 = (Self::MAX_VALUE - Self::MIN_VALUE + 1) as f64;

    /// Create a new river-init layer with the given seed.
    pub fn new(seed: u64) -> Self {
        debug_log!("LayerRiverInit", "new", "seed={}", seed);
        Self { seed }
    }
}

impl BiomeLayer for LayerRiverInit {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        _ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerRiverInit",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        for dy in 0..height {
            for dx in 0..width {
                let wx = x + dx as i32;
                let wy = y + dy as i32;
                let chance = coord_hash_to_f64(self.seed, wx, wy);
                let value = Self::MIN_VALUE + (chance * Self::RANGE) as u32;
                // Clamp to handle floating-point edge case where chance == 1.0.
                let value = value.min(Self::MAX_VALUE);
                output[dy * width + dx] = value;
            }
        }

        ext_debug_log!(
            "LayerRiverInit",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::world_gen::biome_layer::GenContext;

    #[test]
    fn test_island_deterministic() {
        let island = LayerIsland::new(42);
        let mut ctx = GenContext::new(42);
        let mut a = vec![0u32; 16];
        let mut b = vec![0u32; 16];
        island.generate(0, 0, 4, 4, &mut ctx, &mut a);
        island.generate(0, 0, 4, 4, &mut ctx, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_island_binary() {
        let island = LayerIsland::new(42);
        let mut ctx = GenContext::new(42);
        let mut out = vec![0u32; 100];
        island.generate(0, 0, 10, 10, &mut ctx, &mut out);
        for &v in &out {
            assert!(v == 0 || v == 1, "expected 0 or 1, got {}", v);
        }
    }

    #[test]
    fn test_river_init_range() {
        let river = LayerRiverInit::new(42);
        let mut ctx = GenContext::new(42);
        let mut out = vec![0u32; 100];
        river.generate(0, 0, 10, 10, &mut ctx, &mut out);
        for &v in &out {
            assert!(
                v >= 1 && v <= 10,
                "expected 1..=10, got {}",
                v
            );
        }
    }
}
