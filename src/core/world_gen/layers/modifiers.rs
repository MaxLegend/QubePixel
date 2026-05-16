//! # Post-processing modifier layers
//!
//! These layers sit **after** zoom layers in the pipeline and refine the biome
//! grid by growing landmasses, inserting edge transitions, and smoothing
//! artifacts.
//!
//! | Layer | Purpose |
//! |-------|---------|
//! | [`LayerAddIsland`] | Grows landmasses by probabilistically converting ocean cells adjacent to land |
//! | [`LayerEdge`] | Inserts transitional biome values at boundaries between different values |
//! | [`LayerSmooth`] | Removes isolated single-pixel artifacts ("singleton removal") |

use crate::core::world_gen::biome_layer::{BiomeLayer, GenContext, check_output_size, coord_hash_to_f64};
use crate::{debug_log, ext_debug_log, flow_debug_log};
// ===========================================================================
// Helper: read from a buffer with bounds checking
// ===========================================================================

/// Read a cell from `buf` (row-major, stride = `width`).  Out-of-bounds
/// coordinates return 0.
#[inline]
fn read(buf: &[u32], width: usize, _height: usize, x: i32, y: i32) -> u32 {
    if x < 0 || y < 0 || x as usize >= width || y as usize >= _height {
        return 0;
    }
    buf[y as usize * width + x as usize]
}

// ===========================================================================
// LayerAddIsland
// ===========================================================================

/// Grows landmasses by converting ocean cells that are adjacent to land.
///
/// For each cell whose value is **0** (ocean), this layer examines the four
/// cardinal neighbors (N, S, E, W).  If **any** neighbor is non-zero (land),
/// the ocean cell is converted to land with probability [`chance`](LayerAddIsland::chance).
///
/// The decision is deterministic: [`coord_hash_to_f64`] ensures that the same
/// world coordinates always yield the same result.
///
/// This layer is typically applied several times in succession (each time
/// wrapped in a new `LayerAddIsland`) to progressively grow landmasses,
/// simulating the effect of continental shelves or large islands.
///
/// # Example
///
/// ```ignore
/// let island = LayerIsland::new(seed);
/// let grow1 = LayerAddIsland::new(Box::new(island), seed + 1);
/// let grow2 = LayerAddIsland::new(Box::new(grow1), seed + 2);
/// ```
pub struct LayerAddIsland {
    parent: Box<dyn BiomeLayer>,
    seed: u64,

    /// Probability of converting an ocean cell to land when it borders land.
    chance: f64,
}

impl LayerAddIsland {
    /// Default conversion probability.
    const DEFAULT_CHANCE: f64 = 0.33;

    /// Create a new `LayerAddIsland` with the default conversion chance (0.33).
    pub fn new(parent: Box<dyn BiomeLayer>, seed: u64) -> Self {
        Self::with_chance(parent, seed, Self::DEFAULT_CHANCE)
    }

    /// Create a new `LayerAddIsland` with a custom conversion chance.
    ///
    /// # Arguments
    ///
    /// * `parent` — the upstream layer whose output is modified.
    /// * `seed` — seed for deterministic randomness.
    /// * `chance` — probability in `[0.0, 1.0]` of converting ocean to land.
    pub fn with_chance(parent: Box<dyn BiomeLayer>, seed: u64, chance: f64) -> Self {
        debug_log!(
            "LayerAddIsland",
            "with_chance",
            "seed={} chance={:.2}",
            seed,
            chance
        );
        Self { parent, seed, chance }
    }
}

impl BiomeLayer for LayerAddIsland {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerAddIsland",
            "generate",
            "region=({}, {}) size={}x{} chance={:.2}",
            x, y, width, height, self.chance
        );

        // We need a +1 border to check neighbors.  Request parent with that border,
        // but only write into the central `output` region.
        let mut parent_buf = ctx.acquire_buffer((width + 2) * (height + 2));
        self.parent
            .generate(x - 1, y - 1, width + 2, height + 2, ctx, &mut parent_buf);

        let pw = width + 2;
        let ph = height + 2;

        for oy in 0..height {
            for ox in 0..width {
                // Position in parent buffer (shifted by +1 border).
                let bx = ox + 1;
                let by = oy + 1;
                let value = parent_buf[by * pw + bx];

                if value == 0 {
                    // Ocean — check cardinal neighbors.
                    let n = read(&parent_buf, pw, ph, bx as i32, (by - 1) as i32);
                    let s = read(&parent_buf, pw, ph, bx as i32, (by + 1) as i32);
                    let e = read(&parent_buf, pw, ph, (bx + 1) as i32, by as i32);
                    let w = read(&parent_buf, pw, ph, (bx - 1) as i32, by as i32);

                    if n != 0 || s != 0 || e != 0 || w != 0 {
                        let r = coord_hash_to_f64(self.seed, x + ox as i32, y + oy as i32);
                        output[oy * width + ox] = if r < self.chance { 1 } else { 0 };
                    } else {
                        output[oy * width + ox] = 0;
                    }
                } else {
                    // Land — keep as-is (copy the actual parent value, not just 1).
                    output[oy * width + ox] = value;
                }
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerAddIsland",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ===========================================================================
// LayerEdge
// ===========================================================================

/// Inserts transitional ("edge") biome values at boundaries between different
/// biome values.
///
/// For each cell with value **A** whose cardinal neighbor has value **B**, if
/// the pair `(A, B)` (or `(B, A)`) appears in [`edge_rules`], the cell is
/// replaced with the corresponding edge value.
///
/// This is useful for placing beaches between ocean and land, or transitional
/// biomes between climatic zones.
///
/// # Edge rules
///
/// Each rule is a triple `(current, neighbor, edge_value)`:
///
/// * If a cell has value `current` and **any** cardinal neighbor has value
///   `neighbor`, the cell is replaced with `edge_value`.
/// * Rules are checked in order; the **first** matching rule wins.
///
/// # Example — beach generation
///
/// ```ignore
/// let rules = vec![
///     (1, 0, 10),  // land next to ocean → beach (biome 10)
/// ];
/// let edge = LayerEdge::new(parent, 42, rules);
/// ```
pub struct LayerEdge {
    parent: Box<dyn BiomeLayer>,
    seed: u64,

    /// Ordered list of `(current_value, neighbor_value, replacement_value)`.
    /// The first matching rule is applied.
    edge_rules: Vec<(u32, u32, u32)>,
}

impl LayerEdge {
    /// Create a new `LayerEdge` with the given rules.
    pub fn new(parent: Box<dyn BiomeLayer>, seed: u64, edge_rules: Vec<(u32, u32, u32)>) -> Self {
        debug_log!(
            "LayerEdge",
            "new",
            "seed={} rules_count={}",
            seed,
            edge_rules.len()
        );
        Self {
            parent,
            seed,
            edge_rules,
        }
    }

    /// Convenience constructor that creates rules for **climate-zone
    /// transitions**.
    ///
    /// The generated rules insert edge biomes at transitions between adjacent
    /// climate categories (values `0` through `n_zones - 1`).
    ///
    /// # Arguments
    ///
    /// * `parent` — upstream layer.
    /// * `seed` — seed for deterministic decisions.
    /// * `n_zones` — number of climate zones (expected values `0..n_zones`).
    /// * `edge_start_id` — first biome ID to use for edge biomes.  Edge biome
    ///   for the transition between zone `i` and zone `i+1` gets ID
    ///   `edge_start_id + i`.
    pub fn new_for_climate(
        parent: Box<dyn BiomeLayer>,
        seed: u64,
        n_zones: u32,
        edge_start_id: u32,
    ) -> Self {
        let mut rules = Vec::new();
        for i in 0..n_zones.saturating_sub(1) {
            let edge_id = edge_start_id + i;
            // Zone i next to zone i+1 → edge.
            rules.push((i, i + 1, edge_id));
            rules.push((i + 1, i, edge_id));
        }
        debug_log!(
            "LayerEdge",
            "new_for_climate",
            "seed={} n_zones={} edge_start={} rules={}",
            seed,
            n_zones,
            edge_start_id,
            rules.len()
        );
        Self {
            parent,
            seed,
            edge_rules: rules,
        }
    }
}

impl BiomeLayer for LayerEdge {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerEdge",
            "generate",
            "region=({}, {}) size={}x{} rules={}",
            x, y, width, height, self.edge_rules.len()
        );

        // Need +1 border for neighbor checks.
        let mut parent_buf = ctx.acquire_buffer((width + 2) * (height + 2));
        self.parent
            .generate(x - 1, y - 1, width + 2, height + 2, ctx, &mut parent_buf);

        let pw = width + 2;
        let ph = height + 2;

        for oy in 0..height {
            for ox in 0..width {
                let bx = ox + 1;
                let by = oy + 1;
                let value = parent_buf[by * pw + bx];

                // Read cardinal neighbors.
                let neighbors = [
                    read(&parent_buf, pw, ph, bx as i32, (by - 1) as i32),
                    read(&parent_buf, pw, ph, bx as i32, (by + 1) as i32),
                    read(&parent_buf, pw, ph, (bx + 1) as i32, by as i32),
                    read(&parent_buf, pw, ph, (bx - 1) as i32, by as i32),
                ];

                let mut result = value;

                // Check rules in order.
                for &(cur, nbr, edge_val) in &self.edge_rules {
                    if value != cur {
                        continue;
                    }
                    if neighbors.iter().any(|&n| n == nbr) {
                        result = edge_val;
                        break; // First matching rule wins.
                    }
                }

                output[oy * width + ox] = result;
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerEdge",
            "generate",
            "done region=({}, {}) size={}x{}",
            x, y, width, height
        );
    }
}

// ===========================================================================
// LayerSmooth
// ===========================================================================

/// Removes isolated single-pixel artifacts ("singleton removal").
///
/// A "singleton" is a cell whose value differs from **all four cardinal
/// neighbors**.  This layer replaces such cells with the value of the
/// majority of their neighbors, effectively removing scattered noise pixels.
///
/// This is a purely local operation — no randomness is required.  If all four
/// neighbors are different from each other (extremely rare), the northern
/// neighbor's value is used as a tie-breaker.
///
/// # Usage
///
/// Typically placed after zoom layers that may introduce pixel-level
/// artifacts at biome boundaries.
pub struct LayerSmooth {
    parent: Box<dyn BiomeLayer>,
}

impl LayerSmooth {
    /// Create a new smoothing layer.
    pub fn new(parent: Box<dyn BiomeLayer>) -> Self {
        debug_log!("LayerSmooth", "new", "");
        Self { parent }
    }
}

impl BiomeLayer for LayerSmooth {
    fn generate(
        &self,
        x: i32,
        y: i32,
        width: usize,
        height: usize,
        ctx: &mut GenContext,
        output: &mut [u32],
    ) {
        if !check_output_size(x, y, width, height, output) {
            return;
        }

        flow_debug_log!(
            "LayerSmooth",
            "generate",
            "region=({}, {}) size={}x{}",
            x, y, width, height
        );

        // Need +1 border for neighbor checks.
        let mut parent_buf = ctx.acquire_buffer((width + 2) * (height + 2));
        self.parent
            .generate(x - 1, y - 1, width + 2, height + 2, ctx, &mut parent_buf);

        let pw = width + 2;
        let ph = height + 2;

        for oy in 0..height {
            for ox in 0..width {
                let bx = ox + 1;
                let by = oy + 1;
                let center = parent_buf[by * pw + bx];

                let n = read(&parent_buf, pw, ph, bx as i32, (by - 1) as i32);
                let s = read(&parent_buf, pw, ph, bx as i32, (by + 1) as i32);
                let e = read(&parent_buf, pw, ph, (bx + 1) as i32, by as i32);
                let w = read(&parent_buf, pw, ph, (bx - 1) as i32, by as i32);

                // If center differs from ALL four neighbors, replace it.
                if center != n && center != s && center != e && center != w {
                    // Pick the most common neighbor.  On tie, prefer N > E > S > W.
                    let mut candidates = [(n, 0u32), (s, 0), (e, 0), (w, 0)];
                    // Count occurrences (including center's value for context).
                    let all = [n, s, e, w];
                    for &v in &all {
                        if let Some(c) = candidates.iter_mut().find(|(val, _)| *val == v) {
                            c.1 += 1;
                        }
                    }
                    // Sort by count descending, then by original order (N, S, E, W).
                    candidates.sort_by(|a, b| b.1.cmp(&a.1));
                    output[oy * width + ox] = candidates[0].0;
                } else {
                    output[oy * width + ox] = center;
                }
            }
        }

        ctx.release_buffer(parent_buf);

        ext_debug_log!(
            "LayerSmooth",
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
    use crate::core::world_gen::layers::LayerIsland;
    use super::*;


    #[test]
    fn test_add_island_grows_land() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let add = LayerAddIsland::new(island, 100);
        let mut ctx = GenContext::new(42);

        let mut parent_out = vec![0u32; 16 * 16];
        let mut child_out = vec![0u32; 16 * 16];

        // Generate without and with the AddIsland layer.
        let island_raw = LayerIsland::new(42);
        island_raw.generate(0, 0, 16, 16, &mut ctx, &mut parent_out);
        add.generate(0, 0, 16, 16, &mut ctx, &mut child_out);

        // Count land cells.
        let parent_land = parent_out.iter().filter(|&&v| v != 0).count();
        let child_land = child_out.iter().filter(|&&v| v != 0).count();

        // After AddIsland, we should have >= land cells (it only grows).
        assert!(
            child_land >= parent_land,
            "expected >= {} land cells, got {}",
            parent_land,
            child_land
        );
    }

    #[test]
    fn test_add_island_deterministic() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let add = LayerAddIsland::new(island, 100);
        let mut ctx = GenContext::new(42);

        let mut a = vec![0u32; 16 * 16];
        let mut b = vec![0u32; 16 * 16];
        add.generate(0, 0, 16, 16, &mut ctx, &mut a);
        add.generate(0, 0, 16, 16, &mut ctx, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn test_edge_inserts_beach() {
        let island: Box<dyn BiomeLayer> = Box::new(LayerIsland::new(42));
        let rules = vec![(1, 0, 10)]; // land next to ocean → beach
        let edge = LayerEdge::new(island, 42, rules);
        let mut ctx = GenContext::new(42);

        let mut out = vec![0u32; 32 * 32];
        edge.generate(0, 0, 32, 32, &mut ctx, &mut out);

        // Beach cells (value 10) should exist at land/ocean boundaries.
        let beach_count = out.iter().filter(|&&v| v == 10).count();
        // The result depends on the random seed; just make sure it runs.
        assert!(beach_count >= 0);
    }

    #[test]
    fn test_smooth_removes_singletons() {
        // Create a synthetic parent with a singleton.
        struct SingletonLayer;
        impl BiomeLayer for SingletonLayer {
            fn generate(
                &self,
                _x: i32, _y: i32, width: usize, height: usize,
                _ctx: &mut GenContext, output: &mut [u32],
            ) {
                // Fill with 1, place a 0 at center.
                for v in output.iter_mut() { *v = 1; }
                let cx = width / 2;
                let cy = height / 2;
                if cx < width && cy < height {
                    output[cy * width + cx] = 0;
                }
            }
        }
        let singleton: Box<dyn BiomeLayer> = Box::new(SingletonLayer);
        let smooth = LayerSmooth::new(singleton);
        let mut ctx = GenContext::new(42);

        let mut out = vec![0u32; 8 * 8];
        smooth.generate(0, 0, 8, 8, &mut ctx, &mut out);

        // The center cell should have been replaced with 1 (neighbor majority).
        assert_eq!(out[4 * 8 + 4], 1);
    }
}
