// =============================================================================
// QubePixel — AppConfig (global atomic settings shared across threads)
// =============================================================================

use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::OnceLock;
use serde::Deserialize;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Chunk dimensions — loaded once from assets/chunk_config.json
// ---------------------------------------------------------------------------

/// Per-axis chunk size in blocks. Default: 16×256×16 (columnar).
/// Dimensions should be multiples of 4 so LOD steps (×2, ×4) divide evenly.
#[derive(Debug, Deserialize, Clone)]
pub struct ChunkDimensions {
    pub size_x: usize,
    pub size_y: usize,
    pub size_z: usize,
}

impl Default for ChunkDimensions {
    fn default() -> Self {
        Self { size_x: 16, size_y: 256, size_z: 16 }
    }
}

static CHUNK_DIMS: OnceLock<ChunkDimensions> = OnceLock::new();

/// Returns the global chunk dimensions, loading from JSON on first call.
pub fn chunk_dims() -> &'static ChunkDimensions {
    CHUNK_DIMS.get_or_init(|| {
        let dims: ChunkDimensions = std::fs::read_to_string("assets/chunk_config.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        debug_log!(
            "Config", "chunk_dims",
            "Chunk dimensions: {}x{}x{}", dims.size_x, dims.size_y, dims.size_z
        );
        dims
    })
}

#[inline(always)] pub fn chunk_size_x() -> usize { chunk_dims().size_x }
#[inline(always)] pub fn chunk_size_y() -> usize { chunk_dims().size_y }
#[inline(always)] pub fn chunk_size_z() -> usize { chunk_dims().size_z }

/// Render distance in chunks (horizontal). Range: 1..=128.
pub static RENDER_DISTANCE: AtomicI32 = AtomicI32::new(3);

/// LOD distance multiplier × 100 (e.g. 100 = 1.0×, 150 = 1.5×).
/// Controls how far LOD0/LOD1 zones extend before switching to coarser LOD.
///   LOD0 (full detail):  Chebyshev dist ≤ LOD_NEAR_BASE  × multiplier
///   LOD1 (2× coarser):   Chebyshev dist ≤ LOD_FAR_BASE   × multiplier
///   LOD2 (4× coarser):   everything beyond LOD1 within render distance
pub static LOD_MULTIPLIER: AtomicU32 = AtomicU32::new(100);

/// Base Chebyshev-distance thresholds (in chunks) before multiplier is applied.
pub const LOD_NEAR_BASE: f32 = 3.0;
pub const LOD_FAR_BASE: f32  = 6.0;
/// Vertical chunk loading range — chunks below camera's chunk Y.
/// Default 0: with 256-height columnar chunks the whole world fits in one vertical layer.
pub static VERTICAL_BELOW: AtomicI32 = AtomicI32::new(0);
/// Vertical chunk loading range — chunks above camera's chunk Y.
pub static VERTICAL_ABOVE: AtomicI32 = AtomicI32::new(0);
// ---------------------------------------------------------------------------
// Render distance
// ---------------------------------------------------------------------------
pub fn render_distance() -> i32 {
    RENDER_DISTANCE.load(Ordering::Relaxed).clamp(1, 128)
}

pub fn set_render_distance(v: i32) {
    let clamped = v.clamp(1, 128);
    RENDER_DISTANCE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_render_distance", "Render distance set to {}", clamped);
}

// ---------------------------------------------------------------------------
// LOD multiplier
// ---------------------------------------------------------------------------

/// Returns LOD multiplier as float (1.0 = default).
pub fn lod_multiplier() -> f32 {
    LOD_MULTIPLIER.load(Ordering::Relaxed) as f32 / 100.0
}

/// Set LOD multiplier. Value is clamped to 0.25..=4.0.
pub fn set_lod_multiplier(v: f32) {
    let clamped = (v * 100.0).clamp(25.0, 400.0) as u32;
    LOD_MULTIPLIER.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_lod_multiplier", "LOD multiplier set to {:.2}", clamped as f32 / 100.0);
}
// ---------------------------------------------------------------------------
// Vertical chunk range
// ---------------------------------------------------------------------------

pub fn vertical_below() -> i32 {
    VERTICAL_BELOW.load(Ordering::Relaxed).clamp(0, 16)
}

pub fn set_vertical_below(v: i32) {
    let clamped = v.clamp(0, 16);
    VERTICAL_BELOW.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_vertical_below", "Vertical below set to {}", clamped);
}

pub fn vertical_above() -> i32 {
    VERTICAL_ABOVE.load(Ordering::Relaxed).clamp(0, 16)
}

pub fn set_vertical_above(v: i32) {
    let clamped = v.clamp(0, 16);
    VERTICAL_ABOVE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_vertical_above", "Vertical above set to {}", clamped);
}
/// Compute the LOD level for a chunk at the given chunk coordinates.
/// Returns 0 (full), 1 (2× coarser), or 2 (4× coarser).
pub fn compute_lod_level(cam_cx: i32, cam_cz: i32, cx: i32, cz: i32) -> u8 {
    let dist = (cx - cam_cx).abs().max((cz - cam_cz).abs());
    let mult = lod_multiplier();
    let lod0_range = (LOD_NEAR_BASE * mult).ceil() as i32;
    let lod1_range = (LOD_FAR_BASE * mult).ceil() as i32;

    if dist <= lod0_range { 0 }
    else if dist <= lod1_range { 1 }
    else { 2 }
}
