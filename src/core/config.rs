// =============================================================================
// QubePixel — AppConfig (global atomic settings shared across threads)
// =============================================================================

use std::sync::atomic::{AtomicI32, Ordering};
use crate::debug_log;

/// Render distance in chunks (horizontal). Range: 1..=8.
pub static RENDER_DISTANCE: AtomicI32 = AtomicI32::new(3);

/// Returns the current render distance, always clamped to 1..=8.
pub fn render_distance() -> i32 {
    RENDER_DISTANCE.load(Ordering::Relaxed).clamp(1, 128)
}

/// Sets the render distance. Value is clamped to 1..=8.
pub fn set_render_distance(v: i32) {
    let clamped = v.clamp(1, 128);
    RENDER_DISTANCE.store(clamped, Ordering::Relaxed);
    debug_log!("Config", "set_render_distance", "Render distance set to {}", clamped);
}
