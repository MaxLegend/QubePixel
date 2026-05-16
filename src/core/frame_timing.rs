// =============================================================================
// QubePixel — Frame timing globals
//
// Written by main.rs each frame, read by game_screen profiler feed.
// All values in microseconds (u64 → i64 cast safe up to 18 PB).
// =============================================================================

use std::sync::atomic::{AtomicU64, Ordering};

/// Time blocked in get_current_texture() — waits for GPU to finish previous frame.
pub static GET_TEXTURE_US:    AtomicU64 = AtomicU64::new(0);
/// Time for update() + post_process() on the screen manager.
pub static UPDATE_US:         AtomicU64 = AtomicU64::new(0);
/// Time for egui_ctx.run() (build_ui) + tessellation + texture uploads.
pub static EGUI_BUILD_US:     AtomicU64 = AtomicU64::new(0);
/// Time for egui_mgr.render_draw() (encoding egui draw calls into the encoder).
pub static EGUI_RENDER_US:    AtomicU64 = AtomicU64::new(0);
/// Time for queue.submit() + surface.present().
pub static SUBMIT_PRESENT_US: AtomicU64 = AtomicU64::new(0);

#[inline] pub fn set_get_texture(us: u128)    { GET_TEXTURE_US   .store(us as u64, Ordering::Relaxed); }
#[inline] pub fn set_update(us: u128)          { UPDATE_US        .store(us as u64, Ordering::Relaxed); }
#[inline] pub fn set_egui_build(us: u128)      { EGUI_BUILD_US    .store(us as u64, Ordering::Relaxed); }
#[inline] pub fn set_egui_render(us: u128)     { EGUI_RENDER_US   .store(us as u64, Ordering::Relaxed); }
#[inline] pub fn set_submit_present(us: u128)  { SUBMIT_PRESENT_US.store(us as u64, Ordering::Relaxed); }

#[inline] pub fn get_texture_us()    -> u128 { GET_TEXTURE_US   .load(Ordering::Relaxed) as u128 }
#[inline] pub fn update_us()         -> u128 { UPDATE_US        .load(Ordering::Relaxed) as u128 }
#[inline] pub fn egui_build_us()     -> u128 { EGUI_BUILD_US    .load(Ordering::Relaxed) as u128 }
#[inline] pub fn egui_render_us()    -> u128 { EGUI_RENDER_US   .load(Ordering::Relaxed) as u128 }
#[inline] pub fn submit_present_us() -> u128 { SUBMIT_PRESENT_US.load(Ordering::Relaxed) as u128 }
