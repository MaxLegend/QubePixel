// =============================================================================
// QubePixel — Global GPU info (set once at startup, read from any screen)
// =============================================================================

use std::sync::OnceLock;

static GPU_INFO: OnceLock<String> = OnceLock::new();

/// Stores the GPU adapter string (name + backend) once at startup.
/// Called from `main()` right after `Renderer::new()`.
pub fn init(info: String) {
    let _ = GPU_INFO.set(info);
}

/// Returns the stored GPU info string, or `"N/A"` if not yet initialised.
pub fn get() -> &'static str {
    GPU_INFO.get().map(|s| s.as_str()).unwrap_or("N/A")
}
