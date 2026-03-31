// =============================================================================
// Minecraft Clone — Global debug flag and logging macro
// =============================================================================

use std::sync::atomic::{AtomicBool, Ordering};

/// Global toggle for debug logging.
/// When `false`, all `debug_log!` calls are no-ops (zero cost).
pub(crate) static IS_DEBUG: AtomicBool = AtomicBool::new(true);
pub(crate) static IS_EXTENDED_DEBUG: AtomicBool = AtomicBool::new(false);

pub(crate) static IS_FLOW_DEBUG: AtomicBool = AtomicBool::new(false);
/// Returns the current value of `IS_DEBUG`.
#[allow(dead_code)]
pub(crate) fn is_debug() -> bool {
    IS_DEBUG.load(Ordering::Relaxed)
        || IS_EXTENDED_DEBUG.load(Ordering::Relaxed)
        || IS_FLOW_DEBUG.load(Ordering::Relaxed)
}

/// Sets the value of `IS_DEBUG`.
#[allow(dead_code)]
pub(crate) fn set_debug(value: bool) {
    IS_DEBUG.store(value, Ordering::Relaxed);
}
#[allow(dead_code)]
pub(crate) fn set_ext_debug(value: bool) {
    IS_EXTENDED_DEBUG.store(value, Ordering::Relaxed);
}
#[allow(dead_code)]
pub(crate) fn set_flow_debug(value: bool) {
    IS_FLOW_DEBUG.store(value, Ordering::Relaxed);
}
/// Debug logging macro.
///
/// Format: `[Class][Method] message`
/// All output is in English.
/// Wrapped in `IS_DEBUG` check — no-ops when debug is disabled.
#[macro_export]
macro_rules! debug_log {
    ($class:expr, $method:expr, $($arg:tt)*) => {
        if $crate::logging::IS_DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            log::info!("[{}][{}] {}", $class, $method, format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! ext_debug_log {
    ($class:expr, $method:expr, $($arg:tt)*) => {
        if $crate::logging::IS_EXTENDED_DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            log::info!("[{}][{}] {}", $class, $method, format!($($arg)*));
        }
    };
}

#[macro_export]
macro_rules! flow_debug_log {
    ($class:expr, $method:expr, $($arg:tt)*) => {
        if $crate::logging::IS_FLOW_DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            log::info!("[{}][{}] {}", $class, $method, format!($($arg)*));
        }
    };
}