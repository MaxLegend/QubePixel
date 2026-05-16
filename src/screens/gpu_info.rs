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

// ---------------------------------------------------------------------------
// RAM usage — Windows only, via psapi GetProcessMemoryInfo (no extra crates)
// ---------------------------------------------------------------------------

/// Returns working-set RAM used by this process in megabytes.
/// Returns 0.0 on non-Windows platforms.
pub fn get_ram_mb() -> f64 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb:                              u32,
            page_fault_count:               u32,
            peak_working_set_size:          usize,
            working_set_size:               usize,
            quota_peak_paged_pool_usage:    usize,
            quota_paged_pool_usage:         usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage:     usize,
            pagefile_usage:                 usize,
            peak_pagefile_usage:            usize,
        }

        #[link(name = "psapi")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut core::ffi::c_void;
            fn GetProcessMemoryInfo(
                process:        *mut core::ffi::c_void,
                ppsmemcounters: *mut ProcessMemoryCounters,
                cb:             u32,
            ) -> i32;
        }

        unsafe {
            let mut pmc: ProcessMemoryCounters = std::mem::zeroed();
            pmc.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            let handle = GetCurrentProcess();
            if GetProcessMemoryInfo(handle, &mut pmc, pmc.cb) != 0 {
                pmc.working_set_size as f64 / (1024.0 * 1024.0)
            } else {
                0.0
            }
        }
    }
    #[cfg(not(target_os = "windows"))]
    { 0.0 }
}
