// =============================================================================
// QubePixel — Lighting Module (simplified)
// =============================================================================
//
// Все системы освещения (Radiance Cascades, Voxel GPU Lighting, Volumetric Lights)
// архивированы. Модуль оставлен для обратной совместимости — реэкспортирует
// DayNightCycle и LightingConfig из lighting_legacy.

// Re-export из lighting_legacy для обратной совместимости
pub use crate::core::lighting_legacy::*;
