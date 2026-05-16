# LIGHTING_RESTORE_GUIDE.md — Патч-гайд для восстановления систем освещения

> Дата архивации: 2026-05-02  
> Архив: `A:\Programming\QubePixelAntigravity\archiveLighting\`

---

## Что было удалено

Две системы освещения:
1. **Radiance Cascades (RC)** — probe-based GI с multi-cascade sampling
2. **Voxel GPU Lighting** — 3D texture flood-fill + CSM + Clustered Forward

Также удалён:
3. **Volumetric Lights** — additive-blended halos и god-ray quads для emissive блоков

Вместо них оставлен **дефолтный статичный PBR рендеринг** с directional sun/moon light + ambient.

---

## Структура архива

```
archiveLighting/
├── src/
│   ├── core/
│   │   ├── radiance_cascades/          ← Полная директория RC (12 файлов)
│   │   │   ├── dispatch.rs
│   │   │   ├── merge.rs
│   │   │   ├── sampling.rs
│   │   │   ├── system.rs
│   │   │   ├── types.rs
│   │   │   ├── voxel_tex.rs
│   │   │   └── shaders/
│   │   │       ├── emission_inject.wgsl
│   │   │       ├── merge.wgsl
│   │   │       ├── outline.wgsl
│   │   │       ├── pbr_lighting.wgsl    ← Оригинальный PBR с RC GI
│   │   │       ├── ray_march.wgsl
│   │   │       └── volumetric_lights.wgsl
│   │   ├── lighting/                    ← Полная директория Voxel GPU Lighting (13 файлов)
│   │   │   ├── mod.rs
│   │   │   ├── voxel_grid.rs
│   │   │   ├── light_prop.rs
│   │   │   ├── csm.rs
│   │   │   ├── lightmap_lut.rs
│   │   │   ├── clustered.rs
│   │   │   ├── ssao.rs
│   │   │   ├── voxel_renderer.rs
│   │   │   └── shaders/
│   │   │       ├── cluster_cull.wgsl
│   │   │       ├── light_propagation.wgsl
│   │   │       ├── ssao.wgsl
│   │   │       ├── voxel_lighting.wgsl
│   │   │       └── voxel_shadow.wgsl
│   │   └── volumetric_lights.rs
│   └── gameobjects/
│       └── world.rs                     ← Старый gameobjects/world.rs (с voxel texture)
├── assets/
│   └── lighting_config.json
└── modified_originals/                  ← Оригиналы изменённых файлов
    ├── core.rs
    ├── lighting_mod.rs
    ├── lighting_legacy.rs
    ├── config.rs
    ├── game_3d_pipeline.rs
    ├── game_screen.rs
    ├── settings.rs
    ├── world_gen_world.rs
    ├── gameobjects_world.rs
    ├── sky_renderer.rs
    ├── profiler_overlay.rs
    └── debug_overlay.rs
```

---

## Пошаговая инструкция по восстановлению

### Шаг 1: Восстановить удалённые директории и файлы

```batch
:: Копировать RC систему
xcopy "A:\Programming\QubePixelAntigravity\archiveLighting\src\core\radiance_cascades" "src\core\radiance_cascades\" /E /I /Y

:: Копировать Voxel GPU Lighting систему
xcopy "A:\Programming\QubePixelAntigravity\archiveLighting\src\core\lighting" "src\core\lighting\" /E /I /Y

:: Копировать Volumetric Lights
copy "A:\Programming\QubePixelAntigravity\archiveLighting\src\core\volumetric_lights.rs" "src\core\volumetric_lights.rs" /Y

:: Копировать gameobjects/world.rs (если нужен)
copy "A:\Programming\QubePixelAntigravity\archiveLighting\src\core\gameobjects\world.rs" "src\core\gameobjects\world.rs" /Y
```

### Шаг 2: Восстановить `src/core.rs`

Добавить обратно модули:
```rust
// Добавить после строки `pub(crate) mod lighting_legacy;`:
pub(crate) mod radiance_cascades;
pub(crate) mod volumetric_lights;
```

### Шаг 3: Восстановить `src/core/lighting/mod.rs`

Заменить содержимое на оригинальное из `archiveLighting/modified_originals/lighting_mod.rs`:
```rust
pub mod voxel_grid;
pub mod light_prop;
pub mod csm;
pub mod lightmap_lut;
pub mod clustered;
pub mod ssao;
pub mod voxel_renderer;

pub use crate::core::lighting_legacy::*;
```

### Шаг 4: Восстановить `src/core/config.rs`

Добавить обратно после `use crate::debug_log;`:
```rust
// LightingMode enum + атомарные геттеры/сеттеры
// (скопировать из archiveLighting/modified_originals/config.rs строки 10-108)
```

### Шаг 5: Восстановить `src/screens/game_3d_pipeline.rs`

Полностью заменить на оригинал из `archiveLighting/modified_originals/game_3d_pipeline.rs`.  
Ключевые отличия:
- `include_str!` пути: `"../core/radiance_cascades/shaders/pbr_lighting.wgsl"` и `"../core/radiance_cascades/shaders/outline.wgsl"`
- GI Bind Group Layout (`gi_bgl`) с 6 bindings (C0-C4 intervals)
- Bindings 3-6: voxel_tex, block_lut, normal_tex, normal_sampler
- `render()` принимает `gi_bind_group`, `voxel_view`, `block_lut_view`
- `set_gi_bgl()`, `chunk_meshes()`, `depth_view()` методы

### Шаг 6: Восстановить `src/screens/game_screen.rs`

Полностью заменить на оригинал из `archiveLighting/modified_originals/game_screen.rs`.  
Ключевые отличия:
- Импорты RC, Voxel, Volumetric
- Поля в `GameScreen`: `rc_system`, `voxel_renderer`, `volumetric_pass`, `cached_voxel_emitters`, `cached_emissive_gpu`, и т.д.
- RC dispatch в `render()` (throttled, every 4 frames)
- Voxel GPU Lighting path (lighting_mode switch)
- Volumetric light pass (halos + god rays)
- Emissive block processing

### Шаг 7: Восстановить `src/screens/settings.rs`

Добавить обратно UI lighting mode:
```rust
use crate::core::config::LightingMode;
// И блок UI из оригинала (строки 69-205)
```

### Шаг 8: Восстановить `src/core/world_gen/world.rs`

Добавить обратно:
- `use crate::core::radiance_cascades::voxel_tex::{VOXEL_TEX_SIZE, VoxelTextureBuilder};`
- `VoxelChunkRegion`, `VoxelUpdate` типы
- `EmissiveBlockPos` структуру
- `fill_voxel_texture()` метод
- `scan_emissive_blocks()` метод
- `voxel_data` и `emissive_blocks` поля в `WorldResult`
- Voxel texture update + emissive scan в воркере

### Шаг 9: Восстановить `src/screens/profiler_overlay.rs`

Добавить обратно поля в `ProfilerFrame`:
```rust
pub voxel_upload_time_us: u128,
pub voxel_uploaded: bool,
pub rc_dispatch_time_us: u128,
pub rc_dispatched: bool,
pub rc_was_full: bool,
```

И соответствующие строки в `draw_egui()`.

### Шаг 10: Восстановить `src/screens/debug_overlay.rs`

Добавить обратно RC probe grid debug строки в `draw_egui()`.

### Шаг 11: Удалить упрощённые шейдеры

```batch
del /Q "src\shaders\pbr_lighting.wgsl"
del /Q "src\shaders\outline.wgsl"
rmdir "src\shaders"
```

### Шаг 12: Собрать и проверить

```batch
cargo build --release
```

---

## Заметки

- `lighting_config.json` остался в `assets/` — он используется DayNightCycle
- `lighting_legacy.rs` НЕ был удалён — он содержит DayNightCycle, LightingConfig, pack_lighting_uniforms()
- `sky_renderer.rs` НЕ был изменён — он использует DayNightCycle для sun/moon billboard
- Упрощённые шейдеры в `src/shaders/` — заменяют оригинальные из `src/core/radiance_cascades/shaders/`
- `gameobjects/world.rs` был осиротевшим файлом (не подключён в `gameobjects.rs`) — его можно не восстанавливать
