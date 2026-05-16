# Аудит системы Flow Fill (Voxel GPU Lighting)

**Дата:** 2026-04-18  
**Подсистема:** Voxel GPU Lighting — Flow Fill / Light Propagation  
**Статус:** Альтернативная система (параллельно с Radiance Cascades, переключение через settings)

---

## 1. Обзор архитектуры

Flow Fill (Voxel GPU Lighting) — вторая система освещения в QubePixel, работающая параллельно с Radiance Cascades. Основана на GPU flood fill в 3D-текстуре (128³ RGBA8, double-buffered) с compute shader распространением света. Включает в себя Clustered Forward Rendering для динамических источников, Cascaded Shadow Maps (CSM) для теней от направленного света, Lightmap LUT для SkyLight→цвет, и SSAO.

### Компоненты системы

```
VoxelLightingRenderer
├── VoxelLightGrid          — 128³ RGBA8 double-buffered 3D-текстура + R8 occlusion
├── LightPropagationPass    — Compute shader flood fill (dirty regions)
├── LightmapLut             — 256×64 RGBA8 SkyLight→цвет по time-of-day
├── ClusteredForwardSystem  — 16×8×24 кластеров, max 256 lights
├── Shadow Pipeline         — Depth-only pass (2048²) с точки зрения солнца
├── Main Lighting Pipeline  — PBR + voxel light + CSM + clustered
└── SSAO                    — Screen Space Ambient Occlusion (опционально)
```

### Порядок рендера (per-frame)

```
1. grid.recenter(camera_pos)            — scroll 3D-текстуры, mark dirty
2. grid.upload_emitters(queue, ...)     — загрузка emissive blocks
3. propagation.dispatch_safe(...)       — compute flood fill (до 16 шагов)
4. lut.rebuild(queue, time_of_day)      — перестройка Lightmap LUT
5. cf_system.dispatch_cull(encoder)     — clustered light culling
6. render():
   a. Shadow pass (depth-only, sun POV)
   b. Main pass (PBR + voxel light + clustered + LUT)
```

---

## 2. Файловая структура

### Rust (src/core/lighting/)

| Файл | Роль | Строки |
|------|------|--------|
| [`mod.rs`](src/core/lighting/mod.rs) | Модульный корень, реэкспорт `lighting_legacy` | 27 |
| [`voxel_grid.rs`](src/core/lighting/voxel_grid.rs) | `VoxelLightGrid` — GPU 3D-текстура (double-buffered) + occlusion | 597 |
| [`light_prop.rs`](src/core/lighting/light_prop.rs) | `LightPropagationPass` — compute shader flood fill | 430 |
| [`csm.rs`](src/core/lighting/csm.rs) | `CsmSystem` + `LightmapLut` — каскадные тени + SkyLight LUT | 473 |
| [`clustered.rs`](src/core/lighting/clustered.rs) | `ClusteredForwardSystem` — frustum cluster culling | 454 |
| [`ssao.rs`](src/core/lighting/ssao.rs) | `SsaoPass` — Screen Space Ambient Occlusion | 385 |
| [`voxel_renderer.rs`](src/core/lighting/voxel_renderer.rs) | `VoxelLightingRenderer` — полный рендерер (оркестратор) | 822 |
| [`lightmap_lut.rs`](src/core/lighting/lightmap_lut.rs) | Реэкспорт `LightmapLut` из `csm.rs` | 9 |

### WGSL Shaders (src/core/lighting/shaders/)

| Файл | Роль | Entry point |
|------|------|-------------|
| [`light_propagation.wgsl`](src/core/lighting/shaders/light_propagation.wgsl) | Flood fill: 6-neighbor max с falloff + occlusion | `cs_propagate` |
| [`voxel_lighting.wgsl`](src/core/lighting/shaders/voxel_lighting.wgsl) | PBR + voxel light + CSM shadow + clustered + LUT | `vs_main` / `fs_main` |
| [`voxel_shadow.wgsl`](src/core/lighting/shaders/voxel_shadow.wgsl) | Depth-only shadow pass (sun POV) | `vs_shadow` |
| [`cluster_cull.wgsl`](src/core/lighting/shaders/cluster_cull.wgsl) | Clustered light culling compute | `cs_cluster_cull` |
| [`ssao.wgsl`](src/core/lighting/shaders/ssao.wgsl) | SSAO compute + bilateral blur | `cs_ssao` / `cs_ssao_blur` |

### Связанные файлы

| Файл | Связь |
|------|-------|
| [`src/core/lighting_legacy.rs`](src/core/lighting_legacy.rs) | `DayNightCycle`, `LightingConfig`, `pack_lighting_uniforms()` |
| [`src/core/config.rs`](src/core/config.rs) | Настройки: `voxel_rgb_falloff()`, `voxel_skylight_falloff()`, `voxel_propagation_steps()` |

---

## 3. GPU-структуры

### PropagationParams (48 bytes) — [`light_prop.rs`](src/core/lighting/light_prop.rs:40)

```rust
pub struct PropagationParams {
    pub grid_size: u32,                      // [ 0] = 128
    pub rgb_falloff: f32,                    // [ 4] = 0.95
    pub skylight_horizontal_falloff: f32,     // [ 8] = 0.90
    pub step_count: u32,                      // [12] = 16
    pub world_origin_x/y/z: f32,             // [16..28]
    pub clear_dirty: f32,                     // [28]
    pub _pad0.._pad3: f32,                   // [32..48]
}
```

### DirtyRegionGpu (24 bytes) — [`light_prop.rs`](src/core/lighting/light_prop.rs:78)

```rust
pub struct DirtyRegionGpu {
    pub min_x/y/z: u32,  // [0..12]
    pub max_x/y/z: u32,  // [12..24]
}
```

### GpuLight (32 bytes) — [`clustered.rs`](src/core/lighting/clustered.rs:44)

```rust
pub struct GpuLight {
    pub position_radius: [f32; 4],   // [0..16]
    pub color_intensity: [f32; 4],   // [16..32]
}
```

### SsaoParams (176 bytes) — [`ssao.rs`](src/core/lighting/ssao.rs:83)

```rust
pub struct SsaoParams {
    pub view_proj: [[f32; 4]; 4],      // [0..64]
    pub inv_view_proj: [[f32; 4]; 4],   // [64..128]
    pub camera_pos: [f32; 4],           // [128..144]
    pub screen_params: [f32; 4],        // [144..160]
    pub ssao_params: [f32; 4],          // [160..176]
}
```

---

## 4. Анализ компонентов

### 4.1 VoxelLightGrid — 3D Light Texture

**Характеристики:**
- Размер: 128³ (±64 блока от камеры)
- Формат: RGBA8Unorm (double-buffered, ping-pong)
- Occlusion: отдельная 128³ R8Unorm текстура (1.0 = opaque, 0.0 = air)
- Scroll: привязка к 16-блочной сетке, mark_scroll_dirty при сдвиге
- Dirty regions: до 64 регионов/кадр

**Проблемы:**
1. **128³ = ±64 блока** — этого может быть недостаточно для дальнего освещения (сравните с RC: 256³ = ±128 блоков)
2. **RGBA8 = 8 бит/канал** — низкая точность для плавных градиентов освещения (256 уровней). RC использует Rgba32Float.
3. **Occlusion загружается по 1 вокселю за вызов** `upload_occlusion()` — крайне неэффективно для больших объёмов
4. **`clear()` не реализован** — метод помечен как "помечаем всё dirty", но не очищает данные
5. **`clear_occlusion()` загружает по одному Z-срезу** (128 вызовов `write_texture`) — медленно

### 4.2 LightPropagationPass — Flood Fill

**Алгоритм (`light_propagation.wgsl`):**
- Workgroup: 8×8×4 = 256 invocations
- Для каждого вокселя: проверка 6 соседей (±X, ±Y, ±Z)
- RGB: `result = max(current, max(neighbor.rgb * falloff))`
- SkyLight: вертикальное без затухания, горизонтальное с `skylight_horizontal_falloff`
- Occlusion: непрозрачные воксели блокируют распространение
- Dirty regions: только dirty-области обновляются (если есть), иначе — полная сетка

**Проблемы:**
1. **🔴 UB в `dispatch()`** — строка 360:
   ```rust
   grid.bind_group(/* device needed */ unsafe { &*(1 as *const wgpu::Device) }, &[])
   ```
   Это **неопределённое поведение** — создание ссылки из сырого указателя `1`. Вызовет краш или молчаливую порчу памяти. Существует безопасная альтернатива `dispatch_safe()`, но опасный метод не удалён и не помечен как deprecated.

2. **`unsafe` для slice casts** — используется `std::slice::from_raw_parts` вместо `bytemuck::cast_slice()`

3. **Max 16 шагов/кадр** — при falloff 0.95 свет распространяется на ~51 блок. Для заполнения всего 128³ grid нужно ~128 итераций, т.е. ~8 кадров. Это создаёт задержку (light "ползёт" от источников).

4. **Нет DispatchIndirect** — буфер `indirect_buffer` создаётся, но не используется. Dispatch всегда выполняется на полную сетку.

5. **SkyLight: вертикальное без затухания** — свет "льется" вниз бесконечно через прозрачные воксели. Это может вызывать артефакты: подземные комнаты могут получать SkyLight через 1-блочные потолки.

### 4.3 CsmSystem — Cascaded Shadow Maps

**Характеристики:**
- 4 каскада, 1024² каждый
- Split scheme: logarithmic + uniform (lambda = 0.75)
- Depth bias: constant=2, slope_scale=2.0
- PCF Poisson 16-tap (в voxel_lighting.wgsl)

**Проблемы:**
1. **Только 1 каскад в BGL** — bind group layout содержит одну `texture_depth_2d`, а не массив. Используется только первый каскад.
2. **`update_cascades()` не вызывается** — в `VoxelLightingRenderer` нет вызова `csm_system.update_cascades()`. Каскады инициализируются единичными матрицами и никогда не обновляются.
3. **CSM система создана, но не интегрирована** — `CsmSystem` существует как standalone, но `VoxelLightingRenderer` использует собственный shadow pass с одной orthographic матрицей.

### 4.4 LightmapLut — SkyLight → Color

**Характеристики:**
- 256×64 RGBA8Unorm
- Ось X: SkyLight intensity (0–255)
- Ось Y: Time of day (0.0–1.0)
- Перестраивается каждый кадр (`rebuild()`)

**Проблемы:**
1. **Перестройка каждый кадр** — 256×64×4 = 64KB, генерируется на CPU и загружается через `write_texture`. Это лишняя работа, если time-of-day не изменился.
2. **Цветовые переходы грубые** — `sky_color_for_time()` использует простые линейные интерполяции с жёсткими границами (0.20, 0.30, 0.70, 0.80).

### 4.5 ClusteredForwardSystem

**Характеристики:**
- 16×8×24 = 3072 кластера
- Max 256 lights, max 32/cluster
- Compute culling + fragment lookup

**Проблемы:**
1. **`update_params()` записывает только 32 байта** — не включает `inv_view_proj` матрицу (нужна для reconstruct world pos). Fragment shader `voxel_lighting.wgsl` не использует clustered lighting для реконструкции позиции — вместо этого передаёт `world_pos` из vertex shader.
2. **`cf_cluster_bounds` не заполняется** — буфер создаётся, но никогда не обновляется реальными AABB кластеров. Culling shader, вероятно, вычисляет bounds на лету или использует заглушку.

### 4.6 SsaoPass

**Характеристики:**
- Max 64 kernel samples
- 4×4 noise texture
- Compute + bilateral blur

**Проблемы:**
1. **Kernel buffer не загружается на GPU** — `kernel_buffer` создаётся, данные генерируются, но `queue.write_buffer()` никогда не вызывается для kernel. Bind group не создаётся.
2. **`ssao_view()` возвращает blur_view** — но blur pipeline использует тот же BGL что и SSAO pipeline, что может быть некорректно (binding 4 = storage write, но blur должен читать SSAO output).
3. **SSAO не интегрирован в `VoxelLightingRenderer`** — `SsaoPass` создаётся отдельно, но не вызывается в `update()` или `render()`.

### 4.7 VoxelLightingRenderer — Оркестратор

**Проблемы:**
1. **CSM не используется** — `CsmSystem` не является полем `VoxelLightingRenderer`. Вместо этого используется собственный shadow pass с одной матрицей.
2. **SSAO не используется** — нет поля `SsaoPass` в рендерере.
3. **Bind groups пересоздаются каждый кадр** — `bg_main`, `voxel_bg`, `shadow_bg` создаются через `device.create_bind_group()` каждый кадр. Это не оптимально (должны кэшироваться).
4. **`pending_emitters` — Vec с аллокацией** — `set_emitters()` заменяет весь вектор каждый раз, вместо re-use.
5. **Exposure = 3.4** — захардкожен в `voxel_lighting.wgsl` (строка 443). Сравните с RC: 0.95. Разница в ~3.5× — системы освещения не сбалансированы.

---

## 5. Сравнение с Radiance Cascades

| Аспект | Flow Fill | Radiance Cascades |
|--------|-----------|-------------------|
| **Подход** | Flood fill в 3D текстуре | Ray marching + cascade merge |
| **Размер grid** | 128³ (±64 блока) | 256³ voxel + probe grids |
| **Точность** | RGBA8 (8 бит/канал) | Rgba32Float (32 бит/канал) |
| **GI качество** | Низкое (max-falloff, без bounce) | Высокое (ray-based, с merge chain) |
| **Emissive blocks** | Через emitter upload + propagation | Через emission injection compute |
| **Тени** | CSM (не интегрирован) + voxel shadow | Voxel ray-traced shadow (DDA) |
| **SkyLight** | Вертикальный flood fill | C4 sky radiance + merge chain |
| **Dynamic lights** | Clustered Forward (256 max) | Emission injection (ограничено VRAM) |
| **VRAM** | ~16 MB (128³ × 2 + occlusion) | ~36.5 MB |
| **Задержка обновления** | ~8 кадров для полного заполнения | 1 кадр (C0-C1), 4 кадр (C2-C4) |
| **Exposure** | 3.4 | 0.95 |

---

## 6. Проблемы и баги

### 🔴 Критические

1. **UB в `LightPropagationPass::dispatch()`** — [`light_prop.rs:360`](src/core/lighting/light_prop.rs:360)  
   `unsafe { &*(1 as *const wgpu::Device) }` — создание ссылки из нулевого/невалидного указателя.  
   **Исправление:** Удалить метод `dispatch()`, оставить только `dispatch_safe()`.

2. **CSM не интегрирован** — `CsmSystem` создаётся, но не используется в `VoxelLightingRenderer`. Shadow pass использует единую orthographic матрицу вместо каскадов.

3. **SSAO не интегрирован** — `SsaoPass` существует, но не вызывается в рендерере. Kernel buffer не загружен.

### 🟡 Средние

4. **Cluster bounds не заполняются** — `cluster_bounds_buffer` создаётся, но никогда не обновляется реальными AABB.

5. **LightmapLut перестраивается каждый кадр** — даже если time-of-day не изменился.

6. **Occlusion загружается по 1 вокселю** — `upload_occlusion()` делает один `write_texture` на блок. Для 16³×16³×16³ чанка это 4096 вызовов.

7. **`clear()` не реализован** — `VoxelLightGrid::clear()` не очищает данные.

8. **Bind groups пересоздаются каждый кадр** — 3 bind groups в `render()`.

9. **Exposure не сбалансирован** — 3.4 (Flow Fill) vs 0.95 (RC). При переключении между системами яркость скачкообразно меняется.

### 🟢 Мелкие

10. **`lightmap_lut.rs` — избыточный файл** — содержит только `pub use super::csm::LightmapLut;`  
11. **`ClusterLightHeader` не используется** — структура определена, но не применяется в коде.  
12. **`DispatchIndirectArgs` не используется** — буфер создаётся, но dispatch indirect не вызывается.  
13. **`unsafe` slice casts** — можно заменить на `bytemuck`.  
14. **`SsaoPass::generate_noise()` использует детерминированные "случайные" числа** — не истинный random, но для SSAO это допустимо.

---

## 7. VRAM Usage

| Ресурс | Размер |
|--------|--------|
| Light grid A (128³ × RGBA8) | 8 MB |
| Light grid B (128³ × RGBA8) | 8 MB |
| Occlusion (128³ × R8) | 2 MB |
| Upload buffer (128³ × 4) | 8 MB |
| Shadow map (2048² × Depth32) | 16 MB |
| Cluster SSBOs | ~0.5 MB |
| Lightmap LUT (256×64 × RGBA8) | 64 KB |
| SSAO textures | ~0.5 MB (при 1080p) |
| **Итого** | **~43 MB** |

---

## 8. Производительность

### Bottlenecks

1. **Light propagation:** 128³ / (8×8×4) = 32³ = 32768 workgroups × 16 шагов = 524k dispatch'ей  
   Но реально 1 dispatch с 16 шагами внутри shader — 128³ = 2M вокселей, каждый проверяет 6 соседей.
2. **Shadow pass:** полный рендер сцены с точки зрения солнца (все chunk meshes)  
3. **Lightmap LUT rebuild:** 64KB CPU-side генерация + upload каждый кадр  
4. **Clustered culling:** 16×8×24 = 3072 кластера × 256 lights = 786k checks

### Оптимизации (уже реализованы)

- ✅ Dirty region tracking (не обновляем чистые области)  
- ✅ Double-buffered ping-pong (нет read-write hazard)  
- ✅ Occlusion texture (блокировка распространения через стены)  
- ✅ Clustered Forward (вместо O(n²) light evaluation)

### Возможные оптимизации

- 🔧 Кэшировать bind groups (пересоздавать только при resize/recenter)  
- 🔧 Lightmap LUT: перестраивать только при изменении time-of-day  
- 🔧 Batch occlusion upload (16³ срез за один вызов)  
- 🔧 DispatchIndirect для propagation (только dirty regions)  
- 🔧 Half-resolution propagation (64³ вместо 128³)

---

## 9. Рекомендации

### Приоритет 1 (критические баги)

1. **Удалить или задепрекейтить `dispatch()` в `light_prop.rs`** — UB с сырым указателем  
2. **Интегрировать CSM** в `VoxelLightingRenderer` или удалить `CsmSystem`  
3. **Интегрировать SSAO** или удалить `SsaoPass`  
4. **Заполнить cluster bounds** реальными AABB или вычислять в shader

### Приоритет 2 (качество)

5. **Заменить RGBA8 на Rgba16Float** для light grid — улучшит точность градиентов  
6. **Увеличить grid до 192³ или 256³** — расширить зону покрытия  
7. **Сбалансировать exposure** между Flow Fill и RC (единое значение)  
8. **Кэшировать bind groups** вместо пересоздания каждый кадр  
9. **Batch upload occlusion** — загружать 16³ регионов за один вызов

### Приоритет 3 (настройка)

10. **Вынести exposure (3.4) в конфигурацию**  
11. **Вынести falloff коэффициенты в runtime конфиг**  
12. **Добавить Lightmap LUT cache** — перестраивать только при Δtime > threshold  
13. **Удалить неиспользуемые структуры** (`ClusterLightHeader`, `DispatchIndirectArgs`)

---

## 10. Общий вывод

Система Flow Fill представляет собой **классический voxel GI подход** с flood fill распространением. Она проще в реализации чем Radiance Cascades, но имеет существенные ограничения:

- **Низкое качество GI** — нет bounce lighting, нет directional information, max-falloff даёт плоское освещение  
- **Задержка распространения** — свет "ползёт" от источников со скоростью ~16 блоков/кадр  
- **Низкая точность** — RGBA8 (8 бит) недостаточно для плавных градиентов  
- **Незавершённая интеграция** — CSM, SSAO, cluster bounds не подключены  

**Рекомендация:** Использовать Radiance Cascades как основную систему GI. Flow Fill можно оставить как fallback для слабых GPU (меньше VRAM), но для этого нужно завершить интеграцию CSM + SSAO и исправить критические баги.
