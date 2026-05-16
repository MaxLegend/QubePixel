# Аудит системы Radiance Cascades

**Дата:** 2026-04-18  
**Подсистема:** Global Illumination — Radiance Cascades  
**Статус:** Активная (основная система GI)

---

## 1. Обзор архитектуры

Radiance Cascades (RC) — основная система глобального освещения в QubePixel.  
Реализует иерархию из 5 каскадов (C0=детальный → C4=грубый), каждый из которых хранит probe grid с наборами лучей. Лучи маршируются через 3D воксельную текстуру (DDA), затем каскады сливаются (merge) от грубого к детальному с трилинейной интерполяцией.

### Каскадная иерархия

| Каскад | Spacing | Subdiv | Rays | Grid half-extent | Grid size | Ray length |
|--------|---------|--------|------|------------------|-----------|------------|
| C0     | 8.0     | 2      | 24   | 6                | 13³       | 64.0       |
| C1     | 16.0    | 2      | 24   | 4                | 9³        | 128.0      |
| C2     | 32.0    | 3      | 54   | 2                | 5³        | 256.0      |
| C3     | 64.0    | 4      | 96   | 2                | 5³        | 512.0      |
| C4     | 128.0   | 5      | 150  | 1                | 3³        | 1024.0     |

> **Примечание:** `BASE_SPACING = 8.0`, каждый следующий каскад удваивает spacing.  
> `CUBE_FACE_SUBDIVS = [2, 2, 3, 4, 5]`, `CASCADE_GRID_HALF_EXTENTS = [6, 4, 2, 2, 1]`.

### Порядок dispatch'а (per-frame)

```
1. rc.recenter(camera_pos)              — пересчёт grid_origins + fractional_offsets
2. rc.update_voxel_texture(queue)       — загрузка 256³ R16Uint (block IDs)
3. rc.update_block_lut(queue)           — загрузка 256×1 Rgba32Float (opacity+emission)
4. rc.dispatch(encoder, queue, sky_brightness):
   a. Ray March C0-C1 (каждый кадр), C2-C4 (каждый 4-й кадр)
   b. Emission Injection C0 + C1 (точечные источники)
   c. Merge chain: C3←C4 → C2←C3 → C1←C2 → C0←C1
   d. Обновление GiSampleParams (256B uniform)
5. Fragment shader: multi-cascade GI sampling через gi_bg bind group
```

---

## 2. Файловая структура

### Rust (src/core/radiance_cascades/)

| Файл | Роль | Строки |
|------|------|--------|
| [`radiance_cascades.rs`](src/core/radiance_cascades.rs) | Модульный корень, реэкспорт типов | 20 |
| [`types.rs`](src/core/radiance_cascades/types.rs) | `CascadeConfig`, `RadianceInterval` (48B), `ProbeGrid`, `BlockProperties` (16B), `RadianceCascadeSystem` (CPU) | 797 |
| [`system.rs`](src/core/radiance_cascades/system.rs) | **`RadianceCascadeSystemGpu`** — GPU-оркестратор: pipelines, bind groups, dispatch | 1557 |
| [`dispatch.rs`](src/core/radiance_cascades/dispatch.rs) | `RayMarchPipeline` — standalone pipeline для DDA ray march | 561 |
| [`merge.rs`](src/core/radiance_cascades/merge.rs) | `MergePipeline` — standalone pipeline для cascade merge | 588 |
| [`sampling.rs`](src/core/radiance_cascades/sampling.rs) | Генерация направлений: cube-face subdivision + Fibonacci sphere | 419 |
| [`voxel_tex.rs`](src/core/radiance_cascades/voxel_tex.rs) | `VoxelTextureBuilder` (256³), `BlockLUTBuilder` (256×1) | 453 |

### WGSL Shaders (src/core/radiance_cascades/shaders/)

| Файл | Роль | Entry point |
|------|------|-------------|
| [`ray_march.wgsl`](src/core/radiance_cascades/shaders/ray_march.wgsl) | DDA ray marching через воксельную текстуру | `main` |
| [`merge.wgsl`](src/core/radiance_cascades/shaders/merge.wgsl) | Cascade merge с трилинейной интерполяцией + Trilinear Fix | `main` |
| [`emission_inject.wgsl`](src/core/radiance_cascades/shaders/emission_inject.wgsl) | Point-light injection от emissive blocks в C0/C1 | `main` |
| [`pbr_lighting.wgsl`](src/core/radiance_cascades/shaders/pbr_lighting.wgsl) | PBR + voxel shadow + multi-cascade GI sampling (фрагментный) | `vs_main` / `fs_main` |
| [`gi_sample.wgsl`](src/core/radiance_cascades/shaders/gi_sample.wgsl) | Устаревший single-cascade GI sampler (не используется напрямую) | — |
| [`volumetric_lights.wgsl`](src/core/radiance_cascades/shaders/volumetric_lights.wgsl) | Halo + god-ray billboard quads для emissive blocks | `vs_halo`/`fs_halo` + `vs_ray`/`fs_ray` |
| [`outline.wgsl`](src/core/radiance_cascades/shaders/outline.wgsl) | Block selection outline shader | `vs_main` / `fs_main` |

---

## 3. Критические GPU-структуры (compile-time asserts)

Все размеры проверяются через `const _: () = assert!(...)` в [`system.rs`](src/core/radiance_cascades/system.rs:31):

| Структура | Размер | WGSL-аналог |
|-----------|--------|-------------|
| `CascadeDispatchParams` | 64B | `CascadeParams` в `ray_march.wgsl` |
| `MergeParams` | 96B | `MergeParams` в `merge.wgsl` |
| `GiCascadeEntry` | 48B | `GiCascadeEntry` в `pbr_lighting.wgsl` |
| `GiSampleParams` | 256B | `GiSampleParams` в `pbr_lighting.wgsl` |
| `RadianceInterval` | 48B | `RadianceInterval` (3 × vec4) |
| `EmissiveBlockGpu` | 32B | `EmissiveBlock` в `emission_inject.wgsl` |

> ⚠️ **КРИТИЧЕСКОЕ ПРАВИЛО:** При изменении любой Rust-структуры нужно обновить WGSL-аналог и compile-time assert.

---

## 4. Анализ шейдеров

### 4.1 ray_march.wgsl — DDA Ray March

**Алгоритм:**
- Один invocation = один (probe, ray)
- DDA (Amanatides-Woo) трассировка через 256³ воксельную текстуру
- Для каждого вокселя: чтение block_id → lookup в Block LUT → opacity + emission
- Накопление: `tau`, `transmittance`, `radiance_in`, `radiance_out`
- Sky contribution: **только в C4** (`cascade_level == 4`) — предотвращает double-counting
- `radiance_in.a` = `pre_hit_tau` (tau до opaque hit) — для emission attenuation в fragment shader

**Проблемы:**
1. **Старый fallback для направлений:** Если `ray_idx >= num_stored`, используется `cube_face_dir_gpu`. Но буферы направлений создаются на все ray_count, так что fallback не должен срабатывать. Однако код содержит потенциальный race condition при `num_stored == 0`.
2. **Emission double-counting:** При opaque hit emission вычитается из `radiance_in` (строка 296), но это может привести к отрицательным значениям при определённых условиях.

### 4.2 merge.wgsl — Cascade Merge

**Алгоритм:**
- Trilinear интерполяция 8 parent probes → weighted average
- Trilinear Fix: perpendicular displacement damping (`1 / (1 + perp_len * 0.25)`)
- Merge formula: `M.radiance_in = child.radiance_in + exp(-child.tau) * interp_parent.radiance_in`
- `M.radiance_out = interp_parent.radiance_out` (far-end emission)
- `M.tau = min(child.tau + parent.tau, opaque_threshold)`

**Проблемы:**
1. **Clamp merged_in до 2.0** — может быть слишком агрессивно для ярких emissive блоков.
2. **Trilinear Fix damping = 0.25** — константа, не настраивается. Может быть недостаточной для тонких стен.

### 4.3 emission_inject.wgsl — Point Light Injection

**Алгоритм:**
- Для каждого (probe, ray): итерация по всем emissive blocks
- Attenuation: `1 / (1 + k * d²)` (inverse square)
- NdotL hemisphere check: только лучи, направленные к блоку
- Добавляет `emission * attenuation * ndotl` к `radiance_in`

**Проблемы:**
1. **O(probes × rays × blocks)** — при 256 emissive blocks и C0 (13³×24 = ~52k intervals) это ~13M итераций. При большом количестве emissive blocks может стать bottleneck.
2. **Нет spatial culling** — все emissive blocks проверяются для каждого probe, даже если они далеко за пределами influence radius. Проверка `dist > influence_radius` происходит внутри цикла, но не до него.

### 4.4 pbr_lighting.wgsl — Fragment Shader

**Алгоритм:**
- Multi-cascade GI sampling: C0→C4, finest first
- Boundary fade: quadratic ease на outer 15% каждого каскада
- Stride sampling: max 6 rays per probe (stride = ray_count / 6)
- Solid angle weighting: `4π / ray_count × stride`
- Voxel ray-traced shadows: DDA через voxel texture от fragment к sun
- PBR: GGX + Smith + Schlick Fresnel
- ACES tone mapping + exposure 0.95

**Проблемы:**
1. **GiSampleParams layout mismatch:** WGSL-структура `GiSampleParams` в `pbr_lighting.wgsl` содержит `array<GiCascadeEntry, 5>` (5×48B = 240B) + 16B tail = 256B. Но WGSL `GiCascadeEntry` имеет размер 48B (vec4 + vec2+vec2 + vec3+f32), а Rust-аналог тоже 48B. Однако в WGSL `array<GiCascadeEntry, 5>` в uniform address space требует `AlignOf(element) >= 16`. `GiCascadeEntry` содержит `vec4<f32>` → `AlignOf = 16`. OK.
2. **gi_intensity = 0.08** — захардкожено в [`system.rs:1449`](src/core/radiance_cascades/system.rs:1449). Не вынесено в настройки.
3. **8 debug режимов** (GI_DEBUG_MODE 0-8) — полезно для отладки, но в production нужно отключать (const = 0).

---

## 5. Дублирование кода

### 5.1 Два набора GPU-структур

В проекте существует **дублирование** GPU-структур между standalone pipeline файлами и `system.rs`:

| Структура | В `dispatch.rs` | В `system.rs` |
|-----------|-----------------|---------------|
| `CascadeDispatchParams` | 64B (строка 59) | 64B (строка 65) |
| `MergeParams` | — | 96B (строка 119) |
| `MergeParams` | 96B (строка 53 в `merge.rs`) | — |

**Рекомендация:** Удалить дубликаты из `dispatch.rs` и `merge.rs`, использовать единственные определения из `system.rs` (или вынести в `types.rs`).

### 5.2 Дублирование cube_face_dir_gpu / derive_subdiv

Функции `cube_face_dir_gpu()` и `derive_subdiv()` копируются в **4 шейдера**:
- `ray_march.wgsl`
- `merge.wgsl`
- `emission_inject.wgsl`
- (частично) `pbr_lighting.wgsl` через `nearest_cube_face_dir`

**Рекомендация:** Вынести в общий include-файл (например, `rc_common.wgsl`) и подключать через `#include` или string concatenation.

### 5.3 Два CPU-side cascade system

- [`RadianceCascadeSystem`](src/core/radiance_cascades/types.rs:477) — CPU-side, с `ProbeGrid`, `HashMap`, `gpu_texture: Option<>`. Метод `init_gpu()` — заглушка.
- [`RadianceCascadeSystemGpu`](src/core/radiance_cascades/system.rs:222) — реальный GPU-оркестратор.

**Рекомендация:** Удалить `RadianceCascadeSystem` или объединить с `RadianceCascadeSystemGpu`, так как CPU-side система не используется для рендеринга.

---

## 6. Проблемы и баги

### 🔴 Критические

1. **Voxel texture size mismatch:**  
   - `voxel_tex.rs` определяет `VOXEL_TEX_SIZE = 256`  
   - `ray_march.wgsl` определяет `VOXEL_TEX_SIZE = 256`  
   - Но `system.rs:329` создаёт текстуру с `VOXEL_TEX_SIZE` из `voxel_tex.rs`  
   - **OK** — значения совпадают, но нет compile-time проверки соответствия Rust ↔ WGSL констант.

2. **ARCHITECTURE-GI.md устарел:**  
   - Документация описывает `128³` voxel texture, но реально используется `256³` (`VOXEL_TEX_SIZE = 256`).
   - Таблица cascade hierarchy не соответствует актуальным константам (`BASE_SPACING = 8.0`, не 1.0).

3. **GiCascadeEntry размер:**  
   - Rust: 48B (`origin_and_spacing: [f32;4]` + `counts: [u32;2]` + `_pad: [u32;2]` + `fractional_offset: [f32;3]` + `_pad2: u32`)  
   - WGSL в `pbr_lighting.wgsl`: `origin_and_spacing: vec4<f32>` (16B) + `counts: vec2<u32>` (8B) + `_pad: vec2<u32>` (8B) + `fractional_offset: vec3<f32>` (12B) + `_pad2: f32` (4B) = 48B  
   - **OK**, но WGSL struct имеет AlignOf = 16 из-за `vec4`, что даёт stride = 48 (48 / 16 = 3, целое). Массив в uniform buffer корректен.

### 🟡 Средние

4. **Staggered dispatch не синхронизирован с merge:**  
   - C2-C4 пропускаются 3 из 4 кадров (только ray march)  
   - Но merge chain всегда выполняется полностью, включая C3←C4 и C2←C3  
   - На non-full dispatch merge читает stale данные из C2-C4 output buffers — это по дизайму, но может вызывать артефакты "запаздывания" GI на дальних дистанциях.

5. **Emission injection в C1:**  
   - Emissive blocks инжектируются в оба каскада C0 и C1  
   - Это удваивает стоимость emission injection  
   - Комментарий говорит о "стабилизации при движении grid", но это может быть избыточно.

6. **`unsafe` в `light_prop.rs`:**  
   - `unsafe { std::slice::from_raw_parts(...) }` используется для каста структур в байты  
   - Безопаснее использовать `bytemuck::cast_slice()` или `bytemuck::bytes_of()`.

7. **`dispatch()` в `light_prop.rs` использует сырой указатель:**  
   - Строка 360: `unsafe { &*(1 as *const wgpu::Device) }` — **это UB и вызовет краш**  
   - Есть безопасная альтернатива `dispatch_safe()`, но `dispatch()` не удалён и может быть вызван по ошибке.

### 🟢 Мелкие

8. **Gamma correction закомментирована** в `pbr_lighting.wgsl` (строки 549-559).  
9. **`gi_sample.wgsl` — устаревший файл**, не используется напрямую (multi-cascade sampling встроен в `pbr_lighting.wgsl`).  
10. **Тесты в `types.rs` используют устаревшие константы** — `BASE_SPACING = 1.0` в тестах, хотя реально `8.0`.

---

## 7. VRAM Usage

| Ресурс | Размер |
|--------|--------|
| Voxel texture (256³ × R16Uint) | 32 MB |
| Block LUT (256 × Rgba32Float) | 4 KB |
| C0 output (13³ × 24 × 48B) | ~2.5 MB |
| C1 output (9³ × 24 × 48B) | ~0.8 MB |
| C2 output (5³ × 54 × 48B) | ~0.3 MB |
| C3 output (5³ × 96 × 48B) | ~0.6 MB |
| C4 output (3³ × 150 × 48B) | ~0.2 MB |
| Ray dir buffers (C0-C4) | ~20 KB |
| Uniform buffers | ~1 KB |
| **Итого** | **~36.5 MB** |

---

## 8. Производительность

### Bottlenecks

1. **Ray March C0:** 13³ × 24 = ~52k invocations × 128 max steps = до 6.7M DDA шагов/кадр  
2. **Emission Injection:** O(probes × rays × blocks) — при 100+ emissive blocks может доминировать  
3. **Merge chain:** 4 прохода, каждый читает parent + пишет child  
4. **Fragment shader GI:** 8 probes × 6 rays × 5 cascades = до 240 storage buffer reads/фрагмент (со stride sampling)

### Оптимизации (уже реализованы)

- ✅ Staggered dispatch (C2-C4 каждый 4-й кадр)  
- ✅ Stride sampling в fragment shader (max 6 rays/probe)  
- ✅ Cube-face directions (лучше для axis-aligned мира чем Fibonacci)  
- ✅ Pre-created bind groups (не пересоздаются каждый кадр)  
- ✅ Boundary fade (предотвращает seams между каскадами)

### Возможные оптимизации

- 🔧 Spatial culling для emission injection (grid-based bucketing)  
- 🔧 Half-resolution GI sampling (рендерить GI в половинном разрешении)  
- 🔧 Temporal accumulation (усреднение GI между кадрами)  
- 🔧 Async compute (RC pipeline на отдельной queue, если поддерживается)

---

## 9. Рекомендации

### Приоритет 1 (баги)

1. **Удалить `dispatch()` в `light_prop.rs`** с UB-указателем или пометить `#[deprecated]`  
2. **Обновить `ARCHITECTURE-GI.md`** под актуальные константы (256³ voxel, BASE_SPACING=8.0)  
3. **Исправить тесты** в `types.rs` под актуальные значения констант

### Приоритет 2 (качество кода)

4. **Вынести общие WGSL функции** (`cube_face_dir_gpu`, `derive_subdiv`) в `rc_common.wgsl`  
5. **Удалить дублирование** `CascadeDispatchParams` / `MergeParams` между `dispatch.rs`, `merge.rs` и `system.rs`  
6. **Удалить или объединить** `RadianceCascadeSystem` (CPU) с `RadianceCascadeSystemGpu`  
7. **Заменить `unsafe` slice casts** на `bytemuck::cast_slice()` / `bytemuck::bytes_of()`

### Приоритет 3 (настройка)

8. **Вынести `gi_intensity`** (0.08) и `bounce_intensity` (1.0) в конфигурацию  
9. **Вынести `TRILINEAR_FIX_DAMPING`** (0.25) в конфигурацию  
10. **Добавить runtime toggle** для debug режимов (вместо перекомпиляции)

---

## 10. Связанные файлы (вне radiance_cascades/)

| Файл | Связь |
|------|-------|
| [`src/core/volumetric_lights.rs`](src/core/volumetric_lights.rs) | Halo + god-ray эффекты для emissive blocks (дополнение к RC GI) |
| [`src/core/lighting_legacy.rs`](src/core/lighting_legacy.rs) | `DayNightCycle`, `pack_lighting_uniforms()`, `sky_brightness()` |
| [`src/screens/game_screen.rs`](src/screens/game_screen.rs) | Интеграция: вызывает `recenter()`, `dispatch()`, передаёт `gi_bg` в render pass |
| [`src/screens/game_3d_pipeline.rs`](src/screens/game_3d_pipeline.rs) | Render pipeline, включает `gi_bgl` в layout |
| [`ARCHITECTURE-GI.md`](ARCHITECTURE-GI.md) | Документация (устаревшая) |
