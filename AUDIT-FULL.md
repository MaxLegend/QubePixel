# QubePixel — Полный аудит проекта

> **Дата аудита:** 2026-05-15  
> **Версия проекта:** 0.1.0  
> **Издание Rust:** 2024  
> **Аудитор:** Roo (автоматизированный аудит кодовой базы)

---

## Содержание

1. [Общая информация о проекте](#1-общая-информация-о-проекте)
2. [Архитектура и структура](#2-архитектура-и-структура)
3. [Точка входа и экранная система](#3-точка-входа-и-экранная-система)
4. [Рендеринг и графический конвейер](#4-рендеринг-и-графический-конвейер)
5. [Глобальное освещение (GI)](#5-глобальное-освещение-gi)
6. [Мир и генерация ландшафта](#6-мир-и-генерация-ландшафта)
7. [Система блоков](#7-система-блоков)
8. [Система чанков](#8-система-чанков)
9. [Физика](#9-физика)
10. [Игрок и камера](#10-игрок-и-камера)
11. [Система жидкостей](#11-система-жидкостей)
12. [Система сущностей](#12-система-сущностей)
13. [Модели блоков и Entity-рендеринг](#13-модели-блоков-и-entity-рендеринг)
14. [UI и экраны](#14-ui-и-экраны)
15. [Система сохранений](#15-система-сохранений)
16. [Конфигурация и настройки](#16-конфигурация-и-настройки)
17. [Ассеты и ресурсы](#17-ассеты-и-ресурсы)
18. [Шейдеры](#18-шейдеры)
19. [Многопоточность и производительность](#19-многопоточность-и-производительность)
20. [Логирование](#20-логирование)
21. [Зависимости](#21-зависимости)
22. [Оценка качества кода](#22-оценка-качества-кода)
23. [Выявленные проблемы и риски](#23-выявленные-проблемы-и-риски)
24. [Рекомендации по развитию](#24-рекомендации-по-развитию)
25. [Сводная таблица возможностей](#25-сводная-таблица-возможностей)

---

## 1. Общая информация о проекте

| Параметр | Значение |
|---|---|
| **Название** | QubePixel |
| **Версия** | 0.1.0 |
| **Тип** | Воксельный движок в стиле Minecraft |
| **Язык** | Rust (edition 2024) |
| **Графический API** | wgpu 29.0 (Vulkan / DX12 / Metal) |
| **Шейдеры** | WGSL |
| **Окно** | winit 0.30 |
| **Стартовое разрешение** | 840×480 |
| **Физический движок** | rapier3d 0.32 |
| **Параллелизм** | rayon + tokio + std::thread |

### Ключевые возможности
- Реальное время GI через **VCT** (Voxel Cone Tracing — 128³ flood-fill propagation)
- Полный биомный пайплайн генерации ландшафта (32 биома)
- Симуляция жидкостей (вода/лава с взаимодействиями)
- Скелетная модель игрока (Bedrock Edition формат)
- Система инвентаря с drag-and-drop
- Цикл дня/ночи с солнцем и луной
- Система сохранений миров
- LOD (3 уровня детализации)
- PBR-материалы для блоков

---

## 2. Архитектура и структура

### Дерево исходного кода

```
src/
├── main.rs                          — Точка входа, event loop, App
├── core.rs                          — Модульные декларации ядра
├── screens.rs                       — Модульные декларации экранов
├── core/
│   ├── config.rs                    — Глобальные атомики конфигурации
│   ├── renderer.rs                  — wgpu surface/device/queue
│   ├── screen.rs                    — Screen trait + ScreenAction
│   ├── screen_manager.rs           — Стековая машина состояний
│   ├── input_controller.rs         — Обработка ввода
│   ├── logging.rs                   — Инициализация логгера (fern)
│   ├── frame_timing.rs             — Атомики тайминга кадра
│   ├── user_settings.rs            — Пользовательские настройки (JSON)
│   ├── save_system.rs              — Сохранение/загрузка миров
│   ├── raycast.rs                   — DDA воксельный рейкаст
│   ├── physics.rs                   — rapier3d обёртка
│   ├── player.rs                    — Контроллер игрока
│   ├── player_model.rs             — Парсер Bedrock geometry
│   ├── block_model.rs              — Унифицированный парсер моделей
│   ├── model_messenger.rs          — Асинхронная загрузка моделей
│   ├── entity.rs                    — Система сущностей
│   ├── upload_worker.rs            — Фоновый поток упаковки мешей
│   ├── egui_manager.rs             — egui интеграция
│   ├── egui_fonts.rs               — Шрифты egui
│   ├── egui_style.rs               — Стиль egui
│   ├── lighting_legacy.rs          — Цикл дня/ночи, конфиг освещения
│   ├── lighting/mod.rs             — Реэкспорт lighting_legacy
│   ├── gameobjects/
│   │   ├── block.rs                — BlockDefinition + BlockRegistry (1249 строк)
│   │   ├── chunk.rs                — Chunk: воксельное хранилище + greedy mesh (1105 строк)
│   │   └── texture_atlas.rs        — Атлас текстур
│   ├── world_gen/
│   │   ├── world.rs                — World + WorldWorker (1497 строк)
│   │   ├── pipeline.rs             — BiomePipeline (743 строки)
│   │   ├── geography.rs            — Континентальная карта высот
│   │   ├── climate.rs              — Климатические зоны
│   │   ├── geology.rs              — Геологические регионы (Voronoi)
│   │   ├── stratification.rs       — Стратиграфия колонок
│   │   ├── biome_layer.rs          — BiomeLayer trait
│   │   ├── biome_registry.rs       — BiomeDictionary (32 биома, 931 строка)
│   │   ├── nature.rs               — Деревья и растительность (444 строки)
│   │   ├── layers/                 — Реализации слоёв (zoom, island, river, и т.д.)
│   │   └── ...
│   ├── vct/
│   │   ├── system.rs               — VCTSystem: GPU ресурсы (904 строки)
│   │   ├── voxel_volume.rs         — VoxelSnapshot + упаковка (187 строк)
│   │   ├── dynamic_lights.rs       — Point/Spot/Shadow AABB GPU структуры
│   │   └── mod.rs
│   ├── radiance_cascades/          — ⚠️ АРХИВ: не подключено к GameScreen
│   │   ├── system.rs, dispatch.rs, merge.rs, sampling.rs, types.rs, voxel_tex.rs
│   │   └── shaders/ (6 WGSL файлов)
│   ├── fluid/
│   │   └── mod.rs                  — FluidSimulator (728 строк)
│   └── water/
│       └── mod.rs                  — WaterSimulator (553 строки, устаревший)
├── screens/
│   ├── game_screen.rs              — Главный экран игры (2292 строки)
│   ├── game_3d_pipeline.rs         — GPU пайплайн, камера, фрустум (2114 строк)
│   ├── player_renderer.rs          — Скелетный рендеринг игрока
│   ├── entity_renderer.rs          — Рендерер сущностей (570 строк)
│   ├── block_model_renderer.rs     — Рендерер моделей блоков (559 строк)
│   ├── hand_block_renderer.rs      — Блок в руке (382 строки)
│   ├── sky_renderer.rs             — Небесный купол + спрайты (541 строка)
│   ├── inventory.rs                — Инвентарь + хотбар (1233 строки)
│   ├── main_menu.rs                — Главное меню (409 строк)
│   ├── settings.rs                 — Экран настроек
│   ├── debug_overlay.rs            — F3-оверлей
│   ├── profiler_overlay.rs         — Профилировщик кадра
│   ├── gpu_info.rs                 — Информация о GPU
│   ├── worldgen_visualizer_screen.rs — Визуализатор генерации
│   └── worldgen_visualizer.rs      — Отрисовка биомов
└── shaders/
    ├── pbr_lighting.wgsl           — PBR без GI
    ├── pbr_vct.wgsl                — PBR + VCT GI
    ├── vct_inject.wgsl             — VCT инжекция (compute)
    ├── vct_propagate.wgsl          — VCT распространение (compute)
    ├── water.wgsl                  — Вода
    ├── player_model.wgsl           — Модель игрока
    ├── entity_pbr.wgsl             — PBR для сущностей + VCT
    ├── outline.wgsl                — Wireframe выделения
    ├── debug_lines.wgsl            — Линии отладки
    ├── wireframe.wgsl              — Wireframe мешей
    └── vct_inject.wgsl / vct_propagate.wgsl
```

### Архитектурные документы

Проект содержит 5 подробных архитектурных документов:

| Документ | Содержание |
|---|---|
| [`ARCHITECTURE-ENTRY.md`](ARCHITECTURE-ENTRY.md) | Точка входа, event loop, экранная система |
| [`ARCHITECTURE-RENDERING.md`](ARCHITECTURE-RENDERING.md) | Рендер-пайплайн, шейдеры, камера, фрустум |
| [`ARCHITECTURE-GI.md`](ARCHITECTURE-GI.md) | VCT flood-fill + Radiance Cascades (архив) |
| [`ARCHITECTURE-WORLD.md`](ARCHITECTURE-WORLD.md) | Чанки, биомы, LOD, жидкости |
| [`ARCHITECTURE-CONFIG.md`](ARCHITECTURE-CONFIG.md) | Конфигурация, освещение, физика, игрок |
| [`ARCHITECTURE-INVARIANTS.md`](ARCHITECTURE-INVARIANTS.md) | Критические инварианты GPU |

---

## 3. Точка входа и экранная система

### Boot-последовательность

```
main() → EventLoop → Renderer::new() (async, tokio) → App::new()
  App::new(): ScreenManager::push(MainMenuScreen)
              EguiManager::new()
```

### Экранная система

Реализована как **стековая машина состояний** ([`src/core/screen_manager.rs`](src/core/screen_manager.rs)):

- `push()` → `load()` → `start()` → N× `update()`/`render()`/`build_ui()` → `unload()`
- `ScreenAction`: `Push(screen)`, `Switch(screen)`, `Pop`, `None`
- Только **верхний** экран получает ввод; все экраны в стеке получают `update()` + `render()`

### Экраны

| Экран | Файл | Назначение |
|---|---|---|
| `MainMenuScreen` | [`screens/main_menu.rs`](src/screens/main_menu.rs) | Главное меню, выбор/создание мира |
| `GameScreen` | [`screens/game_screen.rs`](src/screens/game_screen.rs) | 3D мир, игрок, HUD |
| `SettingsScreen` | [`screens/settings.rs`](src/screens/settings.rs) | Настройки (egui) |
| `DebugOverlay` | [`screens/debug_overlay.rs`](src/screens/debug_overlay.rs) | F3-оверлей (координаты, FPS) |
| `ProfilerOverlay` | [`screens/profiler_overlay.rs`](src/screens/profiler_overlay.rs) | Тайминг кадра |
| `GpuInfo` | [`screens/gpu_info.rs`](src/screens/gpu_info.rs) | Информация об адаптере GPU |
| `InventoryUI` | [`screens/inventory.rs`](src/screens/inventory.rs) | Хотбар + инвентарь + drag-and-drop |
| `WorldgenVisualizerScreen` | [`screens/worldgen_visualizer_screen.rs`](src/screens/worldgen_visualizer_screen.rs) | Отладка генерации |

### Захват курсора

`screen.wants_pointer_lock()` → `App::apply_cursor_lock()` — вызывает OS API только при изменении состояния. Сначала пробует `CursorGrabMode::Locked`, затем fallback на `Confined`.

---

## 4. Рендеринг и графический конвейер

### Порядок проходов (каждый кадр)

1. **Sky pass** — купол неба, спрайты солнца/луны ([`sky_renderer.rs`](src/screens/sky_renderer.rs))
2. **3D Geometry pass** — отсечённые по фрустуму меши чанков, PBR + VCT GI
3. **Player model pass** — скелетная модель (один draw call на видимую кость)
4. **Water pass** — полупрозрачные грани воды ([`water.wgsl`](src/shaders/water.wgsl))
5. **Block outline pass** — wireframe целевого блока ([`outline.wgsl`](src/shaders/outline.wgsl))
6. **egui UI overlay** — HUD поверх 3D сцены

### GPU бэкенд

**wgpu 29.0** — мульти-бэкенд: Vulkan / DX12 / Metal. Все шейдеры на WGSL. Запрос `POLYGON_MODE_LINE` для wireframe-отладки.

### Vertex3D — 52 байта (упакованный)

```
[ 0] position:  Float32x3  — world XYZ          (12 байт)
[12] normal:    Snorm8x4   — xyz + unused         (4 байта)
[16] color_ao:  Unorm8x4   — rgb albedo + AO      (4 байта)
[20] texcoord:  Float32x2  — UV                   (8 байт)
[28] material:  Unorm8x4   — roughness, metalness  (4 байта)
[32] emission:  Float32x4  — rgb + intensity       (16 байт)
[48] tangent:   Snorm8x4   — xyz + unused          (4 байта)
```

### Фрустум-каллинг

[`FrustumPlanes::from_view_projection(vp)`](src/screens/game_3d_pipeline.rs:27) → `intersects_aabb(min, max)` на чанк. Отсечённые чанки пропускают draw calls и деприоритизируются для перестройки мешей.

### VRAM бюджет

`CHUNK_MESH_VRAM_BUDGET = 4 ГБ`. При превышении самые дальние чанки вытесняются из GPU.

### Камера

Первое лицо. Поля: position, yaw, pitch, fov, aspect, near=0.1, far=dynamic. Методы: `view_matrix()`, `projection_matrix()`, `view_projection_matrix()`, `forward()`, `right()`, `rotate()`, `update_render_distance()`.

---

## 5. Глобальное освещение (GI)

### Активная система: VCT (Voxel Flood-Fill Propagation)

**Расположение:** [`src/core/vct/`](src/core/vct/)

Свет распространяется через 128³ воксельный объём, центрированный на камере, с итеративным flood-fill на GPU. Два 3D текстуры ping-pong до сходимости (64 итерации на кадр).

#### Per-frame dispatch

```
world_worker → VoxelSnapshot → pending_voxel_snapshot
vct.upload_volume(device, queue, snapshot, registry) → voxel_data + voxel_emission
vct.dispatch_gi(&mut encoder) → inject + 64× propagate
fragment shader → vct.bind_group() → group(1) in pbr_vct.wgsl
```

#### GPU ресурсы VCTSystem

| Текстура | Формат | Размер | Содержимое |
|---|---|---|---|
| `voxel_data` | RGBA8Unorm | 128³ | rgb=albedo, a=opacity |
| `voxel_emission` | RGBA8Unorm | 128³ | rgb=emission, a=intensity |
| `voxel_tint` | RGBA8Unorm | 128³ | rgb=tint (стекло), a=opacity |
| `radiance_a` | RGBA16Float | 128³ | ping buffer |
| `radiance_b` | RGBA16Float | 128³ | pong buffer |

#### Ключевые константы

```rust
PROPAGATION_STEPS = 64     // итераций на кадр; свет до ~64 блоков
PROPAGATION_DECAY = 0.97   // затухание на шаг; видимость ~20 блоков
GI_INTENSITY      = 2.0    // множитель яркости GI
MAX_SHADOW_STEPS  = 64.0   // шаги DDA тени в фрагментном шейдере
```

#### GPU uniform структуры (compile-time size-assert)

```rust
VCTFragParams    = 64 байта   // volume_origin, inv_size, gi_config, light_counts
InjectParams     = 32 байта   // inject compute
PropagateParams  = 32 байта   // propagation compute
```

#### Динамические источники света

| Структура | Размер | Назначение |
|---|---|---|
| `PointLightGPU` | 32 байта | Точечный свет (позиция + цвет) |
| `SpotLightGPU` | 64 байта | Прожектор (направление + углы) |
| `EntityShadowAABB` | 32 байта | Тень от сущностей (DDA) |

### Архивная система: Radiance Cascades

**Расположение:** [`src/core/radiance_cascades/`](src/core/radiance_cascades/)

Код компилируется, но **не подключён** к GameScreen. 5 уровней каскадов (C0–C4), 128³ воксельная текстура. При resurrect — обновить WGSL и compile-time asserts.

---

## 6. Мир и генерация ландшафта

### Обзор файлов

| Файл | Роль | Строк |
|---|---|---|
| [`world.rs`](src/core/world_gen/world.rs) | `World` + `WorldWorker` | 1497 |
| [`pipeline.rs`](src/core/world_gen/pipeline.rs) | `BiomePipeline` | 743 |
| [`geography.rs`](src/core/world_gen/geography.rs) | Континентальная карта высот | — |
| [`climate.rs`](src/core/world_gen/climate.rs) | Климатические зоны | — |
| [`geology.rs`](src/core/world_gen/geology.rs) | Геологические регионы (Voronoi) | — |
| [`stratification.rs`](src/core/world_gen/stratification.rs) | Стратиграфия колонок | — |
| [`biome_registry.rs`](src/core/world_gen/biome_registry.rs) | 32 биома, теговая система | 931 |
| [`nature.rs`](src/core/world_gen/nature.rs) | Деревья и растительность | 444 |

### Пайплайн генерации

```
BiomePipeline::new(seed)
  ├── GeographyLayer   — continent/erosion/peaks/islands noise
  ├── ClimateLayer     — latitude bands + boundary warp + humidity
  ├── GeologyLayer     — macro (2048 блоков) + micro (512 блоков) Voronoi
  ├── StratificationLayer — dirt/clay/claystone/granite толщина
  └── BiomeDictionary  — (ClimateType, GeoZone) → BiomeDefinition

BiomePipeline::sample_column(x, z, registry) → ColumnData
```

### 32 биома по умолчанию

- 5 вариантов океана (глубокий, обычный, замёрзший, тёплый, коралловый)
- 3 варианта пляжа
- 4 варианта равнин
- 4 варианта леса
- 2 варианта пустыни
- 2 варианта болота
- 4 варианта гор
- 2 варианта тундры
- 2 варианта джунглей
- 2 варианта рек

### Streaming чанков

- **Форма:** цилиндрическая — `dx² + dz² ≤ render_distance²`
- **Приоритет:** по направлению взгляда для генерации; по фрустуму для мешей
- **Лимиты:** `MAX_CHUNKS_PER_FRAME = 64`, `MAX_MESH_BUILDS_PER_FRAME = 128`
- **WorldWorker** — фоновый поток, mpsc канал (`WorldResult`)

### LOD (3 уровня)

| LOD | Разрешение | Порог |
|---|---|---|
| 0 | Полное (1:1) | dist ≤ `LOD_NEAR_BASE × multiplier` |
| 1 | 2× грубее | dist ≤ `LOD_FAR_BASE × multiplier` |
| 2 | 4× грубее | всё beyond LOD1 |

Greedy meshing объединяет смежные грани одного материала на каждом уровне LOD.

### Природа (NatureLayer)

14 типов биомов растительности: Plains, Forest, DenseForest, SparseForest, Swamp, Taiga, Savanna, JungleDense, Tundra, CherryBlossom, Mangrove, MushroomPlains, BambooGrove, DeadForest.  
Генерация деревьев: детерминированный хеш на колонку, только внутри чанка (без кросс-чанковой координации).

---

## 7. Система блоков

### BlockRegistry ([`block.rs`](src/core/gameobjects/block.rs) — 1249 строк)

JSON-базируемая система, загружаемая из [`assets/blocks/block_registry.json`](assets/blocks/block_registry.json).

- **ID 0** = Air (неявный)
- **ID 1..=255** = назначаются в порядке реестра
- **80 блоков** зарегистрировано (на момент аудита)

### Свойства блока

| Свойство | Описание |
|---|---|
| Текстуры | До 6 граней + 5 alpha-blended слоёв на грань |
| PBR | metallic, roughness, emission color + intensity |
| Прозрачность | opaque / transparent / translucent |
| Жидкость | flow_rate, gravity_rate, spread_distance |
| Биомный tint | foliage/water color overriding |
| Вращение | 24 направления (ROT_FACE_MAP) |
| Модель | Поддержка Bedrock/Java кастомных моделей |
| Тень | shadow_cubes для DDA voxel shadow |

### Система граней

6 граней: +X (East), -X (West), +Y (Top), -Y (Bottom), +Z (South), -Z (North).  
Приоритет текстур: direction → sides → all → block base.

### Зарегистрированные блоки (80)

**Категории:**
- **Породы (13):** andesite, basalt, deep_granite, gneiss, granite, kimberlite, lherzolite, limestone, marble, ophiolite, sandstone, schist, slate
- **Глина (12):** clay_andesite .. clay_slate
- **Глинистый камень (12):** claystone_andesite .. claystone_slate
- **Земля (12):** dirt_andesite .. dirt_slate
- **Трава (12):** grass_andesite .. grass_slate
- **Дерево (2):** oak_log, oak_leaves
- **Специальные (9):** water, lava, glowstone, spotlight, ice, snow, sand, gravel, glass, glass_red
- **Модельные (2):** test_model_be, redstone_furnace

---

## 8. Система чанков

### Chunk ([`chunk.rs`](src/core/gameobjects/chunk.rs) — 1105 строк)

| Параметр | Значение |
|---|---|
| Размер по умолчанию | 16 × 256 × 16 (колонка) |
| Хранение | `Vec<u8>` (flat), index = `x*SY*SZ + y*SZ + z` |
| Вращения | `Vec<u8>` (поблочные) |
| Уровни жидкости | `Vec<f32>` (поблочные) |
| Биомный tint | `biome_foliage_colors`, `biome_water_colors` (на XZ-колонку) |
| Ambient tint | `biome_ambient_tint` (усреднённый для чанка) |

### Greedy Meshing

Объединяет смежные грани одного материала. Работает на каждом уровне LOD отдельно.

---

## 9. Физика

### PhysicsWorld ([`physics.rs`](src/core/physics.rs) — 363 строки)

Обёртка над **rapier3d 0.32**:

| Компонент | Назначение |
|---|---|
| `PhysicsPipeline` | Симуляция |
| `RigidBodySet` / `ColliderSet` | Тела и коллайдеры |
| `chunk_colliders` | HashMap чанк → ColliderHandle |

- Гравитация: `(0, -20, 0)`
- Коллайдер игрока: капсульный (скользит по углам)
- Коллайдеры чанков: pre-built на фоновом потоке (`PhysicsChunkReady`)
- Void boundary: Y = -64

---

## 10. Игрок и камера

### PlayerController ([`player.rs`](src/core/player.rs) — 502 строки)

| Параметр | Значение |
|---|---|
| Полуширина | 0.3 (итого 0.6 м) |
| Полувысота (стоя) | 0.9 (итого 1.8 м) |
| Высота глаз | 1.62 м от подошвы |
| Скорость | 5.0 м/с (стоя), 2.5 (присед), 1.5 (лёжа) |
| Полёт | 12.0 м/с |
| Прыжок | 7.5 м/с |
| Spawn | (8, 250, 30) |

### Позиции

| Позиция | Полувысота | Скорость |
|---|---|---|
| Standing | 0.9 | 5.0 |
| Crouching (Shift) | 0.65 | 2.5 |
| Prone (Ctrl) | 0.35 | 1.5 |

### Хотбар

- 9 слотов
- Колесо мыши / цифры 1-9 для переключения
- Режимы размещения: Single, Three, Five, Seven, Full (1-блок, крест 3/5/7, полный куб)

### Raycast

DDA voxel raycast (Amanatides & Woo, 1987). Дальность: 8 блоков. Максимум шагов: 256. Возвращает `RaycastResult` с позицией блока + нормаль грани.

---

## 11. Система жидкостей

### FluidSimulator ([`fluid/mod.rs`](src/core/fluid/mod.rs) — 728 строк)

Обобщённая симуляция, поддерживающая несколько типов жидкостей:

| Параметр | Значение |
|---|---|
| MIN_LEVEL | 0.005 |
| FLOW_THRESHOLD | 0.01 |
| MAX_SUB_STEPS | 8 |

**Возможности:**
- Источники (level = 1.0, бесконечные) и потоки (level < 1.0)
- Пер-жидкостные параметры: flow_rate, gravity_rate, spread_distance
- Взаимодействия: вода + лава → камень
- Кросс-чанковое течение: snapshot-based (immutable read → mutable apply)

### WaterSimulator ([`water/mod.rs`](src/core/water/mod.rs) — 553 строки)

⚠️ **Устаревший** — только для воды. Суперседирован FluidSimulator. Код ещё присутствует.

---

## 12. Система сущностей

### EntityManager ([`entity.rs`](src/core/entity.rs) — 103 строки)

| Компонент | Описание |
|---|---|
| `EntityId` | Уникальный u64, не переиспользуется в сессии |
| `EntityTransform` | position + yaw + scale |
| `Entity` | id + transform + velocity + model_id |
| `EntityManager` | HashMap<EntityId, Entity>, spawn/despawn/get/iter |

Игрок — специальная сущность, управляемая `PlayerController`. Остальные сущности (NPC, дропнутые предметы) живут в `EntityManager`.

---

## 13. Модели блоков и Entity-рендеринг

### BlockModel ([`block_model.rs`](src/core/block_model.rs) — 1069 строк)

Унифицированный парсер для:
- **Bedrock Edition** (`minecraft:geometry` JSON)
- **Java Edition** (elements/faces JSON)

Поддержка мультитекстур, UV-ремаппинг, AABB для теней.

### ModelMessenger ([`model_messenger.rs`](src/core/model_messenger.rs) — 472 строки)

Централизованная система загрузки моделей:
1. `register_block_models()` — сканирует BlockRegistry
2. `process_pending()` — загружает модели каждый кадр
3. `drain_responses()` — возвращает загруженные модели + GPU текстуры

### EntityRenderer ([`entity_renderer.rs`](src/screens/entity_renderer.rs) — 570 строк)

- PBR + VCT GI + DDA voxel shadows
- Per-bone uniform = 240 байт (compile-time assert)
- Один draw call на видимую кость

### BlockModelRenderer ([`block_model_renderer.rs`](src/screens/block_model_renderer.rs) — 559 строк)

- Отдельный рендерер для статических блоков с 3D моделями
- CCw winding (отличается от EntityRenderer)
- Per-model texture atlas

### HandBlockRenderer ([`hand_block_renderer.rs`](src/screens/hand_block_renderer.rs) — 382 строки)

- Блок в руке игрока, привязан к правой кости предплечья
- PBR + VCT GI (использует `entity_pbr.wgsl`)
- Масштаб: 0.4 от полного блока

### PlayerRenderer ([`player_renderer.rs`](src/screens/player_renderer.rs))

Скелетная иерархия:
```
body → head (скрыт в 1-м лице)
     → leftArm / rightArm (counter-swing)
     → leftThigh → leftShin (walk cycle)
     → rightThigh → rightShin
```

Per-bone uniform = 192 байта. Поддержка First/Third person `ViewMode`.

---

## 14. UI и экраны

### egui интеграция

[`EguiManager`](src/core/egui_manager.rs) — полный цикл: input → tessellate → render. Кастомные шрифты и стиль.

### InventoryUI ([`inventory.rs`](src/screens/inventory.rs) — 1233 строки)

| Возможность | Статус |
|---|---|
| Хотбар (9 слотов) | ✅ |
| Полный инвентарь (9×4) | ✅ |
| Drag-and-drop | ✅ |
| Режим Creative/Survival | ✅ |
| Вкладки по категориям | ✅ |
| Подсветка текстур блоков | ✅ |

### MainMenuScreen ([`main_menu.rs`](src/screens/main_menu.rs) — 409 строк)

- Главное меню → Список миров → Создание мира
- Генерация seed из времени
- Подтверждение удаления мира

### Debug Overlay (F6)

Тогглы: wireframe, bbox_visible, bbox_physical, bbox_hitboxes, lighting_debug, shadow_vectors.

### Profiler Overlay

Тайминг: get_texture, update, egui_build, egui_render, submit_present (в микросекундах, через AtomicU64).

---

## 15. Система сохранений

### SaveSystem ([`save_system.rs`](src/core/save_system.rs) — 214 строк)

```
saves/
  user_settings.json
  <WorldName>/
    world.json        ← JSON метаданные (seed, player, version)
    chunks/
      1_0_2.bin       ← bincode SavedChunk
```

| Компонент | Формат | Описание |
|---|---|---|
| `WorldMetadata` | JSON (serde) | version, name, seed, created_at, last_played, player |
| `SavedPlayer` | JSON | position, fly_mode, stance, selected_slot, hotbar_ids |
| `SavedChunk` | bincode | cx, cy, cz, blocks, rotations, fluid_sparse |

**Сохранение чанков:** блоки и вращения — полные массивы; fluid_levels — разреженный формат `(flat_index, value_bits)`.

---

## 16. Конфигурация и настройки

### Глобальные атомики ([`config.rs`](src/core/config.rs) — 377 строк)

| Атомик | Тип | По умолчанию | Значение |
|---|---|---|---|
| `RENDER_DISTANCE` | AtomicI32 | 12 | Радиус прогрузки чанков |
| `LOD_MULTIPLIER` | AtomicU32 | 300 (×100) | Порог LOD |
| `VERTICAL_BELOW` | AtomicI32 | 0 | Чанков ниже камеры |
| `VERTICAL_ABOVE` | AtomicI32 | 0 | Чанков выше камеры |
| `BIOME_AMBIENT_TINT_ENABLED` | AtomicU32 | 1 | Биомный ambient tint |
| `SHADOW_SUN_ENABLED` | AtomicU32 | 1 | DDA тень от солнца |
| `SHADOW_BLOCK_ENABLED` | AtomicU32 | 1 | DDA тень от блоков |

### Конфигурационные файлы

| Файл | Назначение |
|---|---|
| [`assets/chunk_config.json`](assets/chunk_config.json) | Размеры чанка (16×256×16) |
| [`assets/world_gen_config.json`](assets/world_gen_config.json) | Генерация мира (geography, climate, geology, stratification, nature) |
| [`assets/lighting_config.json`](assets/lighting_config.json) | Цикл дня/ночи, солнце/луна, ambient, тени, SSAO |
| [`assets/blocks/block_registry.json`](assets/blocks/block_registry.json) | Список всех блоков |
| [`assets/blocks/*.json`](assets/blocks/) | Определения блоков (80 файлов) |
| [`assets/player_model.json`](assets/player_model.json) | Модель игрока (Bedrock geometry) |
| `saves/user_settings.json` | Пользовательские настройки |

### UserSettings ([`user_settings.rs`](src/core/user_settings.rs))

Сохраняются: render_distance, lod_multiplier, vertical_below/above, biome_ambient_tint, shadow_sun, shadow_block.

---

## 17. Ассеты и ресурсы

### Текстуры

```
assets/textures/
├── entity/player/          — player_skin.png
├── modelblocks/            — текстуры для модельных блоков (6 файлов)
├── simpleblocks/           — текстуры простых блоков
│   ├── common/             — grass, gravel, ice, sand, snow, water (10 файлов)
│   ├── clay/               — 12 вариантов
│   ├── claystone/          — 12 вариантов
│   ├── dirt/               — 12 вариантов
│   ├── grass/              — 12 вариантов
│   ├── rocks/              — 14 вариантов (вкл. transparent)
│   └── wood/               — oak_log, oak_log_top, oak_leaves
```

### Модели

```
assets/models/
├── blocks/
│   ├── redstone_furnace_je.json   — Java Edition модель
│   └── test_model_be.json         — Bedrock Edition модель
└── entity/                        — (пока пусто)
```

### TextureAtlas

[`TextureAtlas`](src/core/gameobjects/texture_atlas.rs) компонует все текстуры граней в единый GPU атлас. Каждый блок ссылается на область в `TextureAtlasLayout`.

---

## 18. Шейдеры

### Активные шейдеры ([`src/shaders/`](src/shaders/))

| Файл | Назначение |
|---|---|
| `pbr_lighting.wgsl` | PBR + направленное солнце/луна (без GI) |
| `pbr_vct.wgsl` | PBR + VCT GI sampling через group(1) |
| `vct_inject.wgsl` | VCT inject compute: emission → radiance_a |
| `vct_propagate.wgsl` | VCT propagation compute: ping-pong A↔B |
| `water.wgsl` | Полупрозрачная поверхность воды |
| `player_model.wgsl` | Per-bone скелетная модель игрока |
| `entity_pbr.wgsl` | PBR для сущностей + VCT GI + DDA voxel shadows |
| `outline.wgsl` | Wireframe выделения блока |
| `debug_lines.wgsl` | Отладочные линии |
| `wireframe.wgsl` | Wireframe мешей |

### Архивные шейдеры RC ([`src/core/radiance_cascades/shaders/`](src/core/radiance_cascades/shaders/))

`ray_march.wgsl`, `merge.wgsl`, `emission_inject.wgsl`, `outline.wgsl`, `pbr_lighting.wgsl`, `volumetric_lights.wgsl` — компилируются, но не подключены.

---

## 19. Многопоточность и производительность

### Потоки

| Поток | Назначение |
|---|---|
| Main | Event loop, рендеринг, UI |
| WorldWorker | Генерация чанков, физика, VoxelSnapshot |
| gpu-staging | Упаковка мешей (UploadWorker) |
| rayon pool | Параллельная генерация чанков |

### Синхронизация

- `mpsc` каналы: WorldWorker → GameScreen, UploadWorker
- `AtomicI32/AtomicU32/AtomicU64`: конфигурация, тайминг
- `OnceLock`: размеры чанков, конфиги
- `Arc<BiomePipeline>`: разделённый между rayon workers
- `RwLock`: (где необходимо)

### Frame Timing

Атомики в [`frame_timing.rs`](src/core/frame_timing.rs):
- `GET_TEXTURE_US` — время ожидания GPU
- `UPDATE_US` — update + post_process
- `EGUI_BUILD_US` — egui build
- `EGUI_RENDER_US` — egui render
- `SUBMIT_PRESENT_US` — submit + present

### UploadWorker

Фоновый поток упаковки: `UploadJob` → `UploadPacked` (bytemuck zero-cost cast). Лимит: `MAX_GPU_UPLOADS_PER_FRAME = 32`.

---

## 20. Логирование

### Система

- **Логгер:** fern + chrono (файловый + консольный)
- **Макросы:** `debug_log!`, `flow_debug_log!`, `ext_debug_log!`
- **Формат:** `[Class][Method] — message`
- **Язык:** английский (все логи)
- **Уровни:** IS_DEBUG, IS_FLOW_DEBUG, IS_EXT_DEBUG

---

## 21. Зависимости

| Крейт | Версия | Назначение |
|---|---|---|
| `wgpu` | 29.0.0 | Графика (Vulkan/DX12/Metal) |
| `winit` | 0.30 | Окно и ввод |
| `image` | 0.24 | Загрузка текстур |
| `glam` | 0.24 | Математика (Vec3, Mat4) |
| `noise` | 0.8 | Шум Перлина для генерации |
| `rapier3d` | 0.32.0 | Физика |
| `hecs` | 0.10 | ECS (указан, но не активно используется) |
| `egui` | 0.34 | Immediate-mode UI |
| `egui-wgpu` | 0.34 | egui ↔ wgpu |
| `egui-winit` | 0.34 | egui ↔ winit |
| `serde` | 1.0 | Сериализация |
| `serde_json` | 1.0 | JSON |
| `bincode` | 2.0.1 | Бинарная сериализация (чанки) |
| `rayon` | 1.8 | Параллелизм данных |
| `tokio` | 1.0 | Async runtime |
| `bytemuck` | 1.21 | GPU struct casting |
| `fern` | 0.7.1 | Логирование |
| `chrono` | 0.4.44 | Время |
| `log` | 0.4 | Facade логирования |
| `env_logger` | 0.11 | (Указан, но не используется — fern вместо него) |

---

## 22. Оценка качества кода

### Сильные стороны

| Аспект | Оценка |
|---|---|
| **Архитектурная документация** | ⭐⭐⭐⭐⭐ — 5 подробных ARCHITECTURE-*.md файлов |
| **Комментарии в коде** | ⭐⭐⭐⭐⭐ — Каждый файл с заголовком, секции разделены |
| **GPU safety** | ⭐⭐⭐⭐⭐ — Compile-time size asserts для всех GPU структур |
| **Модульность** | ⭐⭐⭐⭐ — Чёткое разделение: core / screens / shaders |
| **Потокобезопасность** | ⭐⭐⭐⭐ — Atomics, mpsc, Arc, OnceLock |
| **Расширяемость блоков** | ⭐⭐⭐⭐⭐ — JSON-базируемая система, 5 слоёв текстур |
| **PBR материалы** | ⭐⭐⭐⭐ — Full PBR: roughness, metalness, emission, AO |
| **LOD система** | ⭐⭐⭐⭐ — 3 уровня, greedy meshing, настраиваемые пороги |

### Области для улучшения

| Аспект | Оценка | Комментарий |
|---|---|---|
| **Тесты** | ⭐ | Нет модульных тестов (`cargo test` ничего не найдёт) |
| **Обработка ошибок** | ⭐⭐⭐ | Много `.expect()` и `unwrap()` |
| **Дублирование кода** | ⭐⭐⭐ | WaterSimulator vs FluidSimulator, дублирование uniform структур |
| **ECS использование** | ⭐ | hecs указан в зависимостях, но не используется активно |
| **Мёртвый код** | ⭐⭐⭐ | Radiance Cascades (компилируется, не подключено), WaterSimulator |

---

## 23. Выявленные проблемы и риски

### 🔴 Критические

1. **Нет модульных тестов.** `cargo test` не найдет ни одного теста. Критические системы (raycast, fluid, chunk meshing) не покрыты тестами.

2. **`unsafe impl Send for PhysicsChunkReady`** ([`physics.rs:51`](src/core/physics.rs:51)). Хотя комментарий объясняет безопасность, это потенциальный источник UB при изменении rapier3d.

3. **`std::mem::transmute` для Surface** ([`renderer.rs:47`](src/core/renderer.rs:47)). `transmute` для продления времени жизни `Surface` — рискованная практика, может привести к use-after-free.

### 🟡 Средние

4. **Дублирование FluidSimulator / WaterSimulator.** Обе системы существуют одновременно. `WaterSimulator` — устаревший, но код остаётся в проекте (553 строки).

5. **Radiance Cascades — мёртвый код.** 6 файлов + 6 шейдеров компилируются, но не используются. Увеличивает время компиляции.

6. **`hecs` не используется.** Зависимость указана в Cargo.toml, но в коде не найдено использование. Лишняя зависимость.

7. **`env_logger` не используется.** Указан в зависимостях, но логирование через `fern`. Лишняя зависимость.

8. **Отсутствие error handling для GPU.** Многие GPU операции используют `.expect()` — при ошибке паника без восстановления.

9. **Блоки ограничены 255 типами.** `Vec<u8>` в чанке ограничивает количество блоков до 255 (0 = air).

10. **Нет сети / мультиплеера.** tokio указан в зависимостях, но нет сетевого кода.

### 🟢 Низкие

11. **Жёсткие пути к ассетам.** Все пути захардкожены (`"assets/blocks/..."`, `"saves/..."`). Не работает при запуске из другой директории.

12. **Нет локализации.** Вся UI на английском.

13. **SSAO и Shadow Map не реализованы.** Поля в `lighting_config.json` присутствуют, но GPU-реализации нет.

14. **Нет звука.** Отсутствует аудиосистема.

15. **Нет анимаций блоков.** Нет системы анимированных текстур (кроме воды).

---

## 24. Рекомендации по развитию

### Приоритет 1 — Качество и стабильность

| # | Рекомендация | Обоснование |
|---|---|---|
| 1 | Добавить модульные тесты для критических систем | raycast, fluid, chunk meshing, block registry |
| 2 | Удалить или feature-gate Radiance Cascades | Сократить время компиляции |
| 3 | Удалить WaterSimulator (заменён на FluidSimulator) | Устранение дублирования |
| 4 | Удалить `hecs` и `env_logger` из зависимостей | Лишние зависимости |
| 5 | Заменить `.expect()` на `?` или `anyhow` в GPU коде | Улучшить обработку ошибок |

### Приоритет 2 — Функциональность

| # | Рекомендация | Обоснование |
|---|---|---|
| 6 | Расширить ID блоков до u16 | Больше типов блоков |
| 7 | Добавить анимированные текстуры | Текстуры воды, лавы |
| 8 | Реализовать SSAO | Поля в конфиге уже есть |
| 9 | Добавить систему звуков | Полноценный геймплей |
| 10 | Относительные пути к ассетам | Запуск из любой директории |

### Приоритет 3 — Архитектура

| # | Рекомендация | Обоснование |
|---|---|---|
| 11 | Использовать `anyhow` / `thiserror` для ошибок | Единая система обработки |
| 12 | Добавить CI/CD (GitHub Actions) | Автоматическая сборка и тесты |
| 13 | Feature flags для экспериментальных систем | Условная компиляция RC и др. |
| 14 | ECS (hecs или замена) для сущностей | Масштабируемость |
| 15 | Network layer (tokio уже есть) | Мультиплеер |

---

## 25. Сводная таблица возможностей

| Подсистема | Статус | Зрелость | Файлов | Строк кода (примерно) |
|---|---|---|---|---|
| **Графика (wgpu)** | ✅ Активна | Высокая | 3 | ~2 800 |
| **Рендер-пайплайн** | ✅ Активна | Высокая | 6 | ~6 500 |
| **VCT GI** | ✅ Активна | Высокая | 4 | ~1 200 |
| **Radiance Cascades** | ⚠️ Архив | Средняя | 8 | ~2 000 |
| **Генерация мира** | ✅ Активна | Высокая | 15+ | ~4 500 |
| **Биомы (32)** | ✅ Активна | Высокая | 3 | ~1 400 |
| **Система блоков (80)** | ✅ Активна | Высокая | 1 | ~1 250 |
| **Система чанков** | ✅ Активна | Высокая | 1 | ~1 100 |
| **Физика (rapier3d)** | ✅ Активна | Средняя | 1 | ~360 |
| **Игрок** | ✅ Активна | Высокая | 1 | ~500 |
| **Жидкости (FluidSim)** | ✅ Активна | Высокая | 1 | ~730 |
| **Жидкости (WaterSim)** | ⚠️ Устарела | — | 1 | ~550 |
| **Сущности** | ✅ Активна | Базовая | 1 | ~100 |
| **Модели блоков** | ✅ Активна | Высокая | 2 | ~1 500 |
| **Entity rendering** | ✅ Активна | Высокая | 3 | ~1 500 |
| **UI (egui)** | ✅ Активна | Высокая | 5 | ~2 800 |
| **Инвентарь** | ✅ Активна | Высокая | 1 | ~1 200 |
| **Сохранения** | ✅ Активна | Средняя | 1 | ~210 |
| **Конфигурация** | ✅ Активна | Высокая | 2 | ~470 |
| **Шейдеры (10 активных)** | ✅ Активна | Высокая | 10 | — |
| **Логирование** | ✅ Активна | Средняя | 1 | — |
| **Многопоточность** | ✅ Активна | Высокая | 3 | — |
| **Тесты** | ❌ Отсутствуют | — | 0 | 0 |
| **Звук** | ❌ Отсутствует | — | 0 | 0 |
| **Сеть** | ❌ Отсутствует | — | 0 | 0 |

### Общая статистика

| Метрика | Значение |
|---|---|
| **Исходных файлов (src/)** | ~65 |
| **Примерное количество строк кода** | ~25 000+ |
| **Зарегистрированных блоков** | 80 |
| **Биомов** | 32 |
| **Шейдеров (активных)** | 10 |
| **Экранов** | 8 |
| **Зависимостей** | 18 |
| **Архитектурных документов** | 6 |
| **Тестов** | 0 |

---

## Заключение

**QubePixel** — это амбициозный воксельный движок на Rust с впечатляющим набором возможностей для проекта версии 0.1.0. Особенно выделяются:

- **VCT Global Illumination** — полноценное GI через 128³ flood-fill propagation на GPU
- **Биомная система** — 32 биома с теговой классификацией и многослойной генерацией
- **PBR материалы** — full Cook-Torrance с roughness, metalness, emission
- **Расширяемая система блоков** — JSON-базируемая, с мультитекстурами и слоями

**Ключевые области для улучшения:** отсутствие тестов, наличие мёртвого кода (RC, WaterSim), лишние зависимости, и ограничение в 255 типов блоков. Рекомендуется приоритизировать добавление тестов и очистку кодовой базы.

---

*Аудит подготовлен автоматически на основе анализа всех исходных файлов проекта.*
