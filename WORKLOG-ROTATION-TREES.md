# Worklog: Block Rotation System & Tree Generation (Dialog Session)

**Date**: May 2026  
**Status**: ✅ COMPLETED  
**Branch**: master

---

## Overview

Реализована полная система ориентации блоков при размещении + генерация деревьев в мире. Система позволяет блокам иметь 14 различных ориентаций, включая горизонтальные, стеновые и потолочные варианты.

---

## Part 1: Block Rotation System

### Architecture

**Core Components:**

1. **PlacementMode enum** (`src/core/gameobjects/block.rs`)
   - `Fixed`: блок всегда в одном положении
   - `AttachHorizontal`: прилипает к N/S/E/W стенам
   - `AttachFull`: прилипает к любой грани куба
   - `AttachFullRotatable`: любая грань + вращение на верх/низ
   - `LookHorizontal`: 4 ориентации по горизонтальному взгляду (N/S/E/W)
   - `LookFull`: 6 ориентаций по полному взгляду

2. **ROT_FACE_MAP** (`src/core/gameobjects/block.rs`)
   - Таблица `[[u8; 6]; 14]` — face permutation для каждого значения поворота
   - `perm[rotation][world_face] = block_local_face`
   - Face indices: 0=East, 1=West, 2=Top, 3=Bottom, 4=South, 5=North
   - Rotation values (14 штук):
     - 0 = default (south-facing)
     - 1 = facing north (180° Y rotation)
     - 2 = facing south (same as default)
     - 3 = facing east (+90° Y rotation) ⚠️ **WAS SWAPPED**
     - 4 = facing west (-90° Y rotation) ⚠️ **WAS SWAPPED**
     - 5 = upside-down/ceiling
     - 6-9 = wall rotations (N/S/E/W)
     - 10-13 = ceiling + compass directions

3. **Chunk Rotation Storage** (`src/core/gameobjects/chunk.rs`)
   - Добавлено поле: `pub(crate) rotations: Vec<u8>` (параллельно `blocks`)
   - Методы:
     - `set_gen_with_rotation(x, y, z, block_id, rotation)` — без dirty flag
     - `set_with_rotation(x, y, z, block_id, rotation)` — с dirty flag
     - `get_rotation(x, y, z) -> u8`

4. **Mesh Builder Face Remapping** (`src/core/gameobjects/chunk.rs::build_mesh_lod`)
   - Для каждой грани куба: `local_face = remap_face_for_rotation(rot, world_face)`
   - Функция `remap_face_for_rotation` применяет ROT_FACE_MAP
   - Все lookup текстур/цветов теперь переходят через remapping

5. **Block Placement & Rotation Computation** (`src/core/player.rs`)
   - `place_block(target, physics, block_id, look_dir, placement_mode, default_rotation) -> Option<BlockOp>`
   - `compute_block_rotation(placement, default_rotation, surface_normal, look_dir) -> u8`
   - `horizontal_look_rotation(look: Vec3) -> u8` (1-4 для N/S/E/W)
   - `ceiling_look_rotation(look: Vec3) -> u8` (10-13)

6. **World Integration** (`src/core/world_gen/world.rs`)
   - `BlockOp::Place { x, y, z, block_id, rotation }`
   - `World::place_block(wx, wy, wz, block_id, rotation)` вызывает `chunk.set_with_rotation()`

7. **Game Screen Integration** (`src/screens/game_screen.rs`)
   - При размещении блока: извлекает `placement_mode` и `default_rotation` из `BlockRegistry`
   - Передаёт `look` направление в `player.place_block()`
   - Создаёт `BlockOp::Place { rotation }`

### Bug Fixes During Implementation

#### Bug #1: ROT_FACE_MAP rows 3 and 4 were swapped ✅ FIXED
**Problem**: 
- Current row 3 had значения для facing west (должны быть в row 4)
- Current row 4 имел значения для facing east (должны быть в row 3)
- Результат: при rotation=3 (facing east) блок показывал свою спину с восточной стороны

**Root Cause**: При создании таблицы были перепутаны значения строк

**Fix**: 
```rust
// Before (WRONG):
[5, 4, 2, 3, 0, 1], // 3: facing east
[4, 5, 2, 3, 1, 0], // 4: facing west

// After (CORRECT):
[4, 5, 2, 3, 1, 0], // 3: facing east  (+90° Y: local+Z→world+X)
[5, 4, 2, 3, 0, 1], // 4: facing west  (-90° Y: local+Z→world-X)
```

---

## Part 2: Tree Generation System

### Architecture

**Core Components:**

1. **NatureLayer** (`src/core/world_gen/nature.rs` - NEW FILE)
   - Содержит: `seed: u64, oak_log_id: u8, oak_leaves_id: u8`
   - Метод: `decorate_chunk(&mut chunk, columns, base_y)` — добавляет деревья в биом-слой

2. **NatureBiome enum** (14 вариантов)
   - Forest, DenseForest, SparseForest, Swamp, Taiga, Savanna
   - JungleDense, Tundra, CherryBlossom, Mangrove
   - MushroomPlains, BambooGrove, DeadForest, Plains
   - Для каждого: своя плотность деревьев (от 0.0 до 0.8)

3. **assign_nature_biome()** 
   - Входные данные: `climate_type: ClimateType, geo: GeoInfo, humidity: f32`
   - Выход: `NatureBiome` на основе климата и влажности

4. **TreeKind enum**
   - Oak, Pine, Savanna, Swamp, Bamboo, Dead
   - Определяет форму кроны и высоту ствола

5. **Tree Placement & Generation**
   - `place_tree(chunk, chunk_x, chunk_y, chunk_z, bx, by, bz, tree_kind)` — рекурсивно размещает ствол и крону
   - `place_sphere_canopy(...)` — шаровая крона (Oak, Pine)
   - `place_disk(...)` — дисковая крона (специальные деревья)
   - MAX_CANOPY_R = 4 блоков — protection от пересечения границ чанка

### Integration

1. **Module registration** (`src/core/world_gen.rs`)
   - `pub mod nature;`

2. **World initialization** (`src/core/world_gen/world.rs`)
   - `let nature_layer = NatureLayer::new(world_seed, &registry);`
   - Сохранено в `World::nature_layer`

3. **Chunk generation** (`src/core/world_gen/world.rs::generate_chunk_fn`)
   - После заполнения базового ландшафта: `nature.decorate_chunk(&mut chunk, &columns, base_y);`
   - Вызывается в rayon parallel блоке (NatureLayer thread-safe)

### Block Definitions

**oak_log.json** (NEW)
```json
{
  "id": "wood/oak_log",
  "display_name": "Oak Log",
  "color": [0.45, 0.30, 0.15],
  "material": { "albedo": [0.45, 0.30, 0.15], "roughness": 0.95 },
  "solid": true,
  "placement_mode": "look_horizontal",
  "faces": {
    "top": { "texture": "wood/oak_log_top" },
    "bottom": { "texture": "wood/oak_log_top" },
    "sides": { "texture": "wood/oak_log_side" }
  }
}
```

**oak_leaves.json** (NEW)
```json
{
  "id": "wood/oak_leaves",
  "display_name": "Oak Leaves",
  "color": [0.22, 0.52, 0.14],
  "material": { "albedo": [0.22, 0.52, 0.14], "roughness": 0.85 },
  "solid": true,
  "transparent": false,
  "biome_tint": true,
  "placement_mode": "fixed",
  "faces": { "all": { "texture": "wood/oak_leaves" } }
}
```

**block_registry.json** (UPDATED)
- Добавлены в конец: `"wood/oak_log"`, `"wood/oak_leaves"`

---

## Part 3: Model Block Rendering with Rotation

### Architecture

**BlockModelRenderer** (`src/screens/block_model_renderer.rs`)

1. **BlockInstanceGpu struct** - добавлено поле:
   ```rust
   pub rotation: u8,  // Block rotation value (1-4 for horizontal look)
   ```

2. **spawn() method** - сигнатура:
   ```rust
   pub fn spawn(&mut self, device: &wgpu::Device, pos: (i32, i32, i32), model_id: &str, rotation: u8)
   ```

3. **Model matrix with Y rotation** (in `render()` method):
   ```rust
   let y_angle = match inst.rotation {
       1 => std::f32::consts::PI,           // north: 180°
       2 => 0.0,                            // south: 0° (default)
       3 => std::f32::consts::FRAC_PI_2,    // east: +90°
       4 => -std::f32::consts::FRAC_PI_2,   // west: -90°
       _ => 0.0,
   };
   let model_mat = (Mat4::from_translation(world_pos) * Mat4::from_rotation_y(y_angle)).to_cols_array();
   ```

### Integration with World

**WorldWorker::build_meshes()** 
- `model_block_positions` теперь содержит: `Vec<(i32, i32, i32, u8, u8)>` 
  - формат: `(wx, wy, wz, block_id, rotation)`
- При сканировании dirty chunks: `chunk.get_rotation(x, y, z)` читается для каждого блока-модели

**GameScreen::on_update()**
- При распаковке `model_block_positions`:
  ```rust
  for &(wx, wy, wz, bid, rotation) in &result.model_block_positions {
      bmr.spawn(device, (wx, wy, wz), &model_name, rotation);
  }
  ```

---

## Part 4: Collision Box Rotation

### Issue
При размещении блока-модели с поворотом его AABB (ограничивающая коробка для физики) оставалась неповёрнутой. Это вызывает проблемы при взаимодействии с моделями, которые имеют асимметричную форму.

### Solution: AABB Rotation Function

**rotate_aabb_y()** (`src/core/world_gen/world.rs`)
```rust
fn rotate_aabb_y(rot: u8, mn: [f32; 3], mx: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    match rot {
        1 => ([1.0-mx[0], mn[1], 1.0-mx[2]], [1.0-mn[0], mx[1], 1.0-mn[2]]),  // 180° (north)
        3 => ([mn[2], mn[1], 1.0-mx[0]], [mx[2], mx[1], 1.0-mn[0]]),           // +90° (east)
        4 => ([1.0-mx[2], mn[1], mn[0]], [1.0-mn[2], mx[1], mx[0]]),           // -90° (west)
        _ => (mn, mx),  // rot 0, 2, or other
    }
}
```

**Integration in physics builder** (`src/core/world_gen/world.rs`)
```rust
if let Some(&(mn_raw, mx_raw)) = aabbs_phys.get(&bid) {
    let rot = chunk.get_rotation(x, y, z);
    let (mn, mx) = rotate_aabb_y(rot, mn_raw, mx_raw);
    // ... rest of AABB cuboid creation with rotated bounds
}
```

**Rotation Formulas:**
- **Rotation 1 (north, 180°)**: `(x,y,z) → (1-x, y, 1-z)`
- **Rotation 3 (east, +90°)**: `(x,y,z) → (z, y, 1-x)`
- **Rotation 4 (west, -90°)**: `(x,y,z) → (1-z, y, x)`

Вращение применяется вокруг центра блока в XZ-плоскости (0.5, -, 0.5).

---

## Known Limitations & Future Work

### Current Limitations

1. **oak_log with look_horizontal doesn't rotate visually**
   - Причина: `look_horizontal` = Y-axis rotation. Для брёвен это значит, что кольца остаются на верху/низу
   - Чтобы брёвна лежали на боку: нужен режим `look_axis` (осевое ориентирование)
   - Текущее поведение: логически правильное для Y-ротации, просто не видимо для блоков с одинаковой текстурой на боках

2. **Wall rotations (6-13) for models don't apply pitch**
   - Model rotation применяет только Y-axis rotation
   - Для wall/ceiling ориентаций нужна X/Z ось rotation
   - Сейчас wall блоки используют только Y-ротацию (или вообще без неё)

3. **No texture generation for wood textures**
   - oak_log_top, oak_log_side, oak_leaves — текстуры не генерируются
   - Блоки будут отрисовываться с white vertex colors (без текстур)
   - Пользователь может добавить PNG в `assets/textures/`

### Possible Future Implementations

#### `look_axis` Mode (высокий приоритет)
```rust
PlacementMode::LookAxis
```
- Определяет ось (X, Y, Z) по направлению взгляда
- Для брёвен: кольца на торцах той оси
- ROT_FACE_MAP: свопит боковые грани с верхом/низом для горизонтальных осей
- Model rotation: Euler angles around appropriate axis

#### `rotation_with_pitch` Mode
- Full 3D orientation (yaw + pitch + roll)
- Для моделей: `Mat4::from_rotation_zyx(roll, pitch, yaw)`
- Более сложная система ввода (требует трёхмерного выбора направления)

#### `slot_attachable` Mode
- Блок занимает "слот" на другом блоке (рельсы на краю платформы)
- Требует системы определения слотов на других блоках

#### Per-biome Model Variation
- Разные модели для одного блока в зависимости от биома
- Например: różные виды деревьев в разных биомах

---

## Files Changed

### Core Implementation
- ✅ `src/core/gameobjects/block.rs` — PlacementMode, ROT_FACE_MAP, remap_face_for_rotation
- ✅ `src/core/gameobjects/chunk.rs` — rotations array, set_with_rotation, get_rotation, mesh builder update
- ✅ `src/core/player.rs` — place_block, compute_block_rotation, horizontal_look_rotation, ceiling_look_rotation
- ✅ `src/core/world_gen/world.rs` — BlockOp with rotation, place_block, generate_chunk_fn, model_block_positions with rotation, rotate_aabb_y, physics builder integration
- ✅ `src/screens/game_screen.rs` — place_block call with look direction, model spawn with rotation

### Tree Generation
- ✅ `src/core/world_gen/nature.rs` — NEW FILE, NatureLayer, NatureBiome, tree generation logic
- ✅ `src/core/world_gen.rs` — pub mod nature
- ✅ `assets/blocks/wood/oak_log.json` — NEW FILE
- ✅ `assets/blocks/wood/oak_leaves.json` — NEW FILE
- ✅ `assets/blocks/block_registry.json` — added oak_log, oak_leaves

### Model Rendering
- ✅ `src/screens/block_model_renderer.rs` — BlockInstanceGpu.rotation, spawn with rotation, model matrix Y rotation

### Documentation
- ✅ `BLOCKS-CONFIG-GUIDE.md` — NEW FILE, comprehensive block configuration guide

---

## Testing Checklist

- [x] Code compiles (`cargo build --release`)
- [x] ROT_FACE_MAP fix verified (east/west correctly map to front face)
- [x] Chunk rotations stored and read correctly
- [x] Mesh builder applies face remapping
- [x] Model blocks rotate with Y-axis
- [x] AABB rotates for physics
- [x] Tree generation produces valid structure
- [x] Nature biome assignment works
- [ ] Visual test in game: place furnace, verify correct front faces
- [ ] Visual test in game: place trees, verify structure
- [ ] Visual test in game: oak_log placement (should show that Y-rotation is all that happens)

---

## Build Status

✅ **Latest build**: `Finished release profile [optimized]`

No compilation errors. Ready for testing in-game.

---

## Next Steps (If Continuing)

1. **Texture Implementation**
   - Generate or provide PNG files for `oak_log_top`, `oak_log_side`, `oak_leaves`
   - Place in `assets/textures/`

2. **Test Rotation in Game**
   - Place blocks with different `placement_mode` values
   - Verify visual rotation matches placement direction
   - Check collision boxes align with visual model

3. **Implement look_axis for Logs**
   - Add `PlacementMode::LookAxis` variant
   - Compute axis from look direction
   - Create appropriate ROT_FACE_MAP rows for axis rotation
   - Update `compute_block_rotation` to handle axis mode

4. **Add Wall/Ceiling Model Rotation**
   - Support X/Z axis rotations in `block_model_renderer.rs`
   - Map rotations 5-13 to appropriate Euler angles

5. **Nature Generation Tuning**
   - Adjust tree density per biome
   - Add more tree varieties (Pine, Birch, Spruce, etc.)
   - Implement tree variation (size, canopy shape)

---

## References

- **Coordinate System**: +X=East, +Y=Up, +Z=South
- **Rotation Pivot**: Block center in XZ plane: (0.5, -, 0.5)
- **Face Indices**: 0=E, 1=W, 2=Top, 3=Bottom, 4=South, 5=North
- **Model Convention**: Local +Z = "front" (south by default)
- **ROT_FACE_MAP**: Semantic = world face → local face index for texture lookup
