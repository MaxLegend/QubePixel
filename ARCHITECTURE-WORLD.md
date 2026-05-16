# World, Terrain & Simulation

## Files overview

| File | Role |
|---|---|
| `src/core/world_gen/world.rs` | `World` struct + `WorldWorker` thread; chunk streaming, LOD, physics |
| `src/core/world_gen/pipeline.rs` | `BiomePipeline`: assembles all terrain layers, thread-safe |
| `src/core/world_gen/geography.rs` | Continental height map (continent/erosion/peaks/islands noise) |
| `src/core/world_gen/climate.rs` | Climate zones (`ClimateType`), band + boundary noise |
| `src/core/world_gen/geology.rs` | Macro/micro geological regions (Voronoi cells) |
| `src/core/world_gen/stratification.rs` | Column stratigraphy: dirt/clay/claystone/granite layers |
| `src/core/world_gen/biome_registry.rs` | `BiomeDictionary`, `BiomeDefinition`, `ClimateTag` |
| `src/core/world_gen/biome_layer.rs` | `BiomeLayer` trait, `GenContext`, zoom/smooth layers |
| `src/core/world_gen/layers/` | Layer implementations: zoom, smooth, Voronoi, island, river init |
| `src/core/gameobjects/chunk.rs` | 16³ (or config-sized) voxel storage, greedy mesh per LOD |
| `src/core/gameobjects/block.rs` | JSON block registry, PBR material properties |
| `src/core/gameobjects/texture_atlas.rs` | Atlas composition from PNG files in `assets/` |
| `src/core/upload_worker.rs` | Off-thread mesh building (mpsc channels + rayon thread pool) |
| `src/core/fluid/mod.rs` | `FluidSimulator`: generalized fluid (water/lava, interactions) |
| `src/core/water/mod.rs` | `WaterSimulator`: simple water-only CA (older, superseded by fluid) |
| `src/core/player_model.rs` | Minecraft Bedrock Edition geometry format parser |

## Terrain Generation Pipeline

```
BiomePipeline::new(seed)
  │
  ├── GeographyLayer   — continent_scale / erosion_scale / peaks_scale noise stacks
  ├── ClimateLayer     — latitude bands + boundary warp + humidity noise
  ├── GeologyLayer     — macro cells (2048 blocks) + micro cells (512 blocks)
  ├── StratificationLayer — per-column dirt/clay/claystone/granite thicknesses
  └── BiomeDictionary  — maps (ClimateType, GeoZone) → BiomeDefinition

BiomePipeline::sample_column(x, z, registry) → ColumnData
  — contains: biome_id, surface_height, stratigraphy, geo_info, climate_info
```

`BiomePipeline` is constructed once per `World` and shared (via `Arc`) across rayon worker threads.

## Chunk Streaming (`world_gen/world.rs::World`)

- **Shape**: cylindrical — loads chunks where `dx² + dz² ≤ render_distance²`
- **Priority**: view-direction priority for generation; frustum-priority for mesh building
- **Per-frame limits**: `MAX_CHUNKS_PER_FRAME = 64` generated, `MAX_MESH_BUILDS_PER_FRAME = 128`
- **Vertical**: `VERTICAL_BELOW` / `VERTICAL_ABOVE` config atomics (default 0 — one vertical layer)

WorldWorker runs in a background thread, sends results to GameScreen via mpsc channel (`WorldResult`). Block operations (place/break) sent back as `BlockOp`.

## LOD (Level of Detail)

Three LOD levels selected by Chebyshev distance from camera chunk:

| LOD | Resolution | Threshold |
|---|---|---|
| 0 | Full (1:1) | dist ≤ `LOD_NEAR_BASE × lod_multiplier` chunks |
| 1 | 2× coarser | dist ≤ `LOD_FAR_BASE × lod_multiplier` chunks |
| 2 | 4× coarser | everything beyond LOD1 within render distance |

Greedy meshing merges adjacent same-material faces per LOD level.

## Fluid Simulation (`core/fluid/mod.rs`)

`FluidSimulator` handles multiple fluid types (water, lava, etc.) from block registry:

- **Source blocks**: level = 1.0, never deplete
- **Flowing blocks**: level < 1.0, created by spread from sources
- **Spread distance**: per-fluid, limits horizontal reach from source
- **Interactions**: two fluids meeting can produce a solid (e.g. water + lava → stone)
- **Cross-chunk flow**: snapshot-based — immutable read phase, then mutable apply phase
- **Constants**: `MIN_LEVEL = 0.005`, `FLOW_THRESHOLD = 0.01`, `MAX_SUB_STEPS = 8`

`World` owns a `FluidSimulator`; `water_block_id` cached for fast lookup.

## Block Registry (`gameobjects/block.rs`)

JSON-based, loaded from `assets/blocks/*.json`. Per-block properties:
- Texel coordinates (in atlas)
- PBR: metallic, roughness, emission color + intensity
- Transparency (opaque / transparent / translucent)
- Fluid properties: flow_rate, gravity_rate, spread_distance (for fluids)
- Biome tint: foliage/water color overriding

Example block JSONs: `assets/blocks/dirt.json`, `assets/blocks/water.json`, `assets/blocks/lava.json`, etc.

## Texture Atlas (`gameobjects/texture_atlas.rs`)

Composes all block face textures into a single GPU atlas from PNG files in `assets/textures/`. Reduces draw calls. Each block references a texel region in `TextureAtlasLayout`.

## Player Model (`core/player_model.rs`)

Parses Minecraft Bedrock Edition `minecraft:geometry` JSON format from `assets/player_model.json`:
- Right-handed Y-up coordinate system; 1 model unit = 1/16 block
- Box UV mapping: top/bottom/left/right/front/back face layout
- Per-bone: pivot, initial rotation, cube list
- Consumed by `PlayerRenderer` for skeletal rendering (see ARCHITECTURE-RENDERING.md)

## Worldgen Visualizer (`screens/worldgen_visualizer_screen.rs`)

Debug screen that renders the biome/terrain pipeline top-down, showing climate zones, geography, geology, and biome IDs without generating full 3D chunks.
