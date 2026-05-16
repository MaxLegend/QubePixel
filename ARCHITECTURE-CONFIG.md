# Configuration, Lighting, Physics & Player

## Runtime config atomics (`src/core/config.rs`)

All are global atomics, changed at runtime via the Settings screen:

| Atomic | Type | Default | Meaning |
|---|---|---|---|
| `RENDER_DISTANCE` | `AtomicI32` | 3 | Chunk streaming radius (horizontal, in chunks) |
| `LOD_MULTIPLIER` | `AtomicU32` | 100 | LOD threshold scale × 100 (100 = 1.0×) |
| `VERTICAL_BELOW` | `AtomicI32` | 0 | Chunks to stream below camera |
| `VERTICAL_ABOVE` | `AtomicI32` | 0 | Chunks to stream above camera |
| `BIOME_AMBIENT_TINT_ENABLED` | `AtomicU32` | 1 | Per-biome ambient tint (hot→warm, icy→cool) |

Chunk dimensions (`size_x/y/z`) are **not** atomic — loaded once from `assets/chunk_config.json` via `OnceLock` at startup. Default: 16×256×16.

LOD constants (before multiplier): `LOD_NEAR_BASE = 3.0`, `LOD_FAR_BASE = 6.0` (in chunks).

## World generation config (`src/core/config.rs::WorldGenConfig`)

Loaded once from `assets/world_gen_config.json` via `OnceLock`. Has a `test_mode` flag that applies `test_overrides` when enabled (reduces noise scales for faster debugging).

### Sub-configs

**GeographyConfig** — continental height map:
- `sea_level`, `terrain_base`, `terrain_amplitude`
- `continent_scale`, `erosion_scale`, `peaks_scale`, `island_scale`
- `continent_octaves`, `erosion_octaves`, `peaks_octaves`
- `island_threshold`, `island_boost`
- Zone thresholds: `deep_ocean_max`, `ocean_max`, `lowland_max`, `midland_max`

**ClimateConfig** — climate zones:
- `band_scale` — latitude band frequency
- `boundary_noise_scale`, `boundary_warp_amount` — zone boundary irregularity
- `humidity_noise_scale`

**GeologyConfig** — geological regions:
- `macro_cell_size` (default 2048 blocks), `micro_cell_size` (default 512 blocks)
- `mantle_keep_denominator` — fraction of mantle cells retained

**StratificationConfig** — per-column layer thicknesses:
- `deep_granite_threshold`, `deep_granite_transition`
- `dirt_thickness_base/variation`, `clay_thickness_base/variation`
- `claystone_thickness_base/variation`
- `thickness_noise_scale`

## Lighting & day/night cycle (`src/core/lighting_legacy.rs`, re-exported via `lighting/mod.rs`)

`LightingConfig` deserialized from `assets/lighting_config.json`. Sub-configs:
- `DayNightConfig` — cycle duration, start time
- `SunConfig` — noon/sunrise colors, max intensity, sprite texture + scale
- `MoonConfig` — color, max intensity, sprite
- `AmbientConfig` — day/night colors, min level
- `ShadowConfig` — fields present in JSON but shadow mapping not GPU-implemented
- `SsaoConfig` — fields present in JSON but SSAO not GPU-implemented

`DayNightCycle` ticks time, `pack_lighting_uniforms()` produces the GPU uniform struct consumed by fragment shaders (sun/moon direction + color, ambient, time-of-day).

## Physics (`src/core/physics.rs`)

Rapier3D wrapper. Player collider + gravity + block collision detection. `PhysicsChunkReady` carries trimesh data from the world worker to be inserted into Rapier.

## Player & camera (`src/core/player.rs`)

First-person controller. WASD movement, mouse look (raw `DeviceEvent::MouseMotion` delta). Velocity integration via physics. Delegates to `Camera` in `game_3d_pipeline.rs` for view matrix.

## Raycast (`src/core/raycast.rs`)

DDA voxel raycast for block targeting and interaction. `MAX_RAYCAST_DISTANCE` limits reach. Returns `RaycastResult` with hit block coords + face normal.

## Settings screen (`src/screens/settings.rs`)

egui UI that writes directly to the config atomics above. Changes take effect the next frame.
