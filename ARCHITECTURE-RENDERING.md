# Rendering Pipeline & Shaders

## Pass order each frame

1. **Sky pass** — `SkyRenderer` draws sky dome, sun/moon sprites
2. **3D Geometry pass** — frustum-culled chunk meshes, PBR + optional VCT GI sampling
3. **Player model pass** — skeletal player body (one draw call per visible bone)
4. **Water pass** — translucent water faces (`water.wgsl`)
5. **Block outline pass** — target block wireframe (`outline.wgsl`)
6. **egui UI overlay** — immediate-mode HUD on top

## Key files

| File | Role |
|---|---|
| `src/screens/game_3d_pipeline.rs` | GPU pipeline setup, frustum culling, VRAM budget, draw calls |
| `src/screens/sky_renderer.rs` | Sky dome + celestial body sprites |
| `src/screens/player_renderer.rs` | Skeletal player rendering (bone hierarchy, animation) |
| `src/screens/inventory.rs` | Hotbar + inventory UI (egui, drag-and-drop) |
| `src/core/egui_manager.rs` | egui tessellation + render pass |

## Shaders (active — `src/shaders/`)

| File | Purpose |
|---|---|
| `pbr_lighting.wgsl` | PBR material + directional sun/moon lighting (no GI) |
| `pbr_vct.wgsl` | PBR + VCT GI sampling via group(1) bind group |
| `vct_inject.wgsl` | VCT inject compute: emission → radiance_a texture |
| `vct_propagate.wgsl` | VCT propagation compute: ping-pong radiance A↔B |
| `water.wgsl` | Translucent water surface |
| `player_model.wgsl` | Per-bone skeletal player body |
| `outline.wgsl` | Block selection wireframe |

Legacy RC shaders still compile but are not wired to GameScreen — see `src/core/radiance_cascades/shaders/`.

## GPU Backend

**wgpu** (multi-backend: Vulkan/DX12/Metal). All shaders in WGSL.

## Vertex3D — 52 bytes (packed)

Defined in `game_3d_pipeline.rs`:

```
[ 0] position:  Float32x3  — world XYZ          (12 b)
[12] normal:    Snorm8x4   — xyz + unused         ( 4 b)
[16] color_ao:  Unorm8x4   — rgb albedo + AO      ( 4 b)
[20] texcoord:  Float32x2  — UV                   ( 8 b)
[28] material:  Unorm8x4   — roughness, metalness  ( 4 b)
[32] emission:  Float32x4  — rgb + intensity       (16 b)
[48] tangent:   Snorm8x4   — xyz + unused          ( 4 b)
```

## Frustum culling

`FrustumPlanes::from_view_projection(vp)` → `intersects_aabb(min, max)` per chunk. View-projection extracted from the camera's combined matrix. Frustum-failing chunks skip draw calls and are deprioritized for mesh rebuilds.

## VRAM budget

`CHUNK_MESH_VRAM_BUDGET = 4 GiB`. When exceeded, the farthest loaded chunks are evicted from GPU and re-queued for future re-meshing.

## Camera (`game_3d_pipeline.rs::Camera`)

First-person. Fields: position, yaw, pitch, fov, aspect, near=0.1, far=dynamic (render_distance × chunk_size × 1.2). Methods: `view_matrix()`, `projection_matrix()`, `view_projection_matrix()`, `forward()`, `right()`, `rotate()`, `update_render_distance()`.

## Player model rendering (`screens/player_renderer.rs`)

Bone hierarchy (body_yaw drives all limbs):
```
body → head (hidden 1st-person)
     → leftArm / rightArm    (counter-swing)
     → leftThigh → leftShin  (walk cycle)
     → rightThigh → rightShin
```
Per-bone uniform = 192 bytes (view_proj 64 + model 64 + 3×vec4 48, padded). One draw call per visible bone. Supports first/third-person `ViewMode`.
