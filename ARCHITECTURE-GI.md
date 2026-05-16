# Global Illumination Systems

## Active GI: VCT (Voxel Flood-Fill Propagation)

All files: `src/core/vct/`.

### Overview

Light is propagated through a 128³ voxel volume centered on the camera using iterative flood-fill on the GPU. Two 3D textures ping-pong until convergence (fixed 24 iterations per frame).

### Per-frame dispatch (called from GameScreen)

```
world_worker sends VoxelSnapshot → pending_voxel_snapshot
vct.upload_volume(device, queue, snapshot, registry)  → writes voxel_data + voxel_emission
vct.dispatch_gi(&mut encoder)                          → inject + 24× propagate
fragment shader reads radiance via vct.bind_group()   → group(1) in pbr_vct.wgsl
```

### Key files

| File | Role |
|---|---|
| `vct/voxel_volume.rs` | `VoxelSnapshot` (CPU block-ID cube), `VOLUME_SIZE=128`, packing to RGBA8 |
| `vct/dynamic_lights.rs` | `PointLightGPU`, `SpotLightGPU` GPU structs |
| `vct/system.rs` | `VCTSystem`: GPU resources, compute pipelines, dispatch |

### GPU resources (VCTSystem)

| Texture | Format | Size | Contents |
|---|---|---|---|
| `voxel_data` | RGBA8Unorm | 128³ | rgb=albedo, a=opacity |
| `voxel_emission` | RGBA8Unorm | 128³ | rgb=emission color, a=intensity |
| `radiance_a` | RGBA16Float | 128³ | ping buffer |
| `radiance_b` | RGBA16Float | 128³ | pong buffer |

### GPU uniform structs (compile-time size-asserted)

```rust
VCTFragParams    = 64 bytes   // fragment shader: volume origin, inv_size, gi_config, light_counts
InjectParams     = 32 bytes   // inject compute
PropagateParams  = 32 bytes   // propagation compute
```

`VCTFragParams` layout:
```
volume_origin [f32; 4]  — xyz = world origin, w = volume_size
inv_size      [f32; 4]  — xyz = 1/size, w = gi_intensity
gi_config     [f32; 4]  — x = max_shadow_steps, y = decay, z = ambient_min, w = 0
light_counts  [u32; 4]  — x = point lights, y = spot lights, zw = 0
```

### Tuning constants

```rust
PROPAGATION_STEPS = 24    // iterations per frame; 24 = light reaches ~24 blocks
PROPAGATION_DECAY = 0.97  // per-step decay; 0.97 ≈ visible up to 20 blocks
GI_INTENSITY      = 2.0   // multiplier on propagated radiance in fragment shader
MAX_SHADOW_STEPS  = 64.0  // voxel shadow ray steps in fragment
```

### Shaders

- `src/shaders/vct_inject.wgsl` — copies emission texture → radiance_a
- `src/shaders/vct_propagate.wgsl` — flood-fill one step, reads A writes B (ping-pong)
- `src/shaders/pbr_vct.wgsl` — fragment shader: PBR + trilinear sample from radiance texture

---

## Legacy GI: Radiance Cascades (archived, not connected to GameScreen)

All files: `src/core/radiance_cascades/`. Code compiles but `GameScreen` does not instantiate `RadianceCascadeSystemGpu`.

### Cascade hierarchy (5 levels, C0=finest → C4=coarsest)

| Level | Spacing | Subdivs | Rays | Grid half-extent |
|---|---|---|---|---|
| C0 | 4 vox | 2 | 24 | 6 |
| C1 | 8 vox | 2 | 24 | 4 |
| C2 | 16 vox | 3 | 54 | 2 |
| C3 | 32 vox | 4 | 96 | 2 |
| C4 | 64 vox | 5 | 150 | 1 |

Rays = `6 × subdiv²`. C0+C1 intended every frame; C2–C4 every 4th.

### RC key files

| File | Role |
|---|---|
| `types.rs` | `CascadeConfig`, `RadianceInterval` (48B), `ProbeGrid`, `BlockProperties` (16B) |
| `system.rs` | `RadianceCascadeSystemGpu`: GPU resources, pipelines. **Compile-time size asserts.** |
| `dispatch.rs` | Ray march compute — `CascadeDispatchParams` (64B) |
| `merge.rs` | Merge compute — `MergeParams` (96B), trilinear interp coarse→fine |
| `voxel_tex.rs` | 128³ R16Uint voxel texture from chunks; `0xFFFF` = unloaded sentinel |
| `sampling.rs` | Cube-face direction generation |
| `shaders/ray_march.wgsl` | DDA voxel ray marching |
| `shaders/merge.wgsl` | Cascade merge (trilinear, C(i+1)→C(i)) |

### Critical GPU struct sizes (compile-time asserted in system.rs)

```rust
CascadeDispatchParams  = 64 bytes   // must match CascadeParams in ray_march.wgsl
MergeParams            = 96 bytes   // must match MergeUniform in merge.wgsl
GiCascadeEntry         = 48 bytes
GiSampleParams         = 256 bytes
RadianceInterval       = 48 bytes   // 3 × vec4<f32>
BlockProperties        = 16 bytes   // 1 × vec4<f32>
```

**If you resurrect RC: update WGSL counterparts and compile-time asserts for any struct changes.**

### RadianceInterval (48 bytes)

```
radiance_in[4]       — in-scattered RGB + padding
radiance_out[4]      — emission at hit (.rgb) + tau (.w, optical thickness)
direction_length[4]  — ray direction (.xyz) + distance traveled (.w)
```

Merge formula: `M.radiance_in = A.radiance_in + exp(-A.tau) × B.radiance_in`

### RC Block LUT

256 entries (one per block ID), `Rgba32Float`:
- `.r` = opacity (0.0 = transparent, >500 = opaque)
- `.gba` = pre-multiplied emission (emission_color × intensity)
