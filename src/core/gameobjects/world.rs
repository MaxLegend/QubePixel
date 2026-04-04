// =============================================================================
// QubePixel — World  (chunk streaming + Perlin-noise terrain + LOD)
// =============================================================================
//
// CHANGELOG:
//   • Cylindrical chunk loading (dx²+dz² ≤ rd²) instead of square
//   • View-direction priority for chunk generation
//   • Frustum-prioritized mesh building with per-frame limit
//   • camera_forward piped through WorldWorker for sorting/culling
// =============================================================================

use std::collections::HashMap;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use glam::Vec3;
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;
use rapier3d::prelude::*;
use crate::{debug_log, flow_debug_log};
use crate::core::config;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::gameobjects::chunk::Chunk;
use crate::screens::game_3d_pipeline::Vertex3D;
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;
use crate::core::raycast::{dda_raycast, RaycastResult, MAX_RAYCAST_DISTANCE};
use crate::core::radiance_cascades::voxel_tex::{VOXEL_TEX_SIZE, VoxelTextureBuilder};
use crate::core::physics::PhysicsChunkReady;

// ---------------------------------------------------------------------------
// Voxel dirty-region types (used by WorldResult → GameScreen → RC system)
// ---------------------------------------------------------------------------

/// One 16³ voxel-texture sub-region to be uploaded to the GPU.
/// `tx, ty, tz` — texel-space origin (multiple of 16).
/// `data`       — flat [dx + dy*16 + dz*256] u16 slice, 4096 elements (8 KB).
pub struct VoxelChunkRegion {
    pub tx:   u32,
    pub ty:   u32,
    pub tz:   u32,
    pub data: Vec<u16>,
}

/// Describes what the voxel texture needs on the GPU side this frame.
pub enum VoxelUpdate {
    /// Camera crossed a chunk boundary — full 32 MB texture replaced.
    Full { origin: glam::IVec3, data: Vec<u16> },
    /// Only the listed chunk regions changed — much cheaper (~N×8 KB).
    Partial { origin: glam::IVec3, regions: Vec<VoxelChunkRegion> },
}
// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// World-space Y at which the noise maps to zero (the "sea level").
/// Set to half of the default chunk height (256/2 = 128) so the ground
/// sits in the middle of the world column.
const SURFACE_BASE: f64 = 128.0;
/// Half-amplitude of terrain height variation in blocks.
const SURFACE_AMP: f64  = 32.0;
/// Horizontal scale of the noise (larger = smoother hills).
const NOISE_SCALE: f64  = 64.0;

/// Staggered loading — max chunks generated per World::update() call.
const MAX_CHUNKS_PER_FRAME: usize = 64;
/// Max chunk meshes rebuilt per build_meshes() call (frustum-priority order).
const MAX_MESH_BUILDS_PER_FRAME: usize = 128;

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
pub struct World {
    chunks:   HashMap<(i32, i32, i32), Chunk>,
    noise:    Perlin,
    pub registry: BlockRegistry,
    grass_id: u8,
    dirt_id:  u8,
    stone_id: u8,
    atlas_layout: TextureAtlasLayout,
    /// Per-chunk LOD level (0 = full, 1 = 2×, 2 = 4×).
    chunk_lod: HashMap<(i32, i32, i32), u8>,
    /// LOD counts from last update (for profiler).
    last_lod_counts: [usize; 3],
}

impl World {
    pub fn new(atlas_layout: TextureAtlasLayout) -> Self {
        let registry = BlockRegistry::load();
        let grass_id  = registry.id_for("grass").unwrap_or(1);
        let dirt_id   = registry.id_for("dirt").unwrap_or(grass_id);
        let stone_id  = registry.id_for("stone").unwrap_or(grass_id);

        debug_log!(
            "World", "new",
            "Block IDs — grass:{} dirt:{} stone:{}",
            grass_id, dirt_id, stone_id
        );

        Self {
            chunks:   HashMap::new(),
            noise:    Perlin::new(12345u32),
            registry,
            grass_id,
            dirt_id,
            stone_id,
            atlas_layout,
            chunk_lod: HashMap::new(),
            last_lod_counts: [0; 3],
        }
    }

    fn surface_height_fn(noise: &Perlin, wx: i32, wz: i32) -> i32 {
        let nx = wx as f64 / NOISE_SCALE;
        let nz = wz as f64 / NOISE_SCALE;
        let h  = noise.get([nx, nz]);
        (SURFACE_BASE + h * SURFACE_AMP).round() as i32
    }

    fn generate_chunk_fn(
        noise:    &Perlin,
        grass_id: u8,
        dirt_id:  u8,
        stone_id: u8,
        cx: i32, cy: i32, cz: i32,
    ) -> Chunk {
        let sx = config::chunk_size_x();
        let sy = config::chunk_size_y();
        let sz = config::chunk_size_z();

        let mut chunk = Chunk::new(cx, cy, cz);
        chunk.mesh_dirty = true;

        for bx in 0..sx {
            for bz in 0..sz {
                let wx = cx * sx as i32 + bx as i32;
                let wz = cz * sz as i32 + bz as i32;
                let surface = Self::surface_height_fn(noise, wx, wz);

                for by in 0..sy {
                    let wy = cy * sy as i32 + by as i32;
                    let id = if wy > surface {
                        0
                    } else if wy == surface {
                        grass_id
                    } else if wy >= surface - 3 {
                        dirt_id
                    } else {
                        stone_id
                    };
                    chunk.set_gen(bx, by, bz, id);
                }
            }
        }
        chunk
    }

    // -------------------------------------------------------------------
    // update — generate/evict chunks, compute LOD levels
    // -------------------------------------------------------------------

    /// Load chunks around `camera_pos` and unload ones too far away.
    /// Uses cylindrical shape (dx²+dz² ≤ rd²) and prioritizes view direction.
    /// Returns `(changed, evicted, gen_time_us, pending)`.
    pub fn update(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) -> (bool, Vec<(i32, i32, i32)>, u128, usize) {
        let rd      = config::render_distance();
        let v_below = config::vertical_below();
        let v_above = config::vertical_above();
        let rd_sq   = rd * rd;

        let cam_cx = (camera_pos.x / config::chunk_size_x() as f32).floor() as i32;
        let cam_cy = (camera_pos.y / config::chunk_size_y() as f32).floor() as i32;
        let cam_cz = (camera_pos.z / config::chunk_size_z() as f32).floor() as i32;

        // --- Camera forward projected onto XZ plane (for priority sorting) ---
        let fwd_xz_len = (camera_forward.x * camera_forward.x
            + camera_forward.z * camera_forward.z).sqrt();
        let fwd_nx = if fwd_xz_len > 1e-6 { camera_forward.x / fwd_xz_len } else { 0.0 };
        let fwd_nz = if fwd_xz_len > 1e-6 { camera_forward.z / fwd_xz_len } else { 0.0 };

        // --- Collect keys within CYLINDRICAL render distance ---
        let mut to_generate: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -rd..=rd {
            for dz in -rd..=rd {
                // Cylindrical filter: skip corners of the square
                if dx * dx + dz * dz > rd_sq { continue; }

                for dy in -v_below..=v_above {
                    let key = (cam_cx + dx, cam_cy + dy, cam_cz + dz);
                    if self.chunks.contains_key(&key) { continue; }
                    to_generate.push(key);
                }
            }
        }

        // --- Priority sort: view-direction weighted distance ---
        // Lower score = higher priority.
        // Chunks in front of camera get a bonus (negative offset).
        to_generate.sort_by(|a, b| {
            let score = |key: &(i32, i32, i32)| -> i64 {
                let dx = (key.0 - cam_cx) as f64;
                let dy = (key.1 - cam_cy) as f64;
                let dz = (key.2 - cam_cz) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dot = dx * fwd_nx as f64 + dz * fwd_nz as f64;
                // Subtract dot*3 so forward chunks get ~3 chunks distance advantage
                ((dist_sq - dot * 3.0) * 1000.0) as i64
            };
            score(a).cmp(&score(b))
        });

        let total_pending = to_generate.len();
        let gen_count = to_generate.len().min(MAX_CHUNKS_PER_FRAME);
        let was_truncated = to_generate.len() > MAX_CHUNKS_PER_FRAME;
        to_generate.truncate(gen_count);

        let mut changed = false;
        let mut gen_time_us: u128 = 0;

        if !to_generate.is_empty() {
            let t0 = Instant::now();

            let noise    = &self.noise;
            let grass_id = self.grass_id;
            let dirt_id  = self.dirt_id;
            let stone_id = self.stone_id;

            let new_chunks: Vec<((i32, i32, i32), Chunk)> = to_generate
                .into_par_iter()
                .map(|key| {
                    let chunk = Self::generate_chunk_fn(
                        noise, grass_id, dirt_id, stone_id,
                        key.0, key.1, key.2,
                    );
                    (key, chunk)
                })
                .collect();

            gen_time_us = t0.elapsed().as_micros();

            // Mark neighbors dirty
            for &(ref key, _) in &new_chunks {
                let offsets: [(i32, i32, i32); 6] = [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ];
                for &(dx, dy, dz) in &offsets {
                    let nk = (key.0 + dx, key.1 + dy, key.2 + dz);
                    if let Some(neighbor) = self.chunks.get_mut(&nk) {
                        neighbor.mesh_dirty = true;
                    }
                }
            }

            for (key, chunk) in new_chunks {
                self.chunks.insert(key, chunk);
            }

            changed = true;
        }

        // --- Evict chunks beyond CYLINDRICAL render distance ---
        let evict_r   = rd + 2;
        let evict_r_sq = evict_r * evict_r;
        let before = self.chunks.len();

        let evict_keys: Vec<(i32, i32, i32)> = self.chunks.keys()
            .copied()
            .filter(|&(cx, cy, cz)| {
                let dx = cx - cam_cx;
                let dz = cz - cam_cz;
                let too_far_h = dx * dx + dz * dz > evict_r_sq;
                let too_far_v = cy < cam_cy - v_below - 1
                    || cy > cam_cy + v_above + 1;
                too_far_h || too_far_v
            })
            .collect();

        for key in &evict_keys {
            self.chunks.remove(key);
            self.chunk_lod.remove(key);
        }

        let evicted: Vec<(i32, i32, i32)> = evict_keys;

        if self.chunks.len() != before {
            changed = true;
        }

        // --- Compute LOD levels for all loaded chunks ---
        let mut lod_counts = [0usize; 3];

        let loaded_keys: Vec<(i32, i32, i32)> = self.chunks.keys().copied().collect();

        for key in loaded_keys {
            let new_lod = config::compute_lod_level(cam_cx, cam_cz, key.0, key.2);
            let old_lod = self.chunk_lod.get(&key).copied().unwrap_or(255);

            if new_lod != old_lod {
                self.chunk_lod.insert(key, new_lod);
                if old_lod != 255 {
                    if let Some(chunk) = self.chunks.get_mut(&key) {
                        chunk.mesh_dirty = true;
                    }
                    changed = true;
                }
            }

            lod_counts[new_lod as usize] += 1;
        }
        self.last_lod_counts = lod_counts;

        (changed, evicted, gen_time_us, if was_truncated { total_pending - gen_count } else { 0 })
    }

    // -------------------------------------------------------------------
    // build_meshes — frustum-prioritized, limited per frame
    // -------------------------------------------------------------------

    /// Builds meshes for dirty chunks, prioritizing those visible to camera.
    /// At most `MAX_MESH_BUILDS_PER_FRAME` meshes are built; remaining stay dirty.
    pub fn build_meshes(
        &mut self,
        camera_pos: Vec3,
        camera_forward: Vec3,
    ) -> (Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>, u128, usize) {
        let t0 = Instant::now();

        let cam_cx = (camera_pos.x / config::chunk_size_x() as f32).floor() as i32;
        let cam_cz = (camera_pos.z / config::chunk_size_z() as f32).floor() as i32;

        // Camera forward on XZ plane
        let fwd_xz = {
            let len = (camera_forward.x * camera_forward.x
                + camera_forward.z * camera_forward.z).sqrt();
            if len > 1e-6 {
                (camera_forward.x / len, camera_forward.z / len)
            } else {
                (0.0f32, 0.0f32)
            }
        };

        // Collect all dirty keys
        let mut dirty_keys: Vec<(i32, i32, i32)> = self.chunks.iter()
            .filter_map(|(&key, chunk)| {
                if chunk.mesh_dirty { Some(key) } else { None }
            })
            .collect();

        // Sort: visible-first (by dot with forward), then by distance
        dirty_keys.sort_by(|a, b| {
            let score = |key: &(i32, i32, i32)| -> i64 {
                let dx = (key.0 - cam_cx) as f32;
                let dz = (key.2 - cam_cz) as f32;
                let dist_sq = dx * dx + dz * dz;
                let dot = dx * fwd_xz.0 + dz * fwd_xz.1;
                // Lower score = higher priority; forward chunks get a bonus
                ((dist_sq - dot * 4.0) * 1000.0) as i64
            };
            score(a).cmp(&score(b))
        });

        // Limit per frame — the rest stay dirty for next frame
        dirty_keys.truncate(MAX_MESH_BUILDS_PER_FRAME);

        let dirty_count = dirty_keys.len();

        // Clear dirty ONLY for chunks we're about to build
        for &key in &dirty_keys {
            if let Some(chunk) = self.chunks.get_mut(&key) {
                chunk.mesh_dirty = false;
            }
        }

        let chunks   = &self.chunks;
        let registry = &self.registry;
        let atlas    = &self.atlas_layout;
        let lod_map  = &self.chunk_lod;
        let csx = config::chunk_size_x() as i32;
        let csy = config::chunk_size_y() as i32;
        let csz = config::chunk_size_z() as i32;

        let result: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])> = dirty_keys
            .into_par_iter()
            .filter_map(|key| {
                let chunk = match chunks.get(&key) {
                    Some(c) => c,
                    None    => return None,
                };

                let lod = lod_map.get(&key).copied().unwrap_or(0);

                let get_neighbor = |wx: i32, wy: i32, wz: i32| -> u8 {
                    let cx = wx.div_euclid(csx);
                    let cy = wy.div_euclid(csy);
                    let cz = wz.div_euclid(csz);
                    let lx = wx.rem_euclid(csx) as usize;
                    let ly = wy.rem_euclid(csy) as usize;
                    let lz = wz.rem_euclid(csz) as usize;
                    chunks.get(&(cx, cy, cz))
                        .map_or(0, |c| c.get(lx, ly, lz))
                };

                let (verts, idxs) = chunk.build_mesh_lod(registry, atlas, get_neighbor, lod);

                if verts.is_empty() { return None; }

                let sx = csx as f32;
                let sy = csy as f32;
                let sz = csz as f32;
                let min = [chunk.cx as f32 * sx, chunk.cy as f32 * sy, chunk.cz as f32 * sz];
                let max = [min[0] + sx, min[1] + sy, min[2] + sz];
                Some((key, verts, idxs, min, max))
            })
            .collect();

        let build_time_us = t0.elapsed().as_micros();

        (result, build_time_us, dirty_count)
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    /// Returns true if any loaded chunk still needs its mesh rebuilt.
    pub fn has_dirty_chunks(&self) -> bool {
        self.chunks.values().any(|c| c.mesh_dirty)
    }

    fn mark_dirty(&mut self, cx: i32, cy: i32, cz: i32) {
        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.mesh_dirty = true;
        }
    }

    fn dirty_boundary_neighbors(
        &mut self,
        cx: i32, cy: i32, cz: i32,
        lx: usize, ly: usize, lz: usize,
    ) {
        let sx = config::chunk_size_x();
        let sy = config::chunk_size_y();
        let sz = config::chunk_size_z();
        if lx == 0      { self.mark_dirty(cx - 1, cy, cz); }
        if lx == sx - 1 { self.mark_dirty(cx + 1, cy, cz); }
        if ly == 0      { self.mark_dirty(cx, cy - 1, cz); }
        if ly == sy - 1 { self.mark_dirty(cx, cy + 1, cz); }
        if lz == 0      { self.mark_dirty(cx, cy, cz - 1); }
        if lz == sz - 1 { self.mark_dirty(cx, cy, cz + 1); }
    }

    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn lod_counts(&self) -> [usize; 3] { self.last_lod_counts }
    // -----------------------------------------------------------------------
    // Volumetric light: scan nearby chunks for emissive blocks
    // -----------------------------------------------------------------------

    /// Scan chunks within `radius` world-blocks of `camera_pos` for blocks
    /// whose emission is set. Returns at most `max_count` results.
    ///
    /// Called by WorldWorker when `has_dirty` — updates the per-frame emissive
    /// block list used by the volumetric light pass.
    pub fn scan_emissive_blocks(
        &self,
        camera_pos: Vec3,
        radius:     i32,
        max_count:  usize,
    ) -> Vec<EmissiveBlockPos> {
        let mut result = Vec::new();
        let radius_sq  = radius * radius;
        let cam_wx = camera_pos.x.floor() as i32;
        let cam_wy = camera_pos.y.floor() as i32;
        let cam_wz = camera_pos.z.floor() as i32;

        let csx       = config::chunk_size_x() as i32;
        let csy       = config::chunk_size_y() as i32;
        let csz       = config::chunk_size_z() as i32;
        let r_chunks  = (radius / csx) + 1;
        let cam_cx    = cam_wx.div_euclid(csx);
        let cam_cy    = cam_wy.div_euclid(csy);
        let cam_cz    = cam_wz.div_euclid(csz);

        'outer: for dcx in -r_chunks..=r_chunks {
            for dcy in -r_chunks..=r_chunks {
                for dcz in -r_chunks..=r_chunks {
                    let key = (cam_cx + dcx, cam_cy + dcy, cam_cz + dcz);
                    if let Some(chunk) = self.chunks.get(&key) {
                        let wx_base = key.0 * csx;
                        let wy_base = key.1 * csy;
                        let wz_base = key.2 * csz;
                        for bx in 0..config::chunk_size_x() {
                            for by in 0..config::chunk_size_y() {
                                for bz in 0..config::chunk_size_z() {
                                    let block_id = chunk.get(bx, by, bz);
                                    if block_id == 0 { continue; }
                                    if let Some(def) = self.registry.get(block_id) {
                                        if !def.emission.emit_light { continue; }
                                        let wx = wx_base + bx as i32;
                                        let wy = wy_base + by as i32;
                                        let wz = wz_base + bz as i32;
                                        let dx = wx - cam_wx;
                                        let dy = wy - cam_wy;
                                        let dz = wz - cam_wz;
                                        if dx*dx + dy*dy + dz*dz > radius_sq { continue; }
                                        result.push(EmissiveBlockPos {
                                            wx: wx as f32 + 0.5,
                                            wy: wy as f32 + 0.5,
                                            wz: wz as f32 + 0.5,
                                            block_id,
                                        });
                                        if result.len() >= max_count { break 'outer; }
                                    }
                                }}}
                    }
                }}}
        result
    }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;
        match self.chunks.get(&(cx, cy, cz)) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => 0,
        }
    }

    pub fn remove_block(&mut self, wx: i32, wy: i32, wz: i32) {
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set(lx, ly, lz, 0);
            debug_log!("World", "remove_block", "Broke block at ({}, {}, {})", wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }

    pub fn place_block(&mut self, wx: i32, wy: i32, wz: i32, block_id: u8) {
        if block_id == 0 { return; }
        let (csx, csy, csz) = (
            config::chunk_size_x() as i32,
            config::chunk_size_y() as i32,
            config::chunk_size_z() as i32,
        );
        let cx = wx.div_euclid(csx);
        let cy = wy.div_euclid(csy);
        let cz = wz.div_euclid(csz);
        let lx = wx.rem_euclid(csx) as usize;
        let ly = wy.rem_euclid(csy) as usize;
        let lz = wz.rem_euclid(csz) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set(lx, ly, lz, block_id);
            debug_log!("World", "place_block", "Placed block {} at ({}, {}, {})", block_id, wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }

    // -----------------------------------------------------------------------
    // Radiance Cascades: fill voxel texture from loaded chunks
    // -----------------------------------------------------------------------

    /// Fill a VoxelTextureBuilder with block IDs from all loaded chunks.
    ///
    /// Iterates over the 128³ volume around the camera and writes each block
    /// into the builder's CPU buffer. The caller should recenter() the builder
    /// first (so world_to_texel mappings are correct), then upload to GPU.
    ///
    /// This is called each frame from GameScreen::render() before RC dispatch.
    pub fn fill_voxel_texture(&self, builder: &mut VoxelTextureBuilder) {
        let origin = builder.world_origin();
        let size   = VOXEL_TEX_SIZE as i32;
        let csx    = config::chunk_size_x() as i32;
        let csy    = config::chunk_size_y() as i32;
        let csz    = config::chunk_size_z() as i32;

        for (key, chunk) in &self.chunks {
            let wx_base = key.0 * csx;
            let wy_base = key.1 * csy;
            let wz_base = key.2 * csz;

            // Skip chunks that don't overlap the voxel texture
            if wx_base + csx <= origin.x || wx_base >= origin.x + size { continue; }
            if wy_base + csy <= origin.y || wy_base >= origin.y + size { continue; }
            if wz_base + csz <= origin.z || wz_base >= origin.z + size { continue; }

            for bx in 0..config::chunk_size_x() {
                for by in 0..config::chunk_size_y() {
                    for bz in 0..config::chunk_size_z() {
                        let block_id = chunk.get(bx, by, bz);
                        builder.set_block(
                            wx_base + bx as i32,
                            wy_base + by as i32,
                            wz_base + bz as i32,
                            block_id,
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// WorldWorker — runs World on a background thread
// =============================================================================

enum WorldRequest {
    Update {
        camera_pos:        Vec3,
        camera_forward:    Vec3,
        ray_dir:           Option<Vec3>,
        block_ops:         Vec<BlockOp>,
        /// Chunks evicted from GPU VRAM — mark dirty so they get re-meshed
        /// when the camera brings them back into view.
        force_dirty_keys:  Vec<(i32, i32, i32)>,
    },
    Shutdown,
}
/// World-space center of an emissive block, used by the volumetric light pass.
#[derive(Debug, Clone)]
pub struct EmissiveBlockPos {
    pub wx: f32,
    pub wy: f32,
    pub wz: f32,
    pub block_id: u8,
}
#[derive(Debug, Clone)]
pub enum BlockOp {
    Break { x: i32, y: i32, z: i32 },
    Place { x: i32, y: i32, z: i32, block_id: u8 },
}

pub struct WorldResult {
    pub meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    pub evicted: Vec<(i32, i32, i32)>,
    pub chunk_count: usize,
    pub gen_time_us: u128,
    pub mesh_time_us: u128,
    pub pending: usize,
    pub raycast_result: Option<RaycastResult>,
    /// Pre-built physics colliders — compound shapes already constructed off-thread.
    pub physics_ready: Vec<PhysicsChunkReady>,
    // Profiler data
    pub lod_counts: [usize; 3],
    pub dirty_count: usize,
    pub worker_total_us: u128,
    pub voxel_data: Option<VoxelUpdate>,
    /// Emissive blocks near the camera — fed to the volumetric light pass.
    /// `None` means the list has not changed since last frame.
    pub emissive_blocks: Option<Vec<EmissiveBlockPos>>,
}

pub struct WorldWorker {
    request_tx:          mpsc::Sender<WorldRequest>,
    result_rx:           mpsc::Receiver<WorldResult>,
    thread:              Option<thread::JoinHandle<()>>,
    busy:                bool,
    last_chunk_count:    usize,
    pending_ops:         Vec<BlockOp>,
    /// Chunk keys evicted from GPU VRAM, buffered until next request_update.
    pending_force_dirty: Vec<(i32, i32, i32)>,
}

impl WorldWorker {
    pub fn new(atlas_layout: TextureAtlasLayout) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WorldRequest>();
        let (result_tx,  result_rx)  = mpsc::channel::<WorldResult>();

        let handle = thread::Builder::new()
            .name("world-worker".into())
            .spawn(move || {
                let mut world = World::new(atlas_layout);
                let mut vox_builder = VoxelTextureBuilder::new();
                // Sentinel: force Full rebuild on first dirty frame.
                let mut vox_known_origin = glam::IVec3::new(i32::MIN, i32::MIN, i32::MIN);
                // Track last player chunk position for physics rebuild triggers.
                // (cx, cam_wy_coarse, cz) — coarse Y prevents constant rebuild during walking on flat ground.
                let mut last_phys_cam: (i32, i32, i32) = (i32::MIN, i32::MIN, i32::MIN);
                loop {
                    match request_rx.recv() {
                        Ok(WorldRequest::Update {
                               camera_pos,
                               camera_forward,
                               ray_dir,
                               block_ops,
                               force_dirty_keys,
                           }) => {
                            let t_total = Instant::now();

                            // Re-mark GPU-evicted chunks dirty so they get re-meshed
                            // next time the camera moves near them.
                            for key in &force_dirty_keys {
                                if let Some(chunk) = world.chunks.get_mut(key) {
                                    chunk.mesh_dirty = true;
                                }
                            }

                            let (changed, evicted, gen_time, pending) =
                                world.update(camera_pos, camera_forward);

                            for op in &block_ops {
                                match op {
                                    BlockOp::Break { x, y, z } => {
                                        world.remove_block(*x, *y, *z);
                                    }
                                    BlockOp::Place { x, y, z, block_id } => {
                                        world.place_block(*x, *y, *z, *block_id);
                                    }
                                }
                            }
                            let has_block_ops = !block_ops.is_empty();

                            // Also rebuild when deferred dirty chunks remain
                            let has_dirty = changed
                                || has_block_ops
                                || world.has_dirty_chunks();

                            let (meshes, mesh_time, dirty_count) = if has_dirty {
                                world.build_meshes(camera_pos, camera_forward)
                            } else {
                                (Vec::new(), 0, 0)
                            };

                            let raycast_result = ray_dir.and_then(|dir| {
                                dda_raycast(
                                    camera_pos,
                                    dir,
                                    MAX_RAYCAST_DISTANCE,
                                    |wx, wy, wz| world.get_block(wx, wy, wz),
                                )
                            });

                            let chunk_sx = config::chunk_size_x() as f32;
                            let chunk_sy = config::chunk_size_y() as f32;
                            let chunk_sz = config::chunk_size_z() as f32;
                            let phys_sx  = config::chunk_size_x();
                            let phys_sy  = config::chunk_size_y();
                            let phys_sz  = config::chunk_size_z();

                            // ---------------------------------------------------------------------------
                            // Physics — proximity scan + compound build (all off main thread).
                            //
                            // Only the 5×5 chunk area around the player, Y-sliced to ±PHYSICS_Y_MARGIN.
                            // On block-op only the single affected chunk is rebuilt (incremental).
                            // Compound shapes are built here via rayon — main thread just inserts handles.
                            // ---------------------------------------------------------------------------
                            const PHYSICS_H_RADIUS: i32 = 2;
                            const PHYSICS_Y_MARGIN: i32 = 10;

                            let cam_chunk_x = (camera_pos.x / chunk_sx).floor() as i32;
                            let cam_chunk_z = (camera_pos.z / chunk_sz).floor() as i32;
                            let cam_cy      = (camera_pos.y / chunk_sy).floor() as i32;
                            let cam_wy      = camera_pos.y.floor() as i32;

                            let cam_wy_coarse = cam_wy / 4;
                            let player_moved  = (cam_chunk_x, cam_wy_coarse, cam_chunk_z) != last_phys_cam;

                            // Collect affected chunk keys from block-ops for incremental rebuild
                            let mut block_op_chunk_keys: Vec<(i32, i32, i32)> = Vec::new();
                            if has_block_ops {
                                let csx_i = phys_sx as i32;
                                let csy_i = phys_sy as i32;
                                let csz_i = phys_sz as i32;
                                for op in &block_ops {
                                    let (wx, wy, wz) = match op {
                                        BlockOp::Break { x, y, z } => (*x, *y, *z),
                                        BlockOp::Place { x, y, z, .. } => (*x, *y, *z),
                                    };
                                    let key = (
                                        wx.div_euclid(csx_i),
                                        wy.div_euclid(csy_i),
                                        wz.div_euclid(csz_i),
                                    );
                                    if !block_op_chunk_keys.contains(&key) {
                                        block_op_chunk_keys.push(key);
                                    }
                                }
                            }

                            let build_physics = has_dirty || has_block_ops || player_moved;

                            if build_physics {
                                last_phys_cam = (cam_chunk_x, cam_wy_coarse, cam_chunk_z);
                            }

                            let v_below = config::vertical_below() as i32;
                            let v_above = config::vertical_above() as i32;

                            let physics_ready: Vec<PhysicsChunkReady> = if build_physics {
                                // Decide which chunk keys to scan:
                                // • incremental (block-op only) → just the affected chunk(s)
                                // • full → all chunks within PHYSICS_H_RADIUS
                                let keys_to_build: Vec<(i32, i32, i32)> =
                                    if !block_op_chunk_keys.is_empty() && !changed && !player_moved {
                                        // Incremental: only the chunks that were modified
                                        block_op_chunk_keys.clone()
                                    } else {
                                        // Full scan
                                        let mut keys = Vec::new();
                                        for dx in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                            for dz in -PHYSICS_H_RADIUS..=PHYSICS_H_RADIUS {
                                                for dy in -v_below..=v_above {
                                                    keys.push((cam_chunk_x + dx, cam_cy + dy, cam_chunk_z + dz));
                                                }
                                            }
                                        }
                                        keys
                                    };

                                flow_debug_log!(
                                    "WorldWorker", "thread",
                                    "[Physics] rebuild keys={} cam=({},{}) wy={} dirty={} ops={} moved={}",
                                    keys_to_build.len(), cam_chunk_x, cam_chunk_z, cam_wy,
                                    has_dirty, has_block_ops, player_moved
                                );

                                // Build compound shapes in parallel with rayon
                                keys_to_build.into_par_iter()
                                    .filter_map(|key| {
                                        let wy_base = key.1 * phys_sy as i32;
                                        let y_lo = (cam_wy - PHYSICS_Y_MARGIN - wy_base)
                                            .max(0) as usize;
                                        let y_hi = (cam_wy + PHYSICS_Y_MARGIN + 1 - wy_base)
                                            .min(phys_sy as i32).max(0) as usize;
                                        if y_lo >= y_hi { return None; }

                                        let chunk = world.chunks.get(&key)?;
                                        let cuboid = SharedShape::cuboid(0.5, 0.5, 0.5);
                                        let mut shapes: Vec<(Pose, SharedShape)> = Vec::new();
                                        for x in 0..phys_sx {
                                            for y in y_lo..y_hi {
                                                for z in 0..phys_sz {
                                                    if chunk.get(x, y, z) != 0 {
                                                        let wx = key.0 as f32 * chunk_sx + x as f32 + 0.5;
                                                        let wy = key.1 as f32 * chunk_sy + y as f32 + 0.5;
                                                        let wz = key.2 as f32 * chunk_sz + z as f32 + 0.5;
                                                        shapes.push((
                                                            Pose::from_translation(
                                                                rapier3d::prelude::Vector::new(wx, wy, wz),
                                                            ),
                                                            cuboid.clone(),
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                        if shapes.is_empty() { return None; }

                                        let compound = SharedShape::compound(shapes);
                                        let collider = ColliderBuilder::new(compound)
                                            .friction(1.0)
                                            .restitution(0.0)
                                            .build();
                                        Some(PhysicsChunkReady { key, collider })
                                    })
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            // ---- Voxel texture update (Partial vs Full) ----
                            let voxel_data = if has_dirty || has_block_ops {
                                let origin_changed = vox_builder.recenter(camera_pos);

                                if origin_changed || vox_known_origin != vox_builder.world_origin() {
                                    // Camera crossed chunk boundary — full rebuild
                                    world.fill_voxel_texture(&mut vox_builder);
                                    vox_known_origin = vox_builder.world_origin();
                                    debug_log!("WorldWorker", "thread", "Voxel FULL rebuild (origin changed)");
                                    Some(VoxelUpdate::Full {
                                        origin: vox_builder.world_origin(),
                                        data:   vox_builder.data().to_vec(),
                                    })
                                } else {
                                    // Origin stable — update only affected sub-regions (~N×8 KB each).
                                    // Chunks may be taller than 16 blocks (columnar), so we
                                    // split each chunk into 16×16×16 voxel-texture tiles.
                                    const VOX_REG: usize = 16;
                                    let csx = config::chunk_size_x();
                                    let csy = config::chunk_size_y();
                                    let csz = config::chunk_size_z();
                                    let csx_i = csx as i32;
                                    let csy_i = csy as i32;
                                    let csz_i = csz as i32;
                                    let tiles_x = (csx + VOX_REG - 1) / VOX_REG;
                                    let tiles_y = (csy + VOX_REG - 1) / VOX_REG;
                                    let tiles_z = (csz + VOX_REG - 1) / VOX_REG;

                                    let mut regions = Vec::new();
                                    for &(cx, cy, cz) in meshes.iter().map(|(k,_,_,_,_)| k) {
                                        let wx_base = cx * csx_i;
                                        let wy_base = cy * csy_i;
                                        let wz_base = cz * csz_i;
                                        if let Some(chunk) = world.chunks.get(&(cx, cy, cz)) {
                                            for ty_i in 0..tiles_y {
                                                for tx_i in 0..tiles_x {
                                                    for tz_i in 0..tiles_z {
                                                        let wx_s = wx_base + (tx_i * VOX_REG) as i32;
                                                        let wy_s = wy_base + (ty_i * VOX_REG) as i32;
                                                        let wz_s = wz_base + (tz_i * VOX_REG) as i32;
                                                        let bx_off = tx_i * VOX_REG;
                                                        let by_off = ty_i * VOX_REG;
                                                        let bz_off = tz_i * VOX_REG;
                                                        if let Some((tx, ty, tz)) = vox_builder.fill_chunk_region_fn(
                                                            wx_s, wy_s, wz_s,
                                                            |bx, by, bz| chunk.get(bx + bx_off, by + by_off, bz + bz_off),
                                                        ) {
                                                            let data = vox_builder.extract_chunk_region(tx, ty, tz);
                                                            regions.push(VoxelChunkRegion { tx, ty, tz, data });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    debug_log!("WorldWorker", "thread",
                                        "Voxel PARTIAL update: {} regions", regions.len());
                                    if regions.is_empty() { None }
                                    else {
                                        Some(VoxelUpdate::Partial {
                                            origin: vox_builder.world_origin(),
                                            regions,
                                        })
                                    }
                                }
                            } else {
                                None
                            };

                            // ---- Emissive block scan for volumetric light pass ----
                            // Only re-scan when the world actually changed (chunk added/removed
                            // or block op applied).  Radius 48 = 3 chunks from camera.
                            let emissive_blocks = if has_dirty || has_block_ops {
                                Some(world.scan_emissive_blocks(camera_pos, 48, 256))
                            } else {
                                None
                            };
                            let worker_total_us = t_total.elapsed().as_micros();
                            let lod_counts = world.lod_counts();

                            let _ = result_tx.send(WorldResult {
                                meshes,
                                evicted,
                                chunk_count: world.chunk_count(),
                                gen_time_us: gen_time,
                                mesh_time_us: mesh_time,
                                pending,
                                raycast_result,
                                physics_ready,
                                lod_counts,
                                dirty_count,
                                worker_total_us,
                                voxel_data,
                                emissive_blocks,
                            });
                        }
                        Ok(WorldRequest::Shutdown) | Err(_) => break,
                    }
                }
                debug_log!("WorldWorker", "thread", "Background thread exiting");
            })
            .expect("Failed to spawn world-worker thread");

        debug_log!("WorldWorker", "new", "Spawned background world thread");

        Self {
            request_tx,
            result_rx,
            thread: Some(handle),
            busy:                false,
            last_chunk_count:    0,
            pending_ops:         Vec::new(),
            pending_force_dirty: Vec::new(),
        }
    }

    pub fn request_update(
        &mut self,
        camera_pos:       Vec3,
        camera_forward:   Vec3,
        ray_dir:          Option<Vec3>,
        block_ops:        Vec<BlockOp>,
        force_dirty_keys: Vec<(i32, i32, i32)>,
    ) {
        self.pending_ops.extend(block_ops);
        self.pending_force_dirty.extend(force_dirty_keys);

        if !self.busy {
            let ops         = std::mem::take(&mut self.pending_ops);
            let dirty_keys  = std::mem::take(&mut self.pending_force_dirty);
            let _ = self.request_tx.send(WorldRequest::Update {
                camera_pos,
                camera_forward,
                ray_dir,
                block_ops:        ops,
                force_dirty_keys: dirty_keys,
            });
            self.busy = true;
        }
    }

    pub fn try_recv_result(&mut self) -> Option<WorldResult> {
        match self.result_rx.try_recv() {
            Ok(result) => {
                self.busy = false;
                self.last_chunk_count = result.chunk_count;
                Some(result)
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.busy = false;
                None
            }
        }
    }

    pub fn chunk_count(&self) -> usize {
        self.last_chunk_count
    }
}

impl Drop for WorldWorker {
    fn drop(&mut self) {
        debug_log!("WorldWorker", "drop", "Shutting down background thread");
        let _ = self.request_tx.send(WorldRequest::Shutdown);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}