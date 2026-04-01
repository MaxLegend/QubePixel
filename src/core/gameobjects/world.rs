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
use crate::{debug_log, flow_debug_log};
use crate::core::config;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::gameobjects::chunk::{Chunk, CHUNK_SIZE};
use crate::screens::game_3d_pipeline::Vertex3D;
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;
use crate::core::raycast::{dda_raycast, RaycastResult, MAX_RAYCAST_DISTANCE};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// World-space Y at which the noise maps to zero (the "sea level").
const SURFACE_BASE: f64 = 8.0;
/// Half-amplitude of terrain height variation in blocks.
const SURFACE_AMP: f64  = 5.0;
/// Horizontal scale of the noise (larger = smoother hills).
const NOISE_SCALE: f64  = 48.0;

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
        let mut chunk = Chunk::new(cx, cy, cz);
        chunk.mesh_dirty = true;

        for bx in 0..CHUNK_SIZE {
            for bz in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + bx as i32;
                let wz = cz * CHUNK_SIZE as i32 + bz as i32;
                let surface = Self::surface_height_fn(noise, wx, wz);

                for by in 0..CHUNK_SIZE {
                    let wy = cy * CHUNK_SIZE as i32 + by as i32;
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

        let cam_cx = (camera_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let cam_cy = (camera_pos.y / CHUNK_SIZE as f32).floor() as i32;
        let cam_cz = (camera_pos.z / CHUNK_SIZE as f32).floor() as i32;

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

        let cam_cx = (camera_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let cam_cz = (camera_pos.z / CHUNK_SIZE as f32).floor() as i32;

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
        let cs       = CHUNK_SIZE as i32;

        let result: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])> = dirty_keys
            .into_par_iter()
            .filter_map(|key| {
                let chunk = match chunks.get(&key) {
                    Some(c) => c,
                    None    => return None,
                };

                let lod = lod_map.get(&key).copied().unwrap_or(0);

                let get_neighbor = |wx: i32, wy: i32, wz: i32| -> u8 {
                    let cx = wx.div_euclid(cs);
                    let cy = wy.div_euclid(cs);
                    let cz = wz.div_euclid(cs);
                    let lx = wx.rem_euclid(cs) as usize;
                    let ly = wy.rem_euclid(cs) as usize;
                    let lz = wz.rem_euclid(cs) as usize;
                    chunks.get(&(cx, cy, cz))
                        .map_or(0, |c| c.get(lx, ly, lz))
                };

                let (verts, idxs) = chunk.build_mesh_lod(registry, atlas, get_neighbor, lod);
                if verts.is_empty() { return None; }

                let s   = CHUNK_SIZE as f32;
                let min = [chunk.cx as f32 * s, chunk.cy as f32 * s, chunk.cz as f32 * s];
                let max = [min[0] + s, min[1] + s, min[2] + s];
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
        if lx == 0                 { self.mark_dirty(cx - 1, cy, cz); }
        if lx == CHUNK_SIZE - 1    { self.mark_dirty(cx + 1, cy, cz); }
        if ly == 0                 { self.mark_dirty(cx, cy - 1, cz); }
        if ly == CHUNK_SIZE - 1    { self.mark_dirty(cx, cy + 1, cz); }
        if lz == 0                 { self.mark_dirty(cx, cy, cz - 1); }
        if lz == CHUNK_SIZE - 1    { self.mark_dirty(cx, cy, cz + 1); }
    }

    pub fn chunk_count(&self) -> usize { self.chunks.len() }
    pub fn lod_counts(&self) -> [usize; 3] { self.last_lod_counts }

    pub fn get_block(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        let cs = CHUNK_SIZE as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;
        match self.chunks.get(&(cx, cy, cz)) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => 0,
        }
    }

    pub fn remove_block(&mut self, wx: i32, wy: i32, wz: i32) {
        let cs = CHUNK_SIZE as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set(lx, ly, lz, 0);
            debug_log!("World", "remove_block", "Broke block at ({}, {}, {})", wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }

    pub fn place_block(&mut self, wx: i32, wy: i32, wz: i32, block_id: u8) {
        if block_id == 0 { return; }
        let cs = CHUNK_SIZE as i32;
        let cx = wx.div_euclid(cs);
        let cy = wy.div_euclid(cs);
        let cz = wz.div_euclid(cs);
        let lx = wx.rem_euclid(cs) as usize;
        let ly = wy.rem_euclid(cs) as usize;
        let lz = wz.rem_euclid(cs) as usize;

        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.set(lx, ly, lz, block_id);
            debug_log!("World", "place_block", "Placed block {} at ({}, {}, {})", block_id, wx, wy, wz);
        }
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }
}

// =============================================================================
// WorldWorker — runs World on a background thread
// =============================================================================

enum WorldRequest {
    Update {
        camera_pos:     Vec3,
        camera_forward: Vec3,
        ray_dir:        Option<Vec3>,
        block_ops:      Vec<BlockOp>,
    },
    Shutdown,
}

#[derive(Debug)]
pub struct ChunkBlockData {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    pub solid_positions: Vec<[u8; 3]>,
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
    pub physics_chunks: Vec<ChunkBlockData>,
    // Profiler data
    pub lod_counts: [usize; 3],
    pub dirty_count: usize,
    pub worker_total_us: u128,
}

pub struct WorldWorker {
    request_tx:   mpsc::Sender<WorldRequest>,
    result_rx:    mpsc::Receiver<WorldResult>,
    thread:       Option<thread::JoinHandle<()>>,
    busy:         bool,
    last_chunk_count: usize,
    pending_ops:  Vec<BlockOp>,
}

impl WorldWorker {
    pub fn new(atlas_layout: TextureAtlasLayout) -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WorldRequest>();
        let (result_tx,  result_rx)  = mpsc::channel::<WorldResult>();

        let handle = thread::Builder::new()
            .name("world-worker".into())
            .spawn(move || {
                let mut world = World::new(atlas_layout);

                loop {
                    match request_rx.recv() {
                        Ok(WorldRequest::Update {
                               camera_pos,
                               camera_forward,
                               ray_dir,
                               block_ops,
                           }) => {
                            let t_total = Instant::now();

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

                            let physics_chunks: Vec<ChunkBlockData> = if has_dirty {
                                meshes.iter()
                                    .filter_map(|(key, verts, _, _, _)| {
                                        if verts.is_empty() { return None; }
                                        let chunk = world.chunks.get(key)?;
                                        let mut solid = Vec::new();
                                        for x in 0..CHUNK_SIZE {
                                            for y in 0..CHUNK_SIZE {
                                                for z in 0..CHUNK_SIZE {
                                                    if chunk.get(x, y, z) != 0 {
                                                        solid.push([x as u8, y as u8, z as u8]);
                                                    }
                                                }
                                            }
                                        }
                                        Some(ChunkBlockData {
                                            cx: key.0, cy: key.1, cz: key.2,
                                            solid_positions: solid,
                                        })
                                    })
                                    .collect()
                            } else {
                                Vec::new()
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
                                physics_chunks,
                                lod_counts,
                                dirty_count,
                                worker_total_us,
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
            busy: false,
            last_chunk_count: 0,
            pending_ops: Vec::new(),
        }
    }

    pub fn request_update(
        &mut self,
        camera_pos:     Vec3,
        camera_forward: Vec3,
        ray_dir:        Option<Vec3>,
        block_ops:      Vec<BlockOp>,
    ) {
        self.pending_ops.extend(block_ops);

        if !self.busy {
            let ops = std::mem::take(&mut self.pending_ops);
            let _ = self.request_tx.send(WorldRequest::Update {
                camera_pos,
                camera_forward,
                ray_dir,
                block_ops: ops,
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