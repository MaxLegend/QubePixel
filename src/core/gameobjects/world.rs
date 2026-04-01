// =============================================================================
// QubePixel — World  (chunk streaming + Perlin-noise terrain)
// =============================================================================
//
// CHANGELOG vs original:
//   • update() returns (bool, Vec<(i32,i32,i32)>) — changed flag + evicted keys
//   • build_meshes() skips non-dirty chunks (dirty-tracking)
//   • build_meshes() runs via rayon par_iter_mut() for parallel mesh building
//   • generate_chunk_fn() explicitly sets mesh_dirty = true
//   • Staggered loading: MAX_CHUNKS_PER_FRAME limits generation per update
//   • WorldWorker: background thread + channels for non-blocking world ops
//   • Profiling: std::time::Instant on update(), build_meshes(), WorldResult
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
use crate::core::gameobjects::texture_atlas::{TextureAtlas, TextureAtlasLayout};
use crate::core::raycast::{dda_raycast, RaycastResult, MAX_RAYCAST_DISTANCE};
// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Vertical range relative to the camera's current chunk.
const VERTICAL_BELOW: i32 = 2;
const VERTICAL_ABOVE: i32 = 3;

/// World-space Y at which the noise maps to zero (the "sea level").
const SURFACE_BASE: f64 = 8.0;
/// Half-amplitude of terrain height variation in blocks.
const SURFACE_AMP: f64  = 5.0;
/// Horizontal scale of the noise (larger = smoother hills).
const NOISE_SCALE: f64  = 48.0;

/// ★ Staggered loading — max chunks generated per World::update() call.
/// Prevents frame spikes when a large number of chunks need generation.
/// With the background thread, this limits how long ONE worker iteration takes.
const MAX_CHUNKS_PER_FRAME: usize = 64;

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
    /// Cloneable UV layout for the texture atlas (used in background thread).
    atlas_layout: TextureAtlasLayout,
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
        }
    }

    // -----------------------------------------------------------------------
    // Terrain helpers (pure / no &mut self)
    // -----------------------------------------------------------------------

    fn surface_height_fn(noise: &Perlin, wx: i32, wz: i32) -> i32 {
        let nx = wx as f64 / NOISE_SCALE;
        let nz = wz as f64 / NOISE_SCALE;
        let h  = noise.get([nx, nz]);
        (SURFACE_BASE + h * SURFACE_AMP).round() as i32
    }

    // -----------------------------------------------------------------------
    // Chunk generation — static so rayon closures can borrow independently
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Streaming update — rayon parallel generation + staggered loading
    // -----------------------------------------------------------------------

    /// Load chunks around `camera_pos` and unload ones too far away.
    /// Returns `(changed, evicted, gen_time_us, pending)`:
    ///   - `changed`   — true if the chunk set changed
    ///   - `evicted`   — list of chunk keys removed (GPU buffers must be freed)
    ///   - `gen_time`  — wall-clock microseconds for chunk generation (profiling)
    ///   - `pending`   — how many chunks still need generation (staggered)
    pub fn update(
        &mut self,
        camera_pos: Vec3,
    ) -> (bool, Vec<(i32, i32, i32)>, u128, usize) {
        let rd = config::render_distance();

        let cam_cx = (camera_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let cam_cy = (camera_pos.y / CHUNK_SIZE as f32).floor() as i32;
        let cam_cz = (camera_pos.z / CHUNK_SIZE as f32).floor() as i32;

        // --- Collect keys that need to be generated ---
        let mut to_generate: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -rd..=rd {
            for dz in -rd..=rd {
                for dy in -VERTICAL_BELOW..=VERTICAL_ABOVE {
                    let key = (cam_cx + dx, cam_cy + dy, cam_cz + dz);
                    if !self.chunks.contains_key(&key) {
                        to_generate.push(key);
                    }
                }
            }
        }

        // ★ Staggered loading: sort by distance so closest chunks load first
        to_generate.sort_by_key(|&(cx, cy, cz)| {
            let dx = (cx - cam_cx) as f64;
            let dy = (cy - cam_cy) as f64;
            let dz = (cz - cam_cz) as f64;
            (dx * dx + dy * dy + dz * dz) as i64
        });

        let total_pending = to_generate.len();

        // ★ Limit how many chunks we generate this frame
        let gen_count = to_generate.len().min(MAX_CHUNKS_PER_FRAME);
        let was_truncated = to_generate.len() > MAX_CHUNKS_PER_FRAME;
        to_generate.truncate(gen_count);

        let mut changed = false;
        let mut gen_time_us: u128 = 0;

        if !to_generate.is_empty() {
            let t0 = Instant::now();

            // --- Parallel generation via rayon ---
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

            // Before inserting, mark existing neighbor chunks dirty
            // so they rebuild with correct cross-chunk boundary culling.
            for &(key, _) in &new_chunks {
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

        // --- Unload far chunks ---
        let evict_h = rd + 2;

        let evicted: Vec<(i32, i32, i32)> = self.chunks.keys()
            .filter(|&&(cx, cy, cz)| {
                let dx = (cx - cam_cx).abs();
                let dz = (cz - cam_cz).abs();
                let too_far_h = dx > evict_h || dz > evict_h;
                let too_far_v = cy < cam_cy - VERTICAL_BELOW - 1
                    || cy > cam_cy + VERTICAL_ABOVE + 1;
                too_far_h || too_far_v
            })
            .copied()
            .collect();

        let before = self.chunks.len();
        self.chunks.retain(|&(cx, cy, cz), _| {
            let dx = (cx - cam_cx).abs();
            let dz = (cz - cam_cz).abs();
            let too_far_h = dx > evict_h || dz > evict_h;
            let too_far_v = cy < cam_cy - VERTICAL_BELOW - 1
                || cy > cam_cy + VERTICAL_ABOVE + 1;
            !too_far_h && !too_far_v
        });

        if self.chunks.len() != before {
            changed = true;
        }

        if changed {
            flow_debug_log!(
                "World", "update",
                "[PERF] gen={:.2}ms chunks={}/{} loaded={} evicted={} pending={}",
                gen_time_us as f64 / 1000.0,
                gen_count, total_pending,
                self.chunks.len(), evicted.len(),
                if was_truncated { total_pending - gen_count } else { 0 }
            );
        }

        (changed, evicted, gen_time_us, if was_truncated { total_pending - gen_count } else { 0 })
    }

    // -----------------------------------------------------------------------
    // Mesh build — returns meshes ONLY for dirty chunks (rayon parallel)
    // -----------------------------------------------------------------------

    /// Builds GPU-ready mesh data for chunks that are marked `mesh_dirty`.
    /// Uses cross-chunk neighbor lookup to cull faces on chunk boundaries.
    /// Returns `(meshes, build_time_us)`.
    pub fn build_meshes(
        &mut self,
    ) -> (Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>, u128) {
        let total = self.chunks.len();
        let t0 = Instant::now();

        // ── Phase 1: collect dirty keys & clear flags (mutable access) ──
        let dirty_keys: Vec<(i32, i32, i32)> = self.chunks.iter()
            .filter_map(|(&key, chunk)| {
                if chunk.mesh_dirty { Some(key) } else { None }
            })
            .collect();

        for &key in &dirty_keys {
            if let Some(chunk) = self.chunks.get_mut(&key) {
                chunk.mesh_dirty = false;
            }
        }

        let dirty_count = dirty_keys.len();

        // ── Phase 2: parallel mesh build with read-only chunk access ──
        // Re-borrow self.chunks as immutable so rayon closures can read
        // neighboring chunks for cross-boundary face culling.
        let chunks   = &self.chunks;
        let registry = &self.registry;
        let atlas    = &self.atlas_layout;
        let cs       = CHUNK_SIZE as i32;

        let result: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])> = dirty_keys
            .into_par_iter()
            .filter_map(|key| {
                let chunk = match chunks.get(&key) {
                    Some(c) => c,
                    None    => return None,
                };

                // Neighbor lookup: world coords → block ID.
                // Returns 0 (Air) when the chunk is not loaded → face is
                // drawn (conservative).
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

                let (verts, idxs) = chunk.build_mesh(registry, atlas, get_neighbor);

                if verts.is_empty() { return None; }

                let s   = CHUNK_SIZE as f32;
                let min = [chunk.cx as f32 * s, chunk.cy as f32 * s, chunk.cz as f32 * s];
                let max = [min[0] + s, min[1] + s, min[2] + s];
                Some((key, verts, idxs, min, max))
            })
            .collect();

        let build_time_us = t0.elapsed().as_micros();

        flow_debug_log!(
            "World", "build_meshes",
            "[PERF] build={:.2}ms dirty={}/{} total_chunks={}",
            build_time_us as f64 / 1000.0,
            result.len(), dirty_count, total
        );

        (result, build_time_us)
    }
    // -----------------------------------------------------------------------
    // Neighbor invalidation
    // -----------------------------------------------------------------------

    /// Marks a chunk as dirty for mesh rebuild if it exists.
    fn mark_dirty(&mut self, cx: i32, cy: i32, cz: i32) {
        if let Some(chunk) = self.chunks.get_mut(&(cx, cy, cz)) {
            chunk.mesh_dirty = true;
        }
    }

    /// If `lx/ly/lz` is on a chunk boundary, mark the adjacent chunk dirty
    /// so it rebuilds its mesh with updated cross-chunk face culling.
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
    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Get the block ID at world coordinates (wx, wy, wz).
    /// Returns 0 (Air) if the chunk is not loaded.
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

    /// Remove the block at world coordinates (set to Air = 0).
    /// Marks the containing chunk as dirty for mesh rebuild.
    /// No-op if the chunk is not loaded.
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
            debug_log!(
                "World", "remove_block",
                "Broke block at ({}, {}, {})", wx, wy, wz
            );
        }

        // Rebuild adjacent chunk's mesh if block was on a boundary
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }

    /// Place a block at world coordinates.
    /// Marks the containing chunk as dirty for mesh rebuild.
    /// No-op if the chunk is not loaded or block_id is 0.
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
            debug_log!(
                "World", "place_block",
                "Placed block {} at ({}, {}, {})", block_id, wx, wy, wz
            );
        }

        // Rebuild adjacent chunk's mesh if block was on a boundary
        self.dirty_boundary_neighbors(cx, cy, cz, lx, ly, lz);
    }
}

// =============================================================================
// WorldWorker — runs World on a background thread
// =============================================================================

/// Message sent from main thread to the background world thread.
enum WorldRequest {
    Update {
        camera_pos: Vec3,
        ray_dir: Option<Vec3>,
        block_ops: Vec<BlockOp>,
    },
    Shutdown,
}
/// Solid block positions for a chunk, sent from the background thread
/// to the main thread so PhysicsWorld can create compound colliders.
#[derive(Debug)]
pub struct ChunkBlockData {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// Local (x, y, z) positions of solid blocks within the chunk.
    pub solid_positions: Vec<[u8; 3]>,
}

/// A block modification operation queued by player input.
#[derive(Debug, Clone)]
pub enum BlockOp {
    Break { x: i32, y: i32, z: i32 },
    Place { x: i32, y: i32, z: i32, block_id: u8 },
}
/// Completed work sent back from the background world thread.
pub struct WorldResult {
    pub meshes: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>,
    pub evicted: Vec<(i32, i32, i32)>,
    pub chunk_count: usize,
    /// Wall-clock microseconds for chunk generation (profiling).
    pub gen_time_us: u128,
    /// Wall-clock microseconds for mesh building (profiling).
    pub mesh_time_us: u128,
    /// Chunks still pending generation (staggered loading).
    pub pending: usize,
    /// Raycast result — which block the camera is aiming at (if requested).
    pub raycast_result: Option<RaycastResult>,
    /// Solid block data for chunks that were (re)generated — used to
    /// sync physics colliders on the main thread.
    pub physics_chunks: Vec<ChunkBlockData>,
}

/// Owns a background thread that runs `World::update()` + `World::build_meshes()`
/// asynchronously. The main render thread never blocks on chunk generation or
/// mesh building — it only does a non-blocking `try_recv()` and then a cheap
/// GPU-buffer upload when data is ready.
pub struct WorldWorker {
    request_tx:   mpsc::Sender<WorldRequest>,
    result_rx:    mpsc::Receiver<WorldResult>,
    thread:       Option<thread::JoinHandle<()>>,
    busy:         bool,
    last_chunk_count: usize,
    /// Операции, накопленные пока worker был занят — отправляются при следующем свободном цикле.
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
                        Ok(WorldRequest::Update { camera_pos, ray_dir, block_ops }) => {
                            let t_total = Instant::now();

                            let (changed, evicted, gen_time, pending) =
                                world.update(camera_pos);

                            // ★ Process block operations (break / place)
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

                            // ★ Always build meshes + send result, even if nothing changed.
                            // This ensures `busy` gets cleared so the main thread can
                            // send the next request when the camera moves.
                            let has_dirty = changed || has_block_ops;
                            let (meshes, mesh_time) = if has_dirty {
                                world.build_meshes()
                            } else {
                                (Vec::new(), 0)
                            };

                            // ★ Raycast — find the block the camera is aiming at.
                            let raycast_result = ray_dir.and_then(|dir| {
                                dda_raycast(
                                    camera_pos,
                                    dir,
                                    MAX_RAYCAST_DISTANCE,
                                    |wx, wy, wz| world.get_block(wx, wy, wz),
                                )
                            });
                            // ★ Extract solid block data for physics collider sync.
                            // Iterates meshes keys (dirty chunks) and pulls block
                            // data from world.chunks while the borrow is valid.
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
                            let total_time_us = t_total.elapsed().as_micros();
                            let _ = result_tx.send(WorldResult {
                                meshes,
                                evicted,
                                chunk_count: world.chunk_count(),
                                gen_time_us: gen_time,
                                mesh_time_us: mesh_time,
                                pending,
                                raycast_result,
                                physics_chunks,
                            });

                            flow_debug_log!(
        "WorldWorker", "thread",
        "[PERF] worker_total={:.2}ms changed={} pending={}",
        total_time_us as f64 / 1000.0,
        changed,
        pending
    );
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

    /// Send a world-update request. No-op if a previous request is still in flight.

    /// `block_ops` are queued break/place operations applied this frame.
    pub fn request_update(
        &mut self,
        camera_pos: Vec3,
        ray_dir: Option<Vec3>,
        block_ops: Vec<BlockOp>,
    ) {
        // Всегда накапливаем операции, даже если worker занят
        self.pending_ops.extend(block_ops);

        if !self.busy {
            let ops = std::mem::take(&mut self.pending_ops);
            let _ = self.request_tx.send(WorldRequest::Update {
                camera_pos,
                ray_dir,
                block_ops: ops,
            });
            self.busy = true;
        }
    }

    /// Non-blocking poll for a completed result.
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

    /// Latest chunk count from the most recent completed world update.
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
