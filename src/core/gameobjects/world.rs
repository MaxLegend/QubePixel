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
}

impl World {
    pub fn new() -> Self {
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
            debug_log!(
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
    /// Returns `(meshes, build_time_us)` — mesh data and wall-clock profiling.
    pub fn build_meshes(
        &mut self,
    ) -> (Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])>, u128) {
        let s = CHUNK_SIZE as f32;
        let total = self.chunks.len();

        let chunks   = &mut self.chunks;
        let registry = &self.registry;

        let t0 = Instant::now();

        let result: Vec<((i32, i32, i32), Vec<Vertex3D>, Vec<u32>, [f32; 3], [f32; 3])> = chunks
            .par_iter_mut()
            .filter_map(|(key, chunk)| {
                if !chunk.mesh_dirty {
                    return None;
                }

                let (verts, idxs) = chunk.build_mesh(registry);
                chunk.mesh_dirty = false;

                if verts.is_empty() {
                    return None;
                }

                let min = [
                    chunk.cx as f32 * s,
                    chunk.cy as f32 * s,
                    chunk.cz as f32 * s,
                ];
                let max = [min[0] + s, min[1] + s, min[2] + s];
                Some((*key, verts, idxs, min, max))
            })
            .collect();

        let build_time_us = t0.elapsed().as_micros();

        debug_log!(
            "World", "build_meshes",
            "[PERF] build={:.2}ms dirty={}/{} total_chunks={}",
            build_time_us as f64 / 1000.0,
            result.len(), total, total
        );

        (result, build_time_us)
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

// =============================================================================
// WorldWorker — runs World on a background thread
// =============================================================================

/// Message sent from main thread to the background world thread.
enum WorldRequest {
    Update { camera_pos: Vec3 },
    Shutdown,
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
}

/// Owns a background thread that runs `World::update()` + `World::build_meshes()`
/// asynchronously. The main render thread never blocks on chunk generation or
/// mesh building — it only does a non-blocking `try_recv()` and then a cheap
/// GPU-buffer upload when data is ready.
pub struct WorldWorker {
    request_tx: mpsc::Sender<WorldRequest>,
    result_rx:  mpsc::Receiver<WorldResult>,
    thread:     Option<thread::JoinHandle<()>>,
    /// `true` while a request is being processed — avoids queuing up work.
    busy: bool,
    /// Latest chunk count reported by the background thread (for debug overlay).
    last_chunk_count: usize,
}

impl WorldWorker {
    pub fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<WorldRequest>();
        let (result_tx,  result_rx)  = mpsc::channel::<WorldResult>();

        let handle = thread::Builder::new()
            .name("world-worker".into())
            .spawn(move || {
                let mut world = World::new();

                loop {
                    match request_rx.recv() {
                        Ok(WorldRequest::Update { camera_pos }) => {
                            let t_total = Instant::now();

                            let (changed, evicted, gen_time, pending) =
                                world.update(camera_pos);

                            // ★ Always build meshes + send result, even if nothing changed.
                            // This ensures `busy` gets cleared so the main thread can
                            // send the next request when the camera moves.
                            let (meshes, mesh_time) = if changed {
                                world.build_meshes()
                            } else {
                                (Vec::new(), 0)
                            };

                            let total_time_us = t_total.elapsed().as_micros();
                            let _ = result_tx.send(WorldResult {
                                meshes,
                                evicted,
                                chunk_count: world.chunk_count(),
                                gen_time_us: gen_time,
                                mesh_time_us: mesh_time,
                                pending,
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
        }
    }

    /// Send a world-update request. No-op if a previous request is still in flight.
    pub fn request_update(&mut self, camera_pos: Vec3) {
        if !self.busy {
            let _ = self.request_tx.send(WorldRequest::Update { camera_pos });
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
