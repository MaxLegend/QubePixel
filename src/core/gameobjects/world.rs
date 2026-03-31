// =============================================================================
// QubePixel — World  (chunk streaming + Perlin-noise terrain)
// =============================================================================

use std::collections::HashMap;
use glam::Vec3;
use noise::{NoiseFn, Perlin};
use crate::debug_log;
use crate::core::gameobjects::block::BlockRegistry;
use crate::core::gameobjects::chunk::{Chunk, CHUNK_SIZE};
use crate::screens::game_3d_pipeline::Vertex3D;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// How many chunks to load in each horizontal direction from the camera.
const RENDER_DISTANCE: i32 = 3;

/// Vertical range relative to the camera's current chunk.
const VERTICAL_BELOW: i32 = 2; // chunks below camera chunk
const VERTICAL_ABOVE: i32 = 3; // chunks above camera chunk

/// World-space Y at which the noise maps to zero (the "sea level").
const SURFACE_BASE: f64 = 8.0;  // blocks
/// Half-amplitude of terrain height variation in blocks.
const SURFACE_AMP: f64 = 5.0;
/// Horizontal scale of the noise (larger = smoother hills).
const NOISE_SCALE: f64 = 48.0;

// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
pub struct World {
    chunks: HashMap<(i32, i32, i32), Chunk>,
    noise: Perlin,
    pub registry: BlockRegistry,
    grass_id: u8,
    dirt_id: u8,
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
            chunks: HashMap::new(),
            noise: Perlin::new(12345u32),
            registry,
            grass_id,
            dirt_id,
            stone_id,
        }
    }

    // -----------------------------------------------------------------------
    // Terrain helpers
    // -----------------------------------------------------------------------

    /// Returns the world-space surface Y for a block column (wx, wz).
    fn surface_height(&self, wx: i32, wz: i32) -> i32 {
        let nx = wx as f64 / NOISE_SCALE;
        let nz = wz as f64 / NOISE_SCALE;
        // noise::get returns a value in approximately [-1, 1]
        let h = self.noise.get([nx, nz]);
        (SURFACE_BASE + h * SURFACE_AMP).round() as i32
    }

    // -----------------------------------------------------------------------
    // Chunk generation
    // -----------------------------------------------------------------------

    /// Fill one chunk from noise.
    fn generate_chunk(&self, cx: i32, cy: i32, cz: i32) -> Chunk {
        let mut chunk = Chunk::new(cx, cy, cz);

        for bx in 0..CHUNK_SIZE {
            for bz in 0..CHUNK_SIZE {
                let wx = cx * CHUNK_SIZE as i32 + bx as i32;
                let wz = cz * CHUNK_SIZE as i32 + bz as i32;
                let surface = self.surface_height(wx, wz);

                for by in 0..CHUNK_SIZE {
                    let wy = cy * CHUNK_SIZE as i32 + by as i32;

                    let id = if wy > surface {
                        0                  // air
                    } else if wy == surface {
                        self.grass_id
                    } else if wy >= surface - 3 {
                        self.dirt_id
                    } else {
                        self.stone_id
                    };

                    chunk.set_gen(bx, by, bz, id);
                }
            }
        }

        chunk
    }

    // -----------------------------------------------------------------------
    // Streaming update — call once per frame
    // -----------------------------------------------------------------------

    /// Load chunks around `camera_pos` and unload ones that are too far away.
    ///
    /// Returns `true` if the chunk set (and therefore the mesh) changed.
    pub fn update(&mut self, camera_pos: Vec3) -> bool {
        let cam_cx = (camera_pos.x / CHUNK_SIZE as f32).floor() as i32;
        let cam_cy = (camera_pos.y / CHUNK_SIZE as f32).floor() as i32;
        let cam_cz = (camera_pos.z / CHUNK_SIZE as f32).floor() as i32;

        let mut changed = false;

        // --- Load missing chunks within the frustum ---
        let mut to_generate: Vec<(i32, i32, i32)> = Vec::new();
        for dx in -RENDER_DISTANCE..=RENDER_DISTANCE {
            for dz in -RENDER_DISTANCE..=RENDER_DISTANCE {
                for dy in -VERTICAL_BELOW..=VERTICAL_ABOVE {
                    let key = (cam_cx + dx, cam_cy + dy, cam_cz + dz);
                    if !self.chunks.contains_key(&key) {
                        to_generate.push(key);
                    }
                }
            }
        }

        for key in to_generate {
            let chunk = self.generate_chunk(key.0, key.1, key.2);
            self.chunks.insert(key, chunk);
            changed = true;
        }

        // --- Unload far chunks ---
        let evict_h = RENDER_DISTANCE + 2;
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
                "Chunk set changed — loaded chunks: {}", self.chunks.len()
            );
        }

        changed
    }

    // -----------------------------------------------------------------------
    // Mesh build — call after `update()` returns true
    // -----------------------------------------------------------------------

    /// Build (or rebuild) the CPU-side meshes for all loaded chunks.
    ///
    /// Clears `mesh_dirty` on each chunk.  Returns one `(vertices, indices)`
    /// pair per non-empty chunk.
    pub fn build_meshes(&mut self) -> Vec<(Vec<Vertex3D>, Vec<u32>)> {
        let registry = &self.registry;
        let mut result = Vec::new();

        for chunk in self.chunks.values_mut() {
            let (verts, idxs) = chunk.build_mesh(registry);
            if !verts.is_empty() {
                result.push((verts, idxs));
            }
            chunk.mesh_dirty = false;
        }

        debug_log!(
            "World", "build_meshes",
            "Built {} non-empty chunk meshes from {} total chunks",
            result.len(), self.chunks.len()
        );

        result
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}
