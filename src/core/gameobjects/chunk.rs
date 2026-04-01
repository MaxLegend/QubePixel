// =============================================================================
// QubePixel — Chunk  (16 × 16 × 16 cubic voxel chunk)
// =============================================================================

use crate::{ flow_debug_log};
use crate::core::gameobjects::block::BlockRegistry;
use crate::screens::game_3d_pipeline::Vertex3D;
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;
pub const CHUNK_SIZE: usize = 16;

pub struct Chunk {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// blocks[x][y][z] — block ID (0 = Air)
    blocks: Box<[[[u8; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]>,
    pub mesh_dirty: bool,
}

impl Chunk {
    pub fn new(cx: i32, cy: i32, cz: i32) -> Self {
        Self {
            cx,
            cy,
            cz,
            blocks: Box::new([[[0u8; CHUNK_SIZE]; CHUNK_SIZE]; CHUNK_SIZE]),
            mesh_dirty: true,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.blocks[x][y][z]
    }

    #[allow(dead_code)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[x][y][z] = block_id;
        self.mesh_dirty = true;
    }

    /// Write a block without marking the chunk dirty (used during terrain generation).
    pub(crate) fn set_gen(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[x][y][z] = block_id;
    }

    pub fn is_all_air(&self) -> bool {
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    if self.blocks[x][y][z] != 0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Full-detail mesh (LOD 0)
    // -----------------------------------------------------------------------

    /// Build a visible-face mesh for this chunk.
    /// Returns `(vertices, u32_indices)`. Empty if chunk is all-air.
    ///
    /// `get_neighbor` — closure receiving **world coordinates** and returning
    /// the block ID at that position. Used to cull faces against blocks in
    /// adjacent chunks (returns 0 = Air for unloaded chunks → conservative).
    pub fn build_mesh<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        get_neighbor: F,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> u8,
    {
        self.build_mesh_lod(registry, atlas, get_neighbor, 0)
    }

    // -----------------------------------------------------------------------
    // LOD mesh builder
    // -----------------------------------------------------------------------

    /// Build a mesh with configurable LOD level.
    ///
    /// * `lod_level = 0` → full detail (step = 1, 16³ cells)
    /// * `lod_level = 1` → 2× coarser  (step = 2,  8³ cells, ~1/4–1/8 faces)
    /// * `lod_level = 2` → 4× coarser  (step = 4,  4³ cells, ~1/16–1/64 faces)
    ///
    /// Each "super-block" is represented by the dominant non-air block ID
    /// found in the step³ region.  Faces are emitted at `step × step` scale.
    pub fn build_mesh_lod<F>(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
        get_neighbor: F,
        lod_level: u8,
    ) -> (Vec<Vertex3D>, Vec<u32>)
    where
        F: Fn(i32, i32, i32) -> u8,
    {
        if self.is_all_air() {
            return (Vec::new(), Vec::new());
        }

        let step = 1usize << (lod_level.min(2) as usize); // 1, 2, or 4
        let cells = CHUNK_SIZE / step;
        let scale = step as f32;

        let wx_base = self.cx * CHUNK_SIZE as i32;
        let wy_base = self.cy * CHUNK_SIZE as i32;
        let wz_base = self.cz * CHUNK_SIZE as i32;

        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(if lod_level == 0 { 1024 } else { 256 });
        let mut indices: Vec<u32> = Vec::with_capacity(if lod_level == 0 { 1536 } else { 384 });

        for cx in 0..cells {
            for cy in 0..cells {
                for cz in 0..cells {
                    // Local block origin of this super-block
                    let bx0 = cx * step;
                    let by0 = cy * step;
                    let bz0 = cz * step;

                    // Find the dominant non-air block in this super-block
                    let block_id = self.dominant_block(bx0, by0, bz0, step);
                    if block_id == 0 { continue; }

                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    if !def.solid { continue; }

                    // World-space origin of this super-block
                    let ox = wx_base as f32 + bx0 as f32;
                    let oy = wy_base as f32 + by0 as f32;
                    let oz = wz_base as f32 + bz0 as f32;

                    // Integer world coords for neighbor lookup (center of super-block)
                    let wx = wx_base + bx0 as i32;
                    let wy = wy_base + by0 as i32;
                    let wz = wz_base + bz0 as i32;

                    // +X face (dir = 0)
                    if self.is_adjacent_air(bx0 + step, by0, bz0, step, 0, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(0);
                        let uv = atlas.uv_for(def.texture_for_face(0).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 0, c, uv, scale);
                    }
                    // -X face (dir = 1)
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 1, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(1);
                        let uv = atlas.uv_for(def.texture_for_face(1).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 1, c, uv, scale);
                    }
                    // +Y face (dir = 2)
                    if self.is_adjacent_air(bx0, by0 + step, bz0, step, 2, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(2);
                        let uv = atlas.uv_for(def.texture_for_face(2).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 2, c, uv, scale);
                    }
                    // -Y face (dir = 3)
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 3, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(3);
                        let uv = atlas.uv_for(def.texture_for_face(3).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 3, c, uv, scale);
                    }
                    // +Z face (dir = 4)
                    if self.is_adjacent_air(bx0, by0, bz0 + step, step, 4, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(4);
                        let uv = atlas.uv_for(def.texture_for_face(4).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 4, c, uv, scale);
                    }
                    // -Z face (dir = 5)
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 5, wx, wy, wz, &get_neighbor) {
                        let c = def.color_for_face(5);
                        let uv = atlas.uv_for(def.texture_for_face(5).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 5, c, uv, scale);
                    }
                }
            }
        }

        flow_debug_log!(
            "Chunk", "build_mesh_lod",
            "Chunk ({},{},{}) lod={} step={} => {} verts, {} indices",
            self.cx, self.cy, self.cz, lod_level, step,
            vertices.len(), indices.len()
        );

        (vertices, indices)
    }

    // -----------------------------------------------------------------------
    // LOD helpers
    // -----------------------------------------------------------------------

    /// Find the dominant (most common) non-air block in a step³ region.
    /// Falls back to center sample for speed when step > 1.
    fn dominant_block(&self, bx0: usize, by0: usize, bz0: usize, step: usize) -> u8 {
        if step == 1 {
            return self.blocks[bx0][by0][bz0];
        }

        // For LOD: sample the center block first (fast path)
        let mid = step / 2;
        let cx = (bx0 + mid).min(CHUNK_SIZE - 1);
        let cy = (by0 + mid).min(CHUNK_SIZE - 1);
        let cz = (bz0 + mid).min(CHUNK_SIZE - 1);
        let center = self.blocks[cx][cy][cz];
        if center != 0 { return center; }

        // If center is air, scan the region for any solid block
        for dx in 0..step {
            for dy in 0..step {
                for dz in 0..step {
                    let x = bx0 + dx;
                    let y = by0 + dy;
                    let z = bz0 + dz;
                    if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
                        let id = self.blocks[x][y][z];
                        if id != 0 { return id; }
                    }
                }
            }
        }
        0
    }

    /// Check if the positive-direction adjacent super-block is "air" for face culling.
    /// `pos` is the start coordinate along the face axis (already offset by step).
    fn is_adjacent_air<F>(
        &self,
        bx: usize, by: usize, bz: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        get_neighbor: &F,
    ) -> bool
    where F: Fn(i32, i32, i32) -> u8
    {
        // The coordinate we're checking is already bx0 + step (or similar)
        let (check_local, world_offset) = match dir {
            0 => (bx < CHUNK_SIZE, (step as i32, 0, 0)),
            2 => (by < CHUNK_SIZE, (0, step as i32, 0)),
            4 => (bz < CHUNK_SIZE, (0, 0, step as i32)),
            _ => return true,
        };

        if check_local {
            // Check within this chunk — sample center of adjacent super-block
            let (sx, sy, sz) = match dir {
                0 => (bx, by + step / 2, bz + step / 2),
                2 => (bx + step / 2, by, bz + step / 2),
                4 => (bx + step / 2, by + step / 2, bz),
                _ => (bx, by, bz),
            };
            let sx = sx.min(CHUNK_SIZE - 1);
            let sy = sy.min(CHUNK_SIZE - 1);
            let sz = sz.min(CHUNK_SIZE - 1);
            self.blocks[sx][sy][sz] == 0
        } else {
            get_neighbor(
                wx + world_offset.0,
                wy + world_offset.1,
                wz + world_offset.2,
            ) == 0
        }
    }

    /// Check negative-direction adjacency.
    fn is_adjacent_air_neg<F>(
        &self,
        bx0: usize, by0: usize, bz0: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        get_neighbor: &F,
    ) -> bool
    where F: Fn(i32, i32, i32) -> u8
    {
        let (can_check_local, world_offset) = match dir {
            1 => (bx0 >= step, (-1i32, 0, 0)),
            3 => (by0 >= step, (0, -1i32, 0)),
            5 => (bz0 >= step, (0, 0, -1i32)),
            _ => return true,
        };

        if can_check_local {
            let (sx, sy, sz) = match dir {
                1 => (bx0 - 1, by0 + step / 2, bz0 + step / 2),
                3 => (bx0 + step / 2, by0 - 1, bz0 + step / 2),
                5 => (bx0 + step / 2, by0 + step / 2, bz0 - 1),
                _ => (bx0, by0, bz0),
            };
            let sx = sx.min(CHUNK_SIZE - 1);
            let sy = sy.min(CHUNK_SIZE - 1);
            let sz = sz.min(CHUNK_SIZE - 1);
            self.blocks[sx][sy][sz] == 0
        } else {
            get_neighbor(
                wx + world_offset.0,
                wy + world_offset.1,
                wz + world_offset.2,
            ) == 0
        }
    }
}

// ---------------------------------------------------------------------------
// emit_face_lod — emits a single face of size `scale × scale`
//
// All faces wound CCW from outside (FrontFace::Ccw + CullMode::Back).
// ---------------------------------------------------------------------------
fn emit_face_lod(
    verts: &mut Vec<Vertex3D>,
    idxs:  &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:   u8,
    color: [f32; 3],
    uv:    (f32, f32, f32, f32),
    scale: f32,
) {
    let base = verts.len() as u32;
    let s = scale;

    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        // +X
        0 => ([1.0, 0.0, 0.0], [
            [ox+s, oy,   oz+s],
            [ox+s, oy,   oz  ],
            [ox+s, oy+s, oz  ],
            [ox+s, oy+s, oz+s],
        ]),
        // -X
        1 => ([-1.0, 0.0, 0.0], [
            [ox, oy,   oz  ],
            [ox, oy,   oz+s],
            [ox, oy+s, oz+s],
            [ox, oy+s, oz  ],
        ]),
        // +Y (top)
        2 => ([0.0, 1.0, 0.0], [
            [ox,   oy+s, oz+s],
            [ox+s, oy+s, oz+s],
            [ox+s, oy+s, oz  ],
            [ox,   oy+s, oz  ],
        ]),
        // -Y (bottom)
        3 => ([0.0, -1.0, 0.0], [
            [ox,   oy, oz  ],
            [ox+s, oy, oz  ],
            [ox+s, oy, oz+s],
            [ox,   oy, oz+s],
        ]),
        // +Z
        4 => ([0.0, 0.0, 1.0], [
            [ox,   oy,   oz+s],
            [ox+s, oy,   oz+s],
            [ox+s, oy+s, oz+s],
            [ox,   oy+s, oz+s],
        ]),
        // -Z
        _ => ([0.0, 0.0, -1.0], [
            [ox+s, oy,   oz],
            [ox,   oy,   oz],
            [ox,   oy+s, oz],
            [ox+s, oy+s, oz],
        ]),
    };

    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = match dir {
        2 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        3 => [ [u0, v0], [u1, v0], [u1, v1], [u0, v1] ],
        1 => [ [u0, v1], [u1, v1], [u1, v0], [u0, v0] ],
        4 => [ [u1, v1], [u0, v1], [u0, v0], [u1, v0] ],
        _  => [ [u0, v0], [u1, v0], [u1, v1], [u0, v1] ],
    };

    for i in 0..4 {
        verts.push(Vertex3D {
            position: corners[i],
            normal,
            color,
            texcoord: uvs[i],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
