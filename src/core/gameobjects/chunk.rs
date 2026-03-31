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

    /// Build a visible-face mesh for this chunk.
    /// Returns `(vertices, u32_indices)`. Empty if chunk is all-air.
    ///
    /// Uses per-face colours from `BlockDefinition::color_for_face()` so
    /// blocks like grass can have green tops and brown sides.
    pub fn build_mesh(
        &self,
        registry: &BlockRegistry,
        atlas: &TextureAtlasLayout,
    ) -> (Vec<Vertex3D>, Vec<u32>) {
        if self.is_all_air() {
            return (Vec::new(), Vec::new());
        }

        let wx_base = (self.cx * CHUNK_SIZE as i32) as f32;
        let wy_base = (self.cy * CHUNK_SIZE as i32) as f32;
        let wz_base = (self.cz * CHUNK_SIZE as i32) as f32;

        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(1024);
        let mut indices: Vec<u32>       = Vec::with_capacity(1536);

        for bx in 0..CHUNK_SIZE {
            for by in 0..CHUNK_SIZE {
                for bz in 0..CHUNK_SIZE {
                    let block_id = self.blocks[bx][by][bz];
                    if block_id == 0 { continue; }

                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None    => continue,
                    };
                    if !def.solid { continue; }

                    let ox = wx_base + bx as f32;
                    let oy = wy_base + by as f32;
                    let oz = wz_base + bz as f32;

                    // +X face (dir = 0)
                    if bx + 1 >= CHUNK_SIZE || self.blocks[bx+1][by][bz] == 0 {
                        let c = def.color_for_face(0);
                        let uv = atlas.uv_for(
                            def.texture_for_face(0).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 0, c, uv);
                    }
                    // -X face (dir = 1)
                    if bx == 0 || self.blocks[bx-1][by][bz] == 0 {
                        let c = def.color_for_face(1);
                        let uv = atlas.uv_for(
                            def.texture_for_face(1).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 1, c, uv);
                    }
                    // +Y face / top (dir = 2)
                    if by + 1 >= CHUNK_SIZE || self.blocks[bx][by+1][bz] == 0 {
                        let c = def.color_for_face(2);
                        let uv = atlas.uv_for(
                            def.texture_for_face(2).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 2, c, uv);
                    }
                    // -Y face / bottom (dir = 3)
                    if by == 0 || self.blocks[bx][by-1][bz] == 0 {
                        let c = def.color_for_face(3);
                        let uv = atlas.uv_for(
                            def.texture_for_face(3).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 3, c, uv);
                    }
                    // +Z face (dir = 4)
                    if bz + 1 >= CHUNK_SIZE || self.blocks[bx][by][bz+1] == 0 {
                        let c = def.color_for_face(4);
                        let uv = atlas.uv_for(
                            def.texture_for_face(4).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 4, c, uv);
                    }
                    // -Z face (dir = 5)
                    if bz == 0 || self.blocks[bx][by][bz-1] == 0 {
                        let c = def.color_for_face(5);
                        let uv = atlas.uv_for(
                            def.texture_for_face(5).unwrap_or("")
                        );
                        emit_face(&mut vertices, &mut indices,
                                  ox, oy, oz, 5, c, uv);
                    }
                }
            }
        }

        flow_debug_log!(
            "Chunk", "build_mesh",
            "Chunk ({},{},{}) => {} verts, {} indices",
            self.cx, self.cy, self.cz, vertices.len(), indices.len()
        );

        (vertices, indices)
    }
}

// ---------------------------------------------------------------------------
// emit_face
//
// All faces wound CCW from outside (FrontFace::Ccw + CullMode::Back).
// Cross-product verification (edge1 × edge2 = outward normal):
//   +X : (0,0,-1)×(0,1,-1) → (+1, 0, 0) ✓
//   -X : (0,0,+1)×(0,1,+1) → (-1, 0, 0) ✓
//   +Y : (1,0, 0)×(1,0,-1) → ( 0,+1, 0) ✓  (FIXED — was CW)
//   -Y : (1,0, 0)×(1,0,+1) → ( 0,-1, 0) ✓  (FIXED — was CW)
//   +Z : (1,0, 0)×(1,1, 0) → ( 0, 0,+1) ✓
//   -Z : (-1,0,0)×(-1,1,0) → ( 0, 0,-1) ✓
// ---------------------------------------------------------------------------
fn emit_face(
    verts: &mut Vec<Vertex3D>,
    idxs:  &mut Vec<u32>,
    ox: f32, oy: f32, oz: f32,
    dir:   u8,
    color: [f32; 3],
    uv:    (f32, f32, f32, f32),   // (u0, v0, u1, v1)
) {
    let base = verts.len() as u32;

    let (normal, corners): ([f32; 3], [[f32; 3]; 4]) = match dir {
        // +X
        0 => ([1.0, 0.0, 0.0], [
            [ox+1.0, oy,     oz+1.0],
            [ox+1.0, oy,     oz    ],
            [ox+1.0, oy+1.0, oz    ],
            [ox+1.0, oy+1.0, oz+1.0],
        ]),
        // -X
        1 => ([-1.0, 0.0, 0.0], [
            [ox, oy,     oz    ],
            [ox, oy,     oz+1.0],
            [ox, oy+1.0, oz+1.0],
            [ox, oy+1.0, oz    ],
        ]),
        // +Y (top)
        2 => ([0.0, 1.0, 0.0], [
            [ox,     oy+1.0, oz+1.0],
            [ox+1.0, oy+1.0, oz+1.0],
            [ox+1.0, oy+1.0, oz    ],
            [ox,     oy+1.0, oz    ],
        ]),
        // -Y (bottom)
        3 => ([0.0, -1.0, 0.0], [
            [ox,     oy, oz    ],
            [ox+1.0, oy, oz    ],
            [ox+1.0, oy, oz+1.0],
            [ox,     oy, oz+1.0],
        ]),
        // +Z
        4 => ([0.0, 0.0, 1.0], [
            [ox,     oy,     oz+1.0],
            [ox+1.0, oy,     oz+1.0],
            [ox+1.0, oy+1.0, oz+1.0],
            [ox,     oy+1.0, oz+1.0],
        ]),
        // -Z
        _ => ([0.0, 0.0, -1.0], [
            [ox+1.0, oy,     oz],
            [ox,     oy,     oz],
            [ox,     oy+1.0, oz],
            [ox+1.0, oy+1.0, oz],
        ]),
    };

    // UV corners: (u0,v0)=top-left, (u1,v1)=bottom-right
    // Для каждой грани углы配对 с UV так, чтобы текстура не была зеркальной
    let (u0, v0, u1, v1) = uv;
    let uvs: [[f32; 2]; 4] = match dir {
        2 => [ // +Y: смотрим сверху
            [u0, v1], [u1, v1], [u1, v0], [u0, v0],
        ],
        3 => [ // -Y: смотрим снизу
            [u0, v0], [u1, v0], [u1, v1], [u0, v1],
        ],
        1 => [ // -X
            [u0, v1], [u1, v1], [u1, v0], [u0, v0],
        ],
        4 => [ // +Z
            [u1, v1], [u0, v1], [u0, v0], [u1, v0],
        ],
        _ => [ // +X (0), -Z (5), fallback
            [u0, v0], [u1, v0], [u1, v1], [u0, v1],
        ],
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
