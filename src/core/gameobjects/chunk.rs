// =============================================================================
// QubePixel — Chunk  (columnar voxel chunk with configurable XYZ dimensions)
//
// Dimensions are read once from config (assets/chunk_config.json).
// Default: 16 × 256 × 16 (columnar, one chunk per XZ column).
//
// Block storage: flat Vec<u8>, index = x*SY*SZ + y*SZ + z
// =============================================================================

use crate::{ flow_debug_log};
use crate::core::config;
use crate::core::gameobjects::block::BlockRegistry;
use crate::screens::game_3d_pipeline::{Vertex3D, snorm8, unorm8};
use crate::core::gameobjects::texture_atlas::TextureAtlasLayout;

// ---------------------------------------------------------------------------
// Dimension helpers — read from global config (initialised from JSON at startup)
// ---------------------------------------------------------------------------

#[inline(always)] fn csx() -> usize { config::chunk_size_x() }
#[inline(always)] fn csy() -> usize { config::chunk_size_y() }
#[inline(always)] fn csz() -> usize { config::chunk_size_z() }

/// Flat index into the block array: [x * SY * SZ + y * SZ + z]
#[inline(always)]
fn flat_idx(x: usize, y: usize, z: usize) -> usize {
    x * csy() * csz() + y * csz() + z
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

pub struct Chunk {
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    /// Flat block array: index = x*SY*SZ + y*SZ + z. 0 = Air.
    blocks: Vec<u8>,
    pub mesh_dirty: bool,
}

impl Chunk {
    pub fn new(cx: i32, cy: i32, cz: i32) -> Self {
        let vol = csx() * csy() * csz();
        Self {
            cx,
            cy,
            cz,
            blocks: vec![0u8; vol],
            mesh_dirty: true,
        }
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> u8 {
        self.blocks[flat_idx(x, y, z)]
    }

    #[allow(dead_code)]
    pub fn set(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[flat_idx(x, y, z)] = block_id;
        self.mesh_dirty = true;
    }

    /// Write a block without marking the chunk dirty (used during terrain generation).
    pub(crate) fn set_gen(&mut self, x: usize, y: usize, z: usize, block_id: u8) {
        self.blocks[flat_idx(x, y, z)] = block_id;
    }

    pub fn is_all_air(&self) -> bool {
        self.blocks.iter().all(|&b| b == 0)
    }

    // -----------------------------------------------------------------------
    // Full-detail mesh (LOD 0)
    // -----------------------------------------------------------------------

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
    /// * `lod_level = 0` → step = 1 (full detail)
    /// * `lod_level = 1` → step = 2 (2× coarser)
    /// * `lod_level = 2` → step = 4 (4× coarser)
    ///
    /// Each "super-block" uses the dominant non-air block in its step³ region.
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

        let sx = csx();
        let sy = csy();
        let sz = csz();

        let step = 1usize << (lod_level.min(2) as usize); // 1, 2, or 4
        let cells_x = sx / step;
        let cells_y = sy / step;
        let cells_z = sz / step;
        let scale   = step as f32;

        let wx_base = self.cx * sx as i32;
        let wy_base = self.cy * sy as i32;
        let wz_base = self.cz * sz as i32;

        let cap = if lod_level == 0 { 4096 } else { 512 };
        let mut vertices: Vec<Vertex3D> = Vec::with_capacity(cap);
        let mut indices:  Vec<u32>      = Vec::with_capacity(cap * 3 / 2);

        for cx_cell in 0..cells_x {
            for cy_cell in 0..cells_y {
                for cz_cell in 0..cells_z {
                    let bx0 = cx_cell * step;
                    let by0 = cy_cell * step;
                    let bz0 = cz_cell * step;

                    let block_id = self.dominant_block(bx0, by0, bz0, step, sx, sy, sz);
                    if block_id == 0 { continue; }

                    let def = match registry.get(block_id) {
                        Some(d) => d,
                        None => continue,
                    };
                    if !def.solid { continue; }

                    let ox = wx_base as f32 + bx0 as f32;
                    let oy = wy_base as f32 + by0 as f32;
                    let oz = wz_base as f32 + bz0 as f32;

                    let wx = wx_base + bx0 as i32;
                    let wy = wy_base + by0 as i32;
                    let wz = wz_base + bz0 as i32;

                    let mat = &def.material;
                    let material_data = [mat.roughness, mat.metalness];
                    let emi = &def.emission;
                    let emission_data = if emi.emit_light {
                        let intensity = if emi.no_self_gi { -emi.light_intensity } else { emi.light_intensity };
                        [emi.light_color[0], emi.light_color[1], emi.light_color[2], intensity]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    };

                    let ws = step as i32;

                    // +X face
                    if self.is_adjacent_air(bx0 + step, by0, bz0, step, 0, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(0);
                        let uv = atlas.uv_for(def.texture_for_face(0).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 0, c, uv, scale, material_data, emission_data);
                    }
                    // -X face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 1, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(1);
                        let uv = atlas.uv_for(def.texture_for_face(1).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 1, c, uv, scale, material_data, emission_data);
                    }
                    // +Y face
                    if self.is_adjacent_air(bx0, by0 + step, bz0, step, 2, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(2);
                        let uv = atlas.uv_for(def.texture_for_face(2).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 2, c, uv, scale, material_data, emission_data);
                    }
                    // -Y face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 3, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(3);
                        let uv = atlas.uv_for(def.texture_for_face(3).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 3, c, uv, scale, material_data, emission_data);
                    }
                    // +Z face
                    if self.is_adjacent_air(bx0, by0, bz0 + step, step, 4, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(4);
                        let uv = atlas.uv_for(def.texture_for_face(4).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 4, c, uv, scale, material_data, emission_data);
                    }
                    // -Z face
                    if self.is_adjacent_air_neg(bx0, by0, bz0, step, 5, wx, wy, wz, sx, sy, sz, &get_neighbor) {
                        let c  = def.color_for_face(5);
                        let uv = atlas.uv_for(def.texture_for_face(5).unwrap_or(""));
                        emit_face_lod(&mut vertices, &mut indices, ox, oy, oz, 5, c, uv, scale, material_data, emission_data);
                    }

                    let _ = ws; // suppress unused warning
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

    fn dominant_block(
        &self,
        bx0: usize, by0: usize, bz0: usize,
        step: usize,
        sx: usize, sy: usize, sz: usize,
    ) -> u8 {
        if step == 1 {
            return self.blocks[flat_idx(bx0, by0, bz0)];
        }

        let mid = step / 2;
        let cx = (bx0 + mid).min(sx - 1);
        let cy = (by0 + mid).min(sy - 1);
        let cz = (bz0 + mid).min(sz - 1);
        let center = self.blocks[flat_idx(cx, cy, cz)];
        if center != 0 { return center; }

        for dx in 0..step {
            for dy in 0..step {
                for dz in 0..step {
                    let x = bx0 + dx;
                    let y = by0 + dy;
                    let z = bz0 + dz;
                    if x < sx && y < sy && z < sz {
                        let id = self.blocks[flat_idx(x, y, z)];
                        if id != 0 { return id; }
                    }
                }
            }
        }
        0
    }

    /// Check if the positive-direction adjacent super-block is air (face culling).
    #[allow(clippy::too_many_arguments)]
    fn is_adjacent_air<F>(
        &self,
        bx: usize, by: usize, bz: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        sx: usize, sy: usize, sz: usize,
        get_neighbor: &F,
    ) -> bool
    where F: Fn(i32, i32, i32) -> u8
    {
        let (check_local, world_offset) = match dir {
            0 => (bx < sx, (step as i32, 0, 0)),
            2 => (by < sy, (0, step as i32, 0)),
            4 => (bz < sz, (0, 0, step as i32)),
            _ => return true,
        };

        if check_local {
            let (lx, ly, lz) = match dir {
                0 => (bx,              by + step / 2, bz + step / 2),
                2 => (bx + step / 2,  by,            bz + step / 2),
                4 => (bx + step / 2,  by + step / 2, bz           ),
                _ => (bx, by, bz),
            };
            let lx = lx.min(sx - 1);
            let ly = ly.min(sy - 1);
            let lz = lz.min(sz - 1);
            self.blocks[flat_idx(lx, ly, lz)] == 0
        } else {
            get_neighbor(
                wx + world_offset.0,
                wy + world_offset.1,
                wz + world_offset.2,
            ) == 0
        }
    }

    /// Check negative-direction adjacency.
    #[allow(clippy::too_many_arguments)]
    fn is_adjacent_air_neg<F>(
        &self,
        bx0: usize, by0: usize, bz0: usize,
        step: usize, dir: u8,
        wx: i32, wy: i32, wz: i32,
        sx: usize, sy: usize, sz: usize,
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
            let (lx, ly, lz) = match dir {
                1 => (bx0 - 1,          by0 + step / 2, bz0 + step / 2),
                3 => (bx0 + step / 2,   by0 - 1,        bz0 + step / 2),
                5 => (bx0 + step / 2,   by0 + step / 2, bz0 - 1       ),
                _ => (bx0, by0, bz0),
            };
            let lx = lx.min(sx - 1);
            let ly = ly.min(sy - 1);
            let lz = lz.min(sz - 1);
            self.blocks[flat_idx(lx, ly, lz)] == 0
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
    dir:      u8,
    color:    [f32; 3],
    uv:       (f32, f32, f32, f32),
    scale:    f32,
    material: [f32; 2],   // [roughness, metalness]
    emission: [f32; 4],   // [r, g, b, intensity]
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

    let tangent: [f32; 3] = match dir {
        0 => [0.0,  0.0, -1.0],
        1 => [0.0,  0.0,  1.0],
        2 => [1.0,  0.0,  0.0],
        3 => [1.0,  0.0,  0.0],
        4 => [1.0,  0.0,  0.0],
        _ => [-1.0, 0.0,  0.0],
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
            position:  corners[i],
            normal:    [snorm8(normal[0]), snorm8(normal[1]), snorm8(normal[2]), 0],
            color_ao:  [unorm8(color[0]),  unorm8(color[1]),  unorm8(color[2]),  255],
            texcoord:  uvs[i],
            material:  [unorm8(material[0]), unorm8(material[1]), 0, 0],
            emission,
            tangent:   [snorm8(tangent[0]), snorm8(tangent[1]), snorm8(tangent[2]), 0],
        });
    }
    idxs.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
