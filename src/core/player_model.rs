// =============================================================================
// QubePixel — PlayerModel  (Minecraft Bedrock Edition geometry format)
// =============================================================================
//
// Parses the `minecraft:geometry` JSON format and builds per-bone geometry
// for skeletal animation.
//
// Coordinate system: Bedrock uses right-handed Y-up.
// 1 model unit = 1/16 block.  Scale factor: MODEL_SCALE = 1/16.
//
// UV mapping (box UV): for a cube at (u, v) with W×H×D pixel dimensions:
//   Top    face: (u+D,       v),     size W×D
//   Bottom face: (u+D+W,     v),     size W×D
//   Right  face: (u,         v+D),   size D×H   (east, +X)
//   Front  face: (u+D,       v+D),   size W×H   (south, +Z)
//   Left   face: (u+D+W,     v+D),   size D×H   (west, -X)
//   Back   face: (u+D+W+D,   v+D),   size W×H   (north, -Z)
// =============================================================================

use std::collections::HashMap;
use serde::Deserialize;
use glam::Vec3;

// ---------------------------------------------------------------------------
// JSON deserialization structures
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BedrockModelFile {
    pub format_version: String,
    #[serde(rename = "minecraft:geometry")]
    pub geometries: Vec<BedrockGeometry>,
}

#[derive(Debug, Deserialize)]
pub struct BedrockGeometry {
    pub description: BedrockDescription,
    pub bones: Vec<BedrockBone>,
}

#[derive(Debug, Deserialize)]
pub struct BedrockDescription {
    pub identifier:    String,
    #[serde(default = "default_64")]
    pub texture_width: u32,
    #[serde(default = "default_64")]
    pub texture_height: u32,
}

fn default_64() -> u32 { 64 }

#[derive(Debug, Deserialize)]
pub struct BedrockBone {
    pub name:   String,
    #[serde(default)]
    pub parent: Option<String>,
    /// Pivot point in model space (absolute, not parent-relative).
    #[serde(default)]
    pub pivot: Option<[f32; 3]>,
    /// Initial rotation in degrees [X, Y, Z] (Euler, ZYX order).
    #[serde(default)]
    pub rotation: Option<[f32; 3]>,
    #[serde(default)]
    pub cubes: Vec<BedrockCube>,
    #[serde(default)]
    pub mirror: bool,
}

#[derive(Debug, Deserialize)]
pub struct BedrockCube {
    /// Minimum corner of the cube in model space.
    pub origin: [f32; 3],
    /// [width, height, depth] in model units.
    pub size: [f32; 3],
    /// Box UV top-left corner in texture pixels.
    pub uv: [f32; 2],
    /// Inflation (expands all faces outward by this many units).
    #[serde(default)]
    pub inflate: f32,
}

// ---------------------------------------------------------------------------
// PlayerVertex — GPU vertex for the player model
// ---------------------------------------------------------------------------

/// Vertex for the player model.  32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlayerVertex {
    pub position: [f32; 3],  // world-relative model position
    pub uv:       [f32; 2],  // normalised UV [0..1]
    pub normal:   [f32; 3],  // face normal (outward)
}

// ---------------------------------------------------------------------------
// BoneMesh — geometry for a single bone
// ---------------------------------------------------------------------------

pub struct BoneMesh {
    pub vertices: Vec<PlayerVertex>,
    pub indices:  Vec<u32>,
    /// Bone pivot in scaled (block-unit) space.
    pub pivot:    Vec3,
}

// ---------------------------------------------------------------------------
// PlayerModel — loaded per-bone geometry
// ---------------------------------------------------------------------------

pub struct PlayerModel {
    /// Geometry identifier (e.g. "geometry.npc.steve").
    pub identifier:  String,
    pub tex_width:   u32,
    pub tex_height:  u32,
    /// Per-bone geometry (only bones that have cubes).
    pub bone_meshes: HashMap<String, BoneMesh>,
    /// Bone name → pivot in scaled (block-unit) space (all bones, including empty ones).
    pub bone_pivots: HashMap<String, Vec3>,
}

/// 1 model pixel = 1/16 block
pub const MODEL_SCALE: f32 = 1.0 / 16.0;

impl PlayerModel {
    // -----------------------------------------------------------------------
    // Loading
    // -----------------------------------------------------------------------

    /// Load the first geometry from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let file: BedrockModelFile = serde_json::from_str(json)
            .map_err(|e| format!("PlayerModel JSON parse error: {}", e))?;

        let geom = file.geometries.into_iter().next()
            .ok_or("PlayerModel: no geometry in file")?;

        Self::from_geometry(geom)
    }

    /// Load a specific geometry by identifier.
    pub fn from_json_named(json: &str, ident: &str) -> Result<Self, String> {
        let file: BedrockModelFile = serde_json::from_str(json)
            .map_err(|e| format!("PlayerModel JSON parse error: {}", e))?;

        let geom = file.geometries.into_iter()
            .find(|g| g.description.identifier == ident)
            .ok_or_else(|| format!("PlayerModel: geometry '{}' not found", ident))?;

        Self::from_geometry(geom)
    }

    fn from_geometry(geom: BedrockGeometry) -> Result<Self, String> {
        let tw = geom.description.texture_width  as f32;
        let th = geom.description.texture_height as f32;

        let mut bone_meshes: HashMap<String, BoneMesh> = HashMap::new();
        let mut bone_pivots: HashMap<String, Vec3>     = HashMap::new();

        for bone in &geom.bones {
            let pivot_raw = bone.pivot.unwrap_or([0.0, 0.0, 0.0]);
            let pivot_scaled = Vec3::new(
                pivot_raw[0] * MODEL_SCALE,
                pivot_raw[1] * MODEL_SCALE,
                -pivot_raw[2] * MODEL_SCALE, // Bedrock Z → OpenGL -Z
            );
            bone_pivots.insert(bone.name.clone(), pivot_scaled);

            if !bone.cubes.is_empty() {
                let mut vertices: Vec<PlayerVertex> = Vec::new();
                let mut indices:  Vec<u32>          = Vec::new();

                for cube in &bone.cubes {
                    push_cube(&mut vertices, &mut indices, cube, bone.mirror, tw, th);
                }

                bone_meshes.insert(bone.name.clone(), BoneMesh {
                    vertices,
                    indices,
                    pivot: pivot_scaled,
                });
            }
        }

        Ok(Self {
            identifier:  geom.description.identifier,
            tex_width:   geom.description.texture_width,
            tex_height:  geom.description.texture_height,
            bone_meshes,
            bone_pivots,
        })
    }
}

// ---------------------------------------------------------------------------
// Cube mesh generation
// ---------------------------------------------------------------------------

/// Push 6 faces (12 triangles, 24 vertices) for one cube.
/// All positions are converted to OpenGL space (Bedrock Z flipped to -Z).
fn push_cube(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    cube:    &BedrockCube,
    mirror:  bool,
    tw: f32,
    th: f32,
) {
    let infl = cube.inflate;
    let ox = (cube.origin[0] - infl) * MODEL_SCALE;
    let oy = (cube.origin[1] - infl) * MODEL_SCALE;
    // Bedrock Z → flip to OpenGL -Z, then account for size (depth extends in +Z bedrock = -Z ogl)
    let oz_min = -(cube.origin[2] + cube.size[2] + infl) * MODEL_SCALE;

    let w = (cube.size[0] + infl * 2.0) * MODEL_SCALE;
    let h = (cube.size[1] + infl * 2.0) * MODEL_SCALE;
    let d = (cube.size[2] + infl * 2.0) * MODEL_SCALE;

    let x0 = ox;
    let x1 = ox + w;
    let y0 = oy;
    let y1 = oy + h;
    let z0 = oz_min;
    let z1 = oz_min + d;

    // Box UV base (in pixels)
    let u = cube.uv[0];
    let v = cube.uv[1];
    let pw = cube.size[0]; // pixel width
    let ph = cube.size[1]; // pixel height
    let pd = cube.size[2]; // pixel depth

    // +Y  (Top face)    UV: (u+d, v) → (u+d+w, v+d)
    push_quad(verts, indices,
        [x0,y1,z0], [x1,y1,z0], [x1,y1,z1], [x0,y1,z1],
        [0.0, 1.0, 0.0],
        if mirror { uv_rev(uv_flip_u(u+pd, v, pw, pd, tw, th)) }
        else      { uv_rev(uv_rect(u+pd, v, pw, pd, tw, th)) },
    );

    // -Y  (Bottom face) UV: (u+d+w, v) → (u+d+2w, v+d)
    push_quad(verts, indices,
        [x0,y0,z1], [x1,y0,z1], [x1,y0,z0], [x0,y0,z0],
        [0.0, -1.0, 0.0],
        if mirror { uv_rev(uv_flip_u(u+pd+pw, v, pw, pd, tw, th)) }
        else      { uv_rev(uv_rect(u+pd+pw, v, pw, pd, tw, th)) },
    );

    // -Z  (North/Back face, +Z in Bedrock)  UV: (u+d+w+d, v+d) → (u+2d+2w, v+d+h)
    push_quad(verts, indices,
        [x1,y1,z0], [x0,y1,z0], [x0,y0,z0], [x1,y0,z0],
        [0.0, 0.0, -1.0],
        if mirror { uv_flip_u(u+pd+pw+pd, v+pd, pw, ph, tw, th) }
        else      { uv_rect(u+pd+pw+pd, v+pd, pw, ph, tw, th) },
    );

    // +Z  (South/Front face, -Z in Bedrock)  UV: (u+d, v+d) → (u+d+w, v+d+h)
    push_quad(verts, indices,
        [x0,y1,z1], [x1,y1,z1], [x1,y0,z1], [x0,y0,z1],
        [0.0, 0.0, 1.0],
        if mirror { uv_flip_u(u+pd, v+pd, pw, ph, tw, th) }
        else      { uv_rect(u+pd, v+pd, pw, ph, tw, th) },
    );

    // +X  (East/Right face)  UV: (u, v+d) → (u+d, v+d+h)
    push_quad(verts, indices,
        [x1,y0,z0], [x1,y0,z1], [x1,y1,z1], [x1,y1,z0],
        [1.0, 0.0, 0.0],
        uv_rev(uv_rect(u, v+pd, pd, ph, tw, th)),
    );

    // -X  (West/Left face)  UV: (u+d+w, v+d) → (u+2d+w, v+d+h)
    push_quad(verts, indices,
        [x0,y0,z1], [x0,y0,z0], [x0,y1,z0], [x0,y1,z1],
        [-1.0, 0.0, 0.0],
        uv_rev(uv_rect(u+pd+pw, v+pd, pd, ph, tw, th)),
    );
}

/// UV rect: (px, py) top-left corner, pw×ph size, normalised.
/// Returns [tl, tr, br, bl] UV pairs.
fn uv_rect(px: f32, py: f32, pw: f32, ph: f32, tw: f32, th: f32) -> [[f32; 2]; 4] {
    let u0 = px / tw;
    let u1 = (px + pw) / tw;
    let v0 = py / th;
    let v1 = (py + ph) / th;
    [[u0,v0],[u1,v0],[u1,v1],[u0,v1]]
}

/// Horizontally flipped UV rect (for mirror bones).
fn uv_flip_u(px: f32, py: f32, pw: f32, ph: f32, tw: f32, th: f32) -> [[f32; 2]; 4] {
    let u0 = px / tw;
    let u1 = (px + pw) / tw;
    let v0 = py / th;
    let v1 = (py + ph) / th;
    [[u1,v0],[u0,v0],[u0,v1],[u1,v1]]
}

/// Reverse UV order to match reversed vertex winding (compensates Z-flip for
/// faces whose plane is not perpendicular to Z).
fn uv_rev(uvs: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    [uvs[3], uvs[2], uvs[1], uvs[0]]
}

/// Push one quad (4 verts) as 2 triangles (CCW, indices 0-1-2, 0-2-3).
fn push_quad(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3],
    normal: [f32; 3],
    uvs: [[f32; 2]; 4],
) {
    let base = verts.len() as u32;
    verts.push(PlayerVertex { position: p0, uv: uvs[0], normal });
    verts.push(PlayerVertex { position: p1, uv: uvs[1], normal });
    verts.push(PlayerVertex { position: p2, uv: uvs[2], normal });
    verts.push(PlayerVertex { position: p3, uv: uvs[3], normal });
    indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
}
