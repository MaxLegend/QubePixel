// =============================================================================
// QubePixel — BlockModel  (unified model parser for block custom models)
// =============================================================================
//
// Parses both Minecraft Bedrock Edition geometry format and Java Edition
// block model format into a unified BlockModelData structure.
//
// Supported formats:
//   - Bedrock geometry (minecraft:geometry JSON)
//     Auto-detected by presence of "minecraft:geometry" key.
//   - Java Edition block model (elements/faces JSON)
//     Auto-detected by presence of "elements" key.
//
// Multi-texture support:
//   Java Edition models can reference multiple textures via the "textures"
//   map and per-face "texture" fields (e.g. "#0", "#side").  These are
//   resolved into a per-face texture variable map.  At load time, the
//   ModelMessenger packs all referenced textures into a single per-model
//   atlas and remaps UVs accordingly.
//
// Coordinate conversion:
//   Bedrock: Y-up, right-handed. Z-flipped to OpenGL. Scale: 1 unit = 1/16 block.
//   Java:    Y-up, right-handed. Origin at block corner (0,0,0).
//            Converted to center-origin: (pixel - 8.0) / 16.0
// =============================================================================

use std::collections::HashMap;
use serde::Deserialize;
use glam::{Mat3, Mat4, Vec3};

use crate::core::player_model::{BoneMesh, PlayerModel, PlayerVertex, MODEL_SCALE};

// ---------------------------------------------------------------------------
// Unified model output
// ---------------------------------------------------------------------------

/// A parsed block model ready for GPU upload or conversion to PlayerModel.
#[derive(Debug, Clone)]
pub struct BlockModelData {
    /// Model identifier (from geometry description or generated).
    pub identifier:  String,
    /// All vertices in model-space block units (OpenGL coordinates).
    pub vertices:    Vec<PlayerVertex>,
    /// Triangle indices (uint32).
    pub indices:     Vec<u32>,
    /// Texture width in pixels (for UV normalization).
    pub tex_width:   u32,
    /// Texture height in pixels (for UV normalization).
    pub tex_height:  u32,
    /// Axis-aligned bounding box minimum corner.
    pub aabb_min:    Vec3,
    /// Axis-aligned bounding box maximum corner.
    pub aabb_max:    Vec3,

    /// Per-cube shadow AABBs in block-canonical [0,1]^3 space for voxel shadow DDA.
    /// Each entry: [x_min, x_max, y_min, y_max, z_min, z_max], all in [0,1].
    /// One entry per element (Java) or cube (Bedrock), after all transforms applied.
    /// Both formats use centred XZ → shift X and Z by +0.5. Y is already [0,1].
    pub shadow_cubes: Vec<[f32; 6]>,

    // -- Multi-texture support -----------------------------------------------
    /// Texture variable definitions from the model JSON.
    /// Java format: maps "0" → "block/stone_side", "1" → "block/stone_top", etc.
    /// Bedrock format: empty (single texture assumed).
    pub texture_variables: HashMap<String, String>,

    /// Per-face texture assignment: (first_vertex_index, vertex_count, texture_var).
    /// Each entry records which texture variable a group of vertices belongs to.
    /// Used for UV remapping when packing textures into a per-model atlas.
    pub face_texture_map: Vec<FaceTextureEntry>,
}

/// Records which texture variable a face's vertices use.
#[derive(Debug, Clone)]
pub struct FaceTextureEntry {
    /// Index of the first vertex in this face.
    pub first_vertex: u32,
    /// Number of vertices in this face (always 4 for a quad).
    pub vertex_count: u32,
    /// Texture variable name (e.g. "#0", "#side") — may be unresolved.
    pub texture_var:  String,
}

impl BlockModelData {
    /// Convert to a PlayerModel with a single "root" bone.
    /// This allows using block models with the existing EntityRenderer.
    pub fn to_player_model(&self) -> PlayerModel {
        let mut bone_meshes = HashMap::new();
        bone_meshes.insert("root".to_string(), BoneMesh {
            vertices: self.vertices.clone(),
            indices:  self.indices.clone(),
            pivot:    Vec3::ZERO,
        });
        let mut bone_pivots = HashMap::new();
        bone_pivots.insert("root".to_string(), Vec3::ZERO);

        PlayerModel {
            identifier:  self.identifier.clone(),
            tex_width:   self.tex_width,
            tex_height:  self.tex_height,
            bone_meshes,
            bone_pivots,
        }
    }

    /// Returns true if the model has no geometry.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }

    /// Returns true if this model uses multiple textures.
    pub fn has_multi_texture(&self) -> bool {
        !self.face_texture_map.is_empty()
    }

    /// Returns the height profile (min_y, max_y) of the model in block space [0..1].
    pub fn y_range(&self) -> [f32; 2] {
        let y_min = self.shadow_cubes.iter().map(|c| c[2]).fold(1.0f32, f32::min);
        let y_max = self.shadow_cubes.iter().map(|c| c[3]).fold(0.0f32, f32::max);
        [y_min, y_max]
    }

    /// Collect all unique texture variable names referenced by faces.
    /// Returns variable names without the leading '#'.
    pub fn referenced_textures(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for entry in &self.face_texture_map {
            let name = entry.texture_var.trim_start_matches('#').to_string();
            if !name.is_empty() && seen.insert(name.clone()) {
                result.push(name);
            }
        }
        result
    }

    /// Resolve texture variable names to actual file paths using the model's
    /// texture_variables map and the block's model_textures overrides.
    ///
    /// Resolution chain:
    ///   1. Face references "#0"
    ///   2. Look up "0" in self.texture_variables → gets e.g. "block/stone_side"
    ///   3. Look up "block/stone_side" in block_textures → gets file path
    ///   4. If not found, use the raw value as a file path
    ///
    /// Returns a map: texture_var_name (without #) → resolved file path.
    pub fn resolve_texture_paths(
        &self,
        block_textures: &HashMap<String, String>,
    ) -> HashMap<String, String> {
        let mut resolved = HashMap::new();
        for var_name in self.referenced_textures() {
            // Step 1: resolve from model's texture_variables
            let model_path = self.texture_variables.get(&var_name)
                .map(|s| s.trim_start_matches('#').to_string())
                .unwrap_or_else(|| var_name.clone());

            // Step 2: resolve from block's model_textures
            let final_path = block_textures.get(&model_path)
                .or_else(|| block_textures.get(&var_name))
                .cloned()
                .unwrap_or_else(|| model_path.clone());

            resolved.insert(var_name, final_path);
        }
        resolved
    }

    /// Remap UV coordinates for a packed texture atlas.
    ///
    /// `atlas_layout` maps texture variable names (without #) to their
    /// position in the atlas: (x_pixels, y_pixels, width_pixels, height_pixels).
    /// `atlas_w` and `atlas_h` are the full atlas dimensions in pixels.
    ///
    /// Before remapping, UVs are in [0..1] space relative to the individual
    /// texture. After remapping, UVs point into the correct region of the atlas.
    pub fn remap_uvs_for_atlas(
        &mut self,
        atlas_layout: &HashMap<String, (u32, u32, u32, u32)>,
        atlas_w: u32,
        atlas_h: u32,
    ) {
        if atlas_layout.is_empty() { return; }
        let aw = atlas_w as f32;
        let ah = atlas_h as f32;

        let vert_len = self.vertices.len();
        for entry in &self.face_texture_map {
            let tex_name = entry.texture_var.trim_start_matches('#');
            if let Some(&(x, y, w, h)) = atlas_layout.get(tex_name) {
                let xf = x as f32;
                let yf = y as f32;
                let wf = w as f32;
                let hf = h as f32;

                let start = entry.first_vertex as usize;
                let end   = start + entry.vertex_count as usize;
                for v in &mut self.vertices[start..end.min(vert_len)] {
                    // Current UV is in [0,1] space of the individual texture
                    let u = v.uv[0];
                    let t = v.uv[1];
                    // Remap to atlas space
                    v.uv[0] = (xf + u * wf) / aw;
                    v.uv[1] = (yf + t * hf) / ah;
                }
            }
        }

        // Update texture dimensions to atlas dimensions
        self.tex_width  = atlas_w;
        self.tex_height = atlas_h;
    }
}

// ---------------------------------------------------------------------------
// Format detection
// ---------------------------------------------------------------------------

/// Detected model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Bedrock Edition geometry (minecraft:geometry).
    BedrockGeometry,
    /// Java Edition block model (elements/faces).
    JavaEdition,
}

/// Auto-detect model format from JSON content.
pub fn detect_format(json: &str) -> Result<ModelFormat, String> {
    let val: serde_json::Value = serde_json::from_str(json)
        .map_err(|e| format!("Invalid JSON: {}", e))?;

    if val.get("minecraft:geometry").is_some() {
        return Ok(ModelFormat::BedrockGeometry);
    }
    if val.get("elements").is_some() {
        return Ok(ModelFormat::JavaEdition);
    }

    Err("Cannot detect model format: expected 'minecraft:geometry' or 'elements' key".into())
}

// ---------------------------------------------------------------------------
// Bedrock geometry — JSON deserialization
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BedrockFile {
    #[serde(rename = "minecraft:geometry")]
    geometries: Vec<BedrockGeometry>,
}

#[derive(Debug, Deserialize)]
struct BedrockGeometry {
    description: BedrockDescription,
    #[serde(default)]
    bones: Vec<BedrockBone>,
}

#[derive(Debug, Deserialize)]
struct BedrockDescription {
    identifier:     String,
    #[serde(default = "default_64")]
    texture_width:  u32,
    #[serde(default = "default_64")]
    texture_height: u32,
}

fn default_64() -> u32 { 64 }

#[derive(Debug, Deserialize)]
struct BedrockBone {
    name:   String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    pivot:  Option<[f32; 3]>,
    #[serde(default)]
    rotation: Option<[f32; 3]>,
    #[serde(default)]
    cubes:  Vec<BedrockCube>,
    #[serde(default)]
    mirror: bool,
}

#[derive(Debug, Deserialize)]
struct BedrockCube {
    origin: [f32; 3],
    size:   [f32; 3],
    #[serde(default)]
    uv: Option<BedrockCubeUv>,
    #[serde(default)]
    inflate: f32,
    #[serde(default)]
    mirror: Option<bool>,
    /// Per-cube pivot point in model space (Bedrock units). Used with `rotation`.
    #[serde(default)]
    pivot: Option<[f32; 3]>,
    /// Per-cube Euler rotation [x, y, z] in degrees, ZYX order.
    /// Blockbench exports this when individual cubes are rotated within a bone.
    #[serde(default)]
    rotation: Option<[f32; 3]>,
}

/// Bedrock cube UV: either box UV `[u, v]` or per-face UV object.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BedrockCubeUv {
    Box([f32; 2]),
    PerFace(BedrockPerFaceUv),
}

#[derive(Debug, Deserialize, Default)]
struct BedrockPerFaceUv {
    north: Option<BedrockFaceUv>,
    south: Option<BedrockFaceUv>,
    east:  Option<BedrockFaceUv>,
    west:  Option<BedrockFaceUv>,
    up:    Option<BedrockFaceUv>,
    down:  Option<BedrockFaceUv>,
}

#[derive(Debug, Deserialize)]
struct BedrockFaceUv {
    uv:      [f32; 2],
    uv_size: [f32; 2],
}

// ---------------------------------------------------------------------------
// Java Edition — JSON deserialization
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct JavaFile {
    /// Texture variable definitions: maps "#0" → "block/stone_side", etc.
    #[serde(default)]
    textures: HashMap<String, String>,
    #[serde(default)]
    elements: Vec<JavaElement>,
}

#[derive(Debug, Deserialize)]
struct JavaElement {
    from:     [f32; 3],
    to:       [f32; 3],
    rotation: Option<JavaRotation>,
    faces:    JavaFaces,
}

#[derive(Debug, Deserialize)]
struct JavaRotation {
    angle:  f32,
    axis:   String,
    origin: [f32; 3],
}

#[derive(Debug, Deserialize, Default)]
struct JavaFaces {
    north: Option<JavaFace>,
    south: Option<JavaFace>,
    east:  Option<JavaFace>,
    west:  Option<JavaFace>,
    up:    Option<JavaFace>,
    down:  Option<JavaFace>,
}

#[derive(Debug, Deserialize)]
struct JavaFace {
    uv:       Option<[f32; 4]>,
    #[serde(default)]
    texture:  Option<String>,
    #[serde(default)]
    rotation: u32,
}

// ---------------------------------------------------------------------------
// BlockModelParser — public API
// ---------------------------------------------------------------------------

/// Parser for block models. Auto-detects format and produces BlockModelData.
pub struct BlockModelParser;

impl BlockModelParser {
    /// Parse a model from JSON, auto-detecting the format.
    pub fn parse(json: &str) -> Result<BlockModelData, String> {
        match detect_format(json)? {
            ModelFormat::BedrockGeometry => Self::parse_bedrock(json),
            ModelFormat::JavaEdition     => Self::parse_java(json),
        }
    }

    /// Parse a Bedrock geometry format model (first geometry in file).
    pub fn parse_bedrock(json: &str) -> Result<BlockModelData, String> {
        Self::parse_bedrock_named(json, None)
    }

    /// Parse a specific named geometry from a Bedrock model file.
    pub fn parse_bedrock_named(
        json:   &str,
        target: Option<&str>,
    ) -> Result<BlockModelData, String> {
        let file: BedrockFile = serde_json::from_str(json)
            .map_err(|e| format!("Bedrock model parse error: {}", e))?;

        let geom = if let Some(name) = target {
            file.geometries.into_iter()
                .find(|g| g.description.identifier == name)
                .ok_or_else(|| format!("Geometry '{}' not found", name))?
        } else {
            file.geometries.into_iter().next()
                .ok_or("No geometry in Bedrock model file")?
        };

        let tw = geom.description.texture_width  as f32;
        let th = geom.description.texture_height as f32;

        let mut vertices     = Vec::new();
        let mut indices      = Vec::new();
        let mut shadow_cubes: Vec<[f32; 6]> = Vec::new();

        for bone in &geom.bones {
            let bone_mirror = bone.mirror;
            let pivot = bone.pivot.unwrap_or([0.0, 0.0, 0.0]);
            let rot   = bone.rotation.unwrap_or([0.0, 0.0, 0.0]);

            // Bone transform: translate to pivot, rotate, translate back.
            // pivot[2] is negated because vertex Z is already flipped to OpenGL
            // space in push_bedrock_*_cube (oz_min = -(origin[2]+size[2])*scale).
            // The pivot must be in the same space as the vertices.
            let bone_mat = if rot != [0.0, 0.0, 0.0] {
                let p = Vec3::new(pivot[0], pivot[1], -pivot[2]) * MODEL_SCALE;
                // X-flip (Bedrock → OpenGL YZ-mirror) requires negating Y and Z rotation
                // angles: S·Ry(θ)·S⁻¹ = Ry(-θ), S·Rz(θ)·S⁻¹ = Rz(-θ).
                let rz = Mat4::from_rotation_z(-rot[2].to_radians());
                let rx = Mat4::from_rotation_x(rot[0].to_radians());
                let ry = Mat4::from_rotation_y(-rot[1].to_radians());
                Mat4::from_translation(p) * ry * rx * rz * Mat4::from_translation(-p)
            } else {
                Mat4::IDENTITY
            };

            for cube in &bone.cubes {
                let cube_mirror = cube.mirror.unwrap_or(bone_mirror);
                let base = vertices.len() as u32;

                match &cube.uv {
                    Some(BedrockCubeUv::Box(uv)) => {
                        push_bedrock_box_cube(
                            &mut vertices, &mut indices,
                            cube, *uv, cube_mirror, tw, th,
                        );
                    }
                    Some(BedrockCubeUv::PerFace(pfuv)) => {
                        push_bedrock_perface_cube(
                            &mut vertices, &mut indices,
                            cube, pfuv, tw, th,
                        );
                    }
                    None => {
                        // No UV — generate with default UVs (full texture)
                        push_bedrock_box_cube(
                            &mut vertices, &mut indices,
                            cube, [0.0, 0.0], cube_mirror, tw, th,
                        );
                    }
                }

                // Step 1: Apply cube-level rotation (Blockbench per-cube transform).
                // Must be applied BEFORE the bone transform since cube rotation is
                // defined in the bone's local frame.
                // cp[2] is negated to match vertex Z space (same reason as bone pivot).
                let cube_rot = cube.rotation.unwrap_or([0.0, 0.0, 0.0]);
                if cube_rot != [0.0, 0.0, 0.0] {
                    let cp = cube.pivot.unwrap_or([0.0, 0.0, 0.0]);
                    let pivot = Vec3::new(cp[0], cp[1], -cp[2]) * MODEL_SCALE;
                    // X-flip (Bedrock → OpenGL YZ-mirror) requires negating Y and Z rotation
                    // angles: S·Ry(θ)·S⁻¹ = Ry(-θ), S·Rz(θ)·S⁻¹ = Rz(-θ).
                    let rz = Mat4::from_rotation_z(-cube_rot[2].to_radians());
                    let rx = Mat4::from_rotation_x(cube_rot[0].to_radians());
                    let ry = Mat4::from_rotation_y(-cube_rot[1].to_radians());
                    let cube_mat = Mat4::from_translation(pivot) * ry * rx * rz
                        * Mat4::from_translation(-pivot);
                    let rot3 = Mat3::from_mat4(cube_mat);
                    for v in &mut vertices[base as usize..] {
                        let p = cube_mat.transform_point3(Vec3::from(v.position));
                        let n = (rot3 * Vec3::from(v.normal)).normalize();
                        v.position = p.to_array();
                        v.normal   = n.to_array();
                    }
                }

                // Step 2: Apply bone transform to newly added vertices
                if bone_mat != Mat4::IDENTITY {
                    let rot3 = Mat3::from_mat4(bone_mat);
                    for v in &mut vertices[base as usize..] {
                        let p = bone_mat.transform_point3(Vec3::from(v.position));
                        let n = (rot3 * Vec3::from(v.normal)).normalize();
                        v.position = p.to_array();
                        v.normal   = n.to_array();
                    }
                }

                // Per-cube shadow AABB in block-canonical [0,1]^3 space.
                let (cube_min, cube_max) = compute_aabb(&vertices[base as usize..]);
                shadow_cubes.push([
                    (cube_min.x + 0.5).clamp(0.0, 1.0),
                    (cube_max.x + 0.5).clamp(0.0, 1.0),
                    cube_min.y.clamp(0.0, 1.0),
                    cube_max.y.clamp(0.0, 1.0),
                    (cube_min.z + 0.5).clamp(0.0, 1.0),
                    (cube_max.z + 0.5).clamp(0.0, 1.0),
                ]);
            }
        }

        let (aabb_min, aabb_max) = compute_aabb(&vertices);

        // For single-texture Bedrock models, add a face_texture_map entry spanning
        // all vertices so build_model_atlas can load and apply the texture.
        // The texture variable "default" is resolved by ModelMessenger from the
        // block's model_textures map (typically {"default": "texture.png"}).
        let face_texture_map = if !vertices.is_empty() {
            vec![FaceTextureEntry {
                first_vertex: 0,
                vertex_count: vertices.len() as u32,
                texture_var:  "default".to_string(),
            }]
        } else {
            vec![]
        };

        Ok(BlockModelData {
            identifier:  geom.description.identifier,
            vertices,
            indices,
            tex_width:   geom.description.texture_width,
            tex_height:  geom.description.texture_height,
            aabb_min,
            aabb_max,
            shadow_cubes,
            texture_variables: HashMap::new(),
            face_texture_map,
        })
    }

    /// Parse a Java Edition block model.
    pub fn parse_java(json: &str) -> Result<BlockModelData, String> {
        let file: JavaFile = serde_json::from_str(json)
            .map_err(|e| format!("Java model parse error: {}", e))?;

        // Java Edition default texture size is 16×16
        let tw = 16.0f32;
        let th = 16.0f32;

        let mut vertices         = Vec::new();
        let mut indices          = Vec::new();
        let mut face_texture_map = Vec::new();
        let mut shadow_cubes: Vec<[f32; 6]> = Vec::new();

        for element in &file.elements {
            let base = vertices.len();

            push_java_element(
                &mut vertices, &mut indices,
                &mut face_texture_map,
                element, tw, th,
            );

            // Apply element rotation in pixel space
            if let Some(ref rotation) = element.rotation {
                if rotation.angle != 0.0 {
                    let rot_mat = java_rotation_matrix(rotation);
                    let rot3 = Mat3::from_mat4(rot_mat);
                    for v in &mut vertices[base..] {
                        let p = rot_mat.transform_point3(Vec3::from(v.position));
                        let n = rot3 * Vec3::from(v.normal);
                        v.position = p.to_array();
                        v.normal   = n.normalize().to_array();
                    }
                }
            }

            // Convert from pixel coordinates to block units (centered at origin)
            for v in &mut vertices[base..] {
                v.position[0] = (v.position[0] - 8.0) * MODEL_SCALE;
                v.position[1] = (v.position[1] - 8.0) * MODEL_SCALE;
                v.position[2] = (v.position[2] - 8.0) * MODEL_SCALE;
            }

            // Shift Y so block floor sits at Y=0 (range −0.5…+0.5 → 0…1)
            for v in &mut vertices[base..] {
                v.position[1] += 0.5;
            }

            // Per-element shadow AABB in block-canonical [0,1]^3 space.
            // Java model Z convention (+Z = South) is opposite to the game's voxel
            // Z convention (+Z = North, matching OpenGL/Bedrock).  Flip Z so that
            // shadow_z=0 maps to the South face (frac_z=0) and shadow_z=1 maps to
            // the North face (frac_z=1).  min/max are swapped after the flip.
            let (elem_min, elem_max) = compute_aabb(&vertices[base..]);
            shadow_cubes.push([
                (elem_min.x + 0.5).clamp(0.0, 1.0),
                (elem_max.x + 0.5).clamp(0.0, 1.0),
                elem_min.y.clamp(0.0, 1.0),
                elem_max.y.clamp(0.0, 1.0),
                (0.5 - elem_max.z).clamp(0.0, 1.0),
                (0.5 - elem_min.z).clamp(0.0, 1.0),
            ]);
        }

        let (aabb_min, aabb_max) = compute_aabb(&vertices);

        Ok(BlockModelData {
            identifier: "java:block_model".to_string(),
            vertices,
            indices,
            tex_width:   tw as u32,
            tex_height:  th as u32,
            aabb_min,
            aabb_max,
            shadow_cubes,
            texture_variables: file.textures,
            face_texture_map,
        })
    }
}

// ---------------------------------------------------------------------------
// Outline edge extraction
// ---------------------------------------------------------------------------

/// Extract quad perimeter edges from a block model mesh for wireframe outline rendering.
///
/// Iterates the index buffer in groups of 6 (two triangles per quad, using the
/// pattern `[v0, v1, v2,  v0, v2, v3]` produced by `push_quad`).  Returns the
/// 4 perimeter edges of each quad (v0–v1, v1–v2, v2–v3, v3–v0), skipping the
/// interior diagonal (v0–v2).  Edges shared between adjacent faces are deduplicated.
///
/// Returns a flat list of line-segment endpoint pairs in model-space coordinates
/// (X and Z centered at 0, Y ∈ [0..1]).  The caller must apply the same +0.5
/// X/Z offset that `BlockModelRenderer` uses before uploading to the GPU.
pub fn extract_quad_edges(vertices: &[PlayerVertex], indices: &[u32]) -> Vec<[f32; 3]> {
    use std::collections::HashSet;

    fn encode(p: [f32; 3]) -> (u32, u32, u32) {
        (p[0].to_bits(), p[1].to_bits(), p[2].to_bits())
    }
    fn edge_key(a: [f32; 3], b: [f32; 3]) -> ((u32, u32, u32), (u32, u32, u32)) {
        let (ea, eb) = (encode(a), encode(b));
        if ea <= eb { (ea, eb) } else { (eb, ea) }
    }

    let mut seen: HashSet<((u32, u32, u32), (u32, u32, u32))> = HashSet::new();
    let mut out: Vec<[f32; 3]> = Vec::new();

    for chunk in indices.chunks(6) {
        if chunk.len() < 6 { continue; }
        // push_quad emits: [base, base+1, base+2,  base, base+2, base+3]
        let v0 = vertices[chunk[0] as usize].position;
        let v1 = vertices[chunk[1] as usize].position;
        let v2 = vertices[chunk[2] as usize].position;
        let v3 = vertices[chunk[5] as usize].position; // chunk[3]==v0, chunk[4]==v2

        for (a, b) in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)] {
            if seen.insert(edge_key(a, b)) {
                out.push(a);
                out.push(b);
            }
        }
    }

    out
}

/// Extract all triangles from a mesh's index buffer as flat `[f32; 9]` triples
/// (v0.xyz, v1.xyz, v2.xyz) in model-space coordinates (X/Z centered at 0).
///
/// Indices are expected in triangle-list order (groups of 3); `push_quad`
/// emits `[v0,v1,v2, v0,v2,v3]`, so every 3 indices form one triangle.
/// The caller must apply the +0.5 X/Z offset before use in world-space tests.
pub fn extract_triangles(vertices: &[PlayerVertex], indices: &[u32]) -> Vec<[f32; 9]> {
    let mut out = Vec::with_capacity(indices.len() / 3);
    for tri in indices.chunks(3) {
        if tri.len() < 3 { continue; }
        let v0 = vertices[tri[0] as usize].position;
        let v1 = vertices[tri[1] as usize].position;
        let v2 = vertices[tri[2] as usize].position;
        out.push([v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]]);
    }
    out
}

// ---------------------------------------------------------------------------
// AABB computation
// ---------------------------------------------------------------------------

fn compute_aabb(vertices: &[PlayerVertex]) -> (Vec3, Vec3) {
    if vertices.is_empty() {
        return (Vec3::ZERO, Vec3::ZERO);
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in vertices {
        let p = Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

// ---------------------------------------------------------------------------
// Java Edition rotation matrix
// ---------------------------------------------------------------------------

fn java_rotation_matrix(rot: &JavaRotation) -> Mat4 {
    let origin = Vec3::new(rot.origin[0], rot.origin[1], rot.origin[2]);
    let angle  = rot.angle.to_radians();

    let r = match rot.axis.as_str() {
        "x" => Mat4::from_rotation_x(angle),
        "y" => Mat4::from_rotation_y(angle),
        "z" => Mat4::from_rotation_z(angle),
        _   => Mat4::IDENTITY,
    };

    Mat4::from_translation(origin) * r * Mat4::from_translation(-origin)
}

// ---------------------------------------------------------------------------
// Bedrock box-UV cube generation
// ---------------------------------------------------------------------------

/// Push 6 faces for a Bedrock cube using box UV mapping.
fn push_bedrock_box_cube(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    cube:    &BedrockCube,
    uv:      [f32; 2],
    mirror:  bool,
    tw: f32, th: f32,
) {
    let infl = cube.inflate;
    let ox = (cube.origin[0] - infl) * MODEL_SCALE;
    let oy = (cube.origin[1] - infl) * MODEL_SCALE;
    // Bedrock Z → flip to OpenGL -Z
    let oz_min = -(cube.origin[2] + cube.size[2] + infl) * MODEL_SCALE;

    let w = (cube.size[0] + infl * 2.0) * MODEL_SCALE;
    let h = (cube.size[1] + infl * 2.0) * MODEL_SCALE;
    let d = (cube.size[2] + infl * 2.0) * MODEL_SCALE;

    let x0 = ox;      let x1 = ox + w;
    let y0 = oy;      let y1 = oy + h;
    let z0 = oz_min;  let z1 = oz_min + d;

    let u  = uv[0];
    let v  = uv[1];
    let pw = cube.size[0]; // pixel width
    let ph = cube.size[1]; // pixel height
    let pd = cube.size[2]; // pixel depth

    // +Y (Top face)    UV: (u+d, v) → size w×d
    push_quad_flip(verts, indices,
        [x0,y1,z0], [x1,y1,z0], [x1,y1,z1], [x0,y1,z1],
        [0.0, 1.0, 0.0],
        if mirror { uv_rev(uv_flip_u(u+pd, v, pw, pd, tw, th)) }
        else      { uv_rev(uv_rect(u+pd, v, pw, pd, tw, th)) },
    );

    // -Y (Bottom face) UV: (u+d+w, v) → size w×d
    push_quad_flip(verts, indices,
        [x0,y0,z1], [x1,y0,z1], [x1,y0,z0], [x0,y0,z0],
        [0.0, -1.0, 0.0],
        if mirror { uv_rev(uv_flip_u(u+pd+pw, v, pw, pd, tw, th)) }
        else      { uv_rev(uv_rect(u+pd+pw, v, pw, pd, tw, th)) },
    );

    // -Z (North/Back face) UV: (u+d+w+d, v+d) → size w×h
    push_quad_flip(verts, indices,
        [x1,y1,z0], [x0,y1,z0], [x0,y0,z0], [x1,y0,z0],
        [0.0, 0.0, -1.0],
        if mirror { uv_flip_u(u+pd+pw+pd, v+pd, pw, ph, tw, th) }
        else      { uv_rect(u+pd+pw+pd, v+pd, pw, ph, tw, th) },
    );

    // +Z (South/Front face) UV: (u+d, v+d) → size w×h
    push_quad_flip(verts, indices,
        [x0,y1,z1], [x1,y1,z1], [x1,y0,z1], [x0,y0,z1],
        [0.0, 0.0, 1.0],
        if mirror { uv_flip_u(u+pd, v+pd, pw, ph, tw, th) }
        else      { uv_rect(u+pd, v+pd, pw, ph, tw, th) },
    );

    // +X (East/Right face) UV: (u, v+d) → size d×h
    push_quad_flip(verts, indices,
        [x1,y0,z0], [x1,y0,z1], [x1,y1,z1], [x1,y1,z0],
        [1.0, 0.0, 0.0],
        uv_rev(uv_rect(u, v+pd, pd, ph, tw, th)),
    );

    // -X (West/Left face) UV: (u+d+w, v+d) → size d×h
    push_quad_flip(verts, indices,
        [x0,y0,z1], [x0,y0,z0], [x0,y1,z0], [x0,y1,z1],
        [-1.0, 0.0, 0.0],
        uv_rev(uv_rect(u+pd+pw, v+pd, pd, ph, tw, th)),
    );
}

// ---------------------------------------------------------------------------
// Bedrock per-face UV cube generation
// ---------------------------------------------------------------------------

/// Push faces for a Bedrock cube using per-face UV mapping.
/// Only faces with UV data are rendered.
fn push_bedrock_perface_cube(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    cube:    &BedrockCube,
    pfuv:    &BedrockPerFaceUv,
    tw: f32, th: f32,
) {
    let infl = cube.inflate;
    let ox = (cube.origin[0] - infl) * MODEL_SCALE;
    let oy = (cube.origin[1] - infl) * MODEL_SCALE;
    let oz_min = -(cube.origin[2] + cube.size[2] + infl) * MODEL_SCALE;

    let w = (cube.size[0] + infl * 2.0) * MODEL_SCALE;
    let h = (cube.size[1] + infl * 2.0) * MODEL_SCALE;
    let d = (cube.size[2] + infl * 2.0) * MODEL_SCALE;

    let x0 = ox;      let x1 = ox + w;
    let y0 = oy;      let y1 = oy + h;
    let z0 = oz_min;  let z1 = oz_min + d;

    let face_uvs = |fuv: &BedrockFaceUv| -> [[f32; 2]; 4] {
        let u0 = fuv.uv[0] / tw;
        let u1 = (fuv.uv[0] + fuv.uv_size[0]) / tw;
        let v0 = fuv.uv[1] / th;
        let v1 = (fuv.uv[1] + fuv.uv_size[1]) / th;
        [[u0,v0], [u1,v0], [u1,v1], [u0,v1]]
    };

    // North (-Z)
    if let Some(ref fuv) = pfuv.north {
        push_quad_flip(verts, indices,
            [x1,y1,z0], [x0,y1,z0], [x0,y0,z0], [x1,y0,z0],
            [0.0, 0.0, -1.0], face_uvs(fuv),
        );
    }
    // South (+Z)
    if let Some(ref fuv) = pfuv.south {
        push_quad_flip(verts, indices,
            [x0,y1,z1], [x1,y1,z1], [x1,y0,z1], [x0,y0,z1],
            [0.0, 0.0, 1.0], face_uvs(fuv),
        );
    }
    // East (+X)
    if let Some(ref fuv) = pfuv.east {
        push_quad_flip(verts, indices,
            [x1,y0,z0], [x1,y0,z1], [x1,y1,z1], [x1,y1,z0],
            [1.0, 0.0, 0.0], face_uvs(fuv),
        );
    }
    // West (-X)
    if let Some(ref fuv) = pfuv.west {
        push_quad_flip(verts, indices,
            [x0,y0,z1], [x0,y0,z0], [x0,y1,z0], [x0,y1,z1],
            [-1.0, 0.0, 0.0], face_uvs(fuv),
        );
    }
    // Up (+Y)
    if let Some(ref fuv) = pfuv.up {
        push_quad_flip(verts, indices,
            [x0,y1,z0], [x1,y1,z0], [x1,y1,z1], [x0,y1,z1],
            [0.0, 1.0, 0.0], face_uvs(fuv),
        );
    }
    // Down (-Y)
    if let Some(ref fuv) = pfuv.down {
        push_quad_flip(verts, indices,
            [x0,y0,z1], [x1,y0,z1], [x1,y0,z0], [x0,y0,z0],
            [0.0, -1.0, 0.0], face_uvs(fuv),
        );
    }
}

// ---------------------------------------------------------------------------
// Java Edition element generation
// ---------------------------------------------------------------------------

/// Push one Java Edition face quad, record its texture variable.
/// `flip` = true reverses vertex + UV order to convert CW → CCW winding.
fn push_java_face(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    ftm:     &mut Vec<FaceTextureEntry>,
    p0: [f32;3], p1: [f32;3], p2: [f32;3], p3: [f32;3],
    normal: [f32;3],
    face: &JavaFace,
    default_uv: [f32; 4],
    tw: f32, th: f32,
    flip: bool,
) {
    let first_v = verts.len() as u32;
    let uvs = java_face_uvs(face, default_uv, tw, th);
    if flip {
        push_quad(verts, indices, p3, p2, p1, p0, normal,
            [uvs[3], uvs[2], uvs[1], uvs[0]]);
    } else {
        push_quad(verts, indices, p0, p1, p2, p3, normal, uvs);
    }
    let tex_var = face.texture.as_deref().unwrap_or("#missing").to_string();
    ftm.push(FaceTextureEntry { first_vertex: first_v, vertex_count: 4, texture_var: tex_var });
}

/// Push all faces for a Java Edition element.
/// Vertices are generated in pixel coordinates, then converted to block units
/// by the caller (after rotation).
///
/// Winding convention: CCW from outside (FrontFace::Ccw + CullMode::Back).
/// North/South/Up/Down are naturally CCW in Java format.
/// East/West are naturally CW → `flip = true` reverses them.
fn push_java_element(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    face_texture_map: &mut Vec<FaceTextureEntry>,
    element: &JavaElement,
    tw: f32, th: f32,
) {
    let from = element.from;
    let to   = element.to;

    // Pixel coordinates (ensure min/max ordering)
    let x0 = from[0].min(to[0]);
    let y0 = from[1].min(to[1]);
    let z0 = from[2].min(to[2]);
    let x1 = from[0].max(to[0]);
    let y1 = from[1].max(to[1]);
    let z1 = from[2].max(to[2]);

    // Face dimensions in pixels (for default UV)
    let dx = x1 - x0;
    let dy = y1 - y0;
    let dz = z1 - z0;

    // North (-Z) — CCW from outside ✓
    if let Some(ref face) = element.faces.north {
        push_java_face(verts, indices, face_texture_map,
            [x0,y1,z0], [x1,y1,z0], [x1,y0,z0], [x0,y0,z0],
            [0.0, 0.0, -1.0], face, [0.0, 0.0, dx, dy], tw, th, false);
    }
    // South (+Z) — CCW from outside ✓
    if let Some(ref face) = element.faces.south {
        push_java_face(verts, indices, face_texture_map,
            [x1,y1,z1], [x0,y1,z1], [x0,y0,z1], [x1,y0,z1],
            [0.0, 0.0, 1.0], face, [0.0, 0.0, dx, dy], tw, th, false);
    }
    // East (+X) — CW from outside → flip to CCW
    if let Some(ref face) = element.faces.east {
        push_java_face(verts, indices, face_texture_map,
            [x1,y1,z1], [x1,y1,z0], [x1,y0,z0], [x1,y0,z1],
            [1.0, 0.0, 0.0], face, [0.0, 0.0, dz, dy], tw, th, true);
    }
    // West (-X) — CW from outside → flip to CCW
    if let Some(ref face) = element.faces.west {
        push_java_face(verts, indices, face_texture_map,
            [x0,y1,z0], [x0,y1,z1], [x0,y0,z1], [x0,y0,z0],
            [-1.0, 0.0, 0.0], face, [0.0, 0.0, dz, dy], tw, th, true);
    }
    // Up (+Y) — CCW from outside ✓
    if let Some(ref face) = element.faces.up {
        push_java_face(verts, indices, face_texture_map,
            [x0,y1,z1], [x1,y1,z1], [x1,y1,z0], [x0,y1,z0],
            [0.0, 1.0, 0.0], face, [0.0, 0.0, dx, dz], tw, th, false);
    }
    // Down (-Y) — CCW from outside ✓
    if let Some(ref face) = element.faces.down {
        push_java_face(verts, indices, face_texture_map,
            [x0,y0,z0], [x1,y0,z0], [x1,y0,z1], [x0,y0,z1],
            [0.0, -1.0, 0.0], face, [0.0, 0.0, dx, dz], tw, th, false);
    }
}

/// Compute UV coordinates for a Java Edition face.
fn java_face_uvs(
    face:       &JavaFace,
    default_uv: [f32; 4],
    tw: f32, th: f32,
) -> [[f32; 2]; 4] {
    let uv = face.uv.unwrap_or(default_uv);
    let u0 = uv[0] / tw;
    let u1 = uv[2] / tw;
    let v0 = uv[1] / th;
    let v1 = uv[3] / th;

    let base = [[u0,v0], [u1,v0], [u1,v1], [u0,v1]];
    rotate_uvs(&base, face.rotation)
}

/// Apply UV rotation (0, 90, 180, 270 degrees).
fn rotate_uvs(uvs: &[[f32; 2]; 4], rotation: u32) -> [[f32; 2]; 4] {
    match rotation {
        90  => [uvs[3], uvs[0], uvs[1], uvs[2]],
        180 => [uvs[2], uvs[3], uvs[0], uvs[1]],
        270 => [uvs[1], uvs[2], uvs[3], uvs[0]],
        _   => *uvs,
    }
}

// ---------------------------------------------------------------------------
// Shared quad / UV helpers
// ---------------------------------------------------------------------------

/// Push one quad (4 verts) as 2 triangles, CCW winding (FrontFace::Ccw).
/// Vertex order p0→p1→p2→p3 must already be CCW from the face's outside.
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

/// Same as push_quad but reverses vertex + UV order to flip CW → CCW winding.
/// Use this when source geometry is CW from outside (Bedrock faces, Java E/W).
#[inline]
fn push_quad_flip(
    verts:   &mut Vec<PlayerVertex>,
    indices: &mut Vec<u32>,
    p0: [f32; 3], p1: [f32; 3], p2: [f32; 3], p3: [f32; 3],
    normal: [f32; 3],
    uvs: [[f32; 2]; 4],
) {
    push_quad(verts, indices,
        p3, p2, p1, p0, normal,
        [uvs[3], uvs[2], uvs[1], uvs[0]],
    );
}

/// UV rect: (px, py) top-left corner, pw×ph size, normalised.
fn uv_rect(px: f32, py: f32, pw: f32, ph: f32, tw: f32, th: f32) -> [[f32; 2]; 4] {
    let u0 = px / tw;
    let u1 = (px + pw) / tw;
    let v0 = py / th;
    let v1 = (py + ph) / th;
    [[u0,v0], [u1,v0], [u1,v1], [u0,v1]]
}

/// Horizontally flipped UV rect (for mirror bones).
fn uv_flip_u(px: f32, py: f32, pw: f32, ph: f32, tw: f32, th: f32) -> [[f32; 2]; 4] {
    let u0 = px / tw;
    let u1 = (px + pw) / tw;
    let v0 = py / th;
    let v1 = (py + ph) / th;
    [[u1,v0], [u0,v0], [u0,v1], [u1,v1]]
}

/// Reverse UV order to match reversed vertex winding.
fn uv_rev(uvs: [[f32; 2]; 4]) -> [[f32; 2]; 4] {
    [uvs[3], uvs[2], uvs[1], uvs[0]]
}
