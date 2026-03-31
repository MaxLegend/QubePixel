// =============================================================================
// QubePixel — BlockDefinition + BlockRegistry  (loaded from JSON files)
// =============================================================================
//
// Loading pipeline:
//   1. Read  assets/blocks/block_registry.json  → list of block IDs
//   2. For each ID, read  assets/blocks/<id>.json  → full block definition
//   3. If the assets directory is missing, fall back to embedded defaults
//      so the game always works even without external files.
//
// Adding a new block:
//   1. Open  assets/blocks/block_registry.json  and add the block's ID
//      to the "blocks" array.
//   2. Create  assets/blocks/<id>.json  with the desired properties
//      (see any existing block file for the schema).
//   3. Restart the game — the new block is available immediately.
//
// Block IDs:
//   Numeric ID 0 is always Air (implicit, not in any file).
//   Numeric IDs 1..=255 are assigned in registry order (first entry → 1, …).
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use crate::debug_log;

// ---------------------------------------------------------------------------
// BlockDefinition — deserialised from an individual <id>.json
// ---------------------------------------------------------------------------
/// A single face colour override. Expandable (texture, emission, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceOverride {
    /// RGB colour (0..1). If absent, falls back to block's base `color`.
    #[serde(default)]
    pub color: [f32; 3],
    /// Name of the texture in the atlas (e.g. "grass_top", "dirt", "stone_side").
    /// If absent, this face uses vertex colour only (white tile in atlas).
    #[serde(default)]
    pub texture: Option<String>,
}
/// Optional per-face colour overrides.
/// If a face entry is missing, the block's base `color` is used.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct FaceColors {
    #[serde(default)]
    pub top:    Option<FaceOverride>,
    #[serde(default)]
    pub bottom: Option<FaceOverride>,
    #[serde(default)]
    pub sides:  Option<FaceOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDefinition {
    /// Unique string identifier, e.g. "grass", "dirt", "stone".
    pub id: String,

    /// Human-readable name shown in tooltips / UI.
    #[serde(default)]
    pub display_name: String,

    /// Short description of the block.
    #[serde(default)]
    pub description: String,

    /// Base RGB colour (0..1). Used when face-specific colours are absent.
    pub color: [f32; 3],

    /// Name of the texture in the atlas (e.g. "grass_top", "dirt", "stone_side").
    /// If absent, this face uses vertex colour only (white tile in atlas).
    #[serde(default)]
    pub texture: Option<String>,

    /// Whether the block blocks movement and is meshed with solid faces.
    #[serde(default = "default_true")]
    pub solid: bool,

    /// Whether light can pass through this block.
    #[serde(default)]
    pub transparent: bool,

    /// Light emission level (0..15).  0 = no light emitted.
    #[serde(default)]
    pub emit_light: u8,

    /// Optional per-face colour overrides.
    #[serde(default)]
    pub faces: FaceColors,
}

fn default_true() -> bool { true }

impl BlockDefinition {
    /// Return the colour for a given face direction.
    ///
    /// `face`: 0 = +X, 1 = -X, 2 = +Y (top), 3 = -Y (bottom),
    ///         4 = +Z, 5 = -Z.
    pub fn color_for_face(&self, face: u8) -> [f32; 3] {
        match face {
            2 => self.faces.top.as_ref().map(|f| f.color).unwrap_or(self.color),
            3 => self.faces.bottom.as_ref().map(|f| f.color).unwrap_or(self.color),
            _ => self.faces.sides.as_ref().map(|f| f.color).unwrap_or(self.color),
        }
    }

    /// Return the atlas texture name for a given face direction.
    /// Returns `None` if this face has no texture (falls back to white tile).
    ///
    /// `face`: 0 = +X, 1 = -X, 2 = +Y (top), 3 = -Y (bottom),
    ///         4 = +Z, 5 = -Z.
    pub fn texture_for_face(&self, face: u8) -> Option<&str> {
        match face {
            2 => self.faces.top.as_ref().and_then(|f| f.texture.as_deref()),
            3 => self.faces.bottom.as_ref().and_then(|f| f.texture.as_deref()),
            _ => self.faces.sides.as_ref().and_then(|f| f.texture.as_deref()),
        }
    }
}

// ---------------------------------------------------------------------------
// BlockRegistryFile — the top-level block_registry.json schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BlockRegistryFile {
    /// Schema version — currently must be 1.
    format_version: u32,
    /// Ordered list of block IDs.  Order determines numeric IDs (1-indexed).
    blocks: Vec<String>,
}

// ---------------------------------------------------------------------------
// BlockRegistry — holds all loaded block types
//
// Block ID 0 is always Air (implicit, not in any file).
// Block IDs 1..=255 map to definitions[id-1].
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct BlockRegistry {
    definitions: Vec<BlockDefinition>,
    id_to_index: HashMap<String, u8>,
}

/// Default relative path to the block assets directory.
const BLOCKS_DIR: &str = "assets/blocks";
/// Name of the master registry file inside the blocks directory.
const REGISTRY_FILE: &str = "block_registry.json";

impl BlockRegistry {
    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /// Load the block registry.
    ///
    /// Tries to read from `<assets_dir>/block_registry.json` and then each
    /// individual `<assets_dir>/<block_id>.json`.  Falls back to embedded
    /// defaults when the directory or files are not found, so the game can
    /// always start even without external assets.
    pub fn load() -> Self {
        Self::load_from(BLOCKS_DIR)
    }

    /// Same as `load()` but allows specifying a custom assets directory.
    pub fn load_from(assets_dir: &str) -> Self {
        let dir = PathBuf::from(assets_dir);
        let registry_path = dir.join(REGISTRY_FILE);

        // Attempt filesystem load
        if registry_path.exists() {
            debug_log!(
                "BlockRegistry", "load",
                "Found block registry at {:?}",
                registry_path
            );
            match Self::load_from_fs(&dir, &registry_path) {
                Ok(reg) => return reg,
                Err(e) => {
                    debug_log!(
                        "BlockRegistry", "load",
                        "Filesystem load failed: {}, falling back to embedded defaults",
                        e
                    );
                }
            }
        } else {
            debug_log!(
                "BlockRegistry", "load",
                "Registry file not found at {:?}, using embedded defaults",
                registry_path
            );
        }

        // Fallback: embedded defaults
        Self::load_embedded()
    }

    /// Look up a block by numeric ID.  Returns `None` for Air (id = 0) and
    /// unknown ids.
    pub fn get(&self, block_id: u8) -> Option<&BlockDefinition> {
        if block_id == 0 {
            return None;
        }
        self.definitions.get((block_id - 1) as usize)
    }

    /// Look up numeric ID by string identifier.
    pub fn id_for(&self, name: &str) -> Option<u8> {
        self.id_to_index.get(name).copied()
    }

    /// Number of registered block types (excluding Air).
    pub fn len(&self) -> usize {
        self.definitions.len()
    }

    /// Returns true if no blocks are registered.
    pub fn is_empty(&self) -> bool {
        self.definitions.is_empty()
    }

    // -----------------------------------------------------------------------
    // Filesystem loader
    // -----------------------------------------------------------------------

    fn load_from_fs(dir: &Path, registry_path: &Path) -> Result<Self, String> {
        // 1. Read block_registry.json
        let registry_str = std::fs::read_to_string(registry_path)
            .map_err(|e| format!("Failed to read {:?}: {}", registry_path, e))?;

        let registry: BlockRegistryFile = serde_json::from_str(&registry_str)
            .map_err(|e| format!("Failed to parse block_registry.json: {}", e))?;

        if registry.format_version != 1 {
            return Err(format!(
                "Unsupported block_registry format_version: {} (expected 1)",
                registry.format_version
            ));
        }

        debug_log!(
            "BlockRegistry", "load_from_fs",
            "Registry lists {} block(s): {:?}",
            registry.blocks.len(),
            &registry.blocks
        );

        // 2. Load each individual block JSON
        let mut definitions = Vec::with_capacity(registry.blocks.len());
        let mut id_to_index = HashMap::new();

        for (i, block_id) in registry.blocks.iter().enumerate() {
            let numeric_id = (i + 1) as u8;

            let block_path = dir.join(format!("{}.json", block_id));
            if !block_path.exists() {
                debug_log!(
                    "BlockRegistry", "load_from_fs",
                    "WARNING: block file {:?} not found — skipping '{}'",
                    block_path, block_id
                );
                continue;
            }

            let block_str = std::fs::read_to_string(&block_path)
                .map_err(|e| {
                    format!("Failed to read block file {:?}: {}", block_path, e)
                })?;

            let def: BlockDefinition = serde_json::from_str(&block_str)
                .map_err(|e| {
                    format!(
                        "Failed to parse block '{}': {}",
                        block_id, e
                    )
                })?;

            // Sanity check: the `id` inside the JSON should match the
            // filename / registry entry.
            if def.id != *block_id {
                debug_log!(
                    "BlockRegistry", "load_from_fs",
                    "WARNING: block file '{}' declares id='{}' but registry says '{}' — using registry id",
                    block_path.display(), def.id, block_id
                );
            }

            debug_log!(
                "BlockRegistry", "load_from_fs",
                "Registered block '{}' => numeric id {}, display_name='{}'",
                block_id, numeric_id, def.display_name
            );

            id_to_index.insert(block_id.clone(), numeric_id);
            definitions.push(def);
        }

        debug_log!(
            "BlockRegistry", "load_from_fs",
            "Total block types loaded from filesystem: {}",
            definitions.len()
        );

        Ok(Self { definitions, id_to_index })
    }

    // -----------------------------------------------------------------------
    // Embedded fallback — used when assets/ directory is absent
    // -----------------------------------------------------------------------

    fn load_embedded() -> Self {
        debug_log!(
            "BlockRegistry", "load_embedded",
            "Loading embedded default blocks"
        );

        let defaults = include_str!("../../../assets/block_registry.json");
        let registry: BlockRegistryFile = serde_json::from_str(defaults)
            .expect("[BlockRegistry][load_embedded] Failed to parse embedded block_registry.json");

        let default_defs: &[(&str, &str)] = &[
            // (block_id, embedded JSON)
            ("grass", include_str!("../../../assets/blocks/grass.json")),
            ("dirt",  include_str!("../../../assets/blocks/dirt.json")),
            ("stone", include_str!("../../../assets/blocks/stone.json")),
        ];

        let mut definitions = Vec::with_capacity(registry.blocks.len());
        let mut id_to_index = HashMap::new();

        for (i, block_id) in registry.blocks.iter().enumerate() {
            let numeric_id = (i + 1) as u8;

            // Find matching embedded definition
            let json_str = default_defs
                .iter()
                .find(|(id, _)| *id == block_id.as_str())
                .map(|(_, json)| *json);

            if let Some(json_str) = json_str {
                let def: BlockDefinition = serde_json::from_str(json_str)
                    .expect("[BlockRegistry][load_embedded] Failed to parse embedded block JSON");

                debug_log!(
                    "BlockRegistry", "load_embedded",
                    "Registered block '{}' => id {} (embedded)",
                    def.id, numeric_id
                );

                id_to_index.insert(block_id.clone(), numeric_id);
                definitions.push(def);
            } else {
                debug_log!(
                    "BlockRegistry", "load_embedded",
                    "WARNING: no embedded definition for '{}' — skipping",
                    block_id
                );
            }
        }

        debug_log!(
            "BlockRegistry", "load_embedded",
            "Total embedded block types loaded: {}",
            definitions.len()
        );

        Self { definitions, id_to_index }
    }
}
