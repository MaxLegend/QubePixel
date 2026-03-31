// =============================================================================
// QubePixel — BlockDefinition + BlockRegistry  (loaded from asset files)
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use crate::debug_log;

// -----------------------------------------------------------------------------
// BlockDefinition — loaded from individual JSON files in assets/blocks/
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTextures {
    #[serde(default)]
    pub all: Option<String>,
    #[serde(default)]
    pub top: Option<String>,
    #[serde(default)]
    pub bottom: Option<String>,
    #[serde(default)]
    pub side: Option<String>,
    #[serde(default)]
    pub front: Option<String>,
    #[serde(default)]
    pub back: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDefinition {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub color: [f32; 3],
    pub solid: bool,
    pub transparent: bool,
    #[serde(default = "default_hardness")]
    pub hardness: f32,
    #[serde(default = "default_blast_resistance")]
    pub blast_resistance: f32,
    #[serde(default)]
    pub light_emission: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default = "default_slipperiness")]
    pub slipperiness: f32,
    #[serde(default)]
    pub requires_tool: bool,
    #[serde(default = "default_stack_size")]
    pub stack_size: u8,
    #[serde(default)]
    pub textures: Option<BlockTextures>,
}

fn default_hardness() -> f32 { 0.5 }
fn default_blast_resistance() -> f32 { 0.5 }
fn default_friction() -> f32 { 0.6 }
fn default_slipperiness() -> f32 { 0.6 }
fn default_stack_size() -> u8 { 64 }

// -----------------------------------------------------------------------------
// BlockRegistryConfig — loaded from assets/block_registry.json
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRegistryConfig {
    pub blocks: Vec<String>,
}

// -----------------------------------------------------------------------------
// BlockRegistry — holds all loaded block types
//
// Block ID 0 is always Air (implicit, not in registry).
// Block IDs 1..=255 map to definitions[id-1].
// -----------------------------------------------------------------------------
pub struct BlockRegistry {
    definitions: Vec<BlockDefinition>,
    id_to_index: HashMap<String, u8>,
}

impl BlockRegistry {
    /// Load block definitions from asset files.
    /// 
    /// Directory structure:
    /// - assets/block_registry.json — список всех блоков
    /// - assets/blocks/<id>.json — определение каждого блока
    pub fn load() -> Self {
        let registry_path = PathBuf::from("assets/block_registry.json");
        
        let registry_config: BlockRegistryConfig = match fs::read_to_string(&registry_path) {
            Ok(content) => {
                debug_log!("BlockRegistry", "load", "Loading registry from {:?}", registry_path);
                serde_json::from_str(&content)
                    .expect("[BlockRegistry][load] Failed to parse block_registry.json")
            }
            Err(e) => {
                panic!("[BlockRegistry][load] Failed to read block_registry.json: {}", e);
            }
        };

        let mut definitions = Vec::new();
        let mut id_to_index = HashMap::new();

        for block_id in &registry_config.blocks {
            let block_path = PathBuf::from(format!("assets/blocks/{}.json", block_id));
            
            debug_log!("BlockRegistry", "load", "Loading block '{}' from {:?}", block_id, block_path);
            
            let content = match fs::read_to_string(&block_path) {
                Ok(c) => c,
                Err(e) => {
                    panic!("[BlockRegistry][load] Failed to read block file {:?}: {}", block_path, e);
                }
            };

            let def: BlockDefinition = match serde_json::from_str(&content) {
                Ok(d) => d,
                Err(e) => {
                    panic!("[BlockRegistry][load] Failed to parse block definition from {:?}: {}", block_path, e);
                }
            };

            // Validate that the file ID matches the expected ID
            if def.id != *block_id {
                panic!(
                    "[BlockRegistry][load] Block ID mismatch: expected '{}', found '{}' in {:?}",
                    block_id, def.id, block_path
                );
            }

            let numeric_id = (definitions.len() + 1) as u8; // 1-indexed; 0 reserved for Air
            
            debug_log!(
                "BlockRegistry", "load",
                "Registered block '{}' => id {} (solid={}, transparent={})",
                def.id, numeric_id, def.solid, def.transparent
            );
            
            id_to_index.insert(def.id.clone(), numeric_id);
            definitions.push(def);
        }

        debug_log!(
            "BlockRegistry", "load",
            "Total block types loaded: {}",
            definitions.len()
        );

        Self { definitions, id_to_index }
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
}
