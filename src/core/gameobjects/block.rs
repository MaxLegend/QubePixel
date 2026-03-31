// =============================================================================
// QubePixel — BlockDefinition + BlockRegistry  (loaded from embedded JSON)
// =============================================================================

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Embedded block definitions (JSON).
// In production this can be replaced by include_str!("blocks.json") or
// runtime file loading — the struct layout stays the same.
// ---------------------------------------------------------------------------
const BLOCKS_JSON: &str = r#"[
  {
    "id": "grass",
    "color": [0.22, 0.52, 0.14],
    "solid": true,
    "transparent": false
  },
  {
    "id": "dirt",
    "color": [0.40, 0.26, 0.10],
    "solid": true,
    "transparent": false
  },
  {
    "id": "stone",
    "color": [0.50, 0.50, 0.50],
    "solid": true,
    "transparent": false
  }
]"#;

// ---------------------------------------------------------------------------
// BlockDefinition — one entry from the JSON array
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockDefinition {
    pub id: String,
    pub color: [f32; 3],
    pub solid: bool,
    pub transparent: bool,
}

// ---------------------------------------------------------------------------
// BlockRegistry — holds all loaded block types
//
// Block ID 0 is always Air (implicit, not in JSON).
// Block IDs 1..=255 map to definitions[id-1].
// ---------------------------------------------------------------------------
pub struct BlockRegistry {
    definitions: Vec<BlockDefinition>,
    id_to_index: HashMap<String, u8>,
}

impl BlockRegistry {
    /// Parse block definitions from the embedded JSON constant.
    pub fn load() -> Self {
        let defs: Vec<BlockDefinition> =
            serde_json::from_str(BLOCKS_JSON)
                .expect("[BlockRegistry][load] Failed to parse block definitions JSON");

        let mut id_to_index = HashMap::new();
        for (i, def) in defs.iter().enumerate() {
            let numeric_id = (i + 1) as u8; // 1-indexed; 0 reserved for Air
            debug_log!(
                "BlockRegistry", "load",
                "Registered block '{}' => id {}",
                def.id, numeric_id
            );
            id_to_index.insert(def.id.clone(), numeric_id);
        }

        debug_log!(
            "BlockRegistry", "load",
            "Total block types loaded: {}",
            defs.len()
        );

        Self { definitions: defs, id_to_index }
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
