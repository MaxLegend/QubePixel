//! # Biome Dictionary & Registry
//!
//! Provides a centralized, tag-based biome classification system for
//! QubePixel's terrain generation pipeline.
//!
//! ## Overview
//!
//! Each biome is defined as a [`BiomeDefinition`] carrying:
//!
//! * A unique numeric **ID** (`u32`).
//! * A human-readable **name**.
//! * A set of [`ClimateTag`]s describing its temperature, moisture, and
//!   terrain characteristics.
//! * Preferred **surface** and **floor** block names.
//!
//! Biomes are registered into a [`BiomeDictionary`] which maintains:
//!
//! * A primary lookup table (`ID → definition`).
//! * A reverse tag index (`tag → set of IDs`) for efficient multi-tag
//!   queries.
//!
//! ## Usage
//!
//! ```ignore
//! use qubepixel_terrain::biome_registry::{BiomeDictionary, ClimateTag};
//!
//! let dict = BiomeDictionary::create_default();
//!
//! // Find all ocean biomes.
//! let oceans = dict.find_by_tags(&[ClimateTag::Ocean]);
//!
//! // Find the best match for cold, mountainous terrain.
//! let biome_id = dict.find_best_matching(&[
//!     ClimateTag::Mountain,
//!     ClimateTag::Icy,
//! ]);
//! ```
//!
//! ## Default biome set
//!
//! [`BiomeDictionary::create_default`] registers **32 biomes** covering:
//!
//! * 5 ocean variants (deep, normal, frozen, warm, coral).
//! * 3 beach variants (temperate, snowy, desert).
//! * 4 plains variants (temperate, snowy, tropical, savanna).
//! * 4 forest variants (temperate, snowy, tropical, taiga).
//! * 2 desert variants (sand, red sand).
//! * 2 swamp variants (temperate, mangrove).
//! * 4 mountain variants (temperate, snowy, volcanic, mesa).
//! * 2 tundra / ice plains variants.
//! * 2 jungle variants.
//! * 2 river variants.
//!
//! ## Thread safety
//!
//! `BiomeDictionary` is `Send + Sync` and designed to be shared across rayon
//! workers. Lookups are read-only after construction (no interior mutability).

use std::collections::HashMap;
use crate::{debug_log, ext_debug_log};
// ---------------------------------------------------------------------------
// ClimateTag
// ---------------------------------------------------------------------------

/// Describes a single axis of a biome's climate / terrain classification.
///
/// Biomes carry **multiple** tags so that they can be found via intersection
/// queries (e.g. "ocean AND cold").
///
/// # Hash / Eq
///
/// Implements `Hash` + `Eq` so tags can be used as `HashMap` / `HashSet` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClimateTag {
    /// High temperature biome (deserts, savannas, jungles).
    Hot,

    /// Moderate temperature biome (temperate forests, plains).
    Warm,

    /// Low temperature biome (taiga, tundra).
    Cold,

    /// Extremely low temperature (ice plains, frozen ocean, glaciers).
    Icy,

    /// Fully aquatic (oceans, deep oceans).
    Ocean,

    /// River or stream channel.
    River,

    /// High-elevation terrain (mountains, peaks).
    Mountain,

    /// Flat, low-elevation terrain.
    Plains,

    /// Dense tree coverage.
    Forest,

    /// Very low precipitation (deserts).
    Desert,

    /// Waterlogged, low-elevation terrain (swamps, bogs).
    Swamp,

    /// Coastal transition zone between land and ocean.
    Beach,
}

impl std::fmt::Display for ClimateTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClimateTag::Hot => write!(f, "Hot"),
            ClimateTag::Warm => write!(f, "Warm"),
            ClimateTag::Cold => write!(f, "Cold"),
            ClimateTag::Icy => write!(f, "Icy"),
            ClimateTag::Ocean => write!(f, "Ocean"),
            ClimateTag::River => write!(f, "River"),
            ClimateTag::Mountain => write!(f, "Mountain"),
            ClimateTag::Plains => write!(f, "Plains"),
            ClimateTag::Forest => write!(f, "Forest"),
            ClimateTag::Desert => write!(f, "Desert"),
            ClimateTag::Swamp => write!(f, "Swamp"),
            ClimateTag::Beach => write!(f, "Beach"),
        }
    }
}

// ---------------------------------------------------------------------------
// BiomeDefinition
// ---------------------------------------------------------------------------

/// Full definition of a single biome.
///
/// Biomes are **data-only** — they do not contain any generation logic.
/// The [`BiomePipeline`](crate::pipeline::BiomePipeline) is responsible for
/// mapping world conditions to biome IDs.
///
/// # Block names
///
/// `surface_block` and `floor_block` are string names that must be resolved
/// through the [`BlockRegistry`](crate::core::gameobjects::block::BlockRegistry)
/// at generation time.
#[derive(Debug, Clone)]
pub struct BiomeDefinition {
    /// Unique biome identifier.
    ///
    /// Convention: ID ranges are grouped by category:
    ///
    /// | Range | Category |
    /// |-------|----------|
    /// | 100–109 | Oceans |
    /// | 200–209 | Beaches |
    /// | 300–309 | Plains |
    /// | 400–409 | Forests |
    /// | 500–509 | Deserts |
    /// | 600–609 | Swamps |
    /// | 700–709 | Mountains |
    /// | 800–809 | Tundra / Ice |
    /// | 900–909 | Jungles |
    /// | 1000–1009 | Rivers |
    pub id: u32,

    /// Human-readable biome name (static, interned).
    pub name: &'static str,

    /// Climate tags for classification and lookup.
    pub tags: Vec<ClimateTag>,

    /// Preferred surface block name (e.g. `"grass_gneiss"`, `"sand"`).
    pub surface_block: &'static str,

    /// Underwater floor block name (e.g. `"sand"`, `"gravel"`).
    pub floor_block: &'static str,

    /// RGB multiplier applied to foliage blocks (grass, leaves) in this biome.
    /// [1.0, 1.0, 1.0] = neutral (no tint). Desert = yellowish, jungle = vivid green.
    pub foliage_color: [f32; 3],

    /// RGB multiplier applied to water surfaces in this biome.
    /// Deep ocean = dark blue, tropical = teal, swamp = murky green.
    pub water_color: [f32; 3],

    /// Subtle RGB tint blended into the ambient light when the camera is in this biome.
    /// Hot biomes get a warm yellow cast; icy biomes get a cool blue cast.
    /// Controlled by the `BIOME_AMBIENT_TINT_ENABLED` config toggle.
    pub ambient_tint: [f32; 3],
}

// ---------------------------------------------------------------------------
// BiomeDictionary
// ---------------------------------------------------------------------------

/// Central registry for all biome definitions with tag-based indexing.
///
/// This is the single source of truth for biome metadata during terrain
/// generation. Construct once (via [`new`](Self::new) or
/// [`create_default`](Self::create_default)), then share by reference across
/// all worker threads.
///
/// # Performance
///
/// * `get(id)` — **O(1)** via `HashMap`.
/// * `find_by_tags(tags)` — **O(m · n)** where `m` = number of tags and
///   `n` = number of candidate biomes (typically small due to tag pruning).
/// * `find_best_matching(tags)` — **O(b · m)** where `b` = total biomes and
///   `m` = query tag count.
///
/// # Thread safety
///
/// `BiomeDictionary` is `Send + Sync`. After construction it is effectively
/// immutable (no `&mut` methods), so sharing is trivial.
pub struct BiomeDictionary {
    /// Primary storage: biome ID → definition.
    biomes: HashMap<u32, BiomeDefinition>,

    /// Reverse index: tag → list of biome IDs possessing that tag.
    tag_index: HashMap<ClimateTag, Vec<u32>>,
}

impl BiomeDictionary {
    /// Create an empty dictionary.
    ///
    /// Use [`register`](Self::register) to populate, or call
    /// [`create_default`](Self::create_default) for the built-in set.
    pub fn new() -> Self {
        debug_log!("BiomeDictionary", "new", "creating empty dictionary");
        Self {
            biomes: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Register a biome definition.
    ///
    /// If a biome with the same ID already exists it is **replaced** and a
    /// warning is logged.
    ///
    /// # Panics
    ///
    /// Does not panic, but logs a warning on duplicate IDs.
    pub fn register(&mut self, biome: BiomeDefinition) {
        let id = biome.id;
        if self.biomes.contains_key(&id) {
            ext_debug_log!(
                "BiomeDictionary",
                "register",
                "overwriting existing biome id={}",
                id
            );
        }

        // Update tag index.
        for &tag in &biome.tags {
            self.tag_index
                .entry(tag)
                .or_insert_with(Vec::new)
                .push(id);
        }

        self.biomes.insert(id, biome);
        debug_log!(
            "BiomeDictionary",
            "register",
            "registered biome id={} total={}",
            id,
            self.biomes.len()
        );
    }

    /// Look up a biome definition by its numeric ID.
    ///
    /// Returns `None` if no biome with that ID has been registered.
    pub fn get(&self, id: u32) -> Option<&BiomeDefinition> {
        self.biomes.get(&id)
    }

    /// Find all biome IDs that have **every** tag in `tags`.
    ///
    /// Returns an empty `Vec` if no biome matches all tags.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cold_oceans = dict.find_by_tags(&[ClimateTag::Ocean, ClimateTag::Cold]);
    /// ```
    pub fn find_by_tags(&self, tags: &[ClimateTag]) -> Vec<u32> {
        if tags.is_empty() {
            // No filter → return all biome IDs.
            return self.biomes.keys().copied().collect();
        }

        // Start with the candidate set from the first tag, then intersect.
        let mut candidates: Option<Vec<u32>> = None;

        for tag in tags {
            if let Some(ids) = self.tag_index.get(tag) {
                match &mut candidates {
                    None => {
                        candidates = Some(ids.clone());
                    }
                    Some(cands) => {
                        // Keep only IDs present in both sets.
                        cands.retain(|id| ids.contains(id));
                    }
                }
            } else {
                // Tag not present in any biome → no matches possible.
                return Vec::new();
            }
        }

        candidates.unwrap_or_default()
    }

    /// Check whether a specific biome has a given tag.
    ///
    /// Returns `false` if the biome ID is not found.
    pub fn has_tag(&self, biome_id: u32, tag: ClimateTag) -> bool {
        match self.biomes.get(&biome_id) {
            Some(def) => def.tags.contains(&tag),
            None => {
                ext_debug_log!(
                    "BiomeDictionary",
                    "has_tag",
                    "unknown biome id={}",
                    biome_id
                );
                false
            }
        }
    }

    /// Get the first biome ID that has **all** of the specified tags.
    ///
    /// Convenience wrapper around [`find_by_tags`](Self::find_by_tags).
    pub fn first_with_tags(&self, tags: &[ClimateTag]) -> Option<u32> {
        let ids = self.find_by_tags(tags);
        ids.into_iter().next()
    }

    /// Find the biome whose tag set has the **largest intersection** with the
    /// given query tags.
    ///
    /// This is the core matching algorithm used by
    /// [`BiomePipeline::determine_biome`](crate::pipeline::BiomePipeline::determine_biome).
    ///
    /// # Scoring
    ///
    /// Each candidate biome is scored by counting how many of the query tags
    /// it possesses. The biome with the highest score wins. Ties are broken
    /// by smallest ID (deterministic).
    ///
    /// # Returns
    ///
    /// `Some(id)` of the best-matching biome, or `None` if the dictionary is
    /// empty.
    pub fn find_best_matching(&self, query_tags: &[ClimateTag]) -> Option<u32> {
        if query_tags.is_empty() || self.biomes.is_empty() {
            return None;
        }

        let query_set: std::collections::HashSet<ClimateTag> =
            query_tags.iter().copied().collect();

        let mut best_id: Option<u32> = None;
        let mut best_score: usize = 0;

        for (&id, def) in &self.biomes {
            let score = def.tags.iter().filter(|t| query_set.contains(t)).count();
            if score > best_score || (score == best_score && best_id.map_or(true, |b| id < b)) {
                best_score = score;
                best_id = Some(id);
            }
        }

        ext_debug_log!(
            "BiomeDictionary",
            "find_best_matching",
            "query={:?} best_id={:?} score={}",
            query_tags,
            best_id,
            best_score
        );

        best_id
    }

    /// Return the total number of registered biomes.
    pub fn len(&self) -> usize {
        self.biomes.len()
    }

    /// Return `true` if no biomes are registered.
    pub fn is_empty(&self) -> bool {
        self.biomes.is_empty()
    }

    /// Create a dictionary pre-populated with QubePixel's default biome set
    /// (32 biomes covering all major climate / geology combinations).
    ///
    /// # Default biome list
    ///
    /// | ID | Name | Tags |
    /// |----|------|------|
    /// | 100 | Deep Ocean | Ocean, Cold |
    /// | 101 | Ocean | Ocean, Warm |
    /// | 102 | Frozen Ocean | Ocean, Icy |
    /// | 103 | Warm Ocean | Ocean, Hot |
    /// | 104 | Coral Reef | Ocean, Hot |
    /// | 200 | Beach | Beach, Warm |
    /// | 201 | Snowy Beach | Beach, Icy |
    /// | 202 | Desert Beach | Beach, Hot, Desert |
    /// | 300 | Plains | Plains, Warm |
    /// | 301 | Snowy Plains | Plains, Cold |
    /// | 302 | Tropical Plains | Plains, Hot |
    /// | 303 | Savanna | Plains, Hot |
    /// | 400 | Forest | Forest, Warm |
    /// | 401 | Snowy Forest | Forest, Cold |
    /// | 402 | Tropical Forest | Forest, Hot |
    /// | 403 | Taiga | Forest, Cold |
    /// | 500 | Desert | Desert, Hot |
    /// | 501 | Red Desert | Desert, Hot |
    /// | 600 | Swamp | Swamp, Warm |
    /// | 601 | Mangrove Swamp | Swamp, Hot |
    /// | 700 | Mountains | Mountain, Cold |
    /// | 701 | Snowy Mountains | Mountain, Icy |
    /// | 702 | Volcanic Mountains | Mountain, Hot |
    /// | 703 | Mesa | Mountain, Hot, Desert |
    /// | 800 | Tundra | Cold, Plains |
    /// | 801 | Ice Plains | Icy, Plains |
    /// | 900 | Jungle | Forest, Hot |
    /// | 901 | Dense Jungle | Forest, Hot |
    /// | 1000 | River | River, Warm |
    /// | 1001 | Frozen River | River, Icy |
    pub fn create_default() -> Self {
        debug_log!("BiomeDictionary", "create_default", "building default biome set");
        let mut dict = Self::new();

        // -- Oceans (100–104) --------------------------------------------
        dict.register(BiomeDefinition {
            id: 100,
            name: "Deep Ocean",
            tags: vec![ClimateTag::Ocean, ClimateTag::Cold],
            surface_block: "water",
            floor_block: "gravel",
            foliage_color: [0.15, 0.35, 0.65],
            water_color:   [0.05, 0.15, 0.55],
            ambient_tint:  [0.80, 0.90, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 101,
            name: "Ocean",
            tags: vec![ClimateTag::Ocean, ClimateTag::Warm],
            surface_block: "water",
            floor_block: "sand",
            foliage_color: [0.25, 0.45, 0.75],
            water_color:   [0.10, 0.30, 0.70],
            ambient_tint:  [0.85, 0.95, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 102,
            name: "Frozen Ocean",
            tags: vec![ClimateTag::Ocean, ClimateTag::Icy],
            surface_block: "ice",
            floor_block: "gravel",
            foliage_color: [0.65, 0.75, 0.85],
            water_color:   [0.50, 0.65, 0.80],
            ambient_tint:  [0.75, 0.85, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 103,
            name: "Warm Ocean",
            tags: vec![ClimateTag::Ocean, ClimateTag::Hot],
            surface_block: "water",
            floor_block: "sand",
            foliage_color: [0.15, 0.65, 0.65],
            water_color:   [0.05, 0.55, 0.70],
            ambient_tint:  [1.00, 0.95, 0.85],
        });
        dict.register(BiomeDefinition {
            id: 104,
            name: "Coral Reef",
            tags: vec![ClimateTag::Ocean, ClimateTag::Hot],
            surface_block: "water",
            floor_block: "coral_sand",
            foliage_color: [0.10, 0.70, 0.60],
            water_color:   [0.00, 0.60, 0.65],
            ambient_tint:  [1.00, 0.95, 0.85],
        });

        // -- Beaches (200–202) -------------------------------------------
        dict.register(BiomeDefinition {
            id: 200,
            name: "Beach",
            tags: vec![ClimateTag::Beach, ClimateTag::Warm],
            surface_block: "sand",
            floor_block: "sand",
            foliage_color: [0.75, 0.80, 0.40],
            water_color:   [0.20, 0.50, 0.80],
            ambient_tint:  [1.00, 0.97, 0.88],
        });
        dict.register(BiomeDefinition {
            id: 201,
            name: "Snowy Beach",
            tags: vec![ClimateTag::Beach, ClimateTag::Icy],
            surface_block: "snow",
            floor_block: "sand",
            foliage_color: [0.60, 0.70, 0.60],
            water_color:   [0.45, 0.60, 0.80],
            ambient_tint:  [0.85, 0.90, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 202,
            name: "Desert Beach",
            tags: vec![ClimateTag::Beach, ClimateTag::Hot, ClimateTag::Desert],
            surface_block: "sand",
            floor_block: "sandstone",
            foliage_color: [0.80, 0.70, 0.30],
            water_color:   [0.10, 0.55, 0.70],
            ambient_tint:  [1.00, 0.93, 0.78],
        });

        // -- Plains (300–303) --------------------------------------------
        dict.register(BiomeDefinition {
            id: 300,
            name: "Plains",
            tags: vec![ClimateTag::Plains, ClimateTag::Warm],
            surface_block: "grass",
            floor_block: "dirt",
            foliage_color: [0.55, 0.75, 0.25],
            water_color:   [0.15, 0.40, 0.75],
            ambient_tint:  [1.00, 1.00, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 301,
            name: "Snowy Plains",
            tags: vec![ClimateTag::Plains, ClimateTag::Cold],
            surface_block: "snow",
            floor_block: "dirt",
            foliage_color: [0.55, 0.65, 0.45],
            water_color:   [0.45, 0.60, 0.80],
            ambient_tint:  [0.85, 0.90, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 302,
            name: "Tropical Plains",
            tags: vec![ClimateTag::Plains, ClimateTag::Hot],
            surface_block: "grass_tropical",
            floor_block: "dirt_tropical",
            foliage_color: [0.20, 0.85, 0.20],
            water_color:   [0.10, 0.60, 0.65],
            ambient_tint:  [1.00, 0.95, 0.82],
        });
        dict.register(BiomeDefinition {
            id: 303,
            name: "Savanna",
            tags: vec![ClimateTag::Plains, ClimateTag::Hot],
            surface_block: "grass_savanna",
            floor_block: "dirt_savanna",
            foliage_color: [0.75, 0.70, 0.20],
            water_color:   [0.20, 0.50, 0.65],
            ambient_tint:  [1.00, 0.93, 0.75],
        });

        // -- Forests (400–403) -------------------------------------------
        dict.register(BiomeDefinition {
            id: 400,
            name: "Forest",
            tags: vec![ClimateTag::Forest, ClimateTag::Warm],
            surface_block: "grass",
            floor_block: "dirt",
            foliage_color: [0.35, 0.65, 0.20],
            water_color:   [0.15, 0.45, 0.70],
            ambient_tint:  [0.95, 1.00, 0.90],
        });
        dict.register(BiomeDefinition {
            id: 401,
            name: "Snowy Forest",
            tags: vec![ClimateTag::Forest, ClimateTag::Cold],
            surface_block: "snow",
            floor_block: "dirt",
            foliage_color: [0.45, 0.65, 0.35],
            water_color:   [0.40, 0.60, 0.75],
            ambient_tint:  [0.88, 0.93, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 402,
            name: "Tropical Forest",
            tags: vec![ClimateTag::Forest, ClimateTag::Hot],
            surface_block: "grass_tropical",
            floor_block: "dirt_tropical",
            foliage_color: [0.10, 0.75, 0.15],
            water_color:   [0.10, 0.60, 0.60],
            ambient_tint:  [0.95, 1.00, 0.85],
        });
        dict.register(BiomeDefinition {
            id: 403,
            name: "Taiga",
            tags: vec![ClimateTag::Forest, ClimateTag::Cold],
            surface_block: "grass_taiga",
            floor_block: "dirt_taiga",
            foliage_color: [0.35, 0.60, 0.30],
            water_color:   [0.30, 0.55, 0.70],
            ambient_tint:  [0.88, 0.93, 1.00],
        });

        // -- Deserts (500–501) -------------------------------------------
        dict.register(BiomeDefinition {
            id: 500,
            name: "Desert",
            tags: vec![ClimateTag::Desert, ClimateTag::Hot],
            surface_block: "sand",
            floor_block: "sandstone",
            foliage_color: [0.85, 0.75, 0.25],
            water_color:   [0.20, 0.50, 0.55],
            ambient_tint:  [1.00, 0.90, 0.72],
        });
        dict.register(BiomeDefinition {
            id: 501,
            name: "Red Desert",
            tags: vec![ClimateTag::Desert, ClimateTag::Hot],
            surface_block: "red_sand",
            floor_block: "red_sandstone",
            foliage_color: [0.80, 0.40, 0.15],
            water_color:   [0.30, 0.45, 0.50],
            ambient_tint:  [1.00, 0.85, 0.72],
        });

        // -- Swamps (600–601) --------------------------------------------
        dict.register(BiomeDefinition {
            id: 600,
            name: "Swamp",
            tags: vec![ClimateTag::Swamp, ClimateTag::Warm],
            surface_block: "grass_swamp",
            floor_block: "clay",
            foliage_color: [0.35, 0.55, 0.20],
            water_color:   [0.15, 0.35, 0.30],
            ambient_tint:  [0.88, 0.95, 0.82],
        });
        dict.register(BiomeDefinition {
            id: 601,
            name: "Mangrove Swamp",
            tags: vec![ClimateTag::Swamp, ClimateTag::Hot],
            surface_block: "grass_mangrove",
            floor_block: "mud",
            foliage_color: [0.25, 0.50, 0.15],
            water_color:   [0.10, 0.30, 0.25],
            ambient_tint:  [0.90, 0.95, 0.80],
        });

        // -- Mountains (700–703) -----------------------------------------
        dict.register(BiomeDefinition {
            id: 700,
            name: "Mountains",
            tags: vec![ClimateTag::Mountain, ClimateTag::Cold],
            surface_block: "stone",
            floor_block: "stone",
            foliage_color: [0.45, 0.50, 0.35],
            water_color:   [0.20, 0.45, 0.70],
            ambient_tint:  [0.90, 0.93, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 701,
            name: "Snowy Mountains",
            tags: vec![ClimateTag::Mountain, ClimateTag::Icy],
            surface_block: "snow",
            floor_block: "stone",
            foliage_color: [0.55, 0.65, 0.55],
            water_color:   [0.45, 0.65, 0.80],
            ambient_tint:  [0.80, 0.87, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 702,
            name: "Volcanic Mountains",
            tags: vec![ClimateTag::Mountain, ClimateTag::Hot],
            surface_block: "basalt",
            floor_block: "basalt",
            foliage_color: [0.30, 0.30, 0.25],
            water_color:   [0.30, 0.30, 0.30],
            ambient_tint:  [1.00, 0.88, 0.75],
        });
        dict.register(BiomeDefinition {
            id: 703,
            name: "Mesa",
            tags: vec![ClimateTag::Mountain, ClimateTag::Hot, ClimateTag::Desert],
            surface_block: "red_sand",
            floor_block: "terracotta",
            foliage_color: [0.70, 0.45, 0.20],
            water_color:   [0.25, 0.45, 0.55],
            ambient_tint:  [1.00, 0.88, 0.72],
        });

        // -- Tundra / Ice (800–801) --------------------------------------
        dict.register(BiomeDefinition {
            id: 800,
            name: "Tundra",
            tags: vec![ClimateTag::Cold, ClimateTag::Plains],
            surface_block: "snow",
            floor_block: "dirt",
            foliage_color: [0.50, 0.55, 0.45],
            water_color:   [0.35, 0.55, 0.70],
            ambient_tint:  [0.85, 0.90, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 801,
            name: "Ice Plains",
            tags: vec![ClimateTag::Icy, ClimateTag::Plains],
            surface_block: "ice",
            floor_block: "dirt",
            foliage_color: [0.60, 0.70, 0.65],
            water_color:   [0.55, 0.70, 0.85],
            ambient_tint:  [0.75, 0.83, 1.00],
        });

        // -- Jungles (900–901) -------------------------------------------
        dict.register(BiomeDefinition {
            id: 900,
            name: "Jungle",
            tags: vec![ClimateTag::Forest, ClimateTag::Hot],
            surface_block: "grass_jungle",
            floor_block: "dirt_jungle",
            foliage_color: [0.10, 0.80, 0.10],
            water_color:   [0.05, 0.45, 0.50],
            ambient_tint:  [0.92, 1.00, 0.80],
        });
        dict.register(BiomeDefinition {
            id: 901,
            name: "Dense Jungle",
            tags: vec![ClimateTag::Forest, ClimateTag::Hot],
            surface_block: "grass_jungle_dense",
            floor_block: "dirt_jungle",
            foliage_color: [0.05, 0.70, 0.10],
            water_color:   [0.05, 0.40, 0.45],
            ambient_tint:  [0.90, 1.00, 0.78],
        });

        // -- Rivers (1000–1001) ------------------------------------------
        dict.register(BiomeDefinition {
            id: 1000,
            name: "River",
            tags: vec![ClimateTag::River, ClimateTag::Warm],
            surface_block: "water",
            floor_block: "sand",
            foliage_color: [0.40, 0.65, 0.30],
            water_color:   [0.15, 0.45, 0.75],
            ambient_tint:  [0.92, 0.96, 1.00],
        });
        dict.register(BiomeDefinition {
            id: 1001,
            name: "Frozen River",
            tags: vec![ClimateTag::River, ClimateTag::Icy],
            surface_block: "ice",
            floor_block: "sand",
            foliage_color: [0.55, 0.65, 0.55],
            water_color:   [0.50, 0.65, 0.80],
            ambient_tint:  [0.80, 0.88, 1.00],
        });

        debug_log!(
            "BiomeDictionary",
            "create_default",
            "registered {} biomes",
            dict.len()
        );
        dict
    }
}

impl Default for BiomeDictionary {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Send + Sync assertions
// ---------------------------------------------------------------------------

const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn _check() {
        assert_send_sync::<BiomeDictionary>();
        assert_send_sync::<BiomeDefinition>();
        assert_send_sync::<ClimateTag>();
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let dict = BiomeDictionary::new();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_register_and_get() {
        let mut dict = BiomeDictionary::new();
        dict.register(BiomeDefinition {
            id: 42,
            name: "Test Biome",
            tags: vec![ClimateTag::Plains, ClimateTag::Warm],
            surface_block: "grass",
            floor_block: "dirt",
            foliage_color: [0.5, 0.8, 0.2],
            water_color:   [0.1, 0.4, 0.7],
            ambient_tint:  [1.0, 1.0, 1.0],
        });

        assert_eq!(dict.len(), 1);
        let def = dict.get(42).unwrap();
        assert_eq!(def.name, "Test Biome");
        assert_eq!(def.tags.len(), 2);
    }

    #[test]
    fn test_find_by_tags_single() {
        let dict = BiomeDictionary::create_default();
        let oceans = dict.find_by_tags(&[ClimateTag::Ocean]);
        assert!(!oceans.is_empty());
        // All returned IDs should be ocean biomes.
        for &id in &oceans {
            assert!(dict.has_tag(id, ClimateTag::Ocean));
        }
    }

    #[test]
    fn test_find_by_tags_intersection() {
        let dict = BiomeDictionary::create_default();
        // Ocean AND Icy → Frozen Ocean (102).
        let frozen_ocean = dict.find_by_tags(&[ClimateTag::Ocean, ClimateTag::Icy]);
        assert!(frozen_ocean.contains(&102));
    }

    #[test]
    fn test_find_by_tags_no_match() {
        let dict = BiomeDictionary::create_default();
        let impossible = dict.find_by_tags(&[ClimateTag::Ocean, ClimateTag::Mountain]);
        assert!(impossible.is_empty());
    }

    #[test]
    fn test_has_tag() {
        let dict = BiomeDictionary::create_default();
        assert!(dict.has_tag(102, ClimateTag::Ocean));
        assert!(dict.has_tag(102, ClimateTag::Icy));
        assert!(!dict.has_tag(102, ClimateTag::Hot));
        assert!(!dict.has_tag(9999, ClimateTag::Ocean)); // unknown ID
    }

    #[test]
    fn test_first_with_tags() {
        let dict = BiomeDictionary::create_default();
        let first_forest = dict.first_with_tags(&[ClimateTag::Forest, ClimateTag::Warm]);
        assert_eq!(first_forest, Some(400)); // Forest biome
    }

    #[test]
    fn test_find_best_matching() {
        let dict = BiomeDictionary::create_default();

        // Query: ocean + cold → should match "Deep Ocean" (Ocean, Cold).
        let best = dict.find_best_matching(&[ClimateTag::Ocean, ClimateTag::Cold]);
        assert_eq!(best, Some(100));
    }

    #[test]
    fn test_create_default_count() {
        let dict = BiomeDictionary::create_default();
        assert_eq!(dict.len(), 30);
    }

    #[test]
    fn test_create_default_all_retrievable() {
        let dict = BiomeDictionary::create_default();
        for id in [100, 101, 102, 103, 104, 200, 201, 202, 300, 301, 302, 303,
                   400, 401, 402, 403, 500, 501, 600, 601, 700, 701, 702, 703,
                   800, 801, 900, 901, 1000, 1001] {
            assert!(
                dict.get(id).is_some(),
                "missing biome id={}",
                id
            );
        }
    }

    #[test]
    fn test_climate_tag_display() {
        assert_eq!(format!("{}", ClimateTag::Ocean), "Ocean");
        assert_eq!(format!("{}", ClimateTag::Icy), "Icy");
        assert_eq!(format!("{}", ClimateTag::Mountain), "Mountain");
    }

    #[test]
    fn test_register_overwrite() {
        let mut dict = BiomeDictionary::new();
        dict.register(BiomeDefinition {
            id: 1,
            name: "First",
            tags: vec![ClimateTag::Warm],
            surface_block: "a",
            floor_block: "b",
            foliage_color: [1.0, 1.0, 1.0],
            water_color:   [1.0, 1.0, 1.0],
            ambient_tint:  [1.0, 1.0, 1.0],
        });
        dict.register(BiomeDefinition {
            id: 1,
            name: "Second",
            tags: vec![ClimateTag::Cold],
            surface_block: "c",
            floor_block: "d",
            foliage_color: [1.0, 1.0, 1.0],
            water_color:   [1.0, 1.0, 1.0],
            ambient_tint:  [1.0, 1.0, 1.0],
        });
        assert_eq!(dict.get(1).unwrap().name, "Second");
        assert_eq!(dict.len(), 1);
    }
}
