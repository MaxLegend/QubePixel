pub mod world;
pub mod biome_layer;
pub mod nature;
pub mod biome_registry;
pub mod climate;
pub mod geology;
pub mod geography;
pub mod layers;
pub mod pipeline;
pub mod stratification;

// ---------------------------------------------------------------------------
// Convenience re-exports — biome layer core
// ---------------------------------------------------------------------------

pub use biome_layer::{BiomeLayer, GenContext, GenMode};
pub use biome_layer::{coord_hash, coord_hash_to_f64};

// ---------------------------------------------------------------------------
// Convenience re-exports — geography
// ---------------------------------------------------------------------------

pub use geography::{GeoInfo, GeographyLayer, DEEP_OCEAN_MAX, LOWLAND_MAX, MIDLAND_MAX, OCEAN_MAX, SEA_LEVEL};

// ---------------------------------------------------------------------------
// Convenience re-exports — climate
// ---------------------------------------------------------------------------

pub use climate::{ClimateInfo, ClimateLayer, ClimateType};

// ---------------------------------------------------------------------------
// Convenience re-exports — geology
// ---------------------------------------------------------------------------

pub use geology::{GeologyInfo, GeologyLayer, MacroRegion, MicroRegion};

// ---------------------------------------------------------------------------
// Convenience re-exports — stratification
// ---------------------------------------------------------------------------

pub use stratification::{ColumnStratigraphy, StratificationLayer};

// ---------------------------------------------------------------------------
// Convenience re-exports — biome registry
// ---------------------------------------------------------------------------

pub use biome_registry::{BiomeDefinition, BiomeDictionary, ClimateTag};

// ---------------------------------------------------------------------------
// Convenience re-exports — pipeline
// ---------------------------------------------------------------------------

pub use pipeline::{build_layer_pipeline, build_river_pipeline, BiomePipeline, ColumnData};