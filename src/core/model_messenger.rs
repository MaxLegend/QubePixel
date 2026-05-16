// =============================================================================
// QubePixel — ModelMessenger  (async model loading + registry + atlas packing)
// =============================================================================
//
// Centralised system for loading, caching, and distributing block models.
//
// Architecture:
//   ModelMessengerRequest  — messages sent to request model loading
//   ModelMessengerResponse — responses with loaded model data + GPU texture
//   ModelMessenger         — the handler that processes requests
//
// Multi-texture support:
//   Each block model can reference multiple textures via named variables.
//   The messenger packs all referenced textures into a single per-model atlas
//   and remaps UVs accordingly.  The resulting atlas is uploaded as a GPU
//   texture and passed back in the response.
//
// Usage:
//   1. Create a ModelMessenger.
//   2. Call `register_block_models()` to scan BlockRegistry for models.
//   3. Call `process_pending()` each frame to load queued models.
//   4. Call `drain_responses()` to get loaded models + GPU textures.
//   5. Register models with EntityRenderer using the provided GPU texture.
// =============================================================================

use std::collections::HashMap;
use std::path::PathBuf;

use crate::core::block_model::{BlockModelData, BlockModelParser, ModelFormat};
use crate::core::gameobjects::block::BlockRegistry;
use crate::debug_log;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// A request to load a model from a file path.
#[derive(Debug, Clone)]
pub struct ModelMessengerRequest {
    /// Unique name to register the model under (e.g. "block/spotlight").
    pub model_name: String,
    /// Filesystem path to the model JSON file.
    pub file_path:  String,
    /// Named texture paths: variable name → image file path.
    /// For single-texture models, use {"default": "path/to/texture.png"}.
    /// For multi-texture Java models: {"0": "path/a.png", "1": "path/b.png"}.
    pub textures:   HashMap<String, String>,
    /// Expected format hint. If None, auto-detect.
    pub format_hint: Option<ModelFormat>,
    /// Associated block ID (if loaded from block registry).
    pub block_id:    Option<u8>,
}

/// Result of a model loading operation.
#[derive(Debug)]
pub struct ModelMessengerResponse {
    /// The model name from the request.
    pub model_name: String,
    /// The loaded model data (None if loading failed).
    pub model:      Option<BlockModelData>,
    /// Packed per-model atlas texture (GPU), if model loaded successfully.
    pub atlas_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Error message if loading failed.
    pub error:      Option<String>,
    /// Associated block ID.
    pub block_id:    Option<u8>,
    /// Per-cube shadow AABBs in block-canonical [0,1]^3 space.
    pub model_shadow_cubes: Option<Vec<[f32; 6]>>,
}

// ---------------------------------------------------------------------------
// ModelMessenger — the handler
// ---------------------------------------------------------------------------

/// Centralised model loading and caching system.
pub struct ModelMessenger {
    /// Successfully loaded models, keyed by model_name.
    models: HashMap<String, ModelEntry>,
    /// Pending load requests.
    pending: Vec<ModelMessengerRequest>,
    /// Completed responses not yet consumed.
    responses: Vec<ModelMessengerResponse>,
}

/// A loaded model entry with its atlas texture path info.
#[derive(Debug)]
struct ModelEntry {
    model: BlockModelData,
}

impl ModelMessenger {
    /// Create a new empty ModelMessenger.
    pub fn new() -> Self {
        Self {
            models:    HashMap::new(),
            pending:   Vec::new(),
            responses: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Submit requests
    // -----------------------------------------------------------------------

    /// Submit a model loading request.
    pub fn submit(&mut self, request: ModelMessengerRequest) {
        if self.models.contains_key(&request.model_name) {
            return;
        }
        debug_log!(
            "ModelMessenger", "submit",
            "Queued model '{}' from '{}'", request.model_name, request.file_path
        );
        self.pending.push(request);
    }

    // -----------------------------------------------------------------------
    // Process pending requests
    // -----------------------------------------------------------------------

    /// Process all pending load requests synchronously.
    /// Loads model JSON + all referenced textures, packs into per-model atlas.
    /// Returns the number of models successfully loaded this call.
    pub fn process_pending(&mut self) -> usize {
        let requests = std::mem::take(&mut self.pending);
        let mut loaded_count = 0;

        for req in requests {
            if self.models.contains_key(&req.model_name) {
                continue;
            }

            match Self::load_from_file(&req) {
                Ok(mut model) => {
                    debug_log!(
                        "ModelMessenger", "process_pending",
                        "Loaded model '{}' ({} verts, {} indices, {} textures) from '{}'",
                        req.model_name,
                        model.vertices.len(),
                        model.indices.len(),
                        req.textures.len(),
                        req.file_path
                    );

                    // Resolve texture variable names → actual file paths and store
                    // them back into texture_variables so build_model_atlas can read them.
                    // Without this, texture_variables would contain model-internal names
                    // like "block/stone_side" instead of real filesystem paths.
                    if !req.textures.is_empty() {
                        let resolved = model.resolve_texture_paths(&req.textures);
                        debug_log!(
                            "ModelMessenger", "process_pending",
                            "Model '{}' resolved {} textures: {:?}",
                            req.model_name, resolved.len(), resolved
                        );
                        model.texture_variables = resolved;
                    }

                    self.responses.push(ModelMessengerResponse {
                        model_name:    req.model_name.clone(),
                        model:         Some(model.clone()),
                        atlas_texture: None, // GPU texture created later in process_pending_gpu()
                        error:         None,
                        block_id:      req.block_id,
                        model_shadow_cubes: Some(model.shadow_cubes.clone()),
                    });
                    self.models.insert(req.model_name, ModelEntry {
                        model,
                    });
                    loaded_count += 1;
                }
                Err(e) => {
                    debug_log!(
                        "ModelMessenger", "process_pending",
                        "Failed to load model '{}' from '{}': {}",
                        req.model_name, req.file_path, e
                    );
                    self.responses.push(ModelMessengerResponse {
                        model_name:    req.model_name,
                        model:         None,
                        atlas_texture: None,
                        error:         Some(e),
                        block_id:      req.block_id,
                        model_shadow_cubes: None,
                    });
                }
            }
        }

        loaded_count
    }

    /// Process pending GPU-side work: pack textures into per-model atlases.
    /// Must be called with valid device/queue after `process_pending()`.
    pub fn process_pending_gpu(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
    ) {
        for resp in &mut self.responses {
            if let Some(ref mut model) = resp.model {
                // Collect texture file paths from the request
                // We need to get them from the model's referenced textures
                // For now, we'll create the atlas from the model's texture info
                if let Some(atlas) = Self::build_model_atlas(device, queue, model) {
                    resp.atlas_texture = Some(atlas);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Query loaded models
    // -----------------------------------------------------------------------

    /// Get a loaded model by name.
    pub fn get(&self, model_name: &str) -> Option<&BlockModelData> {
        self.models.get(model_name).map(|e| &e.model)
    }

    /// Check if a model is loaded.
    pub fn is_loaded(&self, model_name: &str) -> bool {
        self.models.contains_key(model_name)
    }

    /// Number of loaded models.
    pub fn loaded_count(&self) -> usize {
        self.models.len()
    }

    /// Number of pending (not yet loaded) requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // -----------------------------------------------------------------------
    // Consume responses
    // -----------------------------------------------------------------------

    /// Drain all completed responses since the last call.
    pub fn drain_responses(&mut self) -> Vec<ModelMessengerResponse> {
        std::mem::take(&mut self.responses)
    }

    // -----------------------------------------------------------------------
    // Bulk loading from BlockRegistry
    // -----------------------------------------------------------------------

    /// Scan a BlockRegistry for block definitions that reference model files
    /// and submit load requests for all of them.
    pub fn register_block_models(
        &mut self,
        registry:   &BlockRegistry,
        assets_dir: &str,
    ) {
        let dir = PathBuf::from(assets_dir);
        // Block models live in assets/models/blocks/
        let model_dir = dir.join("models").join("blocks");
        // Block model textures live in assets/textures/modelblocks/
        let texture_dir = dir.join("textures").join("modelblocks");

        for block_id in 1u8..=255u8 {
            let Some(def) = registry.get(block_id) else { continue };
            let Some(ref model_path) = def.model else { continue };

            let full_path = if PathBuf::from(model_path).is_absolute() {
                model_path.clone()
            } else {
                model_dir.join(model_path)
                    .to_str()
                    .unwrap_or(model_path)
                    .to_string()
            };

            let model_name = format!("block/{}", def.id);

            // Resolve texture paths from model_textures map
            // Textures for block models are in assets/textures/modelblocks/
            let textures: HashMap<String, String> = def.model_textures.iter()
                .map(|(k, v)| {
                    let path = if PathBuf::from(v).is_absolute() {
                        v.clone()
                    } else {
                        texture_dir.join(v)
                            .to_str()
                            .unwrap_or(v)
                            .to_string()
                    };
                    (k.clone(), path)
                })
                .collect();

            self.submit(ModelMessengerRequest {
                model_name,
                file_path: full_path,
                textures,
                format_hint: None,
                block_id: Some(block_id),
            });
        }

        debug_log!(
            "ModelMessenger", "register_block_models",
            "Submitted {} block model requests",
            self.pending.len()
        );
    }

    // -----------------------------------------------------------------------
    // Internal: file loading
    // -----------------------------------------------------------------------

    fn load_from_file(req: &ModelMessengerRequest) -> Result<BlockModelData, String> {
        let json = std::fs::read_to_string(&req.file_path)
            .map_err(|e| format!("Failed to read '{}': {}", req.file_path, e))?;

        match req.format_hint {
            Some(ModelFormat::BedrockGeometry) => BlockModelParser::parse_bedrock(&json),
            Some(ModelFormat::JavaEdition)     => BlockModelParser::parse_java(&json),
            None                               => BlockModelParser::parse(&json),
        }
    }

    // -----------------------------------------------------------------------
    // Internal: per-model atlas building
    // -----------------------------------------------------------------------

    /// Build a per-model texture atlas from all referenced textures.
    /// Packs textures horizontally into a single strip, then remaps UVs.
    fn build_model_atlas(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        model:  &mut BlockModelData,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        // Collect unique texture variable names from the model
        let tex_names = model.referenced_textures();
        if tex_names.is_empty() {
            return None;
        }

        // Load each texture image
        let mut images: Vec<(String, image::RgbaImage)> = Vec::new();
        for name in &tex_names {
            // Try to find the texture path from the model's texture_variables
            let path = model.texture_variables.get(name)
                .cloned()
                .unwrap_or_default();

            if path.is_empty() {
                continue;
            }

            match std::fs::read(&path) {
                Ok(bytes) => {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let rgba = img.to_rgba8();
                        debug_log!(
                            "ModelMessenger", "build_model_atlas",
                            "Loaded texture '{}' ({}x{}) from '{}'",
                            name, rgba.width(), rgba.height(), path
                        );
                        images.push((name.clone(), rgba));
                    } else {
                        debug_log!(
                            "ModelMessenger", "build_model_atlas",
                            "Failed to decode image '{}'", path
                        );
                    }
                }
                Err(e) => {
                    debug_log!(
                        "ModelMessenger", "build_model_atlas",
                        "Failed to read texture '{}': {}", path, e
                    );
                }
            }
        }

        if images.is_empty() {
            // No textures loaded — create a 1x1 white fallback
            let data = [255u8; 4];
            return Some(Self::upload_rgba(device, queue, &data, 1, 1));
        }

        // If only one texture, no packing needed — just upload it directly
        if images.len() == 1 && !model.has_multi_texture() {
            let (_, img) = &images[0];
            let (w, h) = img.dimensions();
            return Some(Self::upload_rgba(device, queue, img.as_raw(), w, h));
        }

        // Multi-texture: pack horizontally into a strip atlas
        let max_h: u32 = images.iter().map(|(_, img)| img.height()).max().unwrap_or(1);
        let total_w: u32 = images.iter().map(|(_, img)| img.width()).sum();

        // Create atlas image
        let mut atlas_data = vec![0u8; (total_w * max_h * 4) as usize];
        let mut atlas_layout: HashMap<String, (u32, u32, u32, u32)> = HashMap::new();
        let mut x_offset: u32 = 0;

        for (name, img) in &images {
            let (w, h) = img.dimensions();
            atlas_layout.insert(name.clone(), (x_offset, 0, w, h));

            // Copy image data into atlas
            for y in 0..h {
                for x in 0..w {
                    let src_idx = ((y * w + x) * 4) as usize;
                    let dst_idx = (((y * total_w) + x_offset + x) * 4) as usize;
                    if dst_idx + 4 <= atlas_data.len() && src_idx + 4 <= img.as_raw().len() {
                        atlas_data[dst_idx..dst_idx + 4]
                            .copy_from_slice(&img.as_raw()[src_idx..src_idx + 4]);
                    }
                }
            }
            x_offset += w;
        }

        debug_log!(
            "ModelMessenger", "build_model_atlas",
            "Built atlas {}x{} for model '{}' with {} textures",
            total_w, max_h, model.identifier, images.len()
        );

        // Remap UVs to point into the atlas
        model.remap_uvs_for_atlas(&atlas_layout, total_w, max_h);

        Some(Self::upload_rgba(device, queue, &atlas_data, total_w, max_h))
    }

    fn upload_rgba(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        data:   &[u8],
        w: u32, h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Block Model Atlas"),
            size:            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &tex,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let view = tex.create_view(&Default::default());
        (tex, view)
    }
}

impl Default for ModelMessenger {
    fn default() -> Self {
        Self::new()
    }
}
