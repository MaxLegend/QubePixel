// =============================================================================
// QubePixel — TextureAtlas  (loads PNGs, stitches into atlas, lazy GPU upload)
// =============================================================================

use std::collections::HashMap;
use std::path::PathBuf;
use image::{imageops, Rgba, RgbaImage};  // removed unused: DynamicImage, ImageFormat
use crate::debug_log;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const TILE_SIZE:    u32 = 16;
pub const ATLAS_COLS:   u32 = 16;
pub const ATLAS_ROWS:   u32 = 16;
pub const ATLAS_WIDTH:  u32 = TILE_SIZE * ATLAS_COLS; // 256
pub const ATLAS_HEIGHT: u32 = TILE_SIZE * ATLAS_ROWS; // 256

const TEXTURES_DIR: &str = "assets/textures";

// ---------------------------------------------------------------------------
// TextureAtlasLayout
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct TextureAtlasLayout {
    uv_map: HashMap<String, (f32, f32, f32, f32)>,
}

impl TextureAtlasLayout {
    pub fn uv_for(&self, name: &str) -> (f32, f32, f32, f32) {
        self.uv_map.get(name).copied().unwrap_or_else(|| Self::white_uv())
    }

    fn white_uv() -> (f32, f32, f32, f32) {
        let u1 = 1.0 / ATLAS_COLS as f32;
        let v1 = 1.0 / ATLAS_ROWS as f32;
        (0.0, 0.0, u1, v1)
    }
}

impl Default for TextureAtlasLayout {
    fn default() -> Self { Self { uv_map: HashMap::new() } }
}

// ---------------------------------------------------------------------------
// TextureAtlas
// ---------------------------------------------------------------------------
pub struct TextureAtlas {
    image:       RgbaImage,
    layout:      TextureAtlasLayout,
    gpu_texture: Option<wgpu::Texture>,
    gpu_view:    Option<wgpu::TextureView>,
}

impl TextureAtlas {
    pub fn load() -> Self { Self::load_from_dir(TEXTURES_DIR) }

    pub fn load_from_dir(dir: &str) -> Self {
        let path = PathBuf::from(dir);
        if path.is_dir() {
            debug_log!("TextureAtlas", "load", "Loading textures from {:?}", path);
            match Self::build_from_dir(&path) {
                Ok(atlas) => return atlas,
                Err(e)    => debug_log!(
                    "TextureAtlas", "load",
                    "Directory load failed: {}, using embedded fallback", e
                ),
            }
        } else {
            debug_log!(
                "TextureAtlas", "load",
                "Directory {:?} not found, using embedded fallback", path
            );
        }
        Self::build_embedded()
    }

    pub fn layout(&self) -> &TextureAtlasLayout { &self.layout }

    pub fn ensure_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu_texture.is_some() { return; }

        debug_log!(
            "TextureAtlas", "ensure_gpu",
            "Uploading atlas {}x{} to GPU", ATLAS_WIDTH, ATLAS_HEIGHT
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("TextureAtlas"),
            size:            wgpu::Extent3d { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        // wgpu 22: ImageCopyTexture → TexelCopyTextureInfo
        //          ImageDataLayout  → TexelCopyBufferLayout
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture:   &texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &self.image,
            wgpu::TexelCopyBufferLayout {
                offset:         0,
                bytes_per_row:  Some(ATLAS_WIDTH * 4),
                rows_per_image: Some(ATLAS_HEIGHT),
            },
            wgpu::Extent3d { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, depth_or_array_layers: 1 },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.gpu_texture = Some(texture);
        self.gpu_view    = Some(view);

        debug_log!("TextureAtlas", "ensure_gpu", "Atlas uploaded to GPU");
    }

    pub fn texture_view(&self) -> Option<&wgpu::TextureView> { self.gpu_view.as_ref() }

    // -----------------------------------------------------------------------
    // Directory loader
    // -----------------------------------------------------------------------

    fn build_from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let mut atlas_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut uv_map    = HashMap::new();

        fill_tile_solid(&mut atlas_img, 0, 0, 255, 255, 255);

        let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(dir)
            .map_err(|e| format!("Cannot read dir: {}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "png"))
            .collect();

        entries.sort_by_key(|e| e.file_name());

        let mut next_slot: u32 = 1;

        for entry in &entries {
            let path = entry.path();
            let stem = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            let dyn_img = image::open(&path)
                .map_err(|e| format!("Failed to load {:?}: {}", path, e))?;
            let rgba = dyn_img.to_rgba8();

            let tile = if rgba.width() != TILE_SIZE || rgba.height() != TILE_SIZE {
                imageops::resize(&rgba, TILE_SIZE, TILE_SIZE, imageops::FilterType::Nearest)
            } else { rgba };

            let col = next_slot % ATLAS_COLS;
            let row = next_slot / ATLAS_COLS;
            if row >= ATLAS_ROWS { return Err("Atlas full — too many textures".into()); }
            next_slot += 1;

            imageops::overlay(&mut atlas_img, &tile, (col * TILE_SIZE) as i64, (row * TILE_SIZE) as i64);

            let u0 = col as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;

            debug_log!(
                "TextureAtlas", "build_from_dir",
                "Loaded '{}' => slot ({},{})", stem, col, row
            );
            uv_map.insert(stem, (u0, v0, u1, v1));
        }

        debug_log!(
            "TextureAtlas", "build_from_dir",
            "Total textures loaded: {}", uv_map.len()
        );

        Ok(Self { image: atlas_img, layout: TextureAtlasLayout { uv_map }, gpu_texture: None, gpu_view: None })
    }

    // -----------------------------------------------------------------------
    // Embedded fallback
    // -----------------------------------------------------------------------

    fn build_embedded() -> Self {
        debug_log!("TextureAtlas", "build_embedded", "Generating procedural embedded textures");

        let mut atlas_img = RgbaImage::new(ATLAS_WIDTH, ATLAS_HEIGHT);
        let mut uv_map    = HashMap::new();

        fill_tile_solid(&mut atlas_img, 0, 0, 255, 255, 255);

        let textures: [(&str, u8, u8, u8); 7] = [
            ("grass_top",    56, 133,  36),
            ("grass_side",   77, 102,  38),
            ("grass_bottom", 102, 66,  26),
            ("dirt",         102, 66,  26),
            ("stone_top",    128, 128, 128),
            ("stone_side",   122, 122, 122),
            ("stone_bottom", 128, 128, 128),
        ];

        for (i, &(name, r, g, b)) in textures.iter().enumerate() {
            let slot = (i + 1) as u32;
            let col  = slot % ATLAS_COLS;
            let row  = slot / ATLAS_COLS;

            fill_tile_noisy(&mut atlas_img, col, row, r, g, b, 20);

            let u0 = col as f32 / ATLAS_COLS as f32;
            let v0 = row as f32 / ATLAS_ROWS as f32;
            let u1 = (col + 1) as f32 / ATLAS_COLS as f32;
            let v1 = (row + 1) as f32 / ATLAS_ROWS as f32;
            uv_map.insert(name.to_string(), (u0, v0, u1, v1));

            debug_log!(
                "TextureAtlas", "build_embedded",
                "Generated '{}' at slot ({},{})", name, col, row
            );
        }

        debug_log!("TextureAtlas", "build_embedded", "Total procedural textures: {}", uv_map.len());

        Self { image: atlas_img, layout: TextureAtlasLayout { uv_map }, gpu_texture: None, gpu_view: None }
    }
}

// ---------------------------------------------------------------------------
// Tile-fill helpers
// ---------------------------------------------------------------------------

fn fill_tile_solid(atlas: &mut RgbaImage, col: u32, row: u32, r: u8, g: u8, b: u8) {
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    for y in y0..y0+TILE_SIZE { for x in x0..x0+TILE_SIZE {
        atlas.put_pixel(x, y, Rgba([r, g, b, 255]));
    }}
}

fn fill_tile_noisy(atlas: &mut RgbaImage, col: u32, row: u32, r: u8, g: u8, b: u8, noise_strength: u8) {
    let x0 = col * TILE_SIZE;
    let y0 = row * TILE_SIZE;
    for y in y0..y0+TILE_SIZE { for x in x0..x0+TILE_SIZE {
        let h         = simple_hash(x, y);
        let variation = (h as i16 - 128) * noise_strength as i16 / 128;
        let nr = (r as i16 + variation).clamp(0, 255) as u8;
        let ng = (g as i16 + variation).clamp(0, 255) as u8;
        let nb = (b as i16 + variation).clamp(0, 255) as u8;
        atlas.put_pixel(x, y, Rgba([nr, ng, nb, 255]));
    }}
}

fn simple_hash(x: u32, y: u32) -> u8 {
    let h = x.wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(1013904223);
    ((h >> 16) & 0xFF) as u8
}