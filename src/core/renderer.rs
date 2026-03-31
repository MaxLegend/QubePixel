// =============================================================================
// Minecraft Clone — Renderer (wgpu surface, device, queue, clear pass)
// =============================================================================

use crate::debug_log;
use std::marker::PhantomData;
use winit::window::Window;

/// Manages the wgpu rendering context: surface, device, queue, and surface config.
///
/// Provides a simple `render()` that clears the screen with a solid color.
/// Accessors (`device()`, `queue()`, `surface_format()`) allow screens to create
/// their own pipelines and buffers in future tasks.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pending_size: Option<winit::dpi::PhysicalSize<u32>>,
    /// Prevents the struct from being Send/Sync across threads without
    /// ensuring the window is still alive. Not functionally used.
    _window_ref: PhantomData<&'static Window>,
}

impl Renderer {
    /// Initialise wgpu: instance, surface, adapter, device, queue.
    ///
    /// # Safety
    /// The returned `Renderer` must not outlive the window it was created from.
    /// In practice this is guaranteed because `App` owns both the window and the renderer.
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        debug_log!(
            "Renderer",
            "new",
            "Initialising wgpu renderer: {}x{}",
            size.width,
            size.height
        );

        // ── Instance ──────────────────────────────────────────────────────
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // ── Surface ──────────────────────────────────────────────────────
        // create_surface returns Surface<'window> tied to the window's lifetime.
        // We extend it to 'static because Renderer and Window are owned by the
        // same App struct, so the window always outlives the surface.
        let surface = instance.create_surface(window).expect("Failed to create wgpu surface");
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        debug_log!("Renderer", "new", "Surface created");

        // ── Adapter ───────────────────────────────────────────────────────
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        debug_log!(
            "Renderer",
            "new",
            "Adapter selected: {:?}",
            adapter.get_info()
        );

        // ── Device + Queue ────────────────────────────────────────────────
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .expect("Failed to create GPU device");

        debug_log!("Renderer", "new", "Device and queue created");

        // ── Surface configuration ─────────────────────────────────────────
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("Failed to get default surface config");

        config.desired_maximum_frame_latency = 2;

        surface.configure(&device, &config);

        debug_log!(
            "Renderer",
            "new",
            "Surface configured: {}x{}, format={:?}, present_mode={:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );

        Self {
            surface,
            device,
            queue,
            config,
            size,
            _window_ref: PhantomData,
            pending_size: None,
        }
    }

    // -----------------------------------------------------------------------
    // Resize
    // -----------------------------------------------------------------------

    /// Reconfigure the surface for a new window size.
    /// Ignores zero-size events (minimised window).
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.pending_size = Some(new_size);
        }
    }
    // pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
    //     if new_size.width == 0 || new_size.height == 0 {
    //         return;
    //     }
    //     self.size = new_size;
    //     self.config.width = new_size.width;
    //     self.config.height = new_size.height;
    //     self.surface.configure(&self.device, &self.config);
    //     debug_log!(
    //         "Renderer",
    //         "resize",
    //         "Surface resized to {}x{}",
    //         new_size.width,
    //         new_size.height
    //     );
    // }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    /// Acquire a frame, clear it with a solid colour, and present.

    pub fn render<F>(
        &mut self,
        clear_color: wgpu::Color,
        draw_fn: F,
    ) -> Result<(), wgpu::SurfaceError>
    where
        F: FnOnce(
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            &wgpu::Device,
            &wgpu::Queue,
            wgpu::TextureFormat,
        ),
    {
        // Безопасное обновление размера перед получением текстуры
        if let Some(size) = self.pending_size.take() {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),

                        store: wgpu::StoreOp::Store,
                    },
                })],

                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

        }
        // 2) Delegate screen-specific drawing
        draw_fn(
            &mut encoder,
            &view,
            &self.device,
            &self.queue,
            self.config.format,
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Accessors (for screens to use in future tasks)
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }


    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }
}