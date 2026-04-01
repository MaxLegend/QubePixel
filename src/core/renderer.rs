// =============================================================================
// QubePixel — Renderer (wgpu surface, device, queue, clear pass)
// =============================================================================

use crate::debug_log;
use std::marker::PhantomData;
use winit::window::Window;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    pending_size: Option<winit::dpi::PhysicalSize<u32>>,
    _window_ref: PhantomData<&'static Window>,
}

impl Renderer {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();
        debug_log!(
            "Renderer", "new",
            "Initialising wgpu renderer: {}x{}", size.width, size.height
        );

        // ── Instance ──────────────────────────────────────────────────────

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags:    wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        // ── Surface ───────────────────────────────────────────────────────
        let surface = instance.create_surface(window).expect("Failed to create wgpu surface");
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(surface) };

        debug_log!("Renderer", "new", "Surface created");

        // ── Adapter ───────────────────────────────────────────────────────
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable GPU adapter");

        debug_log!("Renderer", "new", "Adapter selected: {:?}", adapter.get_info());

        // ── Device + Queue ────────────────────────────────────────────────
        // wgpu 22: request_device takes ONE argument (no trace path)
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label:             Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits:   wgpu::Limits::default(),
                    // wgpu 22: new memory_hints field
                    experimental_features: Default::default(),
                    memory_hints:      Default::default(),
                    trace: Default::default(),
                },
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
            "Renderer", "new",
            "Surface configured: {}x{}, format={:?}, present_mode={:?}",
            config.width, config.height, config.format, config.present_mode
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

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size        = new_size;
            self.pending_size = Some(new_size);
        }
    }
    /// Apply any pending surface resize. Call before acquiring a frame.
    pub fn process_pending_resize(&mut self) {
        if let Some(size) = self.pending_size.take() {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            debug_log!(
                "Renderer", "process_pending_resize",
                "Surface resized to {}x{}", size.width, size.height
            );
        }
    }
    /// Get the next frame texture from the swapchain.
    pub fn surface_get_current_texture(
        &self,
    ) {
        self.surface.get_current_texture();
    }
    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    pub fn render<F>(
        &mut self,
        clear_color: wgpu::Color,
        draw_fn: F,
    )
    where
        F: FnOnce(
            &mut wgpu::CommandEncoder,
            &wgpu::TextureView,
            &wgpu::Device,
            &wgpu::Queue,
            wgpu::TextureFormat,
        ),
    {
        if let Some(size) = self.pending_size.take() {
            self.config.width  = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.resize(self.size);
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let view   = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view:           &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes:         None,
                occlusion_query_set:      None,
                multiview_mask:           None,
            });
        }

        draw_fn(&mut encoder, &view, &self.device, &self.queue, self.config.format);

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    pub fn device(&self) -> &wgpu::Device { &self.device }

    #[allow(dead_code)]
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    pub fn size(&self) -> winit::dpi::PhysicalSize<u32> { self.size }

    #[allow(dead_code)]
    pub fn surface_format(&self) -> wgpu::TextureFormat { self.config.format }

    #[allow(dead_code)]
    pub fn config(&self) -> &wgpu::SurfaceConfiguration { &self.config }
}