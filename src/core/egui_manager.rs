// =============================================================================
// QubePixel — EguiManager (egui ↔ wgpu ↔ winit integration)
// =============================================================================

use crate::debug_log;
use egui_wgpu::Renderer as EguiRenderer;

pub struct EguiManager {
    ctx:                egui::Context,
    winit_state:        egui_winit::State,
    renderer:           EguiRenderer,
    screen_descriptor:  egui_wgpu::ScreenDescriptor,
}

impl EguiManager {
    pub fn new(
        window:         &winit::window::Window,
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        debug_log!("EguiManager", "new", "Initializing egui integration");

        let ctx = egui::Context::default();

        // --- egui_winit state (6 arguments) ---------------------------------
        // Order: egui_ctx, viewport_id, display_target,
        //        native_pixels_per_point, theme, max_texture_side
        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,                                    // &dyn HasDisplayHandle
            Some(window.scale_factor() as f32),        // native_pixels_per_point
            None,                                      // theme (auto-detect)
            Some(4096),                                // max_texture_side
        );

        debug_log!("EguiManager", "new", "egui_winit::State created");

        // --- egui_wgpu renderer (3 arguments: device, format, options) ------
        let renderer = EguiRenderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        debug_log!("EguiManager", "new", "egui_wgpu::Renderer created");

        // --- Screen descriptor ---------------------------------------------
        let size = window.inner_size();
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels:  [size.width, size.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        Self {
            ctx,
            winit_state,
            renderer,
            screen_descriptor,
        }
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }

    // -------------------------------------------------------------------
    // Event handling
    // -------------------------------------------------------------------

    /// Feed a winit WindowEvent to egui.
    /// Returns EventResponse with `consumed` and `repaint` flags.
    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event:  &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.winit_state.on_window_event(window, event)
    }

    /// Feed raw mouse motion delta to egui (from DeviceEvent).
    /// Returns true if egui consumed the event.
    pub fn on_mouse_motion(&mut self, delta: (f64, f64)) -> bool {
        self.winit_state.on_mouse_motion(delta)
    }

    /// Update viewport info before calling take_egui_input.
    pub fn update_viewport_info(&mut self, window: &winit::window::Window) {
        let viewport_info = self.winit_state.egui_input_mut()
            .viewports
            .entry(egui::ViewportId::ROOT)
            .or_default();
        egui_winit::update_viewport_info(
            viewport_info,
            &self.ctx,
            window,
            false,
        );
    }

    /// Extract accumulated egui::RawInput from winit state.
    /// Must be called after update_viewport_info.
    pub fn take_egui_input(&mut self, window: &winit::window::Window) -> egui::RawInput {
        self.winit_state.take_egui_input(window)
    }

    /// Handle platform output (cursor icon, clipboard, IME, etc.).
    pub fn handle_platform_output(
        &mut self,
        window:         &winit::window::Window,
        platform_output: egui::PlatformOutput,
    ) {
        self.winit_state.handle_platform_output(window, platform_output);
    }

    // -------------------------------------------------------------------
    // Frame lifecycle
    // -------------------------------------------------------------------

    /// Upload new/modified textures and tessellate shapes.
    /// Returns Vec<ClippedPrimitive> ready for render_draw.
    pub fn end_frame_and_tessellate(
        &mut self,
        device:      &wgpu::Device,
        queue:       &wgpu::Queue,
        full_output: egui::FullOutput,
    ) -> Vec<egui::ClippedPrimitive> {
        // Upload new / modified textures to GPU
        for (id, image_delta) in full_output.textures_delta.set {
            self.renderer.update_texture(device, queue, id, &image_delta);
        }

        // Free removed textures
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        // Tessellate — 2 arguments (shapes, pixels_per_point)
        self.ctx.tessellate(
            full_output.shapes,
            full_output.pixels_per_point,
        )
    }

    /// Upload buffers and render egui primitives into an existing render pass.
    /// This creates its own render pass (LoadOp::Load) on top of the 3D scene.
    pub fn render_draw(
        &mut self,
        encoder:           &mut wgpu::CommandEncoder,
        view:              &wgpu::TextureView,
        device:            &wgpu::Device,
        queue:             &wgpu::Queue,
        clipped_primitives: &[egui::ClippedPrimitive],
    ) {
        if clipped_primitives.is_empty() {
            return;
        }

        // 1) Upload vertex/index/uniform data (MUST call before render)
        let _callback_cmd_bufs = self.renderer.update_buffers(
            device, queue, encoder, clipped_primitives, &self.screen_descriptor,
        );

        // 2) Create a LoadOp::Load render pass on top of the existing content
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });

        // 3) Render — requires &'static mut RenderPass.
        //    SAFETY: render_pass is dropped at end of this scope, before
        //    encoder is reused. The 'static is an egui-wgpu API requirement.
        let static_pass: &mut wgpu::RenderPass<'static> = unsafe {
            &mut *(&mut render_pass
                as *mut wgpu::RenderPass<'_>
                as *mut wgpu::RenderPass<'static>)
        };

        self.renderer.render(
            static_pass,
            clipped_primitives,
            &self.screen_descriptor,
        );

        // render_pass drops here — render pass ends
    }

    // -------------------------------------------------------------------
    // Resize
    // -------------------------------------------------------------------

    /// Update the internal screen descriptor on window resize or DPI change.
    pub fn update_screen_size(
        &mut self,
        width:        u32,
        height:       u32,
        scale_factor: f32,
    ) {
        self.screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels:  [width, height],
            pixels_per_point: scale_factor,
        };
    }
}