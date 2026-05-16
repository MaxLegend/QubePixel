// =============================================================================
// QubePixel — WorldGenVisualizerScreen
//
// Screen wrapper for the world generation visualizer.
// Accessible from main menu via "World Gen" button.
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::screens::worldgen_visualizer::WorldGenVisualizer;
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;

pub struct WorldGenVisualizerScreen {
    visualizer: WorldGenVisualizer,
    pending_action: ScreenAction,
}

impl WorldGenVisualizerScreen {
    pub fn new() -> Self {
        debug_log!("WorldGenVisualizerScreen", "new", "Creating WorldGenVisualizerScreen");
        Self {
            visualizer: WorldGenVisualizer::new(),
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for WorldGenVisualizerScreen {
    fn name(&self) -> &str { "WorldGenVisualizer" }
    fn load(&mut self) {
        debug_log!("WorldGenVisualizerScreen", "load", "Loading WorldGenVisualizerScreen");
    }
    fn start(&mut self) {
        debug_log!("WorldGenVisualizerScreen", "start", "First-frame init");
        self.visualizer.visible = true;
    }
    fn update(&mut self, _dt: f64) {
        flow_debug_log!("WorldGenVisualizerScreen", "update", "Update tick");
    }

    /// No 3D rendering — egui draws everything via build_ui().
    fn render(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        _view:    &wgpu::TextureView,
        _device:  &wgpu::Device,
        _queue:   &wgpu::Queue,
        _format:  wgpu::TextureFormat,
        _width:   u32,
        _height:  u32,
    ) {}

    fn post_process(&mut self, _dt: f64) {}

    fn build_ui(&mut self, ctx: &egui::Context) {
        // Allow Escape to go back
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            debug_log!("WorldGenVisualizerScreen", "build_ui", "Escape -> Pop");
            self.pending_action = ScreenAction::Pop;
        }

        // Draw the visualizer (which manages its own window)
        self.visualizer.draw_egui(ctx);
    }

    fn on_event(&mut self, _event: &WindowEvent) {}

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }
    }
}
