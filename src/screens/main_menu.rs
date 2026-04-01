// =============================================================================
// QubePixel — MainMenuScreen (egui-based main menu)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;
use crate::screens::game_screen::GameScreen;
use crate::screens::settings::SettingsScreen;

pub struct MainMenuScreen {
    pending_action: ScreenAction,
}

impl MainMenuScreen {
    pub fn new() -> Self {
        debug_log!("MainMenuScreen", "new", "Creating MainMenuScreen (egui)");
        Self {
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for MainMenuScreen {
    fn name(&self) -> &str { "MainMenu" }
    fn load(&mut self) {
        debug_log!("MainMenuScreen", "load", "Loading MainMenuScreen");
    }
    fn start(&mut self) {
        debug_log!("MainMenuScreen", "start", "First-frame init");
    }
    fn update(&mut self, _dt: f64) {
        flow_debug_log!("MainMenuScreen", "update", "Update tick");
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
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);

                // --- Title ---
                ui.label(
                    egui::RichText::new("QubePixel")
                        .size(48.0)
                        .color(egui::Color32::WHITE),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("A Minecraft-inspired voxel engine")
                        .size(14.0)
                        .color(egui::Color32::GRAY),
                );

                ui.add_space(40.0);

                // --- Play button ---
                let play_btn = egui::Button::new(
                    egui::RichText::new("\u{25B6}  Play").size(22.0),
                )
                    .min_size(egui::vec2(220.0, 50.0))
                    .fill(egui::Color32::from_rgb(40, 160, 60));

                if ui.add(play_btn).clicked() {
                    debug_log!("MainMenuScreen", "build_ui", "Play clicked");
                    self.pending_action =
                        ScreenAction::Switch(Box::new(GameScreen::new()));
                }

                ui.add_space(12.0);

                // --- Settings button ---
                let settings_btn = egui::Button::new(
                    egui::RichText::new("\u{2699}  Settings").size(22.0),
                )
                    .min_size(egui::vec2(220.0, 50.0))
                    .fill(egui::Color32::from_rgb(60, 60, 140));

                if ui.add(settings_btn).clicked() {
                    debug_log!("MainMenuScreen", "build_ui", "Settings clicked");
                    self.pending_action =
                        ScreenAction::Push(Box::new(SettingsScreen::new()));
                }

                ui.add_space(24.0);

                ui.label(
                    egui::RichText::new("v0.0.1")
                        .size(12.0)
                        .color(egui::Color32::DARK_GRAY),
                );
            });
        });
    }

    fn on_event(&mut self, _event: &WindowEvent) {
        // All UI is egui-driven — no manual event handling needed.
    }

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.08, g: 0.08, b: 0.18, a: 1.0 }
    }
}