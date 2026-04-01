// =============================================================================
// QubePixel — SettingsScreen (egui-based settings)
// =============================================================================

use crate::core::config;
use crate::core::screen::{Screen, ScreenAction};
use crate::{debug_log, flow_debug_log};
use winit::event::WindowEvent;

pub struct SettingsScreen {
    pending_action: ScreenAction,
}

impl SettingsScreen {
    pub fn new() -> Self {
        debug_log!("SettingsScreen", "new", "Creating SettingsScreen (egui)");
        Self {
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for SettingsScreen {
    fn name(&self) -> &str { "Settings" }
    fn load(&mut self) {
        debug_log!("SettingsScreen", "load", "Loading SettingsScreen");
    }
    fn start(&mut self) {
        debug_log!("SettingsScreen", "start", "First-frame init");
    }
    fn update(&mut self, _dt: f64) {
        flow_debug_log!("SettingsScreen", "update", "Update tick");
    }

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
        // Allow Escape to go back (checked via ctx.input)
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            debug_log!("SettingsScreen", "build_ui", "Escape -> Pop");
            self.pending_action = ScreenAction::Pop;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);

                // --- Vertical Below slider ---
                ui.vertical_centered(|ui| {
                    ui.set_width(320.0);

                    ui.label(
                        egui::RichText::new("Chunks Below Camera")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut vb = config::vertical_below() as f32;
                    let slider = egui::Slider::new(&mut vb, 1.0..=16.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Below");
                    if ui.add(slider).changed() {
                        config::set_vertical_below(vb as i32);
                        debug_log!("SettingsScreen", "build_ui",
                            "Vertical below -> {}", vb as i32);
                    }

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "How many chunk layers load below the camera. \
                             Lower = better underground performance."
                        )
                            .size(12.0)
                            .color(egui::Color32::GRAY),
                    );
                });

                ui.add_space(20.0);

                // --- Vertical Above slider ---
                ui.vertical_centered(|ui| {
                    ui.set_width(320.0);

                    ui.label(
                        egui::RichText::new("Chunks Above Camera")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut va = config::vertical_above() as f32;
                    let slider = egui::Slider::new(&mut va, 1.0..=16.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Above");
                    if ui.add(slider).changed() {
                        config::set_vertical_above(va as i32);
                        debug_log!("SettingsScreen", "build_ui",
                            "Vertical above -> {}", va as i32);
                    }

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "How many chunk layers load above the camera. \
                             Lower = fewer sky chunks loaded."
                        )
                            .size(12.0)
                            .color(egui::Color32::GRAY),
                    );
                });

                ui.add_space(40.0);

                // --- Title ---
                ui.label(
                    egui::RichText::new("Settings")
                        .size(36.0)
                        .color(egui::Color32::WHITE),
                );

                ui.add_space(30.0);

                // --- Render Distance slider ---
                ui.vertical_centered(|ui| {
                    ui.set_width(320.0);

                    ui.label(
                        egui::RichText::new("Render Distance")
                            .size(18.0)
                            .color(egui::Color32::LIGHT_GRAY),
                    );
                    ui.add_space(8.0);

                    let mut rd = config::render_distance() as f32;
                    let slider = egui::Slider::new(&mut rd, 1.0..=128.0)
                        .step_by(1.0)
                        .suffix(" chunks")
                        .text("Distance");
                    if ui.add(slider).changed() {
                        config::set_render_distance(rd as i32);
                        debug_log!("SettingsScreen", "build_ui",
                            "Render distance -> {}", rd as i32);
                    }

                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(
                            "Higher values increase world visibility \
                             but may reduce performance."
                        )
                            .size(12.0)
                            .color(egui::Color32::GRAY),
                    );
                });

                ui.add_space(40.0);

                // --- Back button ---
                let back_btn = egui::Button::new(
                    egui::RichText::new("\u{2190}  Back").size(20.0),
                )
                    .min_size(egui::vec2(220.0, 48.0))
                    .fill(egui::Color32::from_rgb(140, 40, 40));

                if ui.add(back_btn).clicked() {
                    debug_log!("SettingsScreen", "build_ui", "Back clicked");
                    self.pending_action = ScreenAction::Pop;
                }
            });
        });
    }

    fn on_event(&mut self, _event: &WindowEvent) {}

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color { r: 0.1, g: 0.1, b: 0.15, a: 1.0 }
    }
}