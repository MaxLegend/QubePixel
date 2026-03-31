// =============================================================================
// QubePixel — MainMenuScreen (Play / Settings buttons)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::debug_log;
use crate::flow_debug_log;
use crate::screens::bitmap_font::BitmapFont;
use crate::screens::button::Button;
use crate::screens::game_screen::GameScreen;
use crate::screens::settings::SettingsScreen;
use crate::screens::ui_renderer::UiRenderer;
use winit::event::{ElementState, WindowEvent};

/// The main menu — the first screen the player sees.
///
/// Displays a title, two buttons (Play / Settings), and text labels
/// centered in the window.
pub struct MainMenuScreen {
    ui: Option<UiRenderer>,
    font: Option<BitmapFont>,
    buttons: Vec<Button>,
    cursor_x: f64,
    cursor_y: f64,
    #[allow(dead_code)]
    mouse_down: bool,
    pending_action: ScreenAction,
}

impl MainMenuScreen {
    pub fn new() -> Self {
        debug_log!("MainMenuScreen", "new", "Creating MainMenuScreen");

        // Buttons are defined with zero position — the actual position
        // is computed in render() for centering within the window.
        let play_button = Button::new(
            0.0, 0.0, 200.0, 50.0,
            [0.2, 0.7, 0.2, 1.0],
            [0.3, 0.9, 0.3, 1.0],
            "Play",
        );
        let settings_button = Button::new(
            0.0, 0.0, 200.0, 50.0,
            [0.3, 0.3, 0.7, 1.0],
            [0.4, 0.4, 0.9, 1.0],
            "Settings",
        );

        let buttons = vec![play_button, settings_button];

        debug_log!(
            "MainMenuScreen",
            "new",
            "MainMenuScreen created with {} buttons",
            buttons.len()
        );

        Self {
            ui: None,
            font: None,
            buttons,
            cursor_x: 0.0,
            cursor_y: 0.0,
            mouse_down: false,
            pending_action: ScreenAction::None,
        }
    }
}

impl Screen for MainMenuScreen {
    fn name(&self) -> &str {
        "MainMenu"
    }

    fn load(&mut self) {
        debug_log!("MainMenuScreen", "load", "Loading MainMenuScreen resources");
    }

    fn start(&mut self) {
        debug_log!("MainMenuScreen", "start", "MainMenuScreen first-frame init");
    }

    fn update(&mut self, _dt: f64) {
        flow_debug_log!("MainMenuScreen", "update", "Update tick");
    }

    fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) {
        // Lazy-init renderers on the first render call.
        if self.ui.is_none() {
            self.ui = Some(UiRenderer::new(device, format));
            debug_log!("MainMenuScreen", "render", "UiRenderer initialised lazily");
        }
        if self.font.is_none() {
            self.font = Some(BitmapFont::new(device, format));
            debug_log!("MainMenuScreen", "render", "BitmapFont initialised lazily");
        }
        let ui = self.ui.as_ref().unwrap();
        let font = self.font.as_ref().unwrap();

        let w = width as f32;
        let h = height as f32;
        let text_scale = 2.0;

        // --- Center buttons vertically within the window --------------------------
        let btn_w = self.buttons[0].width;
        let btn_h = self.buttons[0].height;
        let gap = 16.0;
        let total_h = self.buttons.len() as f32 * btn_h
            + (self.buttons.len() - 1) as f32 * gap;
        let start_y = (h - total_h) / 2.0;

        for (i, btn) in self.buttons.iter_mut().enumerate() {
            btn.x = (w - btn.width) / 2.0;
            btn.y = start_y + i as f32 * (btn_h + gap);
        }

        // --- Draw title -------------------------------------------------------
        let title = "QubePixel";
        let tw = title.len() as f32 * BitmapFont::char_width() * text_scale;
        let th = BitmapFont::char_height() * text_scale;
        let title_x = (w - tw) / 2.0;
        let title_y = start_y - th - 20.0;
        font.draw_text(
            encoder, view, device, queue,
            title, title_x, title_y,
            [1.0, 1.0, 1.0, 1.0],
            text_scale, width, height,
        );

        // --- Draw button rectangles with hover colour ----------------------------
        let rects: Vec<(f32, f32, f32, f32, [f32; 4])> = self
            .buttons
            .iter()
            .map(|b| {
                let c = b.current_color(self.cursor_x, self.cursor_y);
                (b.x, b.y, b.width, b.height, c)
            })
            .collect();

        ui.draw_rects(encoder, view, device, queue, &rects, width, height);

        // --- Draw text labels on buttons -----------------------------------------
        for btn in &self.buttons {
            let cw = BitmapFont::char_width() * text_scale;
            let ch = BitmapFont::char_height() * text_scale;
            let tx = btn.x + (btn.width - btn.label.len() as f32 * cw) / 2.0;
            let ty = btn.y + (btn.height - ch) / 2.0;
            font.draw_text(
                encoder, view, device, queue,
                &btn.label, tx, ty,
                [1.0, 1.0, 1.0, 1.0],
                text_scale, width, height,
            );
        }
    }

    fn post_process(&mut self, _dt: f64) {}

    fn on_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            self.mouse_down = true;
                        }
                        ElementState::Released => {
                            self.mouse_down = false;
                            for btn in &self.buttons {
                                if btn.contains_point(self.cursor_x, self.cursor_y) {
                                    debug_log!(
                                        "MainMenuScreen",
                                        "on_event",
                                        "Button '{}' clicked",
                                        btn.label
                                    );
                                    match btn.label.as_str() {
                                        "Play" => {
                                            self.pending_action =
                                                ScreenAction::Switch(Box::new(
                                                    GameScreen::new(),
                                                ));
                                        }
                                        "Settings" => {
                                            self.pending_action =
                                                ScreenAction::Push(Box::new(
                                                    SettingsScreen::new(),
                                                ));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            _ => {}
        }
    }

    fn poll_action(&mut self) -> ScreenAction {
        std::mem::replace(&mut self.pending_action, ScreenAction::None)
    }

    fn clear_color(&self) -> wgpu::Color {
        wgpu::Color {
            r: 0.1,
            g: 0.1,
            b: 0.3,
            a: 1.0,
        }
    }
}
