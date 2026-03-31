// =============================================================================
// QubePixel — SettingsScreen (Back button, placeholder settings UI)
// =============================================================================

use crate::core::screen::{Screen, ScreenAction};
use crate::debug_log;
use crate::flow_debug_log;
use crate::screens::bitmap_font::BitmapFont;
use crate::screens::button::Button;
use crate::screens::ui_renderer::UiRenderer;
use winit::event::{ElementState, WindowEvent};

/// A minimal settings screen.
///
/// Currently displays only a **Back** button that pops this screen off the stack,
/// returning to whatever screen lies beneath (typically the MainMenuScreen).
/// Real settings controls (volume, resolution, key bindings) will be added later.
pub struct SettingsScreen {
    ui: Option<UiRenderer>,
    font: Option<BitmapFont>,
    buttons: Vec<Button>,
    cursor_x: f64,
    cursor_y: f64,
    #[allow(dead_code)]
    mouse_down: bool,
    pending_action: ScreenAction,
}

impl SettingsScreen {
    pub fn new() -> Self {
        debug_log!("SettingsScreen", "new", "Creating SettingsScreen");

        let back_button = Button::new(
            0.0, 0.0, 200.0, 50.0,
            [0.6, 0.2, 0.2, 1.0],   // red
            [0.8, 0.3, 0.3, 1.0],   // hover red
            "Back",
        );

        let buttons = vec![back_button];

        debug_log!(
            "SettingsScreen",
            "new",
            "SettingsScreen created with {} button(s)",
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

impl Screen for SettingsScreen {
    fn name(&self) -> &str {
        "Settings"
    }

    fn load(&mut self) {
        debug_log!("SettingsScreen", "load", "Loading SettingsScreen resources");
    }

    fn start(&mut self) {
        debug_log!("SettingsScreen", "start", "SettingsScreen first-frame init");
    }

    fn update(&mut self, _dt: f64) {
        flow_debug_log!("SettingsScreen", "update", "Update tick");
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
            debug_log!("SettingsScreen", "render", "UiRenderer initialised lazily");
        }
        if self.font.is_none() {
            self.font = Some(BitmapFont::new(device, format));
            debug_log!("SettingsScreen", "render", "BitmapFont initialised lazily");
        }
        let ui = self.ui.as_ref().unwrap();
        let font = self.font.as_ref().unwrap();

        let w = width as f32;
        let h = height as f32;
        let text_scale = 2.0;

        // --- Draw a decorative title bar ---------------------------------------
        let title = "Settings";
        let tw = title.len() as f32 * BitmapFont::char_width() * text_scale;
        let th = BitmapFont::char_height() * text_scale;
        let title_x = (w - tw) / 2.0;
        let title_y = 40.0;

        let title_bar: (f32, f32, f32, f32, [f32; 4]) = (
            title_x - 12.0,
            title_y - 6.0,
            tw + 24.0,
            th + 12.0,
            [0.4, 0.4, 0.5, 1.0],
        );

        // --- Center the Back button in the window ------------------------------
        let btn = &mut self.buttons[0];
        btn.x = (w - btn.width) / 2.0;
        btn.y = h - 80.0;

        // --- Draw rectangles (title bar + buttons) -----------------------------
        let mut rects: Vec<(f32, f32, f32, f32, [f32; 4])> = vec![title_bar];
        for b in &self.buttons {
            let c = b.current_color(self.cursor_x, self.cursor_y);
            rects.push((b.x, b.y, b.width, b.height, c));
        }
        ui.draw_rects(encoder, view, device, queue, &rects, width, height);

        // --- Draw title text ---------------------------------------------------
        font.draw_text(
            encoder, view, device, queue,
            title, title_x, title_y,
            [1.0, 1.0, 1.0, 1.0],
            text_scale, width, height,
        );

        // --- Draw text labels on buttons ---------------------------------------
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
                                        "SettingsScreen",
                                        "on_event",
                                        "Button '{}' clicked",
                                        btn.label
                                    );
                                    match btn.label.as_str() {
                                        "Back" => {
                                            self.pending_action = ScreenAction::Pop;
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Escape also goes back.
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let winit::keyboard::PhysicalKey::Code(
                        winit::keyboard::KeyCode::Escape,
                    ) = event.physical_key
                    {
                        debug_log!(
                            "SettingsScreen",
                            "on_event",
                            "Escape pressed, requesting Pop"
                        );
                        self.pending_action = ScreenAction::Pop;
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
            r: 0.15,
            g: 0.15,
            b: 0.2,
            a: 1.0,
        }
    }
}
