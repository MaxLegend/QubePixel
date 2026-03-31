// =============================================================================
// QubePixel — SettingsScreen  (render distance slider + Back button)
// =============================================================================

use crate::core::config;
use crate::core::screen::{Screen, ScreenAction};
use crate::debug_log;
use crate::flow_debug_log;
use crate::screens::bitmap_font::BitmapFont;
use crate::screens::button::Button;
use crate::screens::ui_renderer::UiRenderer;
use winit::event::{ElementState, WindowEvent};

// ---------------------------------------------------------------------------
// Slider helpers
// ---------------------------------------------------------------------------

/// Convert render distance (1..=8) to normalised slider position (0..=1).
fn rd_to_t(rd: i32) -> f32 {
    (rd - 1) as f32 / 127.0
}

fn t_to_rd(t: f32) -> i32 {
    (t * 127.0).round() as i32 + 1
}

// ---------------------------------------------------------------------------
// Slider layout constants (pixel sizes — independent of screen dimensions)
// ---------------------------------------------------------------------------
const TRACK_W: f32  = 320.0;
const TRACK_H: f32  = 10.0;
const HANDLE_W: f32 = 20.0;
const HANDLE_H: f32 = 32.0;

// ---------------------------------------------------------------------------
// SettingsScreen
// ---------------------------------------------------------------------------
pub struct SettingsScreen {
    ui:   Option<UiRenderer>,
    font: Option<BitmapFont>,

    buttons:  Vec<Button>,
    cursor_x: f64,
    cursor_y: f64,
    #[allow(dead_code)]
    mouse_down: bool,
    pending_action: ScreenAction,

    // -- Slider state --------------------------------------------------------
    /// Normalised slider position [0, 1] ↔ render distance [1, 8].
    slider_t:        f32,
    slider_dragging: bool,

    // Last rendered slider track rect (x, y, w, h) — used for hit-testing
    // in on_event which does not receive screen dimensions.
    slider_track: (f32, f32, f32, f32),
}

impl SettingsScreen {
    pub fn new() -> Self {
        debug_log!("SettingsScreen", "new", "Creating SettingsScreen");

        let back_button = Button::new(
            0.0, 0.0, 200.0, 50.0,
            [0.6, 0.2, 0.2, 1.0],
            [0.8, 0.3, 0.3, 1.0],
            "Back",
        );

        let rd = config::render_distance();
        debug_log!("SettingsScreen", "new", "Initial render distance: {}", rd);

        Self {
            ui:   None,
            font: None,
            buttons: vec![back_button],
            cursor_x: 0.0,
            cursor_y: 0.0,
            mouse_down: false,
            pending_action: ScreenAction::None,
            slider_t:        rd_to_t(rd),
            slider_dragging: false,
            slider_track:    (0.0, 0.0, TRACK_W, TRACK_H),
        }
    }

    // -----------------------------------------------------------------------
    // Apply slider position → write to global config
    // -----------------------------------------------------------------------
    fn apply_slider_at_cursor(&mut self) {
        let (tx, _ty, tw, _th) = self.slider_track;
        let raw_t = ((self.cursor_x as f32 - tx) / tw).clamp(0.0, 1.0);
        // Snap to integer steps
        let rd = t_to_rd(raw_t);
        self.slider_t = rd_to_t(rd);
        config::set_render_distance(rd);
    }
}

impl Screen for SettingsScreen {
    fn name(&self) -> &str { "Settings" }

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
        view:    &wgpu::TextureView,
        device:  &wgpu::Device,
        queue:   &wgpu::Queue,
        format:  wgpu::TextureFormat,
        width:   u32,
        height:  u32,
    ) {
        // Lazy-init
        if self.ui.is_none() {
            self.ui   = Some(UiRenderer::new(device, format));
            debug_log!("SettingsScreen", "render", "UiRenderer initialised lazily");
        }
        if self.font.is_none() {
            self.font = Some(BitmapFont::new(device, format));
            debug_log!("SettingsScreen", "render", "BitmapFont initialised lazily");
        }
        let ui   = self.ui.as_ref().unwrap();
        let font = self.font.as_ref().unwrap();

        let w = width  as f32;
        let h = height as f32;
        let ts = 2.0_f32;

        // ---- Title -----------------------------------------------------------
        let title   = "Settings";
        let tw_px   = title.len() as f32 * BitmapFont::char_width() * ts;
        let th_px   = BitmapFont::char_height() * ts;
        let title_x = (w - tw_px) / 2.0;
        let title_y = 40.0_f32;

        // ---- Back button (bottom-centre) --------------------------------------
        let btn = &mut self.buttons[0];
        btn.x = (w - btn.width)  / 2.0;
        btn.y = h - 80.0;

        // ---- Slider geometry -------------------------------------------------
        // Track centred horizontally, a bit above centre vertically
        let track_x = (w - TRACK_W) / 2.0;
        let track_y = h / 2.0 - TRACK_H / 2.0 + 20.0;
        self.slider_track = (track_x, track_y, TRACK_W, TRACK_H);

        let handle_x = track_x + self.slider_t * TRACK_W - HANDLE_W / 2.0;
        let handle_y = track_y - (HANDLE_H - TRACK_H) / 2.0;

        // Filled portion of the track (left of handle)
        let fill_w = (self.slider_t * TRACK_W).max(0.0);

        // ---- Collect rectangles ----------------------------------------------
        let title_bar: (f32, f32, f32, f32, [f32; 4]) = (
            title_x - 12.0,
            title_y - 6.0,
            tw_px + 24.0,
            th_px + 12.0,
            [0.4, 0.4, 0.5, 1.0],
        );

        // Track background
        let track_bg  = (track_x, track_y, TRACK_W, TRACK_H, [0.25, 0.25, 0.35, 1.0_f32]);
        // Filled portion
        let track_fill= (track_x, track_y, fill_w, TRACK_H,  [0.3, 0.5, 0.9, 1.0_f32]);
        // Handle
        let cx_f = self.cursor_x as f32;
        let cy_f = self.cursor_y as f32;
        let on_handle = cx_f >= handle_x && cx_f <= handle_x + HANDLE_W
            && cy_f >= handle_y && cy_f <= handle_y + HANDLE_H;
        let handle_color: [f32; 4] = if on_handle || self.slider_dragging {
            [0.9, 0.9, 1.0, 1.0]
        } else {
            [0.7, 0.7, 0.85, 1.0]
        };
        let handle_rect = (handle_x, handle_y, HANDLE_W, HANDLE_H, handle_color);

        let mut rects: Vec<(f32, f32, f32, f32, [f32; 4])> =
            vec![title_bar, track_bg, track_fill, handle_rect];

        for b in &self.buttons {
            rects.push((b.x, b.y, b.width, b.height,
                        b.current_color(self.cursor_x, self.cursor_y)));
        }

        ui.draw_rects(encoder, view, device, queue, &rects, width, height);

        // ---- Text labels -----------------------------------------------------
        // Title
        font.draw_text(
            encoder, view, device, queue,
            title, title_x, title_y,
            [1.0, 1.0, 1.0, 1.0], ts, width, height,
        );

        // Slider label (above track)
        let rd       = t_to_rd(self.slider_t);
        let rd_label = format!("Render Distance: {} chunks", rd);
        let lw       = rd_label.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            &rd_label,
            (w - lw) / 2.0,
            track_y - BitmapFont::char_height() * 1.5 - 12.0,
            [1.0, 1.0, 1.0, 1.0], 1.5, width, height,
        );

        // Min / max labels beneath track
        let label_min = "1";
        let label_max = "128";
        let lh = BitmapFont::char_height() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            label_min, track_x, track_y + TRACK_H + 6.0,
            [0.7, 0.7, 0.7, 1.0], 1.5, width, height,
        );
        let rw = label_max.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            label_max, track_x + TRACK_W - rw, track_y + TRACK_H + 6.0,
            [0.7, 0.7, 0.7, 1.0], 1.5, width, height,
        );

        // Hint below slider
        let hint   = "Drag slider or click track";
        let hw     = hint.len() as f32 * BitmapFont::char_width() * 1.5;
        font.draw_text(
            encoder, view, device, queue,
            hint, (w - hw) / 2.0, track_y + TRACK_H + lh + 16.0,
            [0.6, 0.6, 0.6, 0.8], 1.5, width, height,
        );

        // Back button label
        for b in &self.buttons {
            let cw = BitmapFont::char_width() * ts;
            let ch = BitmapFont::char_height() * ts;
            let tx = b.x + (b.width  - b.label.len() as f32 * cw) / 2.0;
            let ty = b.y + (b.height - ch) / 2.0;
            font.draw_text(
                encoder, view, device, queue,
                &b.label, tx, ty,
                [1.0, 1.0, 1.0, 1.0], ts, width, height,
            );
        }
    }

    fn post_process(&mut self, _dt: f64) {}

    fn on_event(&mut self, event: &WindowEvent) {
        match event {
            // -- Cursor position -----------------------------------------------
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
                if self.slider_dragging {
                    self.apply_slider_at_cursor();
                }
            }

            // -- Mouse buttons ------------------------------------------------
            WindowEvent::MouseInput { state, button, .. } => {
                if *button == winit::event::MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            self.mouse_down = true;

                            let (tx, ty, tw, th) = self.slider_track;
                            let cx = self.cursor_x as f32;
                            let cy = self.cursor_y as f32;

                            // Hit-test: track area (generous vertical tolerance)
                            let in_track = cx >= tx && cx <= tx + tw
                                && cy >= ty - HANDLE_H / 2.0
                                && cy <= ty + TRACK_H + HANDLE_H / 2.0;

                            if in_track {
                                self.slider_dragging = true;
                                self.apply_slider_at_cursor();
                                debug_log!(
                                    "SettingsScreen", "on_event",
                                    "Slider drag start, t={:.2}", self.slider_t
                                );
                            }
                        }
                        ElementState::Released => {
                            self.mouse_down      = false;
                            self.slider_dragging = false;

                            // Button hit-testing
                            for btn in &self.buttons {
                                if btn.contains_point(self.cursor_x, self.cursor_y) {
                                    debug_log!(
                                        "SettingsScreen", "on_event",
                                        "Button '{}' clicked", btn.label
                                    );
                                    if btn.label == "Back" {
                                        self.pending_action = ScreenAction::Pop;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // -- Keyboard -----------------------------------------------------
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    if let winit::keyboard::PhysicalKey::Code(
                        winit::keyboard::KeyCode::Escape,
                    ) = event.physical_key
                    {
                        debug_log!(
                            "SettingsScreen", "on_event",
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
        wgpu::Color { r: 0.12, g: 0.12, b: 0.18, a: 1.0 }
    }
}
