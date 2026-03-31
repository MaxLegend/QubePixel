// =============================================================================
// Minecraft Clone — InputController (keyboard + mouse state tracking)
// =============================================================================

use std::collections::HashSet;

use crate::{debug_log, ext_debug_log};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

// ---------------------------------------------------------------------------
// InputController — tracks keyboard and mouse state each frame
// ---------------------------------------------------------------------------
pub struct InputController {
    /// Currently held-down keys (physical key codes).
    keys_pressed: HashSet<PhysicalKey>,

    /// Currently held-down mouse buttons.
    buttons_pressed: HashSet<MouseButton>,

    /// Mouse cursor position in physical pixels (relative to window top-left).
    cursor_x: f64,
    cursor_y: f64,

    /// Mouse scroll delta (accumulated per frame, cleared after read).
    scroll_delta_x: f32,
    scroll_delta_y: f32,
}

impl InputController {
    pub fn new() -> Self {
        debug_log!("InputController", "new", "InputController created");
        Self {
            keys_pressed: HashSet::new(),
            buttons_pressed: HashSet::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            scroll_delta_x: 0.0,
            scroll_delta_y: 0.0,
        }
    }

    // -----------------------------------------------------------------------
    // Event processing — call this for every WindowEvent
    // -----------------------------------------------------------------------

    /// Process a window event. Returns `true` if the event was handled
    /// (keyboard, mouse, or scroll), `false` otherwise.
    pub fn handle_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            // ── Keyboard ──────────────────────────────────────────────────
            WindowEvent::KeyboardInput { event, .. } => {
                match event.state {
                    ElementState::Pressed => {
                        ext_debug_log!(
                            "InputController",
                            "handle_event",
                            "Key pressed: {:?}",
                            event.logical_key
                        );
                        self.keys_pressed.insert(event.physical_key);
                    }
                    ElementState::Released => {
                        ext_debug_log!(
                            "InputController",
                            "handle_event",
                            "Key released: {:?}",
                            event.logical_key
                        );
                        self.keys_pressed.remove(&event.physical_key);
                    }
                }
                true
            }

            // ── Mouse buttons ─────────────────────────────────────────────
            WindowEvent::MouseInput { state, button, .. } => {
                match state {
                    ElementState::Pressed => {
                        ext_debug_log!(
                            "InputController",
                            "handle_event",
                            "Mouse button pressed: {:?}",
                            button
                        );
                        self.buttons_pressed.insert(*button);
                    }
                    ElementState::Released => {
                        ext_debug_log!(
                            "InputController",
                            "handle_event",
                            "Mouse button released: {:?}",
                            button
                        );
                        self.buttons_pressed.remove(button);
                    }
                }
                true
            }

            // ── Cursor position ───────────────────────────────────────────
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x;
                self.cursor_y = position.y;
                true
            }

            // ── Scroll ────────────────────────────────────────────────────
            WindowEvent::MouseWheel { delta, .. } => {
                use winit::event::MouseScrollDelta;
                match delta {
                    MouseScrollDelta::LineDelta(x, y) => {
                        self.scroll_delta_x += *x;
                        self.scroll_delta_y += *y;
                    }
                    MouseScrollDelta::PixelDelta(offset) => {
                        self.scroll_delta_x += offset.x as f32;
                        self.scroll_delta_y += offset.y as f32;
                    }
                }
                true
            }

            // ── Not an input event ────────────────────────────────────────
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Queries — called by screens during update()
    // -----------------------------------------------------------------------

    /// Returns `true` if the given key is currently held down.
    #[allow(dead_code)]
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&PhysicalKey::Code(key))
    }

    /// Returns `true` if the given mouse button is currently held down.
    #[allow(dead_code)]
    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.buttons_pressed.contains(&button)
    }

    /// Returns the mouse cursor position in physical pixels.
    #[allow(dead_code)]
    pub fn cursor_position(&self) -> (f64, f64) {
        (self.cursor_x, self.cursor_y)
    }

    /// Returns and clears the accumulated scroll delta for this frame.
    #[allow(dead_code)]
    pub fn consume_scroll_delta(&mut self) -> (f32, f32) {
        let delta = (self.scroll_delta_x, self.scroll_delta_y);
        self.scroll_delta_x = 0.0;
        self.scroll_delta_y = 0.0;
        delta
    }

    /// Returns `true` if no keys or buttons are currently pressed.
    #[allow(dead_code)]
    pub fn is_idle(&self) -> bool {
        self.keys_pressed.is_empty() && self.buttons_pressed.is_empty()
    }
}