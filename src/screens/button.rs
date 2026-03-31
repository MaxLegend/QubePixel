// =============================================================================
// QubePixel — Button (clickable UI rectangle with hover state)
// =============================================================================

use crate::debug_log;

/// A simple UI button defined by a screen-space rectangle and two colours
/// (normal and hover). Used by screens for mouse-click hit testing.
pub struct Button {
    /// Top-left X in physical pixels.
    pub x: f32,
    /// Top-left Y in physical pixels.
    pub y: f32,
    /// Width in physical pixels.
    pub width: f32,
    /// Height in physical pixels.
    pub height: f32,
    /// RGBA colour when the cursor is NOT over the button.
    pub color: [f32; 4],
    /// RGBA colour when the cursor IS over the button.
    pub hover_color: [f32; 4],
    /// Human-readable label ( informational, not rendered yet).
    pub label: String,
}

impl Button {
    /// Creates a new button with the given bounds, colours, and label.
    pub fn new(
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: [f32; 4],
        hover_color: [f32; 4],
        label: &str,
    ) -> Self {
        debug_log!(
            "Button",
            "new",
            "Button created: '{}' at ({}, {}) size {}x{}",
            label,
            x,
            y,
            width,
            height
        );
        Self {
            x,
            y,
            width,
            height,
            color,
            hover_color,
            label: label.to_owned(),
        }
    }

    /// Returns `true` if the point `(px, py)` falls inside this button's bounds.
    pub fn contains_point(&self, px: f64, py: f64) -> bool {
        px >= self.x as f64
            && px <= (self.x + self.width) as f64
            && py >= self.y as f64
            && py <= (self.y + self.height) as f64
    }

    /// Returns the colour that should be rendered right now, depending on
    /// whether the cursor is hovering over the button.
    pub fn current_color(&self, cursor_x: f64, cursor_y: f64) -> [f32; 4] {
        if self.contains_point(cursor_x, cursor_y) {
            self.hover_color
        } else {
            self.color
        }
    }
}
