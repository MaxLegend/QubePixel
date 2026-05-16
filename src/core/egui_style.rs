// =============================================================================
// Techpixel — Unified pixel-style egui theme
// =============================================================================
//
// Centralised colour palette + theme applicator.
// Every menu, button, slider and checkbox shares the same teal/cyan
// pixel-art look with zero corner rounding.
// =============================================================================

use egui::{Color32, Rounding, Stroke, Vec2};

// ---------------------------------------------------------------------------
// Colour palette — teal / cyan pixel theme
// ---------------------------------------------------------------------------

/// Primary button fill (dark teal)
pub const BTN_FILL: Color32   = Color32::from_rgb(45, 125, 154);
/// Button hover fill
pub const BTN_HOVER: Color32  = Color32::from_rgb(61, 157, 186);
/// Button active / pressed fill
pub const BTN_ACTIVE: Color32 = Color32::from_rgb(29, 93, 122);
/// Button border
pub const BTN_BORDER: Color32 = Color32::from_rgb(20, 60, 80);
/// Button text colour
pub const BTN_TEXT: Color32   = Color32::WHITE;

/// Accent colour (bright mint) — used for highlights, hover borders, links
pub const ACCENT: Color32 = Color32::from_rgb(0, 229, 176);

/// Panel / sidebar background
pub const PANEL_BG: Color32 = Color32::from_rgb(18, 22, 36);
/// Window / overlay background
pub const WINDOW_BG: Color32 = Color32::from_rgb(21, 26, 40);
/// Window / panel border
pub const WINDOW_BORDER: Color32 = Color32::from_rgb(58, 68, 96);

/// Very dark background
pub const BG_DARK: Color32 = Color32::from_rgb(12, 16, 24);

/// Primary text
pub const TEXT: Color32      = Color32::from_rgb(232, 236, 240);
/// Dim / secondary text
pub const TEXT_DIM: Color32  = Color32::from_rgb(136, 144, 160);
/// Dark / tertiary text
pub const TEXT_DARK: Color32 = Color32::from_rgb(80, 88, 104);

/// Separator colour
pub const SEPARATOR: Color32 = Color32::from_rgb(58, 68, 96);

// ---------------------------------------------------------------------------
// Apply the pixel theme to an egui context
// ---------------------------------------------------------------------------

/// Apply the unified Techpixel pixel-art visual theme.
/// Call once after creating the `egui::Context`.
pub fn apply_pixel_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // --- Visuals (colours & widget appearance) ---
    let mut v = egui::Visuals::dark();
    v.dark_mode = true;

    let zero = Rounding::ZERO;

    // Zero rounding on every widget state
    v.widgets.noninteractive.corner_radius = zero;
    v.widgets.inactive.corner_radius      = zero;
    v.widgets.hovered.corner_radius       = zero;
    v.widgets.active.corner_radius        = zero;
    v.widgets.open.corner_radius          = zero;

    // Inactive (normal) widgets — buttons, sliders, checkboxes …
    v.widgets.inactive.bg_fill      = BTN_FILL;
    v.widgets.inactive.weak_bg_fill = BTN_FILL;
    v.widgets.inactive.bg_stroke    = Stroke::new(2.0, BTN_BORDER);
    v.widgets.inactive.fg_stroke    = Stroke::new(1.5, BTN_TEXT);

    // Hovered widgets
    v.widgets.hovered.bg_fill      = BTN_HOVER;
    v.widgets.hovered.weak_bg_fill = BTN_HOVER;
    v.widgets.hovered.bg_stroke    = Stroke::new(2.0, ACCENT);
    v.widgets.hovered.fg_stroke    = Stroke::new(1.5, BTN_TEXT);
    v.widgets.hovered.expansion    = 1.0;

    // Active (pressed) widgets
    v.widgets.active.bg_fill      = BTN_ACTIVE;
    v.widgets.active.weak_bg_fill = BTN_ACTIVE;
    v.widgets.active.bg_stroke    = Stroke::new(2.0, ACCENT);
    v.widgets.active.fg_stroke    = Stroke::new(2.0, BTN_TEXT);

    // Noninteractive — labels, separators, panel backgrounds
    v.widgets.noninteractive.bg_fill   = PANEL_BG;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, WINDOW_BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_DIM);

    // Selection / text selection
    v.selection.bg_fill = Color32::from_rgba_unmultiplied(0, 229, 176, 80);
    v.selection.stroke  = Stroke::new(1.0, ACCENT);

    // Misc colours
    v.hyperlink_color       = ACCENT;
    v.faint_bg_color        = BG_DARK;
    v.extreme_bg_color      = Color32::BLACK;
    v.code_bg_color         = Color32::from_rgb(30, 38, 56);
    v.override_text_color   = Some(TEXT);

    style.visuals = v;

    // --- Spacing (tight, pixel-friendly) ---
    style.spacing.item_spacing   = Vec2::new(8.0, 4.0);
    style.spacing.button_padding = Vec2::new(10.0, 5.0);

    ctx.set_style(style);
}
