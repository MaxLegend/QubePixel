mod core;
mod screens;

use crate::screens::main_menu::MainMenuScreen;
use std::time::Instant;

use winit::event::{DeviceEvent, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};


use core::logging;
use core::input_controller::InputController;
use core::screen_manager::ScreenManager;
use crate::core::renderer::Renderer;
use crate::core::egui_manager::EguiManager;

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
struct App {
    window:           Option<winit::window::Window>,
    screen_manager:   ScreenManager,
    #[allow(dead_code)]
    input_controller: InputController,
    renderer:         Renderer,
    egui_manager: EguiManager,
    cursor_locked:    bool,
    tokio_runtime:    tokio::runtime::Runtime,

}

impl App {
    fn new(
        window: winit::window::Window,
        renderer: Renderer,
        tokio_runtime: tokio::runtime::Runtime,
    ) -> Self {
        let mut screen_manager = ScreenManager::new();
        let egui_manager = EguiManager::new(&window, &renderer.device(), renderer.surface_format());

        screen_manager.push(Box::new(MainMenuScreen::new()));
        // Создаём EguiManager ПЕРЕД Self {}, потому что нужен device и window

        Self {
            window: Some(window),
            screen_manager,
            input_controller: InputController::new(),
            renderer,
            egui_manager,              // <--- ДОБАВИТЬ
            cursor_locked: false,
            tokio_runtime,
        }
    }

    /// Apply cursor grab / visibility based on what the active screen requests.
    /// Only calls the OS API when the lock state actually changes.
    fn apply_cursor_lock(&mut self) {
        let wants = self.screen_manager.wants_pointer_lock();
        if wants == self.cursor_locked {
            return; // no change
        }

        if let Some(window) = self.window.as_ref() {
            if wants {
                // Try Locked first (hides and centres cursor), fall back to Confined.
                let result = window
                    .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                    .or_else(|_| window.set_cursor_grab(winit::window::CursorGrabMode::Confined));

                if let Err(e) = result {
                    debug_log!("Main", "apply_cursor_lock", "set_cursor_grab failed: {}", e);
                } else {
                    window.set_cursor_visible(false);
                    self.cursor_locked = true;
                    debug_log!("Main", "apply_cursor_lock", "Cursor locked");
                }
            } else {
                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                window.set_cursor_visible(true);
                self.cursor_locked = false;
                debug_log!("Main", "apply_cursor_lock", "Cursor released");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    debug_log!("Main", "init", "Starting QubePixel v0.0.1");

    let event_loop = EventLoop::new()
        .expect("Failed to create winit event loop");

    debug_log!("Main", "init", "EventLoop created");

    let window = event_loop
        .create_window(
            winit::window::WindowAttributes::default()
                .with_title("QubePixel")
                .with_inner_size(winit::dpi::LogicalSize::new(840, 480)),
        )
        .expect("Failed to create winit window");

    debug_log!("Main", "init", "Window created");

    let tokio_runtime = tokio::runtime::Runtime::new()
        .expect("Failed to create tokio runtime");
    let renderer = tokio_runtime.block_on(Renderer::new(&window));
    let mut app  = App::new(window, renderer, tokio_runtime);
    let mut last_time = Instant::now();

    debug_log!("Main", "init", "App initialised, entering event loop");

    event_loop
        .run(move |event, target| {
            target.set_control_flow(ControlFlow::Wait);
            // ---- Feed ALL events to egui first ----

            match event {
                // ---- Device events (raw mouse motion when cursor is locked) ----
                Event::DeviceEvent {
                    event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                    ..
                } => {
                    // Forward raw delta to the active screen regardless of window focus.
                    // The screen decides whether to use it (camera_locked check).
                    app.screen_manager.on_mouse_motion(dx, dy);
                }

                // ---- Window events -----------------------------------------------
                Event::WindowEvent { event, window_id } => {
                    let is_ours = app
                        .window
                        .as_ref()
                        .map(|w| w.id() == window_id)
                        .unwrap_or(false);
                    if !is_ours { return; }
                    if let Some(window) = app.window.as_ref() {
                        app.egui_manager.on_window_event(window, &event);
                    }
                    match event {
                        WindowEvent::CloseRequested => {
                            debug_log!("Main", "window_event", "CloseRequested");
                            target.exit();
                        }

                        WindowEvent::Resized(size) => {
                            debug_log!(
                                "Main", "window_event",
                                "Resized: {}x{}", size.width, size.height
                            );
                            app.renderer.resize(size);
                            let scale = app.window.as_ref()
                                .map(|w| w.scale_factor() as f32)
                                .unwrap_or(1.0);
                            app.egui_manager.update_screen_size(
                                size.width, size.height, scale,
                            );
                        }

                        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                            debug_log!(
                                "Main", "window_event",
                                "Scale factor: {}", scale_factor
                            );
                            let s = app.renderer.size();
                            app.egui_manager.update_screen_size(
                                s.width, s.height, scale_factor as f32,
                            );
                        }

                        WindowEvent::RedrawRequested => {
                            let now = Instant::now();
                            let dt  = now.duration_since(last_time).as_secs_f64();
                            last_time = now;

                            app.screen_manager.update(dt);
                            app.screen_manager.post_process(dt);

                            // Apply cursor lock based on active screen's request
                            app.apply_cursor_lock();

                            // Window title
                            if let Some(window) = app.window.as_ref() {
                                if let Some(screen) = app.screen_manager.active_screen() {
                                    window.set_title(
                                        &format!("QubePixel — {}", screen.name())
                                    );
                                }
                            }

                            // --- egui frame: build UI + tessellate ----------------
                            let clipped_primitives = {
                                // Обновляем размер экрана для egui
                                let size = app.renderer.size();
                                let scale = app.window.as_ref()
                                    .map(|w| w.scale_factor() as f32)
                                    .unwrap_or(1.0);
                                app.egui_manager.update_screen_size(
                                    size.width, size.height, scale,
                                );
                                if let Some(window) = app.window.as_ref() {
                                    app.egui_manager.update_viewport_info(window);
                                }
                                // Берём RawInput из winit state
                                let raw_input = app.egui_manager

                                    .take_egui_input(app.window.as_ref().unwrap());

                                // Строим UI через ScreenManager
                                let egui_ctx = app.egui_manager.ctx().clone();
                                let full_output = egui_ctx.run(raw_input, |ctx| {
                                    app.screen_manager.build_ui(ctx);
                                });
                                // if let Some(window) = app.window.as_ref() {
                                //     app.egui_manager.handle_platform_output(
                                //         window,
                                //         full_output.platform_output,
                                //     );
                                // }
                                // Текстуры + тесселяция
                                app.egui_manager.end_frame_and_tessellate(
                                    app.renderer.device(),
                                    app.renderer.queue(),
                                    full_output,
                                )
                            };

                            // --- Рендер (3D + egui) --------------------------------
                            {
                                let clear_color = app.screen_manager.clear_color();
                                let size        = app.renderer.size();
                                let sm          = &mut app.screen_manager;
                                let egui_mgr    = &mut app.egui_manager;
                                let renderer    = &mut app.renderer;

                                renderer.render(
                                    clear_color,
                                    |encoder, view, device, queue, format| {
                                        // 3D сцена
                                        sm.render(
                                            encoder, view, device, queue,
                                            format, size.width, size.height,
                                        );
                                        // egui HUD поверх 3D
                                        egui_mgr.render_draw(
                                            encoder, view, device, queue,
                                            &clipped_primitives,
                                        );
                                    },
                                );
                            }

                            if let Some(window) = app.window.as_ref() {
                                window.request_redraw();
                            }
                        }

                        other => {
                            app.screen_manager.on_event(&other);
                        }
                    }
                }

                Event::AboutToWait => {
                    if let Some(window) = app.window.as_ref() {
                        window.request_redraw();
                    }
                }

                _ => {}
            }
        })
        .expect("Event loop error");
}