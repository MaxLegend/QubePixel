# Entry Point & Screen System

## Files
- `src/main.rs` — winit event loop, `App` struct, frame timing, cursor lock
- `src/core/screen_manager.rs` — stack-based screen state machine
- `src/core/screen.rs` — `Screen` trait + `ScreenAction` enum
- `src/core/renderer.rs` — wgpu surface, device, queue, resize
- `src/core/egui_manager.rs` — egui integration (input → tessellate → render)

## Boot sequence

```
main() → EventLoop → Renderer::new() (async, tokio) → App::new()
  App::new(): ScreenManager::push(MainMenuScreen)
              EguiManager::new()
```

## Per-frame flow (RedrawRequested)

```
screen_manager.update(dt)
screen_manager.post_process(dt)
apply_cursor_lock()          ← OS cursor grab/release, only on state change
egui_ctx.run → screen_manager.build_ui(ctx)   ← egui UI tree
egui_manager.end_frame_and_tessellate()
renderer.render():
    screen_manager.render(encoder, view, ...)  ← 3D scene
    egui_manager.render_draw(encoder, view, ...) ← HUD on top
window.request_redraw()      ← drives next frame
```

## Screen lifecycle

`push()` → `load()` → (first update) `start()` → N× `update()` / `render()` / `build_ui()` → `unload()`

`ScreenAction` returned from `update()` triggers transitions:
- `Push(screen)` — overlay (pause menu, settings)
- `Switch(screen)` — replace entire stack
- `Pop` — go back
- `None` — stay

Only the **top** screen receives input events. All screens in the stack get `update()` + `render()` (bottom to top).

## Screens

| Screen | File | Purpose |
|---|---|---|
| `MainMenuScreen` | `screens/main_menu.rs` | Title screen, launch into game |
| `GameScreen` | `screens/game_screen.rs` | 3D world, player, HUD |
| `SettingsScreen` | `screens/settings.rs` | egui settings UI (writes to config atomics) |
| `DebugOverlay` | `screens/debug_overlay.rs` | F3-style overlay (coords, FPS, etc.) |
| `ProfilerOverlay` | `screens/profiler_overlay.rs` | Frame timing breakdown |
| `GpuInfo` | `screens/gpu_info.rs` | GPU adapter info panel |
| `SkyRenderer` | `screens/sky_renderer.rs` | Sky dome / sun / moon sprites |
| `PlayerRenderer` | `screens/player_renderer.rs` | Skeletal player body (per-bone draw calls) |
| `InventoryUI` | `screens/inventory.rs` | Hotbar + grid + drag-and-drop inventory |
| `WorldgenVisualizerScreen` | `screens/worldgen_visualizer_screen.rs` | Debug view of biome/terrain pipeline |

## Cursor lock

`screen.wants_pointer_lock()` → `App::apply_cursor_lock()` calls OS API only when state changes. Tries `CursorGrabMode::Locked`, falls back to `Confined`.
