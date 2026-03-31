// =============================================================================
// QubePixel — UiRenderer (2D coloured-rectangle pipeline for screens)
// =============================================================================

use crate::debug_log;

// ---------------------------------------------------------------------------
// Vertex — screen-space position + RGBA colour
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
}

// ---------------------------------------------------------------------------
// UiRenderer — creates a single wgpu pipeline used by all 2D screens
// ---------------------------------------------------------------------------
pub struct UiRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
}

impl UiRenderer {
    /// Creates the 2D UI rendering pipeline.
    ///
    /// * `device` — the wgpu device (borrowed, the pipeline is independent).
    /// * `format`  — the surface texture format to blend into.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        debug_log!("UiRenderer", "new", "Creating 2D UI pipeline, format={:?}", format);

        // -- WGSL shaders -------------------------------------------------------
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r"
                struct Uniforms {
                    screen_size: vec2<f32>,
                };

                @group(0) @binding(0) var<uniform> u: Uniforms;

                struct VertexOutput {
                    @builtin(position) position: vec4<f32>,
                    @location(0) color: vec4<f32>,
                };

                @vertex
                fn vs_main(@location(0) pos: vec2<f32>,
                           @location(1) col: vec4<f32>) -> VertexOutput {
                    var out: VertexOutput;
                    // Convert pixel coordinates (origin top-left) to clip space
                    let clip = (pos / u.screen_size) * 2.0 - vec2<f32>(1.0, 1.0);
                    out.position = vec4<f32>(clip.x, -clip.y, 0.0, 1.0);
                    out.color = col;
                    return out;
                }

                @fragment
                fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                    return in.color;
                }
                ",
            )),
        });

        // -- Bind group layout (uniform: screen size) ----------------------------
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UI Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("UI Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // -- Vertex buffer layout -----------------------------------------------
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
        };

        // -- Render pipeline ----------------------------------------------------
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // -- Uniform buffer (2 x f32 = screen width/height) ---------------------
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Uniform Buffer"),
            size: 8, // 2 x f32
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        debug_log!("UiRenderer", "new", "2D UI pipeline created successfully");

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
        }
    }

    // -----------------------------------------------------------------------
    // draw_rects — batch-render a list of coloured rectangles
    // -----------------------------------------------------------------------

    /// Draws one or more coloured rectangles in a single GPU draw call.
    ///
    /// Each entry is `(x, y, width, height, [r, g, b, a])` in physical pixels.
    /// All coordinates are in screen space with origin at the top-left.
    pub fn draw_rects(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rects: &[(f32, f32, f32, f32, [f32; 4])],
        screen_width: u32,
        screen_height: u32,
    ) {
        if rects.is_empty() {
            return;
        }

        // 1) Update the uniform with the current screen dimensions
        let size_data: [f32; 2] = [screen_width as f32, screen_height as f32];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                size_data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 2]>(),
            )
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytes);

        // 2) Create a bind group for this frame (uniform buffer is already updated)
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Bind Group (frame)"),
            layout: &self.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniform_buffer.as_entire_binding(),
            }],
        });

        // 3) Build the vertex buffer: 2 triangles per rectangle = 6 vertices
        let vertices: Vec<Vertex> = rects
            .iter()
            .flat_map(|&(x, y, w, h, color)| {
                // Triangle 1: top-left, top-right, bottom-left
                // Triangle 2: top-right, bottom-right, bottom-left
                let v = |px: f32, py: f32| Vertex {
                    position: [px, py],
                    color,
                };
                [
                    v(x, y),
                    v(x + w, y),
                    v(x, y + h),
                    v(x + w, y),
                    v(x + w, y + h),
                    v(x, y + h),
                ]
            })
            .collect();

        let vertex_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<Vertex>(),
            )
        };
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Vertex Buffer (frame)"),
            size: vertex_bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, vertex_bytes);

        // 4) Render pass
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("UI Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve the clear colour behind
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..(vertices.len() as u32), 0..1);
    }
}
