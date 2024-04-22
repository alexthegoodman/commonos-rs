use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 2], // x, y coordinates
    color: [f32; 3],    // RGB color
}

// Ensure Vertex is Pod and Zeroable
unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, // Corresponds to layout(location = 0) in shader
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1, // Corresponds to layout(location = 1) in shader
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn create_vertex_and_index_buffers(
    device: &wgpu::Device,
    // bind_group_layout: &wgpu::BindGroupLayout,
    vertices: &[Vertex],
    indices: &[u16],
) -> (wgpu::Buffer, wgpu::Buffer) {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(vertices), // Convert slice of Vertex to bytes
        usage: wgpu::BufferUsages::VERTEX,
    });

    // Create a Bind Group using the layout and buffer
    // let vertex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //     layout: &bind_group_layout,
    //     entries: &[wgpu::BindGroupEntry {
    //         binding: 0,
    //         resource: wgpu::BindingResource::Buffer(vertex_buffer.as_entire_buffer_binding()),
    //     }],
    //     label: Some("Bind Group"),
    // });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(indices), // Convert slice of u16 to bytes
        usage: wgpu::BufferUsages::INDEX,
    });

    // Create a Bind Group using the layout and buffer
    // let index_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
    //     layout: &bind_group_layout,
    //     entries: &[wgpu::BindGroupEntry {
    //         binding: 1,
    //         resource: wgpu::BindingResource::Buffer(index_buffer.as_entire_buffer_binding()),
    //     }],
    //     label: Some("Bind Group"),
    // });

    (
        vertex_buffer,
        index_buffer,
        // vertex_bind_group,
        // index_bind_group,
    )
}

fn create_layouts(device: &wgpu::Device) -> (wgpu::PipelineLayout) {
    // Create a Bind Group Layout
    // let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //     label: Some("Bind Group Layout"),
    //     entries: &[wgpu::BindGroupLayoutEntry {
    //         binding: 0,
    //         visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
    //         ty: wgpu::BindingType::Buffer {
    //             ty: wgpu::BufferBindingType::Uniform,
    //             has_dynamic_offset: false,
    //             min_binding_size: None,
    //         },
    //         count: None,
    //     }],
    // });

    // Create a Pipeline Layout
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        // bind_group_layouts: &[&bind_group_layout],
        bind_group_layouts: &[], // No bind group layouts
        push_constant_ranges: &[],
    });

    (pipeline_layout)
}

async fn initialize_core(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    // Create logical components (instance, adapter, device, queue, surface, etc.)
    let dx12_compiler = wgpu::Dx12Compiler::Dxc {
        dxil_path: None, // Specify a path to custom location
        dxc_path: None,  // Specify a path to custom location
    };

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        dx12_shader_compiler: dx12_compiler,
        flags: wgpu::InstanceFlags::empty(),
        gles_minor_version: wgpu::Gles3MinorVersion::Version2,
    });

    let surface = unsafe {
        instance
            .create_surface(&window)
            .expect("Couldn't create GPU surface")
    };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Couldn't fetch GPU adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None, // Trace path can be specified here for debugging purposes
        )
        .await
        .expect("Failed to create device");

    // Define the layouts
    let (pipeline_layout) = create_layouts(&device);

    // let width = 800;  // Viewport width
    // let height = 600; // Viewport height
    let rect_width = 200; // Rectangle width in pixels
    let rect_height = 100; // Rectangle height in pixels
    let scale_factor = 1.5; // TODO: fetch dynamic scaling factor

    // Adjust rectangle dimensions according to the scaling factor
    let scaled_rect_width = rect_width as f32 * scale_factor;
    let scaled_rect_height = rect_height as f32 * scale_factor;

    let ndc_width = (scaled_rect_width / size.width as f32) * 2.0;
    let ndc_height = (scaled_rect_height / size.height as f32) * 2.0;

    let vertices = [
        Vertex {
            position: [-ndc_width / 2.0, -ndc_height / 2.0],
            color: [1.0, 0.0, 0.0],
        }, // Bottom left
        Vertex {
            position: [ndc_width / 2.0, -ndc_height / 2.0],
            color: [0.0, 1.0, 0.0],
        }, // Bottom right
        Vertex {
            position: [ndc_width / 2.0, ndc_height / 2.0],
            color: [0.0, 0.0, 1.0],
        }, // Top right
        Vertex {
            position: [-ndc_width / 2.0, ndc_height / 2.0],
            color: [1.0, 1.0, 0.0],
        }, // Top left
    ];

    // Set up the vertex and index buffer data
    let (vertex_buffer, index_buffer) = create_vertex_and_index_buffers(
        &device,
        // &bind_group_layout,
        &vertices,
        &[0, 1, 2, 2, 3, 0],
    );

    // Load the shaders
    let shader_module_vert_primary = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Primary Vert Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vert_primary.wgsl").into()),
    });

    let shader_module_frag_primary = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Primary Frag Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/frag_primary.wgsl").into()),
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0]; // Choosing the first available format

    // Configure the render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("EverydayGUI Primary Render Pipeline"),
        layout: Some(&pipeline_layout),
        multiview: None,
        vertex: wgpu::VertexState {
            module: &shader_module_vert_primary,
            entry_point: "vs_main", // Entry point for vertex shader
            buffers: &[Vertex::desc()],
        },
        fragment: Some(wgpu::FragmentState {
            // Optional, needed for coloring
            module: &shader_module_frag_primary,
            entry_point: "fs_main", // Entry point for fragment shader
            targets: &[Some(wgpu::ColorTargetState {
                // format: surface.get_preferred_format(&adapter).unwrap(),
                format: swapchain_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to false is useful for debugging
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None, // Optional, configure if using depth testing
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    });

    let mut config = surface
        .get_default_config(&adapter, size.width, size.height)
        .unwrap();
    surface.configure(&device, &config);

    // execute winit render loop
    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&instance, &adapter, &shader, &pipeline_layout);

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        // Reconfigure the surface with the new size
                        config.width = new_size.width.max(1);
                        config.height = new_size.height.max(1);
                        surface.configure(&device, &config);
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });
                        {
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            rpass.set_index_buffer(
                                index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            rpass.draw_indexed(0..6, 0, 0..1); // TODO: make dynamic
                        }

                        queue.submit(Some(encoder.finish()));
                        frame.present();
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

fn main() {
    print!("Welcome to EverydayGUI!");

    // establish winit window and render loop
    let event_loop = EventLoop::new().expect("Failed to create an event loop");
    let window = WindowBuilder::new()
        .with_title("EverydayGUI Demo Application")
        .with_resizable(true)
        .with_transparent(false)
        .build(&event_loop)
        .unwrap();

    futures::executor::block_on(initialize_core(event_loop, window));
}
