use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

// TODO: move to components/button.rs
enum ButtonVariant {
    Green,
    Dark,
    Light,
}

enum ButtonKind {
    SmallIcon,
    SmallShort,
    SmallWide,
    LargeIcon,
    LargeShort,
    LargeWide,
}

struct Button {
    position: (f32, f32),
    variant: ButtonVariant,
    kind: ButtonKind,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    // texture: wgpu::Texture,
    texture: Arc<wgpu::Texture>,
    // sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
    sampler: Arc<wgpu::Sampler>,
    index_count: u32,
}

struct ButtonConfig {
    button_id: u32,
    position: (f32, f32),
    variant: ButtonVariant,
    kind: ButtonKind,
    // texture: wgpu::Texture,
    texture: Arc<wgpu::Texture>,
    // texture_view: wgpu::TextureView,
    // bind_group_layout: wgpu::BindGroupLayout,
    // sampler: wgpu::Sampler,
    texture_view: Arc<wgpu::TextureView>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

struct AtlasConfig {
    window_size: winit::dpi::PhysicalSize<u32>,
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 2], // x, y coordinates
    tex_coords: [f32; 2], // u, v coordinates
                        // color: [f32; 3],      // RGB color
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
                    format: wgpu::VertexFormat::Float32x2, // x2 for uv or 3 for color
                },
            ],
        }
    }
}

fn create_button(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: ButtonConfig,
    atlasConfig: AtlasConfig,
) -> Button {
    // Define vertices based on the position and size provided in config
    let (vertices, indices) = get_button_vertices_indices(
        atlasConfig.window_size,
        config.position,
        atlasConfig.width,
        atlasConfig.height,
    );

    // Create vertex and index buffers
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer {config.button_id}"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer {config.button_id}"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Texture and sampler setup should be done here (similar to earlier texture loading steps)

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &config.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&config.texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&config.sampler),
            },
        ],
        label: Some("Primary Atlas Texture Bind Group {config.button_id}"),
    });

    Button {
        position: config.position,
        variant: config.variant,
        kind: config.kind,
        vertex_buffer,
        index_buffer,
        texture: config.texture,
        sampler: config.sampler,
        bind_group,
        index_count: indices.len() as u32,
    }
}

fn get_button_vertices_indices(
    size: winit::dpi::PhysicalSize<u32>,
    position: (f32, f32), // Position relative to the top-left of the viewport
    atlas_width: u32,
    atlas_height: u32,
) -> ([Vertex; 4], [u16; 6]) {
    let uv_x = 26.0 / atlas_width as f32;
    let uv_y = 0.0 / atlas_height as f32;
    let uv_width = 80.0 / atlas_width as f32;
    let uv_height = 25.0 / atlas_height as f32;

    let rect_width = 80; // Rectangle width in pixels
    let rect_height = 25; // Rectangle height in pixels
    let scale_factor = 1.5; // Scaling factor for the button

    let scaled_rect_width = rect_width as f32 * scale_factor;
    let scaled_rect_height = rect_height as f32 * scale_factor;

    // Calculate width and height in NDC
    let ndc_width = scaled_rect_width / size.width as f32 * 2.0;
    let ndc_height = scaled_rect_height / size.height as f32 * 2.0;

    // Transform screen position to NDC
    let ndc_x = position.0 / size.width as f32 * 2.0 - 1.0;
    let ndc_y = -(position.1 / size.height as f32 * 2.0 - 1.0);

    let vertices = [
        Vertex {
            position: [ndc_x - ndc_width / 2.0, ndc_y + ndc_height / 2.0],
            tex_coords: [uv_x, uv_y],
        }, // Top left
        Vertex {
            position: [ndc_x + ndc_width / 2.0, ndc_y + ndc_height / 2.0],
            tex_coords: [uv_x + uv_width, uv_y],
        }, // Top right
        Vertex {
            position: [ndc_x + ndc_width / 2.0, ndc_y - ndc_height / 2.0],
            tex_coords: [uv_x + uv_width, uv_y + uv_height],
        }, // Bottom right
        Vertex {
            position: [ndc_x - ndc_width / 2.0, ndc_y - ndc_height / 2.0],
            tex_coords: [uv_x, uv_y + uv_height],
        }, // Bottom left
    ];

    let indices = [0, 1, 2, 2, 3, 0]; // Two triangles to form a quad

    (vertices, indices)
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

use image::GenericImageView; // Make sure you have the `image` crate in your Cargo.toml

async fn load_texture_from_file(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    file_path: &str,
) -> (wgpu::Texture, u32, u32) {
    let img = image::open(file_path)
        .expect("Failed to open image")
        .to_rgba8();
    let (width, height) = img.dimensions();

    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("Texture Atlas"),
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );

    let atlas_width = width / 3;
    let atlas_height = height / 3;

    (texture, atlas_width, atlas_height)
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

    let (texture, atlas_width, atlas_height) =
        load_texture_from_file(&device, &queue, "./src/textures/texture_atlas_01.png").await;

    let texture = Arc::new(texture);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let sampler = Arc::new(sampler);

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let texture_view = Arc::new(texture_view);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some("Primary Atlas Texture Bind Group Layout"),
    });

    let bind_group_layout = Arc::new(bind_group_layout);

    // Define the layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        // bind_group_layouts: &[], // No bind group layouts
        push_constant_ranges: &[],
    });

    let button_config = ButtonConfig {
        button_id: 0,
        position: (60.0, 20.0),
        variant: ButtonVariant::Green,
        kind: ButtonKind::SmallShort,
        texture: Arc::clone(&texture),
        texture_view: Arc::clone(&texture_view),
        bind_group_layout: Arc::clone(&bind_group_layout),
        sampler: Arc::clone(&sampler),
    };

    let button = create_button(
        &device,
        &queue,
        button_config,
        AtlasConfig {
            window_size: size,
            width: atlas_width,
            height: atlas_height,
        },
    );

    let button_config2 = ButtonConfig {
        button_id: 0,
        position: (260.0, 20.0),
        variant: ButtonVariant::Green,
        kind: ButtonKind::SmallShort,
        texture: Arc::clone(&texture),
        texture_view: Arc::clone(&texture_view),
        bind_group_layout: Arc::clone(&bind_group_layout),
        sampler: Arc::clone(&sampler),
    };

    let button2 = create_button(
        &device,
        &queue,
        button_config2,
        AtlasConfig {
            window_size: size,
            width: atlas_width,
            height: atlas_height,
        },
    );

    let buttons = vec![button, button2];

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
        // vertex: wgpu::VertexState {
        //     module: &shader_module_vert_primary,
        //     entry_point: "vs_main", // Entry point for vertex shader
        //     buffers: &[Vertex::desc()],
        // },
        // fragment: Some(wgpu::FragmentState {
        //     // Optional, needed for coloring
        //     module: &shader_module_frag_primary,
        //     entry_point: "fs_main", // Entry point for fragment shader
        //     targets: &[Some(wgpu::ColorTargetState {
        //         // format: surface.get_preferred_format(&adapter).unwrap(),
        //         format: swapchain_format,
        //         blend: Some(wgpu::BlendState::REPLACE),
        //         write_mask: wgpu::ColorWrites::ALL,
        //     })],
        // }),
        // primitive: wgpu::PrimitiveState {
        //     topology: wgpu::PrimitiveTopology::TriangleList,
        //     strip_index_format: None,
        //     front_face: wgpu::FrontFace::Ccw,
        //     cull_mode: Some(wgpu::Face::Back),
        //     // Setting this to false is useful for debugging
        //     unclipped_depth: false,
        //     polygon_mode: wgpu::PolygonMode::Fill,
        //     conservative: false,
        // },
        // depth_stencil: None, // Optional, configure if using depth testing
        // multisample: wgpu::MultisampleState {
        //     count: 1,
        //     mask: !0,
        //     alpha_to_coverage_enabled: false,
        // },
        vertex: wgpu::VertexState {
            module: &shader_module_vert_primary,
            entry_point: "vs_main", // name of the entry point in your vertex shader
            buffers: &[Vertex::desc()], // Make sure your Vertex::desc() matches your vertex structure
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module_frag_primary,
            entry_point: "fs_main", // name of the entry point in your fragment shader
            targets: &[Some(wgpu::ColorTargetState {
                format: swapchain_format,
                // blend: Some(wgpu::BlendState::REPLACE),
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
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
                                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                                r: 0.15,
                                                g: 0.15,
                                                b: 0.15,
                                                a: 1.0,
                                            }),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            /**rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &bind_group, &[]);
                            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            rpass.set_index_buffer(
                                index_buffer.slice(..),
                                wgpu::IndexFormat::Uint16,
                            );
                            rpass.draw_indexed(0..6, 0, 0..1);*/
                            // TODO: make dynamic
                            rpass.set_pipeline(&render_pipeline); // Set once if all buttons use the same pipeline

                            for button in &buttons {
                                // Set the vertex buffer for the current button
                                rpass.set_vertex_buffer(0, button.vertex_buffer.slice(..));

                                // Set the index buffer for the current button
                                rpass.set_index_buffer(
                                    button.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint16,
                                );

                                // Assuming each button uses the same bind group; if not, this would also be set per button
                                rpass.set_bind_group(0, &button.bind_group, &[]);

                                // Draw the button; you need to know the number of indices
                                rpass.draw_indexed(0..button.index_count, 0, 0..1);
                            }
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
