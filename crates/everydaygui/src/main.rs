use wgpu::util::DeviceExt;
use wgpu::Face;
use wgpu::FrontFace;
use wgpu::PrimitiveTopology;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use std::sync::Arc;

use fontdue::layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle};
use fontdue::Font;

use everydaygui::button::create_button;
use everydaygui::label::create_label;
use everydaygui::shared::{
    create_font_texture_atlas, load_texture_from_file, point_in_rect, screen_to_world, AtlasConfig,
    Button, ButtonConfig, ButtonKind, ButtonVariant, Label, LabelConfig, Vertex,
};

fn handle_click(
    window_size: (f64, f64),
    mouse_pos: (f64, f64),
    buttons: &[Button],
    labels: &[Label],
) {
    // let window_size = (800.0, 600.0); // Replace with your actual window size
    let world_pos = screen_to_world(mouse_pos, window_size);

    println!("World Position: {:?}", world_pos);

    for button in buttons {
        if point_in_rect(world_pos, button.world_position, button.world_size) {
            (button.on_click)();
        }
    }

    for label in labels {
        if point_in_rect(world_pos, label.world_position, label.world_size) {
            (label.on_click)();
        }
    }
}

async fn initialize_core(event_loop: EventLoop<()>, window: Window) {
    let mut size = window.inner_size();
    size.width = size.width.max(1);
    size.height = size.height.max(1);

    println!("Window Size: {:?}", size);

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

    // Load the font
    let font_data = include_bytes!("./fonts/inter/Inter-Regular.ttf") as &[u8];
    let font =
        Font::from_bytes(font_data, fontdue::FontSettings::default()).expect("Couldn't load font");

    // Create a texture atlas for the font
    let (font_texture, glyph_infos) = create_font_texture_atlas(
        &device,
        &queue,
        &font,
        16.0,
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    )
    .await;

    let font_texture = Arc::new(font_texture);

    let font_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let font_sampler = Arc::new(font_sampler);

    let font_texture_view = font_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let font_texture_view = Arc::new(font_texture_view);

    // note: this path is correct when running from within this crate
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
        label: Some("Primary Atlas Texture Bind Group Layout"),
    });

    let bind_group_layout = Arc::new(bind_group_layout);

    // create renderMode uniform for button backgrounds
    let render_mode_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Background Render Mode Buffer"),
        contents: bytemuck::cast_slice(&[0i32]), // Default to normal mode
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let render_mode_buffer = Arc::new(render_mode_buffer);

    // Create a buffer for the renderMode uniform
    let text_render_mode_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Text Render Mode Buffer"),
        contents: bytemuck::cast_slice(&[1i32]), // Default to text mode
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let text_render_mode_buffer = Arc::new(text_render_mode_buffer);

    // Define the layouts
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        // bind_group_layouts: &[], // No bind group layouts
        push_constant_ranges: &[],
    });

    // create the label
    let label = create_label(
        &device,
        &queue,
        &glyph_infos,
        "Get Started",
        &font,
        LabelConfig {
            label_id: 0,
            position: (12.0, 9.0, 0.01),
            font_size: 16.0,
            texture_view: Arc::clone(&font_texture_view),
            bind_group_layout: Arc::clone(&bind_group_layout),
            sampler: Arc::clone(&font_sampler),
            render_mode_buffer: Arc::clone(&text_render_mode_buffer),
            on_click: Box::new(|| {
                println!("Get Started clicked (label)!");
            }),
        },
        AtlasConfig {
            window_size: size,
            width: 1024, // TODO: dynamic or dry
            height: 1024,
        },
    );

    let label2 = create_label(
        &device,
        &queue,
        &glyph_infos,
        "Export MP4",
        &font,
        LabelConfig {
            label_id: 0,
            position: (216.0, 9.0, 0.01),
            font_size: 16.0,
            texture_view: Arc::clone(&font_texture_view),
            bind_group_layout: Arc::clone(&bind_group_layout),
            sampler: Arc::clone(&font_sampler),
            render_mode_buffer: Arc::clone(&text_render_mode_buffer),
            on_click: Box::new(|| {
                println!("Export MP4 clicked (label)!");
            }),
        },
        AtlasConfig {
            window_size: size,
            width: 1024, // TODO: dynamic or dry
            height: 1024,
        },
    );

    let labels = vec![label, label2];

    let button_config = ButtonConfig {
        button_id: 0,
        position: (60.0, 20.0, 0.02),
        variant: ButtonVariant::Green,
        kind: ButtonKind::SmallShort,
        texture: Arc::clone(&texture),
        texture_view: Arc::clone(&texture_view),
        bind_group_layout: Arc::clone(&bind_group_layout),
        sampler: Arc::clone(&sampler),
        render_mode_buffer: Arc::clone(&render_mode_buffer),
        on_click: Box::new(|| {
            println!("Get Started clicked (back)!");
        }),
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
        position: (260.0, 20.0, 0.02),
        variant: ButtonVariant::Green,
        kind: ButtonKind::SmallShort,
        texture: Arc::clone(&texture),
        texture_view: Arc::clone(&texture_view),
        bind_group_layout: Arc::clone(&bind_group_layout),
        sampler: Arc::clone(&sampler),
        render_mode_buffer: Arc::clone(&render_mode_buffer),
        on_click: Box::new(|| {
            println!("Export MP4 clicked (back)!");
        }),
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

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: window.inner_size().width,
            height: window.inner_size().height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth24Plus,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        label: Some("Depth Texture"),
        view_formats: &[],
    });

    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_stencil_state = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    };

    // Configure the render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("EverydayGUI Primary Render Pipeline"),
        layout: Some(&pipeline_layout),
        multiview: None,
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
        // primitive: wgpu::PrimitiveState::default(),
        // depth_stencil: None,
        // multisample: wgpu::MultisampleState::default(),
        primitive: wgpu::PrimitiveState {
            conservative: false,
            topology: PrimitiveTopology::TriangleList, // how vertices are assembled into geometric primitives
            strip_index_format: None,
            front_face: FrontFace::Ccw, // Counter-clockwise is considered the front face
            // none cull_mode
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            // Other properties such as conservative rasterization can be set here
            unclipped_depth: false,
        },
        depth_stencil: Some(depth_stencil_state), // Optional, only if you are using depth testing
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

    let mut mouse_position = (0.0, 0.0);

    // execute winit render loop
    let window = &window;
    event_loop
        .run(move |event, target| {
            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::CursorMoved { position, .. } => {
                        // Update the mouse position
                        // println!("Mouse Position: {:?}", position);
                        mouse_position = (position.x as f64, position.y as f64);
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } => {
                        let window_size = (size.width as f64, size.height as f64);
                        handle_click(window_size, mouse_position, &buttons, &labels);
                    }
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
                                    // depth_stencil_attachment: None,
                                    depth_stencil_attachment: Some(
                                        wgpu::RenderPassDepthStencilAttachment {
                                            view: &depth_view, // This is the depth texture view
                                            depth_ops: Some(wgpu::Operations {
                                                load: wgpu::LoadOp::Clear(1.0), // Clear to max depth
                                                store: wgpu::StoreOp::Store,
                                            }),
                                            stencil_ops: None, // Set this if using stencil
                                        },
                                    ),
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });

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

                            for label in &labels {
                                // Set the vertex buffer for the current label
                                rpass.set_vertex_buffer(0, label.vertex_buffer.slice(..));

                                // Set the index buffer for the current label
                                rpass.set_index_buffer(
                                    label.index_buffer.slice(..),
                                    wgpu::IndexFormat::Uint16,
                                );

                                // Assuming each label uses the same bind group; if not, this would also be set per label
                                rpass.set_bind_group(0, &label.bind_group, &[]);

                                // Draw the label; you need to know the number of indices
                                rpass.draw_indexed(0..label.index_count, 0, 0..1);
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

    // for debugging: print current directory
    // let current_dir = std::env::current_dir().unwrap();
    // println!("Current directory: {:?}", current_dir);

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
