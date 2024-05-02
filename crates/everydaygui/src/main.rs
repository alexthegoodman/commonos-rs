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
use image::{GrayImage, Luma};

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
    texture: Arc<wgpu::Texture>,
    bind_group: wgpu::BindGroup,
    sampler: Arc<wgpu::Sampler>,
    index_count: u32,
}

struct ButtonConfig {
    button_id: u32,
    position: (f32, f32),
    variant: ButtonVariant,
    kind: ButtonKind,
    texture: Arc<wgpu::Texture>,
    texture_view: Arc<wgpu::TextureView>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

struct AtlasConfig {
    window_size: winit::dpi::PhysicalSize<u32>,
    width: u32,
    height: u32,
}

// Define the Label struct to hold the label's rendering resources
struct Label {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    bind_group: wgpu::BindGroup,
}

struct LabelConfig {
    label_id: u32,
    position: (f32, f32),
    font_size: f32,
    texture_view: Arc<wgpu::TextureView>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

struct GlyphInfo {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

struct UVRect {
    u_min: f32,
    v_min: f32,
    u_max: f32,
    v_max: f32,
}

impl GlyphInfo {
    fn calculate_uv(&self, atlas_width: u32, atlas_height: u32) -> UVRect {
        UVRect {
            u_min: self.x as f32 / atlas_width as f32,
            v_min: self.y as f32 / atlas_height as f32,
            u_max: (self.x + self.width) as f32 / atlas_width as f32,
            v_max: (self.y + self.height) as f32 / atlas_height as f32,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 2],   // x, y coordinates
    tex_coords: [f32; 2], // u, v coordinates
    // color: [f32; 3],      // RGB color
    color: wgpu::Color, // RGBA color
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
                    format: wgpu::VertexFormat::Float32x2, // x2 for uv or 3 or 4 for color
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2, // Corresponds to layout(location = 2) in shader
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

fn create_label(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    glyph_infos: &HashMap<char, GlyphInfo>,
    text: &str,
    font: &Font,
    label_config: LabelConfig,
    atlas_config: AtlasConfig,
    // color: wgpu::Color,
) -> Label {
    let color = wgpu::Color {
        r: 0.5,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    // Create a layout for your text
    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.append(
        &[font],
        &TextStyle::new(text, label_config.font_size, 0),
        // &LayoutSettings {
        //     ..Default::default()
        // },
    );

    // Generate vertices for each glyph in the layout
    let mut vertices = Vec::new();
    let mut indices: Vec<
        u16, // note: changed from Vec<u32> to Vec<u16> to match the index buffer type
    > = Vec::new();
    // let mut indices: [u16; 6]; // note: hardcode for now
    let mut index_offset = 0;

    let mut i = 0;
    for (glyph) in layout.glyphs() {
        // note: collects info from A-Z, a-z, 0-9, not supplied label text
        let glyph_info = glyph_infos.get(&glyph.parent).unwrap();
        let metrics = font.metrics(glyph.parent, label_config.font_size);
        let glyph_width = metrics.advance_width;
        // let glyph_height = metrics.advance_height; // note: is 0
        // let glyph_height = label_config.font_size;
        let glyph_height = glyph_info.height as f32;

        println!(
            "{:?} x: {:?} y: {:?} width: {:?} height: {:?}",
            glyph.parent, glyph.x, glyph.y, glyph_width, glyph_height
        );

        // Calculate the bottom-left corner of the glyph
        let x0 = label_config.position.0 + glyph.x;
        let y0 = label_config.position.1 + glyph.y - glyph_height;

        // Calculate positions of the rectangle corners
        let positions = [
            [x0, y0 + glyph_height],
            [x0 + glyph_width, y0 + glyph_height],
            [x0 + glyph_width, y0],
            [x0, y0],
        ];
        // hardcode positions
        // let positions = [
        //     [100.0, 100.0],
        //     [200.0, 100.0],
        //     [200.0, 200.0],
        //     [100.0, 200.0],
        // ];

        println!("Positions: {:?}", positions);

        // Convert screen coordinates to NDC
        let ndc_positions: Vec<[f32; 2]> = positions
            .iter()
            .map(|&pos| {
                [
                    2.0 * pos[0] / atlas_config.window_size.width as f32 - 1.0,
                    -(2.0 * (pos[1] as f32) / atlas_config.window_size.height as f32 - 1.0),
                ]
            })
            .collect();

        println!("NDC Positions: {:?}", ndc_positions);

        // Create vertices for this glyph, adding color and texture coordinates
        let uv_rect = glyph_info.calculate_uv(atlas_config.width, atlas_config.height);

        // Corresponding UV coordinates for each vertex of the quad
        let uvs = [
            [uv_rect.u_min, uv_rect.v_max], // Top-left
            [uv_rect.u_max, uv_rect.v_max], // Top-right
            [uv_rect.u_max, uv_rect.v_min], // Bottom-right
            [uv_rect.u_min, uv_rect.v_min], // Bottom-left
        ];
        // // hardcode uvs
        // let uvs = [
        //     [0.0, 0.0], // Top-left
        //     [0.5, 0.0], // Top-right
        //     [0.5, 0.5], // Bottom-right
        //     [0.0, 0.5], // Bottom-left
        // ];

        // Extend vertices with position and corresponding UV coordinates
        vertices.extend(
            ndc_positions
                .iter()
                .zip(uvs.iter())
                .map(|(&pos, &uv)| Vertex {
                    position: pos,
                    tex_coords: uv,
                    color,
                }),
        );

        // Define two triangles for the rectangle
        // println!("Index Offset: {:?}", index_offset);
        indices.extend([
            index_offset,     // Top-left
            index_offset + 1, // Top-right
            index_offset + 2, // Bottom-right
            index_offset + 2, // Bottom-right
            index_offset + 3, // Bottom-left
            index_offset,     // Top-left
        ]);
        index_offset += 4;

        i += 1;
    }

    // hardcode for debugging
    // indices = [0, 1, 2, 2, 3, 0];

    println!("Vertices: {:?}", vertices);
    println!("Indices: {:?} {:?}", indices, indices.len());

    // Create vertex and index buffers
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Label Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Label Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &label_config.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&label_config.texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&label_config.sampler),
            },
        ],
        label: Some("Font Atlas Texture Bind Group {config.label_id}"),
    });

    Label {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
        bind_group,
    }
}

async fn create_font_texture_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    font: &Font,
    font_size: f32,
    characters: &str,
) -> (wgpu::Texture, HashMap<char, GlyphInfo>) {
    // Calculate necessary texture size (simplified)
    let atlas_width = 1024;
    let atlas_height = 1024;
    let mut texture_data = vec![0u8; (atlas_width * atlas_height) as usize];

    let mut x_cursor = 0;
    let mut y_cursor = 0;
    let mut row_height = 0;

    let mut glyph_infos = HashMap::new();

    for ch in characters.chars() {
        let (metrics, bitmap) = font.rasterize(ch, font_size);
        if x_cursor + metrics.width as u32 > atlas_width {
            x_cursor = 0;
            y_cursor += row_height;
            row_height = 0;
        }

        if y_cursor + metrics.height as u32 > atlas_height {
            // Handle texture overflow, increase texture size or use multiple textures
            break;
        }

        // Copy bitmap into the texture data array
        for (i, &byte) in bitmap.iter().enumerate() {
            let x = x_cursor + (i % metrics.width) as u32;
            let y = y_cursor + (i / metrics.width) as u32;
            texture_data[(y * atlas_width + x) as usize] = byte;
        }

        glyph_infos.insert(
            ch,
            GlyphInfo {
                x: x_cursor,
                y: y_cursor,
                width: metrics.width as u32,
                height: metrics.height as u32,
            },
        );

        x_cursor += metrics.width as u32;
        row_height = row_height.max(metrics.height as u32);
    }

    // Create the wgpu texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: atlas_width,
            height: atlas_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        label: Some("Font Texture Atlas {font_size}"),
        view_formats: &[],
    });

    // for debugging: save the texture to a file
    // let img = GrayImage::from_raw(1024, 1024, texture_data)
    //     .expect("Failed to create image from raw data");
    // img.save("font_atlas.png").expect("Failed to save image");

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texture_data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(atlas_width),
            rows_per_image: Some(atlas_height),
        },
        wgpu::Extent3d {
            width: atlas_width,
            height: atlas_height,
            depth_or_array_layers: 1,
        },
    );

    (texture, glyph_infos)
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
    let defaultColor = wgpu::Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    let uv_x = 26.0 / atlas_width as f32;
    let uv_y = 0.0 / atlas_height as f32;
    let uv_width = 80.0 / atlas_width as f32;
    let uv_height = 25.0 / atlas_height as f32;

    println!("UV: {:?} {:?} {:?} {:?}", uv_x, uv_y, uv_width, uv_height);

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
            color: defaultColor,
        }, // Top left
        Vertex {
            position: [ndc_x + ndc_width / 2.0, ndc_y + ndc_height / 2.0],
            tex_coords: [uv_x + uv_width, uv_y],
            color: defaultColor,
        }, // Top right
        Vertex {
            position: [ndc_x + ndc_width / 2.0, ndc_y - ndc_height / 2.0],
            tex_coords: [uv_x + uv_width, uv_y + uv_height],
            color: defaultColor,
        }, // Bottom right
        Vertex {
            position: [ndc_x - ndc_width / 2.0, ndc_y - ndc_height / 2.0],
            tex_coords: [uv_x, uv_y + uv_height],
            color: defaultColor,
        }, // Bottom left
    ];

    let indices = [0, 1, 2, 2, 3, 0]; // Two triangles to form a quad

    println!("Background Vertices: {:?}", vertices);
    println!("Background Indices: {:?}", indices);

    (vertices, indices)
}

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
        36.0,
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

    // let font_bind_group_layout =
    //     device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    //         entries: &[
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 0,
    //                 visibility: wgpu::ShaderStages::FRAGMENT,
    //                 ty: wgpu::BindingType::Texture {
    //                     multisampled: false,
    //                     view_dimension: wgpu::TextureViewDimension::D2,
    //                     sample_type: wgpu::TextureSampleType::Float { filterable: true },
    //                 },
    //                 count: None,
    //             },
    //             wgpu::BindGroupLayoutEntry {
    //                 binding: 1,
    //                 visibility: wgpu::ShaderStages::FRAGMENT,
    //                 ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    //                 count: None,
    //             },
    //         ],
    //         label: Some("Font Atlas Texture Bind Group Layout"),
    //     });

    // let font_bind_group_layout: Arc<wgpu::BindGroupLayout> = Arc::new(font_bind_group_layout);

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

    // create the label
    let label = create_label(
        &device,
        &queue,
        &glyph_infos,
        "Get Started",
        &font,
        LabelConfig {
            label_id: 0,
            position: (100.0, 100.0),
            font_size: 36.0,
            texture_view: Arc::clone(&font_texture_view),
            bind_group_layout: Arc::clone(&bind_group_layout),
            sampler: Arc::clone(&font_sampler),
        },
        AtlasConfig {
            window_size: size,
            width: 1024, // TODO: dynamic or dry
            height: 1024,
        },
    );

    let labels = vec![label];

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
