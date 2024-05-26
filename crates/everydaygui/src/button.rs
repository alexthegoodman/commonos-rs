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

use crate::shared::{
    create_font_texture_atlas, load_texture_from_file, AtlasConfig, Button, ButtonConfig,
    ButtonKind, ButtonVariant, Label, LabelConfig, Vertex,
};

pub fn create_button(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    config: ButtonConfig,
    atlasConfig: AtlasConfig,
) -> Button {
    // Define vertices based on the position and size provided in config
    let (vertices, indices, button_size, world_position, world_size) = get_button_vertices_indices(
        atlasConfig.window_size,
        config.position,
        &config.variant,
        &config.kind,
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &config.render_mode_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
        label: Some("Primary Atlas Texture Bind Group {config.button_id}"),
    });

    Button {
        position: config.position,
        world_position: world_position,
        variant: config.variant,
        kind: config.kind,
        vertex_buffer,
        index_buffer,
        texture: config.texture,
        sampler: config.sampler,
        bind_group,
        index_count: indices.len() as u32,
        size: button_size,
        world_size: world_size,
        on_click: config.on_click,
    }
}

// dimensions and positioning based on Variant and Kind
fn get_button_size_and_position(
    variant: &ButtonVariant,
    kind: &ButtonKind,
) -> (f32, f32, f32, f32) {
    match variant {
        ButtonVariant::Green => match kind {
            ButtonKind::ThinIcon => {
                let (width, height) = (15.0, 15.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::ThinShort => {
                let (width, height) = (60.0, 15.0);
                let (x, y) = (16.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::ThinWide => {
                let (width, height) = (80.0, 15.0);
                let (x, y) = (86.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::SmallIcon => {
                let (width, height) = (25.0, 25.0);
                let (x, y) = (0.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::SmallShort => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::SmallWide => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeIcon => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeShort => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeWide => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::MediumShadow => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
        },
        ButtonVariant::Light => match kind {
            ButtonKind::ThinIcon => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::ThinShort => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::ThinWide => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::SmallIcon => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::SmallShort => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::SmallWide => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeIcon => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeShort => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::LargeWide => {
                let (width, height) = (80.0, 25.0);
                let (x, y) = (26.0, 16.0);
                (width, height, x, y)
            }
            ButtonKind::MediumShadow => {
                let (width, height) = (75.0, 75.0);
                let (x, y) = (24.0, 78.0);
                (width, height, x, y)
            }
        },
        ButtonVariant::Dark => match kind {
            ButtonKind::ThinIcon => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::ThinShort => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::ThinWide => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::SmallIcon => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::SmallShort => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::SmallWide => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::LargeIcon => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::LargeShort => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::LargeWide => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
            ButtonKind::MediumShadow => {
                let (width, height) = (0.0, 0.0);
                let (x, y) = (0.0, 0.0);
                (width, height, x, y)
            }
        },
    }
}

pub fn get_button_vertices_indices(
    size: winit::dpi::PhysicalSize<u32>,
    position: (f32, f32, f32), // Position relative to the top-left of the viewport
    variant: &ButtonVariant,
    kind: &ButtonKind,
    atlas_width: u32,
    atlas_height: u32,
) -> ([Vertex; 4], [u16; 6], (f32, f32), (f32, f32), (f32, f32)) {
    let defaultColor = wgpu::Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };

    // Define the button size and UV coordinates
    let (width, height, x, y) = get_button_size_and_position(variant, kind);

    let button_size = (width, height); // Button size in pixels
    let uv_x = x / atlas_width as f32;
    let uv_y = y / atlas_height as f32;
    let uv_width = width / atlas_width as f32;
    let uv_height = height / atlas_height as f32;

    println!("UV: {:?} {:?} {:?} {:?}", uv_x, uv_y, uv_width, uv_height);

    let rect_width = width; // Rectangle width in pixels
    let rect_height = height; // Rectangle height in pixels
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
            position: [
                ndc_x - ndc_width / 2.0,
                ndc_y + ndc_height / 2.0,
                position.2, // note: added z coordinate
            ],
            tex_coords: [uv_x, uv_y],
            color: defaultColor,
        }, // Top left
        Vertex {
            position: [
                ndc_x + ndc_width / 2.0,
                ndc_y + ndc_height / 2.0,
                position.2, // note: added z coordinate
            ],
            tex_coords: [uv_x + uv_width, uv_y],
            color: defaultColor,
        }, // Top right
        Vertex {
            position: [
                ndc_x + ndc_width / 2.0,
                ndc_y - ndc_height / 2.0,
                position.2, // note: added z coordinate
            ],
            tex_coords: [uv_x + uv_width, uv_y + uv_height],
            color: defaultColor,
        }, // Bottom right
        Vertex {
            position: [
                ndc_x - ndc_width / 2.0,
                ndc_y - ndc_height / 2.0,
                position.2, // note: added z coordinate
            ],
            tex_coords: [uv_x, uv_y + uv_height],
            color: defaultColor,
        }, // Bottom left
    ];

    let indices = [0, 1, 2, 2, 3, 0]; // Two triangles to form a quad

    println!("Background Vertices: {:?}", vertices);
    println!("Background Indices: {:?}", indices);

    let ndc_bottom_left = (ndc_x - ndc_width / 2.0, ndc_y - ndc_height / 2.0);

    // let world_position = (ndc_x, ndc_y);

    let world_size = (ndc_width, ndc_height);

    (vertices, indices, button_size, ndc_bottom_left, world_size)
}
