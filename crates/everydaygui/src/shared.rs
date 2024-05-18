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

pub enum ButtonVariant {
    Green,
    Dark,
    Light,
}

pub enum ButtonKind {
    SmallIcon,
    SmallShort,
    SmallWide,
    LargeIcon,
    LargeShort,
    LargeWide,
}

pub struct Button {
    pub position: (f32, f32, f32),
    pub world_position: (f32, f32),
    pub variant: ButtonVariant,
    pub kind: ButtonKind,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub texture: Arc<wgpu::Texture>,
    pub bind_group: wgpu::BindGroup,
    pub sampler: Arc<wgpu::Sampler>,
    pub index_count: u32,
    pub size: (f32, f32),
    pub world_size: (f32, f32),
    pub on_click: Box<dyn Fn()>,
}

pub struct ButtonConfig {
    pub button_id: u32,
    pub position: (f32, f32, f32),
    pub variant: ButtonVariant,
    pub kind: ButtonKind,
    pub texture: Arc<wgpu::Texture>,
    pub texture_view: Arc<wgpu::TextureView>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub sampler: Arc<wgpu::Sampler>,
    pub render_mode_buffer: Arc<wgpu::Buffer>,
    pub on_click: Box<dyn Fn()>,
}

pub struct AtlasConfig {
    pub window_size: winit::dpi::PhysicalSize<u32>,
    pub width: u32,
    pub height: u32,
}

// Define the Label struct to hold the label's rendering resources
pub struct Label {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub bind_group: wgpu::BindGroup,
    pub size: (f32, f32),
    pub world_size: (f32, f32),
    pub position: (f32, f32, f32),
    pub world_position: (f32, f32),
    pub on_click: Box<dyn Fn()>,
}

pub struct LabelConfig {
    pub label_id: u32,
    pub position: (f32, f32, f32),
    pub font_size: f32,
    pub texture_view: Arc<wgpu::TextureView>,
    pub bind_group_layout: Arc<wgpu::BindGroupLayout>,
    pub sampler: Arc<wgpu::Sampler>,
    pub render_mode_buffer: Arc<wgpu::Buffer>,
    pub on_click: Box<dyn Fn()>,
}

pub struct GlyphInfo {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct UVRect {
    pub u_min: f32,
    pub v_min: f32,
    pub u_max: f32,
    pub v_max: f32,
}

impl GlyphInfo {
    pub fn calculate_uv(&self, atlas_width: u32, atlas_height: u32) -> UVRect {
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
pub struct Vertex {
    pub position: [f32; 3],   // x, y, z coordinates
    pub tex_coords: [f32; 2], // u, v coordinates
    // color: [f32; 3],      // RGB color
    pub color: wgpu::Color, // RGBA color
}

// Ensure Vertex is Pod and Zeroable
unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0, // Corresponds to layout(location = 0) in shader
                    format: wgpu::VertexFormat::Float32x3, // x3 for position
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1, // Corresponds to layout(location = 1) in shader
                    format: wgpu::VertexFormat::Float32x2, // x2 for uv
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2, // Corresponds to layout(location = 2) in shader
                    format: wgpu::VertexFormat::Float32x4, // x4 for color
                },
            ],
        }
    }
}

pub async fn create_font_texture_atlas(
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

pub async fn load_texture_from_file(
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

pub fn screen_to_world(screen_pos: (f64, f64), window_size: (f64, f64)) -> (f64, f64) {
    let (sx, sy) = screen_pos;
    let (ww, wh) = window_size;

    // Convert from screen space to normalized device coordinates (NDC)
    let x = (sx / ww) * 2.0 - 1.0;
    let y = 1.0 - (sy / wh) * 2.0; // Flip y-axis

    // Assuming a simple orthographic projection where NDC directly maps to world coordinates
    (x, y)
}

pub fn point_in_rect(point: (f64, f64), rect_pos: (f32, f32), rect_size: (f32, f32)) -> bool {
    let (px, py) = point;
    let (rx, ry) = rect_pos;
    let (rw, rh) = rect_size;

    // cast
    let rx = rx as f64;
    let ry = ry as f64;
    let rw = rw as f64;
    let rh = rh as f64;

    px >= rx && px <= rx + rw && py >= ry && py <= ry + rh
}
