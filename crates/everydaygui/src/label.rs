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

use crate::shared::{AtlasConfig, GlyphInfo, Label, LabelConfig, Vertex};

pub fn create_label(
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
        let y0 = label_config.position.1 + glyph.y;
        let z0 = label_config.position.2;

        // Calculate positions of the rectangle corners
        let positions = [
            [x0, y0 + glyph_height, z0],
            [x0 + glyph_width, y0 + glyph_height, z0],
            [x0 + glyph_width, y0, z0],
            [x0, y0, z0],
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
        let ndc_positions: Vec<[f32; 3]> = positions
            .iter()
            .map(|&pos| {
                [
                    2.0 * pos[0] / atlas_config.window_size.width as f32 - 1.0,
                    -(2.0 * (pos[1] as f32) / atlas_config.window_size.height as f32 - 1.0),
                    z0, // note: added z coordinate
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &label_config.render_mode_buffer,
                    offset: 0,
                    size: None,
                }),
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
