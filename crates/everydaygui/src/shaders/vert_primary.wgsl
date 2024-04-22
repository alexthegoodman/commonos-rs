// Vertex input structure definition
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

// Vertex output structure definition
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

// Vertex shader function
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // No transformation applied, assuming orthogonal projection
    out.clip_position = vec4(in.position, 0.0, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}


