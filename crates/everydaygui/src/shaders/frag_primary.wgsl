// Group binding for texture and sampler
@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

struct FragmentInput {
    @location(0) tex_coords: vec2<f32>,
    @location(1) color: vec4<f32>,  // Receive color from vertex shader
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    let textureColor = textureSample(myTexture, mySampler, in.tex_coords);
    return textureColor;  // Multiply texture color by vertex color
}

