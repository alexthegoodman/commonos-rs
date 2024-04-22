// Fragment input structure definition
struct FragmentInput {
    @location(0) tex_coords: vec2<f32>,
};

// Group binding for texture and sampler
@group(0) @binding(0) var myTexture: texture_2d<f32>;
@group(0) @binding(1) var mySampler: sampler;

// Fragment shader function
@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return textureSample(myTexture, mySampler, in.tex_coords);
}

