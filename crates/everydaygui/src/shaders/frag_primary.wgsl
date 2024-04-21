struct FragmentInput {
    @location(0) color: vec3<f32>,
};

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0); // Output color with full opacity
}
