use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

async fn initialize_core(window: Window) {
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

    initialize_core(window);

    // execute winit render loop
    event_loop
        .run(move |event, _event_loop_window_target| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => {
                    // Implement logic to shut down gracefully
                    std::process::exit(0); // Exit immediately for simplicity
                }
                // Handle other events here
                _ => {}
            }
        })
        .expect("EverydayGUI failed to run");
}
