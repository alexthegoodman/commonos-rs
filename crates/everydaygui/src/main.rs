use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    print!("Welcome to EverydayGUI!");
    let event_loop = EventLoop::new().expect("Failed to create an event loop");
    //let event_loop = winit::event_loop::EventLoopBuilder::<Event>::with_user_event().build();
    let window = WindowBuilder::new()
        .with_title("EverydayGUI Demo Application")
        .with_resizable(true)
        .with_transparent(false)
        .build(&event_loop)
        .unwrap();

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
