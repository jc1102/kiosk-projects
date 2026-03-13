# Camera Capture System for Jetson Orin Nano

## Overview

High-performance camera capture system designed for the Kiosk CV application on Jetson Orin Nano. Features V4L2 driver integration, zero-copy CUDA buffers, and multi-camera synchronization.

## Features

- ✅ V4L2 driver integration for USB webcams
- ✅ 1080p @ 60fps capture support
- ✅ Zero-copy buffer allocation for GPU processing
- ✅ Multi-camera synchronization (up to 3 cameras)
- ✅ Camera parameter control (exposure, focus, white balance, brightness, contrast, saturation)
- ✅ Hardware timestamping
- ✅ Performance monitoring and statistics
- ✅ Dynamic reconfiguration support

## Requirements

### Hardware
- Jetson Orin Nano (8GB)
- USB webcam(s) with V4L2 support
- USB 3.0 for full 1080p @ 60fps

### Software
- JetPack SDK 5.x+
- OpenCV 4.x with CUDA support
- CUDA Toolkit 11.x+
- CMake 3.18+

## Building

```bash
cd /path/to/kiosk-projects/zane/real-time
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Usage

### Single Camera

```cpp
#include "camera_capture.h"

using namespace kiosk::cv;

// Create configuration
CameraConfig config;
config.device_path = "/dev/video0";
config.resolution = cv::Size(1920, 1080);
config.target_fps = 60.0;
config.zero_copy = true;
config.hardware_timestamp = true;
config.brightness = 128;
config.contrast = 128;
config.saturation = 128;

// Initialize camera
CameraCapture camera;
if (!camera.initialize(config)) {
    // Handle error
    return -1;
}

// Set callback for new frames
camera.setFrameCallback([](const CameraFrame& frame) {
    // Process frame here
    std::cout << "Frame " << frame.sequence_number << std::endl;
});

// Start capture
if (!camera.start()) {
    // Handle error
    return -1;
}

// ... main loop ...

// Stop capture
camera.stop();
```

### Multi-Camera

```cpp
#include "camera_capture.h"

using namespace kiosk::cv;

// Create multi-camera manager
MultiCameraCapture manager;

// Add cameras
for (int i = 0; i < 3; i++) {
    CameraConfig config;
    config.camera_id = i;
    config.device_path = "/dev/video" + std::to_string(i);
    config.resolution = cv::Size(1920, 1080);
    config.target_fps = 60.0;
    config.zero_copy = true;

    if (manager.addCamera(config) == -1) {
        // Handle error
    }
}

// Start all cameras
if (!manager.startAll()) {
    // Handle error
}

// ... main loop ...

// Stop all cameras
manager.stopAll();
```

## Running Tests

```bash
# Single camera test (default device: /dev/video0)
./camera_test

# Specify device
./camera_test /dev/video1
```

## Performance

Target performance metrics:
- **Resolution**: 1920x1080 @ 60fps
- **Capture latency**: <10ms
- **Frame drops**: 0 under normal load
- **Memory usage**: Optimized with zero-copy buffers

## Camera Configuration Options

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `camera_id` | int | 0+ | 0 | Camera identifier |
| `device_path` | string | - | /dev/video0 | V4L2 device path |
| `resolution` | Size | - | 1920x1080 | Frame resolution |
| `target_fps` | double | 1-120 | 60.0 | Target frame rate |
| `zero_copy` | bool | true/false | true | Enable zero-copy buffers |
| `hardware_timestamp` | bool | true/false | true | Enable hardware timestamping |
| `exposure` | int | -1 or 1-10000 | -1 | Exposure (-1 = auto) |
| `focus` | int | -1 or 0-255 | -1 | Focus (-1 = auto) |
| `white_balance` | int | -1 or 2000-10000 | -1 | White balance (-1 = auto) |
| `brightness` | int | 0-255 | 128 | Brightness |
| `contrast` | int | 0-255 | 128 | Contrast |
| `saturation` | int | 0-255 | 128 | Saturation |

## Statistics

The system provides real-time statistics:
- Frames captured
- Frames dropped
- Actual FPS
- Average capture latency (ms)
- Maximum capture latency (ms)

Access via `camera.getStats()` or `manager.getCombinedStats()`.

## Architecture

```
┌─────────────────┐
│  Camera Capture │
│   (V4L2)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Zero-Copy      │
│  CUDA Buffer    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Frame Callback │
│  (User Code)    │
└─────────────────┘
```

## Troubleshooting

### Camera not opening
- Check device path: `ls /dev/video*`
- Verify permissions: `sudo chmod 666 /dev/video0`
- Check if another process is using the camera

### Low frame rate
- Use USB 3.0 port
- Check USB bandwidth: `lsusb -t`
- Reduce resolution or frame rate

### High latency
- Ensure zero-copy is enabled
- Check GPU utilization: `tegrastats`
- Reduce processing load in callback

## Dependencies

- OpenCV with CUDA support
- V4L2 drivers (usually pre-installed on Jetson)
- CUDA Runtime API

## Next Steps

- Integration with preprocessing pipeline
- Face detection implementation
- Real-time processing optimization
- Performance benchmarking

## License

Part of the Kiosk Project - Champion Robotics Pty Ltd
