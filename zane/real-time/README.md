# Kiosk Face Detection & Tracking System

Real-time face detection, tracking, and emotion recognition system for the Kiosk Teacher Assistant AI.

## Features

- **Multi-face Detection**: Detect up to 20+ faces simultaneously at 30 FPS @ 1080p
- **Person Tracking**: SORT/DeepSORT-based tracking with ID persistence
- **Face Landmarks**: 68-point facial landmark extraction
- **Emotion Recognition**: Classify emotions (happy, sad, neutral, confused, engaged)
- **Attention Tracking**: Monitor student engagement and focus levels
- **GPU Acceleration**: CUDA support for Jetson Orin Nano Tensor cores
- **Low Latency**: <50ms end-to-end processing time

## Requirements

### System Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Hardware**: NVIDIA GPU (Jetson Orin Nano recommended) or CPU fallback
- **RAM**: 2GB minimum for full system
- **Camera**: V4L2-compatible webcam (USB or MIPI CSI)

### Software Dependencies
- **OpenCV**: 4.5+ (with CUDA support for GPU acceleration)
- **CMake**: 3.18+
- **C++ Compiler**: GCC 9+ or Clang 10+
- **Python**: 3.8+ (for model training/validation)

## Installation

### Build from Source

```bash
cd zane/real-time
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libopencv-contrib-dev \
    python3-opencv
```

#### macOS
```bash
brew install opencv
```

#### Jetson Orin Nano
```bash
# OpenCV comes pre-installed with CUDA support
# Install additional packages
sudo apt-get install -y \
    build-essential \
    cmake \
    python3-opencv
```

## Usage

### Test Face Detection with Camera

```bash
./test_face_detection --camera --device /dev/video0
```

### Test without Camera (Simulation Mode)

```bash
./test_face_detection
```

### Test Camera Capture Only

```bash
./test_camera /dev/video0
```

## Configuration

Edit the configuration in `face_detection.h` or pass parameters programmatically:

```cpp
FaceDetectionConfig config;
config.confidence_threshold = 0.7f;      // Detection confidence threshold
config.nms_threshold = 0.4f;              // Non-maximum suppression
config.input_size = 640;                  // Model input size
config.use_gpu = true;                    // Enable CUDA
config.max_distance = 0.7f;               // Tracking distance threshold
config.max_age = 30;                      // Track age limit
config.enable_landmarks = true;           // Enable landmark extraction
config.enable_emotion = true;             // Enable emotion recognition
config.enable_attention = true;           // Enable attention tracking
```

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| Detection FPS | 30 FPS @ 1080p | ⚠️ Needs validation |
| End-to-end Latency | <50ms | ⚠️ Needs validation |
| Detection Accuracy | >95% | ⚠️ Needs validation |
| Tracking ID Retention | >99% over 60s | ⚠️ Needs validation |
| GPU Utilization | <70% | ⚠️ Needs validation |
| Memory Usage | <2GB | ⚠️ Needs validation |

## Architecture

### Core Components

1. **Camera Capture** (`camera_capture.cpp/h`)
   - V4L2-based camera interface
   - Zero-copy buffer management
   - Multi-camera support
   - Performance statistics

2. **Face Detection** (`face_detection.cpp/h`)
   - Real-time face detection
   - Multi-person tracking (SORT/DeepSORT)
   - Landmark extraction
   - Emotion recognition
   - Attention tracking

### Data Flow

```
Camera → Face Detection → Landmarks → Emotion → Tracking → Output
                ↓
            Analytics Buffer
                ↓
            Event Generation
```

## Model Integration

### Production Models (To be integrated)

1. **Face Detection**: RetinaFace or MTCNN (TensorRT optimized)
2. **Landmarks**: MediaPipe Face Mesh
3. **Emotion Recognition**: Custom CNN trained on classroom data
4. **Tracking**: DeepSORT with appearance features

### Current Implementation

- **Face Detection**: OpenCV Haar Cascade (fallback for development)
- **Landmarks**: Placeholder (68-point grid)
- **Emotion Recognition**: Rule-based heuristics
- **Tracking**: Simplified SORT algorithm

## Next Steps

### Immediate (Issue #2)
- [ ] Integrate real face detection model (RetinaFace/MTCNN)
- [ ] Implement accurate landmark extraction (MediaPipe)
- [ ] Train emotion recognition model
- [ ] Optimize for TensorRT on Jetson Orin Nano
- [ ] Performance benchmarking
- [ ] End-to-end testing

### Follow-up (Issue #3)
- [ ] GPU acceleration pipeline
- [ ] Zero-copy memory optimization
- [ ] Multi-threading for parallel processing

### Follow-up (Issue #4)
- [ ] Performance benchmarking suite
- [ ] Baseline measurements
- [ ] Optimization iterations

## Troubleshooting

### Camera Not Detected
```bash
# Check available cameras
v4l2-ctl --list-devices

# Check camera capabilities
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

### OpenCV Not Found
```bash
# Check OpenCV installation
pkg-config --modversion opencv4

# Find OpenCV path
pkg-config --cflags opencv4
```

### CUDA Errors
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version
```

## Development

### Code Style
- C++17 standard
- 4-space indentation
- Snake_case for variables/functions
- PascalCase for classes
- Google-style comments

### Testing
```bash
# Build and run tests
mkdir build && cd build
cmake -DBUILD_TESTS=ON ..
make
ctest
```

## License

Champion Robotics Pty Ltd - Internal Use Only

## Author

**Zane** - C++/Computer Vision Engineer
- Email: zane@championrobotics.com.au
- Project: Kiosk MVP - Teacher Assistant AI

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [DeepSORT](https://arxiv.org/abs/1703.07402)
- [RetinaFace](https://arxiv.org/abs/1905.00641)
- [Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)

---

**Issue**: #2 - [CV] Implement real-time face detection and tracking system
**Status**: 🟡 In Progress (Core implementation complete, model integration pending)
**Last Updated**: 2026-03-13
