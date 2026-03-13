/**
 * @file camera_capture.cpp
 * @brief Implementation of camera capture system for Jetson Orin Nano
 */

#include "camera_capture.h"
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace kiosk {
namespace cv {

// ============================================================================
// CameraCapture Implementation
// ============================================================================

CameraCapture::CameraCapture()
    : running_(false)
    , initialized_(false)
    , current_buffer_index_(0)
    , sequence_number_(0) {

    // Initialize stats
    stats_ = {
        .frames_captured = 0,
        .frames_dropped = 0,
        .actual_fps = 0.0,
        .avg_capture_latency_ms = 0.0,
        .max_capture_latency_ms = 0.0,
        .last_frame_time = std::chrono::system_clock::now()
    };
}

CameraCapture::~CameraCapture() {
    stop();
}

bool CameraCapture::initialize(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);

    if (initialized_) {
        std::cerr << "Camera already initialized" << std::endl;
        return false;
    }

    config_ = config;

    // Create OpenCV VideoCapture with V4L2 backend
    capture_ = std::make_unique<cv::VideoCapture>(config_.device_path, cv::CAP_V4L2);

    if (!capture_->isOpened()) {
        std::cerr << "Failed to open camera: " << config_.device_path << std::endl;
        return false;
    }

    // Set camera parameters
    capture_->set(cv::CAP_PROP_FRAME_WIDTH, config_.resolution.width);
    capture_->set(cv::CAP_PROP_FRAME_HEIGHT, config_.resolution.height);
    capture_->set(cv::CAP_PROP_FPS, config_.target_fps);
    capture_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // Verify settings
    int actual_width = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_height = static_cast<int>(capture_->get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = capture_->get(cv::CAP_PROP_FPS);

    std::cout << "Camera initialized: " << actual_width << "x" << actual_height
              << " @ " << actual_fps << "fps" << std::endl;

    // Apply additional camera parameters
    if (!applyCameraParams()) {
        std::cerr << "Warning: Failed to apply some camera parameters" << std::endl;
    }

    // Allocate zero-copy buffers if enabled
    if (config_.zero_copy && !allocateZeroCopyBuffer()) {
        std::cerr << "Failed to allocate zero-copy buffers" << std::endl;
        return false;
    }

    initialized_ = true;
    start_time_ = std::chrono::steady_clock::now();

    return true;
}

bool CameraCapture::start() {
    std::lock_guard<std::mutex> lock(config_mutex_);

    if (!initialized_) {
        std::cerr << "Camera not initialized" << std::endl;
        return false;
    }

    if (running_) {
        std::cerr << "Camera already running" << std::endl;
        return false;
    }

    running_ = true;
    sequence_number_ = 0;

    // Start capture thread
    capture_thread_ = std::thread(&CameraCapture::captureThread, this);

    std::cout << "Camera capture started" << std::endl;
    return true;
}

void CameraCapture::stop() {
    running_ = false;

    if (capture_thread_.joinable()) {
        capture_thread_.join();
    }

    capture_.reset();
    initialized_ = false;

    std::cout << "Camera capture stopped" << std::endl;
}

void CameraCapture::setFrameCallback(FrameCallback callback) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    frame_callback_ = callback;
}

CameraConfig CameraCapture::getConfig() const {
    std::lock_guard<std::mutex> lock(config_mutex_);
    return config_;
}

bool CameraCapture::updateConfig(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(config_mutex_);

    if (!running_) {
        config_ = config;
        return true;
    }

    // Update parameters that can be changed while running
    if (config.brightness != -1) {
        capture_->set(cv::CAP_PROP_BRIGHTNESS, config.brightness);
        config_.brightness = config.brightness;
    }
    if (config.contrast != -1) {
        capture_->set(cv::CAP_PROP_CONTRAST, config.contrast);
        config_.contrast = config.contrast;
    }
    if (config.saturation != -1) {
        capture_->set(cv::CAP_PROP_SATURATION, config.saturation);
        config_.saturation = config.saturation;
    }

    return true;
}

CameraStats CameraCapture::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool CameraCapture::isRunning() const {
    return running_;
}

void CameraCapture::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.frames_captured = 0;
    stats_.frames_dropped = 0;
    stats_.avg_capture_latency_ms = 0.0;
    stats_.max_capture_latency_ms = 0.0;
    start_time_ = std::chrono::steady_clock::now();
}

void CameraCapture::captureThread() {
    CameraFrame frame;

    while (running_) {
        auto capture_start = std::chrono::steady_clock::now();

        if (!captureFrame(frame)) {
            stats_.frames_dropped++;
            continue;
        }

        // Update statistics
        auto capture_end = std::chrono::steady_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            capture_end - capture_start).count();

        updateLatencyStats(latency_ms);

        // Invoke callback if set
        if (frame_callback_) {
            frame_callback_(frame);
        }
    }
}

bool CameraCapture::captureFrame(CameraFrame& frame) {
    cv::Mat cpu_frame;

    // Capture frame from camera
    if (!capture_->read(cpu_frame) || cpu_frame.empty()) {
        std::cerr << "Failed to capture frame" << std::endl;
        return false;
    }

    // Upload to GPU with zero-copy if enabled
    if (config_.zero_copy) {
        // Use buffer pool for zero-copy transfer
        frame.frame = buffer_pool_[current_buffer_index_];
        frame.frame.upload(cpu_frame);
        current_buffer_index_ = (current_buffer_index_ + 1) % buffer_pool_.size();
    } else {
        frame.frame.upload(cpu_frame);
    }

    // Set frame metadata
    frame.camera_id = config_.camera_id;
    frame.sequence_number = sequence_number_++;
    frame.capture_time = std::chrono::system_clock::now();
    frame.resolution = cv::Size(cpu_frame.cols, cpu_frame.rows);
    frame.fps = config_.target_fps;

    // Hardware timestamp (simulated - actual implementation would read from V4L2)
    if (config_.hardware_timestamp) {
        frame.hardware_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    return true;
}

bool CameraCapture::applyCameraParams() {
    if (!capture_) return false;

    // Apply camera parameters
    if (config_.exposure != -1) {
        capture_->set(cv::CAP_PROP_EXPOSURE, config_.exposure);
    }
    if (config_.focus != -1) {
        capture_->set(cv::CAP_PROP_FOCUS, config_.focus);
    }
    if (config_.white_balance != -1) {
        capture_->set(cv::CAP_PROP_WHITE_BALANCE_BLUE_U, config_.white_balance);
    }
    capture_->set(cv::CAP_PROP_BRIGHTNESS, config_.brightness);
    capture_->set(cv::CAP_PROP_CONTRAST, config_.contrast);
    capture_->set(cv::CAP_PROP_SATURATION, config_.saturation);

    return true;
}

bool CameraCapture::allocateZeroCopyBuffer() {
    // Create buffer pool for zero-copy transfers
    // On Jetson, we use CUDA unified memory
    size_t pool_size = 3;  // Triple buffering

    for (size_t i = 0; i < pool_size; i++) {
        cv::cuda::GpuMat buffer(config_.resolution.height,
                                config_.resolution.width,
                                CV_8UC3);
        buffer_pool_.push_back(buffer);
    }

    std::cout << "Allocated " << pool_size << " zero-copy buffers" << std::endl;
    return true;
}

void CameraCapture::updateLatencyStats(double latency_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.frames_captured++;
    stats_.last_frame_time = std::chrono::system_clock::now();

    // Update latency stats (exponential moving average)
    if (stats_.frames_captured == 1) {
        stats_.avg_capture_latency_ms = latency_ms;
    } else {
        stats_.avg_capture_latency_ms = 0.9 * stats_.avg_capture_latency_ms +
                                        0.1 * latency_ms;
    }

    if (latency_ms > stats_.max_capture_latency_ms) {
        stats_.max_capture_latency_ms = latency_ms;
    }

    // Update FPS
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time_).count();
    stats_.actual_fps = stats_.frames_captured / elapsed;
}

// ============================================================================
// MultiCameraCapture Implementation
// ============================================================================

MultiCameraCapture::MultiCameraCapture() {}

MultiCameraCapture::~MultiCameraCapture() {
    stopAll();
}

int MultiCameraCapture::addCamera(const CameraConfig& config) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    if (cameras_.size() >= MAX_CAMERAS) {
        std::cerr << "Maximum number of cameras reached" << std::endl;
        return -1;
    }

    auto camera = std::make_unique<CameraCapture>();

    if (!camera->initialize(config)) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return -1;
    }

    cameras_.push_back(std::move(camera));
    return static_cast<int>(cameras_.size() - 1);
}

void MultiCameraCapture::removeCamera(int camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    if (camera_id >= 0 && camera_id < static_cast<int>(cameras_.size())) {
        cameras_[camera_id]->stop();
        cameras_.erase(cameras_.begin() + camera_id);
    }
}

bool MultiCameraCapture::startAll() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    for (auto& camera : cameras_) {
        if (!camera->start()) {
            std::cerr << "Failed to start camera" << std::endl;
            return false;
        }
    }

    return true;
}

void MultiCameraCapture::stopAll() {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    for (auto& camera : cameras_) {
        camera->stop();
    }
}

CameraCapture* MultiCameraCapture::getCamera(int camera_id) {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    if (camera_id >= 0 && camera_id < static_cast<int>(cameras_.size())) {
        return cameras_[camera_id].get();
    }
    return nullptr;
}

CameraStats MultiCameraCapture::getCombinedStats() const {
    std::lock_guard<std::mutex> lock(cameras_mutex_);

    CameraStats combined{0};
    double total_fps = 0.0;
    size_t running_cameras = 0;

    for (const auto& camera : cameras_) {
        if (camera->isRunning()) {
            CameraStats stats = camera->getStats();
            combined.frames_captured += stats.frames_captured;
            combined.frames_dropped += stats.frames_dropped;
            combined.avg_capture_latency_ms += stats.avg_capture_latency_ms;
            if (stats.max_capture_latency_ms > combined.max_capture_latency_ms) {
                combined.max_capture_latency_ms = stats.max_capture_latency_ms;
            }
            total_fps += stats.actual_fps;
            running_cameras++;
        }
    }

    if (running_cameras > 0) {
        combined.avg_capture_latency_ms /= running_cameras;
        combined.actual_fps = total_fps / running_cameras;
    }

    return combined;
}

} // namespace cv
} // namespace kiosk
