/**
 * @file camera_capture.h
 * @brief Camera capture system for Jetson Orin Nano
 *
 * High-performance camera capture system supporting:
 * - V4L2 driver integration
 * - Zero-copy CUDA buffer allocation
 * - Multi-camera synchronization
 * - Hardware timestamping
 */

#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/cudacodec.hpp>

namespace kiosk {
namespace cv {

/**
 * @brief Camera frame metadata
 */
struct CameraFrame {
    cv::cuda::GpuMat frame;           // GPU-resident frame (zero-copy)
    uint64_t hardware_timestamp_ns;   // Hardware timestamp in nanoseconds
    uint64_t sequence_number;         // Frame sequence number
    int camera_id;                    // Camera identifier
    std::chrono::system_clock::time_point capture_time;  // System capture time
    cv::Size resolution;              // Frame resolution
    double fps;                       // Captured frame rate
};

/**
 * @brief Camera configuration parameters
 */
struct CameraConfig {
    int camera_id;                    // Camera identifier
    std::string device_path;          // V4L2 device path (e.g., "/dev/video0")
    cv::Size resolution;              // Desired resolution
    double target_fps;                // Target frame rate
    bool zero_copy;                   // Enable zero-copy CUDA buffers
    bool hardware_timestamp;          // Enable hardware timestamping

    // Camera parameter controls
    int exposure;                     // Exposure time (-1 = auto)
    int focus;                        // Focus value (-1 = auto)
    int white_balance;                // White balance mode (-1 = auto)
    int brightness;                   // Brightness (0-255)
    int contrast;                     // Contrast (0-255)
    int saturation;                   // Saturation (0-255)

    CameraConfig()
        : camera_id(0)
        , device_path("/dev/video0")
        , resolution(1920, 1080)
        , target_fps(60.0)
        , zero_copy(true)
        , hardware_timestamp(true)
        , exposure(-1)
        , focus(-1)
        , white_balance(-1)
        , brightness(128)
        , contrast(128)
        , saturation(128) {}
};

/**
 * @brief Camera statistics and performance metrics
 */
struct CameraStats {
    uint64_t frames_captured;         // Total frames captured
    uint64_t frames_dropped;          // Frames dropped
    double actual_fps;                // Actual frame rate
    double avg_capture_latency_ms;    // Average capture latency
    double max_capture_latency_ms;    // Maximum capture latency
    std::chrono::system_clock::time_point last_frame_time;
};

/**
 * @brief Camera capture callback type
 */
using FrameCallback = std::function<void(const CameraFrame&)>;

/**
 * @brief High-performance camera capture class
 *
 * Features:
 * - V4L2 driver integration via OpenCV
 * - Zero-copy CUDA buffer allocation for Jetson Orin Nano
 * - Hardware timestamping support
 * - Multi-camera synchronization
 * - Frame rate control and monitoring
 */
class CameraCapture {
public:
    CameraCapture();
    ~CameraCapture();

    /**
     * @brief Initialize camera with configuration
     * @param config Camera configuration
     * @return true if successful, false otherwise
     */
    bool initialize(const CameraConfig& config);

    /**
     * @brief Start camera capture
     * @return true if successful, false otherwise
     */
    bool start();

    /**
     * @brief Stop camera capture
     */
    void stop();

    /**
     * @brief Set frame callback for new frames
     * @param callback Function to call for each new frame
     */
    void setFrameCallback(FrameCallback callback);

    /**
     * @brief Get current camera configuration
     * @return Current configuration
     */
    CameraConfig getConfig() const;

    /**
     * @brief Update camera configuration
     * @param config New configuration
     * @return true if successful, false otherwise
     */
    bool updateConfig(const CameraConfig& config);

    /**
     * @brief Get camera statistics
     * @return Current statistics
     */
    CameraStats getStats() const;

    /**
     * @brief Check if camera is running
     * @return true if running, false otherwise
     */
    bool isRunning() const;

    /**
     * @brief Reset frame counters
     */
    void resetStats();

private:
    // Capture thread function
    void captureThread();

    // Capture single frame from camera
    bool captureFrame(CameraFrame& frame);

    // Apply camera parameters
    bool applyCameraParams();

    // Zero-copy buffer allocation
    bool allocateZeroCopyBuffer();

    // Performance monitoring
    void updateLatencyStats(double latency_ms);

private:
    CameraConfig config_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    std::mutex config_mutex_;
    std::mutex stats_mutex_;

    FrameCallback frame_callback_;

    // OpenCV video capture
    std::unique_ptr<cv::VideoCapture> capture_;

    // Zero-copy CUDA buffer pool
    std::vector<cv::cuda::GpuMat> buffer_pool_;
    size_t current_buffer_index_;

    // Statistics
    CameraStats stats_;
    std::chrono::steady_clock::time_point start_time_;

    // Capture thread
    std::thread capture_thread_;
    uint64_t sequence_number_;
};

/**
 * @brief Multi-camera manager for synchronized capture
 *
 * Manages multiple cameras with synchronized timing.
 * Supports up to 3 cameras for kiosk applications.
 */
class MultiCameraCapture {
public:
    static constexpr int MAX_CAMERAS = 3;

    MultiCameraCapture();
    ~MultiCameraCapture();

    /**
     * @brief Add camera to manager
     * @param config Camera configuration
     * @return Camera ID if successful, -1 otherwise
     */
    int addCamera(const CameraConfig& config);

    /**
     * @brief Remove camera
     * @param camera_id Camera ID to remove
     */
    void removeCamera(int camera_id);

    /**
     * @brief Start all cameras
     * @return true if successful, false otherwise
     */
    bool startAll();

    /**
     * @brief Stop all cameras
     */
    void stopAll();

    /**
     * @brief Get camera by ID
     * @param camera_id Camera ID
     * @return Pointer to camera, nullptr if not found
     */
    CameraCapture* getCamera(int camera_id);

    /**
     * @brief Get combined statistics for all cameras
     * @return Aggregated statistics
     */
    CameraStats getCombinedStats() const;

private:
    std::vector<std::unique_ptr<CameraCapture>> cameras_;
    std::mutex cameras_mutex_;
};

} // namespace cv
} // namespace kiosk
