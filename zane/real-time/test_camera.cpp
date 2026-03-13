/**
 * @file test_camera.cpp
 * @brief Test program for camera capture system
 */

#include "camera_capture.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace kiosk::cv;

void printStats(const CameraStats& stats) {
    std::cout << "\n=== Camera Statistics ===" << std::endl;
    std::cout << "Frames captured: " << stats.frames_captured << std::endl;
    std::cout << "Frames dropped: " << stats.frames_dropped << std::endl;
    std::cout << "Actual FPS: " << stats.actual_fps << std::endl;
    std::cout << "Avg capture latency: " << stats.avg_capture_latency_ms << " ms" << std::endl;
    std::cout << "Max capture latency: " << stats.max_capture_latency_ms << " ms" << std::endl;
    std::cout << "==========================\n" << std::endl;
}

void frameCallback(const CameraFrame& frame) {
    static uint64_t frame_count = 0;
    frame_count++;

    if (frame_count % 60 == 0) {
        std::cout << "Received frame #" << frame.sequence_number
                  << " from camera " << frame.camera_id
                  << " [" << frame.resolution.width << "x" << frame.resolution.height << "]"
                  << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== Kiosk Camera Capture Test ===" << std::endl;

    // Parse command line arguments
    std::string device_path = "/dev/video0";
    if (argc > 1) {
        device_path = argv[1];
    }

    std::cout << "Using device: " << device_path << std::endl;

    // Create camera configuration
    CameraConfig config;
    config.device_path = device_path;
    config.resolution = cv::Size(1920, 1080);
    config.target_fps = 60.0;
    config.zero_copy = true;
    config.hardware_timestamp = true;
    config.brightness = 128;
    config.contrast = 128;
    config.saturation = 128;

    // Create camera capture instance
    CameraCapture camera;

    // Initialize camera
    std::cout << "\nInitializing camera..." << std::endl;
    if (!camera.initialize(config)) {
        std::cerr << "Failed to initialize camera!" << std::endl;
        return 1;
    }

    // Set frame callback
    camera.setFrameCallback(frameCallback);

    // Start capture
    std::cout << "\nStarting capture..." << std::endl;
    if (!camera.start()) {
        std::cerr << "Failed to start capture!" << std::endl;
        return 1;
    }

    // Run for 10 seconds
    std::cout << "\nRunning for 10 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // Print statistics
    CameraStats stats = camera.getStats();
    printStats(stats);

    // Update camera parameters (test dynamic reconfiguration)
    std::cout << "\nTesting parameter update..." << std::endl;
    CameraConfig new_config = config;
    new_config.brightness = 160;
    new_config.contrast = 140;
    if (camera.updateConfig(new_config)) {
        std::cout << "Parameters updated successfully" << std::endl;
    }

    // Run for another 5 seconds
    std::cout << "\nRunning for 5 more seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Final statistics
    stats = camera.getStats();
    printStats(stats);

    // Stop capture
    std::cout << "\nStopping capture..." << std::endl;
    camera.stop();

    std::cout << "\n=== Test Complete ===" << std::endl;

    return 0;
}

/**
 * Multi-camera test example (commented out by default)
 */
/*
void multiCameraTest() {
    std::cout << "=== Multi-Camera Test ===" << std::endl;

    MultiCameraCapture manager;

    // Add 3 cameras
    for (int i = 0; i < 3; i++) {
        CameraConfig config;
        config.camera_id = i;
        config.device_path = "/dev/video" + std::to_string(i);
        config.resolution = cv::Size(1920, 1080);
        config.target_fps = 60.0;
        config.zero_copy = true;

        int camera_id = manager.addCamera(config);
        if (camera_id == -1) {
            std::cerr << "Failed to add camera " << i << std::endl;
            continue;
        }

        std::cout << "Added camera " << camera_id << std::endl;
    }

    // Start all cameras
    if (!manager.startAll()) {
        std::cerr << "Failed to start all cameras" << std::endl;
        return;
    }

    // Run for 30 seconds
    std::this_thread::sleep_for(std::chrono::seconds(30));

    // Print combined statistics
    CameraStats stats = manager.getCombinedStats();
    printStats(stats);

    // Stop all cameras
    manager.stopAll();

    std::cout << "=== Multi-Camera Test Complete ===" << std::endl;
}
*/
