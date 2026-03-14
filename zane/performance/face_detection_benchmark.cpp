/**
 * @file face_detection_benchmark.cpp
 * @brief Performance benchmark for face detection system
 * @author Zane - C++/CV Engineer
 * 
 * This benchmark tests the actual face detection system performance
 * with different configurations and loads.
 */

#include "benchmark.h"
#include "../real-time/face_detection.h"
#include "../real-time/camera_capture.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>

using namespace kiosk::benchmark;
using namespace kiosk::cv;

class FaceDetectionBenchmark : public Benchmark {
public:
    FaceDetectionBenchmark() 
        : Benchmark()
        , face_detector_(nullptr)
        , camera_capture_(nullptr) {
    }

    ~FaceDetectionBenchmark() {
        cleanup();
    }

    bool initialize(const BenchmarkConfig& config) override {
        if (!Benchmark::initialize(config)) {
            return false;
        }

        // Initialize face detection system
        FaceDetectionConfig fd_config;
        fd_config.detection_interval = 1;  // Detect every frame
        fd_config.min_face_size = cv::Size(30, 30);
        fd_config.max_face_size = cv::Size(300, 300);
        fd_config.detection_confidence_threshold = 0.5f;
        fd_config.enable_landmarks = true;
        fd_config.enable_emotion = true;
        fd_config.enable_attention = true;
        fd_config.enable_gpu = config.enable_gpu;
        fd_config.max_tracks = 50;
        fd_config.track_timeout_frames = 30;
        fd_config.enable_visualization = false;  // Disable for benchmarking

        face_detector_ = std::make_unique<FaceDetectionSystem>();
        if (!face_detector_->initialize(fd_config)) {
            std::cerr << "Failed to initialize face detection system" << std::endl;
            return false;
        }

        // Initialize camera capture if specified
        if (!config.camera_device.empty()) {
            CameraConfig cam_config;
            cam_config.device_id = config.camera_device;
            cam_config.resolution = config.resolution;
            cam_config.fps = 30;
            cam_config.buffer_size = 3;
            cam_config.enable_auto_exposure = true;
            cam_config.enable_auto_white_balance = true;

            camera_capture_ = std::make_unique<CameraCapture>();
            if (!camera_capture_->initialize(cam_config)) {
                std::cerr << "Failed to initialize camera capture" << std::endl;
                return false;
            }
        }

        return true;
    }

protected:
    void processFrame(int frame_number) override {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Get or generate frame
        cv::Mat frame;
        if (camera_capture_) {
            // Capture from camera
            if (!camera_capture_->capture(frame)) {
                std::cerr << "Failed to capture frame from camera" << std::endl;
                return;
            }
        } else {
            // Generate synthetic frame for testing
            frame = generateSyntheticFrame(frame_number);
        }

        // Process with face detection
        std::vector<FaceAnalysis> results;
        auto detection_start = std::chrono::high_resolution_clock::now();
        
        if (face_detector_->detectAndTrack(frame, results)) {
            auto detection_end = std::chrono::high_resolution_clock::now();
            auto detection_time = std::chrono::duration<double, std::milli>(detection_end - detection_start).count();

            // Update frame metrics
            FrameMetrics metrics;
            metrics.frame_number = frame_number;
            metrics.processing_time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - frame_start).count();
            metrics.faces_detected = results.size();
            metrics.active_tracks = face_detector_->getActiveTrackCount();
            metrics.detection_time_ms = detection_time;
            
            // Get system metrics
            getSystemMetrics(metrics);

            // Store metrics
            addFrameMetrics(metrics);

            // Update progress
            if (config_.print_progress && frame_number % 100 == 0) {
                std::cout << "Processed " << frame_number << " frames, " 
                          << metrics.faces_detected << " faces detected, "
                          << std::fixed << std::setprecision(2) << metrics.processing_time_ms << " ms/frame" << std::endl;
            }
        }
    }

    cv::Mat generateSyntheticFrame(int frame_number) {
        // Create a synthetic frame with faces for testing
        cv::Mat frame(config_.resolution, CV_8UC3, cv::Scalar(100, 100, 100));

        // Add some "faces" as circles that move around
        int num_faces = 2 + (frame_number / 100) % 3;  // Vary between 2-4 faces
        for (int i = 0; i < num_faces; i++) {
            int x = 200 + (i * 200) + static_cast<int>(50 * sin(frame_number * 0.1 + i));
            int y = 200 + static_cast<int>(50 * cos(frame_number * 0.05 + i));
            int radius = 40 + static_cast<int>(10 * sin(frame_number * 0.02 + i));
            
            cv::circle(frame, cv::Point(x, y), radius, cv::Scalar(200, 100, 50), -1);
            
            // Add "eyes"
            cv::circle(frame, cv::Point(x - 15, y - 10), 8, cv::Scalar(255, 255, 255), -1);
            cv::circle(frame, cv::Point(x + 15, y - 10), 8, cv::Scalar(255, 255, 255), -1);
            
            // Add "mouth"
            cv::ellipse(frame, cv::Point(x, y + 15), cv::Size(20, 10), 0, 0, 180, cv::Scalar(255, 255, 255), 2);
        }

        return frame;
    }

    void getSystemMetrics(FrameMetrics& metrics) {
        metrics.cpu_percent = getCpuUsage();
        metrics.memory_mb = getMemoryUsage();
        
        // Try to get GPU metrics if available
        #ifdef __APPLE__
            // macOS doesn't have easy GPU monitoring
            metrics.gpu_memory_mb = 0;
            metrics.gpu_utilization = 0;
            metrics.temperature_c = 0;
            metrics.power_w = 0;
        #elif __linux__
            // Linux with NVIDIA GPU
            metrics.gpu_memory_mb = getGpuMemoryUsage();
            metrics.gpu_utilization = getGpuUtilization();
            metrics.temperature_c = getGpuTemperature();
            metrics.power_w = getGpuPower();
        #else
            metrics.gpu_memory_mb = 0;
            metrics.gpu_utilization = 0;
            metrics.temperature_c = 0;
            metrics.power_w = 0;
        #endif
    }

    void cleanup() override {
        if (face_detector_) {
            face_detector_->stop();
            face_detector_.reset();
        }
        
        if (camera_capture_) {
            camera_capture_->stop();
            camera_capture_.reset();
        }
        
        Benchmark::cleanup();
    }

private:
    std::unique_ptr<FaceDetectionSystem> face_detector_;
    std::unique_ptr<CameraCapture> camera_capture_;
};

int main(int argc, char* argv[]) {
    std::cout << "=== Face Detection Performance Benchmark ===" << std::endl;
    std::cout << "Author: Zane - C++/CV Engineer" << std::endl;
    std::cout << "Date: " << __DATE__ << " " << __TIME__ << std::endl;
    std::cout << "=============================================" << std::endl;

    // Default configuration
    BenchmarkConfig config;
    config.name = "face_detection_benchmark";
    config.duration_seconds = 60;  // 1 minute default for quick test
    config.num_frames = 0;
    config.camera_device = "";  // Use synthetic frames by default
    config.resolution = cv::Size(1280, 720);
    config.enable_gpu = true;
    config.save_results_csv = true;
    config.results_dir = "./results";
    config.enable_profiling = false;
    config.profiling_interval = 100;
    config.warmup_frames = 30;
    config.enable_cpu_monitoring = true;
    config.enable_memory_monitoring = true;
    config.enable_gpu_monitoring = false;
    config.print_progress = true;
    config.print_summary = true;
    config.generate_plots = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --help, -h              Show this help message" << std::endl;
            std::cout << "  --quick                 Run quick benchmark (30 seconds)" << std::endl;
            std::cout << "  --standard              Run standard benchmark (5 minutes)" << std::endl;
            std::cout << "  --full                  Run full benchmark (30 minutes)" << std::endl;
            std::cout << "  --camera <device>       Use camera device (e.g., /dev/video0)" << std::endl;
            std::cout << "  --duration <seconds>    Run for specified duration" << std::endl;
            std::cout << "  --frames <count>        Process specified number of frames" << std::endl;
            std::cout << "  --results <dir>         Results output directory" << std::endl;
            std::cout << "  --no-gpu                Disable GPU acceleration" << std::endl;
            std::cout << "  --load <low|medium|high> Set synthetic load level" << std::endl;
            return 0;
        }
        else if (arg == "--quick") {
            config.name = "face_detection_quick";
            config.duration_seconds = 30;
        }
        else if (arg == "--standard") {
            config.name = "face_detection_standard";
            config.duration_seconds = 300;  // 5 minutes
        }
        else if (arg == "--full") {
            config.name = "face_detection_full";
            config.duration_seconds = 1800;  // 30 minutes
        }
        else if (arg == "--camera") {
            if (i + 1 < argc) {
                config.camera_device = argv[++i];
                config.name = "face_detection_camera";
            } else {
                std::cerr << "Error: --camera requires a device path" << std::endl;
                return 1;
            }
        }
        else if (arg == "--duration") {
            if (i + 1 < argc) {
                config.duration_seconds = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --duration requires a value in seconds" << std::endl;
                return 1;
            }
        }
        else if (arg == "--frames") {
            if (i + 1 < argc) {
                config.num_frames = std::atoi(argv[++i]);
                config.duration_seconds = 0;
            } else {
                std::cerr << "Error: --frames requires a count" << std::endl;
                return 1;
            }
        }
        else if (arg == "--results") {
            if (i + 1 < argc) {
                config.results_dir = argv[++i];
            } else {
                std::cerr << "Error: --results requires a directory path" << std::endl;
                return 1;
            }
        }
        else if (arg == "--no-gpu") {
            config.enable_gpu = false;
        }
        else if (arg == "--load") {
            if (i + 1 < argc) {
                std::string load = argv[++i];
                if (load == "low") {
                    config.resolution = cv::Size(640, 480);
                } else if (load == "medium") {
                    config.resolution = cv::Size(1280, 720);
                } else if (load == "high") {
                    config.resolution = cv::Size(1920, 1080);
                } else {
                    std::cerr << "Error: --load must be low, medium, or high" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --load requires a level" << std::endl;
                return 1;
            }
        }
    }

    // Create and run benchmark
    FaceDetectionBenchmark benchmark;
    
    if (!benchmark.initialize(config)) {
        std::cerr << "Failed to initialize benchmark" << std::endl;
        return 1;
    }

    std::cout << "\nStarting face detection benchmark..." << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Name: " << config.name << std::endl;
    std::cout << "  Duration: " << config.duration_seconds << " seconds" << std::endl;
    std::cout << "  Frames: " << (config.num_frames > 0 ? std::to_string(config.num_frames) : "duration-based") << std::endl;
    std::cout << "  Camera: " << (config.camera_device.empty() ? "synthetic frames" : config.camera_device) << std::endl;
    std::cout << "  Resolution: " << config.resolution.width << "x" << config.resolution.height << std::endl;
    std::cout << "  GPU: " << (config.enable_gpu ? "enabled" : "disabled") << std::endl;
    std::cout << "  Results: " << config.results_dir << std::endl;

    bool success = benchmark.run();

    if (success) {
        // Export results
        BenchmarkSummary summary = benchmark.getSummary();
        
        std::string summary_path = config.results_dir + "/" + config.name + "_summary.txt";
        if (!benchmark.exportSummary(summary_path)) {
            std::cerr << "Warning: Could not export summary" << std::endl;
        }

        std::string report_path = config.results_dir + "/" + config.name + "_report.md";
        if (!benchmark.generateReport(report_path)) {
            std::cerr << "Warning: Could not generate report" << std::endl;
        }

        // Generate face detection specific report
        std::string fd_report_path = config.results_dir + "/" + config.name + "_face_detection_report.md";
        std::ofstream report(fd_report_path);
        if (report.is_open()) {
            report << "# Face Detection Performance Report\n\n";
            report << "## Benchmark Details\n";
            report << "- **Name**: " << config.name << "\n";
            report << "- **Date**: " << __DATE__ << " " << __TIME__ << "\n";
            report << "- **Duration**: " << config.duration_seconds << " seconds\n";
            report << "- **Resolution**: " << config.resolution.width << "x" << config.resolution.height << "\n";
            report << "- **GPU Acceleration**: " << (config.enable_gpu ? "Enabled" : "Disabled") << "\n";
            report << "- **Input Source**: " << (config.camera_device.empty() ? "Synthetic Frames" : config.camera_device) << "\n\n";
            
            report << "## Performance Summary\n";
            report << "| Metric | Value | Target | Status |\n";
            report << "|--------|-------|--------|--------|\n";
            report << "| Average FPS | " << std::fixed << std::setprecision(2) << summary.avg_fps << " | 30 | " 
                   << (summary.avg_fps >= 30 ? "✅ PASS" : "❌ FAIL") << " |\n";
            report << "| Avg Latency | " << std::fixed << std::setprecision(2) << (1000.0 / summary.avg_fps) << " ms | <100 ms | "
                   << ((1000.0 / summary.avg_fps) < 100 ? "✅ PASS" : "❌ FAIL") << " |\n";
            report << "| Max Faces/Frame | " << summary.max_faces_per_frame << " | 20+ | "
                   << (summary.max_faces_per_frame >= 20 ? "✅ PASS" : "⚠️ PARTIAL") << " |\n";
            report << "| CPU Usage | " << std::fixed << std::setprecision(1) << summary.avg_cpu_percent << "% | <80% | "
                   << (summary.avg_cpu_percent < 80 ? "✅ PASS" : "⚠️ HIGH") << " |\n";
            report << "| Memory Usage | " << std::fixed << std::setprecision(1) << summary.avg_memory_mb << " MB | <6000 MB | "
                   << (summary.avg_memory_mb < 6000 ? "✅ PASS" : "⚠️ HIGH") << " |\n\n";
            
            report << "## Detailed Statistics\n";
            report << "- **Total Frames Processed**: " << summary.total_frames << "\n";
            report << "- **Total Faces Detected