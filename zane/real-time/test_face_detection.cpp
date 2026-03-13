/**
 * @file test_face_detection.cpp
 * @brief Test program for face detection and tracking system
 * @author Zane - C++/CV Engineer
 */

#include "face_detection.h"
#include "camera_capture.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace kiosk::cv;

void printFaceAnalysisStats(const FaceAnalysis& analysis, int index) {
    std::cout << "\n--- Face #" << index << " ---" << std::endl;
    std::cout << "Face ID: " << analysis.detection.face_id << std::endl;
    std::cout << "Position: (" << analysis.detection.center.x << ", " << analysis.detection.center.y << ")" << std::endl;
    std::cout << "Size: " << analysis.detection.size.width << "x" << analysis.detection.size.height << std::endl;
    std::cout << "Confidence: " << analysis.detection.confidence << std::endl;

    if (analysis.landmarks.points.size() >= 68) {
        std::cout << "Landmarks: 68 points extracted" << std::endl;
        std::cout << "Eye Openness: L=" << analysis.landmarks.left_eye_openness
                  << ", R=" << analysis.landmarks.right_eye_openness << std::endl;
        std::cout << "Mouth Openness: " << analysis.landmarks.mouth_openness << std::endl;
    }

    std::cout << "Emotion: " << emotionToString(analysis.emotion.emotion)
              << " (" << static_cast<int>(analysis.emotion.confidence * 100) << "%)" << std::endl;

    std::cout << "Attention Score: " << analysis.attention.attention_score
              << " (" << static_cast<int>(analysis.attention.attention_score * 100) << "%)" << std::endl;
    std::cout << "Engagement Level: " << analysis.attention.engagement_level
              << " (" << static_cast<int>(analysis.attention.engagement_level * 100) << "%)" << std::endl;
    std::cout << "Looking at Screen: " << (analysis.attention.looking_at_screen ? "Yes" : "No") << std::endl;
    std::cout << "Is Distracted: " << (analysis.attention.is_distracted ? "Yes" : "No") << std::endl;
}

void printFaceDetectionStats(const FaceDetectionSystem::Stats& stats) {
    std::cout << "\n=== Face Detection Statistics ===" << std::endl;
    std::cout << "Frames Processed: " << stats.frames_processed << std::endl;
    std::cout << "Faces Detected: " << stats.faces_detected << std::endl;
    std::cout << "Active Tracks: " << stats.tracks_active << std::endl;
    std::cout << "Current FPS: " << stats.current_fps << std::endl;
    std::cout << "Avg Detection Time: " << stats.avg_detection_time_ms << " ms" << std::endl;
    std::cout << "Avg Tracking Time: " << stats.avg_tracking_time_ms << " ms" << std::endl;
    std::cout << "Avg Analysis Time: " << stats.avg_analysis_time_ms << " ms" << std::endl;
    std::cout << "Max Latency: " << stats.max_latency_ms << " ms" << std::endl;
    std::cout << "==================================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Kiosk Face Detection Test ===" << std::endl;

    // Parse command line arguments
    std::string device_path = "/dev/video0";
    bool use_camera = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--camera" || arg == "-c") {
            use_camera = true;
        } else if (arg == "--device" || arg == "-d") {
            if (i + 1 < argc) {
                device_path = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --camera, -c      Use camera input" << std::endl;
            std::cout << "  --device, -d      Camera device path (default: /dev/video0)" << std::endl;
            std::cout << "  --help, -h        Show this help message" << std::endl;
            return 0;
        }
    }

    // Initialize face detection system
    FaceDetectionConfig config;
    config.confidence_threshold = 0.7f;
    config.nms_threshold = 0.4f;
    config.input_size = 640;
    config.use_gpu = true;
    config.use_tensorrt = true;
    config.max_distance = 0.7f;
    config.max_age = 30;
    config.n_init = 3;
    config.enable_landmarks = true;
    config.enable_emotion = true;
    config.enable_attention = true;
    config.target_fps = 30.0f;
    config.max_latency_ms = 50.0f;

    FaceDetectionSystem face_system;
    std::cout << "\nInitializing face detection system..." << std::endl;

    if (!face_system.initialize(config)) {
        std::cerr << "Failed to initialize face detection system!" << std::endl;
        return 1;
    }

    if (!face_system.start()) {
        std::cerr << "Failed to start face detection system!" << std::endl;
        return 1;
    }

    std::cout << "Face detection system initialized successfully" << std::endl;

    CameraCapture camera;
    cv::Mat frame;

    if (use_camera) {
        std::cout << "\nInitializing camera..." << std::endl;

        CameraConfig cam_config;
        cam_config.device_path = device_path;
        cam_config.resolution = cv::Size(1280, 720);
        cam_config.target_fps = 30.0;
        cam_config.zero_copy = true;

        if (!camera.initialize(cam_config)) {
            std::cerr << "Failed to initialize camera!" << std::endl;
            return 1;
        }

        if (!camera.start()) {
            std::cerr << "Failed to start camera!" << std::endl;
            return 1;
        }

        std::cout << "Camera initialized successfully" << std::endl;
    }

    // Processing loop
    int frame_count = 0;
    int print_interval = 30;  // Print stats every 30 frames
    auto test_start = std::chrono::steady_clock::now();

    std::cout << "\nStarting processing loop..." << std::endl;
    std::cout << "Press ESC to stop" << std::endl;

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();

        // Get frame from camera or use test image
        if (use_camera) {
            cv::Mat cpu_frame;
            camera.capture_->read(cpu_frame);
            if (!cpu_frame.empty()) {
                frame = cpu_frame;
            } else {
                std::cerr << "Failed to capture frame" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
                continue;
            }
        } else {
            // For testing without camera, create a synthetic frame
            frame = cv::Mat::zeros(720, 1280, CV_8UC3);
            cv::putText(frame, "No Camera - Simulation Mode",
                       cv::Point(400, 360), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                       cv::Scalar(255, 255, 255), 2);

            // Add simulated faces (circles)
            cv::circle(frame, cv::Point(300, 300), 80, cv::Scalar(255, 200, 150), -1);
            cv::circle(frame, cv::Point(700, 250), 90, cv::Scalar(200, 180, 150), -1);
            cv::circle(frame, cv::Point(900, 350), 70, cv::Scalar(210, 190, 160), -1);
        }

        if (frame.empty()) {
            std::cerr << "Empty frame!" << std::endl;
            continue;
        }

        // Resize frame for performance
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(640, 480));

        // Process frame
        std::vector<FaceAnalysis> analyses = face_system.analyzeFrame(resized_frame);

        // Draw results
        cv::Mat result = drawAllFaceAnalyses(resized_frame, analyses);

        // Print statistics periodically
        frame_count++;
        if (frame_count % print_interval == 0) {
            FaceDetectionSystem::Stats stats = face_system.getStats();
            printFaceDetectionStats(stats);

            // Print detailed analysis for each face
            for (size_t i = 0; i < analyses.size(); i++) {
                printFaceAnalysisStats(analyses[i], i);
            }
        }

        // Display result
        cv::imshow("Face Detection & Tracking", result);

        auto frame_end = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        // Print frame time every 100 frames
        if (frame_count % 100 == 0) {
            std::cout << "Frame " << frame_count << " processed in " << frame_time << " ms" << std::endl;
        }

        // Check for ESC key
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) {  // ESC
            break;
        }
    }

    // Clean up
    auto test_end = std::chrono::steady_clock::now();
    double test_duration = std::chrono::duration<double>(test_end - test_start).count();

    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "Total runtime: " << test_duration << " seconds" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Average FPS: " << frame_count / test_duration << std::endl;

    // Print final statistics
    FaceDetectionSystem::Stats stats = face_system.getStats();
    printFaceDetectionStats(stats);

    // Stop systems
    face_system.stop();
    if (use_camera) {
        camera.stop();
    }

    cv::destroyAllWindows();

    return 0;
}
