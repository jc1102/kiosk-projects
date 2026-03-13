/**
 * @file face_detection.h
 * @brief Real-time face detection and tracking system for Kiosk Teacher Assistant
 * @author Zane - C++/CV Engineer
 */

#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <vector>
#include <memory>
#include <mutex>
#include <chrono>
#include <unordered_map>

namespace kiosk {
namespace cv {

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief Face detection result with bounding box
 */
struct FaceDetection {
    cv::Rect bbox;              ///< Bounding box (x, y, width, height)
    float confidence;           ///< Detection confidence [0, 1]
    int face_id;                ///< Unique face ID (-1 if not tracked)
    cv::Point2f center;         ///< Face center point
    cv::Size size;              ///< Face size
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Face landmarks (68 points)
 */
struct FaceLandmarks {
    std::vector<cv::Point2f> points;  ///< 68 landmark points
    float left_eye_openness;          ///< Left eye openness
    float right_eye_openness;         ///< Right eye openness
    float mouth_openness;             ///< Mouth openness
    float head_roll;                  ///< Head roll angle (degrees)
    float head_pitch;                 ///< Head pitch angle (degrees)
    float head_yaw;                   ///< Head yaw angle (degrees)
};

/**
 * @brief Emotion classification result
 */
enum class Emotion {
    HAPPY,
    SAD,
    NEUTRAL,
    CONFUSED,
    ENGAGED,
    UNKNOWN
};

struct EmotionResult {
    Emotion emotion;
    float confidence;
    std::unordered_map<Emotion, float> probabilities;  ///< All emotion probabilities
};

/**
 * @brief Attention/focus tracking result
 */
struct AttentionResult {
    float attention_score;      ///< Attention score [0, 1]
    float engagement_level;     ///< Engagement level [0, 1]
    bool looking_at_screen;     ///< True if looking at teacher/board
    bool is_distracted;         ///< True if not paying attention
    float distraction_duration; ///< How long they've been distracted (seconds)
};

/**
 * @brief Complete face analysis result
 */
struct FaceAnalysis {
    FaceDetection detection;
    FaceLandmarks landmarks;
    EmotionResult emotion;
    AttentionResult attention;
    bool is_valid;              ///< True if all analyses completed successfully
};

/**
 * @brief Face tracking state (for DeepSORT-like tracking)
 */
struct Track {
    int track_id;                           ///< Unique track ID
    cv::Rect bbox;                          ///< Current bounding box
    std::vector<cv::Rect> history;          ///< Position history (for smoothing)
    std::chrono::system_clock::time_point first_seen;  ///< First seen timestamp
    std::chrono::system_clock::time_point last_seen;   ///< Last seen timestamp
    int age;                                ///< Number of frames since first seen
    int time_since_update;                  ///< Frames since last update
    int hit_streak;                         ///< Consecutive frames with detection
    int state;                              ///< Track state (0: tentative, 1: confirmed, 2: deleted)
    cv::Mat feature;                        ///< Feature vector for re-identification
    float confidence;                       ///< Detection confidence
};

// ============================================================================
// Configuration
// ============================================================================

struct FaceDetectionConfig {
    // Model paths (relative to workspace)
    std::string face_detection_model = "models/face_detection.onnx";
    std::string landmarks_model = "models/face_landmarks.onnx";
    std::string emotion_model = "models/emotion_recognition.onnx";

    // Detection parameters
    float confidence_threshold = 0.7f;      ///< Minimum detection confidence
    float nms_threshold = 0.4f;             ///< Non-maximum suppression threshold
    int input_size = 640;                    ///< Input size for face detection model
    bool use_gpu = true;                    ///< Use CUDA for inference
    bool use_tensorrt = true;               ///< Use TensorRT optimization

    // Tracking parameters
    float max_distance = 0.7f;             ///< Maximum distance for track matching
    int max_age = 30;                        ///< Maximum frames to keep lost tracks
    int n_init = 3;                          ///< Number of detections to confirm track

    // Landmark parameters
    bool enable_landmarks = true;           ///< Enable landmark extraction
    bool enable_emotion = true;             ///< Enable emotion recognition
    bool enable_attention = true;           ///< Enable attention tracking

    // Performance targets
    float target_fps = 30.0f;               ///< Target processing FPS
    float max_latency_ms = 50.0f;           ///< Maximum allowed latency per frame
};

// ============================================================================
// Face Detection System
// ============================================================================

class FaceDetectionSystem {
public:
    FaceDetectionSystem();
    ~FaceDetectionSystem();

    // Initialization
    bool initialize(const FaceDetectionConfig& config);
    bool start();
    void stop();
    bool isRunning() const;

    // Processing
    std::vector<FaceDetection> detectFaces(const cv::Mat& frame);
    std::vector<FaceAnalysis> analyzeFrame(const cv::Mat& frame);
    FaceAnalysis analyzeFace(const cv::Mat& face_image);

    // Tracking
    std::vector<Track> updateTracks(const std::vector<FaceDetection>& detections);
    void resetTracks();

    // Configuration
    FaceDetectionConfig getConfig() const;
    bool updateConfig(const FaceDetectionConfig& config);

    // Statistics
    struct Stats {
        uint64_t frames_processed;
        uint64_t faces_detected;
        uint64_t tracks_active;
        double avg_detection_time_ms;
        double avg_tracking_time_ms;
        double avg_analysis_time_ms;
        double current_fps;
        double max_latency_ms;
    };

    Stats getStats() const;
    void resetStats();

private:
    // Core detection
    std::vector<cv::Rect> detectFacesRaw(const cv::Mat& frame);
    cv::Mat preprocessForDetection(const cv::Mat& frame);
    std::vector<int> nmsBoxes(const std::vector<cv::Rect>& boxes,
                              const std::vector<float>& scores,
                              float nms_threshold);

    // Landmark extraction
    FaceLandmarks extractLandmarks(const cv::Mat& face_image, const cv::Rect& bbox);
    cv::Point2f getPoint(const std::vector<cv::Point2f>& landmarks, int index);

    // Emotion recognition
    EmotionResult classifyEmotion(const cv::Mat& face_image, const FaceLandmarks& landmarks);

    // Attention tracking
    AttentionResult trackAttention(const FaceLandmarks& landmarks,
                                    const std::vector<cv::Point2f>& history);

    // Tracking (SORT/DeepSORT)
    std::vector<Track> updateTracksImpl(const std::vector<FaceDetection>& detections);
    cv::Mat extractFeature(const cv::Mat& face_image);
    float calculateDistance(const Track& track, const FaceDetection& detection);
    void updateTrack(Track& track, const FaceDetection& detection);

    // Statistics
    void updateLatency(double elapsed_ms);

private:
    FaceDetectionConfig config_;
    bool initialized_;
    bool running_;

    // DNN models
    cv::Ptr<cv::dnn::Net> face_detector_;
    cv::Ptr<cv::dnn::Net> landmarks_extractor_;
    cv::Ptr<cv::dnn::Net> emotion_classifier_;

    // Tracking
    std::vector<Track> tracks_;
    int next_track_id_;
    std::mutex tracks_mutex_;

    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    std::chrono::steady_clock::time_point start_time_;
};

// ============================================================================
// Utility Functions
// ============================================================================

std::string emotionToString(Emotion emotion);
cv::Scalar emotionToColor(Emotion emotion);
cv::Mat drawFaceAnalysis(const cv::Mat& frame, const FaceAnalysis& analysis);
cv::Mat drawAllFaceAnalyses(const cv::Mat& frame, const std::vector<FaceAnalysis>& analyses);

} // namespace cv
} // namespace kiosk

#endif // FACE_DETECTION_H
