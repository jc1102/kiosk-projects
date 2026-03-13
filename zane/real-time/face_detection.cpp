/**
 * @file face_detection.cpp
 * @brief Implementation of real-time face detection and tracking system
 * @author Zane - C++/CV Engineer
 */

#include "face_detection.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace kiosk {
namespace cv {

// ============================================================================
// FaceDetectionSystem Implementation
// ============================================================================

FaceDetectionSystem::FaceDetectionSystem()
    : initialized_(false)
    , running_(false)
    , next_track_id_(0) {

    // Initialize stats
    stats_ = {
        .frames_processed = 0,
        .faces_detected = 0,
        .tracks_active = 0,
        .avg_detection_time_ms = 0.0,
        .avg_tracking_time_ms = 0.0,
        .avg_analysis_time_ms = 0.0,
        .current_fps = 0.0,
        .max_latency_ms = 0.0
    };
    start_time_ = std::chrono::steady_clock::now();
}

FaceDetectionSystem::~FaceDetectionSystem() {
    stop();
}

bool FaceDetectionSystem::initialize(const FaceDetectionConfig& config) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (initialized_) {
        std::cerr << "FaceDetectionSystem already initialized" << std::endl;
        return false;
    }

    config_ = config;

    try {
        // Initialize face detection model
        // Note: Using OpenCV's DNN module with a pre-trained model
        // In production, we'd use TensorRT-optimized models on Jetson
        face_detector_ = cv::dnn::readNetFromONNX(config_.face_detection_model);

        if (face_detector_.empty()) {
            std::cerr << "Failed to load face detection model: " << config_.face_detection_model << std::endl;
            // Continue with alternative model for development
            std::cout << "Note: Using OpenCV Haar Cascade as fallback for development" << std::endl;
        } else {
            // Configure backend
            if (config_.use_gpu) {
                face_detector_->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                face_detector_->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } else {
                face_detector_->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                face_detector_->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }

        // Initialize landmarks extractor (placeholder - would use MediaPipe or dlib)
        if (config_.enable_landmarks) {
            // In production, load MediaPipe or dlib model here
            std::cout << "Face landmarks extraction enabled" << std::endl;
        }

        // Initialize emotion classifier (placeholder - would load custom CNN)
        if (config_.enable_emotion) {
            // In production, load emotion recognition model here
            std::cout << "Emotion recognition enabled" << std::endl;
        }

        initialized_ = true;
        start_time_ = std::chrono::steady_clock::now();
        std::cout << "FaceDetectionSystem initialized successfully" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing FaceDetectionSystem: " << e.what() << std::endl;
        return false;
    }
}

bool FaceDetectionSystem::start() {
    if (!initialized_) {
        std::cerr << "FaceDetectionSystem not initialized" << std::endl;
        return false;
    }

    running_ = true;
    std::cout << "FaceDetectionSystem started" << std::endl;
    return true;
}

void FaceDetectionSystem::stop() {
    running_ = false;
    std::cout << "FaceDetectionSystem stopped" << std::endl;
}

bool FaceDetectionSystem::isRunning() const {
    return running_;
}

std::vector<FaceDetection> FaceDetectionSystem::detectFaces(const cv::Mat& frame) {
    auto start = std::chrono::steady_clock::now();

    std::vector<FaceDetection> detections;

    if (frame.empty()) {
        std::cerr << "Empty frame provided to detectFaces" << std::endl;
        return detections;
    }

    // Detect faces
    std::vector<cv::Rect> bboxes = detectFacesRaw(frame);

    // Convert to FaceDetection structures
    auto timestamp = std::chrono::system_clock::now();
    for (const auto& bbox : bboxes) {
        FaceDetection detection;
        detection.bbox = bbox;
        detection.confidence = 0.9f; // Placeholder - would use actual model confidence
        detection.face_id = -1;      // Will be assigned by tracking
        detection.center = cv::Point2f(bbox.x + bbox.width / 2.0f,
                                       bbox.y + bbox.height / 2.0f);
        detection.size = cv::Size(bbox.width, bbox.height);
        detection.timestamp = timestamp;
        detections.push_back(detection);
    }

    // Update statistics
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    updateLatency(elapsed_ms);

    stats_.frames_processed++;
    stats_.faces_detected += detections.size();

    return detections;
}

std::vector<FaceAnalysis> FaceDetectionSystem::analyzeFrame(const cv::Mat& frame) {
    auto start = std::chrono::steady_clock::now();

    std::vector<FaceAnalysis> analyses;

    if (frame.empty()) {
        return analyses;
    }

    // Detect faces
    std::vector<FaceDetection> detections = detectFaces(frame);

    // Update tracks
    std::vector<Track> tracks = updateTracks(detections);

    // Analyze each face
    for (size_t i = 0; i < detections.size(); i++) {
        FaceAnalysis analysis;
        analysis.detection = detections[i];

        // Assign track ID
        if (i < tracks.size()) {
            analysis.detection.face_id = tracks[i].track_id;
        }

        // Extract face ROI
        cv::Mat face_roi = frame(detections[i].bbox);

        // Extract landmarks
        if (config_.enable_landmarks) {
            analysis.landmarks = extractLandmarks(face_roi, detections[i].bbox);
        }

        // Classify emotion
        if (config_.enable_emotion) {
            analysis.emotion = classifyEmotion(face_roi, analysis.landmarks);
        }

        // Track attention
        if (config_.enable_attention) {
            analysis.attention = trackAttention(analysis.landmarks,
                                                 tracks[i < tracks.size() ? i : 0].history.size() > 0 ?
                                                 std::vector<cv::Point2f>() : std::vector<cv::Point2f>());
        }

        analysis.is_valid = true;
        analyses.push_back(analysis);
    }

    // Update statistics
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (stats_.avg_analysis_time_ms == 0.0) {
        stats_.avg_analysis_time_ms = elapsed_ms;
    } else {
        stats_.avg_analysis_time_ms = 0.9 * stats_.avg_analysis_time_ms + 0.1 * elapsed_ms;
    }

    return analyses;
}

FaceAnalysis FaceDetectionSystem::analyzeFace(const cv::Mat& face_image) {
    FaceAnalysis analysis;

    if (face_image.empty()) {
        analysis.is_valid = false;
        return analysis;
    }

    analysis.detection.bbox = cv::Rect(0, 0, face_image.cols, face_image.rows);
    analysis.detection.confidence = 1.0f;
    analysis.detection.center = cv::Point2f(face_image.cols / 2.0f, face_image.rows / 2.0f);
    analysis.detection.size = cv::Size(face_image.cols, face_image.rows);

    // Extract landmarks
    if (config_.enable_landmarks) {
        analysis.landmarks = extractLandmarks(face_image, cv::Rect(0, 0, face_image.cols, face_image.rows));
    }

    // Classify emotion
    if (config_.enable_emotion) {
        analysis.emotion = classifyEmotion(face_image, analysis.landmarks);
    }

    analysis.is_valid = true;
    return analysis;
}

std::vector<Track> FaceDetectionSystem::updateTracks(const std::vector<FaceDetection>& detections) {
    auto start = std::chrono::steady_clock::now();
    auto tracks = updateTracksImpl(detections);

    // Update statistics
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::lock_guard<std::mutex> lock(stats_mutex_);
    if (stats_.avg_tracking_time_ms == 0.0) {
        stats_.avg_tracking_time_ms = elapsed_ms;
    } else {
        stats_.avg_tracking_time_ms = 0.9 * stats_.avg_tracking_time_ms + 0.1 * elapsed_ms;
    }

    return tracks;
}

void FaceDetectionSystem::resetTracks() {
    std::lock_guard<std::mutex> lock(tracks_mutex_);
    tracks_.clear();
    next_track_id_ = 0;
}

FaceDetectionConfig FaceDetectionSystem::getConfig() const {
    return config_;
}

bool FaceDetectionSystem::updateConfig(const FaceDetectionConfig& config) {
    config_ = config;
    return true;
}

FaceDetectionSystem::Stats FaceDetectionSystem::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    Stats stats = stats_;

    // Update current FPS
    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time_).count();
    if (elapsed > 0) {
        stats.current_fps = stats.frames_processed / elapsed;
    }

    // Update active tracks count
    stats.tracks_active = tracks_.size();

    return stats;
}

void FaceDetectionSystem::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = {
        .frames_processed = 0,
        .faces_detected = 0,
        .tracks_active = 0,
        .avg_detection_time_ms = 0.0,
        .avg_tracking_time_ms = 0.0,
        .avg_analysis_time_ms = 0.0,
        .current_fps = 0.0,
        .max_latency_ms = 0.0
    };
    start_time_ = std::chrono::steady_clock::now();
}

// ============================================================================
// Private Methods
// ============================================================================

std::vector<cv::Rect> FaceDetectionSystem::detectFacesRaw(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;

    try {
        // Use OpenCV Haar Cascade as fallback for development
        // In production, this would use the loaded DNN model
        static cv::CascadeClassifier haar_cascade;
        static bool cascade_loaded = false;

        if (!cascade_loaded) {
            // Load OpenCV's built-in Haar cascade
            std::string cascade_path = cv::samples::findFile("haarcascade_frontalface_alt.xml");
            if (haar_cascade.load(cascade_path)) {
                cascade_loaded = true;
                std::cout << "Loaded Haar Cascade for face detection" << std::endl;
            } else {
                std::cerr << "Warning: Could not load Haar Cascade" << std::endl;
            }
        }

        if (cascade_loaded) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);

            haar_cascade.detectMultiScale(gray, faces,
                                          1.1,  // scaleFactor
                                          3,    // minNeighbors
                                          0,
                                          cv::Size(30, 30));  // minSize

            // Filter by confidence threshold (simulated)
            std::vector<cv::Rect> filtered;
            for (const auto& face : faces) {
                // Simple size-based confidence simulation
                float size_score = std::min(1.0f, (face.width * face.height) / 10000.0f);
                if (size_score >= config_.confidence_threshold) {
                    filtered.push_back(face);
                }
            }
            faces = filtered;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in detectFacesRaw: " << e.what() << std::endl;
    }

    return faces;
}

cv::Mat FaceDetectionSystem::preprocessForDetection(const cv::Mat& frame) {
    cv::Mat processed;

    // Resize to model input size
    cv::resize(frame, processed, cv::Size(config_.input_size, config_.input_size));

    // Normalize to [0, 1] and convert to blob
    cv::Mat blob;
    cv::dnn::blobFromImage(processed, blob, 1.0 / 255.0,
                           cv::Size(config_.input_size, config_.input_size),
                           cv::Scalar(0, 0, 0), true, false);

    return blob;
}

std::vector<int> FaceDetectionSystem::nmsBoxes(const std::vector<cv::Rect>& boxes,
                                               const std::vector<float>& scores,
                                               float nms_threshold) {
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, config_.confidence_threshold, nms_threshold, indices);
    return indices;
}

FaceLandmarks FaceDetectionSystem::extractLandmarks(const cv::Mat& face_image,
                                                       const cv::Rect& bbox) {
    FaceLandmarks landmarks;
    landmarks.points.resize(68);

    // Placeholder: In production, this would use MediaPipe or dlib
    // For now, generate approximate landmarks based on face bbox
    cv::Rect inner = bbox;
    inner.x += bbox.width * 0.1f;
    inner.y += bbox.height * 0.1f;
    inner.width *= 0.8f;
    inner.height *= 0.8f;

    // Generate 68 points in a grid pattern (placeholder)
    for (int i = 0; i < 68; i++) {
        float row = (i / 10) / 6.0f;
        float col = (i % 10) / 9.0f;

        landmarks.points[i].x = inner.x + col * inner.width;
        landmarks.points[i].y = inner.y + row * inner.height;
    }

    // Calculate eye openness (placeholder)
    landmarks.left_eye_openness = 0.8f;
    landmarks.right_eye_openness = 0.75f;
    landmarks.mouth_openness = 0.2f;

    // Calculate head pose (placeholder)
    landmarks.head_roll = 0.0f;
    landmarks.head_pitch = 0.0f;
    landmarks.head_yaw = 0.0f;

    return landmarks;
}

cv::Point2f FaceDetectionSystem::getPoint(const std::vector<cv::Point2f>& landmarks, int index) {
    if (index >= 0 && index < static_cast<int>(landmarks.size())) {
        return landmarks[index];
    }
    return cv::Point2f(0, 0);
}

EmotionResult FaceDetectionSystem::classifyEmotion(const cv::Mat& face_image,
                                                     const FaceLandmarks& landmarks) {
    EmotionResult result;

    // Placeholder: In production, this would use a custom CNN model
    // For now, use a simple rule-based approach based on landmarks

    // Initialize probabilities (uniform distribution)
    result.probabilities[Emotion::HAPPY] = 0.2f;
    result.probabilities[Emotion::SAD] = 0.2f;
    result.probabilities[Emotion::NEUTRAL] = 0.2f;
    result.probabilities[Emotion::CONFUSED] = 0.2f;
    result.probabilities[Emotion::ENGAGED] = 0.2f;

    // Simple heuristic based on mouth openness and eye openness
    float mouth_score = landmarks.mouth_openness;
    float eye_avg = (landmarks.left_eye_openness + landmarks.right_eye_openness) / 2.0f;

    if (mouth_score > 0.3f) {
        result.probabilities[Emotion::HAPPY] += 0.4f;
        result.probabilities[Emotion::ENGAGED] += 0.2f;
    } else if (eye_avg < 0.5f) {
        result.probabilities[Emotion::SAD] += 0.3f;
        result.probabilities[Emotion::CONFUSED] += 0.2f;
    } else {
        result.probabilities[Emotion::NEUTRAL] += 0.3f;
        result.probabilities[Emotion::ENGAGED] += 0.2f;
    }

    // Normalize probabilities
    float sum = 0.0f;
    for (auto& pair : result.probabilities) {
        sum += pair.second;
    }
    for (auto& pair : result.probabilities) {
        pair.second /= sum;
    }

    // Find top emotion
    auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
    result.emotion = max_it->first;
    result.confidence = max_it->second;

    return result;
}

AttentionResult FaceDetectionSystem::trackAttention(const FaceLandmarks& landmarks,
                                                     const std::vector<cv::Point2f>& history) {
    AttentionResult result;

    // Placeholder: Calculate attention based on head pose and gaze
    // In production, this would use gaze estimation models

    // Simple heuristic based on head orientation
    float head_deviation = std::abs(landmarks.head_yaw) + std::abs(landmarks.head_pitch);

    if (head_deviation < 15.0f) {
        result.attention_score = 0.9f;
        result.looking_at_screen = true;
    } else if (head_deviation < 30.0f) {
        result.attention_score = 0.6f;
        result.looking_at_screen = true;
    } else {
        result.attention_score = 0.3f;
        result.looking_at_screen = false;
    }

    result.engagement_level = result.attention_score * 0.9f + 0.1f;
    result.is_distracted = result.attention_score < 0.4f;
    result.distraction_duration = result.is_distracted ? 1.0f : 0.0f;

    return result;
}

std::vector<Track> FaceDetectionSystem::updateTracksImpl(const std::vector<FaceDetection>& detections) {
    std::lock_guard<std::mutex> lock(tracks_mutex_);

    // This is a simplified SORT-like tracking algorithm
    // In production, use full DeepSORT with appearance features

    std::vector<Track> updated_tracks;

    // 1. Predict next positions (Kalman filter placeholder)
    for (auto& track : tracks_) {
        // Simple prediction: use velocity to predict next position
        if (track.history.size() >= 2) {
            cv::Rect last = track.history.back();
            cv::Rect prev = track.history[track.history.size() - 2];

            float dx = last.x - prev.x;
            float dy = last.y - prev.y;

            track.bbox.x += static_cast<int>(dx);
            track.bbox.y += static_cast<int>(dy);
        }

        track.time_since_update++;
    }

    // 2. Match detections to tracks (Hungarian algorithm placeholder)
    std::vector<bool> detection_matched(detections.size(), false);
    std::vector<bool> track_matched(tracks_.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        for (size_t j = 0; j < tracks_.size(); j++) {
            if (track_matched[j]) continue;

            float distance = calculateDistance(tracks_[j], detections[i]);

            if (distance < config_.max_distance) {
                // Match found
                updateTrack(tracks_[j], detections[i]);
                track_matched[j] = true;
                detection_matched[i] = true;
                break;
            }
        }
    }

    // 3. Create new tracks for unmatched detections
    for (size_t i = 0; i < detections.size(); i++) {
        if (!detection_matched[i]) {
            Track new_track;
            new_track.track_id = next_track_id_++;
            new_track.bbox = detections[i].bbox;
            new_track.first_seen = detections[i].timestamp;
            new_track.last_seen = detections[i].timestamp;
            new_track.age = 1;
            new_track.time_since_update = 0;
            new_track.hit_streak = 1;
            new_track.state = 0;  // Tentative
            new_track.confidence = detections[i].confidence;
            new_track.history.push_back(detections[i].bbox);
            tracks_.push_back(new_track);
        }
    }

    // 4. Remove old tracks
    auto it = tracks_.begin();
    while (it != tracks_.end()) {
        if (it->time_since_update > config_.max_age) {
            it = tracks_.erase(it);
        } else {
            it++;
        }
    }

    // 5. Return confirmed tracks
    for (const auto& track : tracks_) {
        if (track.state == 1) {  // Confirmed
            updated_tracks.push_back(track);
        }
    }

    stats_.tracks_active = tracks_.size();

    return updated_tracks;
}

cv::Mat FaceDetectionSystem::extractFeature(const cv::Mat& face_image) {
    // Placeholder: In production, this would extract appearance features
    // using a CNN for re-identification (DeepSORT)

    cv::Mat feature;
    cv::resize(face_image, feature, cv::Size(128, 128));
    cv::cvtColor(feature, feature, cv::COLOR_BGR2GRAY);
    feature = feature.reshape(1, 1);
    feature.convertTo(feature, CV_32F);

    return feature;
}

float FaceDetectionSystem::calculateDistance(const Track& track, const FaceDetection& detection) {
    // IoU (Intersection over Union) distance
    cv::Rect track_bbox = track.bbox;
    cv::Rect det_bbox = detection.bbox;

    cv::Rect intersection = track_bbox & det_bbox;
    float intersection_area = intersection.area();

    float union_area = track_bbox.area() + det_bbox.area() - intersection_area;

    if (union_area == 0.0f) {
        return 1.0f;  // Maximum distance
    }

    float iou = intersection_area / union_area;
    return 1.0f - iou;  // Convert IoU to distance
}

void FaceDetectionSystem::updateTrack(Track& track, const FaceDetection& detection) {
    track.bbox = detection.bbox;
    track.last_seen = detection.timestamp;
    track.time_since_update = 0;
    track.age++;
    track.hit_streak++;

    // Update history (keep last 10 positions)
    track.history.push_back(detection.bbox);
    if (track.history.size() > 10) {
        track.history.erase(track.history.begin());
    }

    // Promote to confirmed if enough hits
    if (track.state == 0 && track.hit_streak >= config_.n_init) {
        track.state = 1;  // Confirmed
    }
}

void FaceDetectionSystem::updateLatency(double elapsed_ms) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    if (elapsed_ms > stats_.max_latency_ms) {
        stats_.max_latency_ms = elapsed_ms;
    }

    if (stats_.avg_detection_time_ms == 0.0) {
        stats_.avg_detection_time_ms = elapsed_ms;
    } else {
        stats_.avg_detection_time_ms = 0.9 * stats_.avg_detection_time_ms + 0.1 * elapsed_ms;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string emotionToString(Emotion emotion) {
    switch (emotion) {
        case Emotion::HAPPY:    return "Happy";
        case Emotion::SAD:      return "Sad";
        case Emotion::NEUTRAL:  return "Neutral";
        case Emotion::CONFUSED: return "Confused";
        case Emotion::ENGAGED:  return "Engaged";
        default:                return "Unknown";
    }
}

cv::Scalar emotionToColor(Emotion emotion) {
    switch (emotion) {
        case Emotion::HAPPY:    return cv::Scalar(0, 255, 0);    // Green
        case Emotion::SAD:      return cv::Scalar(255, 0, 0);    // Blue
        case Emotion::NEUTRAL:  return cv::Scalar(128, 128, 128); // Gray
        case Emotion::CONFUSED: return cv::Scalar(0, 165, 255);  // Orange
        case Emotion::ENGAGED:  return cv::Scalar(0, 255, 255);  // Yellow
        default:                return cv::Scalar(255, 255, 255); // White
    }
}

cv::Mat drawFaceAnalysis(const cv::Mat& frame, const FaceAnalysis& analysis) {
    cv::Mat result = frame.clone();

    if (!analysis.is_valid) {
        return result;
    }

    // Draw bounding box
    cv::rectangle(result, analysis.detection.bbox,
                   emotionToColor(analysis.emotion.emotion), 2);

    // Draw face ID
    if (analysis.detection.face_id >= 0) {
        std::string id_text = "ID: " + std::to_string(analysis.detection.face_id);
        cv::putText(result, id_text,
                    cv::Point(analysis.detection.bbox.x, analysis.detection.bbox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // Draw emotion
    std::string emotion_text = emotionToString(analysis.emotion.emotion) +
                              " (" + std::to_string(static_cast<int>(analysis.emotion.confidence * 100)) + "%)";
    cv::putText(result, emotion_text,
                cv::Point(analysis.detection.bbox.x, analysis.detection.bbox.y + analysis.detection.bbox.height + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, emotionToColor(analysis.emotion.emotion), 2);

    // Draw attention score
    std::string attention_text = "Attention: " +
                                std::to_string(static_cast<int>(analysis.attention.attention_score * 100)) + "%";
    cv::Scalar attention_color = analysis.attention.is_distracted ?
                                  cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(result, attention_text,
                cv::Point(analysis.detection.bbox.x, analysis.detection.bbox.y + analysis.detection.bbox.height + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, attention_color, 2);

    // Draw landmarks
    if (analysis.landmarks.points.size() >= 68) {
        for (const auto& point : analysis.landmarks.points) {
            cv::circle(result, point, 1, cv::Scalar(0, 255, 255), -1);
        }
    }

    return result;
}

cv::Mat drawAllFaceAnalyses(const cv::Mat& frame, const std::vector<FaceAnalysis>& analyses) {
    cv::Mat result = frame.clone();

    for (const auto& analysis : analyses) {
        result = drawFaceAnalysis(result, analysis);
    }

    // Draw statistics
    std::string stats_text = "Faces: " + std::to_string(analyses.size());
    cv::putText(result, stats_text,
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    return result;
}

} // namespace cv
} // namespace kiosk
