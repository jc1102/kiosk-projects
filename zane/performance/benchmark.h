/**
 * @file benchmark.h
 * @brief Performance benchmarking framework for Kiosk CV systems
 * @author Zane - C++/CV Engineer
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#ifdef OPENCV_AVAILABLE
#include <opencv2/opencv.hpp>
#else
// Define minimal cv::Size for non-OpenCV builds
namespace cv {
    struct Size {
        int width;
        int height;
        Size(int w = 0, int h = 0) : width(w), height(h) {}
    };
    struct Mat {};  // Dummy Mat for non-OpenCV builds
}
#endif

#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

namespace kiosk {
namespace benchmark {

// ============================================================================
// Benchmark Metrics
// ============================================================================

/**
 * @brief Single timing measurement
 */
struct TimingResult {
    double duration_ms;           ///< Duration in milliseconds
    std::chrono::system_clock::time_point timestamp;
    std::string label;            ///< Label for this measurement
};

/**
 * @brief Resource usage snapshot
 */
struct ResourceSnapshot {
    double cpu_percent;           ///< CPU usage percentage
    double memory_mb;             ///< Memory usage in MB
    double gpu_memory_mb;         ///< GPU memory usage in MB (if available)
    double gpu_utilization;       ///< GPU utilization percentage (if available)
    double temperature_c;         ///< Temperature in Celsius (if available)
    double power_w;               ///< Power consumption in watts (if available)
    std::chrono::system_clock::time_point timestamp;
};

/**
 * @brief Frame processing metrics
 */
struct FrameMetrics {
    int frame_number;
    double processing_time_ms;
    double end_to_end_latency_ms;
    int faces_detected;
    int active_tracks;
    double fps;
    ResourceSnapshot resources;
};

/**
 * @brief Benchmark result summary
 */
struct BenchmarkSummary {
    std::string benchmark_name;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    double total_duration_sec;

    // Frame metrics
    int total_frames;
    double avg_fps;
    double min_fps;
    double max_fps;
    double p50_fps;
    double p95_fps;
    double p99_fps;

    // Latency metrics
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    double p50_latency_ms;
    double p95_latency_ms;
    double p99_latency_ms;

    // Resource metrics (averages)
    double avg_cpu_percent;
    double avg_memory_mb;
    double avg_gpu_memory_mb;
    double avg_gpu_utilization;
    double max_temperature_c;

    // Detection metrics
    double avg_faces_per_frame;
    int max_faces_in_frame;
    int total_faces_detected;

    // Frame drops
    int frame_drops;
    double drop_rate;

    bool success;
    std::string error_message;
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    std::string name;                    ///< Benchmark name
    int duration_seconds;               ///< Duration to run (0 = until manual stop)
    int num_frames;                     ///< Number of frames to process (0 = duration-based)
    std::string camera_device;          ///< Camera device path (empty = no camera)
    cv::Size resolution;                ///< Input resolution
    bool enable_gpu;                    ///< Enable GPU
    bool save_results_csv;              ///< Save results to CSV
    std::string results_dir;            ///< Directory to save results
    bool enable_profiling;              ///< Enable detailed profiling
    int profiling_interval;             ///< Profile every N frames

    // Warmup
    int warmup_frames;                  ///< Number of warmup frames (not counted)

    // Resource monitoring
    bool enable_cpu_monitoring;         ///< Monitor CPU usage
    bool enable_memory_monitoring;      ///< Monitor memory usage
    bool enable_gpu_monitoring;        ///< Monitor GPU usage

    // Output
    bool print_progress;                ///< Print progress during benchmark
    bool print_summary;                 ///< Print summary at end
    bool generate_plots;                ///< Generate performance plots (requires matplotlib)
};

// ============================================================================
// Benchmark Class
// ============================================================================

class Benchmark {
public:
    Benchmark();
    ~Benchmark();

    // Configuration
    bool configure(const BenchmarkConfig& config);
    BenchmarkConfig getConfig() const;

    // Execution
    bool run();
    void stop();
    bool isRunning() const;

    // Results
    BenchmarkSummary getSummary() const;
    std::vector<FrameMetrics> getFrameMetrics() const;

    // Export
    bool exportToCSV(const std::string& filepath) const;
    bool exportSummary(const std::string& filepath) const;
    bool generateReport(const std::string& filepath) const;

private:
    // Timing utilities
    void startTimer(const std::string& label);
    double stopTimer(const std::string& label);
    double getCurrentTimeMs() const;

    // Resource monitoring
    ResourceSnapshot captureResourceSnapshot();
    void updateResourceMetrics();

    // Frame processing
    void processFrame(const cv::Mat& frame, int frame_num);
    void updateStatistics(double latency_ms, int faces_detected, int tracks_active);

    // Progress reporting
    void printProgress(int current_frame, int total_frames);
    void printSummary() const;

    // Percentile calculation
    double calculatePercentile(const std::vector<double>& values, double percentile) const;

private:
    BenchmarkConfig config_;
    bool running_;
    bool should_stop_;

    // Timing
    std::map<std::string, std::chrono::steady_clock::time_point> timers_;

    // Metrics storage
    std::vector<FrameMetrics> frame_metrics_;
    std::vector<double> latencies_;
    std::vector<double> fps_values_;

    // Summary
    BenchmarkSummary summary_;
    int frames_processed_;

    // Resource monitoring
    ResourceSnapshot baseline_resources_;
};

// ============================================================================
// Benchmark Suite (Multiple Benchmarks)
// ============================================================================

class BenchmarkSuite {
public:
    BenchmarkSuite();
    ~BenchmarkSuite();

    void addBenchmark(const Benchmark& benchmark);
    void addBenchmarkConfig(const BenchmarkConfig& config);

    bool runAll();
    bool runBenchmark(size_t index);
    bool runBenchmark(const std::string& name);

    std::vector<BenchmarkSummary> getAllSummaries() const;

    bool exportAllToCSV(const std::string& directory) const;
    bool generateComparisonReport(const std::string& filepath) const;

private:
    std::vector<Benchmark> benchmarks_;
};

// ============================================================================
// Utility Functions
// ============================================================================

// System information
struct SystemInfo {
    std::string os_name;
    std::string os_version;
    std::string cpu_model;
    int cpu_cores;
    std::string gpu_model;
    double gpu_memory_gb;
    double system_memory_gb;
    std::string opencv_version;
    bool cuda_available;
};

SystemInfo getSystemInfo();
std::string systemInfoToString(const SystemInfo& info);

// CSV export helpers
std::string escapeCSV(const std::string& value);

// Progress bar
void printProgressBar(int current, int total, int width = 50);

// Timestamp formatting
std::string getCurrentTimestamp();
std::string formatDuration(double seconds);

} // namespace benchmark
} // namespace kiosk

#endif // BENCHMARK_H
