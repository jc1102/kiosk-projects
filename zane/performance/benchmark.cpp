/**
 * @file benchmark.cpp
 * @brief Implementation of performance benchmarking framework
 * @author Zane - C++/CV Engineer
 */

#include "benchmark.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <fstream>
#include <ctime>
#include <sys/stat.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#elif __linux__
#include <sys/sysinfo.h>
#include <fstream>
#include <unistd.h>
#endif

// ============================================================================
// Benchmark Implementation
// ============================================================================

namespace kiosk {
namespace benchmark {

Benchmark::Benchmark()
    : running_(false)
    , should_stop_(false)
    , frames_processed_(0) {

    // Initialize summary
    summary_ = {
        .benchmark_name = "",
        .start_time = std::chrono::system_clock::now(),
        .end_time = std::chrono::system_clock::now(),
        .total_duration_sec = 0.0,

        .total_frames = 0,
        .avg_fps = 0.0,
        .min_fps = std::numeric_limits<double>::max(),
        .max_fps = std::numeric_limits<double>::lowest(),
        .p50_fps = 0.0,
        .p95_fps = 0.0,
        .p99_fps = 0.0,

        .avg_latency_ms = 0.0,
        .min_latency_ms = std::numeric_limits<double>::max(),
        .max_latency_ms = std::numeric_limits<double>::lowest(),
        .p50_latency_ms = 0.0,
        .p95_latency_ms = 0.0,
        .p99_latency_ms = 0.0,

        .avg_cpu_percent = 0.0,
        .avg_memory_mb = 0.0,
        .avg_gpu_memory_mb = 0.0,
        .avg_gpu_utilization = 0.0,
        .max_temperature_c = 0.0,

        .avg_faces_per_frame = 0.0,
        .max_faces_in_frame = 0,
        .total_faces_detected = 0,

        .frame_drops = 0,
        .drop_rate = 0.0,

        .success = false,
        .error_message = ""
    };
}

Benchmark::~Benchmark() {
    stop();
}

bool Benchmark::configure(const BenchmarkConfig& config) {
    config_ = config;

    // Validate configuration
    if (config.duration_seconds <= 0 && config.num_frames <= 0) {
        std::cerr << "Error: Must specify either duration_seconds or num_frames" << std::endl;
        return false;
    }

    // Create results directory if needed
    if (!config.results_dir.empty()) {
        #ifdef __linux__
        mkdir(config.results_dir.c_str(), 0755);
        #elif __APPLE__
        mkdir(config.results_dir.c_str(), 0755);
        #endif
    }

    // Capture baseline resources
    baseline_resources_ = captureResourceSnapshot();

    return true;
}

BenchmarkConfig Benchmark::getConfig() const {
    return config_;
}

bool Benchmark::run() {
    if (running_) {
        std::cerr << "Benchmark already running" << std::endl;
        return false;
    }

    running_ = true;
    should_stop_ = false;

    summary_.benchmark_name = config_.name;
    summary_.start_time = std::chrono::system_clock::now();

    std::cout << "\n=== Running Benchmark: " << config_.name << " ===" << std::endl;
    std::cout << "System Info:" << std::endl;
    std::cout << systemInfoToString(getSystemInfo()) << std::endl;
    std::cout << "\nStarting benchmark..." << std::endl;

    #ifdef OPENCV_AVAILABLE
    // Open video capture
    cv::VideoCapture cap;
    if (!config_.camera_device.empty()) {
        cap.open(config_.camera_device);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera " << config_.camera_device << std::endl;
            running_ = false;
            return false;
        }
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config_.resolution.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config_.resolution.height);
        cap.set(cv::CAP_PROP_FPS, 30.0);
    }

    // Warmup phase
    if (config_.warmup_frames > 0) {
        std::cout << "Warming up with " << config_.warmup_frames << " frames..." << std::endl;
        for (int i = 0; i < config_.warmup_frames && !should_stop_; i++) {
            cv::Mat frame;
            if (config_.camera_device.empty()) {
                // Simulated frame
                frame = cv::Mat::zeros(720, 1280, CV_8UC3);
            } else {
                cap >> frame;
            }

            if (frame.empty()) continue;

            // Simulate processing
            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(640, 480));
            cv::GaussianBlur(resized, resized, cv::Size(5, 5), 0);
        }
    }
    #else
    // Warmup phase (no OpenCV)
    if (config_.warmup_frames > 0) {
        std::cout << "Warming up with " << config_.warmup_frames << " frames..." << std::endl;
        if (!config_.camera_device.empty()) {
            std::cout << "Warning: Camera mode requires OpenCV. Using simulation mode." << std::endl;
        }
        for (int i = 0; i < config_.warmup_frames && !should_stop_; i++) {
            // Simulate processing work
            std::vector<double> dummy_data(1000);
            std::accumulate(dummy_data.begin(), dummy_data.end(), 0.0);
        }
    }
    #endif

    // Main benchmark loop
    int total_iterations = config_.num_frames > 0 ? config_.num_frames : 36000; // Default: 20 min @ 30fps
    int duration_frames = config_.duration_seconds > 0 ? config_.duration_seconds * 30 : total_iterations;
    int target_frames = std::min(total_iterations, duration_frames);

    auto benchmark_start = std::chrono::steady_clock::now();
    std::vector<double> frame_times;

    #ifdef OPENCV_AVAILABLE
    for (int frame_num = 0; frame_num < target_frames && !should_stop_; frame_num++) {
        auto frame_start = std::chrono::steady_clock::now();

        // Get frame
        cv::Mat frame;
        if (config_.camera_device.empty()) {
            // Simulated frame
            frame = cv::Mat::zeros(config_.resolution.height, config_.resolution.width, CV_8UC3);

            // Add simulated faces
            cv::circle(frame, cv::Point(300, 300), 80, cv::Scalar(200, 180, 150), -1);
            cv::circle(frame, cv::Point(700, 250), 90, cv::Scalar(210, 190, 160), -1);
        } else {
            cap >> frame;
        }

        if (frame.empty()) {
            std::cerr << "Warning: Empty frame at " << frame_num << std::endl;
            continue;
        }

        // Simulate face detection processing
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(640, 480));

        // Simulate face detection (Haar cascade would be real implementation)
        int faces_detected = 2; // Simulated
    #else
    // Simulation mode without OpenCV
    if (!config_.camera_device.empty()) {
        std::cout << "Note: Camera mode requires OpenCV. Running in simulation mode." << std::endl;
    }
    for (int frame_num = 0; frame_num < target_frames && !should_stop_; frame_num++) {
        auto frame_start = std::chrono::steady_clock::now();

        // Simulate processing work
        std::vector<double> dummy_data(5000);
        for (size_t i = 0; i < dummy_data.size(); i++) {
            dummy_data[i] = std::sin(i * 0.01) * std::cos(i * 0.02);
        }
        double result = std::accumulate(dummy_data.begin(), dummy_data.end(), 0.0);

        // Simulated faces
        int faces_detected = 2; // Simulated
        (void)result;  // Avoid unused variable warning
    #endif

        // Calculate latency
        auto frame_end = std::chrono::steady_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

        // Capture resources
        ResourceSnapshot resources = captureResourceSnapshot();

        // Calculate FPS
        double fps = 1000.0 / latency_ms;

        // Record frame metrics
        FrameMetrics metrics = {
            .frame_number = frame_num,
            .processing_time_ms = latency_ms,
            .end_to_end_latency_ms = latency_ms,
            .faces_detected = faces_detected,
            .active_tracks = faces_detected,
            .fps = fps,
            .resources = resources
        };

        frame_metrics_.push_back(metrics);
        latencies_.push_back(latency_ms);
        fps_values_.push_back(fps);

        frames_processed_++;
        frame_times.push_back(latency_ms);

        // Update summary statistics
        summary_.total_frames++;
        summary_.total_faces_detected += faces_detected;
        summary_.max_faces_in_frame = std::max(summary_.max_faces_in_frame, faces_detected);

        // Print progress
        if (config_.print_progress && (frame_num % 30 == 0)) {
            printProgress(frame_num + 1, target_frames);
        }
    }

    // Calculate final statistics
    summary_.end_time = std::chrono::system_clock::now();
    summary_.total_duration_sec =
        std::chrono::duration<double>(summary_.end_time - summary_.start_time).count();

    if (!latencies_.empty()) {
        std::sort(latencies_.begin(), latencies_.end());
        std::sort(fps_values_.begin(), fps_values_.end());

        summary_.avg_fps = std::accumulate(fps_values_.begin(), fps_values_.end(), 0.0) / fps_values_.size();
        summary_.min_fps = fps_values_.front();
        summary_.max_fps = fps_values_.back();
        summary_.p50_fps = calculatePercentile(fps_values_, 50.0);
        summary_.p95_fps = calculatePercentile(fps_values_, 95.0);
        summary_.p99_fps = calculatePercentile(fps_values_, 99.0);

        summary_.avg_latency_ms = std::accumulate(latencies_.begin(), latencies_.end(), 0.0) / latencies_.size();
        summary_.min_latency_ms = latencies_.front();
        summary_.max_latency_ms = latencies_.back();
        summary_.p50_latency_ms = calculatePercentile(latencies_, 50.0);
        summary_.p95_latency_ms = calculatePercentile(latencies_, 95.0);
        summary_.p99_latency_ms = calculatePercentile(latencies_, 99.0);
    }

    summary_.avg_faces_per_frame = static_cast<double>(summary_.total_faces_detected) /
                                    std::max(1, summary_.total_frames);

    // Calculate average resource usage
    double total_cpu = 0.0, total_mem = 0.0;
    for (const auto& metrics : frame_metrics_) {
        total_cpu += metrics.resources.cpu_percent;
        total_mem += metrics.resources.memory_mb;
        summary_.max_temperature_c = std::max(summary_.max_temperature_c,
                                               metrics.resources.temperature_c);
    }

    if (!frame_metrics_.empty()) {
        summary_.avg_cpu_percent = total_cpu / frame_metrics_.size();
        summary_.avg_memory_mb = total_mem / frame_metrics_.size();
    }

    summary_.success = true;

    running_ = false;

    // Print summary
    if (config_.print_summary) {
        printSummary();
    }

    // Export results
    if (config_.save_results_csv) {
        std::string csv_path = config_.results_dir + "/" + config_.name + "_results.csv";
        if (!exportToCSV(csv_path)) {
            std::cerr << "Warning: Could not export results to CSV" << std::endl;
        }
    }

    std::cout << "\nBenchmark completed successfully!" << std::endl;

    return true;
}

void Benchmark::stop() {
    should_stop_ = true;
}

bool Benchmark::isRunning() const {
    return running_;
}

BenchmarkSummary Benchmark::getSummary() const {
    return summary_;
}

std::vector<FrameMetrics> Benchmark::getFrameMetrics() const {
    return frame_metrics_;
}

bool Benchmark::exportToCSV(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    // Write header
    file << "frame_number,processing_time_ms,faces_detected,active_tracks,"
         << "fps,cpu_percent,memory_mb,gpu_memory_mb,gpu_utilization,"
         << "temperature_c,power_w" << std::endl;

    // Write data
    for (const auto& metrics : frame_metrics_) {
        file << metrics.frame_number << ","
             << metrics.processing_time_ms << ","
             << metrics.faces_detected << ","
             << metrics.active_tracks << ","
             << metrics.fps << ","
             << metrics.resources.cpu_percent << ","
             << metrics.resources.memory_mb << ","
             << metrics.resources.gpu_memory_mb << ","
             << metrics.resources.gpu_utilization << ","
             << metrics.resources.temperature_c << ","
             << metrics.resources.power_w << std::endl;
    }

    file.close();
    std::cout << "Results exported to: " << filepath << std::endl;
    return true;
}

bool Benchmark::exportSummary(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        return false;
    }

    file << "=== Benchmark Summary ===" << std::endl;
    file << "Name: " << summary_.benchmark_name << std::endl;
    file << "Duration: " << summary_.total_duration_sec << " seconds" << std::endl;
    file << "Total Frames: " << summary_.total_frames << std::endl;
    file << "\nFPS Metrics:" << std::endl;
    file << "  Average: " << summary_.avg_fps << std::endl;
    file << "  Min: " << summary_.min_fps << std::endl;
    file << "  Max: " << summary_.max_fps << std::endl;
    file << "  P50: " << summary_.p50_fps << std::endl;
    file << "  P95: " << summary_.p95_fps << std::endl;
    file << "  P99: " << summary_.p99_fps << std::endl;
    file << "\nLatency Metrics:" << std::endl;
    file << "  Average: " << summary_.avg_latency_ms << " ms" << std::endl;
    file << "  Min: " << summary_.min_latency_ms << " ms" << std::endl;
    file << "  Max: " << summary_.max_latency_ms << " ms" << std::endl;
    file << "  P50: " << summary_.p50_latency_ms << " ms" << std::endl;
    file << "  P95: " << summary_.p95_latency_ms << " ms" << std::endl;
    file << "  P99: " << summary_.p99_latency_ms << " ms" << std::endl;
    file << "\nResource Metrics:" << std::endl;
    file << "  Avg CPU: " << summary_.avg_cpu_percent << "%" << std::endl;
    file << "  Avg Memory: " << summary_.avg_memory_mb << " MB" << std::endl;
    file << "  Max Temperature: " << summary_.max_temperature_c << "°C" << std::endl;
    file << "\nDetection Metrics:" << std::endl;
    file << "  Avg Faces/Frame: " << summary_.avg_faces_per_frame << std::endl;
    file << "  Max Faces: " << summary_.max_faces_in_frame << std::endl;
    file << "  Total Faces: " << summary_.total_faces_detected << std::endl;

    file.close();
    return true;
}

bool Benchmark::generateReport(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << "# Performance Benchmark Report" << std::endl;
    file << "\n## Benchmark: " << summary_.benchmark_name << std::endl;
    file << "\n### System Information" << std::endl;
    file << systemInfoToString(getSystemInfo()) << std::endl;

    file << "\n### Performance Metrics" << std::endl;
    file << "| Metric | Value | Target | Status |" << std::endl;
    file << "|--------|-------|--------|--------|" << std::endl;

    std::string fps_status = summary_.avg_fps >= 30.0 ? "✅ PASS" : "❌ FAIL";
    file << "| Average FPS | " << summary_.avg_fps << " | 30 | " << fps_status << " |" << std::endl;

    std::string latency_status = summary_.avg_latency_ms < 50.0 ? "✅ PASS" : "❌ FAIL";
    file << "| Avg Latency | " << summary_.avg_latency_ms << " ms | <50 ms | " << latency_status << " |" << std::endl;

    file << "\n### Detailed Results" << std::endl;
    file << "```\n";
    exportSummary(filepath + ".tmp");
    std::ifstream tmp(filepath + ".tmp");
    std::string line;
    while (std::getline(tmp, line)) {
        file << line << std::endl;
    }
    tmp.close();
    std::remove((filepath + ".tmp").c_str());
    file << "```\n";

    file.close();
    return true;
}

void Benchmark::printProgress(int current, int total) {
    double percent = static_cast<double>(current) / total * 100.0;
    std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << percent << "% ("
              << current << "/" << total << " frames)" << std::flush;
}

void Benchmark::printSummary() const {
    std::cout << "\n\n=== Benchmark Summary ===" << std::endl;
    std::cout << "Name: " << summary_.benchmark_name << std::endl;
    std::cout << "Duration: " << summary_.total_duration_sec << " seconds" << std::endl;
    std::cout << "Total Frames: " << summary_.total_frames << std::endl;

    std::cout << "\nFPS Metrics:" << std::endl;
    std::cout << "  Average: " << std::fixed << std::setprecision(2) << summary_.avg_fps << std::endl;
    std::cout << "  Min: " << summary_.min_fps << std::endl;
    std::cout << "  Max: " << summary_.max_fps << std::endl;
    std::cout << "  P50: " << summary_.p50_fps << std::endl;
    std::cout << "  P95: " << summary_.p95_fps << std::endl;
    std::cout << "  P99: " << summary_.p99_fps << std::endl;

    std::cout << "\nLatency Metrics:" << std::endl;
    std::cout << "  Average: " << summary_.avg_latency_ms << " ms" << std::endl;
    std::cout << "  Min: " << summary_.min_latency_ms << " ms" << std::endl;
    std::cout << "  Max: " << summary_.max_latency_ms << " ms" << std::endl;
    std::cout << "  P50: " << summary_.p50_latency_ms << " ms" << std::endl;
    std::cout << "  P95: " << summary_.p95_latency_ms << " ms" << std::endl;
    std::cout << "  P99: " << summary_.p99_latency_ms << " ms" << std::endl;

    std::cout << "\nResource Metrics:" << std::endl;
    std::cout << "  Avg CPU: " << summary_.avg_cpu_percent << "%" << std::endl;
    std::cout << "  Avg Memory: " << summary_.avg_memory_mb << " MB" << std::endl;
    std::cout << "  Max Temperature: " << summary_.max_temperature_c << "°C" << std::endl;

    std::cout << "\nDetection Metrics:" << std::endl;
    std::cout << "  Avg Faces/Frame: " << summary_.avg_faces_per_frame << std::endl;
    std::cout << "  Max Faces: " << summary_.max_faces_in_frame << std::endl;
    std::cout << "  Total Faces: " << summary_.total_faces_detected << std::endl;

    std::cout << "\n========================" << std::endl;
}

double Benchmark::calculatePercentile(const std::vector<double>& values, double percentile) const {
    if (values.empty()) return 0.0;

    size_t index = static_cast<size_t>(std::ceil(values.size() * percentile / 100.0)) - 1;
    index = std::min(index, values.size() - 1);
    return values[index];
}

ResourceSnapshot Benchmark::captureResourceSnapshot() {
    ResourceSnapshot snapshot;
    snapshot.timestamp = std::chrono::system_clock::now();

#ifdef __APPLE__
    // macOS: Get CPU usage
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                         (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long total_ticks = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total_ticks += cpuinfo.cpu_ticks[i];
        }
        unsigned long idle_ticks = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
        snapshot.cpu_percent = (100.0 * (1.0 - static_cast<double>(idle_ticks) / total_ticks));
    }

    // Get memory usage
    struct vm_statistics64 vm_stat;
    count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vm_stat, &count) == KERN_SUCCESS) {
        long long total_mem = vm_stat.wire_count + vm_stat.active_count +
                             vm_stat.inactive_count + vm_stat.free_count;
        long long used_mem = total_mem - vm_stat.free_count;
        snapshot.memory_mb = (used_mem * vm_page_size) / (1024.0 * 1024.0);
    }

#elif __linux__
    // Linux: Read /proc/meminfo for memory
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    long long total_mem = 0, free_mem = 0, available_mem = 0;

    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            sscanf(line.c_str(), "MemTotal: %lld kB", &total_mem);
        } else if (line.find("MemFree:") == 0) {
            sscanf(line.c_str(), "MemFree: %lld kB", &free_mem);
        } else if (line.find("MemAvailable:") == 0) {
            sscanf(line.c_str(), "MemAvailable: %lld kB", &available_mem);
        }
    }
    meminfo.close();

    long long used_mem = total_mem - available_mem;
    snapshot.memory_mb = used_mem / 1024.0;  // kB to MB

    // GPU monitoring would require nvidia-smi or similar
    snapshot.gpu_memory_mb = 0.0;
    snapshot.gpu_utilization = 0.0;
#endif

    return snapshot;
}

// ============================================================================
// BenchmarkSuite Implementation
// ============================================================================

BenchmarkSuite::BenchmarkSuite() {}

BenchmarkSuite::~BenchmarkSuite() {}

void BenchmarkSuite::addBenchmark(const Benchmark& benchmark) {
    benchmarks_.push_back(benchmark);
}

void BenchmarkSuite::addBenchmarkConfig(const BenchmarkConfig& config) {
    Benchmark benchmark;
    benchmark.configure(config);
    benchmarks_.push_back(benchmark);
}

bool BenchmarkSuite::runAll() {
    std::cout << "\n=== Running Benchmark Suite ===" << std::endl;
    std::cout << "Total benchmarks: " << benchmarks_.size() << std::endl;

    bool all_success = true;
    for (size_t i = 0; i < benchmarks_.size(); i++) {
        std::cout << "\n[" << (i + 1) << "/" << benchmarks_.size() << "] ";
        if (!benchmarks_[i].run()) {
            all_success = false;
        }
    }

    std::cout << "\n=== Benchmark Suite Complete ===" << std::endl;
    return all_success;
}

bool BenchmarkSuite::runBenchmark(size_t index) {
    if (index >= benchmarks_.size()) {
        std::cerr << "Error: Benchmark index out of range" << std::endl;
        return false;
    }
    return benchmarks_[index].run();
}

bool BenchmarkSuite::runBenchmark(const std::string& name) {
    for (auto& benchmark : benchmarks_) {
        if (benchmark.getConfig().name == name) {
            return benchmark.run();
        }
    }
    std::cerr << "Error: Benchmark '" << name << "' not found" << std::endl;
    return false;
}

std::vector<BenchmarkSummary> BenchmarkSuite::getAllSummaries() const {
    std::vector<BenchmarkSummary> summaries;
    for (const auto& benchmark : benchmarks_) {
        summaries.push_back(benchmark.getSummary());
    }
    return summaries;
}

bool BenchmarkSuite::exportAllToCSV(const std::string& directory) const {
    for (size_t i = 0; i < benchmarks_.size(); i++) {
        std::string filepath = directory + "/" + benchmarks_[i].getConfig().name + "_results.csv";
        if (!benchmarks_[i].exportToCSV(filepath)) {
            return false;
        }
    }
    return true;
}

bool BenchmarkSuite::generateComparisonReport(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return false;
    }

    file << "# Benchmark Comparison Report\n" << std::endl;
    file << "Generated: " << getCurrentTimestamp() << "\n" << std::endl;

    file << "## System Information\n" << std::endl;
    file << systemInfoToString(getSystemInfo()) << "\n" << std::endl;

    file << "## Results Summary\n" << std::endl;
    file << "| Benchmark | FPS | Latency (ms) | CPU (%) | Memory (MB) |" << std::endl;
    file << "|-----------|-----|--------------|---------|--------------|" << std::endl;

    for (const auto& benchmark : benchmarks_) {
        auto summary = benchmark.getSummary();
        file << "| " << summary.benchmark_name
             << " | " << std::fixed << std::setprecision(2) << summary.avg_fps
             << " | " << summary.avg_latency_ms
             << " | " << summary.avg_cpu_percent
             << " | " << summary.avg_memory_mb << " |" << std::endl;
    }

    file.close();
    return true;
}

// ============================================================================
// Utility Functions
// ============================================================================

SystemInfo getSystemInfo() {
    SystemInfo info;

#ifdef __APPLE__
    // macOS
    char buffer[256];
    size_t size = sizeof(buffer);

    // OS Name
    info.os_name = "macOS";

    // OS Version
    if (sysctlbyname("kern.osrelease", buffer, &size, NULL, 0) == 0) {
        info.os_version = std::string(buffer);
    }

    // CPU Model
    size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.brand_string", buffer, &size, NULL, 0) == 0) {
        info.cpu_model = std::string(buffer);
    }

    // CPU Cores
    int cores;
    size = sizeof(cores);
    if (sysctlbyname("hw.ncpu", &cores, &size, NULL, 0) == 0) {
        info.cpu_cores = cores;
    }

    // Memory
    int64_t mem;
    size = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &size, NULL, 0) == 0) {
        info.system_memory_gb = mem / (1024.0 * 1024.0 * 1024.0);
    }

#elif __linux__
    // Linux
    info.os_name = "Linux";

    // Read OS version
    std::ifstream os_release("/etc/os-release");
    std::string line;
    while (std::getline(os_release, line)) {
        if (line.find("PRETTY_NAME=") == 0) {
            info.os_version = line.substr(12);
            info.os_version.pop_back();  // Remove trailing quote
            break;
        }
    }
    os_release.close();

    // CPU info
    std::ifstream cpuinfo("/proc/cpuinfo");
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") == 0) {
            info.cpu_model = line.substr(line.find(":") + 2);
            break;
        }
    }
    cpuinfo.close();

    // CPU cores
    info.cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);

    // Memory
    std::ifstream meminfo("/proc/meminfo");
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            long long total_mem;
            sscanf(line.c_str(), "MemTotal: %lld kB", &total_mem);
            info.system_memory_gb = total_mem / (1024.0 * 1024.0);
            break;
        }
    }
    meminfo.close();
#endif

    // OpenCV version
    #ifdef OPENCV_AVAILABLE
    info.opencv_version = CV_VERSION;
    #else
    info.opencv_version = "Not available";
    #endif

    // CUDA availability (would need actual OpenCV CUDA check)
    #ifdef HAVE_CUDA
    info.cuda_available = true;
    #else
    info.cuda_available = false;
    #endif

    return info;
}

std::string systemInfoToString(const SystemInfo& info) {
    std::stringstream ss;
    ss << "OS: " << info.os_name << " " << info.os_version << "\n";
    ss << "CPU: " << info.cpu_model << " (" << info.cpu_cores << " cores)\n";
    ss << "Memory: " << std::fixed << std::setprecision(1) << info.system_memory_gb << " GB\n";
    ss << "OpenCV: " << info.opencv_version << "\n";
    ss << "CUDA: " << (info.cuda_available ? "Available" : "Not Available");
    return ss.str();
}

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string formatDuration(double seconds) {
    int hours = static_cast<int>(seconds / 3600);
    int minutes = static_cast<int>((seconds - hours * 3600) / 60);
    int secs = static_cast<int>(seconds - hours * 3600 - minutes * 60);
    int ms = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);

    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        ss << minutes << "m ";
    }
    ss << secs << "." << std::setfill('0') << std::setw(3) << ms << "s";
    return ss.str();
}

void printProgressBar(int current, int total, int width) {
    double percent = static_cast<double>(current) / total;
    int filled = static_cast<int>(width * percent);

    std::cout << "\r[";
    for (int i = 0; i < width; i++) {
        std::cout << (i < filled ? "=" : " ");
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (percent * 100.0) << "% ("
              << current << "/" << total << ")" << std::flush;
}

} // namespace benchmark
} // namespace kiosk
