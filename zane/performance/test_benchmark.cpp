/**
 * @file test_benchmark.cpp
 * @brief Test program for performance benchmarking framework
 * @author Zane - C++/CV Engineer
 */

#include "benchmark.h"
#include <iostream>
#include <string>

#ifdef OPENCV_AVAILABLE
#define DEFAULT_RESOLUTION cv::Size(1280, 720)
#else
#define DEFAULT_RESOLUTION cv::Size(1280, 720)
#endif

using namespace kiosk::benchmark;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h              Show this help message" << std::endl;
    std::cout << "  --quick                 Run quick benchmark (30 seconds)" << std::endl;
    std::cout << "  --standard              Run standard benchmark (5 minutes)" << std::endl;
    std::cout << "  --full                  Run full benchmark (30 minutes)" << std::endl;
    std::cout << "  --camera <device>       Use camera device (e.g., /dev/video0)" << std::endl;
    std::cout << "  --duration <seconds>    Run for specified duration" << std::endl;
    std::cout << "  --frames <count>        Process specified number of frames" << std::endl;
    std::cout << "  --results <dir>         Results output directory (default: ./results)" << std::endl;
    std::cout << "  --no-progress           Disable progress display" << std::endl;
    std::cout << "  --no-summary            Disable summary display" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " --quick" << std::endl;
    std::cout << "  " << program_name << " --camera /dev/video0 --duration 60" << std::endl;
    std::cout << "  " << program_name << " --frames 1000 --results ./benchmark_results" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Kiosk CV Performance Benchmark Tool ===" << std::endl;

    // Default configuration
    BenchmarkConfig config;
    config.name = "baseline_benchmark";
    config.duration_seconds = 300;  // 5 minutes default
    config.num_frames = 0;
    config.camera_device = "";
    config.resolution = cv::Size(1280, 720);
    config.enable_gpu = true;
    config.save_results_csv = true;
    config.results_dir = "./results";
    config.enable_profiling = false;
    config.profiling_interval = 100;
    config.warmup_frames = 30;
    config.enable_cpu_monitoring = true;
    config.enable_memory_monitoring = true;
    config.enable_gpu_monitoring = false;  // GPU monitoring requires nvidia-smi
    config.print_progress = true;
    config.print_summary = true;
    config.generate_plots = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--quick") {
            config.name = "quick_benchmark";
            config.duration_seconds = 30;
            config.num_frames = 0;
        }
        else if (arg == "--standard") {
            config.name = "standard_benchmark";
            config.duration_seconds = 300;  // 5 minutes
            config.num_frames = 0;
        }
        else if (arg == "--full") {
            config.name = "full_benchmark";
            config.duration_seconds = 1800;  // 30 minutes
            config.num_frames = 0;
        }
        else if (arg == "--camera") {
            if (i + 1 < argc) {
                config.camera_device = argv[++i];
                config.name = "camera_benchmark";
            } else {
                std::cerr << "Error: --camera requires a device path" << std::endl;
                return 1;
            }
        }
        else if (arg == "--duration") {
            if (i + 1 < argc) {
                config.duration_seconds = std::atoi(argv[++i]);
                config.num_frames = 0;
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
        else if (arg == "--no-progress") {
            config.print_progress = false;
        }
        else if (arg == "--no-summary") {
            config.print_summary = false;
        }
        else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Create and configure benchmark
    Benchmark benchmark;
    if (!benchmark.configure(config)) {
        std::cerr << "Error: Failed to configure benchmark" << std::endl;
        return 1;
    }

    // Run benchmark
    std::cout << "\nStarting benchmark..." << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Name: " << config.name << std::endl;
    std::cout << "  Duration: " << config.duration_seconds << " seconds" << std::endl;
    std::cout << "  Frames: " << (config.num_frames > 0 ? std::to_string(config.num_frames) : "duration-based") << std::endl;
    std::cout << "  Camera: " << (config.camera_device.empty() ? "simulation mode" : config.camera_device) << std::endl;
    std::cout << "  Resolution: " << config.resolution.width << "x" << config.resolution.height << std::endl;
    std::cout << "  GPU: " << (config.enable_gpu ? "enabled" : "disabled") << std::endl;
    std::cout << "  Results: " << config.results_dir << std::endl;

    bool success = benchmark.run();

    if (success) {
        // Export detailed results
        BenchmarkSummary summary = benchmark.getSummary();
        std::string summary_path = config.results_dir + "/" + config.name + "_summary.txt";
        if (!benchmark.exportSummary(summary_path)) {
            std::cerr << "Warning: Could not export summary" << std::endl;
        }

        std::string report_path = config.results_dir + "/" + config.name + "_report.md";
        if (!benchmark.generateReport(report_path)) {
            std::cerr << "Warning: Could not generate report" << std::endl;
        }

        std::cout << "\n✅ Benchmark completed successfully!" << std::endl;
        std::cout << "Results saved to: " << config.results_dir << std::endl;
        return 0;
    } else {
        std::cerr << "\n❌ Benchmark failed!" << std::endl;
        return 1;
    }
}
