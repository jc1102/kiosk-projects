#!/bin/bash
# Camera Capture Performance Benchmark
# Run on Jetson Orin Nano

set -e

echo "=== Camera Capture Performance Benchmark ==="
echo "Timestamp: $(date)"
echo "============================================="

# Default device
DEVICE="/dev/video0"
DURATION=60
RESOLUTION="1920x1080"
FPS=60

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Duration: ${DURATION}s"
echo "  Resolution: $RESOLUTION"
echo "  Target FPS: $FPS"
echo ""

# Check if device exists
if [ ! -e "$DEVICE" ]; then
    echo "Error: Device $DEVICE not found!"
    echo "Available devices:"
    ls -la /dev/video*
    exit 1
fi

# Create output directory
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running benchmark..."

# Run camera test and capture output
timeout $DURATION ./camera_test "$DEVICE" 2>&1 | tee "$OUTPUT_DIR/benchmark.log"

# Extract metrics
echo ""
echo "=== Benchmark Results ===" > "$OUTPUT_DIR/summary.txt"
echo "Configuration:" >> "$OUTPUT_DIR/summary.txt"
echo "  Device: $DEVICE" >> "$OUTPUT_DIR/summary.txt"
echo "  Resolution: $RESOLUTION" >> "$OUTPUT_DIR/summary.txt"
echo "  Target FPS: $FPS" >> "$OUTPUT_DIR/summary.txt"
echo "  Duration: ${DURATION}s" >> "$OUTPUT_DIR/summary.txt"
echo "" >> "$OUTPUT_DIR/summary.txt"

# Extract from log (if available)
if grep -q "Frames captured:" "$OUTPUT_DIR/benchmark.log"; then
    grep -A 5 "Camera Statistics:" "$OUTPUT_DIR/benchmark.log" >> "$OUTPUT_DIR/summary.txt"
fi

# Get system stats
echo "" >> "$OUTPUT_DIR/summary.txt"
echo "=== System Status ===" >> "$OUTPUT_DIR/summary.txt"
echo "CPU:" >> "$OUTPUT_DIR/summary.txt"
cat /proc/cpuinfo | grep "model name" | head -1 >> "$OUTPUT_DIR/summary.txt"
echo "Memory:" >> "$OUTPUT_DIR/summary.txt"
free -h >> "$OUTPUT_DIR/summary.txt"
echo "GPU:" >> "$OUTPUT_DIR/summary.txt"
nvidia-smi >> "$OUTPUT_DIR/summary.txt" 2>&1 || echo "nvidia-smi not available" >> "$OUTPUT_DIR/summary.txt"

cat "$OUTPUT_DIR/summary.txt"

echo ""
echo "Benchmark complete! Results saved to: $OUTPUT_DIR"
echo "Summary:"
cat "$OUTPUT_DIR/summary.txt"
