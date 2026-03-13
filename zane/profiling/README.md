# Profiling Tools

## Purpose
Performance profiling and analysis tools for Kiosk CV systems optimization.

## Profiling Stack

### 1. NVIDIA Tools (Jetson Platform)
#### Nsight Systems
- System-wide performance analysis
- GPU/CPU activity visualization
- Memory access patterns
- Thread synchronization analysis

```bash
nsys profile --trace=cuda,nvtx,osrt --output=profile.nsys-rep ./kiosk_cv_app
```

#### Nsight Compute
- GPU kernel-level analysis
- Instruction throughput
- Memory bandwidth
- Register usage

```bash
ncu --set full --export report.ncu-rep ./kiosk_cv_app
```

### 2. OpenCV Profiling
#### Built-in Timing
```cpp
double t = (double)cv::getTickCount();
// ... code ...
t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
```

#### Performance Modules
- `cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO)`
- CPU/GPU memory tracking
- Module-level timing

### 3. Custom Profiling Instruments
#### Latency Tracker
- Frame-by-frame timing
- Pipeline stage analysis
- Latency percentile calculation (P50, P95, P99)

#### Resource Monitor
- GPU utilization (CUDA cores, Tensor cores)
- Memory usage (GPU, system)
- Power consumption
- Thermal throttling detection

#### Frame Analysis
- Frame drop detection
- Frame time variance
- Processing jitter
- Queue depth monitoring

## Profiling Workflow

### 1. Baseline Profiling
- Run application under normal load
- Capture full system profile
- Establish performance baseline
- Document bottlenecks

### 2. Targeted Profiling
- Focus on specific pipeline stages
- Profile hot functions/kernels
- Memory access analysis
- Identify optimization opportunities

### 3. Optimization Validation
- Profile after each optimization
- Compare to baseline
- Ensure no regressions
- Document improvements

### 4. Regression Testing
- Automated profiling in CI/CD
- Performance gate thresholds
- Alert on degradation
- Historical trend analysis

## Key Metrics to Profile

### Latency Metrics
- End-to-end processing time
- Per-stage latency
- Frame delivery latency
- Event generation latency

### Throughput Metrics
- Frames per second
- Concurrent tracking capacity
- Model inference throughput
- Data transfer rates

### Resource Metrics
- GPU utilization (%)
- GPU memory usage
- CPU utilization (%)
- System memory usage
- Power consumption (Watts)
- Thermal status (°C)

### Quality Metrics
- Detection accuracy
- False positive rate
- ID retention rate
- Confidence distribution

## Profiling Scripts

### Automated Profiling Run
```bash
#!/bin/bash
# profile_benchmark.sh

DURATION=300  # 5 minutes
OUTPUT_DIR="profiling/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

# Start system monitoring
nvidia-smi dmon -s pucvmet -o DT -d 1 > "$OUTPUT_DIR/gpu_stats.csv" &
GPU_PID=$!

# Start application profiling
nsys profile --trace=cuda,nvtx,osrt \
  --stats=true \
  --duration=$DURATION \
  --output="$OUTPUT_DIR/profile" \
  ./kiosk_cv_app

# Stop monitoring
kill $GPU_PID

# Generate report
echo "Profiling complete: $OUTPUT_DIR"
```

### Frame Time Analyzer
```python
# analyze_frames.py
import numpy as np
import matplotlib.pyplot as plt

frame_times = np.loadtxt('frame_times.csv')
latency_p50 = np.percentile(frame_times, 50)
latency_p95 = np.percentile(frame_times, 95)
latency_p99 = np.percentile(frame_times, 99)

print(f"P50: {latency_p50:.2f}ms")
print(f"P95: {latency_p95:.2f}ms")
print(f"P99: {latency_p99:.2f}ms")
```

## Output Format

All profiling data stored with consistent structure:
```
profiling/
└── 20260313_182300/
    ├── profile.nsys-rep          # NVIDIA Nsight system profile
    ├── gpu_stats.csv             # GPU utilization over time
    ├── frame_times.csv           # Per-frame processing times
    ├── latency_analysis.json     # Statistical analysis
    ├── summary.txt               # Human-readable summary
    └── graphs/                   # Performance visualizations
        ├── latency_histogram.png
        ├── gpu_utilization.png
        └── memory_usage.png
```

## Performance Alerts
- Latency P95 > 100ms: CRITICAL
- GPU utilization > 90%: WARNING
- Frame drop rate > 1%: WARNING
- Memory growth over time: CRITICAL (potential leak)
