cmake_minimum_required(VERSION 3.18)
project(KioskPerformanceBenchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV (optional for benchmark framework itself)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_definitions(-DOPENCV_AVAILABLE)
else()
    message(STATUS "OpenCV not found - building without camera support")
    add_definitions(-DOPENCV_NOT_AVAILABLE)
endif()

# Source files
set(SOURCES
    benchmark.cpp
)

set(HEADERS
    benchmark.h
)

# Libraries
if(OpenCV_FOUND)
    set(LIBRARIES
        ${OpenCV_LIBS}
        pthread
    )
else()
    set(LIBRARIES
        pthread
    )
endif()

# Create benchmark executable
add_executable(test_benchmark test_benchmark.cpp ${SOURCES})
target_link_libraries(test_benchmark ${LIBRARIES})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_benchmark PRIVATE -Wall -Wextra -O3)
elseif(MSVC)
    target_compile_options(test_benchmark PRIVATE /W4 /O2)
endif()

# Installation
install(TARGETS test_benchmark
        RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Kiosk Performance Benchmark Configuration ===")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "=================================================")
message(STATUS "")
