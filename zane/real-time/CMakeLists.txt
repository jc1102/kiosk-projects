cmake_minimum_required(VERSION 3.18)
project(KioskCameraCapture VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Jetson Orin Nano specific settings
set(CMAKE_CUDA_ARCHITECTURES 87)  # Orin architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Source files
set(CAMERA_SOURCES
    camera_capture.cpp
)

set(CAMERA_HEADERS
    camera_capture.h
)

# Camera capture library
add_library(camera_capture SHARED
    ${CAMERA_SOURCES}
    ${CAMERA_HEADERS}
)

target_link_libraries(camera_capture
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Enable CUDA linking
set_target_properties(camera_capture PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(camera_capture PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -O3
    )
endif()

# Test executable
add_executable(camera_test
    test_camera.cpp
)

target_link_libraries(camera_test
    camera_capture
)

# Installation rules
install(TARGETS camera_capture
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES ${CAMERA_HEADERS}
    DESTINATION include/kiosk/cv
)

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "Kiosk Camera Capture Configuration:")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "==========================================")
