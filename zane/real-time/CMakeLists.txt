cmake_minimum_required(VERSION 3.18)
project(KioskFaceDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    camera_capture.cpp
    face_detection.cpp
)

set(HEADERS
    camera_capture.h
    face_detection.h
)

# Libraries
set(LIBRARIES
    ${OpenCV_LIBS}
    pthread
)

# Create test executables
add_executable(test_camera test_camera.cpp ${SOURCES})
target_link_libraries(test_camera ${LIBRARIES})

add_executable(test_face_detection test_face_detection.cpp ${SOURCES})
target_link_libraries(test_face_detection ${LIBRARIES})

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(test_camera PRIVATE -Wall -Wextra -O3)
    target_compile_options(test_face_detection PRIVATE -Wall -Wextra -O3)
elseif(MSVC)
    target_compile_options(test_camera PRIVATE /W4 /O2)
    target_compile_options(test_face_detection PRIVATE /W4 /O2)
endif()

# CUDA support (for Jetson Orin Nano)
if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER_EQUAL 4.5.0)
    if(OpenCV_USE_CUDA)
        message(STATUS "OpenCV CUDA support detected")
    endif()
endif()

# Installation
install(TARGETS test_camera test_face_detection
        RUNTIME DESTINATION bin)

# Print configuration summary
message(STATUS "")
message(STATUS "=== Kiosk Face Detection Configuration ===")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
if(OpenCV_USE_CUDA)
    message(STATUS "CUDA support: YES")
else()
    message(STATUS "CUDA support: NO")
endif()
message(STATUS "===========================================")
message(STATUS "")
