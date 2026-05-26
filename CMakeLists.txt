# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tf)

# 1. Base application source files
target_sources(app PRIVATE src/main.cpp)

# ==============================================================================
# 2. Automatically collect TFLite Micro and Signal source tree
# ==============================================================================

# Define the absolute path where TFLM is mounted inside your project
set(TFLITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tflite)

# Recursively grab all .cc and .c source files
file(GLOB_RECURSE TFLITE_SRCS
    "${TFLITE_DIR}/tensorflow/*.cc"
    "${TFLITE_DIR}/tensorflow/*.c"
    "${TFLITE_DIR}/signal/*.cc"
    "${TFLITE_DIR}/signal/*.c"
    "${TFLITE_DIR}/third_party/kissfft/*.c"
)

# ⚠️ Exclude unit tests, benchmarks and mock files that may cause Zephyr build errors
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_test\\.cc$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_test_helpers\\.h$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_benchmark\\.cc$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/testing/.*")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/fake_micro_context\\.cc")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/mock_micro_graph\\.cc")

# Inject the cleaned TFLM + Signal sources into the app target
target_sources(app PRIVATE ${TFLITE_SRCS})

# ==============================================================================
# 3. Include directories configuration (precise matching of third-party deps)
# ==============================================================================
target_include_directories(app PRIVATE
    ${TFLITE_DIR}                                     # Corresponds to #include "tensorflow/..." and "signal/..."
    ${TFLITE_DIR}/third_party/flatbuffers/include     # flatbuffers headers
    ${TFLITE_DIR}/third_party/gemmlowp                # fixed-point math library
    ${TFLITE_DIR}/third_party/kissfft                 # FFT library used for signal processing
    ${TFLITE_DIR}/third_party/ruy                     # ruy shim for matrix multiplication acceleration
)

# ==============================================================================
# 4. Platform-wide compile option adjustments
# ==============================================================================
target_compile_options(app PRIVATE
    -fno-exceptions
    -ffunction-sections
    -fdata-sections
    -DTF_LITE_STATIC_MEMORY
    -DTF_LITE_DISABLE_X86_NEON
)

target_compile_options(app INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)
