# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(tf)

# 1. 基础应用层源码
target_sources(app PRIVATE src/main.cpp)

# ==============================================================================
# 2. TFLite Micro 与 Signal 源码树自动收集
# ==============================================================================

# 定义 TFLM 挂载在你工程下的绝对路径
set(TFLITE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tflite)

# 递归抓取所有的 .cc 和 .c 源文件
file(GLOB_RECURSE TFLITE_SRCS
    "${TFLITE_DIR}/tensorflow/*.cc"
    "${TFLITE_DIR}/tensorflow/*.c"
    "${TFLITE_DIR}/signal/*.cc"
    "${TFLITE_DIR}/signal/*.c"
    "${TFLITE_DIR}/third_party/kissfft/*.c"
)

# 🚨 【硬核避坑】排除掉可能导致 Zephyr 编译报错的单元测试、基准测试及 mock 文件
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_test\\.cc$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_test_helpers\\.h$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX ".*_benchmark\\.cc$")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/testing/.*")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/fake_micro_context\\.cc")
list(FILTER TFLITE_SRCS EXCLUDE REGEX "${TFLITE_DIR}/tensorflow/lite/micro/mock_micro_graph\\.cc")

# 将洗干净的 TFLM + Signal 源码注入 app
target_sources(app PRIVATE ${TFLITE_SRCS})

# ==============================================================================
# 3. 头文件包含路径配置 (精准匹配第三方依赖)
# ==============================================================================
target_include_directories(app PRIVATE
    ${TFLITE_DIR}                                     # 对应 #include "tensorflow/..." 和 "signal/..."
    ${TFLITE_DIR}/third_party/flatbuffers/include     # flatbuffers 头文件
    ${TFLITE_DIR}/third_party/gemmlowp                # 定点数数学库
    ${TFLITE_DIR}/third_party/kissfft                 # 信号处理相关的 FFT 库
    ${TFLITE_DIR}/third_party/ruy                     # 矩阵乘法加速基础垫片
)

# ==============================================================================
# 4. 平台全局编译选项微调
# ==============================================================================
target_compile_options(app PRIVATE
    -fno-exceptions
    -ffunction-sections
    -fdata-sections
    -DTF_LITE_STATIC_MEMORY
    -DTF_LITE_DISABLE_X86_NEON
)

# 如果你后续把 main.c 改成了 main.cpp，TFLM 强制要求关闭标准 C++ 异常机制以省 Flash
target_compile_options(app INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>)