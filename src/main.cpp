/*
 * Copyright (c) 2022 Nordic Semiconductor
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <zephyr/kernel.h>
#include <zephyr/linker/linker-defs.h>

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mnist_cnn_quant_model_data.h"

// 💡 Fix 1: Convert arrays to standard uint8_t (0-255) to match your preprocessing/normalization.
uint8_t sample3_data[784] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0, 12, 41,146,146, 48,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0, 12,129,253,253,253,250,163, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,133,253,253,253,253,253,253,229, 70,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,101,253,252,145,102,107,237,253,247,128, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,181,253,167,  0,  0,  0, 61,235,253,253,163,  5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,255,253, 43,  0,  0,  0,  0, 58,193,253,253,164,  4,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,187,253, 32,  0,  0,  0,  0,  0, 55,236,253,253, 86,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,146,253, 32,  0,100,190, 87, 87, 87,147,253,253,123,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0, 94,253, 78, 40,248,253,253,253,253,253,253,253,223, 84, 15,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0, 14, 92, 12, 35,240,253,253,253,253,253,253,253,253,253,244, 89, 10,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0, 75,161,179,253,253,253,253,253,253,253,253,253,209, 43,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3, 16, 16, 39, 38, 16, 16,145,243,253,253,185, 48,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0, 20, 58,  0,  0,  0,  0,  0,  0,  0,  0, 58,209,253,253,183,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 77,221,247, 79,  0,  0,  0,  0,  0,  0,  0,  0, 13,219,253,240, 72,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 90,247,253,252, 57,  0,  0,  0,  0,  0,  0,  0,  0, 53,251,253,191,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,116,253,253, 59,  0,  0,  0,  0,  0,  0,  0,  0, 99,252,253,145,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,188,253,221,158, 38,  0,  0,  0,  0,111,211,246,253,253,145,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 12,221,246,253,251,249,249,249,249,253,253,253,253,200, 19,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 95,183,228,253,253,253,253,253,253,195,124, 23,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 24, 37,138, 74,126, 88, 37,  7,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

uint8_t sample6_data[784] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,204,253,176,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,150,252,252,125,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,117,252,186, 56,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,141,252,118,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,154,247, 50,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 26,253,196,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,150,253,196,  0,  0,  0,  0,  0,  0,  0, 57, 85, 85, 38,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,225,253, 96,  0,  0,  0,  0,  0,151,226,243,252,252,238,125,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, 10,229,226,  0,  0,  0,  4, 54,229,253,255,234,175,225,255,228, 31,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,110,252,150,  0,  0, 26,128,252,252,227,134, 28,  0,  0,178,252, 56,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,159,252,113,  0,  0,150,253,252,186, 43,  0,  0,  0,  0,141,252, 56,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,185,252,113,  0, 38,237,253,151,  6,  0,  0,  0,  0,  0,141,202,  6,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,198,253,114,  0,147,253,163,  0,  0,  0,  0,  0,  0,  0,154,197,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,197,252,113,  0,172,252,188,  0,  0,  0,  0,  0,  0, 26,253,171,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,197,252,113,  0, 19,231,247,122, 19,  0,  0,  0,  0,200,244, 56,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0, 26,222,252,113,  0,  0, 25,203,252,193, 13,  0, 76,200,249,125,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,185,253,179, 10,  0,  0,  0, 76, 35, 29,154,253,244,125,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0, 28,209,253,196, 82, 57, 57,131,197,252,253,214, 81,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 25,216,252,252,252,253,252,252,252,156, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0, 16,103,139,240,140,139,139, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

uint8_t sample8_data[784] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 43, 47, 47,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,108,249,253,253,208,207,207,207,149, 65, 13,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,184,254,253,253,253,254,253,253,253,254,253,213, 25,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0, 55,203,254,254,199,127,127, 60, 93, 84, 68,151,222,254,161,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,138,253,253,199, 19,  0,  0,  0,  0,  0,  0,  0,155,253,211,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,138,253,253, 17,  0,  0,  0,  0,  0,  0,  0, 74,241,253,211,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,105,253,253,102,  0,  0,  0,  0,  0,  0, 34,229,253,253,160,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,149,254,229, 40,  0,  0,  0, 38,153,254,254,254,180, 25,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 19,198,254,207,  9, 34, 72,235,253,253,224,139, 13,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 17,211,253,215,240,254,253,234,128, 17,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,102,229,253,253,253,228, 77, 13,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0, 70,170,254,254,254,254,254,254,119,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0, 26,130,230,254,253,253,185,115, 64,211,253,248, 21,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,166,232,253,253,247,162, 46, 13,  7, 91,245,253,254, 56,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,128,253,253,253,210, 93,127,159,204,253,253,253,228, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,134,241,254,255,254,254,254,254,254,254,228, 34,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0, 27,115,140,206,206,206,207,206,123, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

uint8_t *p_sample_data = sample3_data;

// 💡 Optimization: allocate 64KB to avoid internal buffer overlap in extreme edge cases
constexpr int kTensorArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// --- Keep your useful debug functions (no changes) ---
void print_image_stats(uint8_t* image, int size, const char* name) {
    std::cout << "\n=== " << name << " Statistics ===" << std::endl;
    int min_val = 255, max_val = 0;
    long sum = 0;
    int non_zero = 0;
    for (int i = 0; i < size; ++i) {
        uint8_t val = image[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        if (val > 50) non_zero++;
    }
    float mean = static_cast<float>(sum) / size;
    float density = static_cast<float>(non_zero) / size * 100.0f;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Mean value: " << mean << std::endl;
    std::cout << "  Non-dark pixels (>50): " << non_zero << " / " << size << " (" << density << "%)" << std::endl;
}

void print_image_ascii(uint8_t* image, int width, int height) {
    std::cout << "\n=== Image ASCII Representation ===" << std::endl;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t pixel = image[y * width + x];
            if (pixel > 200) std::cout << "██";
            else if (pixel > 150) std::cout << "▓▓";
            else if (pixel > 100) std::cout << "▒▒";
            else if (pixel > 50) std::cout << "░░";
            else std::cout << "  ";
        }
        std::cout << std::endl;
    }
}

void print_quantization_debug(uint8_t* raw_image, int8_t* quantized_image, float scale, int zero_point, int size) {
    std::cout << "\n=== Quantization Debug (first 20 pixels) ===" << std::endl;
    std::cout << "Quantization params: scale=" << scale << ", zero_point=" << zero_point << std::endl;
    std::cout << "Raw(0-255) -> Normalized(0-1) -> Quantized(int8)" << std::endl;
    for (int i = 0; i < std::min(20, size); ++i) {
        float raw_val = raw_image[i];
        if (raw_val > 200.0f) {
            raw_val = raw_val * 0.85f;
        }
        float normalized = raw_val / 255.0f;
        float quantized_float = normalized / scale + zero_point;
        int8_t quantized = static_cast<int8_t>(std::round(quantized_float));
        std::cout << "[" << i << "] " << std::setw(3) << (int)raw_image[i] << " -> "
                  << std::fixed << std::setprecision(3) << normalized << " -> "
                  << std::setw(4) << (int)quantized << " (expected: " << (int)quantized_image[i] << ")" << std::endl;
    }
}

void print_output_details(int8_t* output_buffer, float scale, int zero_point) {
    std::cout << "\n=== Model Output Details ===" << std::endl;
    std::cout << "Output quant params: scale=" << scale << ", zero_point=" << zero_point << std::endl;
    float sum = 0.0f;
    for (int i = 0; i < 10; ++i) {
        float probability = (static_cast<float>(output_buffer[i]) - zero_point) * scale;
        probability = std::max(0.0f, std::min(1.0f, probability));
        sum += probability;
        std::cout << "Digit " << i << ": raw=" << std::setw(4) << (int)output_buffer[i] 
                  << " -> prob=" << std::fixed << std::setprecision(4) << probability;
        if (probability > 0.1f) std::cout << " ***";
        std::cout << std::endl;
    }
    std::cout << "Probability sum: " << sum << " (should be ~1.0)" << std::endl;
}

int cnn_mnist() {
    tflite::InitializeTarget();

    std::cout << "\n========================================" << std::endl;
    std::cout << "=== MNIST CNN Model Debug Version ===" << std::endl;
    std::cout << "========================================\n" << std::endl;

    const tflite::Model* model = tflite::GetModel(_tmp_lstm_trained_model_mnist_cnn_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model schema version mismatch!" << std::endl;
        return 1;
    }
    std::cout << "✓ Model loaded successfully" << std::endl;

    // Keep exactly these 10 operators to ensure the model ops are resolved correctly
    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddShape();
    micro_op_resolver.AddStridedSlice();
    micro_op_resolver.AddPack();
    micro_op_resolver.AddCast();
    micro_op_resolver.AddQuantize();

    static tflite::MicroInterpreter interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensor memory!" << std::endl;
        return 1;
    }

    TfLiteTensor* input = interpreter.input(0);
    TfLiteTensor* output = interpreter.output(0);
    
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    
    // 1. Print raw image statistics
    print_image_stats(p_sample_data, 784, "Raw Input Image");
    print_image_ascii(p_sample_data, 28, 28);
    
    // 2. Preprocessing and quantization
    int8_t* input_buffer = input->data.int8;
    int input_size = input->bytes;

    // 💡 Key tweak: apply edge softening for high-contrast images in this loop
    for (int i = 0; i < input_size; ++i) {
        float raw_val = p_sample_data[i];

        // For very bright hard edges (e.g., 253, 255) apply slight scaling to soften into gray transitions
        if (raw_val > 200.0f) {
            raw_val = raw_val * 0.85f;
        }

        float normalized = raw_val / 255.0f;
        float quantized_float = normalized / input_scale + input_zero_point;
        quantized_float = std::max(-128.0f, std::min(127.0f, quantized_float));
        input_buffer[i] = static_cast<int8_t>(std::round(quantized_float));
    }
#if 0
    // 3. Print quantization debug info
    print_quantization_debug(p_sample_data, input_buffer, input_scale, input_zero_point, input_size);
#endif
    // ===== Model inference =====
    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
        std::cerr << "ERROR: Model inference failed!" << std::endl;
        return 1;
    }
    
    // ===== Output details =====
    int8_t* output_buffer = output->data.int8;
    print_output_details(output_buffer, output_scale, output_zero_point);
    
    int max_digit = 0;
    float max_probability = -1.0f;
    for (int i = 0; i < 10; ++i) {
        float prob = (static_cast<float>(output_buffer[i]) - output_zero_point) * output_scale;
        prob = std::max(0.0f, std::min(1.0f, prob));
        if (prob > max_probability) {
            max_probability = prob;
            max_digit = i;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "🎯 FINAL PREDICTION: " << (max_probability * 100) 
              << "% likely the digit [" << max_digit << "]" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

int main(void)
{
	printk("Address of sample %p\n", (void *)__rom_region_start);
	printk("Hello sysbuild with mcuboot! %s\n", CONFIG_BOARD);

	cnn_mnist();

	return 0;
}
