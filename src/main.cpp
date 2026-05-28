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
#include "test_input_digital.h"

uint8_t *p_sample_data = NULL;

// 💡 Optimization: allocate 48KB to avoid internal buffer overlap in extreme edge cases
constexpr int kTensorArenaSize = 48 * 1024;
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
    int printed_count = 0;
    for (int i = 0; i < size && printed_count < 10; ++i) {
        if (raw_image[i] == 0) continue; 
        
        float raw_val = static_cast<float>(raw_image[i]);
        float normalized = raw_val / 255.0f;
        float quantized_float = (normalized / scale) + zero_point;
        int8_t calc_quant = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, std::round(quantized_float))));
        
        std::cout << "  Idx[" << i << "] Raw: " << std::setw(3) << (int)raw_image[i] 
                  << " -> Norm: " << std::fixed << std::setprecision(3) << normalized 
                  << " -> Calc int8: " << std::setw(4) << (int)calc_quant 
                  << " | Actual Buffer: " << std::setw(4) << (int)quantized_image[i] << std::endl;
        printed_count++;
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

    const tflite::Model* model = tflite::GetModel(_mnist_cnn_quant_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        std::cerr << "Model schema version mismatch!" << std::endl;
        return 1;
    }
    std::cout << "✓ Model loaded successfully" << std::endl;

    // Keep exactly these 5 operators to ensure the model ops are resolved correctly
    static tflite::MicroMutableOpResolver<8> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddAdd();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddMul(); 

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
    
    // Run inference for ten sample images.
    uint8_t* samples[] = { 
        (uint8_t*)sample0_data, (uint8_t*)sample1_data, (uint8_t*)sample2_data, 
        (uint8_t*)sample3_data, (uint8_t*)sample4_data, (uint8_t*)sample5_data,
        (uint8_t*)sample6_data, (uint8_t*)sample7_data, (uint8_t*)sample8_data,
        (uint8_t*)sample9_data };
    const char* sample_names[] = { 
        "sample0", "sample1", "sample2",
        "sample3", "sample4", "sample5",
        "sample6", "sample7", "sample8",
        "sample9" };
    const int num_samples = 1;
    const float inv_scale_255 = 1.0f / (255.0f * input_scale);
    const float zero_point_f = static_cast<float>(input_zero_point);

    int8_t* input_buffer = input->data.int8;
    int input_size = input->bytes;

    for (int s = 0; ;(s < num_samples)? s++ : s)
    {
        if(s < num_samples)
        {
            std::cout << "\n----- Inference for " << sample_names[s] << " -----" << std::endl;
            p_sample_data = samples[s];
        }
        else
        {
            std::cout << "\n----- Inference for bsp uart receive -----" << std::endl;

            extern struct k_sem bsp_rx_sem;
            k_sem_take(&bsp_rx_sem, K_FOREVER);
        
            extern uint8_t digital_buffer[784];
            p_sample_data = digital_buffer;
        }

        // 1. Print raw image statistics and ASCII
        print_image_stats(p_sample_data, 784, s < num_samples ? sample_names[s] : "bsp uart receive");
        print_image_ascii(p_sample_data, 28, 28);

        // 2. Preprocessing and quantization (write into interpreter input buffer)
        for (int i = 0; i < input_size; ++i) {
            float raw_val = static_cast<float>(p_sample_data[i]);
            float quantized_float = (raw_val * inv_scale_255) + zero_point_f;
        
            quantized_float = std::max(-128.0f, std::min(127.0f, quantized_float));
            input_buffer[i] = static_cast<int8_t>(std::round(quantized_float));
        }

    #if 0
        print_quantization_debug(p_sample_data, input_buffer, input_scale, input_zero_point, input_size);
    #endif

        // 3. Invoke the model (timed using Zephyr k_uptime_get)
        uint32_t start_ms = k_uptime_get();
        TfLiteStatus invoke_status = interpreter.Invoke();
        uint32_t infer_ms = k_uptime_get() - start_ms;
        std::cout << "Inference time: " << infer_ms << " ms" << std::endl;
        if (invoke_status != kTfLiteOk) {
            std::cerr << "ERROR: Model inference failed for " << sample_names[s] << std::endl;
            continue;
        }

        // 4. Output details and prediction
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
        if(s< num_samples)
        {
            std::cout << "Result for " << sample_names[s] << ": " << (max_probability * 100)
                    << "% likely the digit [" << max_digit << "]" << std::endl;
        }
        else
        {
            std::cout << "Result for bsp uart receive: " << (max_probability * 100)
                    << "% likely the digit [" << max_digit << "]" << std::endl;
        }
    }

    return 0;
}

int main(void)
{
	printk("board : %s\r\n", CONFIG_BOARD);
    printk("frequency : %d MHz (Cortex-M4)\r\n", sys_clock_hw_cycles_per_sec()/1000000);

	cnn_mnist();

	return 0;
}
