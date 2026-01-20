#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace fastllm;

// ==========================================
// 1. 基础工具：FP16 类型与 ACL 辅助
// ==========================================


// FP16 转换工具 (依赖编译器支持 __fp16，通常在 ARM 环境下可用)
uint16_t F32toF16(float value) {
    __fp16 val_f16 = (__fp16)value;
    return *(uint16_t*)&val_f16;
}

float F16toF32(uint16_t value) {
    __fp16 val_f16 = *(__fp16*)&value;
    return (float)val_f16;
}

void FillRandomF16(Data &data, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    uint16_t *ptr = (uint16_t*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = F32toF16(dis(gen));
}


// ==========================================
// 3. 验证逻辑与测试主函数
// ==========================================

bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 1e-3, const char* name = "Op") {
    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData;
    int len = cpuOut.Count(0);
    int errIdx = -1;

    for (int i = 0; i < len; ++i) {
        float diff = std::abs(pCpu[i] - pNpu[i]);
        if (diff > maxErr) { maxErr = diff; errIdx = i; }
    }

    std::cout << "Testing " << name << " -> Max Error: " << std::fixed << std::setprecision(6) << maxErr;
    if (maxErr > threshold) {
        std::cout << " [FAIL] (Thresh: " << threshold << ") @" << errIdx 
                  << " CPU:" << pCpu[errIdx] << " NPU:" << pNpu[errIdx] << std::endl;
        return false;
    }
    std::cout << " [PASS]" << std::endl;
    return true;
}

void Test_SwiGLU() {
    std::cout << "=== Test 1: SwiGLU (FP16) ===" << std::endl;
    int M = 16;
    int K = 256; // Output Hidden Size
    // Input must be [M, 2 * K]
    Data input;
    input.dataType = DataType::FLOAT16; input.Resize({M, 2 * K});
    FillRandomF16(input, -2.0f, 2.0f);

    // CPU Ref
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({M, K}); cpuOut.Allocate();
    float *pOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            // SwiGLU: Input Split -> (A, B)
            // 通常: Out = Swish(A) * B
            // A = input[i, j], B = input[i, j + K]
            float a = F16toF32(pIn[i * (2*K) + j]);
            float b = F16toF32(pIn[i * (2*K) + j + K]);
            
            // Swish(x) = x / (1 + exp(-x))
            float swish_a = a / (1.0f + std::exp(-a));
            pOut[i * K + j] = swish_a * b;
        }
    }

    // NPU Run
    Data npuInput, npuOutput;
    auto ToDevice = [](Data &h, Data &d) {
        d.dataType = h.dataType; d.Resize(h.dims);
        d.deviceData = FastllmAclMalloc(h.GetBytes());
        FastllmAclCopyFromHostToDevice(d.deviceData, h.cpuData, h.GetBytes());
    };
    
    ToDevice(input, npuInput);
    npuOutput.dataType = DataType::FLOAT16; npuOutput.Resize({M, K});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    FastllmAclSwiglu(npuInput, npuOutput);

    // Copy Back
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize({M, K}); npuResHost.Allocate();
    Data tempF16;
    tempF16.dataType = DataType::FLOAT16; tempF16.Resize({M, K}); tempF16.Allocate();
    FastllmAclCopyFromDeviceToHost(tempF16.cpuData, npuOutput.deviceData, tempF16.GetBytes());
    
    uint16_t *ptrF16 = (uint16_t*)tempF16.cpuData;
    float *ptrF32 = (float*)npuResHost.cpuData;
    for(int i=0; i<npuResHost.Count(0); ++i) ptrF32[i] = F16toF32(ptrF16[i]);

    CompareData(cpuOut, npuResHost, 1e-3, "SwiGLU");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

void Test_Softmax() {
    std::cout << "\n=== Test 2: Softmax (FP16, Axis=-1) ===" << std::endl;
    int M = 16;
    int K = 128;
    Data input;
    input.dataType = DataType::FLOAT16; input.Resize({M, K});
    FillRandomF16(input, -5.0f, 5.0f); // Range slightly larger to test exp stability

    // CPU Ref
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({M, K}); cpuOut.Allocate();
    float *pOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;

    for (int i = 0; i < M; ++i) {
        // 1. Find Max
        float maxVal = -1e9;
        for (int j = 0; j < K; ++j) {
            float val = F16toF32(pIn[i * K + j]);
            if (val > maxVal) maxVal = val;
        }
        // 2. Sum Exp
        float sumExp = 0.0f;
        std::vector<float> exps(K);
        for (int j = 0; j < K; ++j) {
            float val = F16toF32(pIn[i * K + j]);
            exps[j] = std::exp(val - maxVal);
            sumExp += exps[j];
        }
        // 3. Div
        for (int j = 0; j < K; ++j) {
            pOut[i * K + j] = exps[j] / sumExp;
        }
    }

    // NPU Run
    Data npuInput, npuOutput;
    auto ToDevice = [](Data &h, Data &d) {
        d.dataType = h.dataType; d.Resize(h.dims);
        d.deviceData = FastllmAclMalloc(h.GetBytes());
        FastllmAclCopyFromHostToDevice(d.deviceData, h.cpuData, h.GetBytes());
    };

    ToDevice(input, npuInput);
    npuOutput.dataType = DataType::FLOAT16; npuOutput.Resize({M, K});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    FastllmAclSoftmax(npuInput, npuOutput, -1);

    // Copy Back
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize({M, K}); npuResHost.Allocate();
    Data tempF16;
    tempF16.dataType = DataType::FLOAT16; tempF16.Resize({M, K}); tempF16.Allocate();
    FastllmAclCopyFromDeviceToHost(tempF16.cpuData, npuOutput.deviceData, tempF16.GetBytes());
    
    uint16_t *ptrF16 = (uint16_t*)tempF16.cpuData;
    float *ptrF32 = (float*)npuResHost.cpuData;
    for(int i=0; i<npuResHost.Count(0); ++i) ptrF32[i] = F16toF32(ptrF16[i]);

    CompareData(cpuOut, npuResHost, 1e-3, "Softmax");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();

    Test_SwiGLU();
    Test_Softmax();

    std::cout << "All tests finished." << std::endl;
    return 0;
}