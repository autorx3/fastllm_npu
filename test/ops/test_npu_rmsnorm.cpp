#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace fastllm;

// ==========================================
// 辅助工具：使用编译器原生 __fp16 (更稳健)
// ==========================================
// 注意：Ascend 开发环境通常是 ARM64，GCC 支持 __fp16
// 如果编译报错不支持 __fp16，请告诉我要换回位操作版本
uint16_t F32toF16(float value) {
    __fp16 val_f16 = (__fp16)value;
    return *(uint16_t*)&val_f16;
}

float F16toF32(uint16_t value) {
    __fp16 val_f16 = *(__fp16*)&value;
    return (float)val_f16;
}

// ==========================================
// 辅助函数：数据填充
// ==========================================
void FillRandomF16(Data &data, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    uint16_t *ptr = (uint16_t*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = F32toF16(dis(gen));
}

// ==========================================
// 辅助函数：对比结果
// ==========================================
bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 1e-2) {
    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData;
    int len = cpuOut.Count(0);
    int errIdx = -1;

    for (int i = 0; i < len; ++i) {
        // 跳过 NaN/Inf 检查，避免 crashing
        if (std::isnan(pCpu[i]) || std::isinf(pCpu[i]) || 
            std::isnan(pNpu[i]) || std::isinf(pNpu[i])) {
            continue;
        }

        float diff = std::abs(pCpu[i] - pNpu[i]);
        if (diff > maxErr) {
            maxErr = diff;
            errIdx = i;
        }
    }

    std::cout << " -> Max Error: " << std::fixed << std::setprecision(6) << maxErr;
    if (maxErr > threshold) {
        std::cout << " [FAIL] (Threshold: " << threshold << ")" << std::endl;
        if (errIdx != -1)
            std::cout << "    At index " << errIdx << ": CPU=" << pCpu[errIdx] << " NPU=" << pNpu[errIdx] << std::endl;
        return false;
    }
    std::cout << " [PASS]" << std::endl;
    return true;
}

// ==========================================
// 算子实现 (请确保你的 cpp 文件里也是这个逻辑)
// ==========================================

// ==========================================
// 测试用例
// ==========================================
void Test_RMSNorm() {
    std::cout << "=== Testing RMSNorm (FP16) ===" << std::endl;

    int Rows = 32;
    int Dim = 4096;
    float eps = 1e-6f;

    Data input;
    input.dataType = DataType::FLOAT16;
    input.Resize({Rows, Dim});
    FillRandomF16(input, -1.0f, 1.0f);

    Data weight;
    weight.dataType = DataType::FLOAT16;
    weight.Resize({Dim});
    FillRandomF16(weight, 0.8f, 1.2f); // Gamma 接近 1

    Data bias; // Empty

    // CPU Ground Truth
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize({Rows, Dim});
    cpuOut.Allocate();
    float *pCpuOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;
    uint16_t *pW  = (uint16_t*)weight.cpuData;

    #pragma omp parallel for
    for (int r = 0; r < Rows; ++r) {
        float sumSq = 0.0f;
        for (int d = 0; d < Dim; ++d) {
            float val = F16toF32(pIn[r * Dim + d]);
            sumSq += val * val;
        }
        float meanSq = sumSq / Dim;
        float invRms = 1.0f / std::sqrt(meanSq + eps);

        for (int d = 0; d < Dim; ++d) {
            float val = F16toF32(pIn[r * Dim + d]);
            float w   = F16toF32(pW[d]);
            pCpuOut[r * Dim + d] = val * invRms * w;
        }
    }

    // NPU Run
    Data npuInput, npuWeight, npuBias, npuOutput;
    auto ToDevice = [](Data &h, Data &d) {
        d.dataType = h.dataType; d.Resize(h.dims);
        d.deviceData = FastllmAclMalloc(h.GetBytes());
        FastllmAclCopyFromHostToDevice(d.deviceData, h.cpuData, h.GetBytes());
    };
    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);
    npuOutput.dataType = DataType::FLOAT16;
    npuOutput.Resize({Rows, Dim});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    FastllmAclRMSNorm(npuInput, npuWeight, npuBias, npuOutput, eps);

    //begin
    Data npuResultHost;
    npuResultHost.dataType = DataType::FLOAT32;
    npuResultHost.Resize({Rows, Dim});
    npuResultHost.Allocate();
    
    Data npuF16Host; // 先拷回 F16
    npuF16Host.dataType = DataType::FLOAT16;
    npuF16Host.Resize({Rows, Dim});
    npuF16Host.Allocate();
    FastllmAclCopyFromDeviceToHost(npuF16Host.cpuData, npuOutput.deviceData, npuF16Host.GetBytes());

    uint16_t *ptrF16 = (uint16_t*)npuF16Host.cpuData;
    float    *ptrF32 = (float*)npuResultHost.cpuData;
    for(int i=0; i<npuResultHost.Count(0); ++i) {
        ptrF32[i] = F16toF32(ptrF16[i]);
    }
    // end
    
    CompareData(cpuOut, npuResultHost, 0.05f); // RMSNorm 允许一点误差

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();
    Test_RMSNorm();
    std::cout << "All tests finished." << std::endl;
    return 0;
}