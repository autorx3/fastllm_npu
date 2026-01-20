#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring>
#include <cstdint> // for uint16_t, int8_t

using namespace fastllm;

// ==========================================
// 辅助工具：FP32 <-> FP16 转换
// ==========================================
uint16_t F32toF16(float value) {
    uint32_t x = *(uint32_t*)&value;
    uint16_t h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
    return h;
}

float F16toF32(uint16_t value) {
    uint32_t t = ((value & 0x8000) << 16) | (((value & 0x7c00) + 0x1C000) << 13) | ((value & 0x03FF) << 13);
    return *(float*)&t;
}

// ==========================================
// 辅助函数：数据填充
// ==========================================
void FillRandomInt8(Data &data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-127, 127);
    if (data.cpuData == nullptr) data.Allocate();
    int8_t *ptr = (int8_t*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = (int8_t)dis(gen);
}

void FillRandomF16(Data &data, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    uint16_t *ptr = (uint16_t*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = F32toF16(dis(gen));
}

void FillRandomF32(Data &data, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    float *ptr = (float*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = dis(gen);
}

// ==========================================
// 辅助函数：对比结果
// ==========================================
bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 0.5f) {
    if (cpuOut.dims != npuOut.dims) {
        std::cerr << "[FAIL] Dims mismatch!" << std::endl;
        return false;
    }

    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData;
    int len = cpuOut.Count(0);
    int errIdx = -1;

    for (int i = 0; i < len; ++i) {
        float diff = std::abs(pCpu[i] - pNpu[i]);
        if (diff > maxErr) {
            maxErr = diff;
            errIdx = i;
        }
    }

    std::cout << " -> Max Error: " << std::fixed << std::setprecision(6) << maxErr;
    if (maxErr > threshold) {
        std::cout << " [FAIL] (Threshold: " << threshold << ")" << std::endl;
        std::cout << "    At index " << errIdx << ": CPU=" << pCpu[errIdx] << " NPU=" << pNpu[errIdx] << std::endl;
        return false;
    }
    std::cout << " [PASS]" << std::endl;
    return true;
}

// ==========================================
// 测试用例: Dynamic QuantLinear (FP16 In, INT8 W, FP16 Out)
// ==========================================
void Test_QuantLinear_Dynamic_W8A16() {
    std::cout << "=== Testing QuantLinear (Dynamic Mode: FP16 In, INT8 W, FP16 Out) ===" << std::endl;

    // 1. 定义维度
    int M = 16;   // Batch * SeqLen
    int K = 256;  // Input Hidden Size
    int N = 128;  // Output Hidden Size

    // 2. 准备 Host 数据
    
    // (A) Input: FP16 [M, K]
    Data input;
    input.dataType = DataType::FLOAT16;
    input.Resize({M, K});
    FillRandomF16(input, -1.0f, 1.0f);

    // (B) Weight: INT8 [N, K]
    Data weight;
    weight.dataType = DataType::INT8; 
    weight.Resize({N, K});
    FillRandomInt8(weight);

    // (C) WeightScale: FP32 [N]
    Data wScale;
    wScale.dataType = DataType::FLOAT32;
    wScale.Resize({N});
    FillRandomF32(wScale, 0.002f, 0.005f);

    // (D) xScale: 【关键点】置空，触发 NPU 动态量化
    Data xScale; // 空 Tensor

    // (E) Bias: 【关键点】置空
    Data bias;   // 空 Tensor

    // 3. CPU Ground Truth 计算 (FP32 Golden Reference)
    // 逻辑：虽然 NPU 内部做了动态量化，但在数学上我们期望结果接近 Input_FP32 * Weight_FP32
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize({M, N});
    cpuOut.Allocate();
    
    uint16_t *pIn = (uint16_t*)input.cpuData;
    int8_t   *pW  = (int8_t*)weight.cpuData;
    float    *pWs = (float*)wScale.cpuData;
    float    *pOut = (float*)cpuOut.cpuData;

    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            float curWScale = pWs[n];
            for (int k = 0; k < K; ++k) {
                // 还原 Input FP16 -> FP32
                float val_x = F16toF32(pIn[m * K + k]);
                // 还原 Weight INT8 -> FP32
                float val_w = (float)pW[n * K + k] * curWScale;
                sum += val_x * val_w;
            }
            pOut[m * N + n] = sum;
        }
    }

    // 4. NPU 计算
    Data npuInput, npuWeight, npuWScale, npuXScale, npuBias, npuOutput;

    auto ToDevice = [](Data &host, Data &dev) {
        if (host.dims.empty()) return; // 空 Tensor 不分配
        dev.dataType = host.dataType;
        dev.Resize(host.dims);
        dev.deviceData = FastllmAclMalloc(host.GetBytes());
        FastllmAclCopyFromHostToDevice(dev.deviceData, host.cpuData, host.GetBytes());
    };

    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);
    ToDevice(wScale, npuWScale);
    // xScale 和 bias 为空，不调用 ToDevice，保持 deviceData 为 nullptr

    // Output: 【关键点】必须是 FP16
    npuOutput.dataType = DataType::FLOAT16; 
    npuOutput.Resize({M, N});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    // 调用算子
    // 传入空的 npuXScale 和 npuBias
    FastllmAclQuantLinearDequant(npuInput, npuWeight, npuWScale, npuXScale, npuBias, npuOutput);

    // 5. 拷回结果并格式转换
    Data npuResultF16;
    npuResultF16.dataType = DataType::FLOAT16;
    npuResultF16.Resize({M, N});
    npuResultF16.Allocate();
    
    // 从 NPU 拷贝 FP16 数据回来
    FastllmAclCopyFromDeviceToHost(npuResultF16.cpuData, npuOutput.deviceData, npuResultF16.GetBytes());

    // 将 NPU 的 FP16 结果转为 FP32 以便和 CPU 结果对比
    Data npuResultF32;
    npuResultF32.dataType = DataType::FLOAT32;
    npuResultF32.Resize({M, N});
    npuResultF32.Allocate();
    
    uint16_t *ptrNpuF16 = (uint16_t*)npuResultF16.cpuData;
    float    *ptrNpuF32 = (float*)npuResultF32.cpuData;
    
    for(int i=0; i<npuResultF32.Count(0); i++) {
        ptrNpuF32[i] = F16toF32(ptrNpuF16[i]);
    }

    // 6. 比较
    CompareData(cpuOut, npuResultF32, 0.5f);

    // 7. 释放
    if(npuInput.deviceData) FastllmAclFree(npuInput.deviceData);
    if(npuWeight.deviceData) FastllmAclFree(npuWeight.deviceData);
    if(npuWScale.deviceData) FastllmAclFree(npuWScale.deviceData);
    if(npuOutput.deviceData) FastllmAclFree(npuOutput.deviceData);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();

    Test_QuantLinear_Dynamic_W8A16();

    std::cout << "All tests finished." << std::endl;
    return 0;
}