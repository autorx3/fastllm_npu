#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring> // for memcpy

using namespace fastllm;

// ==========================================
// 辅助函数：生成随机数据
// ==========================================
void FillRandom(Data &data, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    if (data.cpuData == nullptr) {
        data.Allocate();
    }
    
    float *ptr = (float*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) {
        ptr[i] = dis(gen);
    }
}

// ==========================================
// 辅助函数：对比 CPU 和 NPU 结果
// ==========================================
bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 1e-3) {
    // npuOut 已经在 Test 函数里被 CopyDeviceToHost 填充了 cpuData
    if (cpuOut.dims != npuOut.dims) {
        std::cerr << "[FAIL] Dims mismatch!" << std::endl;
        return false;
    }

    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData;
    int len = cpuOut.Count(0);

    for (int i = 0; i < len; ++i) {
        float diff = std::abs(pCpu[i] - pNpu[i]);
        if (diff > maxErr) maxErr = diff;
    }

    std::cout << " -> Max Error: " << std::fixed << std::setprecision(6) << maxErr;
    
    if (maxErr > threshold) {
        std::cout << " [FAIL] (Threshold: " << threshold << ")" << std::endl;
        return false;
    }
    std::cout << " [PASS]" << std::endl;
    return true;
}

// ==========================================
// 测试用例 1: 基础 Element-wise (Silu)
// ==========================================
void Test_Silu() {
    std::cout << "=== Testing Silu ===";
    
    std::vector<int> dims = {4, 128, 4096};
    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    Data cpuOut;
    // 手动深拷贝 input 到 cpuOut
    cpuOut.dataType = input.dataType;
    cpuOut.Resize(input.dims);
    cpuOut.Allocate();
    memcpy(cpuOut.cpuData, input.cpuData, input.GetBytes());

    // CPU 计算 Silu
    float *p = (float*)cpuOut.cpuData;
    for(int i=0; i<cpuOut.Count(0); i++) {
        float x = p[i];
        p[i] = x / (1.0f + expf(-x));
    }

    // NPU 计算
    // 1. 准备 Device 内存
    Data npuRealInput;
    npuRealInput.dataType = DataType::FLOAT32;
    npuRealInput.Resize(dims);
    npuRealInput.deviceData = FastllmAclMalloc(npuRealInput.GetBytes());
    FastllmAclCopyFromHostToDevice(npuRealInput.deviceData, input.cpuData, input.GetBytes());

    Data npuRealOutput;
    npuRealOutput.dataType = DataType::FLOAT32;
    npuRealOutput.Resize(dims);
    npuRealOutput.deviceData = FastllmAclMalloc(npuRealOutput.GetBytes());

    // 2. 调用算子
    FastllmAclSilu(npuRealInput, npuRealOutput);

    // 3. 拷回 Host
    Data npuOut;
    npuOut.dataType = DataType::FLOAT32;
    npuOut.Resize(dims);
    npuOut.Allocate();
    FastllmAclCopyFromDeviceToHost(npuOut.cpuData, npuRealOutput.deviceData, npuOut.GetBytes());

    CompareData(cpuOut, npuOut);

    FastllmAclFree(npuRealInput.deviceData);
    FastllmAclFree(npuRealOutput.deviceData);
}

// ==========================================
// 测试用例 2: 矩阵乘法 (MatMul)
// ==========================================
void Test_MatMul_FP16() {
    std::cout << "=== Testing MatMul (FP16) ===";
    
    Data input, weight, outputCPU;
    
    input.dataType = DataType::FLOAT32;
    input.Resize({128, 512});
    FillRandom(input);

    weight.dataType = DataType::FLOAT32;
    weight.Resize({1024, 512}); 
    FillRandom(weight);

    // CPU 计算
    outputCPU.dataType = DataType::FLOAT32;
    outputCPU.Resize({128, 1024});
    outputCPU.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* w = (float*)weight.cpuData;
    float* out = (float*)outputCPU.cpuData;
    
    #pragma omp parallel for
    for (int m = 0; m < 128; ++m) {
        for (int n = 0; n < 1024; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < 512; ++k) {
                sum += inp[m * 512 + k] * w[n * 512 + k]; 
            }
            out[m * 1024 + n] = sum;
        }
    }

    // NPU 计算
    Data npuInput, npuWeight, npuOutput;
    // 构造 NPU Tensor wrapper (只分配 deviceData)
    
    npuInput.dataType = DataType::FLOAT32; npuInput.Resize(input.dims);
    npuInput.deviceData = FastllmAclMalloc(input.GetBytes());
    FastllmAclCopyFromHostToDevice(npuInput.deviceData, input.cpuData, input.GetBytes());
    
    npuWeight.dataType = DataType::FLOAT32; npuWeight.Resize(weight.dims);
    npuWeight.deviceData = FastllmAclMalloc(weight.GetBytes());
    FastllmAclCopyFromHostToDevice(npuWeight.deviceData, weight.cpuData, weight.GetBytes());
    
    npuOutput.dataType = DataType::FLOAT32; npuOutput.Resize(outputCPU.dims);
    npuOutput.deviceData = FastllmAclMalloc(outputCPU.GetBytes());

    Data emptyBias;
    FastllmAclMatMulTransB(npuInput, npuWeight, emptyBias, npuOutput, 1, 0);

    Data resHost;
    resHost.dataType = DataType::FLOAT32;
    resHost.Resize(outputCPU.dims);
    resHost.Allocate();
    FastllmAclCopyFromDeviceToHost(resHost.cpuData, npuOutput.deviceData, resHost.GetBytes());

    CompareData(outputCPU, resHost, 1e-1); 

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();

    Test_Silu();
    Test_MatMul_FP16();

    std::cout << "All tests finished." << std::endl;
    return 0;
}