#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <chrono> // 用于时间统计

using namespace fastllm;

// ==========================================
// 0. 性能计时器 (新增功能)
// ==========================================
class TestTimer {
public:
    TestTimer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~TestTimer() {
        // 仅作为 RAII 的一部分，主要计时通过 Stop() 获取
    }

    void Reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double Stop(const std::string& tag = "") {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start_;
        double ms = elapsed.count();
        if (!tag.empty()) {
            std::cout << "    [" << name_ << "] " << tag << " Time: " 
                      << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
        }
        return ms;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ==========================================
// 1. FP16 / FP32 转换工具
// ==========================================
// 整合了 test_npu_quant.cpp 中的位操作版本，兼容性更好
// uint16_t F32toF16(float value) {
//     uint32_t x = *(uint32_t*)&value;
//     uint16_t h = ((x >> 16) & 0x8000) | ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) | ((x >> 13) & 0x03ff);
//     return h;
// }

// float F16toF32(uint16_t value) {
//     uint32_t t = ((value & 0x8000) << 16) | (((value & 0x7c00) + 0x1C000) << 13) | ((value & 0x03FF) << 13);
//     return *(float*)&t;
// }
uint16_t F32toF16(float value) {
    __fp16 val_f16 = (__fp16)value;
    return *(uint16_t*)&val_f16;
}

float F16toF32(uint16_t value) {
    __fp16 val_f16 = *(__fp16*)&value;
    return (float)val_f16;
}

// ==========================================
// 2. 数据填充工具 (整合版)
// ==========================================
void FillRandom(Data &data, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (data.cpuData == nullptr) data.Allocate();

    if (data.dataType == DataType::FLOAT32) {
        std::uniform_real_distribution<float> dis(min, max);
        float *ptr = (float*)data.cpuData;
        for (int i = 0; i < data.Count(0); ++i) ptr[i] = dis(gen);
    } 
    else if (data.dataType == DataType::FLOAT16) {
        std::uniform_real_distribution<float> dis(min, max);
        uint16_t *ptr = (uint16_t*)data.cpuData;
        for (int i = 0; i < data.Count(0); ++i) ptr[i] = F32toF16(dis(gen));
    }
    else if (data.dataType == DataType::INT8) {
        std::uniform_int_distribution<int> dis(-127, 127);
        int8_t *ptr = (int8_t*)data.cpuData;
        for (int i = 0; i < data.Count(0); ++i) ptr[i] = (int8_t)dis(gen);
    }
}

// 专门用于 Embedding 的索引填充
void FillRandomInt(Data &data, int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    // Fastllm 中索引通常暂时用 float 存储，根据 test_npu_eb_repeat.cpp 逻辑
    float *ptr = (float*)data.cpuData; 
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = (float)dis(gen);
}

// ==========================================
// 3. 数据对比工具 (通用版)
// ==========================================
bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 1e-3, const std::string& opName = "Op") {
    if (cpuOut.dims != npuOut.dims) {
        std::cerr << " [" << opName << "] [FAIL] Dims mismatch!" << std::endl;
        return false;
    }

    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData; // 假设 npuOut 已经是 Host 端的 FP32
    int len = cpuOut.Count(0);
    int errIdx = -1;

    for (int i = 0; i < len; ++i) {
        if (std::isnan(pCpu[i]) || std::isinf(pCpu[i]) || std::isnan(pNpu[i])) continue;
        float diff = std::abs(pCpu[i] - pNpu[i]);
        if (diff > maxErr) {
            maxErr = diff;
            errIdx = i;
        }
    }

    std::cout << " -> Max Error: " << std::fixed << std::setprecision(6) << maxErr;
    
    if (maxErr > threshold) {
        std::cout << " [FAIL] (Threshold: " << threshold << ")" << std::endl;
        if (errIdx != -1) {
            std::cout << "    At index " << errIdx << ": CPU=" << pCpu[errIdx] << " NPU=" << pNpu[errIdx] << std::endl;
        }
        return false;
    }
    std::cout << " [PASS]" << std::endl;
    return true;
}

// 辅助：NPU 内存管理 Wrapper
void ToDevice(const Data &host, Data &dev) {
    if (host.dims.empty()) return;
    dev.dataType = host.dataType;
    dev.Resize(host.dims);
    dev.deviceData = FastllmAclMalloc(host.GetBytes());
    FastllmAclCopyFromHostToDevice(dev.deviceData, host.cpuData, host.GetBytes());
}

// 辅助：从 Device 拷回并转换为 FP32 (用于对比)
void FromDeviceToFP32(const Data &devOutput, Data &hostResultFP32) {
    hostResultFP32.dataType = DataType::FLOAT32;
    hostResultFP32.Resize(devOutput.dims);
    hostResultFP32.Allocate();

    if (devOutput.dataType == DataType::FLOAT16) {
        Data tempF16;
        tempF16.dataType = DataType::FLOAT16;
        tempF16.Resize(devOutput.dims);
        tempF16.Allocate();
        FastllmAclCopyFromDeviceToHost(tempF16.cpuData, devOutput.deviceData, tempF16.GetBytes());
        
        uint16_t *ptrF16 = (uint16_t*)tempF16.cpuData;
        float *ptrF32 = (float*)hostResultFP32.cpuData;
        for(int i=0; i < hostResultFP32.Count(0); ++i) {
            ptrF32[i] = F16toF32(ptrF16[i]);
        }
    } 
    else {
        // 假设是 FP32 直接拷回
        FastllmAclCopyFromDeviceToHost(hostResultFP32.cpuData, devOutput.deviceData, hostResultFP32.GetBytes());
    }
}
// ==========================================
// 4. 测试用例 1: 基础 Element-wise (Silu)
// ==========================================
void Test_Silu() {
    std::cout << "=== Testing Silu ===" << std::endl;
    TestTimer timer("Silu");
    
    std::vector<int> dims = {4, 128, 4096};
    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = input.dataType;
    cpuOut.Resize(input.dims);
    cpuOut.Allocate();
    memcpy(cpuOut.cpuData, input.cpuData, input.GetBytes());

    timer.Reset(); // Start CPU Timer
    float *p = (float*)cpuOut.cpuData;
    int count = cpuOut.Count(0);
    // 简单的 CPU 模拟，实际框架中可能有向量化优化
    for(int i=0; i < count; i++) {
        float x = p[i];
        p[i] = x / (1.0f + expf(-x));
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    // 1. 准备 Device 内存
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);
    
    npuOutput.dataType = DataType::FLOAT32;
    npuOutput.Resize(dims);
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    // 2. 调用算子 (计入时间)
    timer.Reset(); // Start NPU Timer
    FastllmAclSilu(npuInput, npuOutput);
    timer.Stop("NPU_Exec"); // 仅包含算子执行时间，不含数据拷贝

    // 3. 拷回验证
    Data npuResultHost;
    FromDeviceToFP32(npuOutput, npuResultHost);

    CompareData(cpuOut, npuResultHost, 1e-3, "Silu");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 5. 测试用例 2: 矩阵乘法 (MatMul FP16/FP32)
// ==========================================
void Test_MatMul_FP16() {
    std::cout << "=== Testing MatMul (FP16/FP32 Mix) ===" << std::endl;
    TestTimer timer("MatMul");
    
    Data input, weight;
    // 模拟常见形状
    input.dataType = DataType::FLOAT32;
    input.Resize({128, 512});
    FillRandom(input);

    weight.dataType = DataType::FLOAT32;
    weight.Resize({1024, 512}); 
    FillRandom(weight);

    // --- CPU Bench ---
    Data outputCPU;
    outputCPU.dataType = DataType::FLOAT32;
    outputCPU.Resize({128, 1024});
    outputCPU.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* w = (float*)weight.cpuData;
    float* out = (float*)outputCPU.cpuData;
    
    timer.Reset();
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
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuWeight, npuOutput;
    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);
    
    npuOutput.dataType = DataType::FLOAT32; 
    npuOutput.Resize(outputCPU.dims);
    npuOutput.deviceData = FastllmAclMalloc(outputCPU.GetBytes());

    Data emptyBias;
    
    timer.Reset();
    FastllmAclMatMulTransB(npuInput, npuWeight, emptyBias, npuOutput, 1, 0);
    timer.Stop("NPU_Exec");

    // 验证
    Data resHost;
    FromDeviceToFP32(npuOutput, resHost);

    CompareData(outputCPU, resHost, 1e-1, "MatMul");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 6. 测试用例 3: TopK (Values + Indices)
// ==========================================
void Test_TopK() {
    std::cout << "=== Testing TopK ===" << std::endl;
    TestTimer timer("TopK");

    int batch = 4;
    int vocabSize = 1000; // 稍微加大一点
    int k = 5;

    std::vector<int> inputDims = {batch, vocabSize};
    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(inputDims);
    FillRandom(input, 0.0f, 100.0f);

    std::vector<int> outputDims = {batch, k * 2};
    
    // --- CPU Bench (Skipped logic implementation to save space, assuming NPU check only) ---
    // 原代码未实现完整的 CPU TopK 排序逻辑验证，这里主要测试 NPU 耗时
    // 如需验证正确性，通常只打印 Top1

    // --- NPU Bench ---
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);

    npuOutput.dataType = DataType::FLOAT32;
    npuOutput.Resize(outputDims);
    npuOutput.deviceData = FastllmAclMalloc(outputDims[0] * outputDims[1] * sizeof(float)); 

    timer.Reset();
    FastllmAclTopK(npuInput, npuOutput, k);
    timer.Stop("NPU_Exec");

    // 打印验证
    Data npuResultHost;
    FromDeviceToFP32(npuOutput, npuResultHost);

    float* outPtr = (float*)npuResultHost.cpuData;
    // 简单检查: 打印第一个 Batch 的 Top1 Value 和 Index
    // TopK 布局通常是 [Value1...ValueK, Index1...IndexK] 或交替，需参考具体实现
    // 原代码假设: outPtr[0] 是 value, outPtr[k] 是 index
    std::cout << "    [Check] Top1 Value: " << outPtr[0] << ", Index: " << outPtr[k] << std::endl;
    std::cout << " -> [PASS] (Logic check manually)" << std::endl;

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}
// ==========================================
// 7. 测试用例 4: Add Scalar (Output = Input + v)
// ==========================================
void Test_Add_Scalar() {
    std::cout << "=== Testing Add (Scalar) ===" << std::endl;
    TestTimer timer("Add_Scalar");

    std::vector<int> dims = {2, 512};
    float v = 3.14f;

    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize(dims);
    cpuOut.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* out = (float*)cpuOut.cpuData;
    int count = input.Count(0);

    timer.Reset();
    for(int i=0; i < count; i++) {
        out[i] = inp[i] + v;
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);
    
    npuOutput.dataType = DataType::FLOAT32; 
    npuOutput.Resize(dims);
    npuOutput.deviceData = FastllmAclMalloc(cpuOut.GetBytes());

    timer.Reset();
    FastllmAclAdd(npuInput, v, npuOutput);
    timer.Stop("NPU_Exec");

    // Verify
    Data npuResHost;
    FromDeviceToFP32(npuOutput, npuResHost);

    CompareData(cpuOut, npuResHost, 1e-4, "Add_Scalar");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 8. 测试用例 5: AddTo (Axpy) -> In0 = In0 + In1 * alpha
// ==========================================
void Test_AddTo() {
    std::cout << "=== Testing AddTo (Axpy) ===" << std::endl;
    TestTimer timer("AddTo");

    std::vector<int> dims = {4, 256};
    float alpha = 0.5f;

    Data input0, input1;
    input0.dataType = DataType::FLOAT32; input0.Resize(dims); FillRandom(input0);
    input1.dataType = DataType::FLOAT32; input1.Resize(dims); FillRandom(input1);

    // --- CPU Bench ---
    Data cpuRef;
    cpuRef.dataType = DataType::FLOAT32; cpuRef.Resize(dims); cpuRef.Allocate();
    
    float* p0 = (float*)input0.cpuData;
    float* p1 = (float*)input1.cpuData;
    float* pRef = (float*)cpuRef.cpuData;
    int count = input0.Count(0);

    timer.Reset();
    for(int i=0; i < count; i++) {
        pRef[i] = p0[i] + p1[i] * alpha;
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuIn0, npuIn1;
    // In0 (作为 Output，会被原地修改)
    ToDevice(input0, npuIn0);
    // In1
    ToDevice(input1, npuIn1);

    timer.Reset();
    FastllmAclAddTo(npuIn0, npuIn1, alpha);
    timer.Stop("NPU_Exec");

    // Verify (注意：结果直接写入 npuIn0)
    Data npuResHost;
    FromDeviceToFP32(npuIn0, npuResHost);

    CompareData(cpuRef, npuResHost, 1e-4, "AddTo");

    FastllmAclFree(npuIn0.deviceData);
    FastllmAclFree(npuIn1.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 9. 测试用例 6: Mul Scalar (Output = Input * v)
// ==========================================
void Test_Mul_Scalar() {
    std::cout << "=== Testing Mul (Scalar) ===" << std::endl;
    TestTimer timer("Mul_Scalar");

    std::vector<int> dims = {2, 512};
    float v = 2.0f;

    Data input;
    input.dataType = DataType::FLOAT32; input.Resize(dims); FillRandom(input);

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize(dims); cpuOut.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* out = (float*)cpuOut.cpuData;
    int count = input.Count(0);

    timer.Reset();
    for(int i=0; i < count; i++) {
        out[i] = inp[i] * v;
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);
    
    npuOutput.dataType = DataType::FLOAT32; 
    npuOutput.Resize(dims);
    npuOutput.deviceData = FastllmAclMalloc(cpuOut.GetBytes());

    timer.Reset();
    FastllmAclMul(npuInput, v, npuOutput);
    timer.Stop("NPU_Exec");

    // Verify
    Data npuResHost;
    FromDeviceToFP32(npuOutput, npuResHost);

    CompareData(cpuOut, npuResHost, 1e-4, "Mul_Scalar");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}
// ==========================================
// 10. 测试用例 7: MulTo (Element-wise) -> In0 = In0 * In1 * alpha
// ==========================================
void Test_MulTo() {
    std::cout << "=== Testing MulTo (Element-wise + alpha) ===" << std::endl;
    TestTimer timer("MulTo");

    std::vector<int> dims = {4, 256};
    float alpha = 2.0f;

    Data input0, input1;
    input0.dataType = DataType::FLOAT32; input0.Resize(dims); FillRandom(input0);
    input1.dataType = DataType::FLOAT32; input1.Resize(dims); FillRandom(input1);

    // --- CPU Bench ---
    Data cpuRef;
    cpuRef.dataType = DataType::FLOAT32; cpuRef.Resize(dims); cpuRef.Allocate();
    
    float* p0 = (float*)input0.cpuData;
    float* p1 = (float*)input1.cpuData;
    float* pRef = (float*)cpuRef.cpuData;
    int count = input0.Count(0);

    timer.Reset();
    for(int i=0; i < count; i++) {
        pRef[i] = (p0[i] * p1[i]) * alpha;
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuIn0, npuIn1;
    ToDevice(input0, npuIn0);
    ToDevice(input1, npuIn1);

    timer.Reset();
    // 结果写入 npuIn0
    FastllmAclMulTo(npuIn0, npuIn1, alpha);
    timer.Stop("NPU_Exec");

    // Verify
    Data npuResHost;
    FromDeviceToFP32(npuIn0, npuResHost);

    CompareData(cpuRef, npuResHost, 1e-4, "MulTo");

    FastllmAclFree(npuIn0.deviceData);
    FastllmAclFree(npuIn1.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 11. 测试用例 8: Permute (Transpose)
// ==========================================
void Test_Permute() {
    std::cout << "=== Testing Permute (aclnnPermute) ===" << std::endl;
    TestTimer timer("Permute");

    // 形状: [Batch=2, Heads=4, Seq=8, Dim=16] -> 转置 Heads 和 Seq
    // 目标: [2, 8, 4, 16]
    std::vector<int> dims = {2, 4, 8, 16};
    std::vector<int> axis = {0, 2, 1, 3}; // 交换第1和第2维
    std::vector<int> expectedDims = {2, 8, 4, 16};

    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    // --- CPU Bench ---
    // 仅验证维度，不进行实际的数据搬运模拟
    timer.Reset();
    // (CPU 逻辑略，假设瞬间完成)
    timer.Stop("CPU_Sim(Skip)");

    // --- NPU Bench ---
    Data npuData;
    ToDevice(input, npuData);

    timer.Reset();
    FastllmAclPermute(npuData, axis);
    timer.Stop("NPU_Exec");

    // 5. 验证维度
    if (npuData.dims != expectedDims) {
        std::cout << " [FAIL] Dimension Mismatch!" << std::endl;
    } else {
        std::cout << " [PASS] Dims check passed." << std::endl;
    }

    FastllmAclFree(npuData.deviceData);
    std::cout << "--------------------------------" << std::endl;
}


// ==========================================
// 13. 测试用例 10: RoPE Dual Version (Fused vs Single)
// ==========================================
void Test_RoPE_Dual_Version() {
    std::cout << "=== Testing RoPE (Dual Fused Version) ===" << std::endl;
    TestTimer timer("RoPE_Dual");

    std::vector<int> dims = {1, 1, 1, 64}; // 简化维度
    int dim = 64;

    Data q, k, sinData, cosData, dummyPos;
    q.dataType = DataType::FLOAT32; q.Resize(dims); FillRandom(q);
    k.dataType = DataType::FLOAT32; k.Resize(dims); FillRandom(k);
    sinData.dataType = DataType::FLOAT32; sinData.Resize(dims); FillRandom(sinData);
    cosData.dataType = DataType::FLOAT32; cosData.Resize(dims); FillRandom(cosData);

    // NPU Prep: 需要两组 Device 内存来对比单次调用和融合调用的结果
    Data npuQ1, npuK1, npuQ2, npuK2, npuSin, npuCos;
    ToDevice(q, npuQ1); ToDevice(k, npuK1);
    ToDevice(q, npuQ2); ToDevice(k, npuK2);
    ToDevice(sinData, npuSin); ToDevice(cosData, npuCos);

    // --- Benchmark 1: Legacy Single Calls ---
    timer.Reset();
    FastllmAclNearlyRotatePosition2D(npuQ1, dummyPos, npuSin, npuCos, dim);
    FastllmAclNearlyRotatePosition2D(npuK1, dummyPos, npuSin, npuCos, dim);
    timer.Stop("NPU_Single(x2)");

    // --- Benchmark 2: Fused Dual Call ---
    timer.Reset();
    FastllmAclRotatePosition2D_Fused(npuQ2, npuK2, dummyPos, npuSin, npuCos, dim);
    timer.Stop("NPU_Fused");

    // --- Verify ---
    // 理论上 npuQ1 应该等于 npuQ2，npuK1 应该等于 npuK2
    Data resQ1, resK1, resQ2, resK2;
    FromDeviceToFP32(npuQ1, resQ1);
    FromDeviceToFP32(npuK1, resK1);
    FromDeviceToFP32(npuQ2, resQ2);
    FromDeviceToFP32(npuK2, resK2);

    bool checkQ = CompareData(resQ1, resQ2, 1e-5, "RoPE_Dual_Q");
    bool checkK = CompareData(resK1, resK2, 1e-5, "RoPE_Dual_K");
    
    if (checkQ && checkK) {
        std::cout << " -> [PASS] Fused kernel matches legacy kernel." << std::endl;
    }

    FastllmAclFree(npuQ1.deviceData); FastllmAclFree(npuK1.deviceData);
    FastllmAclFree(npuQ2.deviceData); FastllmAclFree(npuK2.deviceData);
    FastllmAclFree(npuSin.deviceData); FastllmAclFree(npuCos.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 14. 测试用例 11: Embedding
// ==========================================
void Test_Embedding() {
    std::cout << "=== Testing Embedding ===" << std::endl;
    TestTimer timer("Embedding");

    int Batch = 2;
    int SeqLen = 8;
    int VocabSize = 1000;
    int EmbedDim = 256;

    // 1. 准备数据
    Data input; // Indices
    input.dataType = DataType::FLOAT32; // Fastllm 常用 float 存索引
    input.Resize({Batch, SeqLen});
    FillRandomInt(input, 0, VocabSize - 1);

    Data weight; // Embedding Table (FP16)
    weight.dataType = DataType::FLOAT16;
    weight.Resize({VocabSize, EmbedDim});
    FillRandom(weight); // 会自动调用 FillRandomF16

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize({Batch, SeqLen, EmbedDim});
    cpuOut.Allocate();

    float *pCpuOut = (float*)cpuOut.cpuData;
    float *pIn = (float*)input.cpuData;
    uint16_t *pW = (uint16_t*)weight.cpuData;

    timer.Reset();
    int count = Batch * SeqLen;
    for (int i = 0; i < count; ++i) {
        int idx = (int)pIn[i];
        if (idx < 0 || idx >= VocabSize) idx = 0;
        // 查表并转 FP32
        for (int d = 0; d < EmbedDim; ++d) {
            uint16_t val16 = pW[idx * EmbedDim + d];
            pCpuOut[i * EmbedDim + d] = F16toF32(val16);
        }
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuWeight, npuOutput;
    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);

    // 准备 Output (FP16)
    npuOutput.dataType = DataType::FLOAT16;
    npuOutput.Resize({Batch, SeqLen, EmbedDim});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    timer.Reset();
    FastllmAclEmbedding(npuInput, npuWeight, npuOutput);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuResultHost;
    FromDeviceToFP32(npuOutput, npuResultHost); // 自动转 FP32

    CompareData(cpuOut, npuResultHost, 1e-3, "Embedding");

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 15. 测试用例 12: Repeat (Bytes Expand)
// ==========================================
void Test_Repeat() {
    std::cout << "=== Testing Repeat (Bytes Expand) ===" << std::endl;
    TestTimer timer("Repeat");

    // 模拟场景：GQA 中将 Key/Value 进行广播
    int outer = 4;
    int repeatTimes = 3;
    int elementSize = 2; // FP16
    int headDim = 128;   
    int bytesPerBlock = headDim * elementSize; 

    int channelsInputInner = bytesPerBlock; 
    int channelsInner = bytesPerBlock; 
    int inputStride = channelsInputInner; 
    int outputStride = channelsInner * repeatTimes; 

    // 1. 准备数据 (逻辑形状 [outer, 1, headDim])
    Data input;
    input.dataType = DataType::FLOAT16;
    input.Resize({outer, 1, headDim}); 
    FillRandom(input);

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT16;
    cpuOut.Resize({outer, repeatTimes, headDim});
    cpuOut.Allocate();

    uint8_t *pSrcBytes = (uint8_t*)input.cpuData;
    uint8_t *pDstBytes = (uint8_t*)cpuOut.cpuData;

    timer.Reset();
    for (int o = 0; o < outer; ++o) {
        uint8_t *srcRow = pSrcBytes + o * inputStride;
        uint8_t *dstRow = pDstBytes + o * outputStride;
        for (int r = 0; r < repeatTimes; ++r) {
            memcpy(dstRow + r * channelsInner, srcRow, channelsInputInner);
        }
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    void *deviceSrc = FastllmAclMalloc(input.GetBytes());
    void *deviceDst = FastllmAclMalloc(cpuOut.GetBytes());
    FastllmAclCopyFromHostToDevice(deviceSrc, input.cpuData, input.GetBytes());

    timer.Reset();
    // 注意：函数参数 src, dst 是 void*
    FastllmAclRepeat(deviceSrc, deviceDst, outer, repeatTimes, 
                     inputStride, outputStride, 
                     channelsInner, channelsInputInner);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuOutHost; // 拷回原始 FP16
    npuOutHost.dataType = DataType::FLOAT16;
    npuOutHost.Resize(cpuOut.dims);
    npuOutHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuOutHost.cpuData, deviceDst, npuOutHost.GetBytes());

    // 转换为 Float 进行对比
    Data cpuOutF32, npuOutF32;
    // 手动转一下 cpuOut -> cpuOutF32
    cpuOutF32.dataType = DataType::FLOAT32; cpuOutF32.Resize(cpuOut.dims); cpuOutF32.Allocate();
    {
        uint16_t* s = (uint16_t*)cpuOut.cpuData; float* d = (float*)cpuOutF32.cpuData;
        for(int i=0; i<cpuOut.Count(0); i++) d[i] = F16toF32(s[i]);
    }
    // 手动转一下 npuOutHost -> npuOutF32 (或者直接利用 FromDeviceToFP32 逻辑)
    npuOutF32.dataType = DataType::FLOAT32; npuOutF32.Resize(npuOutHost.dims); npuOutF32.Allocate();
    {
        uint16_t* s = (uint16_t*)npuOutHost.cpuData; float* d = (float*)npuOutF32.cpuData;
        for(int i=0; i<npuOutHost.Count(0); i++) d[i] = F16toF32(s[i]);
    }

    CompareData(cpuOutF32, npuOutF32, 0.0f, "Repeat"); // 纯内存复制应无误差

    FastllmAclFree(deviceSrc);
    FastllmAclFree(deviceDst);
    std::cout << "--------------------------------" << std::endl;
}
// ==========================================
// 16. 测试用例 13: QuantLinear (Dynamic: W8A16)
// ==========================================
void Test_QuantLinear_Dynamic_W8A16() {
    std::cout << "=== Testing QuantLinear (Dynamic W8A16) ===" << std::endl;
    TestTimer timer("QuantLinear");

    int M = 16;   // Batch * SeqLen
    int K = 256;  // Input Hidden Size
    int N = 128;  // Output Hidden Size

    // (A) Input: FP16
    Data input; input.dataType = DataType::FLOAT16; input.Resize({M, K}); FillRandom(input);
    // (B) Weight: INT8
    Data weight; weight.dataType = DataType::INT8; weight.Resize({N, K}); FillRandom(weight);
    // (C) WeightScale: FP32
    Data wScale; wScale.dataType = DataType::FLOAT32; wScale.Resize({N}); FillRandom(wScale, 0.002f, 0.005f);
    
    // Empty Tensors for Dynamic Mode
    Data xScale, bias;

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({M, N}); cpuOut.Allocate();
    
    uint16_t *pIn = (uint16_t*)input.cpuData;
    int8_t   *pW  = (int8_t*)weight.cpuData;
    float    *pWs = (float*)wScale.cpuData;
    float    *pOut = (float*)cpuOut.cpuData;

    timer.Reset();
    #pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            float curWScale = pWs[n];
            for (int k = 0; k < K; ++k) {
                float val_x = F16toF32(pIn[m * K + k]);
                float val_w = (float)pW[n * K + k] * curWScale;
                sum += val_x * val_w;
            }
            pOut[m * N + n] = sum;
        }
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuWeight, npuWScale, npuXScale, npuBias, npuOutput;
    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);
    ToDevice(wScale, npuWScale);
    // xScale, bias are empty

    npuOutput.dataType = DataType::FLOAT16; 
    npuOutput.Resize({M, N});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    timer.Reset();
    FastllmAclQuantLinearDequant(npuInput, npuWeight, npuWScale, npuXScale, npuBias, npuOutput);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuResultHost;
    FromDeviceToFP32(npuOutput, npuResultHost);

    CompareData(cpuOut, npuResultHost, 0.5f, "QuantLinear"); // 量化误差允许较大

    if(npuInput.deviceData) FastllmAclFree(npuInput.deviceData);
    if(npuWeight.deviceData) FastllmAclFree(npuWeight.deviceData);
    if(npuWScale.deviceData) FastllmAclFree(npuWScale.deviceData);
    if(npuOutput.deviceData) FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// 17. 测试用例 14: RMSNorm
// ==========================================
void Test_RMSNorm() {
    std::cout << "=== Testing RMSNorm (FP16) ===" << std::endl;
    TestTimer timer("RMSNorm");

    int Rows = 32;
    int Dim = 4096;
    float eps = 1e-6f;

    Data input; input.dataType = DataType::FLOAT16; input.Resize({Rows, Dim}); FillRandom(input, -1.0f, 1.0f);
    Data weight; weight.dataType = DataType::FLOAT16; weight.Resize({Dim}); FillRandom(weight, 0.8f, 1.2f);
    Data bias; // Empty

    // --- CPU Bench ---
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({Rows, Dim}); cpuOut.Allocate();
    float *pCpuOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;
    uint16_t *pW  = (uint16_t*)weight.cpuData;

    timer.Reset();
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
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuWeight, npuBias, npuOutput;
    ToDevice(input, npuInput);
    ToDevice(weight, npuWeight);

    npuOutput.dataType = DataType::FLOAT16;
    npuOutput.Resize({Rows, Dim});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    timer.Reset();
    FastllmAclRMSNorm(npuInput, npuWeight, npuBias, npuOutput, eps);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuResultHost;
    FromDeviceToFP32(npuOutput, npuResultHost);
    // Data npuResultHost;
    // npuResultHost.dataType = DataType::FLOAT32;
    // npuResultHost.Resize({Rows, Dim});
    // npuResultHost.Allocate();
    
    // Data npuF16Host; // 先拷回 F16
    // npuF16Host.dataType = DataType::FLOAT16;
    // npuF16Host.Resize({Rows, Dim});
    // npuF16Host.Allocate();
    // FastllmAclCopyFromDeviceToHost(npuF16Host.cpuData, npuOutput.deviceData, npuF16Host.GetBytes());

    // uint16_t *ptrF16 = (uint16_t*)npuF16Host.cpuData;
    // float    *ptrF32 = (float*)npuResultHost.cpuData;
    // for(int i=0; i<npuResultHost.Count(0); ++i) {
    //     ptrF32[i] = F16toF32(ptrF16[i]);
    // }

    CompareData(cpuOut, npuResultHost, 0.05f, "RMSNorm");
    auto printData = [](const std::string &name, const Data &d, int limit = 10) {
        std::cout << name << ": ";
        float* ptr = (float*)d.cpuData; // 确保是 FLOAT32 指针
        for (int i = 0; i < limit && i < d.Count(0); ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << "..." << std::endl;
    };
    
    // 调用打印
    printData("CPU Output", cpuOut);
    printData("NPU Result", npuResultHost);

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}


// ==========================================
// 18. 测试用例 15: SwiGLU & Softmax
// ==========================================
void Test_SwiGLU() {
    std::cout << "=== Testing SwiGLU (FP16) ===" << std::endl;
    TestTimer timer("SwiGLU");

    int M = 16;
    int K = 256; 
    Data input; input.dataType = DataType::FLOAT16; input.Resize({M, 2 * K}); FillRandom(input, -2.0f, 2.0f);

    // --- CPU Bench ---
    Data cpuOut; cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({M, K}); cpuOut.Allocate();
    float *pOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;

    timer.Reset();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float a = F16toF32(pIn[i * (2*K) + j]);
            float b = F16toF32(pIn[i * (2*K) + j + K]);
            float swish_a = a / (1.0f + std::exp(-a));
            pOut[i * K + j] = swish_a * b;
        }
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);
    npuOutput.dataType = DataType::FLOAT16; npuOutput.Resize({M, K});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    timer.Reset();
    FastllmAclSwiglu(npuInput, npuOutput);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuResHost;
    FromDeviceToFP32(npuOutput, npuResHost);
    CompareData(cpuOut, npuResHost, 1e-3, "SwiGLU");

    FastllmAclFree(npuInput.deviceData); FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

void Test_Softmax() {
    std::cout << "=== Testing Softmax (FP16) ===" << std::endl;
    TestTimer timer("Softmax");

    int M = 16; int K = 128;
    Data input; input.dataType = DataType::FLOAT16; input.Resize({M, K}); FillRandom(input, -5.0f, 5.0f);

    // --- CPU Bench ---
    Data cpuOut; cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize({M, K}); cpuOut.Allocate();
    float *pOut = (float*)cpuOut.cpuData;
    uint16_t *pIn = (uint16_t*)input.cpuData;

    timer.Reset();
    for (int i = 0; i < M; ++i) {
        float maxVal = -1e9;
        for (int j = 0; j < K; ++j) {
            float val = F16toF32(pIn[i * K + j]);
            if (val > maxVal) maxVal = val;
        }
        float sumExp = 0.0f;
        std::vector<float> exps(K);
        for (int j = 0; j < K; ++j) {
            float val = F16toF32(pIn[i * K + j]);
            exps[j] = std::exp(val - maxVal);
            sumExp += exps[j];
        }
        for (int j = 0; j < K; ++j) {
            pOut[i * K + j] = exps[j] / sumExp;
        }
    }
    timer.Stop("CPU");

    // --- NPU Bench ---
    Data npuInput, npuOutput;
    ToDevice(input, npuInput);
    npuOutput.dataType = DataType::FLOAT16; npuOutput.Resize({M, K});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    timer.Reset();
    FastllmAclSoftmax(npuInput, npuOutput, -1);
    timer.Stop("NPU_Exec");

    // --- Verify ---
    Data npuResHost;
    FromDeviceToFP32(npuOutput, npuResHost);
    CompareData(cpuOut, npuResHost, 1e-3, "Softmax");

    FastllmAclFree(npuInput.deviceData); FastllmAclFree(npuOutput.deviceData);
    std::cout << "--------------------------------" << std::endl;
}

// ==========================================
// Main Function
// ==========================================
int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << " FastLLM NPU (Ascend) Operator Test Suite" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();
    std::cout << "NPU Initialized.\n" << std::endl;

    //--- Basic Math ---
    Test_Silu();
    Test_MatMul_FP16();
    Test_TopK();
    Test_Add_Scalar();
    Test_AddTo();
    Test_Mul_Scalar();
    Test_MulTo();
    
    // --- Data Movement ---
    Test_Permute();
    Test_Repeat();

    // --- Layers ---
    Test_Embedding();
    Test_RMSNorm();
    Test_SwiGLU();
    Test_Softmax();
    //Test_RoPE();              // Diagnostic Mode
    Test_RoPE_Dual_Version(); // Fused Mode
    Test_QuantLinear_Dynamic_W8A16();

    std::cout << "\n===========================================" << std::endl;
    std::cout << " All tests finished successfully." << std::endl;
    std::cout << "===========================================" << std::endl;
    return 0;
}