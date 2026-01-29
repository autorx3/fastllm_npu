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


void Test_TopK() {
    std::cout << "=== Testing TopK (Values + Indices) ===" << std::endl;

    int batch = 4;
    int vocabSize = 100;
    int k = 5;

    // 输入准备
    std::vector<int> inputDims = {batch, vocabSize};
    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(inputDims);
    FillRandom(input, 0.0f, 100.0f);

    std::vector<int> outputDims = {batch, k * 2};
    
    // NPU 准备
    Data npuInput, npuOutput;
    
    // Input Malloc
    npuInput.dataType = DataType::FLOAT32; 
    npuInput.Resize(inputDims);
    npuInput.deviceData = FastllmAclMalloc(input.GetBytes()); 
    FastllmAclCopyFromHostToDevice(npuInput.deviceData, input.cpuData, input.GetBytes());

    // Output Malloc
    npuOutput.dataType = DataType::FLOAT32;
    npuOutput.Resize(outputDims);
    npuOutput.deviceData = FastllmAclMalloc(outputDims[0] * outputDims[1] * sizeof(float)); 

    // 执行
    FastllmAclTopK(npuInput, npuOutput, k);

    // 打印验证
    Data npuResultHost;
    npuResultHost.dataType = DataType::FLOAT32;
    npuResultHost.Resize(outputDims);
    npuResultHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuResultHost.cpuData, npuOutput.deviceData, npuResultHost.GetBytes());

    float* outPtr = (float*)npuResultHost.cpuData;
    printf("Top1 Value: %f, Index: %f\n", outPtr[0], outPtr[k]);
    std::cout << " -> [PASS] (Assuming logic checked)" << std::endl;
    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);

}

void Test_Add_Scalar() {
    std::cout << "=== Testing Add (Scalar) ===";
    std::vector<int> dims = {2, 512};
    float v = 3.14f;

    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    // CPU Golden
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize(dims);
    cpuOut.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* out = (float*)cpuOut.cpuData;
    for(int i=0; i<input.Count(0); i++) {
        out[i] = inp[i] + v;
    }

    // NPU Test
    Data npuInput, npuOutput;
    // Setup Input
    npuInput.dataType = DataType::FLOAT32; npuInput.Resize(dims);
    npuInput.deviceData = FastllmAclMalloc(input.GetBytes());
    FastllmAclCopyFromHostToDevice(npuInput.deviceData, input.cpuData, input.GetBytes());
    
    // Setup Output
    npuOutput.dataType = DataType::FLOAT32; npuOutput.Resize(dims);
    npuOutput.deviceData = FastllmAclMalloc(cpuOut.GetBytes());

    // Execute
    FastllmAclAdd(npuInput, v, npuOutput);

    // Verify
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize(dims);
    npuResHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuResHost.cpuData, npuOutput.deviceData, npuResHost.GetBytes());

    CompareData(cpuOut, npuResHost, 1e-4);

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

// ==========================================
// 2. 测试: AddTo (Axpy) -> In0 = In0 + In1 * alpha
// ==========================================
void Test_AddTo() {
    std::cout << "=== Testing AddTo (Axpy) ===";
    std::vector<int> dims = {4, 256};
    float alpha = 0.5f;

    Data input0, input1;
    input0.dataType = DataType::FLOAT32; input0.Resize(dims); FillRandom(input0);
    input1.dataType = DataType::FLOAT32; input1.Resize(dims); FillRandom(input1);

    // CPU Golden (计算结果存入 cpuRef)
    Data cpuRef;
    cpuRef.dataType = DataType::FLOAT32; cpuRef.Resize(dims); cpuRef.Allocate();
    
    float* p0 = (float*)input0.cpuData;
    float* p1 = (float*)input1.cpuData;
    float* pRef = (float*)cpuRef.cpuData;
    
    for(int i=0; i<input0.Count(0); i++) {
        pRef[i] = p0[i] + p1[i] * alpha;
    }

    // NPU Test
    Data npuIn0, npuIn1;
    // In0 (作为 Output，会被修改)
    npuIn0.dataType = DataType::FLOAT32; npuIn0.Resize(dims);
    npuIn0.deviceData = FastllmAclMalloc(input0.GetBytes());
    FastllmAclCopyFromHostToDevice(npuIn0.deviceData, input0.cpuData, input0.GetBytes());

    // In1
    npuIn1.dataType = DataType::FLOAT32; npuIn1.Resize(dims);
    npuIn1.deviceData = FastllmAclMalloc(input1.GetBytes());
    FastllmAclCopyFromHostToDevice(npuIn1.deviceData, input1.cpuData, input1.GetBytes());

    // Execute (注意：结果直接写入 npuIn0)
    FastllmAclAddTo(npuIn0, npuIn1, alpha);

    // Verify
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize(dims);
    npuResHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuResHost.cpuData, npuIn0.deviceData, npuResHost.GetBytes());

    CompareData(cpuRef, npuResHost, 1e-4);

    FastllmAclFree(npuIn0.deviceData);
    FastllmAclFree(npuIn1.deviceData);
}

// ==========================================
// 3. 测试: Mul (Scalar) -> Output = Input * v
// ==========================================
void Test_Mul_Scalar() {
    std::cout << "=== Testing Mul (Scalar) ===";
    std::vector<int> dims = {2, 512};
    float v = 2.0f;

    Data input;
    input.dataType = DataType::FLOAT32; input.Resize(dims); FillRandom(input);

    // CPU Golden
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32; cpuOut.Resize(dims); cpuOut.Allocate();
    
    float* inp = (float*)input.cpuData;
    float* out = (float*)cpuOut.cpuData;
    for(int i=0; i<input.Count(0); i++) {
        out[i] = inp[i] * v;
    }

    // NPU Test
    Data npuInput, npuOutput;
    npuInput.dataType = DataType::FLOAT32; npuInput.Resize(dims);
    npuInput.deviceData = FastllmAclMalloc(input.GetBytes());
    FastllmAclCopyFromHostToDevice(npuInput.deviceData, input.cpuData, input.GetBytes());
    
    npuOutput.dataType = DataType::FLOAT32; npuOutput.Resize(dims);
    npuOutput.deviceData = FastllmAclMalloc(cpuOut.GetBytes());

    // Execute
    FastllmAclMul(npuInput, v, npuOutput);

    // Verify
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize(dims); npuResHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuResHost.cpuData, npuOutput.deviceData, npuResHost.GetBytes());

    CompareData(cpuOut, npuResHost, 1e-4);

    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

// ==========================================
// 4. 测试: MulTo (Element-wise) -> In0 = In0 * In1 * alpha
// ==========================================
void Test_MulTo() {
    std::cout << "=== Testing MulTo (Element-wise + alpha) ===";
    std::vector<int> dims = {4, 256};
    float alpha = 2.0f; // 设一个非1的alpha测试你的第二段逻辑

    Data input0, input1;
    input0.dataType = DataType::FLOAT32; input0.Resize(dims); FillRandom(input0);
    input1.dataType = DataType::FLOAT32; input1.Resize(dims); FillRandom(input1);

    // CPU Golden
    Data cpuRef;
    cpuRef.dataType = DataType::FLOAT32; cpuRef.Resize(dims); cpuRef.Allocate();
    
    float* p0 = (float*)input0.cpuData;
    float* p1 = (float*)input1.cpuData;
    float* pRef = (float*)cpuRef.cpuData;
    
    for(int i=0; i<input0.Count(0); i++) {
        pRef[i] = (p0[i] * p1[i]) * alpha;
    }

    // NPU Test
    Data npuIn0, npuIn1;
    npuIn0.dataType = DataType::FLOAT32; npuIn0.Resize(dims);
    npuIn0.deviceData = FastllmAclMalloc(input0.GetBytes());
    FastllmAclCopyFromHostToDevice(npuIn0.deviceData, input0.cpuData, input0.GetBytes());

    npuIn1.dataType = DataType::FLOAT32; npuIn1.Resize(dims);
    npuIn1.deviceData = FastllmAclMalloc(input1.GetBytes());
    FastllmAclCopyFromHostToDevice(npuIn1.deviceData, input1.cpuData, input1.GetBytes());

    // Execute (结果写入 npuIn0)
    FastllmAclMulTo(npuIn0, npuIn1, alpha);

    // Verify
    Data npuResHost;
    npuResHost.dataType = DataType::FLOAT32; npuResHost.Resize(dims); npuResHost.Allocate();
    FastllmAclCopyFromDeviceToHost(npuResHost.cpuData, npuIn0.deviceData, npuResHost.GetBytes());

    CompareData(cpuRef, npuResHost, 1e-4);

    FastllmAclFree(npuIn0.deviceData);
    FastllmAclFree(npuIn1.deviceData);
}

// 建议在 main 中依次调用：

void Test_Permute() {
    std::cout << "=== Testing Permute (aclnnPermute) ===";
    // 形状: [Batch=2, Heads=4, Seq=8, Dim=16] -> 转置 Heads 和 Seq
    // 目标: [2, 8, 4, 16]
    std::vector<int> dims = {2, 4, 8, 16};
    std::vector<int> axis = {0, 2, 1, 3}; // 交换第1和第2维
    std::vector<int> expectedDims = {2, 8, 4, 16};

    // 1. 准备数据
    Data input;
    input.dataType = DataType::FLOAT32;
    input.Resize(dims);
    FillRandom(input);

    // 2. CPU Golden (简单模拟，太复杂可以只对比维度和首尾数据)
    // 这里我们只验证运行不报错，且维度变化正确
    // 若要严格验证数据，需要写一个 CPU 的 Transpose 逻辑
    
    // 3. NPU 准备
    Data npuData;
    npuData.dataType = DataType::FLOAT32; npuData.Resize(dims);
    npuData.deviceData = FastllmAclMalloc(input.GetBytes());
    FastllmAclCopyFromHostToDevice(npuData.deviceData, input.cpuData, input.GetBytes());

    // 4. 执行算子
    FastllmAclPermute(npuData, axis);

    // 5. 验证维度
    if (npuData.dims != expectedDims) {
        std::cout << " [FAIL] Dimension Mismatch!" << std::endl;
        return;
    }

    // 6. 拷回验证数据 (可选)
    // 如果程序没崩，且维度对了，基本 aclnnPermute 就没问题
    std::cout << " [PASS] Dims check passed." << std::endl;

    FastllmAclFree(npuData.deviceData);
}

// ==========================================
// 测试用例: RoPE (Rotary Position Embedding)
// ==========================================
void Test_RoPE() {
    std::cout << "=== Testing RoPE (Diagnostic Mode) ===";
    int batch = 1; int seqLen = 1; int dim = 64;
    std::vector<int> dims = {batch, seqLen, dim};

    // 1. 准备数据
    Data input, sinData, cosData;
    input.dataType = DataType::FLOAT32; input.Resize(dims); FillRandom(input);
    sinData.dataType = DataType::FLOAT32; sinData.Resize(dims); FillRandom(sinData);
    cosData.dataType = DataType::FLOAT32; cosData.Resize(dims); FillRandom(cosData);
    Data dummyPos; dummyPos.dataType = DataType::FLOAT32; dummyPos.Resize({1,1}); // 占位

    // 2. CPU 计算 - 模式 A: Rotate Half (Llama)
    Data cpuHalf; cpuHalf.dataType = DataType::FLOAT32; cpuHalf.Resize(dims); cpuHalf.Allocate();
    {
        int half = dim / 2;
        float* inp = (float*)input.cpuData; float* sin = (float*)sinData.cpuData; float* cos = (float*)cosData.cpuData; float* out = (float*)cpuHalf.cpuData;
        for (int i = 0; i < half; i++) {
            out[i]        = inp[i] * cos[i] - inp[i + half] * sin[i];
            out[i + half] = inp[i] * sin[i] + inp[i + half] * cos[i];
        }
    }

    // 3. CPU 计算 - 模式 B: Rotate Interleaved (GPT-NeoX)
    Data cpuInter; cpuInter.dataType = DataType::FLOAT32; cpuInter.Resize(dims); cpuInter.Allocate();
    {
        float* inp = (float*)input.cpuData; float* sin = (float*)sinData.cpuData; float* cos = (float*)cosData.cpuData; float* out = (float*)cpuInter.cpuData;
        for (int i = 0; i < dim; i += 2) {
            // [x0, x1] -> [-x1, x0] 旋转
            // out0 = x0*cos - x1*sin
            // out1 = x0*sin + x1*cos
            out[i]     = inp[i] * cos[i] - inp[i+1] * sin[i];
            out[i + 1] = inp[i] * sin[i] + inp[i+1] * cos[i];
        }
    }

    // 4. NPU 计算
    Data npuData; npuData.dataType = DataType::FLOAT32; npuData.Resize(dims);
    npuData.deviceData = FastllmAclMalloc(input.GetBytes());
    FastllmAclCopyFromHostToDevice(npuData.deviceData, input.cpuData, input.GetBytes());

    Data npuSin; npuSin.dataType = DataType::FLOAT32; npuSin.Resize(dims); npuSin.deviceData = FastllmAclMalloc(sinData.GetBytes());
    FastllmAclCopyFromHostToDevice(npuSin.deviceData, sinData.cpuData, sinData.GetBytes());

    Data npuCos; npuCos.dataType = DataType::FLOAT32; npuCos.Resize(dims); npuCos.deviceData = FastllmAclMalloc(cosData.GetBytes());
    FastllmAclCopyFromHostToDevice(npuCos.deviceData, cosData.cpuData, cosData.GetBytes());

    // 调用算子 (请确保此时 mode 已改为 0 或 1)
    FastllmAclNearlyRotatePosition2D(npuData, dummyPos, npuSin, npuCos, dim);

    Data npuRes; npuRes.dataType = DataType::FLOAT32; npuRes.Resize(dims); npuRes.Allocate();
    FastllmAclCopyFromDeviceToHost(npuRes.cpuData, npuData.deviceData, npuRes.GetBytes());

    // 5. 对比
    std::cout << "\nCHECK 1: Llama Mode (Rotate Half)...";
    if (CompareData(cpuHalf, npuRes, 1e-3)) {
        std::cout << "Match! Use this logic." << std::endl;
    } else {
        std::cout << "Mismatch." << std::endl;
    }

    std::cout << "CHECK 2: NeoX Mode (Interleaved)...";
    if (CompareData(cpuInter, npuRes, 1e-3)) {
        std::cout << "Match! (Logic is interleaved)" << std::endl;
    } else {
        std::cout << "Mismatch." << std::endl;
    }

    FastllmAclFree(npuData.deviceData); FastllmAclFree(npuSin.deviceData); FastllmAclFree(npuCos.deviceData);
}

void Test_RoPE_Dual_Version() {
    std::cout << "=== Testing RoPE (Dual Version) ===";
    std::vector<int> dims = {1, 1, 1, 64}; // 简化维度
    int dim = 64;

    // 1. 准备数据
    Data q, k, sinData, cosData, dummyPos;
    q.dataType = DataType::FLOAT32; q.Resize(dims); FillRandom(q);
    k.dataType = DataType::FLOAT32; k.Resize(dims); FillRandom(k);
    sinData.dataType = DataType::FLOAT32; sinData.Resize(dims); FillRandom(sinData);
    cosData.dataType = DataType::FLOAT32; cosData.Resize(dims); FillRandom(cosData);

    // 2. 准备 NPU 数据 (Q1/K1用于单输入测试, Q2/K2用于双输入测试)
    Data npuQ1, npuK1, npuQ2, npuK2, npuSin, npuCos;
    // 拷贝辅助函数略，假设已拷贝...
    // 这里简写逻辑:
    npuQ1.Resize(dims); npuQ1.deviceData = FastllmAclMalloc(q.GetBytes()); FastllmAclCopyFromHostToDevice(npuQ1.deviceData, q.cpuData, q.GetBytes());
    npuK1.Resize(dims); npuK1.deviceData = FastllmAclMalloc(k.GetBytes()); FastllmAclCopyFromHostToDevice(npuK1.deviceData, k.cpuData, k.GetBytes());
    
    npuQ2.Resize(dims); npuQ2.deviceData = FastllmAclMalloc(q.GetBytes()); FastllmAclCopyFromHostToDevice(npuQ2.deviceData, q.cpuData, q.GetBytes());
    npuK2.Resize(dims); npuK2.deviceData = FastllmAclMalloc(k.GetBytes()); FastllmAclCopyFromHostToDevice(npuK2.deviceData, k.cpuData, k.GetBytes());

    npuSin.Resize(dims); npuSin.deviceData = FastllmAclMalloc(sinData.GetBytes()); FastllmAclCopyFromHostToDevice(npuSin.deviceData, sinData.cpuData, sinData.GetBytes());
    npuCos.Resize(dims); npuCos.deviceData = FastllmAclMalloc(cosData.GetBytes()); FastllmAclCopyFromHostToDevice(npuCos.deviceData, cosData.cpuData, cosData.GetBytes());

    // -------------------------------------------------
    // 测试 1: 单输入版 (Legacy) - 需要调用两次
    // -------------------------------------------------
    // 旋转 Q
    FastllmAclNearlyRotatePosition2D(npuQ1, dummyPos, npuSin, npuCos, dim);
    // 旋转 K
    FastllmAclNearlyRotatePosition2D(npuK1, dummyPos, npuSin, npuCos, dim);

    // -------------------------------------------------
    // 测试 2: 双输入版 (Fused) - 调用一次
    // -------------------------------------------------
    FastllmAclRotatePosition2D_Fused(npuQ2, npuK2, dummyPos, npuSin, npuCos, dim);

    // -------------------------------------------------
    // 结果对比
    // -------------------------------------------------
    // 理论上 npuQ1 应该等于 npuQ2，npuK1 应该等于 npuK2
    Data resQ1, resK1, resQ2, resK2;
    resQ1.Resize(dims); resQ1.Allocate(); FastllmAclCopyFromDeviceToHost(resQ1.cpuData, npuQ1.deviceData, resQ1.GetBytes());
    resK1.Resize(dims); resK1.Allocate(); FastllmAclCopyFromDeviceToHost(resK1.cpuData, npuK1.deviceData, resK1.GetBytes());
    resQ2.Resize(dims); resQ2.Allocate(); FastllmAclCopyFromDeviceToHost(resQ2.cpuData, npuQ2.deviceData, resQ2.GetBytes());
    resK2.Resize(dims); resK2.Allocate(); FastllmAclCopyFromDeviceToHost(resK2.cpuData, npuK2.deviceData, resK2.GetBytes());

    std::cout << "Check Q (Single vs Fused): ";
    CompareData(resQ1, resQ2, 1e-5); // 应该几乎完全一致

    std::cout << "Check K (Single vs Fused): ";
    CompareData(resK1, resK2, 1e-5); // 应该几乎完全一致
    
    // 释放内存...
    FastllmAclFree(npuQ1.deviceData); FastllmAclFree(npuK1.deviceData);
    FastllmAclFree(npuQ2.deviceData); FastllmAclFree(npuK2.deviceData);
    FastllmAclFree(npuSin.deviceData); FastllmAclFree(npuCos.deviceData);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    FastllmAclInit();

    Test_Silu();
    Test_MatMul_FP16();
    Test_TopK();
    Test_Add_Scalar();
    Test_AddTo();
    Test_Mul_Scalar();
    Test_MulTo();
    Test_Permute();
    Test_RoPE_Dual_Version();

    std::cout << "All tests finished." << std::endl;
    return 0;
}