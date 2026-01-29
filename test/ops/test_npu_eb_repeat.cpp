#include "fastllm.h"
#include "devices/npu/fastllm-ascend.h"
#include "devices/npu/ascenddevice.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <cstring> // for memcpy

using namespace fastllm;

// ==========================================
// 辅助工具：FP16 转换
// ==========================================
uint16_t F32toF16(float value) {
    __fp16 val_f16 = (__fp16)value;
    return *(uint16_t*)&val_f16;
}

float F16toF32(uint16_t value) {
    __fp16 val_f16 = *(__fp16*)&value;
    return (float)val_f16;
}

// ==========================================
// 辅助工具：数据填充
// ==========================================
void FillRandomF16(Data &data, float min, float max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    uint16_t *ptr = (uint16_t*)data.cpuData;
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = F32toF16(dis(gen));
}

// 填充整数索引（用于 Embedding）
void FillRandomInt(Data &data, int min, int max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min, max);
    if (data.cpuData == nullptr) data.Allocate();
    float *ptr = (float*)data.cpuData; // fastllm 索引通常用 float 存储
    for (int i = 0; i < data.Count(0); ++i) ptr[i] = (float)dis(gen);
}

// ==========================================
// 辅助工具：对比结果
// ==========================================
bool CompareData(const Data &cpuOut, const Data &npuOut, float threshold = 1e-3) {
    float maxErr = 0.0f;
    float *pCpu = (float*)cpuOut.cpuData;
    float *pNpu = (float*)npuOut.cpuData;
    int len = cpuOut.Count(0);
    int errIdx = -1;

    for (int i = 0; i < len; ++i) {
        if (std::isnan(pCpu[i]) || std::isinf(pCpu[i])) continue;
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
// 1. Embedding 测试用例
// ==========================================
void Test_Embedding() {
    std::cout << "=== Testing Embedding ===" << std::endl;

    int Batch = 2;
    int SeqLen = 8;
    int VocabSize = 1000;
    int EmbedDim = 256;

    // 1. 准备数据
    Data input; // Indices
    input.dataType = DataType::FLOAT32; // Fastllm 常用 float 存索引
    input.Resize({Batch, SeqLen});
    FillRandomInt(input, 0, VocabSize - 1);

    Data weight; // Embedding Table
    weight.dataType = DataType::FLOAT16;
    weight.Resize({VocabSize, EmbedDim});
    FillRandomF16(weight, -1.0f, 1.0f);

    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT32;
    cpuOut.Resize({Batch, SeqLen, EmbedDim});
    cpuOut.Allocate();

    // 2. CPU Ground Truth
    float *pCpuOut = (float*)cpuOut.cpuData;
    float *pIn = (float*)input.cpuData;
    uint16_t *pW = (uint16_t*)weight.cpuData;

    for (int i = 0; i < Batch * SeqLen; ++i) {
        int idx = (int)pIn[i];
        if (idx < 0 || idx >= VocabSize) idx = 0; // 越界保护
        
        // 查表并转 FP32
        for (int d = 0; d < EmbedDim; ++d) {
            uint16_t val16 = pW[idx * EmbedDim + d];
            pCpuOut[i * EmbedDim + d] = F16toF32(val16);
        }
    }

    // 3. NPU Run
    Data npuInput, npuWeight, npuOutput;
    
    // 拷贝 Input 到 Device
    npuInput.dataType = input.dataType; npuInput.Resize(input.dims);
    npuInput.deviceData = FastllmAclMalloc(npuInput.GetBytes());
    FastllmAclCopyFromHostToDevice(npuInput.deviceData, input.cpuData, npuInput.GetBytes());

    // 拷贝 Weight 到 Device
    npuWeight.dataType = weight.dataType; npuWeight.Resize(weight.dims);
    npuWeight.deviceData = FastllmAclMalloc(npuWeight.GetBytes());
    FastllmAclCopyFromHostToDevice(npuWeight.deviceData, weight.cpuData, npuWeight.GetBytes());

    // 准备 Output
    npuOutput.dataType = DataType::FLOAT16; // 输出通常跟随 Weight 类型
    npuOutput.Resize({Batch, SeqLen, EmbedDim});
    npuOutput.deviceData = FastllmAclMalloc(npuOutput.GetBytes());

    // 调用待测函数
    FastllmAclEmbedding(npuInput, npuWeight, npuOutput);

    // 4. 结果拷回与转换
    Data npuResultHost; // 最终 FP32 结果
    npuResultHost.dataType = DataType::FLOAT32;
    npuResultHost.Resize({Batch, SeqLen, EmbedDim});
    npuResultHost.Allocate();

    Data npuF16Host; // 中间 FP16 Buffer
    npuF16Host.dataType = DataType::FLOAT16;
    npuF16Host.Resize(npuOutput.dims);
    npuF16Host.Allocate();

    FastllmAclCopyFromDeviceToHost(npuF16Host.cpuData, npuOutput.deviceData, npuF16Host.GetBytes());

    uint16_t *ptrF16 = (uint16_t*)npuF16Host.cpuData;
    float *ptrF32 = (float*)npuResultHost.cpuData;
    for(int i=0; i < npuResultHost.Count(0); ++i) {
        ptrF32[i] = F16toF32(ptrF16[i]);
    }

    // 5. 对比
    CompareData(cpuOut, npuResultHost);

    // 清理
    FastllmAclFree(npuInput.deviceData);
    FastllmAclFree(npuWeight.deviceData);
    FastllmAclFree(npuOutput.deviceData);
}

// ==========================================
// 2. Repeat (Expand) 测试用例
// ==========================================
void Test_Repeat() {
    std::cout << "=== Testing Repeat (Bytes Expand) ===" << std::endl;

    // 模拟场景：GQA 中将 Key/Value 进行广播
    // 假设 Outer (Batch*Heads) = 4
    // 假设 Inner (HeadDim) = 128 float16 elements (即 256 bytes)
    // Repeat Times = 3
    
    int outer = 4;
    int repeatTimes = 3;
    int elementSize = 2; // FP16
    int headDim = 128;   // 元素个数
    int bytesPerBlock = headDim * elementSize; // 256 bytes

    // 待测函数的参数要求是 stride (字节还是元素取决于 CreateTensor 里的 Type)
    // 你的实现里用了 ACL_UINT8，所以这里的 channelsInputInner 代表 字节数
    int channelsInputInner = bytesPerBlock; 
    int channelsInner = bytesPerBlock; // 假设紧密排列

    int inputStride = channelsInputInner; // [outer, 1, inner] -> stride[0] = inner
    int outputStride = channelsInner * repeatTimes; // [outer, repeat, inner] -> stride[0] = repeat * inner

    // 1. 准备数据 (在 CPU 上用 FP16 模拟，但视作字节流)
    Data input;
    input.dataType = DataType::FLOAT16;
    input.Resize({outer, 1, headDim}); // 逻辑形状
    FillRandomF16(input, -10.0f, 10.0f);

    // CPU Ground Truth 计算
    Data cpuOut;
    cpuOut.dataType = DataType::FLOAT16;
    cpuOut.Resize({outer, repeatTimes, headDim});
    cpuOut.Allocate();

    uint8_t *pSrcBytes = (uint8_t*)input.cpuData;
    uint8_t *pDstBytes = (uint8_t*)cpuOut.cpuData;

    // CPU 模拟 Expand 逻辑 (字节级复制)
    for (int o = 0; o < outer; ++o) {
        uint8_t *srcRow = pSrcBytes + o * inputStride;
        uint8_t *dstRow = pDstBytes + o * outputStride;
        
        for (int r = 0; r < repeatTimes; ++r) {
            // 将 src 的 inner 块复制到 dst 的每一个 repeat 槽位
            memcpy(dstRow + r * channelsInner, srcRow, channelsInputInner);
        }
    }

    // 2. NPU Run
    void *deviceSrc = FastllmAclMalloc(input.GetBytes());
    void *deviceDst = FastllmAclMalloc(cpuOut.GetBytes());

    FastllmAclCopyFromHostToDevice(deviceSrc, input.cpuData, input.GetBytes());
    // Dst 不需要初始化，会被覆盖

    // 调用待测函数
    // 注意：函数参数 src, dst 是 void*，且内部使用 ACL_UINT8 处理
    FastllmAclRepeat(deviceSrc, deviceDst, outer, repeatTimes, 
                     inputStride, outputStride, 
                     channelsInner, channelsInputInner);

    // 3. 结果验证
    Data npuOutHost;
    npuOutHost.dataType = DataType::FLOAT16;
    npuOutHost.Resize(cpuOut.dims);
    npuOutHost.Allocate();

    FastllmAclCopyFromDeviceToHost(npuOutHost.cpuData, deviceDst, npuOutHost.GetBytes());

    // 转换为 Float 进行对比 (复用 CompareData 逻辑)
    // 需要把 FP16 bytes 转回 Float32
    Data cpuOutF32, npuOutF32;
    cpuOutF32.dataType = DataType::FLOAT32; cpuOutF32.Resize(cpuOut.dims); cpuOutF32.Allocate();
    npuOutF32.dataType = DataType::FLOAT32; npuOutF32.Resize(npuOutHost.dims); npuOutF32.Allocate();

    float *pRef = (float*)cpuOutF32.cpuData;
    float *pRes = (float*)npuOutF32.cpuData;
    uint16_t *pRef16 = (uint16_t*)cpuOut.cpuData;
    uint16_t *pRes16 = (uint16_t*)npuOutHost.cpuData;

    for (int i = 0; i < cpuOut.Count(0); ++i) {
        pRef[i] = F16toF32(pRef16[i]);
        pRes[i] = F16toF32(pRes16[i]);
    }

    CompareData(cpuOutF32, npuOutF32, 0.0f); // 纯内存复制，理论上误差应为 0

    // 清理
    FastllmAclFree(deviceSrc);
    FastllmAclFree(deviceDst);
}

int main() {
    std::cout << "Initializing NPU..." << std::endl;
    // 你的初始化函数
    FastllmAclInit(); // 假设已经有初始化逻辑
    
    Test_Embedding();
    std::cout << "--------------------------------" << std::endl;
    Test_Repeat();
    
    std::cout << "All tests finished." << std::endl;
    return 0;
}