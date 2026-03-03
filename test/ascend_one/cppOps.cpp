#include "fastllm.h"
#include "fastllm-ascend.h"
#include <iostream>
#include <vector>
#include <cmath>

void callEmbeddingOp() {
    std::cout << "\n=== Testing EmbeddingOp ===" << std::endl;
    // 词表大小 5，维度 4
    std::vector<float> weightData = {
        0.0f, 0.0f, 0.0f, 0.0f, // Token 0
        1.0f, 1.0f, 1.0f, 1.0f, // Token 1
        2.0f, 2.0f, 2.0f, 2.0f, // Token 2
        3.0f, 3.0f, 3.0f, 3.0f, // Token 3
        4.0f, 4.0f, 4.0f, 4.0f  // Token 4
    };
    // 注意：你的 CanRun 强制要求 input 是 FLOAT32
    std::vector<float> inputData = {2.0f, 4.0f}; 

    fastllm::Data inputs(fastllm::DataType::FLOAT32, {1, 2}, inputData);
    fastllm::Data weights(fastllm::DataType::FLOAT16, {5, 4}, weightData);
    fastllm::Data outputs;

    inputs.ToDevice(fastllm::DataDevice::ASCEND);
    weights.ToDevice(fastllm::DataDevice::ASCEND);

    // 预期输出: 抽取 Token 2 和 4 的行，应为 [2.0, 2.0...], [4.0, 4.0...]
    fastllm::Embedding(inputs, weights, outputs);

    outputs.ToDevice(fastllm::DataDevice::CPU);
    outputs.Print();
}

void callTopKOp() {
    std::cout << "\n=== Testing TopKOp ===" << std::endl;
    // 模拟 5 个 Token 的概率分布 (Logits)
    std::vector<float> inputData = {0.1f, 0.9f, 0.3f, 0.8f, 0.2f};
    fastllm::Data inputs(fastllm::DataType::FLOAT16, {1, 5}, inputData);
    fastllm::Data outputs;

    inputs.ToDevice(fastllm::DataDevice::ASCEND);

    // 取 Top-2
    // 预期输出: 值 [0.9, 0.8], 索引 [1.0, 3.0] 拼接在一起
    fastllm::TopK(inputs, outputs, 2);

    outputs.ToDevice(fastllm::DataDevice::CPU);
    outputs.Print();
}

// void callRepeatOp() {
//     std::cout << "\n=== Testing RepeatOp ===" << std::endl;
//     // 原始张量 [1, 2, 3]
//     std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//     fastllm::Data inputs(fastllm::DataType::FLOAT16, {1, 2, 3}, inputData);
//     fastllm::Data outputs;

//     inputs.ToDevice(fastllm::DataDevice::ASCEND);

//     // 在轴 1 上重复 2 次。预期形状: [1, 4, 3]
//     // 预期数据: 行0, 行0, 行1, 行1 交替
//     fastllm::Repeat(inputs, 1, 2, outputs);

//     outputs.ToDevice(fastllm::DataDevice::CPU);
//     outputs.Print();
// }

// void callQuantLinearOp() {
//     std::cout << "\n=== Testing QuantLinearDequantOp (W8A16) ===" << std::endl;
    
//     // 1. 输入 X: 霸气回归 FLOAT16
//     int M = 1;
//     int K = 32; 
//     int N = 16; 

//     // 2. 构造 FLOAT16 的输入 X
//     std::vector<float> inputData(M * K, 1.0f); 
//     fastllm::Data inputs(fastllm::DataType::FLOAT16, {M, K}, inputData); 
    
//     // 3. 构造 INT8 的权重 W (N=16, K=32)
//     std::vector<float> weightData(N * K, 2.0f); 
//     fastllm::Data weights(fastllm::DataType::INT8, {N, K}, weightData); 
    
//     // 4. 构造 FLOAT32 的 WeightScale
//     std::vector<float> wScaleData(N, 0.5f);
//     fastllm::Data weightScale(fastllm::DataType::FLOAT32, {N}, wScaleData);
    
//     // 4. 输入缩放 xScale: 【设为 FLOAT32】，pertoken 要求 shape 为 (m,) 也就是 (1,)
//     std::vector<float> xScaleData = {1.0f};
//     fastllm::Data inputScale(fastllm::DataType::FLOAT32, {1}, xScaleData);
    
//     fastllm::Data bias; // 传空
//     fastllm::Data outputs;

//     inputs.ToDevice(fastllm::DataDevice::ASCEND);
//     weights.ToDevice(fastllm::DataDevice::ASCEND);
//     weightScale.ToDevice(fastllm::DataDevice::ASCEND);
//     inputScale.ToDevice(fastllm::DataDevice::ASCEND);

//     // 5. 输出: FLOAT16
//     outputs.dataType = fastllm::DataType::FLOAT16;
//     outputs.Resize({1, 3});
//     outputs.dataDevice = fastllm::DataDevice::ASCEND;
//     outputs.Allocate(); 

//     // 呼叫
//     fastllm::QuantLinearDequant(inputs, weights, weightScale, inputScale, bias, outputs);

//     outputs.ToDevice(fastllm::DataDevice::CPU);
//     outputs.Print();
// }

// void callQuantLinearOp() {
//     std::cout << "\n=== Testing QuantLinearDequantOp (Bulletproof Memory) ===" << std::endl;
    
//     int M = 1;
//     int K = 32; 
//     int N = 16; 

//     // 1. 纯手工构造 FLOAT16 输入 X (绕过隐式转换)
//     fastllm::Data inputs(fastllm::DataType::FLOAT16, {M, K});
//     inputs.Allocate();
//     uint16_t* in_ptr = (uint16_t*)inputs.cpuData;
//     // 0x3c00 是 IEEE 754 标准下 FLOAT16 的 1.0f 的十六进制真值
//     for (int i = 0; i < M * K; i++) in_ptr[i] = 0x3c00; 

//     // 2. 纯手工构造 INT8 权重 W (直接操作单字节内存)
//     fastllm::Data weights(fastllm::DataType::INT8, {N, K});
//     weights.Allocate();
//     int8_t* w_ptr = (int8_t*)weights.cpuData;
//     // 写入真实的 INT8 整数 2
//     for (int i = 0; i < N * K; i++) w_ptr[i] = 2; 
    
//     // 3. 构造 FLOAT32 的 WeightScale
//     std::vector<float> wScaleData(N, 0.5f);
//     fastllm::Data weightScale(fastllm::DataType::FLOAT32, {N}, wScaleData);
    
//     // 4. 手动提供 xScale (规避动态量化可能返回 0 的硬件差异)
//     std::vector<float> xScaleData(M, 1.0f);
//     fastllm::Data inputScale(fastllm::DataType::FLOAT32, {M}, xScaleData);
    
//     fastllm::Data bias; 
//     fastllm::Data outputs;

//     // 推入 NPU
//     inputs.ToDevice(fastllm::DataDevice::ASCEND);
//     weights.ToDevice(fastllm::DataDevice::ASCEND);
//     weightScale.ToDevice(fastllm::DataDevice::ASCEND);
//     inputScale.ToDevice(fastllm::DataDevice::ASCEND);

//     // 分配输出坑位
//     outputs.dataType = fastllm::DataType::FLOAT16;
//     outputs.Resize({M, N});
//     outputs.dataDevice = fastllm::DataDevice::ASCEND;
//     outputs.Allocate(); 

//     // 呼叫底层
//     fastllm::QuantLinearDequant(inputs, weights, weightScale, inputScale, bias, outputs);

//     // 拉回主板并打印
//     outputs.ToDevice(fastllm::DataDevice::CPU);
//     outputs.Print();
// }

void callRoPEOp() {
    std::cout << "\n=== Testing NearlyRotatePosition2DOp (RoPE Absolute Compliance) ===" << std::endl;
    
    // CANN 强制规定：最后一维必须是 128！
    int B = 1;
    int S = 2; // Seq_len
    int N = 2; // Head_num
    int D = 128; // Head_dim (雷打不动的 128)
    
    // Query 填充全 1
    std::vector<float> qData(B * S * N * D, 1.0f); 
    fastllm::Data query(fastllm::DataType::FLOAT16, {B, S, N, D}, qData); 
    
    // Cos/Sin 的形状必须是 [B, S, 1, D]
    std::vector<float> cosData(B * S * 1 * D, 0.0f); // Cos 全 0
    std::vector<float> sinData(B * S * 1 * D, 1.0f); // Sin 全 1
    
    fastllm::Data cos(fastllm::DataType::FLOAT16, {B, S, 1, D}, cosData);
    fastllm::Data sin(fastllm::DataType::FLOAT16, {B, S, 1, D}, sinData);
    
    fastllm::Data positionIds; // 占位空张量
    
    query.ToDevice(fastllm::DataDevice::ASCEND);
    cos.ToDevice(fastllm::DataDevice::ASCEND);
    sin.ToDevice(fastllm::DataDevice::ASCEND);
    
    // 呼叫
    fastllm::NearlyRotatePosition2D(query, positionIds, sin, cos, 128);
    
    query.ToDevice(fastllm::DataDevice::CPU);
    
    // 验证部分数据即可，不用全打出来
    std::cout << "shape: ";
    for (int i = 0; i < query.dims.size(); i++) std::cout << query.dims[i] << " ";
    std::cout << "\nfirst 8 elements: \n";
    uint16_t* out_ptr = (uint16_t*)query.cpuData;
    // 打印前 8 个 FP16 的近似十进制值观察变化
    for(int i=0; i<8; i++) {
        std::cout << (out_ptr[i] == 0xbc00 ? "-1.0 " : (out_ptr[i] == 0x3c00 ? "1.0 " : "other ")) ;
    }
    std::cout << std::endl;
}

void callRoPEFusedOp() {
    std::cout << "\n=== Testing NearlyRotatePosition2DFusedOp (RoPE Fused) ===" << std::endl;
    
    int B = 1, S = 2, N = 2, D = 128;
    
    // Query 全 1.0，Key 全 2.0
    std::vector<float> qData(B * S * N * D, 1.0f); 
    std::vector<float> kData(B * S * N * D, 2.0f); 
    
    fastllm::Data query(fastllm::DataType::FLOAT16, {B, S, N, D}, qData); 
    fastllm::Data key(fastllm::DataType::FLOAT16, {B, S, N, D}, kData); 
    
    // Cos 全 0，Sin 全 1
    std::vector<float> cosData(B * S * 1 * D, 0.0f); 
    std::vector<float> sinData(B * S * 1 * D, 1.0f); 
    
    fastllm::Data cos(fastllm::DataType::FLOAT16, {B, S, 1, D}, cosData);
    fastllm::Data sin(fastllm::DataType::FLOAT16, {B, S, 1, D}, sinData);
    
    fastllm::Data positionIds; 
    
    query.ToDevice(fastllm::DataDevice::ASCEND);
    key.ToDevice(fastllm::DataDevice::ASCEND);
    cos.ToDevice(fastllm::DataDevice::ASCEND);
    sin.ToDevice(fastllm::DataDevice::ASCEND);
    
    // 直接呼叫我们刚写好的底层函数
    fastllm::FastllmAclRotatePosition2D_Fused(query, key, positionIds, sin, cos, 128);
    
    query.ToDevice(fastllm::DataDevice::CPU);
    key.ToDevice(fastllm::DataDevice::CPU);
    
    uint16_t* q_ptr = (uint16_t*)query.cpuData;
    uint16_t* k_ptr = (uint16_t*)key.cpuData;
    
    // 验证逻辑：
    // Query (1.0) 应该变成 -1.0 (FP16 十六进制为 0xbc00)
    // Key (2.0) 应该变成 -2.0 (FP16 十六进制为 0xc000)
    std::cout << "Query first 8 elements (expected -1.0): \n";
    for(int i=0; i<8; i++) std::cout << (q_ptr[i] == 0xbc00 ? "-1.0 " : "other ") ;
    
    std::cout << "\nKey first 8 elements (expected -2.0): \n";
    for(int i=0; i<8; i++) std::cout << (k_ptr[i] == 0xc000 ? "-2.0 " : "other ") ;
    
    std::cout << std::endl;
}

void testAdvancedOps() {
    //callEmbeddingOp();
    //callTopKOp();
    //callRepeatOp();
    // 下面两个算子对硬件要求较高，如果报错可以先注释掉单独调试
    callRoPEOp();
    callRoPEFusedOp();
    //callQuantLinearOp(); 
    std::cout << "\n=== All Advanced Ops Tested! ===" << std::endl;
}

int main(){
    fastllm::FastllmAclInit();
    testAdvancedOps();
}