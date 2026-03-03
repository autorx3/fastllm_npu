#include "fastllm.h"
//#ifdef USE_ASCEND
#include "fastllm-ascend.h"
//#endif
// void callBaseOp(int optype=0){
//     // fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5});
//     // fastllm::Data outputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {3, 4});
//     fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT16, {1, 2}, std::vector<float>{1.f, 5.f});
//     fastllm::Data outputs = fastllm::Data(fastllm::DataType::FLOAT16, {1, 2}, std::vector<float>{3.f, 4.f});
//     switch (optype)
//     {
//     case 0:
//         //inputs.ToDevice(fastllm::DataDevice::ASCEND);
//         //outputs.ToDevice(fastllm::DataDevice::ASCEND);
//         fastllm::AddTo(outputs, inputs, 1);
//         break;
//     case 1:
//         fastllm::Cat(inputs, inputs, 0, outputs);
//         break;
//     case 2:
//         fastllm::Mul(inputs, 2, outputs);
//         break;
//     case 3:
//         fastllm::Permute(inputs, {1, 0}, outputs);
//         break;
//     case 4:
//         fastllm::Split(inputs, 0, 0, 1, outputs);
//         break;
//     case 5:
//         fastllm::Permute(inputs, {1, 0}, outputs);
//         fastllm::MatMul(inputs, outputs, outputs);
//         break;
//     default:
//         break;
//     } 
//     outputs.ToDevice(fastllm::DataDevice::CPU);
//     std::cout<<"case:"<<optype<<std::endl;
//     outputs.Print();
// }

void callBaseOp(int optype=0) {
    std::cout << "\n=======================================" << std::endl;
    std::cout << "Running Case: " << optype << std::endl;

    switch (optype) {
        case 0: {
            // 【难度升级】：3D 张量的 In-place AddTo
            std::vector<float> vecA(24); for(int i=0;i<24;++i) vecA[i] = i * 1.0f;
            std::vector<float> vecB(24); for(int i=0;i<24;++i) vecB[i] = 10.0f;
            
            // 直接构造，坚决不用赋值运算符！
            fastllm::Data inputs(fastllm::DataType::FLOAT16, {2, 3, 4}, vecA);
            fastllm::Data outputs(fastllm::DataType::FLOAT16, {2, 3, 4}, vecB);
            
            inputs.ToDevice(fastllm::DataDevice::ASCEND);
            outputs.ToDevice(fastllm::DataDevice::ASCEND);
            
            fastllm::AddTo(outputs, inputs, 2.0f); 
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        case 1: {
            // 【难度升级】：3D 张量在深层轴 (axis=2) 上进行 Cat
            std::vector<float> vecA(12, 1.0f); 
            std::vector<float> vecB(16, 2.0f); 
            fastllm::Data input0(fastllm::DataType::FLOAT16, {2, 2, 3}, vecA);
            fastllm::Data input1(fastllm::DataType::FLOAT16, {2, 2, 4}, vecB); 
            fastllm::Data outputs; // 让算子自己去 Allocate
            
            input0.ToDevice(fastllm::DataDevice::ASCEND);
            input1.ToDevice(fastllm::DataDevice::ASCEND);
            
            fastllm::Cat(input0, input1, 2, outputs);
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        case 2: {
            // 【难度升级】：带负数和浮点的 Mul
            std::vector<float> vecA(12); for(int i=0;i<12;++i) vecA[i] = i * 1.0f;
            fastllm::Data inputs(fastllm::DataType::FLOAT16, {2, 2, 3}, vecA);
            fastllm::Data outputs;
            
            inputs.ToDevice(fastllm::DataDevice::ASCEND);
            fastllm::Mul(inputs, -0.5f, outputs);
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        case 3: {
            // 【地狱难度】：4D 张量的 Permute
            std::vector<float> vecA(24); for(int i=0;i<24;++i) vecA[i] = i * 1.0f;
            fastllm::Data inputs(fastllm::DataType::FLOAT16, {2, 3, 2, 2}, vecA);
            fastllm::Data outputs;
            
            inputs.ToDevice(fastllm::DataDevice::ASCEND);
            fastllm::Permute(inputs, {0, 2, 1, 3}, outputs); 
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        case 4: {
            // 【难度升级】：3D 张量在轴 1 上进行切分
            std::vector<float> vecA(30); for(int i=0;i<30;++i) vecA[i] = i * 1.0f;
            fastllm::Data inputs(fastllm::DataType::FLOAT16, {2, 5, 3}, vecA);
            fastllm::Data outputs;
            
            inputs.ToDevice(fastllm::DataDevice::ASCEND);
            fastllm::Split(inputs, 1, 1, 4, outputs); 
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        case 5: {
            // 【极高难度】：Batched MatMul (BMM) 结合 Permute
            std::vector<float> vecA(12, 1.0f);
            std::vector<float> vecB(24, 2.0f);
            fastllm::Data matA(fastllm::DataType::FLOAT16, {2, 2, 3}, vecA);
            fastllm::Data matB_pre(fastllm::DataType::FLOAT16, {2, 4, 3}, vecB);
            fastllm::Data matB;
            fastllm::Data outputs;
            
            matA.ToDevice(fastllm::DataDevice::ASCEND);
            matB_pre.ToDevice(fastllm::DataDevice::ASCEND);
            
            fastllm::Permute(matB_pre, {0, 2, 1}, matB); 
            fastllm::MatMul(matA, matB, outputs); 
            
            outputs.ToDevice(fastllm::DataDevice::CPU);
            outputs.Print();
            break;
        }
        default:
            break;
    } 
    std::cout << "=======================================\n" << std::endl;
}

void callNormOp(int normType=0){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5}); 
    fastllm::Data weights = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 2});
    fastllm::Data gamma = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 1});
    fastllm::Data beta = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {0, 0});
    fastllm::Data outputs;

    switch (normType)
    {
    case 0:
        fastllm::LayerNorm(inputs, gamma, beta, -1, outputs);
        break;
    case 1:
        fastllm::RMSNorm(inputs, weights, 1e-5, outputs);
        break;
    default:
        break;
    }
    outputs.ToDevice(fastllm::DataDevice::CPU);
    outputs.Print();
}
    
// void callLinearOp() {
//     std::cout << "=== Testing LinearOp on ASCEND (Auto Memory Management) ===" << std::endl;

//     fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 2}); 

//     fastllm::Data weights = fastllm::Data(fastllm::DataType::FLOAT32, {3, 2}, {3, 4, 5, 5, 6, 7});

//     fastllm::Data bias = fastllm::Data(fastllm::DataType::FLOAT32, {1, 3}, {0, 1, 1});

//     fastllm::Data outputs;

//     // inputs.ToDevice(fastllm::DataDevice::ASCEND);
//     // weights.ToDevice(fastllm::DataDevice::ASCEND);
//     // bias.ToDevice(fastllm::DataDevice::ASCEND);

//     // outputs.ToDevice(fastllm::DataDevice::ASCEND);
//     fastllm::Linear(inputs, weights, bias, outputs);

//     //outputs.ToDevice(fastllm::DataDevice::CPU);

//     outputs.Print();
    
//     std::cout << "=== Test Finished ===" << std::endl;
// }



void callLinearOp() {

    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 2}); 

    fastllm::Data weights = fastllm::Data(fastllm::DataType::FLOAT32, {3, 2}, {3, 4, 5, 5, 6, 7});

    fastllm::Data bias = fastllm::Data(fastllm::DataType::FLOAT32, {1, 3}, {0, 1, 1});

    fastllm::Data outputs;

    fastllm::Linear(inputs, weights, bias, outputs);

    float* outPtr = (float*)outputs.cpuData;
    printf("First 5 results: %f, %f, %f, %f, %f\n", 
           outPtr[0], outPtr[1], outPtr[2], outPtr[3], outPtr[4]);

}

void callActivationOp(int activateType=0){
    fastllm::Data inputs = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2}, {1, 5});
    fastllm::Data outputs;
    switch (activateType)
    {
    case 0:
        fastllm::Silu(inputs, outputs);
        break;
    case 1:
        fastllm::Softmax(inputs, outputs, -1);
        break;
    case 2:
        fastllm::Swiglu(inputs, outputs);
        break;
    default:
        break;
    }
    outputs.ToDevice(fastllm::DataDevice::CPU);
    outputs.Print();
}

// void callAttentionOp(int group=1, int attentionType=0){
//     const fastllm::Data q = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {1, 2, 3, 4, 5, 6});
//     const fastllm::Data k = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {5, 6, 7, 8, 9, 10});
//     const fastllm::Data v = fastllm::Data(fastllm::DataType::FLOAT32, {1, 2, 3}, {1, 1, 1, 2, 1, 3});
//     const fastllm::Data mask = fastllm::Data();
//     int dims = q.dims.back();
//     float scale = 1/sqrt(dims);
//     fastllm::Data output;

//     fastllm::Attention(q, k, v, mask, output, group, scale, attentionType);
// }

void callAttentionOp(int group=1, int attentionType=0){
    // fastllm::Data q(fastllm::DataType::FLOAT16, {1, 1, 2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    // fastllm::Data k(fastllm::DataType::FLOAT16, {1, 1, 2, 3}, std::vector<float>{5, 6, 7, 8, 9, 10});
    // fastllm::Data v(fastllm::DataType::FLOAT16, {1, 1, 2, 3}, std::vector<float>{1, 1, 1, 2, 1, 3});
    fastllm::Data q(fastllm::DataType::FLOAT16, {1, 1, 2, 2}, 
                    std::vector<float>{10.0f,  0.0f, 
                                        0.0f, 10.0f});
                                        
    // K 矩阵: 和 Q 一模一样
    fastllm::Data k(fastllm::DataType::FLOAT16, {1, 1, 2, 2}, 
                    std::vector<float>{10.0f,  0.0f, 
                                        0.0f, 10.0f});
                                        
    // V 矩阵: 极具辨识度的阶梯数据
    fastllm::Data v(fastllm::DataType::FLOAT16, {1, 1, 2, 2}, 
                    std::vector<float>{1.0f, 2.0f, 
                                       3.0f, 4.0f});
    fastllm::Data mask; 
    
    int dims = q.dims.back();
    // 缩放因子: 1 / sqrt(d_k)
    float scale = 1.0f / sqrt(dims);
    fastllm::Data output;

    // 呼叫算子执行: Attention(Q, K, V) = Softmax(Q*K^T / sqrt(d_k)) * V
    fastllm::Attention(q, k, v, mask, output, group, scale, attentionType);
    
    // 拉回主板并打印
    output.ToDevice(fastllm::DataDevice::CPU);
    output.Print();
}

void testBase(){
    printf("testing BaseOp...\n");
    for (int i=0;i<6;i++){
        callBaseOp(i);
    }
    printf("test BaseOp finished!\n");
}

void testActivation(){
    printf("testing ActivationOp...\n");
    for (int i=0;i<3;i++){
        callActivationOp(i);
    }
    printf("test ActivationOp finished!\n");
}

void testAttention(){
    printf("testing AttentionOp...\n");
    callAttentionOp();
    printf("test AttentionOp finished!\n");
}

void testLinaer(){
    printf("testing LinearOp...\n");
    callLinearOp();
    printf("test LinearOp finished!\n");
}

void testNorm(){
    printf("testing NormOp...\n");
    for (int i=0;i<2;i++){
        callNormOp(i);
    }
    printf("test NormOp finished!\n");
}

void testAll(){
    testBase();
    testActivation();
    testAttention();
    testNorm();
    testLinaer();
}


int main(){
    fastllm::FastllmAclInit();
    testAll();
}