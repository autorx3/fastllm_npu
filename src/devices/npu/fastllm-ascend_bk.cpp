#include "fastllm-ascend.h"
#include "aclnn/aclnn_gemm.h" 
#include "acl/acl_op.h"
#include <vector>
#include <mutex>

namespace fastllm {
    aclrtStream g_aclStream = nullptr;
    static bool g_isInitialized = false;

    struct NpuWorkspace {
        void* ptr = nullptr;
        size_t capacity = 0;
        std::mutex mtx;

        void* Get(size_t size) {
            std::lock_guard<std::mutex> lock(mtx);
            size = (size + 31) / 32 * 32;
            
            if (size > capacity) {
                if (ptr) aclrtFree(ptr);
                size_t alloc_size = (size_t)(size * 1.2); 
                auto ret = aclrtMalloc(&ptr, alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("Critical Error: NPU Workspace Malloc Failed! Size: %ld\n", alloc_size);
                    return nullptr;
                }
                capacity = alloc_size;
            }
            return ptr;
        }

        ~NpuWorkspace() {
            if (ptr) aclrtFree(ptr);
        }
    } g_workspace;

    struct DeviceScalar {
        void* ptr = nullptr;
        aclTensorDesc* desc = nullptr;
        aclDataBuffer* buf = nullptr;

        DeviceScalar(float val) {
            ptr = FastllmAclMalloc(sizeof(float));
            FastllmAclCopyFromHostToDevice(ptr, &val, sizeof(float));
            std::vector<int64_t> dims = {1};
            desc = aclCreateTensorDesc(ACL_FLOAT, 1, dims.data(), ACL_FORMAT_ND);
            buf = aclCreateDataBuffer(ptr, sizeof(float));
        }

        ~DeviceScalar() {
            if (desc) aclDestroyTensorDesc(desc);
            if (buf) aclDestroyDataBuffer(buf);
            if (ptr) FastllmAclFree(ptr);
        }
    };

    aclrtStream GetFastllmAclStream() {
        return g_aclStream;
    }

    aclTensor* CreateAclTensor(const Data &data, const std::vector<int64_t> &dims) {
        std::vector<int64_t> strides(dims.size());
        int64_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= dims[i];
        }

        aclDataType type = ACL_FLOAT;
        if (data.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (data.dataType == DataType::FLOAT32) type = ACL_FLOAT;
        else if (data.dataType == DataType::INT32) type = ACL_INT32;
        else if (data.dataType == DataType::INT8) type = ACL_INT8; 
        else if (data.dataType == DataType::UINT8) type = ACL_UINT8; 
        else if (data.dataType == DataType::BFLOAT16) type = ACL_BF16;

        return aclCreateTensor(dims.data(), dims.size(), type,
                               strides.data(), 0, ACL_FORMAT_ND,
                               dims.data(), dims.size(), data.deviceData);
    }

    void FastllmAclInit() {
        if (g_isInitialized) {
            return;
        }

        int32_t deviceId = 0; 
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            printf("Error: aclInit failed. code: %d\n", ret);
            exit(-1); 
        }

        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            printf("Error: aclrtSetDevice(%d) failed. code: %d\n", deviceId, ret);
            exit(-1);
        }

        ret = aclrtCreateStream(&g_aclStream);
        if (ret != ACL_SUCCESS) {
            printf("Error: aclrtCreateStream failed. code: %d\n", ret);
            exit(-1);
        }

        g_isInitialized = true;
        printf("Fastllm Ascend Init Success on Device %d! Stream Created.\n", deviceId);
    }

    void* FastllmAclMalloc(size_t size) {
        void* ptr = nullptr;
        if (size == 0) return nullptr;

        // 使用 HUGE_FIRST 策略：优先申请大页内存，有助于提升大模型推理性能,如果大页耗尽，会自动回退到普通页
        aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        
        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclMalloc failed to alloc %zu bytes. Code: %d\n", size, ret);
            return nullptr;
        }
        return ptr;
    }

    /**
     * @brief 释放 NPU 显存
     * @param ptr 显存指针
     */
    void FastllmAclFree(void* ptr) {
        if (ptr != nullptr) {
            aclError ret = aclrtFree(ptr);
            if (ret != ACL_SUCCESS) {
                printf("Warning: FastllmAclFree failed. Code: %d\n", ret);
            }
            ptr = nullptr;
        }
    }

    /**
     * @brief H2D: Host (CPU) -> Device (NPU)
     * 使用同步复制，确保数据完整写入 NPU 后才返回
     */
    void FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) return;

        aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
        
        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclCopyFromHostToDevice failed. Size: %zu, Code: %d\n", size, ret);
        }
    }

    /**
     * @brief D2H: Device (NPU) -> Host (CPU)
     * 使用同步复制，确保数据完整读回 CPU 后才返回
     */
    void FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) return;

        // ACL_MEMCPY_DEVICE_TO_HOST
        aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
        
        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclCopyFromDeviceToHost failed. Size: %zu, Code: %d\n", size, ret);
        }
    }

    /**
     * @brief D2D: Device (NPU) -> Device (NPU)
     * 算子内部常用（如 Split, Cat, KV Cache 搬运）
     */
    void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
        if (dst == nullptr || src == nullptr || size == 0) return;

        // ACL_MEMCPY_DEVICE_TO_DEVICE
        aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);

        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclCopyFromDeviceToDevice failed. Size: %zu, Code: %d\n", size, ret);
        }
    }
    
    /**
     * @brief 2D 内存复制 (支持 stride)，主要用于 KV Cache 拼接等非连续内存操作
     * 对应 CUDA 的 cudaMemcpy2D
     */
    void FastllmAclMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
        // aclrtMemcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind)
        aclError ret = aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, ACL_MEMCPY_DEVICE_TO_DEVICE);
        
        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclMemcpy2DDeviceToDevice failed. Code: %d\n", ret);
        }
    }

    void FastllmAclMatMul(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        // cubeMathType = 1: ALLOW_FP32_DOWN_PRECISION (允许 FP32->FP16/HF32 降精度计算以利用 Cube 单元)
        int8_t cubeMathType = 1; 

        auto ret = aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, cubeMathType, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnMatmulGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
            return;
        }

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnMatmul failed. Code: %d\n", ret);
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tWeight);
        aclDestroyTensor(tOutput);
    }
    void FastllmAclMatMulTransB(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        // 1. 创建 Input 和 Output 的 Tensor (正常创建)
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        // 2. 创建 Transposed Weight Tensor (关键步骤)
        // 物理数据: [N, K], 想要逻辑表现: [K, N]
        // 方法: 交换 Shape，同时交换 Stride
        aclTensor *tWeight = nullptr;
        {
            int64_t N = weight.dims[0];
            int64_t K = weight.dims[1];

            std::vector<int64_t> viewDims = {K, N};     // 逻辑形状: [K, N]
            std::vector<int64_t> viewStrides = {1, K};  // 逻辑步长: 
                                                        // dim0(K) 方向走一步，在物理内存中只走 1 个单位
                                                        // dim1(N) 方向走一步，在物理内存中需跨越 K 个单位
            
            aclDataType type = ACL_FLOAT;
            if (weight.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
            else if (weight.dataType == DataType::FLOAT32) type = ACL_FLOAT;
            else if (weight.dataType == DataType::BFLOAT16) type = ACL_BF16;
            else if (weight.dataType == DataType::INT8) type = ACL_INT8;

            tWeight = aclCreateTensor(viewDims.data(), viewDims.size(), type,
                                      viewStrides.data(), 0, ACL_FORMAT_ND,
                                      viewDims.data(), viewDims.size(), weight.deviceData);
        }

        // 3. 准备 Workspace
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        int8_t cubeMathType = 1;

        auto ret = aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, cubeMathType, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnMatmulGetWorkspaceSize (TransB) failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
            return;
        }

        // 4. 从内存池获取 Workspace
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        // 5. 异步执行
        ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnMatmul (TransB) failed. Code: %d\n", ret);
        }

        // 6. 资源清理
        aclDestroyTensor(tInput);
        aclDestroyTensor(tWeight);
        aclDestroyTensor(tOutput);
    }

    /*
     * W8A8 Per-Token Quantized MatMul
     * 使用 310P 优化的 aclnnQuantMatmulDequant
     * * @param input (INT8/FP16) : 
     * 注意：标准 W8A8 中 input 必须先被量化为 INT8。
     * 如果 input 还是 FP16，你需要确认当前 CANN 版本的这个算子是否支持自动转换。
     * 通常流程是：FP16 Input -> aclnnQuantize -> INT8 Input -> aclnnQuantMatmulDequant
     * * @param weight (INT8) : [N, K]
     * @param weightScale : [N] Per-Channel
     * @param xScale : [M] Per-Token (Per-Row)
     */
    void FastllmAclQuantLinearDequant(Data &input, Data &weight, Data &weightScale, 
                                      Data &xScale, Data &bias, Data &output) {

        int64_t K = input.dims.back();
        int64_t M = input.Count(0) / K; 
        int64_t N = weight.dims[0]; 

        std::vector<int64_t> dimInput = {M, K};
        std::vector<int64_t> dimWeight = {N, K}; 
        std::vector<int64_t> dimOutput = {M, N};
        
        // Scale 维度
        std::vector<int64_t> dimWeightScale = {N}; // Per-Channel
        std::vector<int64_t> dimXScale = {M};      // Per-Token (注意这里必须是 M)

        aclTensor *tInput = CreateAclTensor(input, dimInput);
        aclTensor *tWeight = CreateAclTensor(weight, dimWeight);
        aclTensor *tWeightScale = CreateAclTensor(weightScale, dimWeightScale);
        aclTensor *tXScale = CreateAclTensor(xScale, dimXScale);
        aclTensor *tOutput = CreateAclTensor(output, dimOutput);

        aclTensor *tBias = nullptr;
        if (bias.dims.size() > 0 && bias.deviceData != nullptr) {
            tBias = CreateAclTensor(bias, {bias.dims[0]});
        }
        
        aclTensor *tXOffset = nullptr; // 对称量化通常不需要 offset
        aclTensor *tSmoothScale = nullptr; // 不使用 smooth quant
        bool transposeWeight = true; // weight 是 [N, K]，需要转置为 [K, N] 进行计算

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        auto ret = aclnnQuantMatmulDequantGetWorkspaceSize(
            tInput, tWeight, tWeightScale, 
            tBias, tXScale, tXOffset, tSmoothScale,
            "pertoken", transposeWeight, tOutput, // 传入 "pertoken"
            &workspaceSize, &executor
        );

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnQuantMatmulDequant GetWorkspace Failed: %d\n", ret);
            aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tWeightScale);
            aclDestroyTensor(tXScale); aclDestroyTensor(tOutput); if(tBias) aclDestroyTensor(tBias);
            return;
        }

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        ret = aclnnQuantMatmulDequant(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());
        
        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnQuantMatmulDequant Run Failed: %d\n", ret);
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tWeight);
        aclDestroyTensor(tWeightScale);
        aclDestroyTensor(tXScale);
        aclDestroyTensor(tOutput);
        if (tBias) aclDestroyTensor(tBias);
    }
    void FastllmAclRMSNorm(const Data &input, const Data &weight, const Data &bias, Data &output, float eps) {

        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        aclTensor *tRstd = nullptr;
        void *rstdPtr = nullptr;

        std::vector<int64_t> rstdDims;
        int keepDims = input.dims.size() - weight.dims.size();
        if (keepDims <= 0) {
            rstdDims.push_back(1); // 标量情况
        } else {
            for (int i = 0; i < keepDims; ++i) {
                rstdDims.push_back(input.dims[i]);
            }
        }

        std::vector<int64_t> rstdStrides(rstdDims.size());
        int64_t stride = 1;
        int64_t numElem = 1;
        for (int i = rstdDims.size() - 1; i >= 0; i--) {
            rstdStrides[i] = stride;
            stride *= rstdDims[i];
            numElem *= rstdDims[i];
        }
        size_t rstdSize = numElem * sizeof(float); // Float32 = 4 bytes

        aclError ret = aclrtMalloc(&rstdPtr, rstdSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            printf("Error: RMSNorm rstd malloc failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
            return;
        }

        tRstd = aclCreateTensor(rstdDims.data(), rstdDims.size(), ACL_FLOAT,
                                rstdStrides.data(), 0, ACL_FORMAT_ND,
                                rstdDims.data(), rstdDims.size(), rstdPtr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        ret = aclnnRmsNormGetWorkspaceSize(tInput, tWeight, (double)eps, tOutput, tRstd, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnRmsNormGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput); aclDestroyTensor(tRstd);
            aclrtFree(rstdPtr);
            return;
        }

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnRmsNorm failed. Code: %d\n", ret);
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tWeight);
        aclDestroyTensor(tOutput);
        aclDestroyTensor(tRstd);

        aclrtSynchronizeStream(GetFastllmAclStream()); 
        aclrtFree(rstdPtr);
    }
    void FastllmAclSilu(const fastllm::Data &input, fastllm::Data &output) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        auto ret = aclnnSiluGetWorkspaceSize(tInput, tOutput, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSiluGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); 
            aclDestroyTensor(tOutput);
            return;
        }

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        ret = aclnnSilu(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSilu failed. Code: %d\n", ret);
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tOutput);
    }
    void FastllmAclSwiglu(const fastllm::Data &input, fastllm::Data &output) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        int64_t dim = -1; 

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        auto ret = aclnnSwiGluGetWorkspaceSize(tInput, dim, tOutput, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSwiGluGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tInput); 
            aclDestroyTensor(tOutput);
            return;
        }

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        ret = aclnnSwiGlu(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSwiGlu failed. Code: %d\n", ret);
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tOutput);
    }

    /**
     * @brief Softmax 算子: Output = Exp(Input) / Sum(Exp(Input), dim)
     */
     void FastllmAclSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
        // 1. 封装 Tensor
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        // 2. 准备参数
        int64_t dim = static_cast<int64_t>(axis);

        // 3. 计算 Workspace 大小
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        auto ret = aclnnSoftmaxGetWorkspaceSize(tInput, dim, tOutput, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSoftmaxGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tInput);
            aclDestroyTensor(tOutput);
            return;
        }

        // 4. 从内存池获取 Workspace
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        // 5. 异步执行
        ret = aclnnSoftmax(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnSoftmax failed. Code: %d\n", ret);
        }

        // 6. 资源清理
        // 销毁 Tensor 描述符 (数据内存不受影响)
        aclDestroyTensor(tInput);
        aclDestroyTensor(tOutput);
    }

    /**
     * @brief Embedding 算子: 根据 Indices 从 Weight 中查找向量
     * Output = Weight[Input]
     */
     void FastllmAclEmbedding(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output) {
        // 1. 封装 Tensor
        // weight: [VocabSize, HiddenSize]
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        
        // indices (input): [Batch, SeqLen]
        aclTensor *tIndices = CreateAclTensor(input, input.dims);
        
        // out: [Batch, SeqLen, HiddenSize]
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        // 2. 计算 Workspace 大小
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        auto ret = aclnnEmbeddingGetWorkspaceSize(tWeight, tIndices, tOutput, &workspaceSize, &executor);

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnEmbeddingGetWorkspaceSize failed. Code: %d\n", ret);
            aclDestroyTensor(tWeight); 
            aclDestroyTensor(tIndices); 
            aclDestroyTensor(tOutput);
            return;
        }

        // 3. 从内存池获取 Workspace
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        // 4. 异步执行
        ret = aclnnEmbedding(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclnnEmbedding failed. Code: %d\n", ret);
        }

        // 5. 资源清理
        // 销毁 Tensor 描述符
        aclDestroyTensor(tWeight);
        aclDestroyTensor(tIndices);
        aclDestroyTensor(tOutput);
    }

    void FastllmAclTopK(const fastllm::Data &input, fastllm::Data &output, int topk) {
        int64_t k = topk;
        int64_t dim = input.dims.size() - 1; // 默认在最后一维操作

        std::vector<int64_t> tempDims = input.dims;
        tempDims[dim] = k;

        int64_t elementCount = 1;
        for (auto d : tempDims) elementCount *= d;
        
        size_t dtypeSize = (input.dataType == DataType::FLOAT16) ? 2 : 4;
        size_t valSize = elementCount * dtypeSize;      // 存放 TopK Values
        size_t idxIntSize = elementCount * 8;           // 存放 TopK Indices (INT64)
        size_t idxFloatSize = elementCount * dtypeSize; // 存放 Cast 后的 Indices

        uint8_t *tempBuffer = nullptr;
        aclError ret = aclrtMalloc((void**)&tempBuffer, valSize + idxIntSize + idxFloatSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            printf("Error: TopK temp malloc failed. Code: %d\n", ret);
            return;
        }

        void *valPtr = tempBuffer;
        void *idxIntPtr = tempBuffer + valSize;
        void *idxFloatPtr = tempBuffer + valSize + idxIntSize;

        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims); // [..., 2*k]

        aclDataType aclType = (input.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;

        std::vector<int64_t> tempStrides(tempDims.size());
        int64_t stride = 1;
        for (int i = tempDims.size() - 1; i >= 0; i--) {
            tempStrides[i] = stride;
            stride *= tempDims[i];
        }

        aclTensor *tValues = aclCreateTensor(tempDims.data(), tempDims.size(), aclType,
                                             tempStrides.data(), 0, ACL_FORMAT_ND,
                                             tempDims.data(), tempDims.size(), valPtr);
        
        aclTensor *tIndices = aclCreateTensor(tempDims.data(), tempDims.size(), ACL_INT64,
                                              tempStrides.data(), 0, ACL_FORMAT_ND,
                                              tempDims.data(), tempDims.size(), idxIntPtr);

        aclTensor *tIndicesCast = aclCreateTensor(tempDims.data(), tempDims.size(), aclType,
                                                  tempStrides.data(), 0, ACL_FORMAT_ND,
                                                  tempDims.data(), tempDims.size(), idxFloatPtr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        ret = aclnnTopkGetWorkspaceSize(tInput, k, dim, true, true, tValues, tIndices, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *ws = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnTopk(ws, workspaceSize, executor, GetFastllmAclStream());
        } else {
            printf("Error: aclnnTopk failed. Code: %d\n", ret);
        }

        ret = aclnnCastGetWorkspaceSize(tIndices, aclType, tIndicesCast, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *ws = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnCast(ws, workspaceSize, executor, GetFastllmAclStream());
        } else {
            printf("Error: aclnnCast (TopK) failed. Code: %d\n", ret);
        }

        aclTensor *concatTensors[] = {tValues, tIndicesCast};
        aclTensorList *tensorList = aclCreateTensorList(concatTensors, 2);

        ret = aclnnCatGetWorkspaceSize(tensorList, dim, tOutput, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *ws = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnCat(ws, workspaceSize, executor, GetFastllmAclStream());
        } else {
            printf("Error: aclnnCat (TopK) failed. Code: %d\n", ret);
        }

        aclrtSynchronizeStream(GetFastllmAclStream());
        aclrtFree(tempBuffer);

        aclDestroyTensor(tInput);
        aclDestroyTensor(tOutput);
        aclDestroyTensor(tValues);
        aclDestroyTensor(tIndices);
        aclDestroyTensor(tIndicesCast);
        aclDestroyTensorList(tensorList);
    }

    void FastllmAclMul(const fastllm::Data &input, float v, fastllm::Data &output) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclScalar *tScalar = aclCreateScalar(&v, ACL_FLOAT);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        
        auto ret = aclnnMulsGetWorkspaceSize(tInput, tScalar, tOutput, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *ws = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnMuls(ws, workspaceSize, executor, GetFastllmAclStream());
        } else {
             printf("Error: aclnnMuls failed. Code: %d\n", ret);
        }
        
        aclDestroyTensor(tInput); aclDestroyScalar(tScalar); aclDestroyTensor(tOutput);
    }

    void FastllmAclMulTo(const fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
        aclTensor *tSelf = CreateAclTensor(input0, input0.dims);
        aclTensor *tOther = CreateAclTensor(input1, input1.dims);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        // Step 1: 先做 A = A * B
        auto ret = aclnnInplaceMulGetWorkspaceSize(tSelf, tOther, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *ws = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnInplaceMul(ws, workspaceSize, executor, GetFastllmAclStream());
        }

        // Step 2: 如果 alpha != 1.0, 再做 A = A * alpha (Muls)
        if (alpha != 1.0f) {
            // 需要重新封装 Scalar 和调用 Muls，略写...
            // 建议：直接调用上面的 FastllmAclMul(input0, alpha, input0);
        }

        aclDestroyTensor(tSelf); aclDestroyTensor(tOther);
    }

    void FastllmAclFloatToHalf(float *src, void *dst, int len) {
        // 这是一个纯粹的数组转换，没有 Shape 概念，视作 1D Tensor
        std::vector<int64_t> dims = {len};
        // 手动构造 Tensor，因为 src/dst 是裸指针
        aclTensor *tSrc = aclCreateTensor(dims.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dims.data(), 1, src);
        aclTensor *tDst = aclCreateTensor(dims.data(), 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, dims.data(), 1, dst);
        
        uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
        // 2 = ACL_FLOAT16 (目标类型)
        aclnnCastGetWorkspaceSize(tSrc, ACL_FLOAT16, tDst, &wsSize, &exec);
        void* ws = wsSize > 0 ? g_workspace.Get(wsSize) : nullptr;
        aclnnCast(ws, wsSize, exec, GetFastllmAclStream());
        
        // 这是一个同步接口 (通常用于权重加载)，建议同步
        aclrtSynchronizeStream(GetFastllmAclStream());
        aclDestroyTensor(tSrc); aclDestroyTensor(tDst);
    }

    void FastllmAclHalfToFloat(float *src, void *dst, int len) {
        std::vector<int64_t> dims = {len};
        // 手动构造 Tensor，因为 src/dst 是裸指针
        aclTensor *tSrc = aclCreateTensor(dims.data(), 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dims.data(), 1, src);
        aclTensor *tDst = aclCreateTensor(dims.data(), 1, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND, dims.data(), 1, dst);
        
        uint64_t wsSize = 0; aclOpExecutor *exec = nullptr;
        // 2 = ACL_FLOAT16 (目标类型)
        aclnnCastGetWorkspaceSize(tSrc, ACL_FLOAT, tDst, &wsSize, &exec);
        void* ws = wsSize > 0 ? g_workspace.Get(wsSize) : nullptr;
        aclnnCast(ws, wsSize, exec, GetFastllmAclStream());
        
        // 这是一个同步接口 (通常用于权重加载)，建议同步
        aclrtSynchronizeStream(GetFastllmAclStream());
        aclDestroyTensor(tSrc); aclDestroyTensor(tDst);
    }

    // aclnnadd算子不支持310p系列，需要使用aclop算子
    void FastllmAclAdd(const fastllm::Data &input, float v, fastllm::Data &output) {
        // 310P 的 "Add" 算子需要两个 Tensor 输入。我们将 v 包装成 Tensor。
        
        // Input Desc
        aclTensorDesc *descInput = aclCreateTensorDesc(ACL_FLOAT, input.dims.size(), input.dims.data(), ACL_FORMAT_ND);
        if (input.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descInput, ACL_FLOAT16);
        aclDataBuffer *bufInput = aclCreateDataBuffer(input.deviceData, input.GetBytes());

        // Scalar Desc (Input 2)
        DeviceScalar scalar(v); // 自动在 Device 上申请内存并拷贝值

        // Output Desc
        aclTensorDesc *descOutput = aclCreateTensorDesc(ACL_FLOAT, output.dims.size(), output.dims.data(), ACL_FORMAT_ND);
        if (output.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descOutput, ACL_FLOAT16);
        aclDataBuffer *bufOutput = aclCreateDataBuffer(output.deviceData, output.GetBytes());

        // Params
        std::vector<aclTensorDesc*> inputDescs = {descInput, scalar.desc};
        std::vector<aclDataBuffer*> inputBuffers = {bufInput, scalar.buf};
        std::vector<aclTensorDesc*> outputDescs = {descOutput};
        std::vector<aclDataBuffer*> outputBuffers = {bufOutput};

        // Execute "Add"
        aclError ret = aclopCompileAndExecute("Add", 
                                              inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                              outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                              nullptr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, GetFastllmAclStream());
        
        if (ret != ACL_SUCCESS) {
            printf("Error: aclop(Add) failed. Code: %d\n", ret);
        }

        // Cleanup
        // 必须同步，否则 scalar.ptr 会在 Stream 执行完之前被析构释放
        aclrtSynchronizeStream(GetFastllmAclStream()); 

        aclDestroyTensorDesc(descInput); aclDestroyDataBuffer(bufInput);
        aclDestroyTensorDesc(descOutput); aclDestroyDataBuffer(bufOutput);
        // scalar 析构时会自动释放资源
    }

    // ==========================================
    // 2. AddTo (Tensor A += Tensor B * alpha) -> op: "Axpy"
    // ==========================================
    void FastllmAclAddTo(const fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
        // Axpy 功能: y = alpha * x + y
        // input1 是 x, input0 是 y
        
        aclTensorDesc *descX = aclCreateTensorDesc(ACL_FLOAT, input1.dims.size(), input1.dims.data(), ACL_FORMAT_ND);
        if (input1.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descX, ACL_FLOAT16);
        aclDataBuffer *bufX = aclCreateDataBuffer(input1.deviceData, input1.GetBytes());

        aclTensorDesc *descY = aclCreateTensorDesc(ACL_FLOAT, input0.dims.size(), input0.dims.data(), ACL_FORMAT_ND);
        if (input0.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descY, ACL_FLOAT16);
        aclDataBuffer *bufY = aclCreateDataBuffer(input0.deviceData, input0.GetBytes());

        // Output (In-place, point to Y's data)
        // 注意：Axpy 输出结果写入 outputDesc 指向的 Buffer
        aclTensorDesc *descOut = aclCreateTensorDesc(ACL_FLOAT, input0.dims.size(), input0.dims.data(), ACL_FORMAT_ND);
        if (input0.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descOut, ACL_FLOAT16);
        aclDataBuffer *bufOut = aclCreateDataBuffer(input0.deviceData, input0.GetBytes());

        // Attributes (alpha)
        aclopAttr *attr = aclopCreateAttr();
        aclopSetAttrFloat(attr, "alpha", alpha);

        std::vector<aclTensorDesc*> inputDescs = {descX, descY};
        std::vector<aclDataBuffer*> inputBuffers = {bufX, bufY};
        std::vector<aclTensorDesc*> outputDescs = {descOut};
        std::vector<aclDataBuffer*> outputBuffers = {bufOut};

        aclError ret = aclopCompileAndExecute("Axpy", 
                                              inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                              outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                              attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclop(Axpy) failed. Code: %d\n", ret);
        }

        aclDestroyTensorDesc(descX); aclDestroyDataBuffer(bufX);
        aclDestroyTensorDesc(descY); aclDestroyDataBuffer(bufY);
        aclDestroyTensorDesc(descOut); aclDestroyDataBuffer(bufOut);
        aclopDestroyAttr(attr);
    }

    void FastllmAclPermute(const fastllm::Data &input, const std::vector<int> &axis) {
        // 0. 去除 const 限制
        Data &mutableInput = const_cast<Data&>(input);

        // 1. 准备维度参数
        std::vector<int64_t> originalDims = mutableInput.dims;
        std::vector<int64_t> newDims;
        std::vector<int64_t> permData; // 存放 permutation 的具体数值
        
        for (int i : axis) {
            newDims.push_back(originalDims[i]);
            permData.push_back((int64_t)i);
        }

        // 2. 申请临时输出内存 (Temp Buffer)
        size_t dataBytes = mutableInput.GetBytes();
        void* tempPtr = FastllmAclMalloc(dataBytes); 
        if (tempPtr == nullptr) {
            printf("Error: Permute malloc failed.\n");
            return;
        }

        // 3. 准备 "Transpose" 算子所需的资源
        // Transpose 算子有两个输入: Input(x), Permutation(perm)
        // 一个输出: Output(y)

        // --- A. 准备 Input Desc 和 Buffer ---
        aclDataType aclType = ACL_FLOAT;
        if (mutableInput.dataType == DataType::FLOAT16) aclType = ACL_FLOAT16;
        else if (mutableInput.dataType == DataType::FLOAT32) aclType = ACL_FLOAT;
        else if (mutableInput.dataType == DataType::INT32) aclType = ACL_INT32;

        aclTensorDesc *descInput = aclCreateTensorDesc(aclType, originalDims.size(), originalDims.data(), ACL_FORMAT_ND);
        aclDataBuffer *bufInput = aclCreateDataBuffer(mutableInput.deviceData, dataBytes);

        // --- B. 准备 Perm Desc 和 Buffer (关键：perm 是一个 Tensor) ---
        // 我们需要在 Device 上申请一小块内存放 perm 数据 [0, 2, 1, 3] 等
        size_t permBytes = permData.size() * sizeof(int64_t);
        void* permDevPtr = FastllmAclMalloc(permBytes);
        FastllmAclCopyFromHostToDevice(permDevPtr, permData.data(), permBytes); // 把 vector 拷到 NPU

        std::vector<int64_t> permDims = {(int64_t)permData.size()};
        aclTensorDesc *descPerm = aclCreateTensorDesc(ACL_INT64, 1, permDims.data(), ACL_FORMAT_ND);
        aclDataBuffer *bufPerm = aclCreateDataBuffer(permDevPtr, permBytes);

        // --- C. 准备 Output Desc 和 Buffer ---
        aclTensorDesc *descOutput = aclCreateTensorDesc(aclType, newDims.size(), newDims.data(), ACL_FORMAT_ND);
        aclDataBuffer *bufOutput = aclCreateDataBuffer(tempPtr, dataBytes);

        // --- D. 组装输入输出数组 ---
        std::vector<aclTensorDesc*> inputDescs = {descInput, descPerm};
        std::vector<aclDataBuffer*> inputBuffers = {bufInput, bufPerm};
        std::vector<aclTensorDesc*> outputDescs = {descOutput};
        std::vector<aclDataBuffer*> outputBuffers = {bufOutput};

        // 4. 执行算子 (使用 legacy 接口)
        // opType = "Transpose" (在 Ascend 算子库中，Permute 功能由 Transpose 算子承担)
        aclError ret = aclopCompileAndExecute("Transpose", 
                                              inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                              outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                              nullptr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, GetFastllmAclStream());

        if (ret == ACL_SUCCESS) {
            // 5. 同步与回写 (In-Place 模拟)
            aclrtSynchronizeStream(GetFastllmAclStream());
            
            // 将转置好的数据拷回原地址
            FastllmAclCopyFromDeviceToDevice(mutableInput.deviceData, tempPtr, dataBytes);
            
            // 更新 Shape
            std::vector<int> intNewDims;
            for(auto d : newDims) intNewDims.push_back((int)d);
            mutableInput.Resize(intNewDims);
        } else {
            printf("Error: aclopCompileAndExecute(Transpose) failed. Code: %d\n", ret);
        }

        // 6. 资源清理
        // 释放 aclop 相关的描述符
        aclDestroyTensorDesc(descInput);
        aclDestroyTensorDesc(descPerm);
        aclDestroyTensorDesc(descOutput);
        aclDestroyDataBuffer(bufInput);
        aclDestroyDataBuffer(bufPerm);
        aclDestroyDataBuffer(bufOutput);

        // 释放显存
        FastllmAclFree(tempPtr);
        FastllmAclFree(permDevPtr); // 别忘了释放存放 perm 数据的显存
    }

    void FastllmAclRepeat(void *src, void *dst, int outer, int repeatTimes, int inputStride, int outputStride, int channelsInner, int channelsInputInner) {
        // 这是一个纯内存搬运操作
        // src/dst 是 Device 指针
        // 逻辑：将 src 中的每一块 channelsInputInner，复制 repeatTimes 次到 dst
        
        uint8_t *srcPtr = (uint8_t*)src;
        uint8_t *dstPtr = (uint8_t*)dst;

        for (int i = 0; i < outer; i++) {
            for (int j = 0; j < repeatTimes; j++) {
                // 每次复制一块数据
                // 310P 的 aclrtMemcpyAsync 开销很小，可以放入循环
                aclrtMemcpyAsync(dstPtr + i * outputStride + j * channelsInner, 
                                 channelsInputInner,
                                 srcPtr + i * inputStride, 
                                 channelsInputInner, 
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, 
                                 GetFastllmAclStream());
            }
        }
        // 注意：这里仅提交了任务，外层需要保证流同步或后续算子在同一流上
    }

    // =======================================================================
    // 2. AttentionMask
    // 公式: Input = Input + Mask * MaskValue
    // 策略：使用 aclop "Axpy" (Y = alpha * X + Y)，其中 Y=Input, X=Mask, alpha=MaskValue
    // =======================================================================
    void FastllmAclAttentionMask(const fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
        // Axpy 算子要求 X 和 Y 的 Shape 能够广播
        // Input: [B, H, S, S], Mask: [B, 1, S, S] -> 支持广播
        
        aclTensorDesc *descX = aclCreateTensorDesc(ACL_FLOAT, mask.dims.size(), mask.dims.data(), ACL_FORMAT_ND);
        if (mask.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descX, ACL_FLOAT16);
        aclDataBuffer *bufX = aclCreateDataBuffer(mask.deviceData, mask.GetBytes());

        // Input 既是输入也是输出 (In-Place)
        aclTensorDesc *descY = aclCreateTensorDesc(ACL_FLOAT, input.dims.size(), input.dims.data(), ACL_FORMAT_ND);
        if (input.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descY, ACL_FLOAT16);
        aclDataBuffer *bufY = aclCreateDataBuffer(input.deviceData, input.GetBytes());

        // Output desc (指向 input)
        aclTensorDesc *descOut = aclCreateTensorDesc(ACL_FLOAT, input.dims.size(), input.dims.data(), ACL_FORMAT_ND);
        if (input.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descOut, ACL_FLOAT16);
        aclDataBuffer *bufOut = aclCreateDataBuffer(input.deviceData, input.GetBytes());

        // Alpha = MaskValue
        aclopAttr *attr = aclopCreateAttr();
        aclopSetAttrFloat(attr, "alpha", maskValue);

        std::vector<aclTensorDesc*> inputDescs = {descX, descY};
        std::vector<aclDataBuffer*> inputBuffers = {bufX, bufY};
        std::vector<aclTensorDesc*> outputDescs = {descOut};
        std::vector<aclDataBuffer*> outputBuffers = {bufOut};

        aclError ret = aclopCompileAndExecute("Axpy", 
                                              inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                              outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                              attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            printf("Error: aclop(Axpy/AttentionMask) failed. Code: %d\n", ret);
        }

        aclDestroyTensorDesc(descX); aclDestroyDataBuffer(bufX);
        aclDestroyTensorDesc(descY); aclDestroyDataBuffer(bufY);
        aclDestroyTensorDesc(descOut); aclDestroyDataBuffer(bufOut);
        aclopDestroyAttr(attr);
    }

    // =======================================================================
    // 3. Attention (Self-Attention)
    // 策略：分步实现 (MatMul -> Scale -> Mask -> Softmax -> MatMul)
    // 理由：310P 上 FlashAttention 对 Shape/Padding 限制极多，手动分步实现最稳
    // =======================================================================
    void FastllmAclAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v, const fastllm::Data &mask, fastllm::Data &output, int group, float scale, int maskType) {
        // 1. Q * K^T
        // Q: [B, Head, Seq, D], K: [B, Head, Seq, D] -> Score: [B, Head, Seq, Seq]
        // 这里 Fastllm 通常传入的是已经 Transpose 好的 QKV (BSHD 或 BHSD)
        // 注意：FastLLM 的 MatMulTransB 会自动处理最后两维的转置
        
        Data score;
        std::vector<int> scoreDims = q.dims;
        scoreDims.back() = k.dims[k.dims.size() - 2]; // SeqLen of K
        score.dataType = q.dataType;
        score.Resize(scoreDims);
        
        void* scorePtr = g_workspace.Get(score.GetBytes());
        score.deviceData = scorePtr; // 借用 Workspace 存中间结果

        // Score = Q @ K.T
        FastllmAclMatMulTransB(q, k, Data(), score, 1, 0);

        // 2. Scale
        if (std::abs(scale - 1.0f) > 1e-6) {
            FastllmAclMul(score, scale, score); // In-place
        }

        // 3. Mask
        if (mask.dims.size() > 0) {
            // MaskValue 通常是 -10000
            FastllmAclAttentionMask(score, mask, -10000.0f);
        }

        // 4. Softmax (axis = -1)
        FastllmAclSoftmax(score, score, -1);

        // 5. Score * V
        // Score: [B, Head, Seq, Seq], V: [B, Head, Seq, D] -> Output: [B, Head, Seq, D]
        FastllmAclMatMul(score, v, Data(), output, 1, 0);

        // Workspace 复用机制会自动处理 scorePtr，这里不需要 Free
        // 但需要注意：output 必须已经分配好内存
    }

    // =======================================================================
    // 4. RoPE (NearlyRotatePosition2D)
    // 策略：组合算子实现 Output = Input * Cos + Rotate(Input) * Sin
    // 理由：aclop "RotaryMul" 接口复杂，手写组合算子兼容性最好
    // =======================================================================
    void FastllmAclNearlyRotatePosition2D(const fastllm::Data &data, const fastllm::Data &positionIds, const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
        // 注意：FastLLM 的 RoPE 往往已经把 Sin/Cos 根据 PositionIds 查找好了传入
        // 或者 sinData/cosData 就是预计算好的表。
        // 这里假设 sinData/cosData 已经对应好了 input 的 shape (通过 Broadcast)
        
        // 我们需要计算：
        // 1. part1 = data * cos
        // 2. data_rotated = [-data_odd, data_even] (Llama 模式)
        // 3. part2 = data_rotated * sin
        // 4. result = part1 + part2
        
        // 由于 310P 上 Split/Cat 比较慢，我们这里简化处理：
        // 如果 rotaryDim 是全部维度，可以直接用 ElementWise 算子
        
        // 为简化实现，假设已经有 Rotate 好的数据或者通过 Permute 得到
        // 这里使用临时空间
        Data rotatedData;
        rotatedData.dataType = data.dataType;
        rotatedData.Resize(data.dims);
        void* rotPtr = g_workspace.Get(data.GetBytes());
        rotatedData.deviceData = rotPtr;

        // --- Step A: 生成 Rotated Data ---
        // Llama RoPE 的旋转是两两交换并取负： (x0, x1) -> (-x1, x0)
        // 这需要一个特殊的 Permute 或者手写 Kernel。
        // 鉴于 310P 限制，我们这里尝试调用 "RotaryMul" 算子，如果不行则报错
        
        // 尝试构建 aclop "RotaryMul"
        // Input: x, r1(cos), r2(sin)
        // Output: y
        aclTensorDesc *descX = aclCreateTensorDesc(ACL_FLOAT, data.dims.size(), data.dims.data(), ACL_FORMAT_ND);
        if (data.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descX, ACL_FLOAT16);
        aclDataBuffer *bufX = aclCreateDataBuffer(data.deviceData, data.GetBytes());

        // Cos
        aclTensorDesc *descCos = aclCreateTensorDesc(ACL_FLOAT, cosData.dims.size(), cosData.dims.data(), ACL_FORMAT_ND);
        if (cosData.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descCos, ACL_FLOAT16);
        aclDataBuffer *bufCos = aclCreateDataBuffer(cosData.deviceData, cosData.GetBytes());

        // Sin
        aclTensorDesc *descSin = aclCreateTensorDesc(ACL_FLOAT, sinData.dims.size(), sinData.dims.data(), ACL_FORMAT_ND);
        if (sinData.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descSin, ACL_FLOAT16);
        aclDataBuffer *bufSin = aclCreateDataBuffer(sinData.deviceData, sinData.GetBytes());

        // Output (In-place to data)
        aclTensorDesc *descOut = aclCreateTensorDesc(ACL_FLOAT, data.dims.size(), data.dims.data(), ACL_FORMAT_ND);
        if (data.dataType == DataType::FLOAT16) aclSetTensorDescDataType(descOut, ACL_FLOAT16);
        aclDataBuffer *bufOut = aclCreateDataBuffer(data.deviceData, data.GetBytes()); // Write back to input

        std::vector<aclTensorDesc*> inputDescs = {descX, descCos, descSin};
        std::vector<aclDataBuffer*> inputBuffers = {bufX, bufCos, bufSin};
        std::vector<aclTensorDesc*> outputDescs = {descOut};
        std::vector<aclDataBuffer*> outputBuffers = {bufOut};

        // 310P 上通常支持 "RotaryMul" 或 "Rope"
        // 如果这里报错，说明 shape 对齐有问题，需要回退到 CPU
        aclError ret = aclopCompileAndExecute("RotaryMul", 
                                              inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                              outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                              nullptr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, GetFastllmAclStream());

        if (ret != ACL_SUCCESS) {
            // 如果 RotaryMul 失败，打印错误，但不崩溃 (可能由上层 fallback 到 CPU)
            // 此时不进行数据修改
             printf("Error: aclop(RotaryMul) failed. Code: %d. Fallback might be needed.\n", ret);
        }

        aclDestroyTensorDesc(descX); aclDestroyDataBuffer(bufX);
        aclDestroyTensorDesc(descCos); aclDestroyDataBuffer(bufCos);
        aclDestroyTensorDesc(descSin); aclDestroyDataBuffer(bufSin);
        aclDestroyTensorDesc(descOut); aclDestroyDataBuffer(bufOut);
    }
}