 #include <vector>
 #include <mutex>
 #include <iostream>
 #include <cmath>
 #include <cstring>
 #include <cstdio>

 #include "fastllm-ascend.h"
 #include "acl/acl_op.h" 
 #include "acl/acl.h"
 #include "aclnnop/aclnn_add.h"
 #include "aclnnop/aclnn_mul.h"
 #include "aclnnop/aclnn_permute.h"
 #include "aclnnop/aclnn_apply_rotary_pos_emb_v2.h"
 #include "aclnnop/aclnn_quant_matmul_dequant.h"
 #include "aclnnop/aclnn_softmax.h"
 #include "aclnnop/aclnn_rms_norm.h"
 #include "aclnnop/aclnn_mul.h"
 #include "aclnnop/aclnn_matmul.h"
 #include "aclnnop/aclnn_silu.h"
 #include "aclnnop/aclnn_swi_glu.h"
 #include "aclnnop/aclnn_embedding.h"
 #include "aclnnop/aclnn_topk.h"
 #include "aclnnop/aclnn_cast.h"
 #include "aclnnop/aclnn_cat.h"
 #include "aclnnop/aclnn_expand.h"
 #include "aclnnop/aclnn_prompt_flash_attention_v3.h"

namespace fastllm {

    aclrtStream g_aclStream = nullptr;
    static bool g_isInitialized = false;

    struct NpuWorkspace {
        void* basePtr = nullptr;      
        size_t capacity = 0;          
        size_t currentOffset = 0;     
        std::mutex mtx; 

        // 默认预分配大小：256MB
        const size_t DEFAULT_POOL_SIZE = 256 * 1024 * 1024; 

        // 【关键修改】构造函数什么都不做，避免在 aclInit 前调用 aclrtMalloc
        NpuWorkspace() {}

        void* Get(size_t size) {
            std::lock_guard<std::mutex> lock(mtx);
            
            // 1. 延迟初始化：第一次被调用时才申请内存
            // 此时 main 函数肯定已经运行，aclInit 肯定已经完成了
            if (basePtr == nullptr) {
                aclError ret = aclrtMalloc(&basePtr, DEFAULT_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("CRITICAL ERROR: NpuWorkspace init failed. Code: %d\n", ret);
                    return nullptr;
                }
                capacity = DEFAULT_POOL_SIZE;
                // printf("NpuWorkspace Initialized: %zu MB\n", capacity / 1024 / 1024);
            }

            // 2. 32字节对齐
            size_t alignSize = (size + 31) / 32 * 32;

            // 3. 检查剩余空间与扩容
            if (currentOffset + alignSize > capacity) {
                printf("WARNING: NpuWorkspace expanding! Old: %zu MB, Needed: %zu MB\n", 
                       capacity/1024/1024, (currentOffset + alignSize)/1024/1024);
                
                    if (currentOffset == 0 && basePtr) {
                        aclrtFree(basePtr);
                        basePtr = nullptr;
                    } else {
                        // 如果 currentOffset > 0 且需要扩容，说明当前函数正在使用旧内存块
                        // 这是一个逻辑死锁。建议在这里直接报 Error，提示增加初始 Pool 大小。
                        printf("CRITICAL ERROR: Workspace out of memory during a single operator call!\n");
                    }
                
                size_t newCapacity = std::max(capacity * 2, currentOffset + alignSize + 64 * 1024 * 1024);
                aclError ret = aclrtMalloc(&basePtr, newCapacity, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("CRITICAL ERROR: NpuWorkspace Expand Failed!\n");
                    return nullptr;
                }
                capacity = newCapacity;
                currentOffset = 0; 
            }

            // 4. 返回地址
            void* ptr = (uint8_t*)basePtr + currentOffset;
            currentOffset += alignSize;
            
            return ptr;
        }

        void Reset() {
            std::lock_guard<std::mutex> lock(mtx);
            currentOffset = 0;
        }

        ~NpuWorkspace() {
            if (basePtr) aclrtFree(basePtr);
        }
    } g_workspace;


    aclrtStream GetFastllmAclStream() { return g_aclStream; }

    void FastllmAclInit() {
        if (g_isInitialized) return;

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

    void FastllmAclClearWorkspace() {
        g_workspace.Reset();
    }


    void* FastllmAclMalloc(size_t size) {
        void* ptr = nullptr;
        if (size == 0) return nullptr;
        aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            printf("Error: FastllmAclMalloc failed to alloc %zu bytes. Code: %d\n", size, ret);
            return nullptr;
        }
        return ptr;
    }

    void FastllmAclFree(void* ptr) {
        if (ptr != nullptr) {
            aclError ret = aclrtFree(ptr);
            if (ret != ACL_SUCCESS) {
                printf("Warning: FastllmAclFree failed. Code: %d\n", ret);
            }
        }
    }

    void FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size) {
        aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    void FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size) {
        aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
        aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }
    
    void FastllmAclMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height) {
        aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, ACL_MEMCPY_DEVICE_TO_DEVICE);
    }

    aclTensor* CreateAclTensor(const Data &data, const std::vector<int> &dims, void* customDevPtr = nullptr) {
        std::vector<int64_t> dims64;
        dims64.reserve(dims.size());
        for (int d : dims) dims64.push_back(d);

        std::vector<int64_t> strides(dims.size());
        int64_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= dims[i];
        }

        aclDataType type = ACL_FLOAT;
        if (data.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (data.dataType == DataType::FLOAT32) type = ACL_FLOAT;
        else if (data.dataType == DataType::INT8) type = ACL_INT8; 
        else if (data.dataType == DataType::BFLOAT16) type = ACL_BF16;

        void* ptr = customDevPtr ? customDevPtr : data.deviceData;

        return aclCreateTensor(dims64.data(), dims64.size(), type,
                               strides.data(), 0, ACL_FORMAT_ND,
                               dims64.data(), dims64.size(), ptr);
    }

    void FastllmAclMatMul(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        int8_t cubeMathType = 1; 

        if (aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, cubeMathType, &workspaceSize, &executor) == ACL_SUCCESS) {
            aclnnMatmul(g_workspace.Get(workspaceSize), workspaceSize, executor, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
    }

    void FastllmAclMatMulTransB(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        int64_t N = weight.dims[0];
        int64_t K = weight.dims[1];
        std::vector<int64_t> viewDims = {K, N};
        std::vector<int64_t> viewStrides = {1, K}; 
        
        aclDataType type = ACL_FLOAT;
        if (weight.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (weight.dataType == DataType::FLOAT32) type = ACL_FLOAT;
        else if (weight.dataType == DataType::BFLOAT16) type = ACL_BF16;
        else if (weight.dataType == DataType::INT8) type = ACL_INT8;

        aclTensor *tWeight = aclCreateTensor(viewDims.data(), viewDims.size(), type,
                                             viewStrides.data(), 0, ACL_FORMAT_ND,
                                             viewDims.data(), viewDims.size(), weight.deviceData);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        if (aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, 1, &workspaceSize, &executor) == ACL_SUCCESS) {
            aclnnMatmul(g_workspace.Get(workspaceSize), workspaceSize, executor, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
    }

    void FastllmAclQuantLinearDequant(Data &input, Data &weight, Data &weightScale, 
                                      Data &xScale, Data &bias, Data &output) {
        int64_t K = input.dims.back();
        int64_t M = input.Count(0) / K; 
        int64_t N = weight.dims[0]; 

        std::vector<int> dimInput = {(int)M, (int)K};
        std::vector<int> dimWeight = {(int)N, (int)K}; 
        std::vector<int> dimOutput = {(int)M, (int)N};
        std::vector<int> dimWScale = {(int)N};
        std::vector<int> dimXScale = {(int)M};

        aclTensor *tInput = CreateAclTensor(input, dimInput);
        aclTensor *tWeight = CreateAclTensor(weight, dimWeight);
        aclTensor *tWeightScale = CreateAclTensor(weightScale, dimWScale);
        //aclTensor *tXScale = CreateAclTensor(xScale, dimXScale);
        aclTensor *tOutput = CreateAclTensor(output, dimOutput);

        aclTensor *tXScale = nullptr;
        if (xScale.dims.size() > 0 && xScale.deviceData != nullptr) {
            tXScale = CreateAclTensor(xScale, dimXScale);
        }
        
        aclTensor *tBias = nullptr;
        if (bias.dims.size() > 0 && bias.deviceData != nullptr) {
            tBias = CreateAclTensor(bias, {bias.dims[0]});
        }

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        char mode[] = "pertoken"; 
        if (aclnnQuantMatmulDequantGetWorkspaceSize(tInput, tWeight, tWeightScale, tBias, tXScale, nullptr, nullptr,
            mode, true, tOutput, &workspaceSize, &executor) == ACL_SUCCESS) {
            aclnnQuantMatmulDequant(g_workspace.Get(workspaceSize), workspaceSize, executor, GetFastllmAclStream());
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tWeightScale);
        aclDestroyTensor(tXScale); aclDestroyTensor(tOutput); if(tBias) aclDestroyTensor(tBias);
    }

    //性能优化版本
    void FastllmAclRMSNorm(const Data &input, const Data &weight, const Data &bias, Data &output, float eps) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        std::vector<int64_t> rstdDims;
        int keepDims = input.dims.size() - weight.dims.size();
        for (int i = 0; i < (keepDims > 0 ? keepDims : 1); ++i) {
            rstdDims.push_back(keepDims > 0 ? input.dims[i] : 1);
        }

        int64_t numElem = 1; 
        for(auto d : rstdDims) numElem *= d;
        size_t rstdBytes = numElem * sizeof(float); 

        void *rstdPtr = g_workspace.Get(rstdBytes);
        printf("DEBUG: rstdPtr = %p\n", rstdPtr); // 打印地址

        std::vector<int64_t> rstdStrides(rstdDims.size());
        int64_t stride = 1;
        for (int i = rstdDims.size() - 1; i >= 0; i--) {
            rstdStrides[i] = stride; stride *= rstdDims[i];
        }
        aclTensor *tRstd = aclCreateTensor(rstdDims.data(), rstdDims.size(), ACL_FLOAT,
                                        rstdStrides.data(), 0, ACL_FORMAT_ND,
                                        rstdDims.data(), rstdDims.size(), rstdPtr);

        uint64_t opWorkspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnRmsNormGetWorkspaceSize(tInput, tWeight, (double)eps, tOutput, tRstd, &opWorkspaceSize, &executor);

        void *opWorkspaceAddr = nullptr;
        if (opWorkspaceSize > 0) {
            opWorkspaceAddr = g_workspace.Get(opWorkspaceSize);
                printf("DEBUG: opWorkspaceAddr = %p (Size: %lu)\n", opWorkspaceAddr, opWorkspaceSize); // 打印地址
        } else {
            printf("DEBUG: opWorkspaceSize is 0, lucky pass!\n");
        }

        aclnnRmsNorm(opWorkspaceAddr, opWorkspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tInput); 
        aclDestroyTensor(tWeight); 
        aclDestroyTensor(tOutput); 
        aclDestroyTensor(tRstd);
    }


    void FastllmAclSilu(const Data &input, Data &output) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnSiluGetWorkspaceSize(tInput, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnSilu(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);
    }

    void FastllmAclSwiglu(const Data &input, Data &output) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);
        int64_t dim = -1; 
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnSwiGluGetWorkspaceSize(tInput, dim, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnSwiGlu(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);
    }

    void FastllmAclSoftmax(const Data &input, Data &output, int axis) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnSoftmaxGetWorkspaceSize(tInput, (int64_t)axis, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnSoftmax(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);
    }

    //待测试
    void FastllmAclEmbedding(const Data &input, const Data &weight, Data &output) {
        aclTensor *tW = CreateAclTensor(weight, weight.dims);
        aclTensor *tO = CreateAclTensor(output, output.dims);

        int64_t elemCount = input.Count(0);
        size_t intBytes = (elemCount * sizeof(int64_t) + 255) / 256 * 256; // 256对齐

        size_t maxOpWs = 8 * 1024 * 1024; 

        uint8_t *basePtr = (uint8_t*)g_workspace.Get(intBytes + maxOpWs);
        
        void *tempIntData = basePtr;
        void *opWsAddr    = basePtr + intBytes;

        aclTensor *tI = nullptr;

        if (input.dataType == DataType::FLOAT32) {
            // === Cast Float -> Int64 ===
            aclTensor *tI_Float = CreateAclTensor(input, input.dims);

            std::vector<int64_t> dims; for(auto d : input.dims) dims.push_back(d);
            std::vector<int64_t> strides(dims.size(), 1);
            for(int i=dims.size()-2; i>=0; i--) strides[i] = dims[i+1] * strides[i+1];
            
            aclTensor *tI_Int64 = aclCreateTensor(dims.data(), dims.size(), ACL_INT64, 
                                                strides.data(), 0, ACL_FORMAT_ND, 
                                                dims.data(), dims.size(), tempIntData);

            uint64_t ws = 0; aclOpExecutor *ex = nullptr;
            if (aclnnCastGetWorkspaceSize(tI_Float, ACL_INT64, tI_Int64, &ws, &ex) == ACL_SUCCESS) {
                if (ws > maxOpWs) printf("Warning: Cast WS need %lu\n", ws);
                aclnnCast(opWsAddr, ws, ex, GetFastllmAclStream());
            }
            aclDestroyTensor(tI_Float);
            tI = tI_Int64;
        } else {
            tI = CreateAclTensor(input, input.dims);
        }

        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnEmbeddingGetWorkspaceSize(tW, tI, tO, &ws, &ex) == ACL_SUCCESS) {
            if (ws > maxOpWs) printf("Warning: Embedding WS need %lu\n", ws);
            aclnnEmbedding(opWsAddr, ws, ex, GetFastllmAclStream());
        }

        aclDestroyTensor(tW); aclDestroyTensor(tI); aclDestroyTensor(tO);
    }

    
    void FastllmAclTopK(const Data &input, Data &output, int topk) {
        int64_t k = topk;
        int64_t dim = input.dims.size() - 1; 

        std::vector<int64_t> tempDims;
        for (auto d : input.dims) tempDims.push_back((int64_t)d);
        tempDims[dim] = k;

        std::vector<int64_t> tempStrides(tempDims.size(), 1);
        for (int i = tempDims.size() - 2; i >= 0; i--) {
            tempStrides[i] = tempDims[i + 1] * tempStrides[i + 1];
        }

        int64_t elementCount = 1; 
        for (auto d : tempDims) elementCount *= d;

        size_t dtypeSize = (input.dataType == DataType::FLOAT16) ? 2 : 4;
        size_t valuesBytes = elementCount * dtypeSize;
        size_t indicesBytes = elementCount * sizeof(int64_t); // Indices 是 INT64
        size_t castBytes   = elementCount * dtypeSize;       // IndicesCast 是 FP16/FP32

        size_t totalTempBytes = valuesBytes + indicesBytes + castBytes;

        uint8_t *tempBuffer = (uint8_t*)g_workspace.Get(totalTempBytes);

        void *ptrValues = tempBuffer;
        void *ptrIndices = tempBuffer + valuesBytes;
        void *ptrIndicesCast = tempBuffer + valuesBytes + indicesBytes;

        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims); 
        
        aclDataType aclType = (input.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;

        aclTensor *tValues = aclCreateTensor(tempDims.data(), tempDims.size(), aclType, 
                                            tempStrides.data(), 0, ACL_FORMAT_ND, 
                                            tempDims.data(), tempDims.size(), ptrValues);

        aclTensor *tIndices = aclCreateTensor(tempDims.data(), tempDims.size(), ACL_INT64, 
                                            tempStrides.data(), 0, ACL_FORMAT_ND, 
                                            tempDims.data(), tempDims.size(), ptrIndices);

        aclTensor *tIndicesCast = aclCreateTensor(tempDims.data(), tempDims.size(), aclType, 
                                                tempStrides.data(), 0, ACL_FORMAT_ND, 
                                                tempDims.data(), tempDims.size(), ptrIndicesCast);

        uint64_t opWsSize = 0; 
        aclOpExecutor *executor = nullptr;
        void *opWsAddr = nullptr;

        // Step A: TopK
        if (aclnnTopkGetWorkspaceSize(tInput, k, dim, true, true, tValues, tIndices, &opWsSize, &executor) == ACL_SUCCESS) {
            if (opWsSize > 0) opWsAddr = g_workspace.Get(opWsSize);
            aclnnTopk(opWsAddr, opWsSize, executor, GetFastllmAclStream());
        }

        // Step B: Cast Indices (Int64 -> Float/Half)
        opWsSize = 0; executor = nullptr; opWsAddr = nullptr;
        if (aclnnCastGetWorkspaceSize(tIndices, aclType, tIndicesCast, &opWsSize, &executor) == ACL_SUCCESS) {
            if (opWsSize > 0) opWsAddr = g_workspace.Get(opWsSize);
            aclnnCast(opWsAddr, opWsSize, executor, GetFastllmAclStream());
        }

        // Step C: Cat [Values, Indices] -> Output
        aclTensor *concatTensors[] = {tValues, tIndicesCast};
        aclTensorList *tensorList = aclCreateTensorList(concatTensors, 2);
        
        opWsSize = 0; executor = nullptr; opWsAddr = nullptr;
        if (aclnnCatGetWorkspaceSize(tensorList, dim, tOutput, &opWsSize, &executor) == ACL_SUCCESS) {
            if (opWsSize > 0) opWsAddr = g_workspace.Get(opWsSize);
            aclnnCat(opWsAddr, opWsSize, executor, GetFastllmAclStream());
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tOutput); 
        //aclDestroyTensor(tValues); 
        aclDestroyTensor(tIndices); 
        //aclDestroyTensor(tIndicesCast);
        aclDestroyTensorList(tensorList); //对于aclTensorList内的aclTensor不需要重复释放。
    }


    void FastllmAclFloatToHalf(float *src, void *dst, int len) {
        std::vector<int> dims = {len};
        Data dSrc, dDst; 
        dSrc.dataType = DataType::FLOAT32; dSrc.deviceData = src;
        dDst.dataType = DataType::FLOAT16; dDst.deviceData = dst;
        
        aclTensor *tSrc = CreateAclTensor(dSrc, dims);
        aclTensor *tDst = CreateAclTensor(dDst, dims);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        aclnnCastGetWorkspaceSize(tSrc, ACL_FLOAT16, tDst, &ws, &ex);
        aclnnCast(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());

        aclDestroyTensor(tSrc); aclDestroyTensor(tDst);
    }

    void FastllmAclHalfToFloat(void *src, float *dst, int len) {
        std::vector<int> dims = {len};
        Data dSrc, dDst; 
        dSrc.dataType = DataType::FLOAT16; dSrc.deviceData = src;
        dDst.dataType = DataType::FLOAT32; dDst.deviceData = dst;

        aclTensor *tSrc = CreateAclTensor(dSrc, dims);
        aclTensor *tDst = CreateAclTensor(dDst, dims);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        aclnnCastGetWorkspaceSize(tSrc, ACL_FLOAT, tDst, &ws, &ex);
        aclnnCast(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());

        aclDestroyTensor(tSrc); aclDestroyTensor(tDst);
    }


    void FastllmAclAttentionMask(const Data &input, const Data &mask, float maskValue) {
        FastllmAclAddTo(const_cast<Data&>(input), mask, maskValue);
    }

    void FastllmAclAttention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output, int group, float scale, int maskType) {
        Data score;
        std::vector<int> scoreDims = q.dims;
        scoreDims.back() = k.dims[k.dims.size() - 2];
        score.dataType = q.dataType;
        score.Resize(scoreDims);
        score.deviceData = g_workspace.Get(score.GetBytes());

        FastllmAclMatMulTransB(q, k, Data(), score, 1, 0);
        if (std::abs(scale - 1.0f) > 1e-6) FastllmAclMul(score, scale, score);
        if (mask.dims.size() > 0) FastllmAclAddTo(score, mask, -10000.0f);
        FastllmAclSoftmax(score, score, -1);
        FastllmAclMatMul(score, v, Data(), output, 1, 0);
    }

    //性能优化版 需要详细分析
    void FastllmAclRepeat(void *src, void *dst, int outer, int repeatTimes, int inputStride, int outputStride, int channelsInner, int channelsInputInner) {

        std::vector<int64_t> selfShape = { (int64_t)outer, 1, (int64_t)channelsInputInner };
        std::vector<int64_t> outShape  = { (int64_t)outer, (int64_t)repeatTimes, (int64_t)channelsInputInner };

        std::vector<int64_t> selfStrides = { (int64_t)inputStride, (int64_t)channelsInputInner, 1 };

        std::vector<int64_t> outStrides = { (int64_t)outputStride, (int64_t)channelsInner, 1 };

        aclTensor *tSelf = aclCreateTensor(selfShape.data(), selfShape.size(), ACL_UINT8, 
                                        selfStrides.data(), 0, ACL_FORMAT_ND, 
                                        selfShape.data(), selfShape.size(), src);

        aclTensor *tOut = aclCreateTensor(outShape.data(), outShape.size(), ACL_UINT8, 
                                        outStrides.data(), 0, ACL_FORMAT_ND, 
                                        outShape.data(), outShape.size(), dst);

        aclIntArray *expandSize = aclCreateIntArray(outShape.data(), outShape.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        if (aclnnExpandGetWorkspaceSize(tSelf, expandSize, tOut, &workspaceSize, &executor) == ACL_SUCCESS) {
            void *workspaceAddr = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnExpand(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());
        }

        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOut);
        aclDestroyIntArray(expandSize);
    }


    void FastllmAclAdd(const Data &input, float v, Data &output) {
        aclTensor *tSelf = CreateAclTensor(input, input.dims);
        aclTensor *tOut = CreateAclTensor(output, output.dims);
        aclScalar *sOther = aclCreateScalar(&v, ACL_FLOAT);
        float alphaVal = 1.0f;
        aclScalar *sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnAddsGetWorkspaceSize(tSelf, sOther, sAlpha, tOut, &workspaceSize, &executor);
        
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) workspaceAddr = g_workspace.Get(workspaceSize);
    
        aclnnAdds(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tSelf); aclDestroyTensor(tOut);
        aclDestroyScalar(sOther); aclDestroyScalar(sAlpha);
    }


    void FastllmAclAddTo(const Data &input0, const Data &input1, float alpha) {
 
        aclTensor *tSelf = CreateAclTensor(input0, input0.dims);
        aclTensor *tOther = CreateAclTensor(input1, input1.dims);

        aclScalar *sAlpha = aclCreateScalar(&alpha, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnAddGetWorkspaceSize(tSelf, tOther, sAlpha, tSelf, &workspaceSize, &executor);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }
    
        aclnnAdd(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());


        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOther);
        aclDestroyScalar(sAlpha);
    }


    void FastllmAclMul(const Data &input, float v, Data &output) {
        aclTensor *tSelf = CreateAclTensor(input, input.dims);
        aclTensor *tOut = CreateAclTensor(output, output.dims);

        aclScalar *sOther = aclCreateScalar(&v, ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnMulsGetWorkspaceSize(tSelf, sOther, tOut, &workspaceSize, &executor);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }
    
        aclnnMuls(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOut);
        aclDestroyScalar(sOther);
    }
    

    void FastllmAclMulTo(const Data &input0, const Data &input1, float alpha) {

        aclTensor *tSelf = CreateAclTensor(input0, input0.dims); 
        aclTensor *tOther = CreateAclTensor(input1, input1.dims);

        uint64_t wsMul = 0;
        aclOpExecutor *exMul = nullptr;
        aclnnMulGetWorkspaceSize(tSelf, tOther, tSelf, &wsMul, &exMul);
    
        void *wsAddrMul = nullptr;
        if (wsMul > 0) wsAddrMul = g_workspace.Get(wsMul);
    
        aclnnMul(wsAddrMul, wsMul, exMul, GetFastllmAclStream());

        if (std::abs(alpha - 1.0f) > 1e-6) {
            aclScalar *sAlpha = aclCreateScalar(&alpha, ACL_FLOAT);
            uint64_t wsScale = 0;
            aclOpExecutor *exScale = nullptr;

            aclnnInplaceMulsGetWorkspaceSize(tSelf, sAlpha, &wsScale, &exScale);
    
            void *wsAddrScale = nullptr;
            if (wsScale > 0) wsAddrScale = g_workspace.Get(wsScale);
    
            aclnnInplaceMuls(wsAddrScale, wsScale, exScale, GetFastllmAclStream());
            
            aclDestroyScalar(sAlpha);
        }
    
        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOther);
    }

    void FastllmAclPermute(const Data &input, const std::vector<int> &axis) {
        Data &mutableInput = const_cast<Data&>(input);

        std::vector<int64_t> axisInt64;
        std::vector<int64_t> newDims;
        for (int i : axis) {
            axisInt64.push_back((int64_t)i);
            newDims.push_back(mutableInput.dims[i]);
        }
        aclIntArray *permArray = aclCreateIntArray(axisInt64.data(), axisInt64.size());

        size_t dataBytes = mutableInput.GetBytes();

        void* tempPtr = g_workspace.Get(dataBytes); 

        aclTensor *tSelf = CreateAclTensor(mutableInput, mutableInput.dims);

        std::vector<int64_t> outStrides(newDims.size(), 1);
        for (int i = newDims.size() - 2; i >= 0; i--) {
            outStrides[i] = newDims[i + 1] * outStrides[i + 1];
        }
        
        aclTensor *tOut = aclCreateTensor(newDims.data(), newDims.size(), ACL_FLOAT16, 
                                        outStrides.data(), 0, ACL_FORMAT_ND, 
                                        newDims.data(), newDims.size(), tempPtr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        
        aclnnPermuteGetWorkspaceSize(tSelf, permArray, tOut, &workspaceSize, &executor);
        
        void *opWorkspaceAddr = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
        
        aclnnPermute(opWorkspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        //支持指针交换（Pointer Swapping）且拥有一个持久化内存池 后可优化
        aclrtMemcpyAsync(mutableInput.deviceData, dataBytes, 
                        tempPtr, dataBytes, 
                        ACL_MEMCPY_DEVICE_TO_DEVICE, GetFastllmAclStream());

        mutableInput.Resize(std::vector<int>(newDims.begin(), newDims.end()));

        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOut);
        aclDestroyIntArray(permArray);
    }

    void FastllmAclNearlyRotatePosition2D(const Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        aclTensor *tQuery = CreateAclTensor(data, data.dims);
        aclTensor *tCos = CreateAclTensor(cosData, cosData.dims);
        aclTensor *tSin = CreateAclTensor(sinData, sinData.dims);

        aclTensor *tKey = nullptr; 

        int64_t layout = 1; 
        char *rotaryCoeff = (char*)"half"; 

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnApplyRotaryPosEmbV2GetWorkspaceSize(tQuery, tKey, tCos, tSin, layout, rotaryCoeff, &workspaceSize, &executor);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        aclnnApplyRotaryPosEmbV2(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tQuery);
        aclDestroyTensor(tCos);
        aclDestroyTensor(tSin);
    }

    void FastllmAclRotatePosition2D_Fused(const Data &query, const Data &key, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        aclTensor *tQuery = CreateAclTensor(query, query.dims);
        aclTensor *tKey = CreateAclTensor(key, key.dims); // 新增 Key
        aclTensor *tCos = CreateAclTensor(cosData, cosData.dims);
        aclTensor *tSin = CreateAclTensor(sinData, sinData.dims);

        int64_t layout = 1; 
        char *rotaryCoeff = (char*)"half"; 

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclnnApplyRotaryPosEmbV2GetWorkspaceSize(tQuery, tKey, tCos, tSin, layout, rotaryCoeff, &workspaceSize, &executor);

        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        aclnnApplyRotaryPosEmbV2(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tQuery);
        aclDestroyTensor(tKey);
        aclDestroyTensor(tCos);
        aclDestroyTensor(tSin);
    }

}