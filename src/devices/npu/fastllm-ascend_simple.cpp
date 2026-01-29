#include <vector>
#include <mutex>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm> // for std::max

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

    // ==========================================
    // 1. 高性能线性显存分配器 (修复了初始化顺序和内存覆盖问题)
    // ==========================================
    struct NpuWorkspace {
        void* basePtr = nullptr;      
        size_t capacity = 0;          
        size_t currentOffset = 0;     
        std::mutex mtx; 

        // 默认预分配大小：256MB
        const size_t DEFAULT_POOL_SIZE = 256 * 1024 * 1024; 

        // 构造函数什么都不做，支持延迟初始化
        NpuWorkspace() {}

        void* Get(size_t size) {
            std::lock_guard<std::mutex> lock(mtx);
            
            // 延迟初始化：第一次调用时才申请内存，确保 aclInit 已完成
            if (basePtr == nullptr) {
                aclError ret = aclrtMalloc(&basePtr, DEFAULT_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("CRITICAL ERROR: NpuWorkspace init failed. Code: %d\n", ret);
                    return nullptr;
                }
                capacity = DEFAULT_POOL_SIZE;
            }

            // 32字节对齐
            size_t alignSize = (size + 31) / 32 * 32;

            // 检查剩余空间与扩容
            if (currentOffset + alignSize > capacity) {
                printf("WARNING: NpuWorkspace expanding! Old: %zu MB, Needed: %zu MB\n", 
                       capacity/1024/1024, (currentOffset + alignSize)/1024/1024);
                
                if (basePtr) aclrtFree(basePtr);
                
                size_t newCapacity = std::max(capacity * 2, currentOffset + alignSize + 64 * 1024 * 1024);
                aclError ret = aclrtMalloc(&basePtr, newCapacity, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("CRITICAL ERROR: NpuWorkspace Expand Failed!\n");
                    return nullptr;
                }
                capacity = newCapacity;
                currentOffset = 0; 
            }

            // 返回当前偏移地址
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

    // ==========================================
    // 2. 基础辅助函数
    // ==========================================
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
            aclrtFree(ptr);
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

    // 原始 Tensor 创建函数
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

    // ==========================================
    // 3. 通用执行模板 RunAclnnOp
    // ==========================================
    template<typename FuncGetSize, typename FuncRun, typename... Args>
    void RunAclnnOp(FuncGetSize getSizeFunc, FuncRun runFunc, Args&&... args) {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        
        // 第一阶段：获取大小
        // 使用 std::forward 完美转发参数，保持引用属性，不触发拷贝
        aclError ret = getSizeFunc(std::forward<Args>(args)..., &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            printf("Aclnn GetWorkspaceSize Failed! Code: %d\n", ret);
            return;
        }

        // 第二阶段：申请内存
        void *workspaceAddr = nullptr;
        if (workspaceSize > 0) {
            workspaceAddr = g_workspace.Get(workspaceSize);
        }

        // 第三阶段：执行
        ret = runFunc(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());
        if (ret != ACL_SUCCESS) {
            printf("Aclnn Run Failed! Code: %d\n", ret);
        }
    }

    // ==========================================
    // 4. RAII 资源管理类 ScopedAclTensor
    // ==========================================
    struct ScopedAclTensor {
        aclTensor* ptr = nullptr;
        
        // 构造方式1：通过 Data 创建
        ScopedAclTensor(const Data &data, const std::vector<int> &dims, void* customDevPtr = nullptr) {
            ptr = CreateAclTensor(data, dims, customDevPtr);
        }
        
        // 构造方式2：接管已有的 raw aclTensor*
        ScopedAclTensor(aclTensor* raw) : ptr(raw) {}

        // 禁止拷贝，允许移动 (略) - 这里简化为禁用拷贝
        ScopedAclTensor(const ScopedAclTensor&) = delete;
        ScopedAclTensor& operator=(const ScopedAclTensor&) = delete;

        ~ScopedAclTensor() {
            if (ptr) aclDestroyTensor(ptr);
        }

        // 隐式转换
        operator aclTensor*() const { return ptr; }
        aclTensor* get() const { return ptr; }
    };

    struct ScopedAclScalar {
        aclScalar* ptr = nullptr;
        ScopedAclScalar(float val) { ptr = aclCreateScalar(&val, ACL_FLOAT); }
        ~ScopedAclScalar() { if(ptr) aclDestroyScalar(ptr); }
        operator aclScalar*() const { return ptr; }
    };

    struct ScopedAclIntArray {
        aclIntArray* ptr = nullptr;
        ScopedAclIntArray(const std::vector<int64_t>& data) {
             ptr = aclCreateIntArray(data.data(), data.size());
        }
        ~ScopedAclIntArray() { if(ptr) aclDestroyIntArray(ptr); }
        operator aclIntArray*() const { return ptr; }
    };

    // ==========================================
    // 5. 重构后的算子实现
    // ==========================================

    void FastllmAclMatMul(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tWeight(weight, weight.dims);
        ScopedAclTensor tOutput(output, output.dims);
        int8_t cubeMathType = 1; 

        RunAclnnOp(aclnnMatmulGetWorkspaceSize, aclnnMatmul, 
                   tInput, tWeight, tOutput, cubeMathType);
    }

    void FastllmAclMatMulTransB(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tOutput(output, output.dims);

        int64_t N = weight.dims[0];
        int64_t K = weight.dims[1];
        std::vector<int64_t> viewDims = {K, N};
        std::vector<int64_t> viewStrides = {1, K}; 
        
        aclDataType type = ACL_FLOAT;
        if (weight.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (weight.dataType == DataType::FLOAT32) type = ACL_FLOAT;
        else if (weight.dataType == DataType::BFLOAT16) type = ACL_BF16;
        else if (weight.dataType == DataType::INT8) type = ACL_INT8;

        // 手动创建 Transposed Weight，交给 Scoped 管理
        ScopedAclTensor tWeight(aclCreateTensor(viewDims.data(), viewDims.size(), type,
                                                viewStrides.data(), 0, ACL_FORMAT_ND,
                                                viewDims.data(), viewDims.size(), weight.deviceData));

        RunAclnnOp(aclnnMatmulGetWorkspaceSize, aclnnMatmul, 
                   tInput, tWeight, tOutput, 1);
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

        ScopedAclTensor tInput(input, dimInput);
        ScopedAclTensor tWeight(weight, dimWeight);
        ScopedAclTensor tWeightScale(weightScale, dimWScale);
        ScopedAclTensor tOutput(output, dimOutput);

        // 可选输入，手动管理生命周期或使用裸指针（RunAclnnOp 支持 nullptr）
        aclTensor *tXScale = nullptr;
        if (xScale.dims.size() > 0 && xScale.deviceData != nullptr) {
            tXScale = CreateAclTensor(xScale, dimXScale);
        }
        
        aclTensor *tBias = nullptr;
        if (bias.dims.size() > 0 && bias.deviceData != nullptr) {
            tBias = CreateAclTensor(bias, {bias.dims[0]});
        }

        char mode[] = "pertoken"; 
        
        RunAclnnOp(aclnnQuantMatmulDequantGetWorkspaceSize, aclnnQuantMatmulDequant,
                   tInput, tWeight, tWeightScale, tBias, tXScale, nullptr, nullptr,
                   mode, true, tOutput);

        if(tXScale) aclDestroyTensor(tXScale);
        if(tBias) aclDestroyTensor(tBias);
    }

    void FastllmAclRMSNorm(const Data &input, const Data &weight, const Data &bias, Data &output, float eps) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tWeight(weight, weight.dims);
        ScopedAclTensor tOutput(output, output.dims);

        std::vector<int64_t> rstdDims;
        int keepDims = input.dims.size() - weight.dims.size();
        for (int i = 0; i < (keepDims > 0 ? keepDims : 1); ++i) {
            rstdDims.push_back(keepDims > 0 ? input.dims[i] : 1);
        }

        int64_t numElem = 1; 
        for(auto d : rstdDims) numElem *= d;
        size_t rstdBytes = numElem * sizeof(float); 

        // 第一次 Get: Rstd 结果
        void *rstdPtr = g_workspace.Get(rstdBytes);

        std::vector<int64_t> rstdStrides(rstdDims.size());
        int64_t stride = 1;
        for (int i = rstdDims.size() - 1; i >= 0; i--) {
            rstdStrides[i] = stride; stride *= rstdDims[i];
        }
        
        ScopedAclTensor tRstd(aclCreateTensor(rstdDims.data(), rstdDims.size(), ACL_FLOAT,
                                              rstdStrides.data(), 0, ACL_FORMAT_ND,
                                              rstdDims.data(), rstdDims.size(), rstdPtr));

        // 第二次 Get (隐含在 RunAclnnOp 中): Op Workspace
        // 由于是线性分配，地址安全
        RunAclnnOp(aclnnRmsNormGetWorkspaceSize, aclnnRmsNorm, 
                   tInput, tWeight, (double)eps, tOutput, tRstd);
    }


    void FastllmAclSilu(const Data &input, Data &output) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tOutput(output, output.dims);
        RunAclnnOp(aclnnSiluGetWorkspaceSize, aclnnSilu, tInput, tOutput);
    }

    void FastllmAclSwiglu(const Data &input, Data &output) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tOutput(output, output.dims);
        int64_t dim = -1; 
        RunAclnnOp(aclnnSwiGluGetWorkspaceSize, aclnnSwiGlu, tInput, dim, tOutput);
    }

    void FastllmAclSoftmax(const Data &input, Data &output, int axis) {
        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tOutput(output, output.dims);
        RunAclnnOp(aclnnSoftmaxGetWorkspaceSize, aclnnSoftmax, tInput, (int64_t)axis, tOutput);
    }

    void FastllmAclEmbedding(const Data &input, const Data &weight, Data &output) {
        ScopedAclTensor tW(weight, weight.dims);
        ScopedAclTensor tI(input, input.dims);
        ScopedAclTensor tO(output, output.dims);
        RunAclnnOp(aclnnEmbeddingGetWorkspaceSize, aclnnEmbedding, tW, tI, tO);
    }

    void FastllmAclTopK(const Data &input, Data &output, int topk) {
        int64_t k = topk;
        int64_t dim = input.dims.size() - 1; 

        // 修复：vector<int> -> vector<int64_t>
        std::vector<int64_t> tempDims(input.dims.begin(), input.dims.end());
        tempDims[dim] = k;

        std::vector<int64_t> tempStrides(tempDims.size(), 1);
        for (int i = tempDims.size() - 2; i >= 0; i--) {
            tempStrides[i] = tempDims[i + 1] * tempStrides[i + 1];
        }

        int64_t elementCount = 1; 
        for (auto d : tempDims) elementCount *= d;

        size_t dtypeSize = (input.dataType == DataType::FLOAT16) ? 2 : 4;
        size_t valuesBytes = elementCount * dtypeSize;
        size_t indicesBytes = elementCount * sizeof(int64_t); 
        size_t castBytes   = elementCount * dtypeSize;      

        size_t totalTempBytes = valuesBytes + indicesBytes + castBytes;

        // 线性分配中间 Buffer
        uint8_t *tempBuffer = (uint8_t*)g_workspace.Get(totalTempBytes);

        void *ptrValues = tempBuffer;
        void *ptrIndices = tempBuffer + valuesBytes;
        void *ptrIndicesCast = tempBuffer + valuesBytes + indicesBytes;

        ScopedAclTensor tInput(input, input.dims);
        ScopedAclTensor tOutput(output, output.dims); 
        
        aclDataType aclType = (input.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;

        ScopedAclTensor tValues(aclCreateTensor(tempDims.data(), tempDims.size(), aclType, 
                                                tempStrides.data(), 0, ACL_FORMAT_ND, 
                                                tempDims.data(), tempDims.size(), ptrValues));

        ScopedAclTensor tIndices(aclCreateTensor(tempDims.data(), tempDims.size(), ACL_INT64, 
                                                 tempStrides.data(), 0, ACL_FORMAT_ND, 
                                                 tempDims.data(), tempDims.size(), ptrIndices));

        ScopedAclTensor tIndicesCast(aclCreateTensor(tempDims.data(), tempDims.size(), aclType, 
                                                     tempStrides.data(), 0, ACL_FORMAT_ND, 
                                                     tempDims.data(), tempDims.size(), ptrIndicesCast));

        // Step A: TopK
        RunAclnnOp(aclnnTopkGetWorkspaceSize, aclnnTopk, tInput, k, dim, true, true, tValues, tIndices);

        // Step B: Cast Indices
        RunAclnnOp(aclnnCastGetWorkspaceSize, aclnnCast, tIndices, aclType, tIndicesCast);

        // Step C: Cat
        aclTensor *concatTensors[] = {tValues, tIndicesCast};
        aclTensorList *tensorList = aclCreateTensorList(concatTensors, 2);
        
        RunAclnnOp(aclnnCatGetWorkspaceSize, aclnnCat, tensorList, dim, tOutput);

        aclDestroyTensorList(tensorList);
    }

    void FastllmAclFloatToHalf(float *src, void *dst, int len) {
        std::vector<int> dims = {len};
        Data dSrc, dDst; 
        dSrc.dataType = DataType::FLOAT32; dSrc.deviceData = src;
        dDst.dataType = DataType::FLOAT16; dDst.deviceData = dst;
        
        ScopedAclTensor tSrc(dSrc, dims);
        ScopedAclTensor tDst(dDst, dims);
        RunAclnnOp(aclnnCastGetWorkspaceSize, aclnnCast, tSrc, ACL_FLOAT16, tDst);
    }

    void FastllmAclHalfToFloat(void *src, float *dst, int len) {
        std::vector<int> dims = {len};
        Data dSrc, dDst; 
        dSrc.dataType = DataType::FLOAT16; dSrc.deviceData = src;
        dDst.dataType = DataType::FLOAT32; dDst.deviceData = dst;

        ScopedAclTensor tSrc(dSrc, dims);
        ScopedAclTensor tDst(dDst, dims);
        RunAclnnOp(aclnnCastGetWorkspaceSize, aclnnCast, tSrc, ACL_FLOAT, tDst);
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

    void FastllmAclRepeat(void *src, void *dst, int outer, int repeatTimes, int inputStride, int outputStride, int channelsInner, int channelsInputInner) {
        std::vector<int64_t> selfShape = { (int64_t)outer, 1, (int64_t)channelsInputInner };
        std::vector<int64_t> outShape  = { (int64_t)outer, (int64_t)repeatTimes, (int64_t)channelsInputInner };

        std::vector<int64_t> selfStrides = { (int64_t)inputStride, (int64_t)channelsInputInner, 1 };
        std::vector<int64_t> outStrides = { (int64_t)outputStride, (int64_t)channelsInner, 1 };

        ScopedAclTensor tSelf(aclCreateTensor(selfShape.data(), selfShape.size(), ACL_UINT8, 
                                              selfStrides.data(), 0, ACL_FORMAT_ND, 
                                              selfShape.data(), selfShape.size(), src));

        ScopedAclTensor tOut(aclCreateTensor(outShape.data(), outShape.size(), ACL_UINT8, 
                                             outStrides.data(), 0, ACL_FORMAT_ND, 
                                             outShape.data(), outShape.size(), dst));

        ScopedAclIntArray expandSize(outShape);
        RunAclnnOp(aclnnExpandGetWorkspaceSize, aclnnExpand, tSelf, expandSize, tOut);
    }

    void FastllmAclAdd(const Data &input, float v, Data &output) {
        ScopedAclTensor tSelf(input, input.dims);
        ScopedAclTensor tOut(output, output.dims);
        ScopedAclScalar sOther(v);
        ScopedAclScalar sAlpha(1.0f);
        RunAclnnOp(aclnnAddsGetWorkspaceSize, aclnnAdds, tSelf, sOther, sAlpha, tOut);
    }

    void FastllmAclAddTo(const Data &input0, const Data &input1, float alpha) {
        ScopedAclTensor tSelf(input0, input0.dims);
        ScopedAclTensor tOther(input1, input1.dims);
        ScopedAclScalar sAlpha(alpha);
        RunAclnnOp(aclnnAddGetWorkspaceSize, aclnnAdd, tSelf, tOther, sAlpha, tSelf);
    }

    void FastllmAclMul(const Data &input, float v, Data &output) {
        ScopedAclTensor tSelf(input, input.dims);
        ScopedAclTensor tOut(output, output.dims);
        ScopedAclScalar sOther(v);
        RunAclnnOp(aclnnMulsGetWorkspaceSize, aclnnMuls, tSelf, sOther, tOut);
    }
    
    void FastllmAclMulTo(const Data &input0, const Data &input1, float alpha) {
        ScopedAclTensor tSelf(input0, input0.dims); 
        ScopedAclTensor tOther(input1, input1.dims);

        RunAclnnOp(aclnnMulGetWorkspaceSize, aclnnMul, tSelf, tOther, tSelf);

        if (std::abs(alpha - 1.0f) > 1e-6) {
            ScopedAclScalar sAlpha(alpha);
            RunAclnnOp(aclnnInplaceMulsGetWorkspaceSize, aclnnInplaceMuls, tSelf, sAlpha);
        }
    }

    void FastllmAclPermute(const Data &input, const std::vector<int> &axis) {
        Data &mutableInput = const_cast<Data&>(input);

        std::vector<int64_t> axisInt64;
        std::vector<int64_t> newDims;
        for (int i : axis) {
            axisInt64.push_back((int64_t)i);
            newDims.push_back(mutableInput.dims[i]);
        }
        ScopedAclIntArray permArray(axisInt64);

        size_t dataBytes = mutableInput.GetBytes();
        void* tempPtr = g_workspace.Get(dataBytes); 

        ScopedAclTensor tSelf(mutableInput, mutableInput.dims);

        std::vector<int64_t> outStrides(newDims.size(), 1);
        for (int i = newDims.size() - 2; i >= 0; i--) {
            outStrides[i] = newDims[i + 1] * outStrides[i + 1];
        }
        
        // 动态判断类型
        aclDataType type = (mutableInput.dataType == DataType::FLOAT32) ? ACL_FLOAT : ACL_FLOAT16;

        ScopedAclTensor tOut(aclCreateTensor(newDims.data(), newDims.size(), type, 
                                             outStrides.data(), 0, ACL_FORMAT_ND, 
                                             newDims.data(), newDims.size(), tempPtr));

        RunAclnnOp(aclnnPermuteGetWorkspaceSize, aclnnPermute, tSelf, permArray, tOut);

        aclrtMemcpyAsync(mutableInput.deviceData, dataBytes, 
                         tempPtr, dataBytes, 
                         ACL_MEMCPY_DEVICE_TO_DEVICE, GetFastllmAclStream());

        mutableInput.Resize(std::vector<int>(newDims.begin(), newDims.end()));
    }

    void FastllmAclNearlyRotatePosition2D(const Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        ScopedAclTensor tQuery(data, data.dims);
        ScopedAclTensor tCos(cosData, cosData.dims);
        ScopedAclTensor tSin(sinData, sinData.dims);
        aclTensor *tKey = nullptr; 
        int64_t layout = 1; 
        char *rotaryCoeff = (char*)"half"; 

        RunAclnnOp(aclnnApplyRotaryPosEmbV2GetWorkspaceSize, aclnnApplyRotaryPosEmbV2, 
                   tQuery, tKey, tCos, tSin, layout, rotaryCoeff);
    }

    void FastllmAclRotatePosition2D_Fused(const Data &query, const Data &key, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        ScopedAclTensor tQuery(query, query.dims);
        ScopedAclTensor tKey(key, key.dims); 
        ScopedAclTensor tCos(cosData, cosData.dims);
        ScopedAclTensor tSin(sinData, sinData.dims);

        int64_t layout = 1; 
        char *rotaryCoeff = (char*)"half"; 

        RunAclnnOp(aclnnApplyRotaryPosEmbV2GetWorkspaceSize, aclnnApplyRotaryPosEmbV2, 
                   tQuery, tKey, tCos, tSin, layout, rotaryCoeff);
    }

}