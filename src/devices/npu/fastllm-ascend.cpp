/**
 * @file fastllm-ascend.cpp
 * @brief FastLLM Ascend NPU (CANN) Backend Implementation
 * @details Supports Ascend 310P/910 using a mix of aclnn (OpAPI) and aclop (Legacy) interfaces.
 * Optimized for W8A8 quantization and dynamic shape inference.
 * * Key Features:
 * - Hybrid Backend: Uses aclnn for performance (MatMul, Softmax) and aclop for compatibility (Add, Permute).
 * - Memory Pool: Zero-overhead NpuWorkspace for intermediate tensor calculations.
 * - Quantization: Native support for W8A8 per-token quantization.
 */
 #include <vector>
 #include <mutex>
 #include <iostream>
 #include <cmath>
 #include <cstring>
 #include <cstdio>

 #include "fastllm-ascend.h"
 #include "acl/acl_op.h" // Legacy Ops (Required for 310P compatibility)
 #include "acl/acl.h"
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
                    printf("NPU Workspace Malloc Failed! Size: %zu\n", alloc_size);
                    return nullptr;
                }
                capacity = alloc_size;
            }
            return ptr;
        }
        ~NpuWorkspace() { if (ptr) aclrtFree(ptr); }
    } g_workspace;

    struct DeviceScalar {
        void* ptr = nullptr;
        aclTensorDesc* desc = nullptr;
        aclDataBuffer* buf = nullptr;

        DeviceScalar(float val) {
            aclrtMalloc(&ptr, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(ptr, sizeof(float), &val, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            std::vector<int64_t> dims = {1};
            desc = aclCreateTensorDesc(ACL_FLOAT, 1, dims.data(), ACL_FORMAT_ND);
            buf = aclCreateDataBuffer(ptr, sizeof(float));
        }
        ~DeviceScalar() {
            if (desc) aclDestroyTensorDesc(desc);
            if (buf) aclDestroyDataBuffer(buf);
            if (ptr) aclrtFree(ptr);
        }
    };

    aclrtStream GetFastllmAclStream() { return g_aclStream; }

    void FastllmAclInit() {
        if (g_isInitialized) return;

        int32_t deviceId = 0; 
        auto ret = aclInit(nullptr);
        // Allow repeat initialize (e.g. embedding in other apps)
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
        // Use HUGE_FIRST for better performance on large models
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

    // =======================================================================
    // Helper: CreateAclTensor
    // 修正：直接接受 vector<int>，内部转为 vector<int64_t>
    // =======================================================================
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

    aclTensor* CreateAclTensorRaw(const std::vector<int64_t> &dims, aclDataType type, void* devPtr) {
        std::vector<int64_t> strides(dims.size());
        int64_t stride = 1;
        for (int i = dims.size() - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= dims[i];
        }
        return aclCreateTensor(dims.data(), dims.size(), type,
                               strides.data(), 0, ACL_FORMAT_ND,
                               dims.data(), dims.size(), devPtr);
    }

    // =======================================================================
    // aclnn Implementation (OpAPI)
    // =======================================================================

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

    /**
    * @brief RMSNorm: Y = X * W / Sqrt(Mean(X^2) + eps)
    */
    void FastllmAclRMSNorm(const Data &input, const Data &weight, const Data &bias, Data &output, float eps) {
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims);

        // Prepare rstdOut (Reciprocal Standard Deviation) - Mandatory for aclnn
        std::vector<int64_t> rstdDims;
        int keepDims = input.dims.size() - weight.dims.size();
        for (int i = 0; i < (keepDims > 0 ? keepDims : 1); ++i) 
            rstdDims.push_back(keepDims > 0 ? input.dims[i] : 1);

        int64_t numElem = 1; for(auto d : rstdDims) numElem *= d;
        
        void *rstdPtr = nullptr;
        aclrtMalloc(&rstdPtr, numElem * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        
        std::vector<int64_t> rstdStrides(rstdDims.size());
        int64_t stride = 1;
        for (int i = rstdDims.size() - 1; i >= 0; i--) {
            rstdStrides[i] = stride; stride *= rstdDims[i];
        }

        aclTensor *tRstd = aclCreateTensor(rstdDims.data(), rstdDims.size(), ACL_FLOAT,
                                        rstdStrides.data(), 0, ACL_FORMAT_ND,
                                        rstdDims.data(), rstdDims.size(), rstdPtr);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        auto ret = aclnnRmsNormGetWorkspaceSize(tInput, tWeight, (double)eps, tOutput, tRstd, &workspaceSize, &executor);

        if (ret == ACL_SUCCESS) {
            void *ws = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnRmsNorm(ws, workspaceSize, executor, GetFastllmAclStream());
        } else {
            printf("Error: aclnnRmsNorm failed. Code: %d\n", ret);
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput); aclDestroyTensor(tRstd);
        
        // Sync required before freeing rstdPtr (as it is not managed by workspace pool)
        aclrtSynchronizeStream(GetFastllmAclStream()); 
        aclrtFree(rstdPtr);
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

    void FastllmAclEmbedding(const Data &input, const Data &weight, Data &output) {
        aclTensor *tW = CreateAclTensor(weight, weight.dims);
        aclTensor *tI = CreateAclTensor(input, input.dims);
        aclTensor *tO = CreateAclTensor(output, output.dims);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnEmbeddingGetWorkspaceSize(tW, tI, tO, &ws, &ex) == ACL_SUCCESS) {
            aclnnEmbedding(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tW); aclDestroyTensor(tI); aclDestroyTensor(tO);
    }

    void FastllmAclTopK(const Data &input, Data &output, int topk) {
        int64_t k = topk;
        int64_t dim = input.dims.size() - 1;

        std::vector<int64_t> tempDims;
        for(int d : input.dims) tempDims.push_back(d);
        tempDims[dim] = k;

        int64_t elementCount = 1; for (auto d : tempDims) elementCount *= d;
        size_t dtypeSize = (input.dataType == DataType::FLOAT16) ? 2 : 4;
        
        uint8_t *tempBuffer = nullptr;
        aclrtMalloc((void**)&tempBuffer, elementCount * (dtypeSize + 8 + dtypeSize), ACL_MEM_MALLOC_HUGE_FIRST);
        
        aclTensor *tInput = CreateAclTensor(input, input.dims);
        aclTensor *tOutput = CreateAclTensor(output, output.dims); 
        aclDataType aclType = (input.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;

        aclTensor *tValues = CreateAclTensorRaw(tempDims, aclType, tempBuffer);
        aclTensor *tIndices = CreateAclTensorRaw(tempDims, ACL_INT64, tempBuffer + elementCount * dtypeSize);
        aclTensor *tIndicesCast = CreateAclTensorRaw(tempDims, aclType, tempBuffer + elementCount * (dtypeSize + 8));

        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnTopkGetWorkspaceSize(tInput, k, dim, true, true, tValues, tIndices, &ws, &ex) == ACL_SUCCESS) {
            aclnnTopk(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        if (aclnnCastGetWorkspaceSize(tIndices, aclType, tIndicesCast, &ws, &ex) == ACL_SUCCESS) {
            aclnnCast(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclTensor *concatTensors[] = {tValues, tIndicesCast};
        aclTensorList *tensorList = aclCreateTensorList(concatTensors, 2);
        if (aclnnCatGetWorkspaceSize(tensorList, dim, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnCat(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }

        aclrtSynchronizeStream(GetFastllmAclStream());
        aclrtFree(tempBuffer);
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput); aclDestroyTensor(tValues);
        aclDestroyTensor(tIndices); aclDestroyTensor(tIndicesCast); aclDestroyTensorList(tensorList);
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
        aclrtSynchronizeStream(GetFastllmAclStream());
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
        aclrtSynchronizeStream(GetFastllmAclStream());
        aclDestroyTensor(tSrc); aclDestroyTensor(tDst);
    }

    // =======================================================================
    // aclop Implementation (Legacy/Basic Ops)
    // =======================================================================

    void CreateAclOpTensor(const Data &data, const std::vector<int64_t>& dims, aclTensorDesc** desc, aclDataBuffer** buf) {
        aclDataType type = ACL_FLOAT;
        if (data.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (data.dataType == DataType::FLOAT32) type = ACL_FLOAT;

        *desc = aclCreateTensorDesc(type, dims.size(), dims.data(), ACL_FORMAT_ND);
        *buf = aclCreateDataBuffer(data.deviceData, data.GetBytes());
    }
    
    std::vector<int64_t> ToDims64(const std::vector<int>& dims) {
        std::vector<int64_t> ret;
        for(int d : dims) ret.push_back(d);
        return ret;
    }

    // Basic arithmetic can stick to aclop as it is very stable
    void FastllmAclAdd(const Data &input, float v, Data &output) {
        aclTensorDesc *dIn, *dOut; aclDataBuffer *bIn, *bOut;
        CreateAclOpTensor(input, ToDims64(input.dims), &dIn, &bIn);
        CreateAclOpTensor(output, ToDims64(output.dims), &dOut, &bOut);
        
        DeviceScalar scalar(v);
        std::vector<aclTensorDesc*> inD = {dIn, scalar.desc};
        std::vector<aclDataBuffer*> inB = {bIn, scalar.buf};
        std::vector<aclTensorDesc*> outD = {dOut};
        std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("Add", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       nullptr, GetFastllmAclStream());
        
        aclrtSynchronizeStream(GetFastllmAclStream()); 
        aclDestroyTensorDesc(dIn); aclDestroyDataBuffer(bIn);
        aclDestroyTensorDesc(dOut); aclDestroyDataBuffer(bOut);
    }

    void FastllmAclAddTo(const Data &input0, const Data &input1, float alpha) {
        aclTensorDesc *dX, *dY, *dOut; aclDataBuffer *bX, *bY, *bOut;
        CreateAclOpTensor(input1, ToDims64(input1.dims), &dX, &bX); 
        CreateAclOpTensor(input0, ToDims64(input0.dims), &dY, &bY); 
        CreateAclOpTensor(input0, ToDims64(input0.dims), &dOut, &bOut); 

        aclopAttr *attr = aclopCreateAttr();
        aclopSetAttrFloat(attr, "alpha", alpha);

        std::vector<aclTensorDesc*> inD = {dX, dY};
        std::vector<aclDataBuffer*> inB = {bX, bY};
        std::vector<aclTensorDesc*> outD = {dOut};
        std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("Axpy", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       attr, GetFastllmAclStream());

        aclDestroyTensorDesc(dX); aclDestroyDataBuffer(bX);
        aclDestroyTensorDesc(dY); aclDestroyDataBuffer(bY);
        aclDestroyTensorDesc(dOut); aclDestroyDataBuffer(bOut);
        aclopDestroyAttr(attr);
    }

    void FastllmAclMul(const Data &input, float v, Data &output) {
        aclTensorDesc *dIn, *dOut; aclDataBuffer *bIn, *bOut;
        CreateAclOpTensor(input, ToDims64(input.dims), &dIn, &bIn);
        CreateAclOpTensor(output, ToDims64(output.dims), &dOut, &bOut);

        aclopAttr *attr = aclopCreateAttr();
        aclopSetAttrFloat(attr, "value", v);

        std::vector<aclTensorDesc*> inD = {dIn}; std::vector<aclDataBuffer*> inB = {bIn};
        std::vector<aclTensorDesc*> outD = {dOut}; std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("Muls", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       attr, GetFastllmAclStream());

        aclDestroyTensorDesc(dIn); aclDestroyDataBuffer(bIn);
        aclDestroyTensorDesc(dOut); aclDestroyDataBuffer(bOut);
        aclopDestroyAttr(attr);
    }

    void FastllmAclMulTo(const Data &input0, const Data &input1, float alpha) {
        aclTensorDesc *dX, *dY, *dOut; aclDataBuffer *bX, *bY, *bOut;
        CreateAclOpTensor(input1, ToDims64(input1.dims), &dX, &bX);
        CreateAclOpTensor(input0, ToDims64(input0.dims), &dY, &bY);
        CreateAclOpTensor(input0, ToDims64(input0.dims), &dOut, &bOut);

        std::vector<aclTensorDesc*> inD = {dY, dX};
        std::vector<aclDataBuffer*> inB = {bY, bX};
        std::vector<aclTensorDesc*> outD = {dOut};
        std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("Mul", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       nullptr, GetFastllmAclStream());

        aclDestroyTensorDesc(dX); aclDestroyDataBuffer(bX);
        aclDestroyTensorDesc(dY); aclDestroyDataBuffer(bY);
        if (std::abs(alpha - 1.0f) > 1e-6) {
            FastllmAclMul(input0, alpha, const_cast<Data&>(input0));
        }
    }

    void FastllmAclPermute(const Data &input, const std::vector<int> &axis) {
        Data &mutableInput = const_cast<Data&>(input);
        std::vector<int64_t> permData;
        std::vector<int64_t> newDims;
        for (int i : axis) {
            newDims.push_back(mutableInput.dims[i]);
            permData.push_back((int64_t)i);
        }

        size_t dataBytes = mutableInput.GetBytes();
        void* tempPtr = FastllmAclMalloc(dataBytes); 
        
        aclTensorDesc *dIn, *dOut, *dPerm; aclDataBuffer *bIn, *bOut, *bPerm;
        CreateAclOpTensor(mutableInput, ToDims64(mutableInput.dims), &dIn, &bIn);
        
        aclDataType type = (mutableInput.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;
        dOut = aclCreateTensorDesc(type, newDims.size(), newDims.data(), ACL_FORMAT_ND);
        bOut = aclCreateDataBuffer(tempPtr, dataBytes);

        void* permDevPtr = FastllmAclMalloc(permData.size() * sizeof(int64_t));
        aclrtMemcpy(permDevPtr, permData.size() * sizeof(int64_t), permData.data(), permData.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int64_t> permShape = {(int64_t)permData.size()};
        dPerm = aclCreateTensorDesc(ACL_INT64, 1, permShape.data(), ACL_FORMAT_ND);
        bPerm = aclCreateDataBuffer(permDevPtr, permData.size() * sizeof(int64_t));

        std::vector<aclTensorDesc*> inD = {dIn, dPerm}; std::vector<aclDataBuffer*> inB = {bIn, bPerm};
        std::vector<aclTensorDesc*> outD = {dOut}; std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("Transpose", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       nullptr, GetFastllmAclStream());

        aclrtSynchronizeStream(GetFastllmAclStream());
        aclrtMemcpy(mutableInput.deviceData, dataBytes, tempPtr, dataBytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
        
        std::vector<int> intNewDims(newDims.begin(), newDims.end());
        mutableInput.Resize(intNewDims);

        aclDestroyTensorDesc(dIn); aclDestroyDataBuffer(bIn);
        aclDestroyTensorDesc(dPerm); aclDestroyDataBuffer(bPerm);
        aclDestroyTensorDesc(dOut); aclDestroyDataBuffer(bOut);
        FastllmAclFree(tempPtr);
        FastllmAclFree(permDevPtr);
    }

    void FastllmAclNearlyRotatePosition2D(const Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        aclTensorDesc *dX, *dCos, *dSin, *dOut; aclDataBuffer *bX, *bCos, *bSin, *bOut;
        CreateAclOpTensor(data, ToDims64(data.dims), &dX, &bX);
        CreateAclOpTensor(cosData, ToDims64(cosData.dims), &dCos, &bCos);
        CreateAclOpTensor(sinData, ToDims64(sinData.dims), &dSin, &bSin);
        CreateAclOpTensor(data, ToDims64(data.dims), &dOut, &bOut);

        std::vector<aclTensorDesc*> inD = {dX, dCos, dSin};
        std::vector<aclDataBuffer*> inB = {bX, bCos, bSin};
        std::vector<aclTensorDesc*> outD = {dOut};
        std::vector<aclDataBuffer*> outB = {bOut};

        aclopExecuteV2("RotaryMul", inD.size(), inD.data(), inB.data(), outD.size(), outD.data(), outB.data(),
                       nullptr, GetFastllmAclStream());

        aclDestroyTensorDesc(dX); aclDestroyDataBuffer(bX);
        aclDestroyTensorDesc(dCos); aclDestroyDataBuffer(bCos);
        aclDestroyTensorDesc(dSin); aclDestroyDataBuffer(bSin);
        aclDestroyTensorDesc(dOut); aclDestroyDataBuffer(bOut);
    }

    void FastllmAclAttentionMask(const Data &input, const Data &mask, float maskValue) {
        FastllmAclAddTo(const_cast<Data&>(input), mask, maskValue);
    }

    void FastllmAclRepeat(void *src, void *dst, int outer, int repeatTimes, int inputStride, int outputStride, int channelsInner, int channelsInputInner) {
        uint8_t *s = (uint8_t*)src;
        uint8_t *d = (uint8_t*)dst;
        for (int i = 0; i < outer; i++) {
            for (int j = 0; j < repeatTimes; j++) {
                aclrtMemcpyAsync(d + i * outputStride + j * channelsInner, channelsInputInner,
                                 s + i * inputStride, channelsInputInner,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE, GetFastllmAclStream());
            }
        }
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

}