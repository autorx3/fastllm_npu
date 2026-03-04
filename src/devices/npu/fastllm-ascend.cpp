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
 #include "aclnnop/aclnn_slice.h"

namespace fastllm {

    aclrtStream g_aclStream = nullptr;
    static bool g_isInitialized = false;

    struct NpuWorkspace {
        void* basePtr = nullptr;      
        size_t capacity = 0;          
        size_t currentOffset = 0;     
        //std::mutex mtx; 

        // 默认预分配大小：8GB
        const size_t DEFAULT_POOL_SIZE = 8UL * 1024 * 1024 * 1024; 

        NpuWorkspace() {}

        NpuWorkspace(const NpuWorkspace&) = delete;
        NpuWorkspace& operator=(const NpuWorkspace&) = delete;

        void* Get(size_t size) {
            //std::lock_guard<std::mutex> lock(mtx);
            if (basePtr == nullptr) {
                aclError ret = aclrtMalloc(&basePtr, DEFAULT_POOL_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    printf("CRITICAL ERROR: NpuWorkspace init failed. Code: %d\n", ret);
                    return nullptr;
                }
                capacity = DEFAULT_POOL_SIZE;
                printf("NpuWorkspace Initialized: %zu MB\n", capacity / 1024 / 1024);
            }

            size_t alignSize = (size + 255) / 256 * 256;

            if (currentOffset + alignSize > capacity) {
                printf("\n[CRITICAL OOM] NpuWorkspace exhausted!\n");
                printf("  Total Capacity: %zu MB\n", capacity / 1024 / 1024);
                printf("  Used:           %zu MB\n", currentOffset / 1024 / 1024);
                printf("  Requested:      %zu Bytes\n", alignSize);
                printf("SOLUTION: Increase DEFAULT_POOL_SIZE in NpuWorkspace.\n");
                exit(-1); 
            }

            //返回地址
            void* ptr = (uint8_t*)basePtr + currentOffset;
            currentOffset += alignSize;
            return ptr;
        }

        void Reset() {
            //std::lock_guard<std::mutex> lock(mtx);
            if (currentOffset > 0) {
                aclrtSynchronizeStream(g_aclStream); 
                currentOffset = 0;
            }
        }

        ~NpuWorkspace() {
            if (basePtr){
                aclrtFree(basePtr);
                basePtr = nullptr;
            }
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

    void FastllmAclMemset0(void *devPtr, size_t size) {
        aclrtMemset(devPtr, size, 0, size);
    }

    void *FastllmAclPrepareInput(const fastllm::Data &input) {
        void *ret = nullptr;
        if (input.dataDevice == fastllm::DataDevice::ASCEND) {
            ret = (void*)input.deviceData;
        } else {
            if (input.expansionBytes > 0) {
                ret = FastllmAclMalloc(input.expansionBytes);
                if (ret == nullptr) {
                    printf("[FATAL] FastllmAclPrepareInput Malloc failed!\n");
                    return nullptr;
                }
                FastllmAclCopyFromHostToDevice(ret, input.cpuData, input.expansionBytes);
            }
        }
        return ret;
    }

    void FastllmAclFinishInput(const fastllm::Data &input, void *data) {
        if (input.dataDevice != fastllm::DataDevice::ASCEND && data != nullptr) {
            FastllmAclFree(data);
        }
    }

    void *FastllmAclPrepareOutput(fastllm::Data &output) {
        void *ret = nullptr;
        if (output.dataDevice == fastllm::DataDevice::ASCEND) {
            ret = (void*)output.deviceData;
        } else {
            if (output.expansionBytes > 0) {
                ret = FastllmAclMalloc(output.expansionBytes);
                if (ret == nullptr) {
                    printf("[FATAL] FastllmAclPrepareOutput Malloc failed!\n");
                }
            }
        }
        return ret;
    }

    void FastllmAclFinishOutput(fastllm::Data &output, void *data) {
        if (data == nullptr) return;

        if (output.dataDevice != fastllm::DataDevice::ASCEND) {
            aclrtSynchronizeStream(GetFastllmAclStream());
            if (output.cpuData == nullptr) {
                output.Allocate(); 
            }
            FastllmAclCopyFromDeviceToHost(output.cpuData, data, output.expansionBytes);

            FastllmAclFree(data);
        }
    }

    static inline aclTensor* CreateAclTensor(const Data &data, const std::vector<int> &dims, void* customDevPtr = nullptr) {
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

    // create acl bool tensor
    static inline aclTensor* CreateBoolTensorFromDataND(void* devPtr, const std::vector<int>& dims) {
        std::vector<int64_t> dims64;
        dims64.reserve(dims.size());
        for (int d : dims) dims64.push_back(d);
    
        std::vector<int64_t> strides(dims.size());
        int64_t stride = 1;
        for (int i = (int)dims.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= dims[i];
        }
    
        return aclCreateTensor(dims64.data(), dims64.size(),
                               ACL_BOOL,
                               strides.data(),
                               0, ACL_FORMAT_ND,
                               dims64.data(), dims64.size(),
                               devPtr);
    }

    void FastllmAclMatMul(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclWeight = FastllmAclPrepareInput(weight);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims, aclWeight);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        //cubeMathType = 1 means ALLOW_FP32_DOWN_PRECISION --> FP32 -> FP16
        int8_t cubeMathType = 1; 

        if (aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, cubeMathType, &workspaceSize, &executor) == ACL_SUCCESS) {
            aclnnMatmul(g_workspace.Get(workspaceSize), workspaceSize, executor, GetFastllmAclStream());
        }
        
        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
        
        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishInput(weight, aclWeight);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclMatMulTransB(const Data &input, const Data &weight, const Data &bias, Data &output, int alpha, int beta) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclWeight = FastllmAclPrepareInput(weight);
        void *aclOutput = FastllmAclPrepareOutput(output);

        int rank = weight.dims.size();
        int64_t N = weight.dims[rank - 2];
        int64_t K = weight.dims[rank - 1];
        
        std::vector<int64_t> viewDims;
        for (int i = 0; i < rank - 2; ++i) viewDims.push_back(weight.dims[i]);
        viewDims.push_back(K);
        viewDims.push_back(N);

        std::vector<int64_t> viewStrides(rank);
        
        // ==========================================
        // 【核心修复】：转置后的内层跨度必须是老矩阵的内层大小 K！
        // ==========================================
        viewStrides[rank - 2] = 1;
        viewStrides[rank - 1] = K; 
        
        int64_t stride = N * K;
        for (int i = rank - 3; i >= 0; i--) {
            viewStrides[i] = stride;
            stride *= weight.dims[i];
        }
        
        aclDataType type = ACL_FLOAT;
        if (weight.dataType == DataType::FLOAT16) type = ACL_FLOAT16;

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        
        aclTensor *tWeight = aclCreateTensor(viewDims.data(), viewDims.size(), type,
                                             viewStrides.data(), 0, ACL_FORMAT_ND,
                                             viewDims.data(), viewDims.size(), aclWeight);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);


        int8_t cubeMathType = 1; 
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        aclError ret = aclnnMatmulGetWorkspaceSize(tInput, tWeight, tOutput, cubeMathType, &ws, &ex);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = ws > 0 ? g_workspace.Get(ws) : nullptr;
            aclnnMatmul(wsAddr, ws, ex, GetFastllmAclStream());
        } else {
            printf("[FATAL] FastllmAclMatMulTransB failed! Error Code: %d\n", ret);
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tOutput);
        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishInput(weight, aclWeight);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclQuantLinearDequant(Data &input, Data &weight, Data &weightScale, 
                                      Data &xScale, Data &bias, Data &output) {
        void *aclInput       = FastllmAclPrepareInput(input);
        void *aclWeight      = FastllmAclPrepareInput(weight);
        void *aclWeightScale = FastllmAclPrepareInput(weightScale);
        void *aclXScale      = xScale.dims.size() > 0 ? FastllmAclPrepareInput(xScale) : nullptr;
        void *aclBias        = bias.dims.size() > 0 ? FastllmAclPrepareInput(bias) : nullptr;
        void *aclOutput      = FastllmAclPrepareOutput(output);

        int64_t K = input.dims.back();
        int64_t M = input.Count(0) / K; 
        int64_t N = weight.dims[0]; 

        std::vector<int> dimInput = {(int)M, (int)K};
        std::vector<int> dimWeight = {(int)N, (int)K}; 
        std::vector<int> dimOutput = {(int)M, (int)N};
        std::vector<int> dimWScale = {(int)N};
        std::vector<int> dimXScale = {(int)M};

        aclTensor *tInput = CreateAclTensor(input, dimInput, aclInput);
        aclTensor *tWeight = CreateAclTensor(weight, dimWeight, aclWeight);
        aclTensor *tWeightScale = CreateAclTensor(weightScale, dimWScale, aclWeightScale);
        aclTensor *tOutput = CreateAclTensor(output, dimOutput, aclOutput);

        aclTensor *tXScale = aclXScale ? CreateAclTensor(xScale, dimXScale, aclXScale) : nullptr;
        // NO USE
        aclTensor *tBias = aclBias ? CreateAclTensor(bias, {bias.dims[0]}, aclBias) : nullptr;

        uint64_t workspaceSize = 0; 
        aclOpExecutor *executor = nullptr;
        char mode[] = "pertoken"; 

        aclError ret = aclnnQuantMatmulDequantGetWorkspaceSize(
            tInput, 
            tWeight, 
            tWeightScale, 
            nullptr,       // 必须为 nullptr
            tXScale,       // xScaleOptional
            nullptr,       // xOffsetOptional 必须为 nullptr
            nullptr,       // smoothScaleOptional 必须为 nullptr
            mode,    // xQuantMode
            true,          // transposeWeight 必须为 true
            tOutput, 
            &workspaceSize, 
            &executor);
            
        if (ret == ACL_SUCCESS) {
            void *wsAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclError execRet = aclnnQuantMatmulDequant(wsAddr, workspaceSize, executor, GetFastllmAclStream());
            if (execRet != ACL_SUCCESS) {
                printf("[FATAL] aclnnQuantMatmulDequant Execution failed! Error Code: %d\n", execRet);
            }
        } else {
            printf("[FATAL] aclnnQuantMatmulDequantGetWorkspaceSize failed! Error Code: %d\n", ret);
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tWeight); aclDestroyTensor(tWeightScale);
        aclDestroyTensor(tOutput); 
        if(tXScale) aclDestroyTensor(tXScale); 
        if(tBias) aclDestroyTensor(tBias);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishInput(weight, aclWeight);
        FastllmAclFinishInput(weightScale, aclWeightScale);
        if (aclXScale) FastllmAclFinishInput(xScale, aclXScale);
        if (aclBias) FastllmAclFinishInput(bias, aclBias);
        FastllmAclFinishOutput(output, aclOutput);
    }

    //性能优化版本
    //需要先计算rstd
    void FastllmAclRMSNorm(const Data &input, const Data &weight, const Data &bias, Data &output, float eps) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclWeight = FastllmAclPrepareInput(weight);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tWeight = CreateAclTensor(weight, weight.dims, aclWeight);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);

        std::vector<int64_t> rstdDims;
        int keepDims = input.dims.size() - weight.dims.size();
        for (int i = 0; i < (keepDims > 0 ? keepDims : 1); ++i) {
            rstdDims.push_back(keepDims > 0 ? input.dims[i] : 1);
        }

        int64_t numElem = 1; 
        for(auto d : rstdDims) numElem *= d;
        size_t rstdBytes = numElem * sizeof(float); 

        void *rstdPtr = g_workspace.Get(rstdBytes);
        //printf("DEBUG: rstdPtr = %p\n", rstdPtr); // 打印地址

        std::vector<int64_t> rstdStrides(rstdDims.size());
        int64_t stride = 1;
        for (int i = rstdDims.size() - 1; i >= 0; i--) {
            rstdStrides[i] = stride;
            stride *= rstdDims[i];
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
                //printf("DEBUG: opWorkspaceAddr = %p (Size: %lu)\n", opWorkspaceAddr, opWorkspaceSize); // 打印地址
        }

        aclnnRmsNorm(opWorkspaceAddr, opWorkspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tInput); 
        aclDestroyTensor(tWeight); 
        aclDestroyTensor(tOutput); 
        aclDestroyTensor(tRstd);
        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishInput(weight, aclWeight);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclSilu(const Data &input, Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnSiluGetWorkspaceSize(tInput, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnSilu(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);
        
        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    //to analysis
    void FastllmAclSwiglu(const Data &input, Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);
        
        int64_t dim = input.dims.size() - 1; 
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;

        aclError ret = aclnnSwiGluGetWorkspaceSize(tInput, dim, tOutput, &ws, &ex);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = ws > 0 ? g_workspace.Get(ws) : nullptr;
            aclnnSwiGlu(wsAddr, ws, ex, GetFastllmAclStream());
        } else {
            int64_t split_dim = input.dims.size() - 1;
            int64_t half_size = input.dims.back() / 2;
            
            // 步骤 A: 切割左半部分 (Slice Left)
            void *tempLeft = g_workspace.Get(output.GetBytes());
            aclTensor *tLeft = CreateAclTensor(output, output.dims, tempLeft);
            uint64_t wsSlice1 = 0; aclOpExecutor *exSlice1 = nullptr;
            aclnnSliceGetWorkspaceSize(tInput, split_dim, 0, half_size, 1, tLeft, &wsSlice1, &exSlice1);
            aclnnSlice(wsSlice1 > 0 ? g_workspace.Get(wsSlice1) : nullptr, wsSlice1, exSlice1, GetFastllmAclStream());

            // 步骤 B: 对左半部分执行 Silu，结果写入 tOutput
            uint64_t wsSilu = 0; aclOpExecutor *exSilu = nullptr;
            aclnnSiluGetWorkspaceSize(tLeft, tOutput, &wsSilu, &exSilu);
            aclnnSilu(wsSilu > 0 ? g_workspace.Get(wsSilu) : nullptr, wsSilu, exSilu, GetFastllmAclStream());

            // 步骤 C: 切割右半部分 (Slice Right)
            void *tempRight = g_workspace.Get(output.GetBytes());
            aclTensor *tRight = CreateAclTensor(output, output.dims, tempRight);
            uint64_t wsSlice2 = 0; aclOpExecutor *exSlice2 = nullptr;
            aclnnSliceGetWorkspaceSize(tInput, split_dim, half_size, input.dims.back(), 1, tRight, &wsSlice2, &exSlice2);
            aclnnSlice(wsSlice2 > 0 ? g_workspace.Get(wsSlice2) : nullptr, wsSlice2, exSlice2, GetFastllmAclStream());

            // 步骤 D: 将 Silu(Left) 与 Right 相乘，结果覆盖回 tOutput
            uint64_t wsMul = 0; aclOpExecutor *exMul = nullptr;
            aclnnMulGetWorkspaceSize(tOutput, tRight, tOutput, &wsMul, &exMul);
            aclnnMul(wsMul > 0 ? g_workspace.Get(wsMul) : nullptr, wsMul, exMul, GetFastllmAclStream());

            aclDestroyTensor(tLeft);
            aclDestroyTensor(tRight);
        }

        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);
        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclSoftmax(const Data &input, Data &output, int axis) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnSoftmaxGetWorkspaceSize(tInput, (int64_t)axis, tOutput, &ws, &ex) == ACL_SUCCESS) {
            aclnnSoftmax(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }
        aclDestroyTensor(tInput); aclDestroyTensor(tOutput);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    //待测试
    //to analysis
    void FastllmAclEmbedding(const Data &input, const Data &weight, Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclWeight = FastllmAclPrepareInput(weight);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tW = CreateAclTensor(weight, weight.dims, aclWeight);
        aclTensor *tO = CreateAclTensor(output, output.dims, aclOutput);
        aclTensor *tI = nullptr;

        if (input.dataType == DataType::FLOAT32) {
            // === Cast Float -> Int64 ===
            aclTensor *tI_Float = CreateAclTensor(input, input.dims, aclInput);

            std::vector<int64_t> dims; 
            for(auto d : input.dims) dims.push_back(d);
            std::vector<int64_t> strides(dims.size(), 1);
            for(int i = (int)dims.size() - 2; i >= 0; i--) strides[i] = dims[i+1] * strides[i+1];

            int64_t elemCount = input.Count(0);
            size_t intBytes = elemCount * sizeof(int64_t);
            void *tempIntData = g_workspace.Get(intBytes); 

            aclTensor *tI_Int64 = aclCreateTensor(dims.data(), dims.size(), ACL_INT64, 
                                                strides.data(), 0, ACL_FORMAT_ND, 
                                                dims.data(), dims.size(), tempIntData);

            uint64_t castWs = 0; aclOpExecutor *castEx = nullptr;
            aclError ret = aclnnCastGetWorkspaceSize(tI_Float, ACL_INT64, tI_Int64, &castWs, &castEx);
            if (ret == ACL_SUCCESS) {
                void *castWsAddr = castWs > 0 ? g_workspace.Get(castWs) : nullptr;
                aclError execRet = aclnnCast(castWsAddr, castWs, castEx, GetFastllmAclStream());
                if (execRet != ACL_SUCCESS) {
                    printf("[FATAL] aclnnCast Execution failed in Embedding! Error Code: %d\n", execRet);
                }
            } else {
                printf("[FATAL] aclnnCastGetWorkspaceSize failed! Error Code: %d\n", ret);
            }
            
            aclDestroyTensor(tI_Float);
            tI = tI_Int64;
        } else {
            tI = CreateAclTensor(input, input.dims, aclInput); 
        }

        uint64_t embWs = 0; aclOpExecutor *embEx = nullptr;
        aclError ret = aclnnEmbeddingGetWorkspaceSize(tW, tI, tO, &embWs, &embEx);
        if (ret == ACL_SUCCESS) {
            void *embWsAddr = embWs > 0 ? g_workspace.Get(embWs) : nullptr;
            aclError execRet = aclnnEmbedding(embWsAddr, embWs, embEx, GetFastllmAclStream());
            if (execRet != ACL_SUCCESS) {
                printf("[FATAL] aclnnEmbedding Execution failed! Error Code: %d\n", execRet);
            }
        } else {
            printf("[FATAL] aclnnEmbeddingGetWorkspaceSize failed! Error Code: %d\n", ret);
        }

        aclDestroyTensor(tW); 
        if (tI) aclDestroyTensor(tI); 
        aclDestroyTensor(tO);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishInput(weight, aclWeight);
        FastllmAclFinishOutput(output, aclOutput);
    }


    void FastllmAclTopK(const Data &input, Data &output, int topk) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);
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

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput); 
        
        aclDataType aclType = (input.dataType == DataType::FLOAT16) ? ACL_FLOAT16 : ACL_FLOAT;
        aclTensor *tValues = aclCreateTensor(tempDims.data(), tempDims.size(), aclType, tempStrides.data(), 0, ACL_FORMAT_ND, tempDims.data(), tempDims.size(), ptrValues);
        aclTensor *tIndices = aclCreateTensor(tempDims.data(), tempDims.size(), ACL_INT64, tempStrides.data(), 0, ACL_FORMAT_ND, tempDims.data(), tempDims.size(), ptrIndices);
        aclTensor *tIndicesCast = aclCreateTensor(tempDims.data(), tempDims.size(), aclType, tempStrides.data(), 0, ACL_FORMAT_ND, tempDims.data(), tempDims.size(), ptrIndicesCast);

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

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclCat(const Data &input0, const Data &input1, Data &output, int axis){
        void *aclInput0  = FastllmAclPrepareInput(input0);
        void *aclInput1  = FastllmAclPrepareInput(input1);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tI0 = CreateAclTensor(input0, input0.dims, aclInput0);
        aclTensor *tI1 = CreateAclTensor(input1, input1.dims, aclInput1);
        aclTensor *tO  = CreateAclTensor(output, output.dims, aclOutput);

        aclTensor *concatTensors[] = {tI0, tI1};
        aclTensorList *tensorList = aclCreateTensorList(concatTensors, 2);

        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        if (aclnnCatGetWorkspaceSize(tensorList, axis, tO, &ws, &ex) == ACL_SUCCESS) {
            aclnnCat(g_workspace.Get(ws), ws, ex, GetFastllmAclStream());
        }

        //aclDestroyTensor(tI0); aclDestroyTensor(tI1); 
        aclDestroyTensor(tO);
        aclDestroyTensorList(tensorList);

        FastllmAclFinishInput(input0, aclInput0);
        FastllmAclFinishInput(input1, aclInput1);
        FastllmAclFinishOutput(output, aclOutput);
    }


    void FastllmAclSplit(const fastllm::Data &input, int axis, int start, int end, fastllm::Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        int dimsLen = input.dims.size();
        int64_t dim = (axis % dimsLen + dimsLen) % dimsLen;
        int64_t start_idx = std::max(0, std::min((int)input.dims[dim], start));
        int64_t end_idx = std::max(0, std::min((int)input.dims[dim], end));
        int64_t step = 1; // 连续切片，步长为 1

        aclTensor *tInput = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOutput = CreateAclTensor(output, output.dims, aclOutput);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        if (aclnnSliceGetWorkspaceSize(tInput, dim, start_idx, end_idx, step, tOutput, &workspaceSize, &executor) == ACL_SUCCESS) {
            void *workspaceAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclnnSlice(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());
        } else {
            printf("[FATAL] FastllmAclSplit aclnnSliceGetWorkspaceSize failed!\n");
        }

        aclDestroyTensor(tInput);
        aclDestroyTensor(tOutput);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
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
        std::vector<int> scoreDims = q.dims; // [1, 1, 2, 3]
        scoreDims.back() = k.dims[k.dims.size() - 2]; // -> [1, 1, 2, 2]
        score.dataType = q.dataType;
        score.Resize(scoreDims);

        score.dataDevice = fastllm::DataDevice::ASCEND;
        score.Allocate(); 

        FastllmAclMatMulTransB(q, k, Data(), score, 1, 0);

        if (std::abs(scale - 1.0f) > 1e-6) FastllmAclMul(score, scale, score);
        if (mask.dims.size() > 0) FastllmAclAddTo(score, mask, -10000.0f);
        FastllmAclSoftmax(score, score, -1);

        std::vector<int> outDims = q.dims;
        outDims.back() = v.dims.back();
        output.dataType = q.dataType;

        output.dataDevice = fastllm::DataDevice::ASCEND;
        output.Resize(outDims);
        output.Allocate(); 

        void *aclScore = FastllmAclPrepareInput(score);
        void *aclV = FastllmAclPrepareInput(v);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tScore = CreateAclTensor(score, score.dims, aclScore);
        aclTensor *tV = CreateAclTensor(v, v.dims, aclV);
        aclTensor *tOut = CreateAclTensor(output, output.dims, aclOutput);

        uint64_t wsSize = 0; aclOpExecutor *executor = nullptr;
        aclError ret = aclnnMatmulGetWorkspaceSize(tScore, tV, tOut, 1, &wsSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = wsSize > 0 ? g_workspace.Get(wsSize) : nullptr;
            aclnnMatmul(wsAddr, wsSize, executor, GetFastllmAclStream());
        } else {
            printf("[FATAL] Attention Score * V aclnnMatmul failed! Error Code: %d\n", ret);
        }

        aclDestroyTensor(tScore); aclDestroyTensor(tV); aclDestroyTensor(tOut);
        FastllmAclFinishInput(score, aclScore);
        FastllmAclFinishInput(v, aclV);
        FastllmAclFinishOutput(output, aclOutput);
    }


    void FastllmAclFlashAttentionV3(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output, int group, float scale, int maskType) {
        if (q.dims.size()!=4 || k.dims.size()!=4 || v.dims.size()!=4 || output.dims.size()!=4) {
            printf(" ERROR: q/k/v/output must be 4D BNSD.\n");
            return;
        }
        if (q.dataType!=DataType::FLOAT16 || k.dataType!=DataType::FLOAT16 ||
            v.dataType!=DataType::FLOAT16 || output.dataType!=DataType::FLOAT16) {
            printf(" ERROR: q/k/v/output must be FP16 on Atlas inference cards.\n");
            return;
        }
        if (!q.deviceData || !k.deviceData || !v.deviceData || !output.deviceData) {
            printf(" ERROR: deviceData is null.\n");
            return;
        }
        const int64_t B   = q.dims[0];
        const int64_t N   = q.dims[1];
        const int64_t Sq  = q.dims[2];
        const int64_t D   = q.dims[3];
        const int64_t Nk  = k.dims[1];
        const int64_t Skv = k.dims[2];

        if (Nk != N) {
            printf("ERROR: Nk(%ld) != N(%ld). GQA not supported here on Atlas inference cards.\n",
                   (long)Nk, (long)N);
            return;
        }
        if (k.dims[0]!=B || v.dims[0]!=B || v.dims[1]!=Nk || v.dims[2]!=Skv || k.dims[3]!=D || v.dims[3]!=D) {
            printf("ERROR: k/v shape mismatch.\n");
            return;
        }
        if (output.dims[0]!=B || output.dims[1]!=N || output.dims[2]!=Sq || output.dims[3]!=D) {
            printf("ERROR: output shape must match q.\n");
            return;
        }

        void *aclQ    = FastllmAclPrepareInput(q);
        void *aclK    = FastllmAclPrepareInput(k);
        void *aclV    = FastllmAclPrepareInput(v);
        void *aclOut  = FastllmAclPrepareOutput(output);
        void *aclMask = (maskType != 0 && !mask.dims.empty()) ? FastllmAclPrepareInput(mask) : nullptr;

        aclTensor *tQ   = CreateAclTensor(q, q.dims, aclQ);
        aclTensor *tK   = CreateAclTensor(k, k.dims, aclK);
        aclTensor *tV   = CreateAclTensor(v, v.dims, aclV);
        aclTensor *tOut = CreateAclTensor(output, output.dims, aclOut);
        
        aclTensor *tMask = nullptr;
        if (aclMask) {
            tMask = CreateBoolTensorFromDataND(aclMask, mask.dims);
        }

        char inputLayout[] = "BNSD";
        const int64_t numHeads = N;
        const int64_t numKeyValueHeads = 0;    // Nk==N
        const double  scaleValue = (double)scale;
        const int64_t preTokens  = 2147483647;
        const int64_t nextTokens = 2147483647;
        const int64_t sparseMode = 0;
        const int64_t innerPrecise = 1;

        aclTensor* pseShift = nullptr;
        aclIntArray* actualSeqLengths = nullptr;
        aclIntArray* actualSeqLengthsKv = nullptr;
        aclTensor* deqScale1 = nullptr;
        aclTensor* quantScale1 = nullptr;
        aclTensor* deqScale2 = nullptr;
        aclTensor* quantScale2 = nullptr;
        aclTensor* quantOffset2 = nullptr;

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        
        aclError ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(
            tQ, tK, tV,
            pseShift,
            tMask,
            actualSeqLengths,
            actualSeqLengthsKv,
            deqScale1,
            quantScale1,
            deqScale2,
            quantScale2,
            quantOffset2,
            numHeads,
            scaleValue,
            preTokens,
            nextTokens,
            inputLayout,
            numKeyValueHeads,
            sparseMode,
            innerPrecise,
            tOut,
            &workspaceSize,
            &executor
        );
        if(ret != ACL_SUCCESS) {
            printf("ERROR: GetWorkspaceSize failed. ret=%d\n", (int)ret);
            return ;
        }
        void *workspaceAddr = (workspaceSize > 0) ? g_workspace.Get(workspaceSize) : nullptr;
        ret = aclnnPromptFlashAttentionV3(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());
        if (ret != ACL_SUCCESS) {
            printf("ERROR: aclnnPromptFlashAttentionV3 failed. ret=%d\n", (int)ret);
            return ;
        }

        aclDestroyTensor(tQ);
        aclDestroyTensor(tK);
        aclDestroyTensor(tV);
        aclDestroyTensor(tOut);
        if (tMask) aclDestroyTensor(tMask);

        FastllmAclFinishInput(q, aclQ);
        FastllmAclFinishInput(k, aclK);
        FastllmAclFinishInput(v, aclV);
        if (aclMask) FastllmAclFinishInput(mask, aclMask);
        FastllmAclFinishOutput(output, aclOut);
    }

    void FastllmAclRepeat(const Data &input, Data &output, int axis, int repeatTimes) {
        int64_t outer = input.Count(0) / input.Count(axis);
        int64_t block = input.Count(axis); 
        
        std::vector<int64_t> selfShape = { outer, 1, block };
        std::vector<int64_t> outShape  = { outer, (int64_t)repeatTimes, block };

        std::vector<int64_t> selfStrides = { block, block, 1 }; 
        std::vector<int64_t> outStrides = { (int64_t)(repeatTimes * block), block, 1 };

        void *aclInput = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclDataType type = ACL_FLOAT;
        if (input.dataType == DataType::FLOAT16) type = ACL_FLOAT16;
        else if (input.dataType == DataType::FLOAT32) type = ACL_FLOAT;
        else if (input.dataType == DataType::INT8) type = ACL_INT8;

        aclTensor *tSelf = aclCreateTensor(selfShape.data(), selfShape.size(), type, 
                                           selfStrides.data(), 0, ACL_FORMAT_ND, 
                                           selfShape.data(), selfShape.size(), aclInput);

        aclTensor *tOut = aclCreateTensor(outShape.data(), outShape.size(), type, 
                                          outStrides.data(), 0, ACL_FORMAT_ND, 
                                          outShape.data(), outShape.size(), aclOutput);

        aclIntArray *expandSize = aclCreateIntArray(outShape.data(), outShape.size());

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclError ret = aclnnExpandGetWorkspaceSize(tSelf, expandSize, tOut, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            ret = aclnnExpand(wsAddr, workspaceSize, executor, GetFastllmAclStream());
            if (ret != ACL_SUCCESS) {
                printf("[FATAL] aclnnExpand Execution failed! Error Code: %d\n", ret);
            }
        } else {
            printf("[FATAL] aclnnExpandGetWorkspaceSize failed! Error Code: %d\n", ret);
        }

        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOut);
        aclDestroyIntArray(expandSize);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclAdd(const Data &input, float v, Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tSelf = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOut = CreateAclTensor(output, output.dims, aclOutput);
        aclScalar *sOther = aclCreateScalar(&v, ACL_FLOAT);
        float alphaVal = 1.0f;
        aclScalar *sAlpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        aclnnAddsGetWorkspaceSize(tSelf, sOther, sAlpha, tOut, &workspaceSize, &executor);
        
        void *workspaceAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
        aclnnAdds(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tSelf); aclDestroyTensor(tOut);
        aclDestroyScalar(sOther); aclDestroyScalar(sAlpha);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }

    void FastllmAclAddTo(const Data &input0, const Data &input1, float alpha) {
        // 【In-place 魔法】：input0 即是输入也是输出
        void *aclInput0 = FastllmAclPrepareInput(input0); 
        void *aclInput1 = FastllmAclPrepareInput(input1);

        aclTensor *tSelf = CreateAclTensor(input0, input0.dims, aclInput0);
        aclTensor *tOther = CreateAclTensor(input1, input1.dims, aclInput1);
        aclScalar *sAlpha = aclCreateScalar(&alpha, ACL_FLOAT);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        aclnnAddGetWorkspaceSize(tSelf, tOther, sAlpha, tSelf, &workspaceSize, &executor);

        void *workspaceAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
        aclnnAdd(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tSelf); aclDestroyTensor(tOther); aclDestroyScalar(sAlpha);

        FastllmAclFinishInput(input1, aclInput1);
        // 使用 FinishOutput 将修改后的结果同步回 CPU 
        FastllmAclFinishOutput(const_cast<Data&>(input0), aclInput0); 
    }

    void FastllmAclMul(const Data &input, float v, Data &output) {
        void *aclInput  = FastllmAclPrepareInput(input);
        void *aclOutput = FastllmAclPrepareOutput(output);

        aclTensor *tSelf = CreateAclTensor(input, input.dims, aclInput);
        aclTensor *tOut = CreateAclTensor(output, output.dims, aclOutput);
        aclScalar *sOther = aclCreateScalar(&v, ACL_FLOAT);

        uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
        aclnnMulsGetWorkspaceSize(tSelf, sOther, tOut, &workspaceSize, &executor);

        void *workspaceAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
        aclnnMuls(workspaceAddr, workspaceSize, executor, GetFastllmAclStream());

        aclDestroyTensor(tSelf); aclDestroyTensor(tOut); aclDestroyScalar(sOther);

        FastllmAclFinishInput(input, aclInput);
        FastllmAclFinishOutput(output, aclOutput);
    }
    
    void FastllmAclMulTo(const Data &input0, const Data &input1, float alpha) {
        // 【In-place 魔法】
        void *aclInput0 = FastllmAclPrepareInput(input0);
        void *aclInput1 = FastllmAclPrepareInput(input1);

        aclTensor *tSelf = CreateAclTensor(input0, input0.dims, aclInput0); 
        aclTensor *tOther = CreateAclTensor(input1, input1.dims, aclInput1);

        uint64_t wsMul = 0; aclOpExecutor *exMul = nullptr;
        aclnnMulGetWorkspaceSize(tSelf, tOther, tSelf, &wsMul, &exMul);
        void *wsAddrMul = wsMul > 0 ? g_workspace.Get(wsMul) : nullptr;
        aclnnMul(wsAddrMul, wsMul, exMul, GetFastllmAclStream());

        if (std::abs(alpha - 1.0f) > 1e-6) {
            aclScalar *sAlpha = aclCreateScalar(&alpha, ACL_FLOAT);
            uint64_t wsScale = 0; aclOpExecutor *exScale = nullptr;
            aclnnInplaceMulsGetWorkspaceSize(tSelf, sAlpha, &wsScale, &exScale);
            void *wsAddrScale = wsScale > 0 ? g_workspace.Get(wsScale) : nullptr;
            aclnnInplaceMuls(wsAddrScale, wsScale, exScale, GetFastllmAclStream());
            aclDestroyScalar(sAlpha);
        }
    
        aclDestroyTensor(tSelf); aclDestroyTensor(tOther);

        FastllmAclFinishInput(input1, aclInput1);
        FastllmAclFinishOutput(const_cast<Data&>(input0), aclInput0);
    }

    void FastllmAclPermute(const Data &input, const std::vector<int> &axis) {
        Data &mutableInput = const_cast<Data&>(input);

        void *aclInput = FastllmAclPrepareInput(mutableInput);

        std::vector<int64_t> axisInt64;
        std::vector<int64_t> newDims;
        for (int i : axis) {
            axisInt64.push_back((int64_t)i);
            newDims.push_back(mutableInput.dims[i]);
        }
        aclIntArray *permArray = aclCreateIntArray(axisInt64.data(), axisInt64.size());

        size_t dataBytes = mutableInput.GetBytes();

        void* tempPtr = g_workspace.Get(dataBytes); 

        aclTensor *tSelf = CreateAclTensor(mutableInput, mutableInput.dims, aclInput);

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

        aclrtMemcpyAsync(aclInput, dataBytes, 
                         tempPtr, dataBytes, 
                         ACL_MEMCPY_DEVICE_TO_DEVICE, GetFastllmAclStream());

        mutableInput.Resize(std::vector<int>(newDims.begin(), newDims.end()));

        aclDestroyTensor(tSelf);
        aclDestroyTensor(tOut);
        aclDestroyIntArray(permArray);

        FastllmAclFinishOutput(mutableInput, aclInput);
    }

    void FastllmAclNearlyRotatePosition2D(const Data &data, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        void *aclData = FastllmAclPrepareInput(data);
        void *aclCos  = FastllmAclPrepareInput(cosData);
        void *aclSin  = FastllmAclPrepareInput(sinData);
        
        // ==========================================
        // 将 Query 强行补齐为 4D [q_b, q_s, q_n, q_d]
        // ==========================================
        std::vector<int> qDims = data.dims;
        while (qDims.size() < 4) qDims.insert(qDims.begin(), 1);
        
        // ==========================================
        // Cos 和 Sin 的第三维必须是 1，第四维必须是 128 (q_d)
        // ==========================================
        std::vector<int> cosDims = {qDims[0], qDims[1], 1, qDims[3]};
        std::vector<int> sinDims = {qDims[0], qDims[1], 1, qDims[3]};

        aclTensor *tQuery = CreateAclTensor(data, qDims, aclData);
        aclTensor *tCos = CreateAclTensor(cosData, cosDims, aclCos);
        aclTensor *tSin = CreateAclTensor(sinData, sinDims, aclSin);

        // ==========================================
        // 伪造 Dummy Key，完全复刻 Query 的维度
        // ==========================================
        fastllm::Data dummyKey(data.dataType, qDims);
        dummyKey.dataDevice = data.dataDevice; 
        dummyKey.Allocate(); // NPU 占坑
        void *aclDummyKey = FastllmAclPrepareInput(dummyKey);
        aclTensor *tKey = CreateAclTensor(dummyKey, qDims, aclDummyKey);

        // layout 只支持 1
        int64_t layout = 1; 
        char rotaryCoeff[] = "half"; 

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclError ret = aclnnApplyRotaryPosEmbV2GetWorkspaceSize(tQuery, tKey, tCos, tSin, layout, rotaryCoeff, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclError execRet = aclnnApplyRotaryPosEmbV2(wsAddr, workspaceSize, executor, GetFastllmAclStream());
            if (execRet != ACL_SUCCESS) {
                printf("[FATAL] aclnnApplyRotaryPosEmbV2 Execution failed! Error Code: %d\n", execRet);
            }
        } else {
            printf("[FATAL] aclnnApplyRotaryPosEmbV2GetWorkspaceSize failed! Error Code: %d\n", ret);
        }

        // 硬件同步
        // aclError syncRet = aclrtSynchronizeStream(GetFastllmAclStream());
        // if (syncRet != ACL_SUCCESS) printf("[FATAL] Stream Sync failed in RoPE! Error Code: %d\n", syncRet);

        aclDestroyTensor(tQuery); 
        aclDestroyTensor(tKey);   
        aclDestroyTensor(tCos); 
        aclDestroyTensor(tSin);

        FastllmAclFinishInput(cosData, aclCos);
        FastllmAclFinishInput(sinData, aclSin);
        FastllmAclFinishOutput(const_cast<Data&>(data), aclData);
        // dummyKey 随函数结束自动释放显存
    }

    void FastllmAclRotatePosition2D_Fused(const Data &query, const Data &key, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim) {
        void *aclQuery = FastllmAclPrepareInput(query);
        void *aclKey   = FastllmAclPrepareInput(key);
        void *aclCos   = FastllmAclPrepareInput(cosData);
        void *aclSin   = FastllmAclPrepareInput(sinData);

        // ==========================================
        // 将 Query 和 Key 强行补齐为 4D
        // ==========================================
        std::vector<int> qDims = query.dims;
        while (qDims.size() < 4) qDims.insert(qDims.begin(), 1);
        
        std::vector<int> kDims = key.dims;
        while (kDims.size() < 4) kDims.insert(kDims.begin(), 1);

        // Cos 和 Sin 的第三维必须是 1，第四维必须是 128 (q_d)
        std::vector<int> cosDims = {qDims[0], qDims[1], 1, qDims[3]};
        std::vector<int> sinDims = {qDims[0], qDims[1], 1, qDims[3]};

        aclTensor *tQuery = CreateAclTensor(query, qDims, aclQuery);
        aclTensor *tKey   = CreateAclTensor(key, kDims, aclKey); 
        aclTensor *tCos   = CreateAclTensor(cosData, cosDims, aclCos);
        aclTensor *tSin   = CreateAclTensor(sinData, sinDims, aclSin);

        // layout 只支持 1
        int64_t layout = 1; 
        char rotaryCoeff[] = "half"; 

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        aclError ret = aclnnApplyRotaryPosEmbV2GetWorkspaceSize(tQuery, tKey, tCos, tSin, layout, rotaryCoeff, &workspaceSize, &executor);
        if (ret == ACL_SUCCESS) {
            void *wsAddr = workspaceSize > 0 ? g_workspace.Get(workspaceSize) : nullptr;
            aclError execRet = aclnnApplyRotaryPosEmbV2(wsAddr, workspaceSize, executor, GetFastllmAclStream());
            if (execRet != ACL_SUCCESS) {
                printf("[FATAL] Fused aclnnApplyRotaryPosEmbV2 Execution failed! Error Code: %d\n", execRet);
            }
        } else {
            printf("[FATAL] Fused aclnnApplyRotaryPosEmbV2GetWorkspaceSize failed! Error Code: %d\n", ret);
        }

        // 强制同步
        // aclError syncRet = aclrtSynchronizeStream(GetFastllmAclStream());
        // if (syncRet != ACL_SUCCESS) printf("[FATAL] Stream Sync failed in Fused RoPE! Error Code: %d\n", syncRet);

        aclDestroyTensor(tQuery); 
        aclDestroyTensor(tKey); 
        aclDestroyTensor(tCos); 
        aclDestroyTensor(tSin);

        FastllmAclFinishInput(cosData, aclCos); 
        FastllmAclFinishInput(sinData, aclSin);
        FastllmAclFinishOutput(const_cast<Data&>(query), aclQuery);
        FastllmAclFinishOutput(const_cast<Data&>(key), aclKey);
    }

}