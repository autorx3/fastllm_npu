/**
 * @file fastllm-ascend.h
 * @brief Header definition for FastLLM Ascend NPU backend.
 * @details Exposes low-level NPU operations (aclnn/aclop) to the NpuDevice class.
 */

 #ifndef FASTLLM_ASCEND_H
 #define FASTLLM_ASCEND_H
 
 #include "fastllm.h" // 必须包含，用于识别 fastllm::Data
 #include <vector>
 
 namespace fastllm {
 
     // =======================================================================
     // 1. 基础资源管理 (Initialization & Memory)
     // =======================================================================
     
     void FastllmAclInit(); 
     void* FastllmAclMalloc(size_t size);
     void FastllmAclFree(void* ptr);
     
     // Memory Copy
     void FastllmAclCopyFromHostToDevice(void *dst, void *src, size_t size);
     void FastllmAclCopyFromDeviceToHost(void *dst, void *src, size_t size);
     void FastllmAclCopyFromDeviceToDevice(void *dst, void *src, size_t size);
     void FastllmAclMemcpy2DDeviceToDevice(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height);
 
     // =======================================================================
     // 2. 核心计算算子 (Core Math)
     // =======================================================================
 
     void FastllmAclMatMul(const fastllm::Data &input, const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int alpha, int beta);
     void FastllmAclMatMulTransB(const fastllm::Data &input, const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int alpha, int beta);
     
     // W8A8 量化计算 (注意：输入 input/weight 在定义中是非 const 的，与你的 extern 保持一致)
     void FastllmAclQuantLinearDequant(fastllm::Data &input, fastllm::Data &weight, fastllm::Data &weightScale, fastllm::Data &xScale, fastllm::Data &bias, fastllm::Data &output);
 
     // =======================================================================
     // 3. 激活与归一化 (Activation & Norm)
     // =======================================================================
 
     void FastllmAclRMSNorm(const fastllm::Data &input, const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, float eps);
     void FastllmAclSilu(const fastllm::Data &input, fastllm::Data &output);
     void FastllmAclSwiglu(const fastllm::Data &input, fastllm::Data &output);
     void FastllmAclSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis);
 
     // =======================================================================
     // 4. 基础运算 (Basic Math - aclop)
     // =======================================================================
 
     void FastllmAclAdd(const fastllm::Data &input, float v, fastllm::Data &output);
     void FastllmAclAddTo(const fastllm::Data &input0, const fastllm::Data &input1, float alpha);
     void FastllmAclMul(const fastllm::Data &input, float v, fastllm::Data &output);
     void FastllmAclMulTo(const fastllm::Data &input0, const fastllm::Data &input1, float alpha);
 
     // =======================================================================
     // 5. Tensor 变换 (Manipulation)
     // =======================================================================
 
     void FastllmAclPermute(const fastllm::Data &input, const std::vector<int> &axis);
     void FastllmAclRepeat(void *src, void *dst, int outer, int repeatTimes, int inputStride, int outputStride, int channelsInner, int channelsInputInner);
     void FastllmAclTopK(const fastllm::Data &input, fastllm::Data &output, int topk);
     void FastllmAclEmbedding(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
 
     // =======================================================================
     // 6. 类型转换 (Type Casting)
     // =======================================================================
 
     void FastllmAclFloatToHalf(float *src, void *dst, int len);
     void FastllmAclHalfToFloat(void *src, float *dst, int len);
 
     // =======================================================================
     // 7. 复杂算子 (Complex Ops)
     // =======================================================================
 
     void FastllmAclAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v, const fastllm::Data &mask, fastllm::Data &output, int group, float scale, int maskType);
     void FastllmAclAttentionMask(const fastllm::Data &input, const fastllm::Data &mask, float maskValue);
     void FastllmAclNearlyRotatePosition2D(const fastllm::Data &data, const fastllm::Data &positionIds, const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim);
     void FastllmAclRotatePosition2D_Fused(const Data &query, const Data &key, const Data &positionIds, const Data &sinData, const Data &cosData, int rotaryDim);
 } // namespace fastllm
 
 #endif // FASTLLM_ASCEND_H