//
// Created by fastllm-npu-adapter
//

#include "devices/cpu/cpudevice.h"
#include "devices/npu/ascenddevice.h" 
#include "devices/npu/fastllm-ascend.h"
#include "utils.h"
#include <algorithm>

namespace fastllm {

    NpuDevice::NpuDevice() {
        this->deviceType = "Npu"; 
        
        // 1. 类型转换
        this->ops["ToFloat16"] = (BaseOperator*)(new NpuToFloat16());
        this->ops["ToFloat32"] = (BaseOperator*)(new NpuToFloat32());
        this->ops["ConvertToFloat16"] = (BaseOperator*)(new NpuConvertToFloat16());
        this->ops["ConvertToFloat32"] = (BaseOperator*)(new NpuConvertToFloat32());

        // 2. 核心计算
        this->ops["Attention"] = (BaseOperator*)(new NpuAttention());
        this->ops["Embedding"] = (BaseOperator*)(new NpuEmbedding());
        this->ops["RMSNorm"] = (BaseOperator*)(new NpuRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new NpuLinearOp());
        this->ops["QuantLinearDequant"] = (BaseOperator*)(new NpuQuantLinearDequantOp()); // W8A8 关键
        this->ops["MatMul"] = (BaseOperator*)(new NpuMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new NpuMatMulTransBOp());

        // 3. 激活与归一化
        this->ops["SoftMax"] = (BaseOperator*)(new NpuSoftMaxOp());
        this->ops["Silu"] = (BaseOperator*)(new NpuSiluOp());
        this->ops["Swiglu"] = (BaseOperator*)(new NpuSwigluOp());
        // this->ops["Sigmoid"] = (BaseOperator*)(new NpuSigmoidOp()); // 如果底层未实现 Sigmoid，先注释掉或回退 CPU

        // 4. 基础运算
        this->ops["Add"] = (BaseOperator*)(new NpuAddOp());
        this->ops["Mul"] = (BaseOperator*)(new NpuMulOp());
        this->ops["AddTo"] = (BaseOperator*)(new NpuAddToOp());
        this->ops["MulTo"] = (BaseOperator*)(new NpuMulToOp());
        
        // 5. Tensor 操作
        this->ops["Split"] = (BaseOperator*)(new NpuSplitOp());
        this->ops["Repeat"] = (BaseOperator*)(new NpuRepeatOp());
        this->ops["Cat"] = (BaseOperator*)(new NpuCatOp());
        this->ops["CatDirect"] = (BaseOperator*)(new NpuCatDirectOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new NpuPermuteSelfOp());
        this->ops["TopK"] = (BaseOperator*)(new NpuTopKOp());

        // 6. 其他
        this->ops["AttentionMask"] = (BaseOperator*)(new NpuAttentionMaskOp());
        // 注册 RoPE 算子 (LlamaRotatePosition2D 是 DeepSeek 用的名字)
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new NpuNearlyRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D_Fused"] = (BaseOperator*)(new NpuNearlyRotatePosition2DFusedOp());
        
        // 初始化 ACL
        FastllmAclInit();
    }

    bool NpuDevice::Malloc(void **ret, size_t size) {
        *ret = FastllmAclMalloc(size);
        return true;
    }

    bool NpuDevice::Free(void *ret) {
        FastllmAclFree(ret);
        return true;
    }

    bool NpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        FastllmAclCopyFromHostToDevice(dst, src, size);
        return true;
    }

    bool NpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        FastllmAclCopyFromDeviceToHost(dst, src, size);
        return true;
    }

    // --- 类型转换算子实现 ---

    void NpuToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT16) return;
        
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            return;
        }

        if (data.dataType == DataType::FLOAT32) {
            float *old = (float*)data.deviceData; 
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            data.deviceData = FastllmAclMalloc(data.GetBytes());
            int len = data.Count(0);
            FastllmAclFloatToHalf(old, data.deviceData, len);
            FastllmAclFree(old);
        } else {
            ErrorInFastLLM("NpuToFloat16: unsupport dataType.\n");
        }
    }

    void NpuToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT32) return;

        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            return;
        }

        if (data.dataType == DataType::FLOAT16) {
            void *old = data.deviceData;
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            data.deviceData = FastllmAclMalloc(data.GetBytes());
            int len = data.Count(0);
            FastllmAclHalfToFloat(old, (float*)data.deviceData, len);
            FastllmAclFree(old);
        } else {
            ErrorInFastLLM("NpuToFloat32: unsupport dataType.\n");
        }
    }

    void NpuConvertToFloat16::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT16;
        output->Resize(input->dims);
        if (input->expansionDims.size() != 0) output->Expansion(input->expansionDims);
    }

    void NpuConvertToFloat16::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT16) {
            FastllmAclCopyFromDeviceToDevice(output.deviceData, input.deviceData, input.GetBytes());
        } else if (input.dataType == DataType::FLOAT32) {
            FastllmAclFloatToHalf((float*)input.deviceData, output.deviceData, input.Count(0));
        } else {
            ErrorInFastLLM("NpuConvertToFloat16: unsupport dataType.\n");
        }
    }
    
    void NpuConvertToFloat32::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT32;
        output->Resize(input->dims);
        if (input->expansionDims.size() != 0) output->Expansion(input->expansionDims);
    }

    void NpuConvertToFloat32::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT32) {
            FastllmAclCopyFromDeviceToDevice(output.deviceData, input.deviceData, input.GetBytes());
        } else if (input.dataType == DataType::FLOAT16) {
            FastllmAclHalfToFloat(input.deviceData, (float*)output.deviceData, input.Count(0));
        } else {
            ErrorInFastLLM("NpuConvertToFloat32: unsupport dataType.\n");
        }
    }

    // --- 核心计算算子实现 ---

    void DoNpuAttentionReshape(Data &q, Data &v, Data &output) {
        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    void NpuAttention::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        DoNpuAttentionReshape(q, v, output);
    }

    void NpuAttention::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data emptyData;
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &mask = datas.find("mask") != datas.end() ? *(datas.find("mask")->second) : emptyData; // Fix check
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        int maskType = intParams.find("maskType") != intParams.end() ? intParams.find("maskType")->second : 0;

        output.Allocate();
        FastllmAclAttention(q, k, v, mask, output, group, scale, maskType);
    }

    bool NpuEmbedding::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        // NPU supports INT32 inputs for embedding
        return (input.dataType == DataType::FLOAT32); 
    }

    void NpuEmbedding::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        FastllmAclEmbedding(input, weight, output);
    }

    bool NpuRMSNormOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        return true;
    }

    void NpuRMSNormOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        Data emptyBias;
        output.Allocate();
        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-6;
        FastllmAclRMSNorm(input, weight, emptyBias, output, eps);
    }

    void DoNpuLinearReshape(Data &input, Data &weight, Data &output) {
        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    bool NpuLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) return false;
        return true;
    }

    void NpuLinearOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        DoNpuLinearReshape(input, weight, output);
    }

    void NpuLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        Data emptyBias;
        Data &bias = datas.find("bias") != datas.end() ? *(datas.find("bias")->second) : emptyBias;
        
        output.Allocate();
        FastllmAclMatMulTransB(input, weight, bias, output, 1, 0); 
    }

    // W8A8 量化 Linear
    bool NpuQuantLinearDequantOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        return datas.find("input") != datas.end() && datas.find("weight") != datas.end();
    }
    
    void NpuQuantLinearDequantOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second); // [N, K]
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];
        output.dataType = DataType::FLOAT16; // NPU output is FP16
        output.Resize(dims);
    }
    
    void NpuQuantLinearDequantOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        Data &weightScale = *(datas.find("weightScale")->second);
        Data &xScale = *(datas.find("inputScale")->second);
        Data emptyBias;
        Data &bias = datas.find("bias") != datas.end() ? *(datas.find("bias")->second) : emptyBias; // Bias support
        
        output.Allocate();
        FastllmAclQuantLinearDequant(input, weight, weightScale, xScale, bias, output);
    }

    void NpuMatMulOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims.back();
        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void NpuMatMulOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        Data emptyBias;
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        
        output.Allocate();
        FastllmAclMatMul(input0, input1, emptyBias, output, alpha, 0);
    }
    
    void NpuMatMulTransBOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[0]; 
        output.dataType = input0.dataType;
        output.Resize(dims);
    }
    
    void NpuMatMulTransBOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        Data emptyBias;
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        
        output.Allocate();
        FastllmAclMatMulTransB(input0, input1, emptyBias, output, alpha, 0);
    }


    void NpuSiluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        FastllmAclSilu(input, output);
    }

    void NpuSwigluOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        std::vector <int> dims = input.dims;
        dims.back() /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void NpuSwigluOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        FastllmAclSwiglu(input, output);
    }
    
    bool NpuSoftMaxOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        return input.Count(axis + 1) == 1; 
    }

    void NpuSoftMaxOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        output.Allocate();
        FastllmAclSoftmax(input, output, axis);
    }


    void NpuAddOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        output.Allocate();
        FastllmAclAdd(input, v, output);
    }
    
    void NpuAddToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
        FastllmAclAddTo(input0, input1, alpha);
    }

    void NpuMulOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        output.Allocate();
        FastllmAclMul(input, v, output);
    }
    
    void NpuMulToOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
        FastllmAclMulTo(input0, input1, alpha);
    }


    void DoNpuSplitReshape(Data &input, int axis, int start, int end, Data &output) {
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void DoNpuSplit(Data &input, int axis, int start, int end, Data &output) {
        output.Allocate();
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        
        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        FastllmAclMemcpy2DDeviceToDevice((uint8_t*)output.deviceData, outputStride * unitSize,
                                         (uint8_t*)input.deviceData + start * inner * unitSize, inputStride * unitSize,
                                         (end - start) * inner * unitSize, outer);
    }

    void NpuSplitOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;
        DoNpuSplitReshape(input, axis, start, end, output);
    }

    void NpuSplitOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;
        DoNpuSplit(input, axis, start, end, output);
    }

    void NpuRepeatOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;
        
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        output.Allocate();

        int outer = output.Count(0) / output.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;
        
        FastllmAclRepeat(input.deviceData, output.deviceData, outer, repeatTimes, inputStride * unitSize, outputStride * unitSize, channels * inner * unitSize, channels * inner * unitSize);
    }

    void NpuCatDirectOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        
        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            FastllmAclMemcpy2DDeviceToDevice((uint8_t *) input0.deviceData, input0Stride * unitSize,
                                             (uint8_t *) input1.deviceData, input1Stride * unitSize,
                                             input1.dims[axis] * inner * unitSize, outer);
            return;
        }

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        std::vector<int> dims = input0.dims;
        std::vector<int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmAclMemcpy2DDeviceToDevice((uint8_t *) input0.deviceData + oldDims[axis] * inner * unitSize,
                                         input0Stride * unitSize,
                                         (uint8_t *) input1.deviceData, input1Stride * unitSize,
                                         input1.dims[axis] * inner * unitSize, outer);
    }

    void NpuCatOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = output.Count(0) / output.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        FastllmAclMemcpy2DDeviceToDevice((uint8_t *) output.deviceData, outputStride * unitSize,
                                         (uint8_t *) input0.deviceData, input0Stride * unitSize,
                                         input0.dims[axis] * inner * unitSize, outer);
        FastllmAclMemcpy2DDeviceToDevice((uint8_t *) output.deviceData + input0.dims[axis] * inner * unitSize, outputStride * unitSize,
                                         (uint8_t *) input1.deviceData, input1Stride * unitSize,
                                         input1.dims[axis] * inner * unitSize, outer);
    }

    void NpuPermuteSelfOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }
        FastllmAclPermute(input, axis);
    }

    void NpuTopKOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        std::vector<int> dims = input.dims;
        dims.back() = topk * 2; 
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    bool NpuTopKOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        return topk <= 100;
    }

    void NpuTopKOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        output.Allocate();
        FastllmAclTopK(input, output, topk);
    }

    void NpuAttentionMaskOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        FastllmAclAttentionMask(input, mask, maskValue);
    }

    void NpuNearlyRotatePosition2DOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        FastllmAclNearlyRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }

    void NpuNearlyRotatePosition2DFusedOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &query = *(datas.find("query")->second);
        Data &key = *(datas.find("key")->second);
        
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);

        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        FastllmAclRotatePosition2D_Fused(query, key, positionIds, sinData, cosData, rotaryDim);
    }

}