cmake_minimum_required(VERSION 3.10)
project(test_npu_eb_repeat_standalone)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -g")

# 1. 路径设置
get_filename_component(FASTLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)

# 2. Ascend NPU 环境
add_compile_definitions(USE_ASCEND)
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif()
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

include_directories(${ASCEND_PATH}/include)
include_directories(${ASCEND_PATH}/include/aclnn)

# 3. 包含头文件
include_directories(${FASTLLM_ROOT}/include)
include_directories(${FASTLLM_ROOT}/include/utils)
include_directories(${FASTLLM_ROOT}/include/models) 
include_directories(${FASTLLM_ROOT}/include/devices/cpu)
include_directories(${FASTLLM_ROOT}/include/devices/npu)
include_directories(${FASTLLM_ROOT}/third_party/json11)
include_directories(${FASTLLM_ROOT}/third_party/gguf)

# 4. 收集源文件
set(FASTLLM_CORE_SOURCES
    "${FASTLLM_ROOT}/src/fastllm.cpp" 
    "${FASTLLM_ROOT}/src/device.cpp"
    "${FASTLLM_ROOT}/src/executor.cpp"
    "${FASTLLM_ROOT}/src/template.cpp"
    "${FASTLLM_ROOT}/src/tokenizer.cpp"
    
    # 第三方库
    "${FASTLLM_ROOT}/third_party/json11/json11.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/gguf.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-quant.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/gguf-adapter.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-dequantize.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-iqk.cpp"
)

# [关键修正] 收集 src/devices/cpu 下的所有 cpp 文件
# 这包括 cpudevice.cpp, kernel_*.cpp, computeutils.cpp, amx.cpp 等
file(GLOB CPU_DEVICE_SOURCES "${FASTLLM_ROOT}/src/devices/cpu/*.cpp")
list(APPEND FASTLLM_CORE_SOURCES ${CPU_DEVICE_SOURCES})

# NPU 后端
set(FASTLLM_NPU_SOURCES
    "${FASTLLM_ROOT}/src/devices/npu/ascenddevice.cpp"
    "${FASTLLM_ROOT}/src/devices/npu/fastllm-ascend.cpp"
)

# 5. 生成可执行文件
add_executable(test_npu_eb_repeat 
    test_npu_eb_repeat.cpp 
    ${FASTLLM_CORE_SOURCES} 
    ${FASTLLM_NPU_SOURCES}
)

# 6. 链接库
target_link_libraries(test_npu_eb_repeat PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libopapi.so
    pthread
)