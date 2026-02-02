cmake_minimum_required(VERSION 3.10)
project(cppOps_standalone LANGUAGES CXX)

option(USE_ASCEND "Enable Ascend NPU backend" ON)  # 默认 ON，你也可以改 OFF

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O2 -g")

# 1. 路径设置
get_filename_component(FASTLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)

# 2. fastllm 公共头文件
include_directories(${FASTLLM_ROOT}/include)
include_directories(${FASTLLM_ROOT}/include/utils)
include_directories(${FASTLLM_ROOT}/include/models)
include_directories(${FASTLLM_ROOT}/include/devices/cpu)
include_directories(${FASTLLM_ROOT}/third_party/json11)
include_directories(${FASTLLM_ROOT}/third_party/gguf)

# 3. fastllm 核心源文件
set(FASTLLM_CORE_SOURCES
    "${FASTLLM_ROOT}/src/fastllm.cpp"
    "${FASTLLM_ROOT}/src/device.cpp"
    "${FASTLLM_ROOT}/src/executor.cpp"
    "${FASTLLM_ROOT}/src/template.cpp"
    "${FASTLLM_ROOT}/src/tokenizer.cpp"

    "${FASTLLM_ROOT}/third_party/json11/json11.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/gguf.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-quant.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/gguf-adapter.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-dequantize.cpp"
    "${FASTLLM_ROOT}/third_party/gguf/ggml-iqk.cpp"
)

file(GLOB CPU_DEVICE_SOURCES "${FASTLLM_ROOT}/src/devices/cpu/*.cpp")
list(APPEND FASTLLM_CORE_SOURCES ${CPU_DEVICE_SOURCES})

# 4. Ascend NPU（可选）
set(FASTLLM_NPU_SOURCES "")
set(ASCEND_LIBS "")

if (USE_ASCEND)
    add_compile_definitions(USE_ASCEND)

    if (DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
    message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

    include_directories(${ASCEND_PATH}/include)
    include_directories(${ASCEND_PATH}/include/aclnn)
    include_directories(${FASTLLM_ROOT}/include/devices/npu)

    set(FASTLLM_NPU_SOURCES
        "${FASTLLM_ROOT}/src/devices/npu/ascenddevice.cpp"
        "${FASTLLM_ROOT}/src/devices/npu/fastllm-ascend.cpp"
    )

    set(ASCEND_LIBS
        ${ASCEND_PATH}/lib64/libascendcl.so
        ${ASCEND_PATH}/lib64/libnnopbase.so
        ${ASCEND_PATH}/lib64/libopapi.so
        pthread
    )
endif()

# 5. 生成可执行文件
add_executable(cppOps
    cppOps.cpp
    ${FASTLLM_CORE_SOURCES}
    ${FASTLLM_NPU_SOURCES}
)

# 6. 链接库（仅 Ascend 时链接 Ascend so）
target_link_libraries(cppOps PRIVATE ${ASCEND_LIBS})
