cmake_minimum_required(VERSION 3.18)
project(dgnn CXX)
enable_language(CUDA)

set(TARGET_LIB "dgnn")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set cuda arch
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
message(STATUS "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")


# rmm
find_package(rmm REQUIRED)

find_package(PythonLibs REQUIRED)
add_subdirectory(third_party/pybind11)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/dgnn/src)
include_directories($ENV{CONDA_PREFIX}/include)
include_directories(/usr/local/cuda/include)
include_directories(third_party/pybind11/include)

file(GLOB_RECURSE DGNN_SRC_FILES ${PROJECT_SOURCE_DIR}/dgnn/src/*.cc)
set_source_files_properties(${PROJECT_SOURCE_DIR}/dgnn/src/api.cc PROPERTIES LANGUAGE CUDA)
file(GLOB_RECURSE DGNN_SRC_CUDA_FILES ${PROJECT_SOURCE_DIR}/dgnn/src/*.cu)
set_source_files_properties(${DGNN_SRC_CUDA_FILES} PROPERTIES LANGUAGE CUDA)
list(APPEND DGNN_SRC_FILES ${DGNN_SRC_CUDA_FILES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")

pybind11_add_module(${TARGET_LIB} ${DGNN_SRC_FILES})
