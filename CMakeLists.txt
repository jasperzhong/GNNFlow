cmake_minimum_required(VERSION 3.18)
project(dgnn CXX)
enable_language(CUDA)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "60")

set(TARGET_LIB "dgnn")

# PYTHON
if(NOT PYTHON_EXECUTABLE)
    find_package(Python 3.8 COMPONENTS Interpreter REQUIRED)
    set(PY_EXE ${Python_EXECUTABLE})
else()
    set(PY_EXE ${PYTHON_EXECUTABLE})
endif()
message(STATUS "Using command ${PY_EXE}")

# rmm
find_package(rmm REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/dgnn/common)
include_directories($ENV{CONDA_PREFIX}/include)
include_directories(/usr/local/cuda/include)

# Get Python suffix
execute_process(COMMAND ${PY_EXE} -c "import sysconfig; print(next(x for x in [sysconfig.get_config_var('EXT_SUFFIX'), sysconfig.get_config_var('SO'), '.so'] if x))"
    OUTPUT_VARIABLE Python_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
message(STATUS "Python suffix is ${Python_SUFFIX}")

list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/dgnn/src/temporal_block.cc"
    "${PROJECT_SOURCE_DIR}/dgnn/src/logging.cc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")


add_library(${TARGET_LIB} SHARED ${SOURCES})
set_target_properties(${TARGET_LIB} PROPERTIES SUFFIX "${Python_SUFFIX}")
set_target_properties(${TARGET_LIB} PROPERTIES PREFIX "")
set_target_properties(${TARGET_LIB} PROPERTIES OUTPUT_NAME "")

