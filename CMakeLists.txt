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

include_directories(${PROJECT_SOURCE_DIR}/dgnn/src)
include_directories($ENV{CONDA_PREFIX}/include)
include_directories(/usr/local/cuda/include)

# Get Python suffix
execute_process(COMMAND ${PY_EXE} -c "import sysconfig; print(next(x for x in [sysconfig.get_config_var('EXT_SUFFIX'), sysconfig.get_config_var('SO'), '.so'] if x))"
    OUTPUT_VARIABLE Python_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
message(STATUS "Python suffix is ${Python_SUFFIX}")

file(GLOB_RECURSE DGNN_SRC_FILES ${PROJECT_SOURCE_DIR}/dgnn/src/*.cc)
list(APPEND SOURCES ${DGNN_SRC_FILES})

file(GLOB_RECURSE DGNN_SRC_CUDA_FILES ${PROJECT_SOURCE_DIR}/dgnn/src/*.cu)
set_source_files_properties(${DGNN_SRC_CUDA_FILES} PROPERTIES LANGUAGE CUDA)
list(APPEND SOURCES ${DGNN_SRC_CUDA_FILES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -ftree-vectorize")
set(ARCH_FLAGS "-march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")


add_library(${TARGET_LIB} SHARED ${SOURCES})
set_target_properties(${TARGET_LIB} PROPERTIES SUFFIX "${Python_SUFFIX}")
set_target_properties(${TARGET_LIB} PROPERTIES PREFIX "")
set_target_properties(${TARGET_LIB} PROPERTIES OUTPUT_NAME "")

# cpp tests
enable_testing()
add_subdirectory(./third_party/googletest)
include_directories(./third_party/googletest/googletest/include)
file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/src/*.cc)
add_executable(runUnitTests ${TEST_SRC_FILES})
target_link_libraries(runUnitTests gtest gtest_main)
target_link_libraries(runUnitTests ${TARGET_LIB})
add_test(UnitTests runUnitTests)
