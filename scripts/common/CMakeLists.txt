cmake_minimum_required(VERSION 3.14)
project(common)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../dgnn/common)

file(GLOB common_sources "*.cc")
foreach (source ${common_sources})
    get_filename_component(target ${source} NAME_WLE)
    add_executable(${target} ${source})
endforeach()
