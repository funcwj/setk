cmake_minimum_required(VERSION 3.4)
project(kaldi_enhan)

set(CMAKE_CXX_STANDARD 11)

set(KALDI_DIR /gruntdata/jianwu.wj/kaldi)
set(DEPEND_LIBS kaldi-base kaldi-util kaldi-matrix kaldi-feat)

add_definitions(-O3 -g -std=c++11 -DHAVE_CLAPACK)
include_directories(${KALDI_DIR}/tools/CLAPACK ${KALDI_DIR}/src ${KALDI_DIR}/tools/openfst/include ${CMAKE_SOURCE_DIR})
link_directories(${KALDI_DIR}/src/lib)

SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
add_subdirectory(src)
