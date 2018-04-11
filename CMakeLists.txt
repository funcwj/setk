cmake_minimum_required(VERSION 3.4)
project("SETK: speech enhancement tookit on kaldi")

set(CMAKE_CXX_STANDARD 11)

MESSAGE(STATUS "detected OS: ${CMAKE_SYSTEM}")  

set(KALDI_ROOT $ENV{KALDI_ROOT})
if(NOT DEFINED KALDI_ROOT)
    message(FATAL_ERROR "Can not find definition of KALDI_ROOT, 
                         export KALDI_ROOT=/path/to/kaldi first")
endif()

message(STATUS "defined KALDI_ROOT=" $ENV{KALDI_ROOT})
set(DEPEND_LIBS kaldi-base kaldi-util kaldi-matrix kaldi-feat)

add_definitions(-O3 -g -std=c++11)
include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/tools/openfst/include ${CMAKE_SOURCE_DIR})
link_directories(${KALDI_ROOT}/src/lib ${CMAKE_SOURCE_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

if(APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
    add_definitions(-DHAVE_CLAPACK)
    include_directories(${KALDI_ROOT}/tools/CLAPACK)
elseif(UNIX)
    add_definitions(-DHAVE_OPENBLAS)
    set(OPENBLAS_ROOT $ENV{OPENBLAS_ROOT})
    if(NOT DEFINED OPENBLAS_ROOT)
        message(FATAL_ERROR "Can not find definition of OPENBLAS_ROOT, 
                             export OPENBLAS_ROOT=/path/to/openfst first")
    endif()
    include_directories(${OPENBLAS_ROOT}/include)
    link_directories(${OPENBLAS_ROOT}/lib)
    set(DEPEND_LIBS ${DEPEND_LIBS} openblas)
    message(STATUS "defined OPENBLAS_ROOT=" $ENV{OPENBLAS_ROOT})
endif()

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)
