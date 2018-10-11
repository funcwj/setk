cmake_minimum_required(VERSION 3.0)

set(SETK_SRC ${CMAKE_SOURCE_DIR}/include/complex-matrix.cc
             ${CMAKE_SOURCE_DIR}/include/complex-vector.cc
             ${CMAKE_SOURCE_DIR}/include/stft.cc
             ${CMAKE_SOURCE_DIR}/include/srp-phat.cc
             ${CMAKE_SOURCE_DIR}/include/rir-generator.cc
             ${CMAKE_SOURCE_DIR}/include/beamformer.cc)
if(APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-framework Accelerate")
endif()

add_library(setk SHARED ${SETK_SRC})
target_link_libraries(setk ${DEPEND_LIBS})
