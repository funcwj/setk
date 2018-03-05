cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_STANDARD 11)
aux_source_directory(${CMAKE_SOURCE_DIR}/include STFT)

# add_executable(test-stft test-stft.cc stft.cc)
add_executable(compute-stft-stats compute-stft-stats.cc ${STFT})
add_executable(compute-masks compute-masks.cc ${STFT})
add_executable(wav-seperate wav-seperate.cc ${STFT})
add_executable(wav-estimate wav-estimate.cc ${STFT})

# target_link_libraries(test-stft ${DEPEND_LIBS})
target_link_libraries(compute-stft-stats ${DEPEND_LIBS})
target_link_libraries(compute-masks ${DEPEND_LIBS})
target_link_libraries(wav-seperate ${DEPEND_LIBS})
target_link_libraries(wav-estimate ${DEPEND_LIBS})

