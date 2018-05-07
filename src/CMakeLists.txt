cmake_minimum_required(VERSION 3.4)

add_executable(compute-stft-stats compute-stft-stats.cc)
add_executable(compute-masks compute-masks.cc)
add_executable(wav-seperate wav-seperate.cc)
add_executable(wav-estimate wav-estimate.cc)
add_executable(perform-snmf perform-snmf.cc)
add_executable(apply-supervised-mvdr apply-supervised-mvdr.cc)
add_executable(apply-supervised-max-snr apply-supervised-max-snr.cc)

target_link_libraries(compute-stft-stats ${DEPEND_LIBS} setk)
target_link_libraries(compute-masks ${DEPEND_LIBS} setk)
target_link_libraries(wav-seperate ${DEPEND_LIBS} setk)
target_link_libraries(wav-estimate ${DEPEND_LIBS} setk)
target_link_libraries(perform-snmf ${DEPEND_LIBS} setk)
target_link_libraries(apply-supervised-mvdr ${DEPEND_LIBS} setk)
target_link_libraries(apply-supervised-max-snr ${DEPEND_LIBS} setk)

