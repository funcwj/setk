cmake_minimum_required(VERSION 3.0)

add_executable(test-stft test-stft.cc)
add_executable(test-srp-phat test-srp-phat.cc)
add_executable(test-complex test-complex.cc)
add_executable(test-beamformer test-beamformer.cc)

target_link_libraries(test-stft ${DEPEND_LIBS} setk)
target_link_libraries(test-srp-phat ${DEPEND_LIBS} setk)
target_link_libraries(test-complex ${DEPEND_LIBS} setk)
target_link_libraries(test-beamformer ${DEPEND_LIBS} setk)

