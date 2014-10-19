include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(pass_histograms pass_histograms.cc)
target_link_libraries(pass_histograms pgbm ${MANDATORY_LIBS})
