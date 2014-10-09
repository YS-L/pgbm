include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(susy susy.cc)
target_link_libraries(susy pgbm ${MANDATORY_LIBS})
