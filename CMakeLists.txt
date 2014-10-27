project(pgbm)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
# Based on the output of mpic++ -show
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wl,-rpath -Wl,/usr/lib/openmpi -Wl,--enable-new-dtags -L/usr/lib/openmpi -lmpi_cxx -lmpi")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# TODO: How to properly include these two as submodules?

# glog
set(MANDATORY_LIBS glog)

# gflags
add_subdirectory(gflags)
include_directories("${PROJECT_BINARY_DIR}/gflags/include")
set(MANDATORY_LIBS "${MANDATORY_LIBS} ${PROJECT_BINARY_DIR}/gflags/lib/libgflags.a")
# BUt the following does not work, not like glog?
#find_package (gflags REQUIRED)
#include_directories (${gflags_INCLUDE_DIR})
#set(MANDATORY_LIBS "${MANDATORY_LIBS} libgflags.a")

# Boost MPI
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.56 COMPONENTS serialization mpi REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(MANDATORY_LIBS "${MANDATORY_LIBS} ${Boost_LIBRARIES}")

add_subdirectory(gtest-1.7.0)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
