project(pgbm)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -std=c++11")
# Based on the output of mpic++ -show
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wl,-rpath -Wl,/usr/lib/openmpi -Wl,--enable-new-dtags -L/usr/lib/openmpi -lmpi_cxx -lmpi")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# For glogs, gflags and folly
link_directories("/usr/lib")
include_directories("/usr/include")

# TODO:
# - How to properly include glog and gflags as submodules? Now retreat gflags
#   to pre-installed lib since the folly-git package requires that.
# - Why must full paths be specified for gflags and folly, but not glog?

# glog
set(MANDATORY_LIBS glog)

# gflags
set(MANDATORY_LIBS "${MANDATORY_LIBS} /usr/lib/libgflags.so")

# folly
set(MANDATORY_LIBS "${MANDATORY_LIBS} /usr/lib/libfolly.so")

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
