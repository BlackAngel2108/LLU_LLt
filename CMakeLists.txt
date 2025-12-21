cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME LUDecompositionProject)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if(OPENMP_FOUND)
    message( STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LIB_NAME "matrix_lib")
set(TEST_NAME "run_tests")
set(PROJ_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${PROJ_INCLUDE}" gtest)

# Build subdirectories
add_subdirectory(src)
add_subdirectory(gtest)
add_subdirectory(test)

message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
if( MSVC )
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()

# Это должно правильно настроить зависимость для MSBuild.
gtest_discover_tests(run_tests)

# --- Benchmark Executable ---
add_executable(benchmark_runner benchmark/main_benchmark.cpp)
target_link_libraries(benchmark_runner PRIVATE
    matrix_lib
)

