cmake_minimum_required(VERSION 2.8.11)

set(PROJECT_NAME LLU_LLt_project)
project(${PROJECT_NAME})

#set(CMAKE_CXX_COMPILER "/common/software/gcc-9.5.0/bin/g++")
#set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++")
if(COMPILER_SUPPORTS_CXX17)
    message( STATUS "CXX17_supported")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message( STATUS "OpenMP_found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# TODO(Kornyakov): not sure if these lines are needed
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(LLU_LIBRARY "${PROJECT_NAME}")
#set(LLU_CUSTOM "${PROJECT_NAME}")
set(LLU_TESTS   "test_${PROJECT_NAME}")
set(LLU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${LLU_INCLUDE}" gtest)

enable_testing()

# BUILD
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(gtest)
add_subdirectory(test)
add_subdirectory(benchmark)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message( STATUS "")

if( MSVC ) # VS doesn't support correctly the tuples yet
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()
