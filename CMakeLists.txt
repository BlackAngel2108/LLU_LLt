cmake_minimum_required(VERSION 3.23.5)

# Возможные компиляторы для выбора
set(CMAKE_C_COMPILER "gcc" CACHE STRING "C compiler")
set(CMAKE_CXX_COMPILER "g++" CACHE STRING "C++ compiler")

# Intel компилятор можно будет выбрать через интерфейс
option(USE_INTEL "Use Intel compiler" OFF)
if(USE_INTEL)
    set(CMAKE_C_COMPILER "icx")
    set(CMAKE_CXX_COMPILER "icpx")
endif()

set(PROJECT_NAME LU_LLt_project)
project(${PROJECT_NAME})

# set(CMAKE_CXX_COMPILER "icpx")
# set(CMAKE_C_COMPILER "icx")

if(COMPILER_SUPPORTS_CXX17)
    message( STATUS "CXX17_supported")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Поиск OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    # Проверяем, что компилятор Intel (в разных вариантах имени)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" OR 
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
       CMAKE_CXX_COMPILER MATCHES "icpx")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        message(STATUS "Using Intel OpenMP flag: -qopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "Using standard OpenMP flags: ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Поиск OpenBLAS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(BLAS_FOUND)
    message( STATUS "BLAS_found: ${BLAS_LIBRARIES}")
    message( STATUS "LAPACK_found: ${LAPACK_LIBRARIES}")
endif()

find_package(LAPACK REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LAPACKE REQUIRED lapacke)

if(LAPACK_FOUND)
    message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
    message(STATUS "LAPACKE found: ${LAPACKE_LIBRARIES}")
endif()

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE})

set(LU_LIBRARY "${PROJECT_NAME}")
set(LU_TESTS   "test_${PROJECT_NAME}")
set(LU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${LU_INCLUDE}" gtest)

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
message( STATUS "   Compiler: ${CMAKE_CXX_COMPILER}")
message( STATUS "   BLAS: ${BLAS_LIBRARIES}")
message( STATUS "   LAPACK: ${LAPACK_LIBRARIES}")
message( STATUS "")

if( MSVC )
  add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()