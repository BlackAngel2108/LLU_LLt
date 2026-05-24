cmake_minimum_required(VERSION 3.23.5)

# ============================================================
# Project
# ============================================================

set(PROJECT_NAME LU_LLt_project)

# Intel компилятор можно выбрать через:
# cmake -B build -DUSE_INTEL=ON
option(USE_INTEL "Use Intel compiler" OFF)

if(USE_INTEL)
    set(CMAKE_C_COMPILER "icx" CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "C++ compiler" FORCE)
else()
    set(CMAKE_C_COMPILER "gcc" CACHE STRING "C compiler")
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "C++ compiler")
endif()

project(${PROJECT_NAME} C CXX)

# ============================================================
# C++ standard
# ============================================================

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Build type
# ============================================================

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# RISC-V Vector intrinsics support
# ============================================================

option(ENABLE_RVV "Enable RISC-V Vector extension intrinsics" OFF)

if(ENABLE_RVV)
    set(RVV_FLAGS
        -march=rv64gcv_zvfh
        -mabi=lp64d
    )

    add_compile_options(${RVV_FLAGS})
    add_link_options(${RVV_FLAGS})
    add_compile_definitions(USE_RVV)

    message(STATUS "RVV enabled")
    message(STATUS "RVV flags: ${RVV_FLAGS}")
endif()

# ============================================================
# BLAS / LAPACK
# ============================================================

#option(USE_BLAS "Use BLAS" ON)
#option(USE_LAPACK "Use LAPACK" ON)

#if(CMAKE_CROSSCOMPILING)
 #   set(USE_BLAS OFF CACHE BOOL "Disable BLAS for cross-compilation" FORCE)
  #  set(USE_LAPACK OFF CACHE BOOL "Disable LAPACK for cross-compilation" FORCE)

   # message(STATUS "Cross-compiling: BLAS/LAPACK disabled")
#endif()

#if(USE_BLAS)
 #   find_package(BLAS REQUIRED)

  #  if(BLAS_FOUND)
   #     message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    #endif()
#endif()

#if(USE_LAPACK)
 #   find_package(LAPACK REQUIRED)
  #  find_package(PkgConfig REQUIRED)
   # pkg_check_modules(LAPACKE REQUIRED lapacke)

 #   if(LAPACK_FOUND)
  #      message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
   # endif()

 #   if(LAPACKE_FOUND)
  #      message(STATUS "LAPACKE found: ${LAPACKE_LIBRARIES}")
   # endif()
#endif()

# ============================================================
# OpenMP
# ============================================================

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel" OR
       CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
       CMAKE_CXX_COMPILER MATCHES "icpx")

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")

        message(STATUS "Using Intel OpenMP flag: -qopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

        message(STATUS "Using standard OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
        message(STATUS "Using standard OpenMP C flags: ${OpenMP_C_FLAGS}")
    endif()
endif()

# ============================================================
# Intel release optimization flags
# ============================================================

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost -qopt-zmm-usage=high -ffast-math")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -xHost -qopt-zmm-usage=high -ffast-math")
endif()

# ============================================================
# Output directories
# ============================================================

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

# ============================================================
# Project variables
# ============================================================

set(LU_LIBRARY "${PROJECT_NAME}")
set(LU_TESTS   "test_${PROJECT_NAME}")
set(LU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories("${LU_INCLUDE}" gtest)

# ============================================================
# Testing
# ============================================================

enable_testing()

# ============================================================
# Subdirectories
# ============================================================

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(gtest)
add_subdirectory(test)
add_subdirectory(benchmark)

# ============================================================
# MSVC
# ============================================================

if(MSVC)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()

# ============================================================
# Report
# ============================================================

message(STATUS "")
message(STATUS "General configuration for ${PROJECT_NAME}")
message(STATUS "======================================")
message(STATUS "")
message(STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "   C compiler:    ${CMAKE_C_COMPILER}")
message(STATUS "   CXX compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "   CXX standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "   ENABLE_RVV:    ${ENABLE_RVV}")
message(STATUS "   USE_BLAS:      ${USE_BLAS}")
message(STATUS "   USE_LAPACK:    ${USE_LAPACK}")
message(STATUS "   BLAS:          ${BLAS_LIBRARIES}")
message(STATUS "   LAPACK:        ${LAPACK_LIBRARIES}")
message(STATUS "")