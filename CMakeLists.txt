cmake_minimum_required(VERSION 3.20.2)

# ============================================================
# Project
# ============================================================

set(PROJECT_NAME LU_LLt_project)
project(${PROJECT_NAME} C CXX)

# ============================================================
# Compiler selection
# ============================================================

option(USE_INTEL "Use Intel compiler" OFF)

if(USE_INTEL)
    set(CMAKE_C_COMPILER "icx" CACHE STRING "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "icpx" CACHE STRING "C++ compiler" FORCE)
else()
    set(CMAKE_C_COMPILER "gcc" CACHE STRING "C compiler")
    set(CMAKE_CXX_COMPILER "g++" CACHE STRING "C++ compiler")
endif()

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
# Output directories
# ============================================================

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

# ============================================================
# Project paths
# ============================================================

set(LU_LIBRARY "LU_LLt_lib")
set(LU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LU_TESTS "test_${PROJECT_NAME}")

# ============================================================
# OpenMP
# ============================================================

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    if(USE_INTEL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -qopenmp")
        message(STATUS "Using Intel OpenMP flag: -qopenmp")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        message(STATUS "Using standard OpenMP flags: ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# ============================================================
# BLAS / LAPACK (optional)
# ============================================================

option(USE_BLAS "Use BLAS/LAPACK libraries" OFF)

# Всегда инициализируем переменные
set(BLAS_FOUND OFF)
set(LAPACK_FOUND OFF)
set(LAPACKE_FOUND OFF)

if(USE_BLAS)
    find_package(BLAS)
    find_package(LAPACK)
    find_package(PkgConfig QUIET)
    
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(LAPACKE lapacke)
    endif()
    
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
        message(STATUS "LAPACK found: ${LAPACK_LIBRARIES}")
        if(LAPACKE_FOUND)
            message(STATUS "LAPACKE found: ${LAPACKE_LIBRARIES}")
        endif()
        add_compile_definitions(USE_BLAS)
    else()
        message(WARNING "BLAS/LAPACK not found. Building without BLAS support.")
        set(USE_BLAS OFF)
    endif()
else()
    message(STATUS "BLAS/LAPACK disabled by user (USE_BLAS=OFF)")
endif()

# ============================================================
# Intel specific optimizations
# ============================================================

if(USE_INTEL)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -xHost -qopt-zmm-usage=high -ffast-math")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -xHost -qopt-zmm-usage=high -ffast-math")
endif()

# ============================================================
# MSVC
# ============================================================

if(MSVC)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
endif()

# ============================================================
# Enable testing
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
# Final report
# ============================================================

message(STATUS "")
message(STATUS "General configuration for ${PROJECT_NAME}")
message(STATUS "======================================")
message(STATUS "  Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler:    ${CMAKE_C_COMPILER}")
message(STATUS "  CXX compiler:  ${CMAKE_CXX_COMPILER}")
message(STATUS "  CXX standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  USE_INTEL:     ${USE_INTEL}")
message(STATUS "  USE_BLAS:      ${USE_BLAS}")
if(USE_BLAS)
    message(STATUS "  BLAS:          ${BLAS_LIBRARIES}")
    message(STATUS "  LAPACK:        ${LAPACK_LIBRARIES}")
endif()
message(STATUS "")