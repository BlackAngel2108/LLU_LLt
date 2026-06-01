# ============================================================
# Sources
# ============================================================

set(target ${LU_LIBRARY})
set(SOURCES matrix.cpp)
# Собираем все .cpp файлы в текущей директории
file(GLOB srcs "*.cpp")

# Убираем benchmarks.cpp (он будет отдельным исполняемым файлом)
list(REMOVE_ITEM srcs "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks.cpp")

# ============================================================
# Основная библиотека
# ============================================================

add_library(${target} ${srcs})

target_include_directories(${target} PUBLIC ${LU_INCLUDE})

if(OPENMP_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_BLAS AND BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(${target} PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_compile_definitions(${target} PUBLIC USE_BLAS USE_LAPACK)
endif()

# ============================================================
# Бенчмарки (исполняемый файл)
# ============================================================

add_executable(${PROJECT_NAME}_benchmarks benchmarks.cpp)

target_link_libraries(${PROJECT_NAME}_benchmarks PRIVATE ${target})

if(OPENMP_FOUND)
    target_link_libraries(${PROJECT_NAME}_benchmarks PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_BLAS AND BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(${PROJECT_NAME}_benchmarks PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME}_benchmarks PUBLIC USE_BLAS USE_LAPACK)
endif()

# LAPACKE (опционально)
if(USE_BLAS AND LAPACKE_FOUND)
    target_link_libraries(${PROJECT_NAME}_benchmarks PRIVATE ${LAPACKE_LIBRARIES})
    target_include_directories(${PROJECT_NAME}_benchmarks PRIVATE ${LAPACKE_INCLUDE_DIRS})
endif()

target_include_directories(${PROJECT_NAME}_benchmarks PRIVATE ${LU_INCLUDE})

set_target_properties(${PROJECT_NAME}_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)