set(target ${LU_LIBRARY})

file(GLOB srcs "*.cpp")
list(REMOVE_ITEM srcs "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks.cpp")

add_library(${target} ${srcs})

target_include_directories(${target} PUBLIC ${LU_INCLUDE})

if(OPENMP_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endif()

# Добавляем BLAS и LAPACK
if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(${target} PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_compile_definitions(${target} PRIVATE -DUSE_BLAS -DUSE_LAPACK)
endif()

# Добавляем исполняемый файл для бенчмарков
add_executable(${PROJECT_NAME}_benchmarks benchmarks.cpp)

target_link_libraries(${PROJECT_NAME}_benchmarks PRIVATE ${target})

if(OPENMP_FOUND)
    target_link_libraries(${PROJECT_NAME}_benchmarks PUBLIC OpenMP::OpenMP_CXX)
endif()

if(BLAS_FOUND AND LAPACK_FOUND)
    target_link_libraries(${PROJECT_NAME}_benchmarks PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

target_include_directories(${PROJECT_NAME}_benchmarks PRIVATE ${LU_INCLUDE})

set_target_properties(${PROJECT_NAME}_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


target_link_libraries(${PROJECT_NAME}_benchmarks PRIVATE
    ${target}
    ${LAPACK_LIBRARIES}
    ${LAPACKE_LIBRARIES}
    openblas
)

target_include_directories(${PROJECT_NAME}_benchmarks PRIVATE
    ${LU_INCLUDE}
    ${LAPACKE_INCLUDE_DIRS}
)