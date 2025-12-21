set(target ${LIB_NAME})

file(GLOB srcs "*.cpp")

add_library(${target} ${srcs})

target_include_directories(${target} PUBLIC ${PROJ_INCLUDE})

if(OPENMP_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endif()
