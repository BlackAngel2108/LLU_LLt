set(target ${LLU_LIBRARY})

file(GLOB srcs "*.cpp")

add_library(${target} ${srcs})

target_include_directories(${target} PUBLIC ${LLU_INCLUDE})

if(OPENMP_FOUND)
    target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
endif()
