set(target ${LLU_TESTS})

file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs})
target_link_libraries(${target} gtest ${LLU_LIBRARY})
target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR})
add_test(${target} ${target})
