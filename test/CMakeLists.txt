set(target ${TEST_NAME})

file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs})
target_link_libraries(${target} gtest ${LIB_NAME})
target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR}/gtest)
add_test(${target} ${target})
