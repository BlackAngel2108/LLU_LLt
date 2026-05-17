set(target ${LU_TESTS})

file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs})
target_link_libraries(${target} gtest ${LU_LIBRARY})
target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR})

# Добавляем тест (только один, с правильным путем)
add_test(NAME ${target}
         COMMAND $<TARGET_FILE:${target}>  # автоматически подставит полный путь к исполняемому файлу
)

# Перед сборкой вывести флаги
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")