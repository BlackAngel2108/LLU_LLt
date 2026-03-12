set(target ${LU_TESTS})

file(GLOB srcs "*.cpp")

add_executable(${target} ${srcs})
target_link_libraries(${target} gtest ${LU_LIBRARY})
target_include_directories(${target} PUBLIC ${CMAKE_SOURCE_DIR})

# Добавляем тест (только один, с правильным путем)
add_test(NAME ${target}
         COMMAND $<TARGET_FILE:${target}>  # автоматически подставит полный путь к исполняемому файлу
)