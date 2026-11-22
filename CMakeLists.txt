cmake_minimum_required(VERSION 3.16)
project(gaussinversefullpivot14 LANGUAGES CXX)

# Используем стандартный C++17 без расширений синтаксиса компилятора.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Дополнительный режим для процессора сборки с раздельным округлением операций.
option(GAUSS_NATIVE "Optimize for the build CPU with floating-point contraction disabled" OFF)
if(GAUSS_NATIVE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$" AND NOT MSVC)
        add_compile_options(-march=native -ffp-contract=off)
    else()
        message(FATAL_ERROR "GAUSS_NATIVE requires GCC or Clang with GNU-style options")
    endif()
endif()

# Основная программа включает обращение, подготовку данных, вывод и невязку.
add_executable(program
    src/task.cpp
    src/matrices.cpp
    src/input.cpp
    src/output.cpp
    src/nevazka.cpp
    src/main.cpp
)

target_include_directories(program PRIVATE include)

# Threads::Threads передаёт нужные компилятору и линкеру параметры pthread.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(program PRIVATE Threads::Threads)

# Тест проверяет алгоритм напрямую, без разборщика командной строки.
include(CTest)
if(BUILD_TESTING)
    add_executable(task_tests tests/task_tests.cpp src/task.cpp src/matrices.cpp)
    target_include_directories(task_tests PRIVATE include)
    target_link_libraries(task_tests PRIVATE Threads::Threads)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(task_tests PRIVATE -Wall -Wextra -Werror -pedantic)
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            # Перехватываем pthread_create только в тестах для имитации отказа.
            target_compile_definitions(task_tests PRIVATE GAUSS_TEST_CREATE_FAILURE)
            target_link_options(task_tests PRIVATE "LINKER:--wrap=pthread_create")
        endif()
    endif()
    add_test(NAME full_pivot_inverse COMMAND task_tests)
    # Ограничиваем время теста, чтобы обнаружить зависание на барьере.
    set_tests_properties(full_pivot_inverse PROPERTIES TIMEOUT 120)
endif()

# Предупреждения считаются ошибками. Для GCC также заданы лимиты встраивания.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(program PRIVATE
        -W
        -Wall
        -Wfloat-equal
        -Wpointer-arith
        -Wwrite-strings
        -Wcast-align
        -Wformat-security
        -Wmissing-format-attribute
        -Wformat=1
        -Wno-long-long
        -Winline
        -Werror
        -pedantic
        -pedantic-errors
        -Wunused
        -Wuninitialized
        "SHELL:--param inline-unit-growth=1000000"
        "SHELL:--param max-inline-insns-single=10000000"
        "SHELL:--param large-function-growth=10000000"
        -fPIC
    )
elseif(MSVC)
    target_compile_options(program PRIVATE /W4 /WX /permissive-)
endif()
