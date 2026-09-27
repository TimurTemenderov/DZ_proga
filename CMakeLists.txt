cmake_minimum_required(VERSION 3.16)
project(gaussinversefullpivot14 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(program
    src/task.cpp
    src/matrices.cpp
    src/input.cpp
    src/output.cpp
    src/nevazka.cpp
    src/main.cpp
)

target_include_directories(program PRIVATE include)

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
