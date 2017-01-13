cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(model_loading)

set(CMAKE_CXX_COMPILER_ID Clang)
set(CMAKE_CXX_STANDARD 14)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(deps/glad)

find_package(assimp REQUIRED)
find_package(DevIL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

include_directories(
    include
    ${GLAD_INCLUDE_DIRS}
    )

add_executable(
    ${PROJECT_NAME}
    src/draw.cpp
    src/main.cpp
    src/image.cpp
    src/shader.cpp
    src/model.cpp
    )

target_link_libraries(
    ${PROJECT_NAME}
    ${ASSIMP_LIBRARIES}
    ${IL_LIBRARIES}
    ${ILU_LIBRARIES}
    glad
    glfw
    ${OPENGL_gl_LIBRARIES}
    dl
    )
