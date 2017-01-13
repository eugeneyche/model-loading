#pragma once
#include <glad/glad.h>
#include <vector>

class ShaderManager
{
public:
    ShaderManager() = default;
    virtual ~ShaderManager() = default;

    GLuint make_shader(GLenum type, const char* path);
    GLuint make_program(const std::vector<GLuint>& parts);
};

