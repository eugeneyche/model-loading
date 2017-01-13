#include "shader.hpp"
#include <cstdio>

GLuint ShaderManager::make_shader(GLenum type, const char* path)
{
    FILE* file = fopen(path, "rb");
    if (file == nullptr) {
        fprintf(stderr, "Failed to open file \"%s\".\n", path);
        return 0u;
    }
    fseek(file, 0, SEEK_END);
    GLint length = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::vector<char> file_buffer (length);
    fread(file_buffer.data(), 1, length, file);

    const GLchar* source = file_buffer.data();
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, &length);
    glCompileShader(shader);

    GLint is_compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
    if (is_compiled == 0) {
        fprintf(stderr, "Failed to compiled shader \"%s\".\n", path);
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log_buffer (log_length + 1);
        glGetShaderInfoLog(shader, log_length + 1, &log_length, log_buffer.data());
        fprintf(stderr, "%s", log_buffer.data());
        glDeleteShader(shader);
        return 0u;
    }
    return shader;
}

GLuint ShaderManager::make_program(const std::vector<GLuint>& parts)
{
    GLuint program = glCreateProgram();

    for (GLuint part : parts) {
        glAttachShader(program, part);
    }

    glLinkProgram(program);

    for (GLuint part : parts) {
        glDetachShader(program, part);
    }

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == 0) {
        fprintf(stderr, "Failed to link program.\n");
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
        std::vector<char> log_buffer (log_length + 1);
        glGetProgramInfoLog(program, log_length + 1, &log_length, log_buffer.data());
        fprintf(stderr, "%s", log_buffer.data());
        glDeleteProgram(program);
        return 0u;
    }

    return program;
}
