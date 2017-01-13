#pragma once
#include "shader.hpp"
#include <vector>
#include <glm/glm.hpp>

struct VertPC
{
    glm::vec3 position;
    glm::vec3 color;
};

class DrawUtil
{
public:
    DrawUtil(ShaderManager* sm);
    virtual ~DrawUtil() = default;

    bool init();
    void make_n_colors(std::vector<glm::vec3>& colors, size_t n);
    void draw(
        GLenum mode,
        const glm::mat4& projection,
        const glm::mat4& view,
        const std::vector<VertPC>& vertices
        );

private:
    static const size_t BATCH_SIZE = 1000;

    ShaderManager* sm_;
    GLuint program_;
    GLuint vao_;
    GLuint vbo_;
    GLint loc_projection_;
    GLint loc_view_;
};
