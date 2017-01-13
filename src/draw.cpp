#include "draw.hpp"
#include <glm/gtc/type_ptr.hpp>

DrawUtil::DrawUtil(ShaderManager* sm)
  : sm_ {sm}
{
}

bool DrawUtil::init()
{
    GLuint vert = sm_->make_shader(GL_VERTEX_SHADER, "shaders/draw.vert");
    GLuint frag = sm_->make_shader(GL_FRAGMENT_SHADER, "shaders/draw.frag");
    program_ = sm_->make_program({vert, frag});
    glDeleteShader(vert);
    glDeleteShader(frag);
    if (program_ == 0u) return false;
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertPC) * BATCH_SIZE, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertPC), reinterpret_cast<GLvoid*>(offsetof(VertPC, position)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertPC), reinterpret_cast<GLvoid*>(offsetof(VertPC, color)));
    loc_projection_ = glGetUniformLocation(program_, "projection");
    loc_view_ = glGetUniformLocation(program_, "view");
    return true;
}

void DrawUtil::draw(
        GLenum mode,
        const glm::mat4& projection,
        const glm::mat4& view,
        const std::vector<VertPC>& vertices
        )
{
    glBindVertexArray(vao_);
    glUseProgram(program_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glUniformMatrix4fv(loc_projection_, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(loc_view_, 1, GL_FALSE, glm::value_ptr(view));
    for (size_t i = 0; i < vertices.size(); i += BATCH_SIZE) {
        size_t batch_size = BATCH_SIZE;
        if (i + BATCH_SIZE > vertices.size()) {
            batch_size = vertices.size() - i;
        }
        glBufferSubData(GL_ARRAY_BUFFER, 0, batch_size * sizeof(VertPC), vertices.data());
        glDrawArrays(mode, 0, batch_size);
    }
}
