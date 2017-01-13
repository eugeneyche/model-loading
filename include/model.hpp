#pragma once
#include "image.hpp"
#include "shader.hpp"
#include <cstdint>
#include <array>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

const size_t MAX_MESHES = 20;
const size_t MAX_BONES = 100;

struct VertPNUBiBw
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec2 tex_coord;
    glm::ivec4 bone_ids;
    glm::vec4 bone_weights;
};

struct Mesh
{
    uint8_t material_h;
    GLsizei offset;
    GLsizei count;
};

struct Material
{
    GLuint diffuse_tex;
};


using Pose = std::array<glm::mat4, MAX_BONES>;

struct Model
{
    size_t n_meshes = 0;
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    std::array<Mesh, MAX_MESHES> meshes;
    std::array<Material, MAX_MESHES> materials;

    std::unordered_map<std::string, uint8_t> bone_mapping;
    size_t n_bones = 0;
    std::array<uint8_t, MAX_BONES> parent_ids;
    Pose offsets;
    Pose default_pose;
};

template <typename T>
struct Key
{
    float time;
    T value;
};

struct Channel
{
    uint8_t bone_id;
    std::vector<Key<glm::vec3>> position_keys;
    std::vector<Key<glm::quat>> rotation_keys;
};

struct Animation
{
    float duration;
    size_t n_channels = 0;
    std::array<Channel, MAX_BONES> channels;
};

class ModelManager
{
public:
    ModelManager(ShaderManager* sm, ImageLoader* il);
    virtual ~ModelManager() = default;

    bool init();
    bool analyze_model(const char* path);
    bool load_model(Model* model, Animation* animation, const char* path);
    void update_pose(Model* model, Pose& pose, Animation* animation, float time);
    void draw_model(
        Model* model,
        const Pose& pose,
        const glm::mat4& projection,
        const glm::mat4& view
        );
    void draw_skeleton(
        Model* model,
        const Pose& pose,
        const glm::mat4& projection,
        const glm::mat4& view
        );

private:
    ShaderManager* sm_;
    ImageLoader* il_;
    Assimp::Importer importer_;

    GLuint mr_program_;
    GLint mr_loc_projection_;
    GLint mr_loc_view_;
    GLint mr_loc_pose_;
    GLint mr_loc_diffuse_tex_;

    GLuint skr_program_;
    GLuint skr_vao_;
    GLuint skr_vbo_;
    GLint skr_loc_projection_;
    GLint skr_loc_view_;

    struct BoneInfo
    {
        int depth;
        aiNode* node;
        glm::mat4 offset;

        bool operator<(const BoneInfo& that) const {
            if (this->depth != that.depth) {
                return this->depth < that.depth;
            }
            return this->node < that.node;
        }
    };

    bool gather_bones(
        std::set<BoneInfo>& included_bones,
        aiNode* node,
        int depth,
        const glm::mat4& transform,
        const std::unordered_map<std::string, glm::mat4>& bone_offsets
        );
    void process_bones(Model* model, const aiScene* scene);
    void process_mesh(
        Mesh* mesh,
        std::vector<VertPNUBiBw>& vertices,
        std::vector<GLuint>& indices,
        const glm::mat4& transform,
        const std::unordered_map<std::string, uint8_t>& bone_mapping,
        aiMesh* ai_mesh
        );
    void process_material(Material* mat, aiMaterial* ai_mat, const std::string& base_dir);
};
