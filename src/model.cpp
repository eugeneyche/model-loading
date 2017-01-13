#include "image.hpp"
#include "model.hpp"
#include "shader.hpp"
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <glm/gtc/type_ptr.hpp>

static std::string make_prefix(int offset)
{
    std::string prefix;
    while (offset-- > 0) {
        prefix += "\t";
    }
    return prefix;
}

static void print_mat4(glm::mat4 mat, int offset = 0)
{
    std::string prefix = make_prefix(offset);
    printf(
        "%s{ %.2f, %.2f, %.2f, %.2f,\n"
        "%s, %.2f, %.2f, %.2f, %.2f,\n"
        "%s, %.2f, %.2f, %.2f, %.2f,\n"
        "%s, %.2f, %.2f, %.2f, %.2f}\n",
        prefix.c_str(), mat[0][0], mat[1][0], mat[2][0], mat[3][0],
        prefix.c_str(), mat[0][1], mat[1][1], mat[2][1], mat[3][1],
        prefix.c_str(), mat[0][2], mat[1][2], mat[2][2], mat[3][2],
        prefix.c_str(), mat[0][3], mat[1][3], mat[2][3], mat[3][3]
        );
}

static glm::vec3 ai_to_glm_vec3(aiVector3D ai_vec3)
{
    return glm::make_vec3(reinterpret_cast<float*>(&ai_vec3));
}

static glm::mat4 ai_to_glm_mat4(aiMatrix4x4 ai_mat4)
{
    return glm::make_mat4(reinterpret_cast<float*>(&ai_mat4.Transpose()));
}

static glm::quat ai_to_glm_quat(aiQuaternion ai_quat)
{
    return glm::normalize(glm::quat{
        ai_quat.w,
        ai_quat.x,
        ai_quat.y,
        ai_quat.z});
}

static uint8_t get_min_index(const glm::vec4& v)
{
    uint8_t min_lhs = (v[0] < v[1]) ? 0 : 1;
    uint8_t min_rhs = (v[2] < v[3]) ? 2 : 3;
    return (v[min_lhs] < v[min_rhs]) ? min_lhs : min_rhs;
}

static PosRotScale mat4_to_pos_rot_scale(const glm::mat4& mat)
{
    PosRotScale prs;
    prs.position = glm::vec3(mat[3]);
    glm::mat3 basis = glm::mat3(mat);
    glm::mat3 norm_basis {
        glm::normalize(basis[0]),
        glm::normalize(basis[1]),
        glm::normalize(basis[2])
    };
    prs.rotation = glm::normalize(glm::quat_cast(norm_basis));
    glm::mat3 scale = glm::transpose(norm_basis) * basis;
    prs.scale = glm::vec3{scale[0][0], scale[1][1], scale[2][2]};
    return prs;
}

ModelManager::ModelManager(ShaderManager* sm, ImageLoader* il, DrawUtil* du)
  : sm_ {sm}
  , il_ {il}
  , du_ {du}
{
}

bool ModelManager::init()
{
    GLuint vert, frag;
    vert = sm_->make_shader(GL_VERTEX_SHADER, "shaders/model.vert");
    frag = sm_->make_shader(GL_FRAGMENT_SHADER, "shaders/model.frag");
    program_ = sm_->make_program({vert, frag});
    glDeleteShader(vert);
    glDeleteShader(frag);
    if (program_ == 0u) return false;
    loc_projection_ = glGetUniformLocation(program_, "projection");
    loc_view_ = glGetUniformLocation(program_, "view");
    loc_pose_ = glGetUniformLocation(program_, "pose");
    loc_diffuse_tex_ = glGetUniformLocation(program_, "diffuse_tex");

    du_->make_n_colors(bone_colors_, 12);
    return true;
}

bool ModelManager::analyze_model(const char* path)
{
    const aiScene* scene = importer_.ReadFile(path, aiProcess_Triangulate);
    if (scene == nullptr) {
        fprintf(stderr, "Failed to load model \"%s\".\n", path);
        return false;
    }

    std::stack<std::pair<int, aiNode*>> to_explore;
    to_explore.push({0, scene->mRootNode});

    while (not to_explore.empty()) {
        std::pair<int, aiNode*> depth_node = to_explore.top();
        int depth = depth_node.first;
        aiNode* node = depth_node.second;
        to_explore.pop();
        std::string prefix = make_prefix(depth);
        aiMatrix4x4 mat = node->mTransformation;
        printf("%sFound node \"%s\", %d.\n", prefix.c_str(), node->mName.C_Str(), node->mNumMeshes);
        print_mat4(ai_to_glm_mat4(node->mTransformation), depth);
        for (size_t i = 0; i < node->mNumChildren; i++) {
            to_explore.push({depth + 1, node->mChildren[i]});
        }
    }

    for (size_t i = 0; i < scene->mNumAnimations; i++) {
        printf("Found animation \"%s\".\n", scene->mAnimations[i]->mName.C_Str());
    }

}

bool ModelManager::gather_bones(
        std::set<BoneInfo>& included_bones,
        std::vector<aiNode*>& ai_bone_ends,
        aiNode* node,
        bool is_parent_bone,
        int depth,
        const glm::mat4& transform,
        const std::unordered_map<std::string, glm::mat4>& bone_offsets
        )
{
    glm::mat4 local_transform = ai_to_glm_mat4(node->mTransformation);
    glm::mat4 full_transform = transform * local_transform;
    bool should_include = false;
    glm::mat4 offset;
    bool is_bone = false;
    if (bone_offsets.count(node->mName.C_Str()) > 0) {
        should_include = true;
        is_bone = true;
        offset = bone_offsets.at(node->mName.C_Str());
    }
    for (size_t i = 0; i < node->mNumChildren; i++) {
         if (gather_bones(
                included_bones,
                ai_bone_ends,
                node->mChildren[i],
                is_bone,
                depth + 1,
                full_transform,
                bone_offsets
                )) {
             should_include = true;
         }
    }
    if (should_include) {
        BoneInfo bone = {
            depth,
            node,
            offset
        };
        included_bones.insert(bone);
    } else if (is_parent_bone) {
        ai_bone_ends.push_back(node);
    }
    return should_include;
}

void ModelManager::process_bones(Model* model, const aiScene* scene)
{
    std::unordered_map<std::string, glm::mat4> bone_offsets;
    std::stack<std::pair<glm::mat4, aiNode*>> to_explore;
    to_explore.push({glm::mat4{1.f}, scene->mRootNode});
    while (not to_explore.empty()) {
        glm::mat4 transform;
        aiNode* node;
        std::tie(transform, node) = to_explore.top();
        to_explore.pop();
        glm::mat4 full_transform = transform * ai_to_glm_mat4(node->mTransformation);
        for (size_t i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            for (size_t j = 0; j < mesh->mNumBones; j++) {
                aiBone* bone = mesh->mBones[j];
                bone_offsets[bone->mName.C_Str()] = ai_to_glm_mat4(bone->mOffsetMatrix) * glm::inverse(full_transform);
            }
        }
        for (size_t i = 0; i < node->mNumChildren; i++) {
            to_explore.push({full_transform, node->mChildren[i]});
        }
    }

    std::set<BoneInfo> included_bones;
    std::vector<aiNode*> ai_bone_ends;
    gather_bones(included_bones, ai_bone_ends, scene->mRootNode, false, 0, glm::mat4{1.f}, bone_offsets);
    model->n_bones++;
    for (auto bone : included_bones) {
        uint8_t bone_id = model->n_bones++;
        if (bone.node->mName.length > 0) {
            model->bone_mapping[bone.node->mName.C_Str()] = bone_id;
        }
        if (bone.node->mParent and model->bone_mapping.count(bone.node->mParent->mName.C_Str()) > 0) {
            model->parent_ids[bone_id] = model->bone_mapping.at(bone.node->mParent->mName.C_Str());
        }
        model->offsets[bone_id] = bone.offset;
        model->default_pose[bone_id] = ai_to_glm_mat4(bone.node->mTransformation);
        model->default_pose_prs[bone_id] = mat4_to_pos_rot_scale(model->default_pose[bone_id]);
    }
    for (auto node : ai_bone_ends) {
        glm::mat4 transform = ai_to_glm_mat4(node->mTransformation);
        model->bone_ends.push_back({
            model->bone_mapping.at(node->mParent->mName.C_Str()),
            glm::vec3{transform[3]}
            });
    } 
}

void ModelManager::process_mesh(
        Mesh* mesh,
        std::vector<VertPNUBiBw>& vertices,
        std::vector<GLuint>& indices,
        const glm::mat4& transform,
        const std::unordered_map<std::string, uint8_t>& bone_mapping,
        aiMesh* ai_mesh
        )
{
    glm::mat4 it_transform = glm::transpose(glm::inverse(transform));
    GLuint mesh_offset = vertices.size();

    for (size_t i = 0; i < ai_mesh->mNumVertices; i++) {
        VertPNUBiBw vertex;
        glm::vec4 position = glm::vec4(ai_to_glm_vec3(ai_mesh->mVertices[i]), 1.f);
        glm::vec4 normal = glm::vec4(ai_to_glm_vec3(ai_mesh->mNormals[i]), 0.f);
        glm::vec2 tex_coord = glm::vec2(ai_to_glm_vec3(ai_mesh->mTextureCoords[0][i]));
        vertex.position = glm::vec3(transform * position);
        vertex.normal = glm::vec3(it_transform * normal);
        vertex.tex_coord = tex_coord;
        vertex.bone_ids =  {0, 0, 0, 0};
        vertex.bone_weights = {0.f, 0.f, 0.f, 0.f};
        vertices.push_back(vertex);
    }

    for (size_t i = 0; i < ai_mesh->mNumBones; i++) {
        aiBone* bone = ai_mesh->mBones[i];
        uint8_t bone_id = bone_mapping.at(bone->mName.C_Str());
        for (size_t j = 0; j < bone->mNumWeights; j++) {
            aiVertexWeight weight = bone->mWeights[j];
            VertPNUBiBw& vert = vertices[mesh_offset + weight.mVertexId];
            uint8_t min_index = get_min_index(vert.bone_weights);
            if (weight.mWeight > vert.bone_weights[min_index]) {
                vert.bone_ids[min_index] = bone_id;
                vert.bone_weights[min_index] = weight.mWeight;
            }
        }
    }

    for (size_t i = mesh_offset; i < vertices.size(); i++) {
        VertPNUBiBw& vertex = vertices[i];
        float weight_total = (
            vertex.bone_weights[0] + 
            vertex.bone_weights[1] + 
            vertex.bone_weights[2] + 
            vertex.bone_weights[3]
            );
        if (weight_total > 0.f) {
            vertex.bone_weights /= weight_total;
        } else {
            vertex.bone_ids[0] = 0;
            vertex.bone_weights[0] = 1.f;
        }
    }

    GLuint offset = indices.size();
    GLuint count = 0;

    for (size_t i = 0; i < ai_mesh->mNumFaces; i++) {
        aiFace face = ai_mesh->mFaces[i];
        for (size_t j = 0; j < face.mNumIndices; j++) {
            indices.push_back(mesh_offset + face.mIndices[j]);
            count++;
        }
    }

    mesh->material_h = ai_mesh->mMaterialIndex;
    mesh->offset = offset;
    mesh->count = count;
}

void ModelManager::process_material(Material* mat, aiMaterial* ai_mat, const std::string& base_dir)
{
    aiString tex_path;
    ai_mat->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path, nullptr, nullptr, nullptr, nullptr, nullptr);
    std::string full_path = base_dir + "/" + tex_path.C_Str();
    mat->diffuse_tex = il_->make_texture_from_image(full_path.c_str());
}

bool ModelManager::load_model(Model* model, Animation* animation, const char* path)
{
    std::string s_path (path);
    std::string base_dir = s_path.substr(0, s_path.find_last_of('/'));
    
    const aiScene* scene = importer_.ReadFile(path, aiProcess_Triangulate);
    if (scene == nullptr) {
        fprintf(stderr, "Failed to load model \"%s\".\n", path);
        return false;
    }

    for (size_t i = 0; i < scene->mNumMaterials; i++) {
        process_material(&model->materials[i], scene->mMaterials[i], base_dir);
    }

    process_bones(model, scene);

    std::vector<VertPNUBiBw> vertices;
    std::vector<GLuint> indices;

    std::stack<std::pair<glm::mat4, aiNode*>> to_explore;
    to_explore.push({glm::mat4{1.f}, scene->mRootNode});
    while (not to_explore.empty()) {
        glm::mat4 transform;
        aiNode* node;
        std::tie(transform, node) = to_explore.top();
        to_explore.pop();
        glm::mat4 full_transform = transform * ai_to_glm_mat4(node->mTransformation);
        for (size_t i = 0; i < node->mNumMeshes; i++) {
            Mesh* mesh = &model->meshes[model->n_meshes++];
            aiMesh* ai_mesh = scene->mMeshes[node->mMeshes[i]];
            process_mesh(mesh, vertices, indices, full_transform, model->bone_mapping, ai_mesh);
        }
        for (size_t i = 0; i < node->mNumChildren; i++) {
            to_explore.push({full_transform, node->mChildren[i]});
        }
    }

    Pose global_pose;
    convert_local_to_global_pose(global_pose, model, model->default_pose, true);
    for (size_t i = 0; i < vertices.size(); i++) {
        const VertPNUBiBw& vert = vertices[i];
        glm::mat4 model_transform = (
            vert.bone_weights[0] * global_pose[vert.bone_ids[0]] + 
            vert.bone_weights[1] * global_pose[vert.bone_ids[1]] + 
            vert.bone_weights[2] * global_pose[vert.bone_ids[2]] + 
            vert.bone_weights[3] * global_pose[vert.bone_ids[3]]
            );
        glm::vec3 global_position = glm::vec3(model_transform * glm::vec4{vertices[i].position, 1.f});
        model->bbox.merge_in(global_position);
    }

    glGenVertexArrays(1, &model->vao);
    glGenBuffers(1, &model->vbo);
    glGenBuffers(1, &model->ebo);
    glBindVertexArray(model->vao);
    glBindBuffer(GL_ARRAY_BUFFER, model->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertPNUBiBw) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertPNUBiBw), reinterpret_cast<GLvoid*>(offsetof(VertPNUBiBw, position)));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, sizeof(VertPNUBiBw), reinterpret_cast<GLvoid*>(offsetof(VertPNUBiBw, normal)));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertPNUBiBw), reinterpret_cast<GLvoid*>(offsetof(VertPNUBiBw, tex_coord)));
    glVertexAttribIPointer(3, 4, GL_INT, sizeof(VertPNUBiBw), reinterpret_cast<GLvoid*>(offsetof(VertPNUBiBw, bone_ids)));
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(VertPNUBiBw), reinterpret_cast<GLvoid*>(offsetof(VertPNUBiBw, bone_weights)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model->ebo); 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_READ);

    if (scene->mNumAnimations > 0) {
        aiAnimation* ai_animation = scene->mAnimations[0];
        animation->duration = ai_animation->mDuration;
        for (size_t i = 0; i < ai_animation->mNumChannels; i++) {
            aiNodeAnim* node_anim = ai_animation->mChannels[i];
            Channel& channel = animation->channels[animation->n_channels++];
            if (model->bone_mapping.count(node_anim->mNodeName.C_Str())) {
                channel.bone_id = model->bone_mapping.at(node_anim->mNodeName.C_Str());
                for (size_t j = 0; j < node_anim->mNumPositionKeys; j++) {
                    channel.position_keys.push_back({
                        static_cast<float>(node_anim->mPositionKeys[j].mTime),
                        ai_to_glm_vec3(node_anim->mPositionKeys[j].mValue)
                        });
                }
                for (size_t j = 0; j < node_anim->mNumRotationKeys; j++) {
                    channel.rotation_keys.push_back({
                        static_cast<float>(node_anim->mRotationKeys[j].mTime),
                        ai_to_glm_quat(node_anim->mRotationKeys[j].mValue)
                        });
                }
            }
        }
    }
    return true;
}

template <typename T>
static T get_key_value(const std::vector<Key<T>>& keys, float time)
{
    if (time < keys.front().time) {
        return keys.front().value;
    }
    if (time >= keys.back().time) {
        return keys.back().value;
    }
    size_t bbegin = 0;
    size_t bend = keys.size() - 1;
    while (bbegin < bend - 1) {
        size_t mid = bbegin + (bend - bbegin) / 2;
        if (keys[mid].time > time) {
            bend = mid;
        } else if (mid < bend and keys[mid + 1].time < time) {
            bbegin = mid + 1;
        } else {
            bbegin = mid;
            break;
        }
    }
    float interp = (time - keys[bbegin].time) / (keys[bbegin + 1].time - keys[bbegin].time);
    return glm::mix(keys[bbegin].value, keys[bbegin + 1].value, interp);
}

void ModelManager::update_pose(Model* model, Pose& pose, Animation* animation, float time)
{
    time *= 24.f;
    float looped_time = time - glm::floor(time / animation->duration) * animation->duration;
    for (size_t i = 0; i < model->n_bones; i++) {
        pose[i] = model->default_pose[i];
    } 
    for (size_t i = 0 ; i < animation->n_channels; i++) {
        Channel& channel = animation->channels[i];
        PosRotScale prs = model->default_pose_prs[channel.bone_id];
        if (not channel.position_keys.empty()) {
            prs.position = get_key_value(channel.position_keys, looped_time);
        }
        if (not channel.position_keys.empty()) {
            prs.rotation = get_key_value(channel.rotation_keys, looped_time);
        }
        pose[channel.bone_id] = prs.to_mat4();
    }
}

void ModelManager::draw_model(
        Model* model,
        const Pose& pose,
        const glm::mat4& projection,
        const glm::mat4& view
        )
{
    Pose global_pose;
    convert_local_to_global_pose(global_pose, model, pose, true);

    glUseProgram(program_);
    glBindVertexArray(model->vao);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model->ebo);
    glUniformMatrix4fv(loc_projection_, 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(loc_view_, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(loc_pose_, model->n_bones, GL_FALSE, reinterpret_cast<const GLfloat*>(global_pose.data()));
    glActiveTexture(GL_TEXTURE1);
    for (size_t i = 0; i < model->n_meshes; i++) {
        const Mesh& mesh = model->meshes[i];
        glBindTexture(GL_TEXTURE_2D, model->materials[mesh.material_h].diffuse_tex);
        glUniform1i(loc_diffuse_tex_, 1);
        glDrawElements(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, reinterpret_cast<GLvoid*>(sizeof(GLuint) * mesh.offset));
    }
}

void ModelManager::draw_skeleton(
        Model* model,
        const Pose& pose,
        const glm::mat4& projection,
        const glm::mat4& view
        )
{
    glDisable(GL_DEPTH_TEST);
    Pose global_pose;
    convert_local_to_global_pose(global_pose, model, pose, false);
    std::vector<VertPC> vertices;
    size_t color_id = 0;
    for (size_t i = 0; i < model->n_bones; i++) {
        if (model->parent_ids[i] < model->n_bones) {
            glm::vec3 color = bone_colors_[color_id++ % bone_colors_.size()];
            vertices.push_back({
                glm::vec3{global_pose[i] * glm::vec4{0.f, 0.f, 0.f, 1.f}},
                color
                });
            vertices.push_back({
                glm::vec3{global_pose[model->parent_ids[i]] *
                glm::vec4{0.f, 0.f, 0.f, 1.f}},
                color
                });
        }
    }
    for (auto bone_end : model->bone_ends) {
        glm::vec3 color = bone_colors_[color_id++ % bone_colors_.size()];
        vertices.push_back({
            glm::vec3{global_pose[bone_end.first] * glm::vec4{0.f, 0.f, 0.f, 1.f}},
            color
            });
        vertices.push_back({
            glm::vec3{global_pose[bone_end.first] * glm::vec4{bone_end.second, 1.f}},
            color
            });
    }
    glPointSize(5.f);
    du_->draw(GL_LINES, projection, view, vertices);
    du_->draw(GL_POINTS, projection, view, vertices);
}

void ModelManager::convert_local_to_global_pose(Pose& global_pose, const Model* model, const Pose& local_pose, bool apply_offsets)
{
    for (size_t i = 0; i < model->n_bones; i++) {
        if (model->parent_ids[i] < model->n_bones) {
            global_pose[i] = global_pose[model->parent_ids[i]] * local_pose[i];
        } else {
            global_pose[i] = local_pose[i];
        }
    }
    if (apply_offsets) {
        for (size_t i = 0; i < model->n_bones; i++) {
            global_pose[i] = global_pose[i] * model->offsets[i];
        }
    }
}
