#version 330 core

const int MAX_BONES = 100;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;
layout(location = 3) in ivec4 bone_ids;
layout(location = 4) in vec4 bone_weights;

uniform mat4 projection;
uniform mat4 view;

uniform mat4 pose [MAX_BONES];

out FS_IN
{
    smooth vec3 position;
    smooth vec3 normal;
    smooth vec2 tex_coord;
} vs_out;

void main()
{
    mat4 model = (
        bone_weights[0] * pose[bone_ids[0]] +
        bone_weights[1] * pose[bone_ids[1]] +
        bone_weights[2] * pose[bone_ids[2]] +
        bone_weights[3] * pose[bone_ids[3]]
        );
    mat4 model_view = view * model;
    mat4 it_model_view = transpose(inverse(view * model));
    vs_out.position = vec3(model_view * vec4(position, 1));
    vs_out.normal   = normalize(vec3(it_model_view * vec4(normal, 0)));
    vs_out.tex_coord = tex_coord;
    gl_Position = projection * vec4(vs_out.position, 1);
}
