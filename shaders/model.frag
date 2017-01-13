#version 330 core

in FS_IN
{
    smooth vec3 position;
    smooth vec3 normal;
    smooth vec2 tex_coord;
} fs_in;

uniform sampler2D diffuse_tex;

out vec4 fs_out; 

void main()
{
    float diffuse = clamp(dot(fs_in.normal, vec3(0, 0, 1)), 0.5, 1);
    fs_out = vec4(diffuse * texture(diffuse_tex, fs_in.tex_coord).rgb, 1);
}
