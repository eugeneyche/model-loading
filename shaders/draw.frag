#version 330 core

in FS_IN
{
    smooth vec3 color;
} fs_in; 

out vec4 fs_out;

void main()
{
    fs_out = vec4(fs_in.color, 1);
}
