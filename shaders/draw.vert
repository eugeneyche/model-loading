#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 projection;
uniform mat4 view;

out FS_IN
{
    smooth vec3 color;
} vs_out; 

void main()
{
    vs_out.color = color;
    gl_Position = projection * view * vec4(position, 1);
}
