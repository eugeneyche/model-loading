#pragma once
#include <glad/glad.h>

class ImageLoader
{
public:
    ImageLoader() = default;
    virtual ~ImageLoader() = default;

    bool init();
    GLuint make_texture_from_image(const char* path);
};
