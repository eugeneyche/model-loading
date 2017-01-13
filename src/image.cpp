#include "image.hpp"
#include <cstdio>
#include <IL/il.h>
#include <IL/ilu.h>

bool ImageLoader::init()
{
    ilInit();
    return ilGetError() == IL_NO_ERROR;
}

GLuint ImageLoader::make_texture_from_image(const char* path)
{
    ILuint img = 0u;
    ilGenImages(1, &img);
    ilBindImage(img);

    if (not ilLoadImage(path)) {
        ilBindImage(0u);
        ilDeleteImage(img);
        fprintf(stderr, "Failed to load image \"%s\".\n", path);
        return 0u;
    }

    ILinfo img_info;
    iluGetImageInfo(&img_info);
    if (img_info.Origin == IL_ORIGIN_UPPER_LEFT) {
           iluFlipImage();
    }

    GLuint tex = 0u;
    glGenTextures(1, &tex);

    GLenum internal_format = 0u, format = 0u;
    switch (img_info.Format) {
    case IL_RGB:
        internal_format = GL_RGB;
        format = GL_RGB;
        break;
    case IL_RGBA:
        internal_format = GL_RGBA;
        format = GL_RGBA;
        break;
    case IL_BGR:
        internal_format = GL_RGB;
        format = GL_BGR;
        break;
    case IL_BGRA:
        internal_format = GL_RGBA;
        format = GL_BGRA;
        break;
    case IL_LUMINANCE:
        internal_format = GL_RED;
        format = GL_RED;
        break;
    default:
        fprintf(stderr, "Failed to recognize format of \"%s\".\n", path);
    }
    
    GLenum type = 0u;
    switch (img_info.Type) {
    case IL_BYTE: type = GL_BYTE; break;
    case IL_UNSIGNED_BYTE: type = GL_UNSIGNED_BYTE; break;
    case IL_SHORT: type = GL_SHORT; break;
    case IL_UNSIGNED_SHORT: type = GL_UNSIGNED_SHORT; break;
    case IL_INT: type = GL_INT; break;
    case IL_UNSIGNED_INT: type = GL_UNSIGNED_INT; break;
    case IL_FLOAT: type = GL_FLOAT; break;
    case IL_DOUBLE: type = GL_DOUBLE; break;
    default:
        fprintf(stderr, "Failed to recognize channel type of \"%s\".\n", path);
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(
        GL_TEXTURE_2D,
        0, internal_format, img_info.Width, img_info.Height,
        0, format, type, img_info.Data
        );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    ilBindImage(0u);
    ilDeleteImage(img);

    return tex;
}
