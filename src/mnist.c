#include <stdio.h>

#include "mnist.h"

static uint32_t rev(uint32_t in) {
    uint32_t ret = 0;
    ret |= (in & 0x000000FF) << 24;
    ret |= (in & 0x0000FF00) <<  8;
    ret |= (in & 0x00FF0000) >>  8;
    ret |= (in & 0xFF000000) >> 24;
    return ret;
}

void mnist_load_images(char *path, mnist_image_handle *handle) {

    // printf("Loading images from '%s'...\n", path);

    FILE *fp = fopen(path, "rb");
    
    handle->fp = fp;
    fread(&handle->magic_number, sizeof(uint32_t), 1, fp);
    fread(&handle->nr_images, sizeof(uint32_t), 1, fp);
    fread(&handle->rows, sizeof(uint32_t), 1, fp);
    fread(&handle->cols, sizeof(uint32_t), 1, fp);

    handle->magic_number = rev(handle->magic_number);
    handle->nr_images = rev(handle->nr_images);
    handle->rows = rev(handle->rows);
    handle->cols = rev(handle->cols);
}

void mnist_load_labels(char *path, mnist_label_handle *handle) {

    // printf("Loading labels from '%s'...\n", path);

    FILE *fp = fopen(path, "rb");

    handle->fp = fp;
    fread(&handle->magic_number, sizeof(uint32_t), 1, fp);
    fread(&handle->nr_labels, sizeof(uint32_t), 1, fp);

    handle->magic_number = rev(handle->magic_number);
    handle->nr_labels = rev(handle->nr_labels);
}

void mnist_next_image(mnist_image_handle *handle, mnist_image *ret) {
    fread(ret->image, sizeof(ret->image), 1, handle->fp);
}

void mnist_next_label(mnist_label_handle *handle, mnist_label *ret) {
    fread(&ret->value, sizeof(ret->value), 1, handle->fp);
}