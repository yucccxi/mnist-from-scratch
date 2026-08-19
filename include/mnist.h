#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28

#define MNIST_LABEL_SIZE 10

/*
image:
    4  B - magic number
    4  B - the number of images
    4  B - rows per image
    4  B - cols per image
    784B - image 0
           ...
*/

/*
label:
    4B - magic number
    4B - the number of labels
    1B - label 0
         ...
*/

typedef struct mnist_image mnist_image;
typedef struct mnist_label mnist_label;

typedef struct mnist_image_handle mnist_image_handle;
typedef struct mnist_label_handle mnist_label_handle;

struct mnist_image_handle {
    FILE *fp;
    uint32_t magic_number;
    uint32_t nr_images;
    uint32_t rows;
    uint32_t cols;
};

struct mnist_image {
    uint8_t image[MNIST_IMAGE_ROWS][MNIST_IMAGE_COLS];
};

struct mnist_label_handle {
    FILE *fp;
    uint32_t magic_number;
    uint32_t nr_labels;
};

struct mnist_label {
    uint8_t value;
};

void mnist_load_images(char *path, mnist_image_handle *ret_handle);
void mnist_load_labels(char *path, mnist_label_handle *ret_handle);

void mnist_next_image(mnist_image_handle *handle, mnist_image *ret_image);
void mnist_next_label(mnist_label_handle *handle, mnist_label *ret_label);

#endif