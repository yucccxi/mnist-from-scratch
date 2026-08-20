#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "compute_node.h"
#include "mnist.h"
#include "imath.h"
#include "mlp.h"

#define EPOCHES 8

mnist_image_handle image_handle;
mnist_label_handle label_handle;

mnist_image image;
mnist_label label;

MLP *mlp;

float input[INPUT_SIZE];

char get_pixel(float gval) {
    static const char ramp[] = " .:-=+*#%@";
    int n = sizeof(ramp) - 2;

    if (gval < 0.0f) gval = 0.0f;
    if (gval > 1.0f) gval = 1.0f;

    return ramp[(int)(gval * n)];
}

void get_img_input() {
    for (int i = 0; i < MNIST_IMAGE_ROWS; i++) {
        for (int j = 0; j < MNIST_IMAGE_COLS; j++) {
            input[i * MNIST_IMAGE_COLS + j] = image.image[i][j] / 255.0f;
        }
    }
}

void print_img() {
    for (int i = 0; i < MNIST_IMAGE_ROWS; i++) {
        for (int j = 0; j < MNIST_IMAGE_COLS; j++) {
            printf("%c", get_pixel(image.image[i][j] / 255.0f));
        }
        printf("\n");
    }
}

void train_all() {
    mnist_load_images("./lib/train-images-idx3-ubyte", &image_handle);
    mnist_load_labels("./lib/train-labels-idx1-ubyte", &label_handle);

    assert(image_handle.nr_images == label_handle.nr_labels);

    int tot = (int)image_handle.nr_images;

    int total_count = 0;
    float total_loss = 0.0f;
    float process = 0.0f;
    for (int i = 0; i < tot; i++) {
        mnist_next_image(&image_handle, &image);
        mnist_next_label(&label_handle, &label);
        get_img_input();
        forward_propagate(mlp, input, label.value);
        back_propagate(mlp);
        update_weights(mlp);

        int mxj = -1;
        float mxv = -1;
        for (int j = 0; j < 10; j++) {
            if (mlp->output[j]->data > mxv) {
                mxv = mlp->output[j]->data;
                mxj = j;
            }
        }

        total_count += (label.value == mxj);
        total_loss  += mlp->loss->data;

        float now = (float)(i + 1) / tot;
        if (now - process > 0.01f) {
            process = now;
            printf("\r%f", process);
            fflush(stdout);
        }
    }

    printf("\r[OK] %d images finished. Avg loss: %f. Avg accuracy: %f\n", 
        tot, 
        total_loss  / tot, 
        (float)total_count / tot
    );
    fflush(stdout);

    fclose(image_handle.fp);
    fclose(label_handle.fp);
}

void test_all() {
    mnist_load_images("./lib/t10k-images-idx3-ubyte", &image_handle);
    mnist_load_labels("./lib/t10k-labels-idx1-ubyte", &label_handle);

    assert(image_handle.nr_images == label_handle.nr_labels);

    int tot = image_handle.nr_images;

    int total_count = 0;
    float total_loss = 0.0f;
    float process = 0.0f;
    for (int i = 0; i < tot; i++) {
        mnist_next_image(&image_handle, &image);
        mnist_next_label(&label_handle, &label);
        get_img_input();
        forward_propagate(mlp, input, label.value);

        int mxj = -1;
        float mxv = -1;
        for (int j = 0; j < 10; j++) {
            if (mlp->output[j]->data > mxv) {
                mxv = mlp->output[j]->data;
                mxj = j;
            }
        }

        total_count += (label.value == mxj);
        total_loss  += mlp->loss->data;

        float now = (float)(i + 1) / tot;
        if (now - process > 0.01f) {
            process = now;
            printf("\r%f", process);
            fflush(stdout);
        }
    }

    printf("\r[OK] %d images finished. Avg loss: %f. Avg accuracy: %f\n", 
        tot, 
        total_loss  / tot, 
        (float)total_count / tot
    );
    fflush(stdout);

    fclose(image_handle.fp);
    fclose(label_handle.fp);
}

void generate_model() {
    for (int i = 0; i < EPOCHES; i++) {
        printf("Running epoch #%d ...\n", i + 1);
        train_all();
        test_all();

        char fname[128];
        uint64_t tstp = (uint64_t)time(NULL);

        const char *ffmt = "./model/MLP-epoch%02d-lr%f-t%lu.bin";
        snprintf(fname, sizeof(fname), ffmt, i + 1, LEARNING_RATE, tstp);

        if (!save_model_to_file(mlp, fname)) return;
    }
}

int main() {
    mlp = init_mlp();

    if (
        !load_model_from_file(mlp, 
        "./model/MLP-epoch08-lr0.005000-t1787156672.bin")
    ) return 1;

    mnist_load_images("./lib/t10k-images-idx3-ubyte", &image_handle);
    mnist_load_labels("./lib/t10k-labels-idx1-ubyte", &label_handle);

    assert(image_handle.nr_images == label_handle.nr_labels);

    float prob[OUTPUT_SIZE];

    for (int i = 0; i < (int)image_handle.nr_images; i++) {
        printf("\n");

        mnist_next_image(&image_handle, &image);
        mnist_next_label(&label_handle, &label);

        get_img_input();
        forward_propagate(mlp, input, label.value);

        print_img();

        for (int j = 0; j < 10; j++) {
            prob[j] = mlp->output[j]->data;
        }

        softmax(10, prob);

        int argmax = -1;
        float max_prob = 0;
        for (int j = 0; j < 10; j++) {
            if (prob[j] > max_prob) {
                max_prob = prob[j];
                argmax = j;
            }
        }

        printf("+-------+------------+\n");
        printf("| Class | Probability|\n");
        printf("+-------+------------+\n");

        for (int i = 0; i < 10; i++) {
            printf("| %5d | %10.4f |\n", i, prob[i]);
        }
        printf("+-------+------------+\n");

        bool right = (argmax == label.value);

        printf("%d: True value: %d, %s!\n", i + 1, label.value, (right ? "RIGHT" : "WRONG"));
        printf("Press enter to continue ...");

        // if (!right) while (getchar() != '\n');
        while (getchar() != '\n');
    }

    fclose(image_handle.fp);
    fclose(label_handle.fp);

    free_mlp(mlp);

    return 0;
}