#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include "mlp.h"
#include "imath.h"
#include "compute_node.h"

MLP *init_mlp() {
    srand(time(0));

    MLP *mlp = malloc(sizeof(MLP));

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            mlp->w1[j][i] = leaf_node();
            mlp->w1[j][i]->data = he_init(INPUT_SIZE);
        }
        mlp->b1[j] = leaf_node();
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->w2[k][j] = leaf_node();
            mlp->w2[k][j]->data = he_init(HIDDEN_SIZE);
        }
        mlp->b2[k] = leaf_node();
    }

    for (int i = 0; i < INPUT_SIZE; i++) {
        mlp->input[i] = leaf_node();
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        compute_node *relu = create_node(forward_relu, backward_relu, 1);
        compute_node *val  = create_node(forward_add, backward_add, INPUT_SIZE + 1);
        for (int i = 0; i < INPUT_SIZE; i++) {
            compute_node *mul = create_node(forward_mul, backward_mul, 2);
            mul->nodes[0] = mlp->w1[j][i];
            mul->nodes[1] = mlp->input[i];
            val->nodes[i] = mul;
        }

        val->nodes[INPUT_SIZE] = mlp->b1[j];
        relu->nodes[0] = val;

        mlp->hidden[j] = relu;
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        compute_node *val  = create_node(forward_add, backward_add, HIDDEN_SIZE + 1);
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            compute_node *mul = create_node(forward_mul, backward_mul, 2);
            mul->nodes[0] = mlp->w2[k][j];
            mul->nodes[1] = mlp->hidden[j];
            val->nodes[j] = mul;
        }

        val->nodes[HIDDEN_SIZE] = mlp->b2[k];
        mlp->output[k] = val;
    }

    mlp->loss = create_node(forward_cross_entropy, backward_cross_entropy, OUTPUT_SIZE);
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        mlp->loss->nodes[k] = mlp->output[k];
    }

    mlp->topo = toposort(mlp->loss, &mlp->count);

    return mlp;
}

void forward_propagate(MLP *mlp, float *input, uint8_t ctx) {
    mlp->loss->ctx = ctx;
    for (int i = 0; i < INPUT_SIZE; i++) {
        mlp->input[i]->data = input[i];
    }

    for (int i = 0; i < (int)mlp->count; i++) {
        if (mlp->topo[i]->forward == NULL) continue;
        mlp->topo[i]->forward(mlp->topo[i]);
    }
}

void back_propagate(MLP *mlp) {
    for (int i = (int)mlp->count - 1; i >= 0; i--) {
        mlp->topo[i]->grad = 0.0f;
    }

    mlp->loss->grad = 1.0f;

    for (int i = (int)mlp->count - 1; i >= 0; i--) {
        if (mlp->topo[i]->backward == NULL) continue;
        mlp->topo[i]->backward(mlp->topo[i]);
    }
}

void update_weights(MLP *mlp) {
    float lr = LEARNING_RATE;
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        mlp->b1[j]->data -= lr * mlp->b1[j]->grad;
        for (int i = 0; i < INPUT_SIZE; i++) {
            mlp->w1[j][i]->data -= lr * mlp->w1[j][i]->grad;
        }
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        mlp->b2[k]->data -= lr * mlp->b2[k]->grad;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->w2[k][j]->data -= lr * mlp->w2[k][j]->grad;
        }
    }
}


static const uint8_t magic[8] = {
    'M', 'L', 'P', 'M', 'O', 'D', 'E', 'L'
};

static const uint32_t version = 1;

bool save_model_to_file(const MLP *mlp, const char *path) {
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        printf("[ERR] Cannot open: '%s'.\n", path);
        return false;
    }

    // header
    if (fwrite(magic, sizeof(magic), 1, fp) != 1) {
        printf("[ERR] Failed to write model magic.\n");
        fclose(fp);
        return false;
    }

    if (fwrite(&version, sizeof(version), 1, fp) != 1) {
        printf("[ERR] Failed to write model version.\n");
        fclose(fp);
        return false;
    }

    // W1
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            if (fwrite(
                    &mlp->w1[j][i]->data,
                    sizeof(float),
                    1,
                    fp
                ) != 1) {
                printf("[ERR] Failed to write W1.\n");
                fclose(fp);
                return false;
            }
        }
    }

    // B1
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        if (fwrite(
                &mlp->b1[j]->data,
                sizeof(float),
                1,
                fp
            ) != 1) {
            printf("[ERR] Failed to write B1.\n");
            fclose(fp);
            return false;
        }
    }

    // W2
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            if (fwrite(
                    &mlp->w2[k][j]->data,
                    sizeof(float),
                    1,
                    fp
                ) != 1) {
                printf("[ERR] Failed to write W2.\n");
                fclose(fp);
                return false;
            }
        }
    }

    // B2
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        if (fwrite(
                &mlp->b2[k]->data,
                sizeof(float),
                1,
                fp
            ) != 1) {
            printf("[ERR] Failed to write B2.\n");
            fclose(fp);
            return false;
        }
    }

    if (fclose(fp) != 0) {
        printf("[ERR] Failed to close model file: '%s'.\n", path);
        return false;
    }

    printf("[OK] Successfully saved model to: '%s'.\n", path);

    return true;
}


bool load_model_from_file(MLP *mlp, const char *path) {
    FILE *fp = fopen(path, "rb");

    if (fp == NULL) {
        printf("[ERR] Cannot open: '%s'.\n", path);
        return false;
    }

    uint8_t fmagic[8];
    uint32_t fversion;

    // read header
    if (fread(fmagic, sizeof(fmagic), 1, fp) != 1) {
        printf("[ERR] Cannot read model magic from '%s'.\n", path);
        fclose(fp);
        return false;
    }

    if (fread(&fversion, sizeof(fversion), 1, fp) != 1) {
        printf("[ERR] Cannot read model version from '%s'.\n", path);
        fclose(fp);
        return false;
    }

    if (memcmp(magic, fmagic, sizeof(magic)) != 0) {
        printf(
            "[ERR] Cannot load model from '%s': unsupported format.\n",
            path
        );
        fclose(fp);
        return false;
    }

    if (fversion != version) {
        printf(
            "[ERR] Unsupported file version: %u.\n",
            fversion
        );
        fclose(fp);
        return false;
    }

    size_t w1_count = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t b1_count = (size_t)HIDDEN_SIZE;
    size_t w2_count = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t b2_count = (size_t)OUTPUT_SIZE;

    // 4MB
    float *buf = malloc(sizeof(float) * 1024 * 1024);

    if (buf == NULL) {
        printf("[ERR] Cannot allocate model loading buffer.\n");
        fclose(fp);
        return false;
    }

    // W1
    if (fread(
            buf,
            sizeof(float),
            w1_count,
            fp
        ) != w1_count) {
        printf("[ERR] Cannot read W1 from '%s'.\n", path);
        free(buf);
        fclose(fp);
        return false;
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            mlp->w1[j][i]->data =
                buf[(size_t)j * INPUT_SIZE + i];
        }
    }

    // B1
    if (fread(
            buf,
            sizeof(float),
            b1_count,
            fp
        ) != b1_count) {
        printf("[ERR] Cannot read B1 from '%s'.\n", path);
        free(buf);
        fclose(fp);
        return false;
    }

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        mlp->b1[j]->data = buf[j];
    }

    // W2
    if (fread(
            buf,
            sizeof(float),
            w2_count,
            fp
        ) != w2_count) {
        printf("[ERR] Cannot read W2 from '%s'.\n", path);
        free(buf);
        fclose(fp);
        return false;
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp->w2[k][j]->data =
                buf[(size_t)k * HIDDEN_SIZE + j];
        }
    }

    // B2
    if (fread(
            buf,
            sizeof(float),
            b2_count,
            fp
        ) != b2_count) {
        printf("[ERR] Cannot read B2 from '%s'.\n", path);
        free(buf);
        fclose(fp);
        return false;
    }

    for (int k = 0; k < OUTPUT_SIZE; k++) {
        mlp->b2[k]->data = buf[k];
    }

    free(buf);
    fclose(fp);

    printf(
        "[OK] Successfully loaded model from: '%s'.\n",
        path
    );

    return true;
}

void free_mlp(MLP *mlp) {
    if (mlp == NULL) return;

    for (int i = 0; i < (int)mlp->count; i++) {
        free_node(mlp->topo[i]);
    }

    free(mlp->topo);
    free(mlp);
}