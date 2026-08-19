#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
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

void free_mlp(MLP *mlp) {
    if (mlp == NULL) return;

    for (int i = 0; i < (int)mlp->count; i++) {
        free_node(mlp->topo[i]);
    }

    free(mlp->topo);
    free(mlp);
}