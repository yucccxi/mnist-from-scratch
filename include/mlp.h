#ifndef MLP_H
#define MLP_H

#include "compute_node.h"

#define INPUT_SIZE  784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE  10

#define LEARNING_RATE 0.005

typedef struct MLP MLP;

struct MLP {
    compute_node *input[INPUT_SIZE];
    compute_node *hidden[HIDDEN_SIZE];
    compute_node *output[OUTPUT_SIZE];

    compute_node *w1[HIDDEN_SIZE][INPUT_SIZE];
    compute_node *b1[HIDDEN_SIZE];

    compute_node *w2[OUTPUT_SIZE][HIDDEN_SIZE];
    compute_node *b2[OUTPUT_SIZE];

    compute_node *loss;

    size_t count;
    compute_node **topo;
};

MLP *init_mlp();
void free_mlp(MLP *mlp);

void forward_propagate(MLP *mlp, float *input, uint8_t ctx);
void back_propagate(MLP *mlp);
void update_weights(MLP *mlp);

#endif