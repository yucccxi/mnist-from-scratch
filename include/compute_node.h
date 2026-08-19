#ifndef COMPUTE_NODE_H
#define COMPUTE_NODE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct compute_node compute_node;

typedef void (*forward_func)(compute_node *self);
typedef void (*backward_func)(compute_node *self);

struct compute_node {
    float data;
    float grad;

    size_t argc;
    compute_node **nodes;

    forward_func forward;
    backward_func backward;

    bool marked;

    // the index of the correct class (target label) 
    // for the cross-entropy loss.
    uint8_t ctx;
};

compute_node *create_node(forward_func f, backward_func g, size_t argc);
compute_node *leaf_node();
compute_node **toposort(compute_node *root, size_t *tot);

void forward_add(compute_node *self);
void backward_add(compute_node *self);

void forward_mul(compute_node *self);
void backward_mul(compute_node *self);

void forward_relu(compute_node *self);
void backward_relu(compute_node *self);

void forward_cross_entropy(compute_node *self);
void backward_cross_entropy(compute_node *self);

void free_node(compute_node *node);

#endif