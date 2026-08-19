#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "imath.h"
#include "compute_node.h"

compute_node *create_node(forward_func f, backward_func g, size_t argc) {

    compute_node *node = calloc(1, sizeof(compute_node));
    node->forward  = f;
    node->backward = g;
    node->argc     = argc;

    node->nodes = malloc(argc * sizeof(compute_node *));

    return node;
}

compute_node *leaf_node() {
    return create_node(NULL, NULL, 0);
}

static void dfs_count(compute_node *node, size_t *cnt) {
    if (node == NULL) return;
    node->marked = true;
    ++(*cnt);
    for (int i = 0; i < (int)node->argc; i++) {
        if (!node->nodes[i]->marked)
            dfs_count(node->nodes[i], cnt);
    }
}

static void dfs_clear_mark(compute_node *node) {
    if (node == NULL) return;
    node->marked = false;
    for (int i = 0; i < (int)node->argc; i++) {
        if (node->nodes[i]->marked)
            dfs_clear_mark(node->nodes[i]);
    }
}

static void dfs_toposort(compute_node *node, compute_node **topo, size_t *counter) {
    if (node == NULL) return;
    node->marked = true;
    for (int i = 0; i < (int)node->argc; i++) {
        if (!node->nodes[i]->marked)
            dfs_toposort(node->nodes[i], topo, counter);
    }
    topo[(*counter)++] = node;
}

compute_node **toposort(compute_node *root, size_t *tot) {
    size_t cnt = 0;
    
    dfs_clear_mark(root);
    dfs_count(root, &cnt);

    size_t counter = 0;
    compute_node **topo = malloc(cnt * sizeof(compute_node *));

    dfs_clear_mark(root);
    dfs_toposort(root, topo, &counter);

    assert(counter == cnt);

    *tot = cnt;

    printf("Toposort finished. Node count: %zu\n", cnt);

    return topo;
}

void free_node(compute_node *node) {
    if (node->nodes != NULL) free(node->nodes);
    free(node);
}

void forward_add(compute_node *self) {
    assert(self->argc > 0);

    self->data = 0;

    for (int i = 0; i < (int)self->argc; i++) {
        self->data += self->nodes[i]->data;
    }
}

void backward_add(compute_node *self) {
    assert(self->argc > 0);

    for (int i = 0; i < (int)self->argc; i++) {
        self->nodes[i]->grad += self->grad;
    }
}

// void forward_mul(compute_node *self) {
//     assert(self->argc > 0);
//     float mul = 1;
//     for (int i = 0; i < (int)self->argc; i++) {
//         mul *= self->nodes[i]->data;
//     }
//     self->data = mul;
// }

// void backward_mul(compute_node *self) {
//     assert(self->argc > 0);

//     float *pref = calloc(self->argc, sizeof(float));
//     float *suff = calloc(self->argc, sizeof(float));
//     pref[0] = self->nodes[0]->data;
//     suff[(int)self->argc - 1] = self->nodes[(int)self->argc - 1]->data;
//     for (int i = 1; i < (int)self->argc; i++) {
//         pref[i] = pref[i - 1] * self->nodes[i]->data;
//     }

//     for (int i = (int)self->argc - 2; i >= 0; i--) {
//         suff[i] = suff[i + 1] * self->nodes[i]->data;
//     }

//     for (int i = 0; i < (int)self->argc; i++) {
//         self->nodes[i]->grad += self->grad * (i > 0 ? pref[i - 1] : 1) * (i < (int)self->argc - 1 ? suff[i + 1] : 1);
//     }

//     free(pref);
//     free(suff);
// }

void forward_mul(compute_node *self) {
    assert(self->argc == 2);
    self->data = self->nodes[0]->data * self->nodes[1]->data;
}

void backward_mul(compute_node *self) {
    assert(self->argc == 2);

    float a = self->nodes[0]->data;
    float b = self->nodes[1]->data;

    self->nodes[0]->grad += self->grad * b;
    self->nodes[1]->grad += self->grad * a;
}

void forward_relu(compute_node *self) {
    assert(self->argc == 1);
    self->data = max(0, self->nodes[0]->data);
}

void backward_relu(compute_node *self) {
    assert(self->argc == 1);
    self->nodes[0]->grad += (self->data > 0) * self->grad;
}

void forward_cross_entropy(compute_node *self) {
    assert(self->argc > 0);
    assert(self->ctx < self->argc);

    float mx = self->nodes[0]->data;

    for (int i = 1; i < (int)self->argc; i++) {
        if (self->nodes[i]->data > mx) {
            mx = self->nodes[i]->data;
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < (int)self->argc; i++) {
        sum += expf(self->nodes[i]->data - mx);
    }

    self->data = -(self->nodes[self->ctx]->data - mx) + logf(sum);
}

void backward_cross_entropy(compute_node *self) {
    float grad = self->grad; // 1

    float *vals = malloc(self->argc * sizeof(float));
    for (int i = 0; i < (int)self->argc; i++) {
        vals[i] = self->nodes[i]->data;
    }

    softmax(self->argc, vals);

    for (int i = 0; i < (int)self->argc; i++) {
        self->nodes[i]->grad += grad * (vals[i] - (i == self->ctx));
    }

    free(vals);
}