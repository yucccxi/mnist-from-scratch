#include <math.h>
#include <stdlib.h>

#include "imath.h"

float rand_uniform() {
    return ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
}

float rand_normal() {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * PI * u2);
}

float he_init(int fan_in) {
    return rand_normal() * sqrtf(2.0f / fan_in);
}

void softmax(size_t n, float *vals) {
    float m = 0;
    for (int i = 0; i < (int)n; i++) {
        m = max(m, vals[i]);
    }

    float tot = 0;
    for (int i = 0; i < (int)n; i++) {
        vals[i] -= m;
        tot += exp(vals[i]);
    }

    for (int i = 0; i < (int)n; i++) {
        vals[i] = exp(vals[i]) / tot;
    }
}