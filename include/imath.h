#ifndef IMATH_H
#define IMATH_H

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define PI 3.14159265358979323846f

float rand_uniform();
float rand_normal();
float he_init(int fan_in);

void softmax(size_t n, float *input);

#endif