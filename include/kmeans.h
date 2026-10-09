#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>

typedef struct {
    double *points;
    size_t n_points;
    size_t dimensions;
} Dataset;

typedef struct {
    double *centers;
    size_t k;
    size_t dimensions;
} Centroids;

typedef struct {
    size_t max_iterations;
    double tolerance;
} KMeansConfig;

int kmeans_sequential(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config
);

#endif
