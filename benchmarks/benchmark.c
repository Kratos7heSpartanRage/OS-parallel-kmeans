#define _POSIX_C_SOURCE 200809L





#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/kmeans.h"

static double get_time_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1000000000.0;
}

static void generate_dataset(
    double *points,
    size_t n_points,
    size_t dimensions
)
{
    for (size_t i = 0; i < n_points; i++) {
        for (size_t d = 0; d < dimensions; d++) {
            points[i * dimensions + d] =
                (double)(rand() % 10000) / 100.0;
        }
    }
}

int main(void)
{
    const size_t n_points = 10000;
    const size_t dimensions = 2;
    const size_t k = 3;

    double *points =
        malloc(n_points * dimensions * sizeof(double));

    double *centers =
        malloc(k * dimensions * sizeof(double));

    if (!points || !centers) {
        fprintf(stderr, "Memory allocation failed.\n");

        free(points);
        free(centers);

        return 1;
    }

    srand(42);

    generate_dataset(
        points,
        n_points,
        dimensions
    );

    /*
     * Use the first k points as initial centroids.
     */
    for (size_t c = 0; c < k; c++) {
        for (size_t d = 0; d < dimensions; d++) {
            centers[c * dimensions + d] =
                points[c * dimensions + d];
        }
    }

    Dataset dataset = {
        .points = points,
        .n_points = n_points,
        .dimensions = dimensions
    };

    Centroids centroids = {
        .centers = centers,
        .k = k,
        .dimensions = dimensions
    };

    KMeansConfig config = {
        .max_iterations = 100,
        .tolerance = 0.0001
    };

    double start = get_time_seconds();

    int result = kmeans_sequential(
        &dataset,
        &centroids,
        &config
    );

    double end = get_time_seconds();

    if (result != 0) {
        fprintf(stderr, "K-Means failed.\n");

        free(points);
        free(centers);

        return 1;
    }

    printf("========================================\n");
    printf("        K-Means Benchmark\n");
    printf("========================================\n");
    printf("Dataset size : %zu points\n", n_points);
    printf("Dimensions   : %zu\n", dimensions);
    printf("Clusters     : %zu\n", k);
    printf("Execution    : %.6f seconds\n", end - start);
    printf("========================================\n");

    free(points);
    free(centers);

    return 0;
}
