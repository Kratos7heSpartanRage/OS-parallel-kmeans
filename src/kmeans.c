#include "kmeans.h"
#include <math.h>
#include <stdlib.h>

static double squared_distance(
    const double *a,
    const double *b,
    size_t dimensions
) {
    double distance = 0.0;

    for (size_t d = 0; d < dimensions; d++) {
        double diff = a[d] - b[d];
        distance += diff * diff;
    }

    return distance;
}

static size_t nearest_centroid(
    const double *point,
    const Centroids *centroids
) {
    size_t best_cluster = 0;

    double best_distance = squared_distance(
        point,
        centroids->centers,
        centroids->dimensions
    );

    for (size_t c = 1; c < centroids->k; c++) {
        double distance = squared_distance(
            point,
            centroids->centers + c * centroids->dimensions,
            centroids->dimensions
        );

        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = c;
        }
    }

    return best_cluster;
}

int kmeans_sequential(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config
) {
    if (!dataset || !centroids || !config ||
        !dataset->points || !centroids->centers ||
        dataset->n_points == 0 ||
        dataset->dimensions == 0 ||
        centroids->k == 0 ||
        centroids->dimensions != dataset->dimensions) {
        return -1;
    }

    size_t k = centroids->k;
    size_t dimensions = dataset->dimensions;

    double *sums = calloc(k * dimensions, sizeof(double));
    size_t *counts = calloc(k, sizeof(size_t));

    if (!sums || !counts) {
        free(sums);
        free(counts);
        return -1;
    }

    for (size_t iteration = 0;
         iteration < config->max_iterations;
         iteration++) {

        for (size_t i = 0; i < k * dimensions; i++) {
            sums[i] = 0.0;
        }

        for (size_t c = 0; c < k; c++) {
            counts[c] = 0;
        }

        /*
         * Assignment + accumulation phase
         */
        for (size_t i = 0; i < dataset->n_points; i++) {

            const double *point =
                dataset->points + i * dimensions;

            size_t cluster =
                nearest_centroid(point, centroids);

            counts[cluster]++;

            for (size_t d = 0; d < dimensions; d++) {
                sums[cluster * dimensions + d] += point[d];
            }
        }

        /*
         * Update centroids and calculate maximum movement.
         */
        double max_change = 0.0;

        for (size_t c = 0; c < k; c++) {

            if (counts[c] == 0) {
                continue;
            }

            for (size_t d = 0; d < dimensions; d++) {

                size_t index = c * dimensions + d;

                double new_value =
                    sums[index] / (double)counts[c];

                double change =
                    fabs(new_value - centroids->centers[index]);

                if (change > max_change) {
                    max_change = change;
                }

                centroids->centers[index] = new_value;
            }
        }

        /*
         * Convergence check.
         */
        if (max_change <= config->tolerance) {
            break;
        }
    }

    free(sums);
    free(counts);

    return 0;
}
