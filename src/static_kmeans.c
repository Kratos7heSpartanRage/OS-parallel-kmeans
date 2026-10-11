#include "kmeans.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    const Dataset *dataset;
    const Centroids *centroids;
    size_t start;
    size_t end;
    double *sums;
    size_t *counts;
} WorkerData;

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

static void *worker_function(void *arg)
{
    WorkerData *worker = (WorkerData *)arg;

    size_t dimensions = worker->dataset->dimensions;

    for (size_t i = worker->start; i < worker->end; i++) {
        const double *point =
            worker->dataset->points + i * dimensions;

        size_t cluster = nearest_centroid(
            point,
            worker->centroids
        );

        worker->counts[cluster]++;

        for (size_t d = 0; d < dimensions; d++) {
            worker->sums[cluster * dimensions + d] += point[d];
        }
    }

    return NULL;
}

int kmeans_static(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config,
    size_t num_threads
) {
    if (!dataset || !centroids || !config ||
        !dataset->points || !centroids->centers ||
        dataset->n_points == 0 ||
        dataset->dimensions == 0 ||
        centroids->k == 0 ||
        centroids->dimensions != dataset->dimensions ||
        num_threads == 0) {
        return -1;
    }

    if (num_threads > dataset->n_points) {
        num_threads = dataset->n_points;
    }

    size_t k = centroids->k;
    size_t dimensions = dataset->dimensions;
    size_t accumulator_size = k * dimensions;

    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
    WorkerData *workers = calloc(num_threads, sizeof(WorkerData));

    double *all_sums = calloc(
        num_threads * accumulator_size, sizeof(double)
    );
    size_t *all_counts = calloc(
        num_threads * k, sizeof(size_t)
    );

    double *total_sums = calloc(accumulator_size, sizeof(double));
    size_t *total_counts = calloc(k, sizeof(size_t));

    if (!threads || !workers || !all_sums || !all_counts ||
        !total_sums || !total_counts) {
        free(threads);
        free(workers);
        free(all_sums);
        free(all_counts);
        free(total_sums);
        free(total_counts);
        return -1;
    }

    int result = 0;

    for (size_t iteration = 0;
         iteration < config->max_iterations;
         iteration++) {

        for (size_t i = 0; i < num_threads * accumulator_size; i++) {
            all_sums[i] = 0.0;
        }

        for (size_t i = 0; i < num_threads * k; i++) {
            all_counts[i] = 0;
        }

        size_t created = 0;

        for (size_t t = 0; t < num_threads; t++) {
            size_t start = (dataset->n_points * t) / num_threads;
            size_t end =
                (dataset->n_points * (t + 1)) / num_threads;

            workers[t].dataset = dataset;
            workers[t].centroids = centroids;
            workers[t].start = start;
            workers[t].end = end;
            workers[t].sums = all_sums + t * accumulator_size;
            workers[t].counts = all_counts + t * k;

            int rc = pthread_create(
                &threads[t],
                NULL,
                worker_function,
                &workers[t]
            );

            if (rc != 0) {
                result = -1;
                break;
            }

            created++;
        }

        for (size_t t = 0; t < created; t++) {
            if (pthread_join(threads[t], NULL) != 0) {
                result = -1;
            }
        }

        if (result != 0) {
            break;
        }

        for (size_t i = 0; i < accumulator_size; i++) {
            total_sums[i] = 0.0;
        }

        for (size_t c = 0; c < k; c++) {
            total_counts[c] = 0;
        }

        for (size_t t = 0; t < num_threads; t++) {
            for (size_t i = 0; i < accumulator_size; i++) {
                total_sums[i] +=
                    all_sums[t * accumulator_size + i];
            }

            for (size_t c = 0; c < k; c++) {
                total_counts[c] += all_counts[t * k + c];
            }
        }

        double max_change = 0.0;

        for (size_t c = 0; c < k; c++) {
            if (total_counts[c] == 0) {
                continue;
            }

            for (size_t d = 0; d < dimensions; d++) {
                size_t index = c * dimensions + d;

                double new_value =
                    total_sums[index] / (double)total_counts[c];

                double change = fabs(
                    new_value - centroids->centers[index]
                );

                if (change > max_change) {
                    max_change = change;
                }

                centroids->centers[index] = new_value;
            }
        }

        if (max_change <= config->tolerance) {
            break;
        }
    }

    free(threads);
    free(workers);
    free(all_sums);
    free(all_counts);
    free(total_sums);
    free(total_counts);

    return result;
}
