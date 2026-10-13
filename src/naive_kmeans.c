#include "kmeans.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    const Dataset *dataset;
    const Centroids *centroids;
    size_t start;
    size_t end;
    double *global_sums;
    size_t *global_counts;
    pthread_mutex_t *mutex;
} NaiveWorker;

static double distance_squared(
    const double *a,
    const double *b,
    size_t dimensions
)
{
    double result = 0.0;

    for (size_t d = 0; d < dimensions; d++) {
        double diff = a[d] - b[d];
        result += diff * diff;
    }

    return result;
}

static size_t nearest_cluster(
    const double *point,
    const Centroids *centroids
)
{
    size_t best = 0;

    double best_distance = distance_squared(
        point, centroids->centers, centroids->dimensions
    );

    for (size_t c = 1; c < centroids->k; c++) {
        double dist = distance_squared(
            point,
            centroids->centers + c * centroids->dimensions,
            centroids->dimensions
        );

        if (dist < best_distance) {
            best_distance = dist;
            best = c;
        }
    }

    return best;
}

static void *naive_worker(void *arg)
{
    NaiveWorker *worker = arg;
    size_t dimensions = worker->dataset->dimensions;

    for (size_t i = worker->start; i < worker->end; i++) {
        const double *point =
            worker->dataset->points + i * dimensions;

        size_t cluster = nearest_cluster(
            point, worker->centroids
        );

        /* Deliberately use one shared mutex for accumulation. */
        pthread_mutex_lock(worker->mutex);

        worker->global_counts[cluster]++;

        for (size_t d = 0; d < dimensions; d++) {
            worker->global_sums[cluster * dimensions + d] +=
                point[d];
        }

        pthread_mutex_unlock(worker->mutex);
    }

    return NULL;
}

int kmeans_naive(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config,
    size_t num_threads
)
{
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

    double *sums = calloc(accumulator_size, sizeof(double));
    size_t *counts = calloc(k, sizeof(size_t));
    pthread_t *threads = calloc(num_threads, sizeof(pthread_t));
    NaiveWorker *workers = calloc(num_threads, sizeof(NaiveWorker));

    if (!sums || !counts || !threads || !workers) {
        free(sums);
        free(counts);
        free(threads);
        free(workers);
        return -1;
    }

    pthread_mutex_t mutex;

    if (pthread_mutex_init(&mutex, NULL) != 0) {
        free(sums);
        free(counts);
        free(threads);
        free(workers);
        return -1;
    }

    int result = 0;

    for (size_t iteration = 0;
         iteration < config->max_iterations;
         iteration++) {

        for (size_t i = 0; i < accumulator_size; i++)
            sums[i] = 0.0;

        for (size_t c = 0; c < k; c++)
            counts[c] = 0;

        size_t created = 0;

        for (size_t t = 0; t < num_threads; t++) {
            workers[t].dataset = dataset;
            workers[t].centroids = centroids;
            workers[t].start = dataset->n_points * t / num_threads;
            workers[t].end =
                dataset->n_points * (t + 1) / num_threads;
            workers[t].global_sums = sums;
            workers[t].global_counts = counts;
            workers[t].mutex = &mutex;

            if (pthread_create(
                    &threads[t], NULL, naive_worker, &workers[t]) != 0) {
                result = -1;
                break;
            }

            created++;
        }

        for (size_t t = 0; t < created; t++) {
            if (pthread_join(threads[t], NULL) != 0)
                result = -1;
        }

        if (result != 0)
            break;

        double max_change = 0.0;

        for (size_t c = 0; c < k; c++) {
            if (counts[c] == 0)
                continue;

            for (size_t d = 0; d < dimensions; d++) {
                size_t index = c * dimensions + d;
                double new_value = sums[index] / (double)counts[c];
                double change =
                    fabs(new_value - centroids->centers[index]);

                if (change > max_change)
                    max_change = change;

                centroids->centers[index] = new_value;
            }
        }

        if (max_change <= config->tolerance)
            break;
    }

    pthread_mutex_destroy(&mutex);
    free(sums);
    free(counts);
    free(threads);
    free(workers);

    return result;
}
