/**
 * @file kmeans.h
 * @brief K-Means Clustering API: Sequential, Naive MT, Static MT, and Parallel Thread-Pool Engine.
 */

#ifndef KMEANS_H
#define KMEANS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Person 4: Dataset, Centroids, and Baseline Implementations                */
/* ========================================================================= */

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

int kmeans_static(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config,
    size_t num_threads
);

int kmeans_naive(
    const Dataset *dataset,
    Centroids *centroids,
    const KMeansConfig *config,
    size_t num_threads
);

/* ========================================================================= */
/* Parallel Dynamic Thread Pool Integration Contract                         */
/* ========================================================================= */

#include "thread_pool.h"
#include "accumulator.h"

/**
 * @brief Shared context passed via Task.ctx into the worker thread callback.
 */
typedef struct {
    const double *points;        /**< Input dataset array [N * D] */
    const double *centroids;     /**< Current centroid positions [K * D] */
    int *assignments;            /**< Cluster labels assigned to points [N] */
    size_t n_points;             /**< Total points N */
    size_t k;                    /**< Number of clusters K */
    size_t d;                    /**< Dimensions D */
    AlignedAccumulator *local_accs; /**< Array of P thread-local accumulators */
} KMeansContext;

/**
 * @brief Worker callback function executed dynamically by the thread pool.
 *
 * Implements the distance calculation and assignment for points [start_idx, end_idx).
 *
 * @param task The task specifying point range.
 * @param tid Worker thread index [0, P - 1].
 * @param ctx Pointer to KMeansContext.
 */
void kmeans_assign_chunk_fn(const Task *task, int tid, void *ctx);

/**
 * @brief Parallel K-Means execution driver using Person 1's Thread Pool.
 *
 * @param points Array of N * D data points.
 * @param n Number of data points.
 * @param d Dimensions.
 * @param k Number of clusters.
 * @param max_iters Maximum iterations.
 * @param tolerance Convergence tolerance.
 * @param nthreads Worker thread count P.
 * @param chunk_size Points per dynamic task chunk.
 * @param out_centroids Output array of K * D centroids.
 * @param out_assignments Output array of N cluster labels.
 * @return int Number of iterations until convergence.
 */
int kmeans_parallel_run(
    const double *points,
    size_t n,
    size_t d,
    size_t k,
    int max_iters,
    double tolerance,
    int nthreads,
    size_t chunk_size,
    double *out_centroids,
    int *out_assignments
);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_H */
