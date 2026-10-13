/**
 * @file kmeans.h
 * @brief Integration contract for Parallel & Adaptive K-Means Engine.
 *
 * Implements high-performance clustering with dynamic task distribution,
 * cache-aligned local accumulation, and adaptive sequential cut-off
 * to eliminate parallelization overhead on small workloads.
 */

#ifndef KMEANS_H
#define KMEANS_H

#include "thread_pool.h"
#include "accumulator.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Default empirical threshold below which parallel overhead
 * (thread wakeup, queue locks, condvar signaling, reduction) exceeds computation time.
 */
#define DEFAULT_PARALLEL_THRESHOLD_POINTS 10000

/**
 * @brief Shared context passed via Task.ctx into the worker thread callback.
 */
typedef struct {
    const double *points;        /**< Input dataset array [N * D] */
    const double *centroids;     /**< Current centroid positions [K * D] */
    int *assignments;            /**< Cluster labels assigned to points [N] (optional, can be NULL) */
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
 * @brief Pure Sequential K-Means execution baseline.
 *
 * Runs on the calling thread with zero multi-threading overhead
 * (no mutexes, no condition variables, no task queue, single accumulator).
 *
 * @param points Array of N * D data points.
 * @param n Number of data points.
 * @param d Dimensions.
 * @param k Number of clusters.
 * @param max_iters Maximum iterations.
 * @param tolerance Convergence tolerance.
 * @param out_centroids Output array of K * D centroids.
 * @param out_assignments Output array of N cluster labels (can be NULL).
 * @return int Number of iterations until convergence.
 */
int kmeans_sequential_run(
    const double *points,
    size_t n,
    size_t d,
    size_t k,
    int max_iters,
    double tolerance,
    double *out_centroids,
    int *out_assignments
);

/**
 * @brief Parallel K-Means execution driver using Thread Pool & Dynamic Task Queue.
 *
 * Forces multi-threaded execution across P worker threads.
 *
 * @param points Array of N * D data points.
 * @param n Number of data points.
 * @param d Dimensions.
 * @param k Number of clusters.
 * @param max_iters Maximum iterations.
 * @param tolerance Convergence tolerance.
 * @param nthreads Worker thread count P.
 * @param chunk_size Points per dynamic task chunk (0 for default).
 * @param out_centroids Output array of K * D centroids.
 * @param out_assignments Output array of N cluster labels (can be NULL).
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

/**
 * @brief Adaptive Hybrid K-Means execution driver.
 *
 * Solves the Parallel Overhead problem:
 * If N < threshold or nthreads <= 1, automatically routes execution to
 * sequential execution, bypassing thread synchronization penalties.
 * If N >= threshold, routes execution to the multi-threaded parallel engine.
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
 * @param out_assignments Output array of N cluster labels (can be NULL).
 * @return int Number of iterations until convergence.
 */
int kmeans_adaptive_run(
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

/**
 * @brief Gets the current sequential cut-off threshold (in number of points N).
 */
size_t kmeans_get_parallel_threshold(void);

/**
 * @brief Sets the sequential cut-off threshold (in number of points N).
 */
void kmeans_set_parallel_threshold(size_t threshold);

/**
 * @brief Evaluates whether parallelization is beneficial for the given workload.
 *
 * @param n Number of data points.
 * @param nthreads Number of threads.
 * @return true if parallel computing is expected to be faster than sequential.
 */
bool kmeans_is_parallel_beneficial(size_t n, int nthreads);

#ifdef __cplusplus
}
#endif

#endif /* KMEANS_H */
