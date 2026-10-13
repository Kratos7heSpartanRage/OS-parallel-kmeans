/**
 * @file accumulator.h
 * @brief Thread-local cluster accumulation and reduction engine.
 *
 * Designed with cache-line alignment (alignas(64)) to eliminate false sharing
 * between threads during local accumulation.
 */

#ifndef ACCUMULATOR_H
#define ACCUMULATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdalign.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Thread-local accumulator padded to a multiple of 64 bytes (L1 cache line).
 *
 * Each worker thread `tid` updates exclusively its own `AlignedAccumulator[tid]`.
 * No mutex is acquired during distance computation and cluster assignment.
 */
typedef struct {
    alignas(64) double *sums;     /**< Cluster coordinate sums: size K * D */
    size_t *counts;                /**< Cluster point counts: size K */
    size_t k;                      /**< Number of clusters */
    size_t d;                      /**< Dimensionality */
    char padding[64];              /**< Explicit cache-line boundary padding */
} AlignedAccumulator;

/**
 * @brief Allocates an array of P aligned accumulators.
 *
 * @param nthreads Number of worker threads (P).
 * @param k Number of clusters (K).
 * @param d Dimensionality of data points (D).
 * @return AlignedAccumulator* Array of accumulators [P].
 */
AlignedAccumulator* accumulators_create(int nthreads, size_t k, size_t d);

/**
 * @brief Clears all accumulator sums and counts to zero before an iteration.
 *
 * @param accs Array of accumulators.
 * @param nthreads Number of worker threads.
 */
void accumulators_reset(AlignedAccumulator *accs, int nthreads);

/**
 * @brief Merges (reduces) all P local accumulators into global centroids.
 *
 * Called by main thread after `pool_wait_iteration` completes.
 *
 * @param accs Array of P accumulators.
 * @param nthreads Number of threads.
 * @param out_centroids Output centroid array [K * D].
 * @return double Maximum shift in centroid positions (for convergence check).
 */
double accumulators_reduce(const AlignedAccumulator *accs, int nthreads, double *out_centroids);

/**
 * @brief Frees accumulator array memory.
 *
 * @param accs Array of accumulators.
 * @param nthreads Number of worker threads.
 */
void accumulators_destroy(AlignedAccumulator *accs, int nthreads);

#ifdef __cplusplus
}
#endif

#endif /* ACCUMULATOR_H */
