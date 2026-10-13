/**
 * @file synchronization.h
 * @brief Integration contract & API for Phase Barrier Synchronization & Local Accumulation.
 *
 * Provides:
 * 1. Phase Barrier Synchronization (POSIX pthread_barrier_t with custom fallback).
 * 2. Parallel multi-threaded reduction across cluster centroids to eliminate serial reduction bottleneck.
 * 3. Lock-free thread-local accumulation vs. lock-contended accumulation primitives for analysis.
 */

#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

#define _POSIX_C_SOURCE 200112L

#include "accumulator.h"
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* 1. PHASE BARRIER SYNCHRONIZATION                                          */
/* ========================================================================= */

#if defined(_POSIX_BARRIERS) && (_POSIX_BARRIERS > 0)
#define HAVE_PTHREAD_BARRIER 1
#endif

typedef struct {
#if defined(HAVE_PTHREAD_BARRIER)
    pthread_barrier_t native_barrier;
#else
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned int threshold;
    unsigned int in;
    unsigned int cycle;
#endif
    unsigned int thread_count;
} KMeansBarrier;

/**
 * @brief Initializes a phase barrier for thread synchronization.
 *
 * @param barrier Pointer to KMeansBarrier.
 * @param count Number of threads that must call wait before unblocking.
 * @return 0 on success, non-zero on error.
 */
int kmeans_barrier_init(KMeansBarrier *barrier, unsigned int count);

/**
 * @brief Synchronizes threads at the barrier.
 *
 * Blocks calling thread until all `count` threads have entered the barrier.
 *
 * @param barrier Pointer to KMeansBarrier.
 * @return 1 for one designated thread (serial leader), 0 for all other threads, or negative error.
 */
int kmeans_barrier_wait(KMeansBarrier *barrier);

/**
 * @brief Destroys barrier and releases internal synchronization primitives.
 *
 * @param barrier Pointer to KMeansBarrier.
 * @return 0 on success, non-zero on error.
 */
int kmeans_barrier_destroy(KMeansBarrier *barrier);

/* ========================================================================= */
/* 2. LOCAL ACCUMULATION PRIMITIVES                                          */
/* ========================================================================= */

/**
 * @brief Lock-free update of a thread's local accumulator.
 *
 * Highly optimized, completely non-blocking: each worker thread updates exclusively
 * its own AlignedAccumulator memory slice.
 *
 * @param acc Pointer to thread-local AlignedAccumulator.
 * @param cluster Assigned cluster index.
 * @param point Pointer to D-dimensional point coordinates.
 */
static inline void local_accumulate_point(
    AlignedAccumulator *acc,
    size_t cluster,
    const double *point
) {
    size_t d = acc->d;
    double *dest = &acc->sums[cluster * d];
    for (size_t dim = 0; dim < d; dim++) {
        dest[dim] += point[dim];
    }
    acc->counts[cluster]++;
}

/**
 * @brief Shared locked accumulator state used for benchmarking lock contention.
 */
typedef struct {
    double *sums;             /**< Global coordinate sums [K * D] */
    size_t *counts;           /**< Global point counts [K] */
    size_t k;
    size_t d;
    pthread_mutex_t lock;     /**< Central mutex guarding all updates */
} GlobalLockedAccumulator;

/**
 * @brief Allocates and initializes a global locked accumulator.
 */
GlobalLockedAccumulator* global_locked_accumulator_create(size_t k, size_t d);

/**
 * @brief Accumulates point by acquiring central mutex (demonstrates lock contention).
 */
void global_locked_accumulate_point(
    GlobalLockedAccumulator *gacc,
    size_t cluster,
    const double *point
);

/**
 * @brief Resets sums and counts in global locked accumulator.
 */
void global_locked_accumulator_reset(GlobalLockedAccumulator *gacc);

/**
 * @brief Computes centroids and maximum shift from global locked accumulator.
 */
double global_locked_accumulator_finish(
    const GlobalLockedAccumulator *gacc,
    double *out_centroids
);

/**
 * @brief Frees global locked accumulator.
 */
void global_locked_accumulator_destroy(GlobalLockedAccumulator *gacc);

/* ========================================================================= */
/* 3. PARALLEL & TREE-BASED REDUCTIONS                                       */
/* ========================================================================= */

/**
 * @brief Parallel reduction slice executed by thread tid.
 *
 * Divides the K clusters evenly among P worker threads.
 * Each worker thread sums its assigned slice of clusters [k_start, k_end)
 * across all P thread accumulators and updates the corresponding centroid positions.
 *
 * @param accs Array of P AlignedAccumulator structs.
 * @param nthreads Total number of threads (P).
 * @param tid Current thread ID [0, P - 1].
 * @param out_centroids Output centroid array [K * D].
 * @return double Local maximum shift among the clusters reduced by this thread.
 */
double parallel_reduction_step(
    const AlignedAccumulator *accs,
    int nthreads,
    int tid,
    double *out_centroids
);

/**
 * @brief Full coordinated parallel reduction across all threads.
 *
 * Called cooperatively by all P worker threads using barrier synchronization:
 * 1. Each thread reduces its cluster partition in parallel.
 * 2. Threads synchronize at the barrier.
 * 3. Returns the global maximum centroid shift across all clusters.
 *
 * @param accs Array of P AlignedAccumulator structs.
 * @param nthreads Total worker count P.
 * @param tid Thread index of caller [0, P - 1].
 * @param barrier Phase barrier pointer.
 * @param out_centroids Centroid buffer [K * D].
 * @param max_shifts Array of size [P] used to collect per-thread maximum shifts.
 * @return double Global maximum shift across all K clusters.
 */
double parallel_reduction_barrier(
    const AlignedAccumulator *accs,
    int nthreads,
    int tid,
    KMeansBarrier *barrier,
    double *out_centroids,
    double *max_shifts
);

#ifdef __cplusplus
}
#endif

#endif /* SYNCHRONIZATION_H */
