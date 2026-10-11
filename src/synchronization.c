/**
 * @file synchronization.c
 * @brief Implementation of Person 3: Synchronization & Local Accumulation.
 */

#define _POSIX_C_SOURCE 200112L

#include "synchronization.h"
#include "aligned_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* 1. PHASE BARRIER SYNCHRONIZATION IMPLEMENTATION                           */
/* ========================================================================= */

int kmeans_barrier_init(KMeansBarrier *barrier, unsigned int count) {
    if (!barrier || count == 0) return -1;
    barrier->thread_count = count;

#if defined(HAVE_PTHREAD_BARRIER)
    return pthread_barrier_init(&barrier->native_barrier, NULL, count);
#else
    int rc = pthread_mutex_init(&barrier->mutex, NULL);
    if (rc != 0) return rc;

    rc = pthread_cond_init(&barrier->cond, NULL);
    if (rc != 0) {
        pthread_mutex_destroy(&barrier->mutex);
        return rc;
    }

    barrier->threshold = count;
    barrier->in = 0;
    barrier->cycle = 0;
    return 0;
#endif
}

int kmeans_barrier_wait(KMeansBarrier *barrier) {
    if (!barrier) return -1;

#if defined(HAVE_PTHREAD_BARRIER)
    int rc = pthread_barrier_wait(&barrier->native_barrier);
    if (rc == PTHREAD_BARRIER_SERIAL_THREAD) {
        return 1;
    } else if (rc == 0) {
        return 0;
    }
    return -1;
#else
    pthread_mutex_lock(&barrier->mutex);
    unsigned int current_cycle = barrier->cycle;

    if (++barrier->in == barrier->threshold) {
        /* Last thread to reach the barrier */
        barrier->cycle++;
        barrier->in = 0;
        pthread_cond_broadcast(&barrier->cond);
        pthread_mutex_unlock(&barrier->mutex);
        return 1; /* Serial thread flag */
    }

    /* Wait for the current cycle to complete */
    while (barrier->cycle == current_cycle) {
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
    }

    pthread_mutex_unlock(&barrier->mutex);
    return 0;
#endif
}

int kmeans_barrier_destroy(KMeansBarrier *barrier) {
    if (!barrier) return -1;

#if defined(HAVE_PTHREAD_BARRIER)
    return pthread_barrier_destroy(&barrier->native_barrier);
#else
    pthread_cond_destroy(&barrier->cond);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
#endif
}

/* ========================================================================= */
/* 2. GLOBAL LOCKED ACCUMULATOR (FOR CONTENTION BENCHMARKING)                */
/* ========================================================================= */

GlobalLockedAccumulator* global_locked_accumulator_create(size_t k, size_t d) {
    if (k == 0 || d == 0) return NULL;

    GlobalLockedAccumulator *gacc = (GlobalLockedAccumulator *)calloc(1, sizeof(GlobalLockedAccumulator));
    if (!gacc) return NULL;

    gacc->k = k;
    gacc->d = d;
    gacc->sums = (double *)calloc(k * d, sizeof(double));
    gacc->counts = (size_t *)calloc(k, sizeof(size_t));

    if (!gacc->sums || !gacc->counts) {
        if (gacc->sums) free(gacc->sums);
        if (gacc->counts) free(gacc->counts);
        free(gacc);
        return NULL;
    }

    if (pthread_mutex_init(&gacc->lock, NULL) != 0) {
        free(gacc->sums);
        free(gacc->counts);
        free(gacc);
        return NULL;
    }

    return gacc;
}

void global_locked_accumulate_point(
    GlobalLockedAccumulator *gacc,
    size_t cluster,
    const double *point
) {
    pthread_mutex_lock(&gacc->lock);
    size_t d = gacc->d;
    double *dest = &gacc->sums[cluster * d];
    for (size_t dim = 0; dim < d; dim++) {
        dest[dim] += point[dim];
    }
    gacc->counts[cluster]++;
    pthread_mutex_unlock(&gacc->lock);
}

void global_locked_accumulator_reset(GlobalLockedAccumulator *gacc) {
    if (!gacc) return;
    pthread_mutex_lock(&gacc->lock);
    memset(gacc->sums, 0, gacc->k * gacc->d * sizeof(double));
    memset(gacc->counts, 0, gacc->k * sizeof(size_t));
    pthread_mutex_unlock(&gacc->lock);
}

double global_locked_accumulator_finish(
    const GlobalLockedAccumulator *gacc,
    double *out_centroids
) {
    if (!gacc || !out_centroids) return 0.0;

    size_t k = gacc->k;
    size_t d = gacc->d;
    double max_shift = 0.0;

    for (size_t c = 0; c < k; c++) {
        if (gacc->counts[c] > 0) {
            double shift_sq = 0.0;
            double inv_cnt = 1.0 / (double)gacc->counts[c];
            for (size_t dim = 0; dim < d; dim++) {
                double new_val = gacc->sums[c * d + dim] * inv_cnt;
                double old_val = out_centroids[c * d + dim];
                double diff = new_val - old_val;
                shift_sq += diff * diff;
                out_centroids[c * d + dim] = new_val;
            }
            double shift = sqrt(shift_sq);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }
    }

    return max_shift;
}

void global_locked_accumulator_destroy(GlobalLockedAccumulator *gacc) {
    if (!gacc) return;
    pthread_mutex_destroy(&gacc->lock);
    free(gacc->sums);
    free(gacc->counts);
    free(gacc);
}

/* ========================================================================= */
/* 3. PARALLEL REDUCTION IMPLEMENTATION                                      */
/* ========================================================================= */

double parallel_reduction_step(
    const AlignedAccumulator *accs,
    int nthreads,
    int tid,
    double *out_centroids
) {
    if (!accs || nthreads <= 0 || tid < 0 || tid >= nthreads || !out_centroids) {
        return 0.0;
    }

    size_t k = accs[0].k;
    size_t d = accs[0].d;

    /* Partition K clusters evenly across P threads */
    size_t base_chunk = k / nthreads;
    size_t remainder = k % nthreads;

    size_t k_start, k_end;
    if ((size_t)tid < remainder) {
        k_start = tid * (base_chunk + 1);
        k_end = k_start + (base_chunk + 1);
    } else {
        k_start = remainder * (base_chunk + 1) + (tid - remainder) * base_chunk;
        k_end = k_start + base_chunk;
    }

    double local_max_shift = 0.0;

    /* For each cluster assigned to thread tid */
    for (size_t c = k_start; c < k_end; c++) {
        size_t total_count = 0;
        /* Temporary stack buffer for dimension sums if small, or heap if large */
        double local_sum[128];
        double *sum_buf = local_sum;
        if (d > 128) {
            sum_buf = (double *)calloc(d, sizeof(double));
        } else {
            memset(local_sum, 0, d * sizeof(double));
        }

        /* Sum cluster c across all P thread accumulators */
        for (int t = 0; t < nthreads; t++) {
            total_count += accs[t].counts[c];
            const double *t_sums = &accs[t].sums[c * d];
            for (size_t dim = 0; dim < d; dim++) {
                sum_buf[dim] += t_sums[dim];
            }
        }

        /* If cluster has assigned points, update centroid & shift */
        if (total_count > 0) {
            double shift_sq = 0.0;
            double inv_cnt = 1.0 / (double)total_count;
            for (size_t dim = 0; dim < d; dim++) {
                double new_val = sum_buf[dim] * inv_cnt;
                double old_val = out_centroids[c * d + dim];
                double diff = new_val - old_val;
                shift_sq += diff * diff;
                out_centroids[c * d + dim] = new_val;
            }
            double shift = sqrt(shift_sq);
            if (shift > local_max_shift) {
                local_max_shift = shift;
            }
        }

        if (d > 128) {
            free(sum_buf);
        }
    }

    return local_max_shift;
}

double parallel_reduction_barrier(
    const AlignedAccumulator *accs,
    int nthreads,
    int tid,
    KMeansBarrier *barrier,
    double *out_centroids,
    double *max_shifts
) {
    /* Step 1: Thread computes its cluster slice reduction independently */
    double my_shift = parallel_reduction_step(accs, nthreads, tid, out_centroids);
    if (max_shifts) {
        max_shifts[tid] = my_shift;
    }

    /* Step 2: Global barrier synchronization guarantees all slices complete */
    kmeans_barrier_wait(barrier);

    /* Step 3: Find global max shift */
    double global_max = 0.0;
    if (max_shifts) {
        for (int t = 0; t < nthreads; t++) {
            if (max_shifts[t] > global_max) {
                global_max = max_shifts[t];
            }
        }
    }

    /* Step 4: Final barrier ensures no thread races ahead before reduction completes */
    kmeans_barrier_wait(barrier);

    return global_max;
}
