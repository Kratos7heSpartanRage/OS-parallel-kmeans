/**
 * @file kmeans.c
 * @brief High-performance Parallel, Sequential, and Adaptive K-Means implementations.
 */

#include "kmeans.h"
#include "accumulator.h"
#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static size_t g_parallel_threshold = DEFAULT_PARALLEL_THRESHOLD_POINTS;

size_t kmeans_get_parallel_threshold(void) {
    return g_parallel_threshold;
}

void kmeans_set_parallel_threshold(size_t threshold) {
    g_parallel_threshold = threshold;
}

bool kmeans_is_parallel_beneficial(size_t n, int nthreads) {
    if (nthreads <= 1) return false;
    return (n >= g_parallel_threshold);
}

static inline double euclidean_dist_sq(const double *a, const double *b, size_t d) {
    double dist = 0.0;
    for (size_t i = 0; i < d; i++) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

void kmeans_assign_chunk_fn(const Task *task, int tid, void *ctx) {
    KMeansContext *kctx = (KMeansContext *)ctx;
    AlignedAccumulator *my_acc = &kctx->local_accs[tid];
    const double *points = kctx->points;
    const double *centroids = kctx->centroids;
    int *assignments = kctx->assignments;
    size_t k = kctx->k;
    size_t d = kctx->d;

    for (size_t p = task->start_idx; p < task->end_idx; p++) {
        const double *pt = &points[p * d];
        int best_c = 0;
        double min_d = euclidean_dist_sq(pt, &centroids[0], d);

        for (size_t c = 1; c < k; c++) {
            double dist = euclidean_dist_sq(pt, &centroids[c * d], d);
            if (dist < min_d) {
                min_d = dist;
                best_c = (int)c;
            }
        }

        if (assignments) {
            assignments[p] = best_c;
        }

        /* Accumulate to thread-local cache-padded accumulator */
        for (size_t dim = 0; dim < d; dim++) {
            my_acc->sums[best_c * d + dim] += pt[dim];
        }
        my_acc->counts[best_c]++;
    }
}

int kmeans_sequential_run(
    const double *points,
    size_t n,
    size_t d,
    size_t k,
    int max_iters,
    double tolerance,
    double *out_centroids,
    int *out_assignments
) {
    if (!points || !out_centroids || n == 0 || d == 0 || k == 0 || max_iters <= 0) {
        return 0;
    }

    double *sums = (double *)calloc(k * d, sizeof(double));
    size_t *counts = (size_t *)calloc(k, sizeof(size_t));
    if (!sums || !counts) {
        if (sums) free(sums);
        if (counts) free(counts);
        return 0;
    }

    int iter = 0;
    for (; iter < max_iters; iter++) {
        memset(sums, 0, k * d * sizeof(double));
        memset(counts, 0, k * sizeof(size_t));

        /* Point assignment and accumulation in a tight single-thread loop */
        for (size_t p = 0; p < n; p++) {
            const double *pt = &points[p * d];
            int best_c = 0;
            double min_d = euclidean_dist_sq(pt, &out_centroids[0], d);

            for (size_t c = 1; c < k; c++) {
                double dist = euclidean_dist_sq(pt, &out_centroids[c * d], d);
                if (dist < min_d) {
                    min_d = dist;
                    best_c = (int)c;
                }
            }

            if (out_assignments) {
                out_assignments[p] = best_c;
            }

            for (size_t dim = 0; dim < d; dim++) {
                sums[best_c * d + dim] += pt[dim];
            }
            counts[best_c]++;
        }

        /* Centroid recalculation and convergence shift check */
        double max_shift = 0.0;
        for (size_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                double shift_sq = 0.0;
                for (size_t dim = 0; dim < d; dim++) {
                    double new_val = sums[c * d + dim] / (double)counts[c];
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

        if (max_shift < tolerance) {
            iter++;
            break;
        }
    }

    free(sums);
    free(counts);
    return iter;
}

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
) {
    if (!points || !out_centroids || n == 0 || d == 0 || k == 0 || max_iters <= 0) {
        return 0;
    }

    if (nthreads <= 1) {
        return kmeans_sequential_run(points, n, d, k, max_iters, tolerance, out_centroids, out_assignments);
    }

    if (chunk_size == 0) {
        chunk_size = 2048;
    }
    if (chunk_size > n) {
        chunk_size = n;
    }

    size_t num_tasks = (n + chunk_size - 1) / chunk_size;
    size_t queue_cap = num_tasks * 2 + 32;

    AlignedAccumulator *accs = accumulators_create(nthreads, k, d);
    if (!accs) return 0;

    ThreadPool *pool = pool_create(nthreads, queue_cap, kmeans_assign_chunk_fn);
    if (!pool) {
        accumulators_destroy(accs, nthreads);
        return 0;
    }

    KMeansContext kctx = {
        .points = points,
        .centroids = out_centroids,
        .assignments = out_assignments,
        .n_points = n,
        .k = k,
        .d = d,
        .local_accs = accs
    };

    Task *tasks = (Task *)malloc(num_tasks * sizeof(Task));
    if (!tasks) {
        pool_shutdown(pool);
        accumulators_destroy(accs, nthreads);
        return 0;
    }

    for (size_t i = 0; i < num_tasks; i++) {
        size_t start = i * chunk_size;
        size_t end = (start + chunk_size > n) ? n : (start + chunk_size);
        tasks[i].start_idx = start;
        tasks[i].end_idx = end;
        tasks[i].task_id = (int)i;
        tasks[i].ctx = &kctx;
    }

    int iter = 0;
    for (; iter < max_iters; iter++) {
        accumulators_reset(accs, nthreads);

        pool_submit_batch(pool, tasks, num_tasks);
        pool_wait_iteration(pool);

        double max_shift = accumulators_reduce(accs, nthreads, out_centroids);
        if (max_shift < tolerance) {
            iter++;
            break;
        }
    }

    free(tasks);
    pool_shutdown(pool);
    accumulators_destroy(accs, nthreads);

    return iter;
}

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
) {
    /* 1. Sequential cut-off: below empirical break-even point, avoid threading overhead */
    if (nthreads <= 1 || n < g_parallel_threshold) {
        return kmeans_sequential_run(points, n, d, k, max_iters, tolerance, out_centroids, out_assignments);
    }

    /* 2. Adaptive Concurrency Scaling: avoid oversaturating threads on small-to-medium sizes */
    int effective_threads = nthreads;
    if (n < 10000 && effective_threads > 2) {
        effective_threads = 2;
    } else if (n < 25000 && effective_threads > 4) {
        effective_threads = 4;
    }

    return kmeans_parallel_run(
        points, n, d, k, max_iters, tolerance,
        effective_threads, chunk_size, out_centroids, out_assignments
    );
}
