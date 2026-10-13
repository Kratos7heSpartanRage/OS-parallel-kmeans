/**
 * @file synchronization_benchmark.c
 * @brief Benchmark suite demonstrating Lock Contention Elimination & Reduction Scaling.
 *
 * Compares:
 * 1. Coarse Lock Contention: All threads compete on a global mutex for every point assignment.
 * 2. Lock-Free Thread-Local Accumulation with Sequential Reduction (Thread-Local approach).
 * 3. Lock-Free Thread-Local Accumulation with Parallel Cluster-Partitioned Barrier Reduction (Cluster-Partitioned approach).
 */

#define _POSIX_C_SOURCE 200112L

#include "synchronization.h"
#include "accumulator.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#define N_POINTS 500000
#define K_CLUSTERS 64
#define DIM 16
#define ITERATIONS 5

static uint64_t get_time_ns_sync(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

/* ========================================================================= */
/* BENCHMARK 1: Global Mutex Contention                                      */
/* ========================================================================= */
typedef struct {
    GlobalLockedAccumulator *gacc;
    const double *points;
    const int *assignments;
    size_t start_idx;
    size_t end_idx;
    int iterations;
    KMeansBarrier *barrier;
} LockedBenchArg;

static void* locked_bench_worker(void *arg) {
    LockedBenchArg *a = (LockedBenchArg *)arg;
    size_t d = a->gacc->d;

    for (int iter = 0; iter < a->iterations; iter++) {
        for (size_t p = a->start_idx; p < a->end_idx; p++) {
            int c = a->assignments[p];
            const double *pt = &a->points[p * d];
            /* Heavy lock contention: every point locks mutex */
            global_locked_accumulate_point(a->gacc, c, pt);
        }
        /* Barrier sync between iterations */
        kmeans_barrier_wait(a->barrier);
    }
    return NULL;
}

static double run_locked_contention_bench(
    int num_threads,
    const double *points,
    const int *assignments
) {
    GlobalLockedAccumulator *gacc = global_locked_accumulator_create(K_CLUSTERS, DIM);
    KMeansBarrier barrier;
    kmeans_barrier_init(&barrier, num_threads);

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    LockedBenchArg *args = (LockedBenchArg *)malloc(num_threads * sizeof(LockedBenchArg));
    size_t chunk = N_POINTS / num_threads;

    uint64_t t0 = get_time_ns_sync();

    for (int t = 0; t < num_threads; t++) {
        args[t].gacc = gacc;
        args[t].points = points;
        args[t].assignments = assignments;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t == num_threads - 1) ? N_POINTS : (t + 1) * chunk;
        args[t].iterations = ITERATIONS;
        args[t].barrier = &barrier;
        pthread_create(&threads[t], NULL, locked_bench_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    uint64_t elapsed_ns = get_time_ns_sync() - t0;

    kmeans_barrier_destroy(&barrier);
    global_locked_accumulator_destroy(gacc);
    free(threads);
    free(args);

    return (double)elapsed_ns / 1e6;
}

/* ========================================================================= */
/* BENCHMARK 2: Thread-Local Accumulation with Sequential Reduction          */
/* ========================================================================= */
typedef struct {
    AlignedAccumulator *accs;
    const double *points;
    const int *assignments;
    size_t start_idx;
    size_t end_idx;
    int tid;
    int iterations;
    int nthreads;
    KMeansBarrier *barrier;
    double *centroids;
} LocalSeqBenchArg;

static void* local_seq_bench_worker(void *arg) {
    LocalSeqBenchArg *a = (LocalSeqBenchArg *)arg;
    AlignedAccumulator *my_acc = &a->accs[a->tid];
    size_t d = my_acc->d;

    for (int iter = 0; iter < a->iterations; iter++) {
        for (size_t p = a->start_idx; p < a->end_idx; p++) {
            int c = a->assignments[p];
            const double *pt = &a->points[p * d];
            local_accumulate_point(my_acc, c, pt);
        }

        /* Barrier before reduction */
        int is_leader = kmeans_barrier_wait(a->barrier);

        /* Single leader performs sequential reduction */
        if (is_leader == 1) {
            accumulators_reduce(a->accs, a->nthreads, a->centroids);
            accumulators_reset(a->accs, a->nthreads);
        }

        /* Barrier to let leader finish reduction before next iteration */
        kmeans_barrier_wait(a->barrier);
    }
    return NULL;
}

static double run_local_seq_bench(
    int num_threads,
    const double *points,
    const int *assignments
) {
    AlignedAccumulator *accs = accumulators_create(num_threads, K_CLUSTERS, DIM);
    KMeansBarrier barrier;
    kmeans_barrier_init(&barrier, num_threads);
    double *centroids = (double *)calloc(K_CLUSTERS * DIM, sizeof(double));

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    LocalSeqBenchArg *args = (LocalSeqBenchArg *)malloc(num_threads * sizeof(LocalSeqBenchArg));
    size_t chunk = N_POINTS / num_threads;

    uint64_t t0 = get_time_ns_sync();

    for (int t = 0; t < num_threads; t++) {
        args[t].accs = accs;
        args[t].points = points;
        args[t].assignments = assignments;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t == num_threads - 1) ? N_POINTS : (t + 1) * chunk;
        args[t].tid = t;
        args[t].iterations = ITERATIONS;
        args[t].nthreads = num_threads;
        args[t].barrier = &barrier;
        args[t].centroids = centroids;
        pthread_create(&threads[t], NULL, local_seq_bench_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    uint64_t elapsed_ns = get_time_ns_sync() - t0;

    kmeans_barrier_destroy(&barrier);
    accumulators_destroy(accs, num_threads);
    free(centroids);
    free(threads);
    free(args);

    return (double)elapsed_ns / 1e6;
}

/* ========================================================================= */
/* BENCHMARK 3: Thread-Local Accumulation with Parallel Barrier Reduction    */
/* ========================================================================= */
typedef struct {
    AlignedAccumulator *accs;
    const double *points;
    const int *assignments;
    size_t start_idx;
    size_t end_idx;
    int tid;
    int iterations;
    int nthreads;
    KMeansBarrier *barrier;
    double *centroids;
    double *max_shifts;
} LocalParBenchArg;

static void* local_par_bench_worker(void *arg) {
    LocalParBenchArg *a = (LocalParBenchArg *)arg;
    AlignedAccumulator *my_acc = &a->accs[a->tid];
    size_t d = my_acc->d;

    for (int iter = 0; iter < a->iterations; iter++) {
        for (size_t p = a->start_idx; p < a->end_idx; p++) {
            int c = a->assignments[p];
            const double *pt = &a->points[p * d];
            local_accumulate_point(my_acc, c, pt);
        }

        /* Barrier before reduction */
        kmeans_barrier_wait(a->barrier);

        /* All threads participate in parallel cluster reduction */
        parallel_reduction_barrier(
            a->accs,
            a->nthreads,
            a->tid,
            a->barrier,
            a->centroids,
            a->max_shifts
        );

        /* Reset accumulator for next iteration */
        size_t sums_size = my_acc->k * my_acc->d * sizeof(double);
        if (my_acc->sums) memset(my_acc->sums, 0, sums_size);
        size_t counts_size = my_acc->k * sizeof(size_t);
        if (my_acc->counts) memset(my_acc->counts, 0, counts_size);

        kmeans_barrier_wait(a->barrier);
    }
    return NULL;
}

static double run_local_par_bench(
    int num_threads,
    const double *points,
    const int *assignments
) {
    AlignedAccumulator *accs = accumulators_create(num_threads, K_CLUSTERS, DIM);
    KMeansBarrier barrier;
    kmeans_barrier_init(&barrier, num_threads);
    double *centroids = (double *)calloc(K_CLUSTERS * DIM, sizeof(double));
    double *max_shifts = (double *)calloc(num_threads, sizeof(double));

    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    LocalParBenchArg *args = (LocalParBenchArg *)malloc(num_threads * sizeof(LocalParBenchArg));
    size_t chunk = N_POINTS / num_threads;

    uint64_t t0 = get_time_ns_sync();

    for (int t = 0; t < num_threads; t++) {
        args[t].accs = accs;
        args[t].points = points;
        args[t].assignments = assignments;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t == num_threads - 1) ? N_POINTS : (t + 1) * chunk;
        args[t].tid = t;
        args[t].iterations = ITERATIONS;
        args[t].nthreads = num_threads;
        args[t].barrier = &barrier;
        args[t].centroids = centroids;
        args[t].max_shifts = max_shifts;
        pthread_create(&threads[t], NULL, local_par_bench_worker, &args[t]);
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }

    uint64_t elapsed_ns = get_time_ns_sync() - t0;

    kmeans_barrier_destroy(&barrier);
    accumulators_destroy(accs, num_threads);
    free(centroids);
    free(max_shifts);
    free(threads);
    free(args);

    return (double)elapsed_ns / 1e6;
}

int main(void) {
    printf("======================================================================\n");
    printf(" BENCHMARK: Lock Contention Elimination & Parallel Reduction Scaling\n");
    printf(" Workload: %d Points | %d Clusters | %d Dimensions | %d Iterations\n",
           N_POINTS, K_CLUSTERS, DIM, ITERATIONS);
    printf("======================================================================\n\n");

    /* Allocate synthetic dataset */
    double *points = (double *)malloc(N_POINTS * DIM * sizeof(double));
    int *assignments = (int *)malloc(N_POINTS * sizeof(int));
    for (size_t i = 0; i < N_POINTS * DIM; i++) {
        points[i] = (double)(i % 100) * 0.05;
    }
    for (size_t i = 0; i < N_POINTS; i++) {
        assignments[i] = (int)(i % K_CLUSTERS);
    }

    printf("%-8s | %-16s | %-16s | %-16s | %-10s\n",
           "Threads", "Global Mutex(ms)", "Local+Seq Red(ms)", "Local+Par Red(ms)", "Lock Speedup");
    printf("---------+------------------+------------------+------------------+-----------\n");

    int thread_counts[] = {1, 2, 4, 8};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int idx = 0; idx < num_tests; idx++) {
        int t = thread_counts[idx];

        double t_locked = run_locked_contention_bench(t, points, assignments);
        double t_local_seq = run_local_seq_bench(t, points, assignments);
        double t_local_par = run_local_par_bench(t, points, assignments);

        double speedup = (t_local_par > 0.0) ? (t_locked / t_local_par) : 1.0;

        printf("   %-5d | %16.2f | %16.2f | %16.2f | %8.2fx\n",
               t, t_locked, t_local_seq, t_local_par, speedup);
    }

    printf("======================================================================\n\n");

    free(points);
    free(assignments);
    return 0;
}
