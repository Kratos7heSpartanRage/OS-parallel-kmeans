/**
 * @file test_synchronization.c
 * @brief Unit tests for Person 3: Synchronization & Local Accumulation.
 */

#define _POSIX_C_SOURCE 200112L

#include "synchronization.h"
#include "accumulator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define EPSILON 1e-9

/* ========================================================================= */
/* TEST 1: Phase Barrier Synchronization Correctness                         */
/* ========================================================================= */

typedef struct {
    KMeansBarrier *barrier;
    int nthreads;
    int tid;
    int iterations;
    volatile int *step_counter;
    int *serial_leader_count;
} BarrierTestArg;

static void* barrier_worker(void *arg) {
    BarrierTestArg *a = (BarrierTestArg *)arg;

    for (int iter = 0; iter < a->iterations; iter++) {
        /* Phase 1 */
        __sync_fetch_and_add(&a->step_counter[iter * 2], 1);

        int rc = kmeans_barrier_wait(a->barrier);
        if (rc == 1) {
            a->serial_leader_count[iter]++;
        }

        /* At this point, all threads must have reached phase 1 */
        assert(a->step_counter[iter * 2] == a->nthreads);

        /* Phase 2 */
        __sync_fetch_and_add(&a->step_counter[iter * 2 + 1], 1);

        rc = kmeans_barrier_wait(a->barrier);
        if (rc == 1) {
            a->serial_leader_count[iter]++;
        }

        /* At this point, all threads must have completed phase 2 */
        assert(a->step_counter[iter * 2 + 1] == a->nthreads);
    }

    return NULL;
}

static void test_phase_barrier(int nthreads, int iterations) {
    printf("[TEST 1] Testing Phase Barrier Synchronization (threads=%d, iters=%d)... ", nthreads, iterations);

    KMeansBarrier barrier;
    assert(kmeans_barrier_init(&barrier, nthreads) == 0);

    pthread_t *threads = (pthread_t *)malloc(nthreads * sizeof(pthread_t));
    BarrierTestArg *args = (BarrierTestArg *)malloc(nthreads * sizeof(BarrierTestArg));
    int *step_counter = (int *)calloc(iterations * 2, sizeof(int));
    int *serial_leader_count = (int *)calloc(iterations, sizeof(int));

    for (int t = 0; t < nthreads; t++) {
        args[t].barrier = &barrier;
        args[t].nthreads = nthreads;
        args[t].tid = t;
        args[t].iterations = iterations;
        args[t].step_counter = step_counter;
        args[t].serial_leader_count = serial_leader_count;
        pthread_create(&threads[t], NULL, barrier_worker, &args[t]);
    }

    for (int t = 0; t < nthreads; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Verify barrier integrity */
    for (int i = 0; i < iterations; i++) {
        assert(step_counter[i * 2] == nthreads);
        assert(step_counter[i * 2 + 1] == nthreads);
        /* Exactly one leader per wait call */
        assert(serial_leader_count[i] == 2);
    }

    assert(kmeans_barrier_destroy(&barrier) == 0);

    free(threads);
    free(args);
    free(step_counter);
    free(serial_leader_count);

    printf("PASSED\n");
}

/* ========================================================================= */
/* TEST 2: Local Accumulator vs. Sequential Reduction Equivalence            */
/* ========================================================================= */

typedef struct {
    AlignedAccumulator *accs;
    const double *points;
    const int *assignments;
    size_t start_idx;
    size_t end_idx;
    int tid;
} AccumulatorTestArg;

static void* local_acc_worker(void *arg) {
    AccumulatorTestArg *a = (AccumulatorTestArg *)arg;
    AlignedAccumulator *my_acc = &a->accs[a->tid];
    size_t d = my_acc->d;

    for (size_t p = a->start_idx; p < a->end_idx; p++) {
        int cluster = a->assignments[p];
        const double *pt = &a->points[p * d];
        local_accumulate_point(my_acc, cluster, pt);
    }
    return NULL;
}

static void test_local_accumulation_and_reduction(void) {
    printf("[TEST 2] Testing Local Accumulation & Numerical Equivalence... ");

    const size_t N = 10000;
    const size_t K = 8;
    const size_t D = 4;
    const int P = 4;

    /* Generate synthetic test dataset */
    double *points = (double *)malloc(N * D * sizeof(double));
    int *assignments = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N * D; i++) {
        points[i] = (double)(i % 100) * 0.1;
    }
    for (size_t i = 0; i < N; i++) {
        assignments[i] = (int)(i % K);
    }

    /* Reference sequential centroids calculation */
    double *ref_centroids = (double *)calloc(K * D, sizeof(double));
    size_t *ref_counts = (size_t *)calloc(K, sizeof(size_t));
    for (size_t i = 0; i < N; i++) {
        int c = assignments[i];
        ref_counts[c]++;
        for (size_t d = 0; d < D; d++) {
            ref_centroids[c * D + d] += points[i * D + d];
        }
    }
    for (size_t c = 0; c < K; c++) {
        if (ref_counts[c] > 0) {
            for (size_t d = 0; d < D; d++) {
                ref_centroids[c * D + d] /= (double)ref_counts[c];
            }
        }
    }

    /* Parallel accumulation using AlignedAccumulator */
    AlignedAccumulator *accs = accumulators_create(P, K, D);
    pthread_t threads[P];
    AccumulatorTestArg args[P];
    size_t chunk = N / P;

    for (int t = 0; t < P; t++) {
        args[t].accs = accs;
        args[t].points = points;
        args[t].assignments = assignments;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t == P - 1) ? N : (t + 1) * chunk;
        args[t].tid = t;
        pthread_create(&threads[t], NULL, local_acc_worker, &args[t]);
    }
    for (int t = 0; t < P; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Test Sequential Reduction */
    double *test_centroids = (double *)calloc(K * D, sizeof(double));
    accumulators_reduce(accs, P, test_centroids);

    for (size_t i = 0; i < K * D; i++) {
        assert(fabs(test_centroids[i] - ref_centroids[i]) < EPSILON);
    }

    accumulators_destroy(accs, P);
    free(points);
    free(assignments);
    free(ref_centroids);
    free(ref_counts);
    free(test_centroids);

    printf("PASSED\n");
}

/* ========================================================================= */
/* TEST 3: Parallel Partitioned Reduction Verification                       */
/* ========================================================================= */

typedef struct {
    AlignedAccumulator *accs;
    int nthreads;
    int tid;
    KMeansBarrier *barrier;
    double *centroids;
    double *max_shifts;
    double *returned_shift;
} ParRedTestArg;

static void* par_reduction_worker(void *arg) {
    ParRedTestArg *a = (ParRedTestArg *)arg;
    *a->returned_shift = parallel_reduction_barrier(
        a->accs,
        a->nthreads,
        a->tid,
        a->barrier,
        a->centroids,
        a->max_shifts
    );
    return NULL;
}

static void test_parallel_reduction(void) {
    printf("[TEST 3] Testing Parallel Cluster-Partitioned Reduction... ");

    const size_t N = 50000;
    const size_t K = 16;
    const size_t D = 8;
    const int P = 4;

    double *points = (double *)malloc(N * D * sizeof(double));
    int *assignments = (int *)malloc(N * sizeof(int));
    for (size_t i = 0; i < N * D; i++) {
        points[i] = (double)(i % 500) * 0.05;
    }
    for (size_t i = 0; i < N; i++) {
        assignments[i] = (int)(i % K);
    }

    AlignedAccumulator *accs = accumulators_create(P, K, D);
    pthread_t threads[P];
    AccumulatorTestArg args[P];
    size_t chunk = N / P;

    for (int t = 0; t < P; t++) {
        args[t].accs = accs;
        args[t].points = points;
        args[t].assignments = assignments;
        args[t].start_idx = t * chunk;
        args[t].end_idx = (t == P - 1) ? N : (t + 1) * chunk;
        args[t].tid = t;
        pthread_create(&threads[t], NULL, local_acc_worker, &args[t]);
    }
    for (int t = 0; t < P; t++) {
        pthread_join(threads[t], NULL);
    }

    /* Compute baseline with sequential reduce */
    double *seq_centroids = (double *)calloc(K * D, sizeof(double));
    double seq_shift = accumulators_reduce(accs, P, seq_centroids);

    /* Compute with parallel barrier reduction */
    double *par_centroids = (double *)calloc(K * D, sizeof(double));
    double max_shifts[P];
    double returned_shifts[P];
    KMeansBarrier barrier;
    assert(kmeans_barrier_init(&barrier, P) == 0);

    ParRedTestArg rargs[P];
    for (int t = 0; t < P; t++) {
        rargs[t].accs = accs;
        rargs[t].nthreads = P;
        rargs[t].tid = t;
        rargs[t].barrier = &barrier;
        rargs[t].centroids = par_centroids;
        rargs[t].max_shifts = max_shifts;
        rargs[t].returned_shift = &returned_shifts[t];
        pthread_create(&threads[t], NULL, par_reduction_worker, &rargs[t]);
    }
    for (int t = 0; t < P; t++) {
        pthread_join(threads[t], NULL);
    }

    /* All threads must agree on the same global max shift */
    for (int t = 0; t < P; t++) {
        assert(fabs(returned_shifts[t] - seq_shift) < EPSILON);
    }

    /* Check centroid coordinates match */
    for (size_t i = 0; i < K * D; i++) {
        assert(fabs(par_centroids[i] - seq_centroids[i]) < EPSILON);
    }

    kmeans_barrier_destroy(&barrier);
    accumulators_destroy(accs, P);
    free(points);
    free(assignments);
    free(seq_centroids);
    free(par_centroids);

    printf("PASSED\n");
}

/* ========================================================================= */
/* TEST 4: Global Locked Accumulator Correctness                             */
/* ========================================================================= */

static void test_global_locked_accumulator(void) {
    printf("[TEST 4] Testing Global Locked Accumulator Primitive... ");

    const size_t K = 4;
    const size_t D = 2;
    GlobalLockedAccumulator *gacc = global_locked_accumulator_create(K, D);
    assert(gacc != NULL);

    double pt[2] = {1.5, 2.5};
    global_locked_accumulate_point(gacc, 0, pt);
    global_locked_accumulate_point(gacc, 0, pt);

    double centroids[8] = {0};
    double shift = global_locked_accumulator_finish(gacc, centroids);

    assert(fabs(centroids[0] - 1.5) < EPSILON);
    assert(fabs(centroids[1] - 2.5) < EPSILON);
    assert(shift > 0.0);

    global_locked_accumulator_reset(gacc);
    assert(gacc->counts[0] == 0);

    global_locked_accumulator_destroy(gacc);
    printf("PASSED\n");
}

int main(void) {
    printf("====================================================\n");
    printf(" Running Person 3 Tests: Synchronization & Reduction\n");
    printf("====================================================\n");

    test_phase_barrier(2, 50);
    test_phase_barrier(4, 50);
    test_phase_barrier(8, 50);
    test_local_accumulation_and_reduction();
    test_parallel_reduction();
    test_global_locked_accumulator();

    printf("\n>>> ALL PERSON 3 TESTS PASSED SUCCESSFULLY! <<<\n\n");
    return 0;
}
