#define _DEFAULT_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include "accumulator.h"
#include "aligned_memory.h"

#define NUM_ITERATIONS 50000000
#define NUM_THREADS_MAX 16

typedef struct {
    double *sums;
    size_t *counts;
    size_t k;
    size_t d;
} UnpaddedAccumulator;

typedef struct {
    int thread_id;
    int use_padding;
    AlignedAccumulator *padded_accs;
    UnpaddedAccumulator *unpadded_accs;
} WorkerArgs;

void* worker_thread(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    int tid = args->thread_id;
    
    if (args->use_padding) {
        AlignedAccumulator *acc = &args->padded_accs[tid];
        size_t K = acc->k;
        size_t D = acc->d;
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            size_t cluster = iter % K;
            for (size_t d = 0; d < D; d++) {
                acc->sums[cluster * D + d] += 1.0;
            }
            acc->counts[cluster] += 1;
        }
    } else {
        UnpaddedAccumulator *acc = &args->unpadded_accs[tid];
        size_t K = acc->k;
        size_t D = acc->d;
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            size_t cluster = iter % K;
            for (size_t d = 0; d < D; d++) {
                acc->sums[cluster * D + d] += 1.0;
            }
            acc->counts[cluster] += 1;
        }
    }
    return NULL;
}

double run_benchmark(int num_threads, size_t K, size_t D, int use_padding) {
    pthread_t threads[NUM_THREADS_MAX];
    WorkerArgs args[NUM_THREADS_MAX];

    AlignedAccumulator *padded_accs = NULL;
    UnpaddedAccumulator *unpadded_accs = NULL;
    double *unpadded_sums_buf = NULL;
    size_t *unpadded_counts_buf = NULL;
    
    if (use_padding) {
        padded_accs = accumulators_create(num_threads, K, D);
    } else {
        unpadded_accs = (UnpaddedAccumulator*)malloc(num_threads * sizeof(UnpaddedAccumulator));
        unpadded_sums_buf = (double*)calloc(num_threads * K * D, sizeof(double));
        unpadded_counts_buf = (size_t*)calloc(num_threads * K, sizeof(size_t));
        for (int i = 0; i < num_threads; i++) {
            unpadded_accs[i].k = K;
            unpadded_accs[i].d = D;
            unpadded_accs[i].sums = &unpadded_sums_buf[i * K * D];
            unpadded_accs[i].counts = &unpadded_counts_buf[i * K];
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        args[i].use_padding = use_padding;
        args[i].padded_accs = padded_accs;
        args[i].unpadded_accs = unpadded_accs;
        pthread_create(&threads[i], NULL, worker_thread, &args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    if (use_padding) {
        accumulators_destroy(padded_accs, num_threads);
    } else {
        free(unpadded_sums_buf);
        free(unpadded_counts_buf);
        free(unpadded_accs);
    }

    return time_taken;
}

int main(void) {
    size_t K = 4;
    size_t D = 3;
    
    printf("======================================================================\n");
    printf(" False Sharing Benchmark (Simulating K-Means accumulation)\n");
    printf(" Iterations per thread: %d\n", NUM_ITERATIONS);
    printf(" Workload: K = %zu, D = %zu\n", K, D);
    printf("======================================================================\n\n");
    
    printf("%-10s %-20s %-20s %-15s\n", "Threads", "Time Unpadded (s)", "Time Padded (s)", "Speedup");
    printf("----------------------------------------------------------------------\n");

    for (int t = 1; t <= 8; t *= 2) {
        double time_unpadded = run_benchmark(t, K, D, 0);
        double time_padded = run_benchmark(t, K, D, 1);
        double speedup = time_unpadded / time_padded;
        printf("%-10d %-20.4f %-20.4f %-15.2fx\n", t, time_unpadded, time_padded, speedup);
    }

    return 0;
}
