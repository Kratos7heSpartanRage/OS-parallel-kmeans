#define _DEFAULT_SOURCE 1
/**
 * @file benchmark_scheduler.c
 * @brief Benchmark suite demonstrating Dynamic Scheduling vs. Stragglers & Chunk Tuning.
 *
 * Inspired by Laccetti et al. (JPDC 2020), this benchmark simulates workload
 * heterogeneity and system jitter to demonstrate why dynamic work-pulling
 * outperforms static chunk distribution.
 */

#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static void sleep_us(int us) {
    if (us >= 1000) Sleep(us / 1000);
    else {
        // Busy spin for microsecond granularity on Windows
        LARGE_INTEGER freq, start, cur;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&start);
        int64_t target = (freq.QuadPart * us) / 1000000;
        do {
            QueryPerformanceCounter(&cur);
        } while (cur.QuadPart - start.QuadPart < target);
    }
}
#else
#include <unistd.h>
static void sleep_us(int us) { usleep(us); }
#endif

static uint64_t get_time_ns_bench(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ULL) + (uint64_t)ts.tv_nsec;
}

/* Simulation task context */
typedef struct {
    int straggler_task_id; /**< Task that encounters artificial jitter */
    int jitter_duration_us; /**< Duration of simulated delay */
} BenchCtx;

static void simulate_kmeans_chunk_fn(const Task *task, int tid, void *ctx) {
    (void)tid;
    BenchCtx *bctx = (BenchCtx *)ctx;

    // Simulate compute for chunk
    size_t pts = task->end_idx - task->start_idx;
    volatile double dummy = 1.0;
    for (size_t i = 0; i < pts * 50; i++) {
        dummy += (double)i * 0.0001;
    }
    (void)dummy;

    // Simulate straggler/system jitter on designated task
    if (bctx && task->task_id == bctx->straggler_task_id) {
        sleep_us(bctx->jitter_duration_us);
    }
}

static void run_scheduling_comparison(int num_threads, size_t total_points) {
    printf("\n======================================================================\n");
    printf(" BENCHMARK: Dynamic Scheduling vs. Straggler Resistance (P = %d)\n", num_threads);
    printf(" Dataset Points: %zu | Simulated Jitter Delay: 15 ms\n", total_points);
    printf("======================================================================\n");

    BenchCtx bctx = {
        .straggler_task_id = 0, // Task 0 experiences a delay
        .jitter_duration_us = 15000 // 15 ms delay
    };

    /* Scenario A: Static-like scheduling (Tasks = P) */
    {
        size_t chunk_size = total_points / num_threads;
        size_t num_tasks = num_threads;
        bctx.straggler_task_id = 0;

        ThreadPool *pool = pool_create(num_threads, num_tasks * 2, simulate_kmeans_chunk_fn);
        uint64_t t0 = get_time_ns_bench();

        for (size_t i = 0; i < num_tasks; i++) {
            Task t = {
                .start_idx = i * chunk_size,
                .end_idx = (i == num_tasks - 1) ? total_points : (i + 1) * chunk_size,
                .task_id = (int)i,
                .ctx = &bctx
            };
            pool_submit(pool, t);
        }
        pool_wait_iteration(pool);
        uint64_t elapsed_ns = get_time_ns_bench() - t0;

        printf("\n[SCENARIO A] Coarse Static Partitioning (Tasks = P = %d):\n", num_threads);
        printf(" -> Execution Time: %.2f ms\n", (double)elapsed_ns / 1e6);
        pool_print_stats(pool);
        pool_shutdown(pool);
    }

    /* Scenario B: Dynamic Fine-Grained Chunking (Tasks = 8 * P) */
    {
        size_t num_tasks = num_threads * 8;
        size_t chunk_size = total_points / num_tasks;
        bctx.straggler_task_id = 0;

        ThreadPool *pool = pool_create(num_threads, num_tasks * 2, simulate_kmeans_chunk_fn);
        uint64_t t0 = get_time_ns_bench();

        for (size_t i = 0; i < num_tasks; i++) {
            Task t = {
                .start_idx = i * chunk_size,
                .end_idx = (i == num_tasks - 1) ? total_points : (i + 1) * chunk_size,
                .task_id = (int)i,
                .ctx = &bctx
            };
            pool_submit(pool, t);
        }
        pool_wait_iteration(pool);
        uint64_t elapsed_ns = get_time_ns_bench() - t0;

        printf("\n[SCENARIO B] Dynamic Fine-Grained Partitioning (Tasks = 8*P = %zu):\n", num_tasks);
        printf(" -> Execution Time: %.2f ms\n", (double)elapsed_ns / 1e6);
        pool_print_stats(pool);
        pool_shutdown(pool);
    }
}

int main(void) {
    run_scheduling_comparison(4, 200000);
    return 0;
}
