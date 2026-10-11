#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/kmeans.h"

#define DIMENSIONS 2
#define CLUSTERS 3
#define REPEATS 3
#define NAIVE_MAX_POINTS 1000000

static const size_t dataset_sizes[] = {
    10000, 100000, 500000, 1000000, 5000000
};

static const size_t thread_counts[] = {1, 2, 4, 8, 16};

static double seconds(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1000000000.0;
}

static void initialize_centers(
    double *centers,
    const double *points
)
{
    for (size_t i = 0; i < CLUSTERS * DIMENSIONS; i++) {
        centers[i] = points[i];
    }
}

static int same_centers(
    const double *a,
    const double *b
)
{
    for (size_t i = 0; i < CLUSTERS * DIMENSIONS; i++) {
        if (fabs(a[i] - b[i]) > 1e-6) {
            return 0;
        }
    }

    return 1;
}

static void free_arrays(
    double *points,
    double *initial,
    double *seq,
    double *stat,
    double *naive
)
{
    free(points);
    free(initial);
    free(seq);
    free(stat);
    free(naive);
}

int main(void)
{
    FILE *csv = fopen(
        "benchmarks/results/comparison_full.csv", "w"
    );

    if (!csv) {
        perror("Cannot open results CSV");
        return EXIT_FAILURE;
    }

    fprintf(csv,
        "dataset_size,threads,repeat,"
        "sequential_seconds,static_seconds,naive_seconds,"
        "static_speedup,naive_speedup,static_match,naive_match\n"
    );

    srand(42);

    size_t num_sizes =
        sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);

    size_t num_thread_counts =
        sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = dataset_sizes[si];

        printf("\n===== Dataset: %zu points =====\n", n);
        fflush(stdout);

        double *points =
            malloc(n * DIMENSIONS * sizeof(double));

        double *initial =
            malloc(CLUSTERS * DIMENSIONS * sizeof(double));

        double *seq =
            malloc(CLUSTERS * DIMENSIONS * sizeof(double));

        double *stat =
            malloc(CLUSTERS * DIMENSIONS * sizeof(double));

        double *naive =
            malloc(CLUSTERS * DIMENSIONS * sizeof(double));

        if (!points || !initial || !seq || !stat || !naive) {
            fprintf(stderr,
                "Memory allocation failed for %zu points\n", n);

            free_arrays(points, initial, seq, stat, naive);
            fclose(csv);
            return EXIT_FAILURE;
        }

        for (size_t i = 0; i < n * DIMENSIONS; i++) {
            points[i] = (double)(rand() % 10000) / 100.0;
        }

        initialize_centers(initial, points);

        Dataset dataset = {points, n, DIMENSIONS};
        KMeansConfig config = {100, 0.0001};

        for (size_t ti = 0; ti < num_thread_counts; ti++) {
            size_t threads = thread_counts[ti];
            int run_naive = (n <= NAIVE_MAX_POINTS);

            for (int rep = 1; rep <= REPEATS; rep++) {
                initialize_centers(seq, initial);
                initialize_centers(stat, initial);
                initialize_centers(naive, initial);

                Centroids cs = {seq, CLUSTERS, DIMENSIONS};
                Centroids cp = {stat, CLUSTERS, DIMENSIONS};
                Centroids cn = {naive, CLUSTERS, DIMENSIONS};

                /* Sequential baseline */
                double start = seconds();

                int rs = kmeans_sequential(
                    &dataset, &cs, &config
                );

                double ts = seconds() - start;

                /* Static multithreaded version */
                start = seconds();

                int rp = kmeans_static(
                    &dataset, &cp, &config, threads
                );

                double tp = seconds() - start;

                /* Naive shared-mutex version, where enabled */
                int rn = 0;
                double tn = 0.0;

                if (run_naive) {
                    start = seconds();

                    rn = kmeans_naive(
                        &dataset, &cn, &config, threads
                    );

                    tn = seconds() - start;
                }

                if (rs != 0 || rp != 0 ||
                    (run_naive && rn != 0)) {
                    fprintf(stderr,
                        "K-Means execution failed: "
                        "N=%zu threads=%zu repeat=%d\n",
                        n, threads, rep);

                    free_arrays(
                        points, initial, seq, stat, naive
                    );
                    fclose(csv);
                    return EXIT_FAILURE;
                }

                int sm = same_centers(seq, stat);
                int nm = run_naive
                    ? same_centers(seq, naive)
                    : -1;

                double static_speedup =
                    tp > 0.0 ? ts / tp : 0.0;

                double naive_speedup =
                    run_naive && tn > 0.0 ? ts / tn : 0.0;

                /* Save this experiment immediately. */
                if (run_naive) {
                    fprintf(csv,
                        "%zu,%zu,%d,%.9f,%.9f,%.9f,"
                        "%.6f,%.6f,%d,%d\n",
                        n, threads, rep, ts, tp, tn,
                        static_speedup, naive_speedup,
                        sm, nm);
                } else {
                    fprintf(csv,
                        "%zu,%zu,%d,%.9f,%.9f,,"
                        "%.6f,,%d,-1\n",
                        n, threads, rep, ts, tp,
                        static_speedup, sm);
                }

                fflush(csv);

                /* Display progress immediately. */
                printf(
                    "N=%zu threads=%zu rep=%d "
                    "seq=%.4fs static=%.4fs ",
                    n, threads, rep, ts, tp
                );

                if (run_naive) {
                    printf(
                        "naive=%.4fs static_speedup=%.3fx "
                        "naive_speedup=%.3fx match=%s\n",
                        tn, static_speedup, naive_speedup,
                        (sm && nm) ? "YES" : "NO"
                    );
                } else {
                    printf(
                        "naive=SKIPPED static_speedup=%.3fx "
                        "static_match=%s\n",
                        static_speedup, sm ? "YES" : "NO"
                    );
                }

                fflush(stdout);

                if (!sm || (run_naive && !nm)) {
                    fprintf(stderr,
                        "Centroid mismatch detected: "
                        "N=%zu threads=%zu repeat=%d\n",
                        n, threads, rep);

                    free_arrays(
                        points, initial, seq, stat, naive
                    );
                    fclose(csv);
                    return EXIT_FAILURE;
                }
            }
        }

        free_arrays(points, initial, seq, stat, naive);
    }

    fclose(csv);

    printf(
        "\nAll experiments completed.\n"
        "Results saved to "
        "benchmarks/results/comparison_full.csv\n"
    );

    return EXIT_SUCCESS;
}

