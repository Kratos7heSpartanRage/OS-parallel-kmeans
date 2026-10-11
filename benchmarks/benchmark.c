#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/kmeans.h"

int kmeans_static(
    const Dataset *, Centroids *,
    const KMeansConfig *, size_t
);

#define DIMENSIONS 2
#define CLUSTERS 3
#define REPEATS 3

static const size_t dataset_sizes[] = {
    10000, 100000, 500000, 1000000
};

static const size_t thread_counts[] = {1, 2, 4, 8};

static double seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec +
           (double)ts.tv_nsec / 1e9;
}

static void initialize_centers(
    double *centers, const double *points, size_t k
)
{
    for (size_t i = 0; i < k * DIMENSIONS; i++)
        centers[i] = points[i];
}

static int same_centers(
    const double *a, const double *b, size_t count
)
{
    for (size_t i = 0; i < count; i++) {
        if (fabs(a[i] - b[i]) > 1e-6)
            return 0;
    }
    return 1;
}

int main(void)
{
    FILE *csv = fopen("benchmarks/results/results.csv", "w");

    if (!csv) {
        perror("Cannot open results CSV");
        return 1;
    }

    fprintf(csv,
        "dataset_size,threads,repeat,sequential_seconds,"
        "static_seconds,speedup,centroids_match\n");

    srand(42);

    size_t size_count =
        sizeof(dataset_sizes) / sizeof(dataset_sizes[0]);

    size_t thread_count =
        sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (size_t s = 0; s < size_count; s++) {
        size_t n = dataset_sizes[s];

        double *points = malloc(n * DIMENSIONS * sizeof(double));
        double *initial = malloc(CLUSTERS * DIMENSIONS * sizeof(double));
        double *seq_centers = malloc(CLUSTERS * DIMENSIONS * sizeof(double));
        double *par_centers = malloc(CLUSTERS * DIMENSIONS * sizeof(double));

        if (!points || !initial || !seq_centers || !par_centers) {
            fprintf(stderr, "Allocation failed for %zu points\n", n);
            free(points);
            free(initial);
            free(seq_centers);
            free(par_centers);
            fclose(csv);
            return 1;
        }

        for (size_t i = 0; i < n * DIMENSIONS; i++)
            points[i] = (double)(rand() % 10000) / 100.0;

        initialize_centers(initial, points, CLUSTERS);

        Dataset dataset = {points, n, DIMENSIONS};
        KMeansConfig config = {100, 0.0001};

        for (size_t t = 0; t < thread_count; t++) {
            size_t threads = thread_counts[t];

            for (int repeat = 1; repeat <= REPEATS; repeat++) {
                initialize_centers(seq_centers, initial, CLUSTERS);
                initialize_centers(par_centers, initial, CLUSTERS);

                Centroids sequential = {
                    seq_centers, CLUSTERS, DIMENSIONS
                };
                Centroids parallel = {
                    par_centers, CLUSTERS, DIMENSIONS
                };

                double start = seconds();
                int seq_ok = kmeans_sequential(
                    &dataset, &sequential, &config
                );
                double seq_time = seconds() - start;

                start = seconds();
                int par_ok = kmeans_static(
                    &dataset, &parallel, &config, threads
                );
                double par_time = seconds() - start;

                if (seq_ok != 0 || par_ok != 0) {
                    fprintf(stderr, "K-Means failed\n");
                    free(points);
                    free(initial);
                    free(seq_centers);
                    free(par_centers);
                    fclose(csv);
                    return 1;
                }

                int match = same_centers(
                    seq_centers, par_centers,
                    CLUSTERS * DIMENSIONS
                );

                double speedup = par_time > 0.0
                    ? seq_time / par_time : 0.0;

                printf(
                    "N=%zu threads=%zu repeat=%d "
                    "seq=%.6f static=%.6f speedup=%.3fx match=%s\n",
                    n, threads, repeat, seq_time, par_time,
                    speedup, match ? "YES" : "NO"
                );

                fprintf(csv, "%zu,%zu,%d,%.9f,%.9f,%.6f,%d\n",
                    n, threads, repeat, seq_time, par_time,
                    speedup, match);

                fflush(csv);

                if (!match) {
                    fprintf(stderr,
                        "Centroid mismatch: N=%zu threads=%zu\n",
                        n, threads);
                    free(points);
                    free(initial);
                    free(seq_centers);
                    free(par_centers);
                    fclose(csv);
                    return 1;
                }
            }
        }

        free(points);
        free(initial);
        free(seq_centers);
        free(par_centers);
    }

    fclose(csv);
    printf("\nResults saved to benchmarks/results/results.csv\n");

    return 0;
}

