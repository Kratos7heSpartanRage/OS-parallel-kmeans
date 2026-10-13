#include <stdio.h>
#include <math.h>
#include "include/kmeans.h"

int main(void)
{
    double points[] = {
        1.0, 1.0,
        1.2, 1.1,
        0.8, 0.9,
        9.0, 9.0,
        8.8, 9.1,
        9.2, 8.9
    };

    double seq_centers[] = {1.0, 1.0, 9.0, 9.0};
    double naive_centers[] = {1.0, 1.0, 9.0, 9.0};

    Dataset dataset = {points, 6, 2};
    Centroids sequential = {seq_centers, 2, 2};
    Centroids naive = {naive_centers, 2, 2};
    KMeansConfig config = {100, 0.0001};

    int a = kmeans_sequential(&dataset, &sequential, &config);
    int b = kmeans_naive(&dataset, &naive, &config, 2);

    if (a != 0 || b != 0) {
        printf("TEST FAILED: execution error\n");
        return 1;
    }

    printf("Sequential: %.4f %.4f %.4f %.4f\n",
           seq_centers[0], seq_centers[1],
           seq_centers[2], seq_centers[3]);

    printf("Naive MT:   %.4f %.4f %.4f %.4f\n",
           naive_centers[0], naive_centers[1],
           naive_centers[2], naive_centers[3]);

    for (int i = 0; i < 4; i++) {
        if (fabs(seq_centers[i] - naive_centers[i]) > 1e-6) {
            printf("TEST FAILED: centroid mismatch\n");
            return 1;
        }
    }

    printf("TEST PASSED: centroids match\n");
    return 0;
}
