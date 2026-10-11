#include <stdio.h>
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
    double par_centers[] = {1.0, 1.0, 9.0, 9.0};

    Dataset dataset = {points, 6, 2};
    Centroids sequential = {seq_centers, 2, 2};
    Centroids parallel = {par_centers, 2, 2};
    KMeansConfig config = {100, 0.0001};

    int a = kmeans_sequential(&dataset, &sequential, &config);
    int b = kmeans_static(&dataset, &parallel, &config, 2);

    if (a != 0 || b != 0) {
        printf("TEST FAILED\n");
        return 1;
    }

    printf("Sequential centroids:\n");
    for (int i = 0; i < 4; i++)
        printf("%.4f%c", seq_centers[i], i == 3 ? '\n' : ' ');

    printf("Static parallel centroids:\n");
    for (int i = 0; i < 4; i++)
        printf("%.4f%c", par_centers[i], i == 3 ? '\n' : ' ');

    for (int i = 0; i < 4; i++) {
        double diff = seq_centers[i] - par_centers[i];
        if (diff < -0.000001 || diff > 0.000001) {
            printf("TEST FAILED: centroid mismatch\n");
            return 1;
        }
    }

    printf("TEST PASSED: centroids match\n");
    return 0;
}
