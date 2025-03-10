
/*calculateDistancesKernel:

    Computes the Euclidean distance between each data point and all centroids.
    Assigns each point to the nearest centroid based on the smallest distance.
    Updates d_classMap with the assigned class.

updateCentroidsKernel:

    Updates the accumulated sum of points for each centroid.
    Each thread adds the point's features to the corresponding centroid's sum.
    Uses atomicAdd to handle concurrent updates safely.

finalizeCentroidsKernel:

    Averages the accumulated points for each centroid to compute the new centroid coordinates.
    Updates the centroids in d_centroids.*/



// Kernel for calculating the distance from each point to the centroids
__global__ void calculateDistancesKernel(float* d_data, float* d_centroids, int* d_classMap, int lines, int samples, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lines) {
        float minDist = FLT_MAX;
        int class = -1;

        for (int j = 0; j < K; j++) {
            float dist = 0;
            for (int k = 0; k < samples; k++) {
                float diff = d_data[i * samples + k] - d_centroids[j * samples + k];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            if (dist < minDist) {
                minDist = dist;
                class = j;
            }
        }

        d_classMap[i] = class;
    }
}

// Kernel for updating centroids
__global__ void updateCentroidsKernel(float* d_data, int* d_classMap, float* d_auxCentroids, int* d_pointsPerClass, int lines, int samples, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lines) {
        int class = d_classMap[i];
        atomicAdd(&d_pointsPerClass[class], 1);

        for (int j = 0; j < samples; j++) {
            atomicAdd(&d_auxCentroids[class * samples + j], d_data[i * samples + j]);
        }
    }
}

// Kernel for finalizing centroids (averaging the sums)
__global__ void finalizeCentroidsKernel(float* d_auxCentroids, int* d_pointsPerClass, float* d_centroids, int K, int samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) {
        int count = d_pointsPerClass[i];
        for (int j = 0; j < samples; j++) {
            if (count > 0) {
                d_centroids[i * samples + j] = d_auxCentroids[i * samples + j] / count;
            }
        }
    }
}