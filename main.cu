#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>

// Arrays-of-Struct to Struct-of-Arrays data layout for better memory coalescing
typedef struct {
    float* timestamps;
    float* consumption;
    float* temperature;
    int size;
} EnergyDataset;

// Weather features for ML modeling (would come from external source in production)
typedef struct {
    float* humidity;
    float* cloudCover;
    float* windSpeed;
    int size;
} WeatherFeatures;

// Output features for ML model (extracted features)
typedef struct {
    float* timeFeatures;
    float* tempFeatures;
    float* movingAvgs;
    float* hourlyPatterns;
    float* weatherFeatures;
    int size;
    int numFeatures;
} MLFeatures;

// Error checking helper
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel for optimized preprocessing
// OPTIMISATION: Uses shared memory and ensures coalesced memory access
__global__ void preprocessEnergyData(
    const float* timestamps,
    const float* temperatures,
    float* timeFeatures,
    float* tempFeatures,
    int dataSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dataSize) {
        // OPTIMISATION: direct, coalesced global memory access
        float timestamp = timestamps[idx];
        float temperature = temperatures[idx];

        // Extract time features
        // It also can be calculated by using sine/cosine encoding or One-Hot (whicl less effective for CUDA)
        // fmodf and floorf better applies for the CUDA instructions
        float hourOfDay = fmodf(timestamp, 24.0f) / 24.0f; // Normalization
        float dayOfWeek = fmodf(floorf(timestamp / 24.0f), 7.0f) / 6.0f;

        // Normalize temperature (-10C to 40C range)
        float normalizedTemp = (temperature + 10.0f) / 50.0f;

        // Store features with colasced writes
        timeFeatures[idx] = hourOfDay;
        timeFeatures[idx + dataSize] = dayOfWeek; // Second feature at offset
        timeFeatures[idx] = normalizedTemp;

        // Additional features for ml could be added here
      }
}

// CUDA kernel for calculating multiple moving averages with shared memory
// OPTIMISATION: Uses shared memory to reduce global memory reads
__global__ void calculateMovingAverages(
    const float* consumption,
    float* movingAvgs,
    int dataSize,
    int windowSize // Needs for time series analysis
) {
    extern __shared__ float sharedData[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    // Load data into shared memory with boundary checks
    // OPTIMISATION: Coalesced global memory reads
    if (idx < dataSize) {
        sharedData[localIdx] = consumption[idx];
    } else {
        sharedData[localIdx] = 0.0f;
    }

    // Make sure all thread loaded their data
    __syncthreads();

    // Boundary check for output calculation
    if (idx < dataSize) {
        float sum = 0.0f;
        int count = 0;
        int halfWindow = windowSize / 2; // centered value (from 0-12 to 12-24 range)

        // OPTIMISATION: Calculate indices into shared memory when possible
        // OPTIMISATION: Minimize divergence with structured loop bounds
        for (int i = -halfWindow; i <= halfWindow; i++) {
            int dataIdx = localIdx + i;

            // Check if we need to read from global memory (outside shared memory range)
            if (dataIdx < 0 || dataIdx >= blockDim.x) {
                int globalIdx = idx + i;
                // Boundary check for global memory
                if (globalIdx >= 0 && globalIdx < dataSize) {
                    sum += consumption[globalIdx];
                    count++;
                }
            } else {
                // Read from shared memory (much faster)
                sum += sharedData[dataIdx];
                count++;
            }
        }

        // Write result back to global memory
        if (count > 0) {
            movingAvgs[idx] = sum / count;
        } else {
            movingAvgs[idx] = consumption[idx];
        }
    }
}

// CUDA kernel for calculating hourly averages with partial reductions
// OPTIMISATION: Uses shared memory and per-block reductions to minimize atomic operations
__global__ void calculateHourlyAverages(
    const float* timestamps,
    const float* consumption,
    float* hourlyAverages,
    float* hourlyCounts,
    int dataSize
) {
    // Allocate shared memory fro local hourly sums and counts
    extern __shared__ float sharedData[];
    float* localSums = sharedData;
    float* localCounts = sharedData + 24;

    // Initialize shared memory
    int localIdx = threadIdx.x;
    if (localIdx < 24) {
        localSums[localIdx] = 0.0f;
        localCounts[localIdx] = 0.0f;
    }

    // Make sure shared memory is initialized
    __syncthreads();

    // Process data points assigned to this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        // Get hour from timestamp
        int hour = (int)floorf(fmodf(timestamps[idx], 24.0f));

        // Increment local sums and counts using atomic operations within shared memory
        // OPTIMISATION: Using atomicAdd to shared memory is much faster than global atomics
        atomicAdd(&localSums[hour], consumption[idx]);
        atomicAdd(&localCounts[hour], 1.0f);
    }

    // Make sure all threads have updated shared memory
    __syncthreads();

    // Reduce shared memory results to global memory
    // Only one thread per hour performs global atomics
    if (localIdx < 24) {
        // OPTIMISATION: Using only 24 atomic operations per block instead of one per thread
        atomicAdd(&hourlyAverages[localIdx], localSums[localIdx]);
        atomicAdd(&hourlyCounts[localIdx], localCounts[localIdx]);
    }
}

// CUDA kernel that extracts ML-ready features from energy data
// This prepares data for future machin learning model training
__global__ void extractMLFeatures(
    const float* timestamps,
    const float* consumption,
    const float* temperature,
    const float* weatherData,
    float* features,
    int dataSize,
    int numFeatures,
    float* movingAvgs,
    float* hourlyPatterns
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < dataSize) {
        // Extract time features
        float timestamp = timestamps[idx];
        int hour = (int)floorf(fmodf(timestamp, 24.0f));
        float dayOfWeek = fmodf(floorf(timestamp / 24.0f), 7.0f);
        float weekOfYear = fmodf(floorf(timestamp / (24.0f * 7.0f)), 52.0f);

        // Base output index for this data point
        int outIdx = idx * numFeatures;

        // Store time features with sine/cosine encoding for cyclical features
        // OPTIMISATION: Better feature encoding for ML models
        features[outIdx + 0] = sinf(2.0f * M_PI * hour / 24.0f);
        features[outIdx + 1] = cosf(2.0f * M_PI * hour / 24.0f);
        features[outIdx + 2] = sinf(2.0f * M_PI * dayOfWeek / 7.0f);
        features[outIdx + 3] = cosf(2.0f * M_PI * dayOfWeek / 7.0f);
        features[outIdx + 4] = sinf(2.0f * M_PI * weekOfYear / 52.0f);
        features[outIdx + 5] = cosf(2.0f * M_PI * weekOfYear / 52.0f);

        // Store temperature features - raw and normalised
        features[outIdx + 6] = temperature[idx];
        features[outIdx + 7] = (temperature[idx] + 10.0f) / 50.0f;

        // Store moving average
        features[outIdx + 8] = movingAvgs[idx];

        // Store hourly pattern for this hour
        features[outIdx + 9] = hourlyPatterns[hour];

        // Store weather features (if available)
        if (weatherData != nullptr) {
            features[outIdx + 10] = weatherData[idx]; // humidity
            features[outIdx + 11] = weatherData[idx + dataSize]; // cloud cover
        }

        // Store target variable (consumption)
        features[outIdx + numFeatures - 1] = consumption[idx];
    }
}

// Helper function to create synthetic energy data
void generateSyntheticEnergyData(
    float* timestamps,
    float* temperatures,
    float* consumption,
    int dataSize
) {
    for (int i = 0; i < dataSize; i++) {
        // Calculate day of year (0-364)
        int dayOfYear = i / 24;

        // Calculate hour of a day (0-23)
        int hourOfDay = i % 24;

        // Set timestamp (in hours)
        timestamps[i] = (float)i;

        // Simulate temperature variations (seasonal)
        // Warmer in summer (middle of year), cooler in winter
        float seasonalFactor = sinf((dayOfYear / 365.0f) * 2.0f * M_PI);
        temperatures[i] = 15.0f + 15.0f * seasonalFactor;

        // Simulate energy consumption patterns based on:
        // 1. Time of day (higher during morning and evening peaks)
        float hourlyFactor = 0.5f + 0.5f * sinf((hourOfDay - 10.0f) / 24.0f * 2.0f * M_PI);

        // 2. Season (higher in winter and summer for heating/cooling)
        float seasonalConsumptionFactor = 0.7f + 0.3f * fabsf(seasonalFactor);

        // 3. Random variations
        float randomFactor = 0.8f + 0.4f * ((float)rand() / RAND_MAX);

        // Combine factors to create realistic consumption pattern
        consumption[i] = 5.0f * hourlyFactor * seasonalConsumptionFactor * randomFactor;
    }
}

// Helper function to allocate host dataset
EnergyDataset allocateHostDataset(int dataSize) {
    EnergyDataset dataset;
    dataset.timestamps = (float*)malloc(dataSize * sizeof(float));
    dataset.consumption = (float*)malloc(dataSize * sizeof(float));
    dataset.temperature = (float*)malloc(dataSize * sizeof(float));
    dataset.size = dataSize;
    return dataset;
}

// Helper function to allocate device dataset
EnergyDataset allocateDeviceDataset(int dataSize) {
    EnergyDataset dataset;
    checkCudaError(cudaMalloc((void**)&dataset.timestamps, dataSize * sizeof(float)),
        "cudaMalloc timestamps");
    checkCudaError(cudaMalloc((void**)&dataset.temperature, dataSize * sizeof(float)),
        "cudaMalloc temperature");
    checkCudaError(cudaMalloc((void**)&dataset.consumption, dataSize * sizeof(float)),
        "cudaMalloc consumptions");
    dataset.size = dataSize;
    return dataset;
}

// Helper function to allocate ML features
MLFeatures allocateFeatures(int dataSize, int numFeatures) {
    MLFeatures features;
    features.size = dataSize;
    features.numFeatures = numFeatures;

    // Allocate host memory
    checkCudaError(cudaMalloc((void**)&features.timeFeatures, dataSize * 2 * sizeof(float)),
        "cudaMalloc timeFeatures");
    checkCudaError(cudaMalloc((void**)&features.tempFeatures, dataSize  * sizeof(float)),
        "cudaMalloc tempFeatures");
    checkCudaError(cudaMalloc((void**)&features.movingAvgs, dataSize * sizeof(float)),
        "cudaMalloc movingAvgs");
    checkCudaError(cudaMalloc((void**)&features.hourlyPatterns, 24 * sizeof(float)),
        "cudaMalloc hourlyPatterns");
    checkCudaError(cudaMalloc((void**)&features.weatherFeatures, dataSize * 3 * sizeof(float)),
        "cudaMalloc weatherFeatures");

    return features;
}

// Helper function to free device dataset
void freeDeviceDataset(EnergyDataset dataset) {
    cudaFree(dataset.temperature);
    cudaFree(dataset.timestamps);
    cudaFree(dataset.consumption);
}

// Helper function to free host dataset
void freeHostDataset(EnergyDataset dataset) {
    free(dataset.temperature);
    free(dataset.timestamps);
    free(dataset.consumption);
}

// Helper function to free ML features
void freeMLFeatures(MLFeatures features) {
    cudaFree(features.hourlyPatterns);
    cudaFree(features.tempFeatures);
    cudaFree(features.timeFeatures);
    cudaFree(features.movingAvgs);
    cudaFree(features.weatherFeatures);
}

int main() {
    // Simulation parameters
    const int dataSize = 8760; // Number of hourly samples in a year
    const int windowSize = 24; // 24-hour moving average window
    const int numMlFeatures = 12; // Nuber of features for ML model

    // OPTIMISATION: Use CUDA streams for concurrent kernel execution
    cudaStream_t stream1, stream2;
    checkCudaError(cudaStreamCreate(&stream1), "Stream1 creation");
    checkCudaError(cudaStreamCreate(&stream2), "Stream2 creation");

    printf("Generating synthetic energy data for a year...\n");

    // Allocate host memory with optimized layout
    EnergyDataset h_energyData = allocateHostDataset(dataSize);

    // Generate synthetic data
    generateSyntheticEnergyData(
        h_energyData.timestamps,
        h_energyData.temperature,
        h_energyData.consumption,
        dataSize
        );

    // Allocate host memory for results
    float* h_movingAvg = (float*)malloc(dataSize * sizeof(float));
    float* h_hourlyAverages = (float*)malloc(24 * sizeof(float));
    float* h_hourlyCounts = (float*)malloc(24 * sizeof(float));
    float* h_mlFeatures = (float*)malloc(dataSize * numMlFeatures * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < 24; i++) {
        h_hourlyAverages[i] = 0.0f;
        h_hourlyCounts[i] = 0.0f;
    }

    // Allocate device memory with optimized layout
    EnergyDataset d_energyData = allocateDeviceDataset(dataSize);

    // Allocate device memory for results
    float* d_movingAvg;
    float* d_hourlyAverages;
    float* d_hourlyCounts;
    float* d_mlFeatures;

    checkCudaError(cudaMalloc((void**)&d_movingAvg, dataSize * sizeof(float)),
        "cudaMalloc d_movingAvg");
    checkCudaError(cudaMalloc((void**)&d_hourlyAverages, 24 * sizeof(float)),
        "cudaMalloc d_hourlyAverages");
    checkCudaError(cudaMalloc((void**)&d_hourlyCounts, 24 * sizeof(float)),
        "cudaMalloc d_hourlyCounts");
    checkCudaError(cudaMalloc((void**)&d_mlFeatures, dataSize * numMlFeatures * sizeof(float)),
        "cudaMalloc d_mlFeatures");

    // Initialize hourly averages and counts on device
    checkCudaError(cudaMemset(d_hourlyAverages, 0, 24 * sizeof(float)),
        "cudaMemset d_hourlyAverages");
    checkCudaError(cudaMemset(d_hourlyCounts, 0, 24 * sizeof(float)),
        "cudaMemset d_hourlyCounts");

    // Copy data from host to device
    // OPTIMIZATION: Use cudaMemcpyAsync with streams for concurrent transfers
    checkCudaError(cudaMemcpyAsync(d_energyData.timestamps, h_energyData.timestamps,
        dataSize * sizeof(float), cudaMemcpyHostToDevice, stream1),
        "cudaMemcpy timestamps to device");
    checkCudaError(cudaMemcpyAsync(d_energyData.temperature, h_energyData.temperature,
        dataSize * sizeof(float), cudaMemcpyHostToDevice, stream1),
        "cudaMemcpy temperature to device");
    checkCudaError(cudaMemcpyAsync(d_energyData.consumption, h_energyData.consumption,
        dataSize * sizeof(float), cudaMemcpyHostToDevice, stream2),
        "cudaMemcpy consumption to device");

    // Allocate ML features structure
    MLFeatures mlFeatures = allocateFeatures(dataSize, numMlFeatures);

    // Set CUDA kernel launch parameters
    // OPTIMIZATION: Tuned for better occupancy based on SM resources
    int threadsPerBlock = 256; // Reduced from 512 for better occupancy
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSizeHourly = 2 * 24 * sizeof(float); // 24 sums + 24 counts
    int sharedMemSizeMovingAvg = threadsPerBlock * sizeof(float);

    printf("CUDA kernel launch: Grid with %d blocks, each with %d threads\n",
        blocksPerGrid, threadsPerBlock);

    // Launch preprocessing kernel with stream1
    // OPTIMIZATION: Using stream for concurrent execution
    preprocessEnergyData<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
        d_energyData.timestamps,
        d_energyData.temperature,
        mlFeatures.timeFeatures,
        mlFeatures.tempFeatures,
        dataSize
    );

    // Launch moving average kernel with stream2
    // OPTIMIZATION: Using shared memory to reduce global memory access
    calculateMovingAverages<<<blocksPerGrid, threadsPerBlock, sharedMemSizeMovingAvg, stream2>>>(
        d_energyData.consumption,
        d_movingAvg,
        dataSize,
        windowSize
    );

    // OPTIMIZATION: Use stream1 which should be done with preprocessing by now
    // Launch hourly average kernel with stream1
    calculateHourlyAverages<<<blocksPerGrid, threadsPerBlock, sharedMemSizeHourly, stream1>>>(
        d_energyData.timestamps,
        d_energyData.consumption,
        d_hourlyAverages,
        d_hourlyCounts,
        dataSize
    );

    // Wait for all kernels to complete
    checkCudaError(cudaDeviceSynchronize(), "Kernel synchronization");

    // Normalize hourly averages on the host
    float* h_normalizedHourlyAvgs = (float*)malloc(24 * sizeof(float));

    // Copy results back from device to host
    checkCudaError(cudaMemcpy(h_movingAvg, d_movingAvg,
        dataSize * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy moving avg from device");
    checkCudaError(cudaMemcpy(h_hourlyAverages, d_hourlyAverages,
        24 * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy hourly averages from device");
    checkCudaError(cudaMemcpy(h_hourlyCounts, d_hourlyCounts,
        24 * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy hourly counts from device");

    // Calculate final hourly averages on CPU
    for (int hour = 0; hour < 24; hour++) {
        h_normalizedHourlyAvgs[hour] = (h_hourlyCounts[hour] > 0) ?
            h_hourlyAverages[hour] / h_hourlyCounts[hour] : 0;
    }

    // Copy normalized hourly averages back to device
    checkCudaError(cudaMemcpy(mlFeatures.hourlyPatterns, h_normalizedHourlyAvgs,
        24 * sizeof(float), cudaMemcpyHostToDevice),
        "cudaMemcpy normalized hourly averages to device");

    // Launch the feature extraction kernel for ML
    extractMLFeatures<<<blocksPerGrid, threadsPerBlock>>>(
        d_energyData.timestamps,
        d_energyData.consumption,
        d_energyData.temperature,
        nullptr, // No weather data in this example
        d_mlFeatures,
        dataSize,
        numMlFeatures,
        d_movingAvg,
        mlFeatures.hourlyPatterns
    );

    // Copy ML features back to host
    checkCudaError(cudaMemcpy(h_mlFeatures, d_mlFeatures,
        dataSize * numMlFeatures * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy ML features from device");

    // Calculate final hourly averages
    printf("\nAverage energy consumption by hour of day:\n");
    printf("Hour | Consumption (kWh)\n");
    printf("-----+------------------\n");
    for (int hour = 0; hour < 24; hour++) {
        printf("%4d | %8.2f\n", hour, h_normalizedHourlyAvgs[hour]);
    }

    // Print sample of moving average results
    printf("\nSample of 24-hour moving averages (first day):\n");
    printf("Hour | Raw Consumption | 24h moving average\n");
    printf("-----+-----------------+------------------\n");
    for (int i = 0; i < 24; i++) {
        printf("%4d | %15.2f | %16.2f\n",
            i, h_energyData.consumption[i], h_movingAvg[i]);
    }

    // Print ML feature sample
    printf("\nSample of ML features (first data point):\n");
    printf("Feature | Value\n");
    printf("--------+-------\n");
    const char* featureNames[] = {
        "Hour (sin)", "Hour (cos)", "Day of week (sin)", "Day of week (cos)",
        "Week of year (sin)", "Week of year (cos)", "Temperature", "Normalized temp",
        "Moving avg", "Hourly pattern", "Weather 1", "Weather 2", "Consumption"
    };

    for (int i = 0; i < numMlFeatures; i++) {
        printf("%s | %f\n", featureNames[i], h_mlFeatures[i]);
    }

    // Identify spots for potential quantum computing integration
    printf("\nPotential quantum computing integration points:\n");
    printf("1. Time series forecasting - quantum FFT could accelerate spectral analysis\n");
    printf("2. Feature selection - quantum optimization for optimal feature subset\n");
    printf("3. Pattern recognition - quantum classifiers for demand pattern classification\n");

    // Clean up
    freeHostDataset(h_energyData);
    freeDeviceDataset(d_energyData);
    freeMLFeatures(mlFeatures);

    free(h_movingAvg);
    free(h_hourlyAverages);
    free(h_hourlyCounts);
    free(h_normalizedHourlyAvgs);
    free(h_mlFeatures);

    cudaFree(d_movingAvg);
    cudaFree(d_hourlyAverages);
    cudaFree(d_hourlyCounts);
    cudaFree(d_mlFeatures);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaDeviceReset();

    return 0;
}