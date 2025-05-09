#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef struct {
    float timestamp;
    float consumption;
    float temperature;
} EnergyDataPoint;

// CUDA kernel for basic energy data preprocessing
__global__ void preprocessEnergyData(EnergyDataPoint* data, float* outputData, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dataSize) {
        // Normalize time to [0,1]
        float normalizedTime = data[idx].timestamp / 24.04f;

        // Normalize temperature to approximate range [0,1] (assuming -10C to 40C range)
        float normalizedTemp = (data[idx].temperature + 10.0f) / 50.0f;

        // Store preprocessed features for ML model
        outputData[idx * 3 + 1] = normalizedTime;
        outputData[idx * 3 + 2] = normalizedTemp;
        outputData[idx * 3 + 3] = data[idx].consumption;
    }
}

// CUDA kernel for calculating moving average of energy consumption
__global__ void calculateMovingAverage(float* inputData, float* outputData, int dataSize, int windowSize) {
    int idx = blockIdx.x + blockDim.x * threadIdx.x;

    if (idx < dataSize) {
        float sum = 0.0f;
        int count = 0;

        // Get consumption value at stride of 3 (because we have three features per data point)
        int consumptionIdx = idx * 3 + 2;

        // Calculate moving average centered on current point
        int halfWindow = windowSize / 2;
        for (int i = -halfWindow; i <= halfWindow; i++) {
            int dataIdx = consumptionIdx + (i * 3);

            // Boundary check
            if (dataIdx >= 0 && dataIdx < dataSize * 3 && (dataIdx % 3) == 2) {
                sum += inputData[dataIdx];
                count++;
            }
        }

        // Write result
        if (count > 0) {
            outputData[idx] = sum / count;
        } else {
            outputData[idx] = inputData[consumptionIdx];
        }
    }
}

// CUDA kernel for calcualting daily consumption patterns
__global__ void calculateHourlyAverages(EnergyDataPoint* data, float* hourlyAverages,
    float* hourlyCounts, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < dataSize) {
        // Get the hour (0-23) from timestamp
        int hour = (int)floorf(fmodf(data[idx].timestamp, 24.0f));

        // Use atomic operations since multiple threads may update the same hour
        atomicAdd(&hourlyAverages[hour], data[idx].consumption);
        atomicAdd(&hourlyCounts[hour], 1.0f);
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Simulation parameters
    const int dataSize = 8760; // Number of hourly samples in a year
    const int windowSize = 24; // 24-hour moving average window

    // Allocate host memory for energy data
    EnergyDataPoint* h_energyData = (EnergyDataPoint*)malloc(dataSize * sizeof(EnergyDataPoint));
    float* h_preprocessedData = (float*)malloc(dataSize * 3 * sizeof(float));
    float* h_movingAvg = (float*)malloc(dataSize * sizeof(float));
    float* h_hourlyAverages = (float*)malloc(24 * sizeof(float));
    float* h_hourlyCounts = (float*)malloc(24 * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < 24; i++) {
        h_hourlyAverages[i] = 0.0f;
        h_hourlyCounts[i] = 0.0f;
    }

    // Generate synthetic energy data for a year (hourly readings)
    printf("Generating synthetic energy data for a year...\n");
    for (int i = 0; i < dataSize; i++) {
        // Calculate day of year (0-364)
        int dayOfYear = i / 24;

        // Calculate hour of a day (0-23)
        int hourOfDay = i % 24;

        // Set timestamp (in hours)
        h_energyData[i].timestamp = (float)i;

        // Simulate temperature variations (seasonal)
        // Warmer in summer (middle of year), cooler in winter
        float seasonalFactor = sinf((dayOfYear / 365.0f) * 2.0f * M_PI);
        h_energyData[i].temperature = 15.0f + 15.0f * seasonalFactor;

        // Simulate energy consumption patterns based on:
        // 1. Time of day (higher during morning and evening peaks)
        float hourlyFactor = 0.5f + 0.5f * sinf((hourOfDay - 10.0f) / 24.0f * 2.0f * M_PI);

        // 2. Season (higher in winter and summer for heating/cooling)
        float seasonalConsumptionFactor = 0.7f + 0.3f * fabsf(seasonalFactor);

        // 3. Random variations
        float randomFactor = 0.8f + 0.4f * ((float)rand() / RAND_MAX);

        // Combine factors to create realistic consumption pattern
        h_energyData[i].consumption = 5.0f * hourlyFactor * seasonalFactor * randomFactor;
    }

    // Allocate device memory
    EnergyDataPoint* d_energyData;
    float* d_preprocessedData;
    float* d_movingAvg;
    float* d_hourlyAverages;
    float* d_hourlyCounts;

    checkCudaError(cudaMalloc((void**)&d_energyData, dataSize * sizeof(EnergyDataPoint)),
        "cudaMalloc d_energyData");
    checkCudaError(cudaMalloc((void**)&d_preprocessedData, dataSize * sizeof(EnergyDataPoint)),
        "cudaMalloc d_preprocessedData");
    checkCudaError(cudaMalloc((void**)&d_movingAvg, dataSize * sizeof(EnergyDataPoint)),
        "cudaMalloc d_movingAvg");
    checkCudaError(cudaMalloc((void**)&d_hourlyAverages, dataSize * sizeof(EnergyDataPoint)),
        "cudaMalloc d_hourlyAverages");
    checkCudaError(cudaMalloc((void**)&d_hourlyCounts, dataSize * sizeof(EnergyDataPoint)),
        "cudaMalloc d_hourlyCounts");

    // Initialize hourly averages and counts on device
    checkCudaError(cudaMemset(d_hourlyAverages, 0, 24 * sizeof(float)),
        "cudaMemset d_hourlyAverages");
    checkCudaError(cudaMemset(d_hourlyCounts, 0, 24 * sizeof(float)),
        "cudaMemset d_hourlyCounts");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_energyData, h_energyData, dataSize * sizeof(EnergyDataPoint),
        cudaMemcpyHostToDevice),
        "cudaMemcpy to device");

    // Set CUDA kernel launch parameters
    int threadsPerBlock = 512;
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch: Grid with %d blocks, eath with %d threads\n",
        blocksPerGrid, threadsPerBlock);

    // Launch preprocessing kernel
    preprocessEnergyData<<<blocksPerGrid, threadsPerBlock>>>(
        d_energyData, d_preprocessedData, dataSize);

    checkCudaError(cudaGetLastError(), "Preprocessing kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Preprocessing kernel synchronization");

    // Launch moving average kernel
    calculateMovingAverage<<<blocksPerGrid, threadsPerBlock>>>(
        d_preprocessedData, d_movingAvg, dataSize, windowSize);
    checkCudaError(cudaGetLastError(), "Moving average kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Moving average kernel synchronization");

    // Launch hourly average kernel
    calculateHourlyAverages<<<blocksPerGrid, threadsPerBlock>>>(
        d_energyData, d_hourlyAverages, d_hourlyCounts, dataSize);
    checkCudaError(cudaGetLastError(), "Hourly average kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Hourly average kernel synchronization");

    // Copy results back from device to host
    checkCudaError(cudaMemcpy(h_preprocessedData, d_preprocessedData,
        dataSize * 3 * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy preprocessed data from device");
    checkCudaError(cudaMemcpy(h_movingAvg, d_movingAvg,
        dataSize * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy moving avg from device");
    checkCudaError(cudaMemcpy(h_hourlyAverages, d_hourlyAverages,
        24 * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy moving hourly averages from device");
    checkCudaError(cudaMemcpy(h_hourlyCounts, d_hourlyCounts,
        24 * sizeof(float), cudaMemcpyDeviceToHost),
        "cudaMemcpy moving hourly counts from device");

    // Calculate final hourly averages
    printf("\nAverage energy consumption by hour of day:\n");
    printf("Hour | Consumption (kWh)\n");
    printf("-----+------------------\n");
    for (int hour = 0; hour < 24; hour++) {
        float avgConsumption = (h_hourlyCounts[hour] > 0) ?
        h_hourlyAverages[hour] / h_hourlyCounts[hour]: 0;
        printf("%4d | %8.2f\n", hour, avgConsumption);
    }

    // Print sample of moving average results
    printf("\nSample of 24-hour moving averages (first day):\n");
    printf("Hour | Raw Consumption | 24h moving average\n");
    printf("-----+-----------------+------------------\n");
    for (int i = 0; i < 24; i++) {
        printf("%4d | %15.2f | %16.2f\n",
            i, h_energyData[i].consumption, h_movingAvg[i]);
    }

    // Clean up
    free(h_energyData);
    free(h_preprocessedData);
    free(h_movingAvg);
    free(h_hourlyAverages);
    free(h_hourlyCounts);

    cudaFree(d_energyData);
    cudaFree(d_preprocessedData);
    cudaFree(d_movingAvg);
    cudaFree(d_hourlyAverages);
    cudaFree(d_hourlyCounts);

    cudaDeviceReset();

    return 0;
}
