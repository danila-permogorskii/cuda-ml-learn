// Basic CUDA program demonstrating the structure for energy applications

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function
// This kernel will run on the GPU with many threads in parallel
__global__ void energyDataKernel(float *d_energyData, int dataSize) {
    // Calculate unique thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't access beyond array bounds
    if (idx < dataSize) {
        // Simple computation: simulate processing energy consumption data
        // In a real application, this would be more complex
        d_energyData[idx] = d_energyData[idx] * 0.001f; // Convert to kwh if in Wh
    }
}


int main() {
    // Simulation: Number of energy consumption data points
    const int dataSize = 1024;
    size_t dataBytes = dataSize * sizeof(float);

    // Host (CPU) data arrays
    float *h_energyData = new float[dataSize];

    // Initialize with sample energy consumption data (in Wh)
    for (int i = 0; i < dataSize; i++) {
        h_energyData[i] = static_cast<float>(1000 + i % 1000); // Simulated Wh readings
    }

    // Device (GPU) data arrays
    float *d_energyData = nullptr;

    // Cuda kernel launch configuration
    int threadsPerBlock = 1024; // This is a specific value for my rtx 3050ti
    int blocksPerGrid = (dataSize + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory on the GPU
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void **) &d_energyData, dataBytes);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy data from host to device (CPU to GPU)
    cudaStatus = cudaMemcpy(d_energyData, h_energyData, dataBytes, cudaMemcpyHostToDevice);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    printf("CUDA kernel launch: Grid with %d blocks, each with %d threads\n", blocksPerGrid, threadsPerBlock);

    // Launch CUDA kernel on GPU
    energyDataKernel<<<blocksPerGrid, threadsPerBlock>>>(d_energyData, dataSize);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "energyDataKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Wait for GPU to finish
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy results from device to host (GPU to CPU)
    cudaStatus = cudaMemcpy(h_energyData, d_energyData, dataBytes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy from device failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Print a few results to verify
    printf("Energy data Processing results (first 5 values, in kWh):\n");
    for (int i = 0; i < 5; i++) {
        printf("Data point %d: %.3f kWh\n", i, h_energyData[i]);
    }

    // Success!
    printf("CUDA processing successful!\n");

Error:
    // Free GPU memory
    cudaFree(d_energyData);

    // Free CPU memory
    delete[] h_energyData;

    // Reset device
    cudaDeviceReset();

    return (cudaStatus == cudaSuccess) ? 0 : 1;
}
