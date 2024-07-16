#include <iostream>
#include <cuda_runtime.h>

#include "1_interleaved_addressing_with_divergent_branching.cuh"

void initializeRandomTestingData(int *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 100;
    }
}

int main() {
    const int log_size = 19;
    const int size = 1 << log_size;
    const int bytes = size * sizeof(int);

    int h_idata[size];
    int h_odata[size];

    initializeRandomTestingData(h_idata, size);

    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, bytes);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(int);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Launch kernel
    reduce_using_1_interleaved_addressing_with_divergent_branching<<<blocks, threads, sharedMemSize>>>(d_idata, d_odata, size);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "*****************************************************" << std::endl;

    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Reduction result: " << h_odata[0] << std::endl;

    std::cout << "*****************************************************" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
