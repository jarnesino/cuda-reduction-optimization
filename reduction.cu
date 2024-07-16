#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"
#include "reduction.h"

int main() {
    const int logDataSize = 10;
    const int dataSize = 1 << logDataSize;

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    reduce(dataSize, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

/* Auxiliary */

void reduce(const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    const int dataSizeInBytes = dataSize * sizeof(int);

    int inputData[dataSize];
    int outputData[dataSize];

    initializeRandomTestingDataIn(inputData, dataSize);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **)&deviceInputData, dataSizeInBytes);
    cudaMalloc((void **)&deviceOutputData, dataSizeInBytes);

    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int blocks = (dataSize + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(int);

    // Record the start event.
    cudaEventRecord(startEvent, 0);

    // Launch kernel.
    reduce_using_2_interleaved_addressing_with_bank_conflicts<<<blocks, threadsPerBlock, sharedMemSize>>>(deviceInputData, deviceOutputData, dataSize);

    // Record the stop event.
    cudaEventRecord(stopEvent, 0);

    // Wait for the stop event to complete.
    cudaEventSynchronize(stopEvent);

    // Calculate the elapsed time in milliseconds.
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "*****************************************************" << std::endl;

    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(outputData, deviceOutputData, dataSizeInBytes, cudaMemcpyDeviceToHost);

    std::cout << "Reduction result: " << outputData[0] << std::endl;

    std::cout << "*****************************************************" << std::endl;

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
}

void initializeRandomTestingDataIn(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = 1;
    }
}
