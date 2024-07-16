#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

void initializeRandomTestingDataIn(int *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = 1;
    }
}

int main() {
    const int logDataSize = 10;
    const int dataSize = 1 << logDataSize;
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

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

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

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    return 0;
}
