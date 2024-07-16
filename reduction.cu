#include "reduction.cuh"

int main() {
    const int logDataSize = 10;
    const int dataSize = 1 << logDataSize;

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    reduce(1, reduce_using_1_interleaved_addressing_with_divergent_branching, dataSize, startEvent, stopEvent);
    reduce(2, reduce_using_2_interleaved_addressing_with_bank_conflicts, dataSize, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

/* Auxiliary */

void reduce(const int implementationNumber, reduce_implementation_function implementation, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
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

    // Record the stop event and wait for it to complete.
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(outputData, deviceOutputData, dataSizeInBytes, cudaMemcpyDeviceToHost);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    std::cout << "*****************************************************" << std::endl;
    std::cout << "Elapsed time: " << elapsedTimeInMilliseconds << " ms" << std::endl;
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
