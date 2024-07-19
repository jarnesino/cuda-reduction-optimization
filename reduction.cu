#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

TODO: Add time complexity explanations.

*/

int main() {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // The reduction only works for up to 1024 elements (one block of threads), in order to avoid launching several kernels in different ways for each implementation.
    const int logDataSize = 10;
    const int dataSize = 1 << logDataSize;
    int* testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    reduce(1, reduce_using_1_interleaved_addressing_with_divergent_branching, testingData, dataSize, startEvent, stopEvent);
    reduce(2, reduce_using_2_interleaved_addressing_with_bank_conflicts, testingData, dataSize, startEvent, stopEvent);
    reduce(3, reduce_using_3_sequential_addressing_with_idle_threads, testingData, dataSize, startEvent, stopEvent);
    reduce(4, reduce_using_4_first_add_during_load_with_loop_overhead, testingData, dataSize, startEvent, stopEvent);
    reduce(5, reduce_using_5_loop_unrolling_only_at_warp_level_iterations, testingData, dataSize, startEvent, stopEvent);
    reduce(6, reduce_using_6_complete_loop_unrolling_with_one_reduction, testingData, dataSize, startEvent, stopEvent);
    reduce(7, reduce_using_7_multiple_reduce_operations_per_thread_iteration, testingData, dataSize, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

/* Auxiliary */

void reduce(const int implementationNumber, reduceImplementationFunction implementation, int* inputData, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    int outputData[dataSize];
    const int dataSizeInBytes = dataSize * sizeof(int);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **)&deviceInputData, dataSizeInBytes);
    cudaMalloc((void **)&deviceOutputData, dataSizeInBytes);
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    size_t sharedMemSize = threadsPerBlock * sizeof(int);

    // Record the start event.
    cudaEventRecord(startEvent, 0);

    // Launch kernel.
    const int oneBlock = 1;
    implementation<<<oneBlock, threadsPerBlock, sharedMemSize>>>(deviceInputData, deviceOutputData, dataSize);

    // Record the stop event and wait for it to complete.
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaMemcpy(outputData, deviceOutputData, dataSizeInBytes, cudaMemcpyDeviceToHost);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    printImplementationData(implementationNumber, elapsedTimeInMilliseconds, outputData[0]);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
}

void printImplementationData(const int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    std::cout << "*** Implementation number: " << implementationNumber << "\t Elapsed time: " << elapsedTimeInMilliseconds << "\t" << "Reduction result: " << result << std::endl;
}

void initializeTestingDataIn(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = 1;
    }
}
