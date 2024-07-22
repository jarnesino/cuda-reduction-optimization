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

    const int logDataSize = 30;  // At least one element in the testing data
    const int dataSize = 1 << logDataSize;
    int* testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    reduce(0, reduce_using_0_interleaved_addressing_with_local_memory, 1, testingData, dataSize, startEvent, stopEvent);
    reduce(1, reduce_using_1_interleaved_addressing_with_divergent_branching, 1, testingData, dataSize, startEvent, stopEvent);
    reduce(2, reduce_using_2_interleaved_addressing_with_bank_conflicts, 1, testingData, dataSize, startEvent, stopEvent);
    reduce(3, reduce_using_3_sequential_addressing_with_idle_threads, 1, testingData, dataSize, startEvent, stopEvent);
    reduce(4, reduce_using_4_first_add_during_load_with_loop_overhead, 2, testingData, dataSize, startEvent, stopEvent);
    reduce(5, reduce_using_5_loop_unrolling_only_at_warp_level_iterations, 2, testingData, dataSize, startEvent, stopEvent);
    reduce(6, reduce_using_6_complete_loop_unrolling_with_one_reduction, 2, testingData, dataSize, startEvent, stopEvent);
    reduce(7, reduce_using_7_multiple_reduce_operations_per_thread_iteration, 2, testingData, dataSize, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

/* Auxiliary */

void reduce(
    const int implementationNumber,
    reduceImplementationFunction implementation,
    const int blockSizedChunksReducedPerBlock,
    int* inputData,
    const int dataSize,
    cudaEvent_t startEvent,
    cudaEvent_t stopEvent
) {
    const int threadsPerBlock = BLOCK_SIZE;
    const size_t dataSizeInBytes = dataSize * sizeof(int);
    int amountOfBlocks = amountOfBlocksForReduction(dataSize, threadsPerBlock, blockSizedChunksReducedPerBlock);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **)&deviceInputData, dataSizeInBytes);
    cudaMalloc((void **)&deviceOutputData, amountOfBlocks * sizeof(int) * 2);  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);

    int remainingElements = dataSize;
    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;
    const size_t sharedMemSize = threadsPerBlock * sizeof(int);

    // Record the start event.
    cudaEventRecord(startEvent, 0);

    // Launch kernel for each block.
    while (remainingElements > 1) {
        amountOfBlocks = amountOfBlocksForReduction(remainingElements, threadsPerBlock, blockSizedChunksReducedPerBlock);
        implementation<<<amountOfBlocks, threadsPerBlock, sharedMemSize>>>(inputPointer, outputPointer, remainingElements);
        cudaDeviceSynchronize();

        remainingElements = amountOfBlocks;
        inputPointer = outputPointer;
        outputPointer += remainingElements;
    }

    // Record the stop event and wait for it to complete.
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    int finalResult;
    cudaMemcpy(&finalResult, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    printImplementationData(implementationNumber, elapsedTimeInMilliseconds, finalResult);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);
}

int amountOfBlocksForReduction(const int dataSize, const int threadsPerBlock, const int blockSizedChunksReducedPerBlock) {
    return (dataSize + threadsPerBlock * blockSizedChunksReducedPerBlock - 1) / (threadsPerBlock * blockSizedChunksReducedPerBlock);
}

void printImplementationData(const int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    std::cout << "*** Implementation number: " << implementationNumber;
    std::cout << "\t Elapsed time: " << elapsedTimeInMilliseconds;
    std::cout << "\t Reduction result: " << result << std::endl;
}

void initializeTestingDataIn(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = 1;
    }
}
