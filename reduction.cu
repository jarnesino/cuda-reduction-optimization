#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

int main() {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    const int logDataSize = 30;  // At least one element in the testing data
    const int dataSize = 1 << logDataSize;
    int *testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    reduceAndMeasureTime(
            0, reduce_using_0_interleaved_addressing_with_local_memory,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            1, reduce_using_1_interleaved_addressing_with_divergent_branching,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            2, reduce_using_2_interleaved_addressing_with_bank_conflicts,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            3, reduce_using_3_sequential_addressing_with_idle_threads,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            4, reduce_using_4_first_add_during_load_with_loop_overhead,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            5, reduce_using_5_loop_unrolling_only_at_warp_level_iterations,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            6, reduce_using_6_complete_loop_unrolling_with_one_reduction,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            7, reduce_using_7_multiple_reduce_operations_per_thread_iteration,
            amountOfBlocksForReductionWithMultipleSteps, testingData, dataSize, startEvent, stopEvent
    );
    reduceAndMeasureTime(
            8, reduce_using_8_operations_for_consecutive_memory_addressing,
            amountOfBlocksForReductionWithConsecutiveMemoryAddressing, testingData, dataSize, startEvent, stopEvent
    );

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

/* Auxiliary */

void reduceAndMeasureTime(
        const int implementationNumber,
        reduceImplementationFunction implementation,
        amountOfBlocksFunction amountOfBlocksFor,
        int *inputData,
        const int dataSize,
        cudaEvent_t startEvent,
        cudaEvent_t stopEvent
) {
    const size_t dataSizeInBytes = dataSize * sizeof(int);
    int remainingElements = dataSize;
    int amountOfBlocks = amountOfBlocksFor(remainingElements);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **) &deviceInputData, dataSizeInBytes);
    cudaMalloc((void **) &deviceOutputData,
               amountOfBlocks * sizeof(int) * 2);  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(int);

    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;

    // Record the start event.
    cudaEventRecord(startEvent, 0);

    // Launch kernel for each block.
    while (remainingElements > 1) {
        amountOfBlocks = amountOfBlocksFor(remainingElements);
        implementation<<<amountOfBlocks, BLOCK_SIZE, sharedMemSize>>>(
                inputPointer, outputPointer, remainingElements
        );
        cudaDeviceSynchronize();
        checkForCUDAErrors();

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

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int amountOfBlocksForStandardReduction(const int dataSize) {
    return (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


int amountOfBlocksForReductionWithExtraStep(const int dataSize) {
    const int blockSizedChunksReducedPerBlock = 2;
    return (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
           (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
}


int amountOfBlocksForReductionWithMultipleSteps(const int dataSize) {
    return min(GRID_SIZE, amountOfBlocksForReductionWithExtraStep(dataSize));
}


int amountOfBlocksForReductionWithConsecutiveMemoryAddressing(const int dataSize) {
    const int blockSizedChunksReducedPerBlock = 4;
    const int blocks = (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
                       (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
    return min(GRID_SIZE, blocks);
}

void printImplementationData(const int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Reduction result: %d\n", result);
}

void initializeTestingDataIn(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = rand() % 100;
    }
}
