#include "reduction.cuh"

ReductionResult reduceAndMeasureTime(
        reduceImplementationFunction implementation,
        amountOfBlocksFunction amountOfBlocksFor,
        int *inputData,
        const unsigned int dataSize
) {

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    const size_t dataSizeInBytes = dataSize * sizeof(int);
    unsigned int remainingElements = dataSize;
    unsigned int amountOfBlocks = amountOfBlocksFor(remainingElements);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **) &deviceInputData, dataSizeInBytes);
    cudaMalloc((void **) &deviceOutputData,
               amountOfBlocks * sizeof(int) * 2);  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(int);

    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;

    // Record the start event.
    cudaEventRecord(startEvent, nullptr);

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
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    int finalResult;
    cudaMemcpy(&finalResult, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return ReductionResult{finalResult, elapsedTimeInMilliseconds};
}

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

unsigned int amountOfBlocksForStandardReduction(const unsigned int dataSize) {
    return (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


unsigned int amountOfBlocksForReductionWithExtraStep(const unsigned int dataSize) {
    const int blockSizedChunksReducedPerBlock = 2;
    return (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
           (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
}


unsigned int amountOfBlocksForReductionWithMultipleSteps(const unsigned int dataSize) {
    return unsignedMin(GRID_SIZE, amountOfBlocksForReductionWithExtraStep(dataSize));
}


unsigned int amountOfBlocksForReductionWithConsecutiveMemoryAddressing(const unsigned int dataSize) {
    const unsigned int blockSizedChunksReducedPerBlock = 4;
    const unsigned int blocks = (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
                                (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
    return unsignedMin(GRID_SIZE, blocks);
}

void initializeTestingDataIn(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = 1;
    }
}

unsigned int unsignedMin(unsigned int a, unsigned int b) {
    return a < b ? a : b;
}