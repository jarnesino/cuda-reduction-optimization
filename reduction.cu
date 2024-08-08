#include "reduction.cuh"

int *reduce(
        const ReduceImplementation &reduceImplementation,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
);

ReductionResult reduceAndMeasureTime(
        ReduceImplementation reduceImplementation,
        int *inputData,
        const unsigned int dataSize
) {
    const size_t dataSizeInBytes = dataSize * sizeof(int);
    unsigned int remainingElements = dataSize;
    unsigned int numberOfBlocks = reduceImplementation.numberOfBlocksFunction(remainingElements);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **) &deviceInputData, dataSizeInBytes);
    cudaMalloc((void **) &deviceOutputData,
               numberOfBlocks * sizeof(int) * 2);  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(int);

    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    inputPointer = reduce(
            reduceImplementation, remainingElements, numberOfBlocks, sharedMemSize, inputPointer, outputPointer
    );

    int value;
    cudaMemcpy(&value, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);

    // Record the CUDA stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    // Destroy the CUDA events for timing.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    return ReductionResult{value, elapsedTimeInMilliseconds};
}

int *reduce(
        const ReduceImplementation &reduceImplementation,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        const size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
) {
    // Launch kernel for each block.
    while (remainingElements > 1) {
        numberOfBlocks = reduceImplementation.numberOfBlocksFunction(remainingElements);
        reduceImplementation.function<<<numberOfBlocks, BLOCK_SIZE, sharedMemSize>>>(
                inputPointer, outputPointer, remainingElements
        );
        cudaDeviceSynchronize();
        checkForCUDAErrors();

        remainingElements = numberOfBlocks;
        inputPointer = outputPointer;
        outputPointer += remainingElements;
    }

    return inputPointer;
}

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}
