#include "reduction.cuh"

int reduceWithCustomImplementation(const ReduceImplementationKernel &reduceImplementation, int *inputData, unsigned int dataSize);

int reduceWithKernel(
        const ReduceImplementationKernel &reduceImplementationKernel,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
);

ReductionResult reduceAndMeasureTime(
        const ReduceImplementationKernel &reduceImplementation,
        int *inputData,
        const unsigned int dataSize
) {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    int value = reduceWithCustomImplementation(reduceImplementation, inputData, dataSize);

    // Record the CUDA stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    // Destroy the CUDA events for timing.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return ReductionResult{value, elapsedTimeInMilliseconds};
}

int reduceWithCustomImplementation(const ReduceImplementationKernel &reduceImplementation, int *inputData, unsigned int dataSize) {
    const size_t dataSizeInBytes = dataSize * sizeof(int);
    unsigned int remainingElements = dataSize;
    unsigned int numberOfBlocks = reduceImplementation.numberOfBlocksFunction(remainingElements);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **) &deviceInputData, dataSizeInBytes);
    cudaMalloc(
            (void **) &deviceOutputData,
            numberOfBlocks * sizeof(int) * 2
    );  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(int);

    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;

    int value = reduceWithKernel(
            reduceImplementation, remainingElements, numberOfBlocks, sharedMemSize, inputPointer, outputPointer
    );

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    return value;
}

int reduceWithKernel(
        const ReduceImplementationKernel &reduceImplementationKernel,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        const size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
) {
    // Launch kernel for each block.
    while (remainingElements > 1) {
        numberOfBlocks = reduceImplementationKernel.numberOfBlocksFunction(remainingElements);
        reduceImplementationKernel.function<<<numberOfBlocks, BLOCK_SIZE, sharedMemSize>>>(
                inputPointer, outputPointer, remainingElements
        );
        cudaDeviceSynchronize();
        checkForCUDAErrors();

        remainingElements = numberOfBlocks;
        inputPointer = outputPointer;
        outputPointer += remainingElements;
    }

    int value;
    cudaMemcpy(&value, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);
    return value;
}

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}
