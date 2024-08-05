#include "reduction.cuh"

ReductionResult reduceAndMeasureTime(
        ReduceImplementation reduceImplementation,
        int *inputData,
        const unsigned int dataSize
) {

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

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

    // Record the start event.
    cudaEventRecord(startEvent, nullptr);

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

    // Record the stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    int value;
    cudaMemcpy(&value, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return ReductionResult{value, elapsedTimeInMilliseconds};
}

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void initializeTestingDataIn(int *data, int size) {
    fillDataWith1s(data, size);
}

void fillDataWith1s(int *data, int size) {
    for (int index = 0; index < size; ++index) {
        data[index] = 1;
    }
}