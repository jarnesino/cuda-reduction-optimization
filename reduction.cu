#include "reduction.cuh"

int reduceWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
);

int reduceWithKernelInDevice(
        const ReduceImplementationKernel &reduceImplementationKernel,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
);

ReductionResult reduceAndMeasureTime(
        const ReduceImplementationKernel &reduceKernel,
        int *inputData,
        const unsigned int dataSize
) {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    int value = reduceWithKernel(reduceKernel, inputData, dataSize);

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
