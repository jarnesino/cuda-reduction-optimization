#include "reduction.cuh"

int reduceWithKernelInDevice(
        const ReduceImplementationKernel &reduceImplementationKernel,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
);

TimedReductionResult reduceAndMeasureTimeWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, const unsigned int dataSize
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

    return TimedReductionResult{value, elapsedTimeInMilliseconds};
}

TimedReductionResult reduceAndMeasureTimeWithNonKernel(
        const ReduceNonKernelImplementation &implementation, int *inputData, const unsigned int dataSize
) {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    int value = implementation.function(inputData, dataSize);

    // Record the CUDA stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    // Destroy the CUDA events for timing.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return TimedReductionResult{value, elapsedTimeInMilliseconds};
}
