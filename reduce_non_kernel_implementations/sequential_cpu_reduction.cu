#include "sequential_cpu_reduction.cuh"

ReductionResult reduceAndMeasureTimeWithCPU(int *inputData, unsigned int size) {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    int sum = 0;
    for (unsigned int index = 0; index < size; index++) {
        sum += inputData[index];
    }

    // Record the CUDA stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    // Destroy the CUDA events for timing.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return ReductionResult{sum, elapsedTimeInMilliseconds};
}
