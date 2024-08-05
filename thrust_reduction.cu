#include "thrust_reduction.cuh"

ReductionResult reduceWithCudaThrustLibrary(int *inputData, unsigned int size) {
    thrust::device_vector<int> deviceInputData(inputData, inputData + size);

    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    // Record the CUDA start event.
    cudaEventRecord(startEvent, nullptr);

    int sum = thrust::reduce(deviceInputData.begin(), deviceInputData.end(), 0, thrust::plus<int>());

    // Record the CUDA stop event and wait for it to complete.
    cudaEventRecord(stopEvent, nullptr);
    cudaEventSynchronize(stopEvent);

    float elapsedTimeInMilliseconds;
    cudaEventElapsedTime(&elapsedTimeInMilliseconds, startEvent, stopEvent);

    // Destroy the CUDA events for timing.
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    int value;
    cudaMemcpy(&value, &sum, sizeof(int), cudaMemcpyHostToDevice);

    return ReductionResult{value, elapsedTimeInMilliseconds};
}
