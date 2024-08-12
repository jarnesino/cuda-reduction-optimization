#include "custom_reduce_implementations.cuh"

__global__ void sequential_addressing_with_idle_threads(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for (
            unsigned int numberOfElementsToReduce = blockSize >> 1;
            numberOfElementsToReduce > 0;
            numberOfElementsToReduce >>= 1
            ) {
        if (threadBlockIndex <
            numberOfElementsToReduce) {  // This if statement makes many threads idle threads in each iteration.
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + numberOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

/*

Leaving idle threads is wasting parallel processing power.

In the first loop iteration, the condition (threadIndex < numberOfElementsToReduce) leaves half of the threads idle.
The number of useful threads halves in each iteration.

*/
