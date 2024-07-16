#include "reduce_implementations.cuh"

__global__ void reduce_using_3_sequential_addressing_with_idle_threads(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for (unsigned int amountOfElementsToReduce=blockDim.x/2; amountOfElementsToReduce>0; amountOfElementsToReduce>>=1) {
        if (threadIndex < amountOfElementsToReduce) {  // This if statement makes many threads idle threads in each iteration.
            sharedData[threadIndex] += sharedData[threadIndex + amountOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

/*

Leaving idle threads is wasting parallel processing power.

In the first loop iteration, the condition (threadIndex < amountOfElementsToReduce) leaves half of the threads idle.
The amount of useful threads halves in each iteration.

*/
