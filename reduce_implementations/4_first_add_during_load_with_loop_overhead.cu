#include "reduce_implementations.cuh"

__global__ void reduce_using_4_first_add_during_load_with_loop_overhead(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + blockDim.x];
    __syncthreads();

    // Do reduction in shared memory.
    for (unsigned int amountOfElementsToReduce=blockDim.x/2; amountOfElementsToReduce>0; amountOfElementsToReduce>>=1) {  // This loop produces instruction overhead.
        if (threadIndex < amountOfElementsToReduce) {
            sharedData[threadIndex] += sharedData[threadIndex + amountOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

/*

Loops like this one produce instruction overhead.

Unrolling the loop could be a good solution.
We know all threads in a warp are SIMD-synchronous. Then, we could safely unroll the last warp, when (amountOfElementsToReduce <= 32).
The if statement is also unnecessary, because it doesn't save work between threads in a warp.

*/
