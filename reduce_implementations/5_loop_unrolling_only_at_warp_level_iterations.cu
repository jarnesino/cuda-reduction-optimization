#include "reduce_implementations.cuh"

__device__ void warpReduce(volatile int* sharedData, int threadIndex);

__global__ void reduce_using_5_loop_unrolling_only_at_warp_level_iterations(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + blockDim.x];
    __syncthreads();

    // Do reduction in shared memory.
    for (unsigned int amountOfElementsToReduce=blockDim.x/2; amountOfElementsToReduce>32; amountOfElementsToReduce>>=1) {  // This loop produces instruction overhead.
        if (threadIndex < amountOfElementsToReduce) {
            sharedData[threadIndex] += sharedData[threadIndex + amountOfElementsToReduce];
        }
        __syncthreads();
    }
    if (threadIndex > 32) warpReduce(sharedData, threadIndex);

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

__device__ void warpReduce(volatile int* sharedData, int threadIndex) {
    sharedData[threadIndex] += sharedData[threadIndex + 32];
    sharedData[threadIndex] += sharedData[threadIndex + 16];
    sharedData[threadIndex] += sharedData[threadIndex + 8];
    sharedData[threadIndex] += sharedData[threadIndex + 4];
    sharedData[threadIndex] += sharedData[threadIndex + 2];
    sharedData[threadIndex] += sharedData[threadIndex + 1];
}

/*

Loops like this one produce instruction overhead.

Completely unrolling the loop could be a good solution.
We could know what the limitation for threads per block is.
In this case, it's 1024 (2^10).
We can use this to completely unroll the loop in the kernel.
Given that we don't know the block size at compile time, we can use C++ template parameters, supported by CUDA in host and device functions.

*/