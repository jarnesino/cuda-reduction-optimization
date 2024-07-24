#include "reduce_implementations.cuh"
#include "../reduction.cuh"

__device__ void warpReduce(volatile int *sharedData, int threadIndex);

__global__ void reduce_using_5_loop_unrolling_only_at_warp_level_iterations(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * BLOCK_SIZE * 2 + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + BLOCK_SIZE];
    __syncthreads();

    // Do reduction in shared memory.
    for (
            unsigned int amountOfElementsToReduce = BLOCK_SIZE / 2;
            amountOfElementsToReduce > 32;
            amountOfElementsToReduce >>= 1
            ) {  // This loop produces instruction overhead.
        if (threadBlockIndex < amountOfElementsToReduce) {
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + amountOfElementsToReduce];
        }
        __syncthreads();
    }
    if (threadBlockIndex < 32) warpReduce(sharedData, threadBlockIndex);

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

__device__ void warpReduce(volatile int *sharedData, int threadBlockIndex) {
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 32];
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 16];
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 8];
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 4];
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 2];
    sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 1];
}

/*

Loops like this one produce instruction overhead.

Completely unrolling the loop could be a good solution.
We could know what the limitation for threads per block is.
In this case, it's 1024 (2^10).
We can use this to completely unroll the loop in the kernel.
Given that we don't know the block size at compile time, we can use C++ template parameters, supported by CUDA in host and device functions.

*/
