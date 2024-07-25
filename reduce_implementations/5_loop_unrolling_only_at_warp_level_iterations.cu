#include "reduce_implementations.cuh"

__device__ void warpReduce(volatile int *data, unsigned int threadIndex);

__global__ void reduce_using_5_loop_unrolling_only_at_warp_level_iterations(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize * 2 + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + blockSize];
    __syncthreads();

    // Do reduction in shared memory.
    for (
            unsigned int amountOfElementsToReduce = blockSize >> 1;
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

__device__ void warpReduce(volatile int *data, const unsigned int threadBlockIndex) {
    data[threadBlockIndex] += data[threadBlockIndex + 32];
    data[threadBlockIndex] += data[threadBlockIndex + 16];
    data[threadBlockIndex] += data[threadBlockIndex + 8];
    data[threadBlockIndex] += data[threadBlockIndex + 4];
    data[threadBlockIndex] += data[threadBlockIndex + 2];
    data[threadBlockIndex] += data[threadBlockIndex + 1];
}

/*

Loops like this one produce instruction overhead.

Completely unrolling the loop could be a good solution.
We could know what the limitation for threads per block is.
In this case, it's 1024 (2^10).
We can use this to completely unroll the loop in the kernel.
Given that we don't know the block size at compile time, we can use C++ template parameters, supported by CUDA in host and device functions.

*/
