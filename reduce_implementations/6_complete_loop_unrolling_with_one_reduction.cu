#include "reduce_implementations.cuh"
#include "../reduction.cuh"

template<unsigned int blockSize>
__device__ void warpReduce(volatile int *sharedData, unsigned int threadBlockIndex);

__global__ void reduce_using_6_complete_loop_unrolling_with_one_reduction(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * BLOCK_SIZE * 2 + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + BLOCK_SIZE];
    __syncthreads();

    // Do reduction in shared memory.
    if (BLOCK_SIZE >= 1024) {
        if (threadBlockIndex < 512) { sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 512]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512) {
        if (threadBlockIndex < 256) { sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 256]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (threadBlockIndex < 128) { sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 128]; }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (threadBlockIndex < 64) { sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 64]; }
        __syncthreads();
    }
    if (threadBlockIndex < 32) warpReduce<BLOCK_SIZE>(sharedData, threadBlockIndex);

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

// Template parameters are needed because device functions cannot access constants, and we want it at compile time.
template<unsigned int blockSize>
__device__ void warpReduce(volatile int *data, const unsigned int threadBlockIndex) {
    if (blockSize >= 64) data[threadBlockIndex] += data[threadBlockIndex + 32];
    if (blockSize >= 32) data[threadBlockIndex] += data[threadBlockIndex + 16];
    if (blockSize >= 16) data[threadBlockIndex] += data[threadBlockIndex + 8];
    if (blockSize >= 8) data[threadBlockIndex] += data[threadBlockIndex + 4];
    if (blockSize >= 4) data[threadBlockIndex] += data[threadBlockIndex + 2];
    if (blockSize >= 2) data[threadBlockIndex] += data[threadBlockIndex + 1];
}

/*

Many kernels can be launched, but a GPU has a limited number of blocks in a grid.
When that number (e.g. 16) is reached, the rest of the kernel executions are serialized.
Knowing this, we can use it to our advantage and reduce the data to a grid-sized array before executing the grid-wide reduction.

*/
