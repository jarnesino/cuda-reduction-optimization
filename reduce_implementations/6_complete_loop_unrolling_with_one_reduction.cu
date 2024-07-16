#include "reduce_implementations.cuh"

template <unsigned int blockSize> __device__ void warpReduce(volatile int* sharedData, int threadIndex);

__global__ void reduce_using_6_complete_loop_unrolling_with_one_reduction(int *inputData, int *outputData, unsigned int dataSize) {
    const unsigned int blockSize = 1024;  // Hardcoded for simplicity.
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + blockDim.x];
    __syncthreads();

    // Do reduction in shared memory.
    // This is still doing only one reduction per thread iteration.
    if (blockSize >= 1024) {
        if (threadIndex < 512) { sharedData[threadIndex] += sharedData[threadIndex + + 512]; }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIndex < 256) { sharedData[threadIndex] += sharedData[threadIndex + + 256]; }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIndex < 128) { sharedData[threadIndex] += sharedData[threadIndex + + 128]; }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIndex < 64) { sharedData[threadIndex] += sharedData[threadIndex + + 64]; }
        __syncthreads();
    }
    if (threadIndex > 32) warpReduce<blockSize>(sharedData, threadIndex);

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sharedData, int threadIndex) {
    if (blockSize >= 64) sharedData[threadIndex] += sharedData[threadIndex + 32];
    if (blockSize >= 32) sharedData[threadIndex] += sharedData[threadIndex + 16];
    if (blockSize >= 16) sharedData[threadIndex] += sharedData[threadIndex + 8];
    if (blockSize >= 8) sharedData[threadIndex] += sharedData[threadIndex + 4];
    if (blockSize >= 4) sharedData[threadIndex] += sharedData[threadIndex + 2];
    if (blockSize >= 2) sharedData[threadIndex] += sharedData[threadIndex + 1];
}

/*

************************************************************************************************

*/
