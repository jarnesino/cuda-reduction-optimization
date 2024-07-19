#include "reduce_implementations.cuh"
#include "../reduction.cuh"

template <unsigned int blockSize> __device__ void warpReduce(volatile int* sharedData, int threadIndex);

__global__ void reduce_using_7_multiple_reduce_operations_per_thread_iteration(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;
    sharedData[threadBlockIndex] = 0;
    while (threadIndex < dataSize) {
        sharedData[threadBlockIndex] += inputData[threadIndex] + inputData[threadIndex + BLOCK_SIZE];
        threadIndex += gridSize;
    }
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
