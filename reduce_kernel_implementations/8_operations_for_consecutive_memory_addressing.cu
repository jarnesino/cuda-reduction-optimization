#include "reduce_kernels.cuh"

// Template parameters are needed because device functions cannot access constants, and we want it at compile time.
template<unsigned int blockSize>
__device__ void warpReduce(volatile int *data, unsigned int threadBlockIndex) {
    if (blockSize >= 64) data[threadBlockIndex] += data[threadBlockIndex + 32];
    if (blockSize >= 32) data[threadBlockIndex] += data[threadBlockIndex + 16];
    if (blockSize >= 16) data[threadBlockIndex] += data[threadBlockIndex + 8];
    if (blockSize >= 8) data[threadBlockIndex] += data[threadBlockIndex + 4];
    if (blockSize >= 4) data[threadBlockIndex] += data[threadBlockIndex + 2];
    if (blockSize >= 2) data[threadBlockIndex] += data[threadBlockIndex + 1];
}

__global__ void operations_for_consecutive_memory_addressing(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    int4 *inputDataForConsecutiveAccessing = (int4 *) inputData;
    unsigned int elementsReducedByBlock = BLOCK_SIZE;
    unsigned int index = blockIndex * elementsReducedByBlock + threadBlockIndex;
    unsigned int elementsReducedByGrid = elementsReducedByBlock * gridDim.x;
    sharedData[threadBlockIndex] = 0;
    while (index < (dataSize >> 2)) {
        int4 input = inputDataForConsecutiveAccessing[index];
        sharedData[threadBlockIndex] += input.x + input.y + input.z + input.w;
        index += elementsReducedByGrid;
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

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

int reduceWithOperationsForConsecutiveMemoryAddressing(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            operations_for_consecutive_memory_addressing, numberOfBlocksForReductionWithConsecutiveMemoryAddressing
    };
    return reduceWithKernel(kernel, data, dataSize);
}

/*

This operates over four consecutive memory addresses with operations that are optimized for such usage.
However, memory is still slower than registers. What if we could access other thread's registers instead of shared
memory?

*/
