#include "reduce_kernels.cuh"

// Template parameters are needed because device functions cannot access constants, and we want it at compile time.
template<unsigned int blockSize>
__inline__ __device__ int warpReduceShuffle(int val) {
    // Shuffle from the other thread's sum variable register.
    if (warpSize >= 64) val += __shfl_down_sync(0xFFFFFFFF, val, 32);
    if (warpSize >= 32) val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    if (warpSize >= 16) val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    if (warpSize >= 8) val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    if (warpSize >= 4) val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    if (warpSize >= 2) val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    return val;
}

__global__ void shuffleDownWithLoopUnrolling(
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
    dataSize = dataSize >> 2;
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
    if (BLOCK_SIZE >= 64) {
        if (threadBlockIndex < 32) { sharedData[threadBlockIndex] += sharedData[threadBlockIndex + 32]; }
        __syncthreads();
    }

    if (threadBlockIndex < 32) {
        int sum = sharedData[threadBlockIndex];
        sum = warpReduceShuffle<BLOCK_SIZE>(sum);  // Reduce last warp
        if (threadBlockIndex == 0) outputData[blockIndex] = sum;  // Write this block's result.
    }
}

int reduceWithShuffleDownWithLoopUnrolling(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            shuffleDownWithLoopUnrolling, numberOfBlocksForReductionWithConsecutiveMemoryAddressing
    };
    return reduceWithKernel(kernel, data, dataSize);
}

/*

Loop unrolled shuffle down implementation.

*/
