#include "reduce_implementations.cuh"

__inline__ __device__ int warpReduceSum(int val);

__global__ void shuffle_down(
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
        sum = warpReduceSum(sum);  // Reduce last warp
        if (threadBlockIndex == 0) {

            outputData[blockIndex] = sum;  // Write this block's result.
        }
    }
}

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
