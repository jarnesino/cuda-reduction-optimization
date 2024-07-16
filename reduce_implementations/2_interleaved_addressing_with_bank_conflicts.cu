#include "reduce_implementations.cuh"

__global__ void reduce_using_2_interleaved_addressing_with_bank_conflicts(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for(unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockDim.x; amountOfElementsReduced *= 2) {
        int index = 2 * amountOfElementsReduced * threadIndex;
        if (index < blockDim.x) {
            sharedData[index] += sharedData[index + amountOfElementsReduced];  // This indexing is producing memory bank conflicts.
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

/*

Bank conflicts explanation here.

*/