#include "reduce_implementations.cuh"
#include "../reduction.cuh"

__global__ void reduce_using_0_interleaved_addressing_with_local_memory(int *inputData, int *outputData, unsigned int dataSize) {
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int blockSize = blockDim.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    __syncthreads();

    // Do reduction in shared memory.
    for(unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockSize; amountOfElementsReduced *= 2) {
        if (threadBlockIndex % (2 * amountOfElementsReduced) == 0) {
            inputData[threadIndex] += inputData[threadIndex + amountOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = inputData[threadIndex];
}

/*



*/
