#include "reduce_implementations.cuh"

__global__ void reduce_using_0_interleaved_addressing_with_local_memory(
        int *inputData, int *outputData, unsigned int dataSize
) {
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    __syncthreads();

    // Do reduction in global memory. Causes slow access speeds.
    for (unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockSize; amountOfElementsReduced *= 2) {
        if (threadBlockIndex % (2 * amountOfElementsReduced) == 0) {
            inputData[threadIndex] += inputData[threadIndex + amountOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = inputData[threadIndex];
}

/*

There's three (relevant) types of memory in CUDA.
- Global memory : accessible by all threads in all blocks.
- Shared memory: accessible by all threads in a single block.
- Local memory: accessible by a single thread.
The access speeds in global memory are of slower bandwidth and higher latency compared to the ones in shared memory.

The pointer (inputData) leads to an array in global memory.
Since we access and modify the array in global memory, the access speed is unnecessarily limited, making the kernel execute slower than necessary.

*/
