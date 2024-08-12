#include "custom_reduce_implementations.cuh"

__global__ void interleaved_addressing_with_bank_conflicts(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for (unsigned int numberOfElementsReduced = 1; numberOfElementsReduced < blockSize; numberOfElementsReduced <<= 1) {
        unsigned int index = (numberOfElementsReduced * threadBlockIndex) << 1;  // This may produce memory bank conflicts.
        if (index < blockSize) {
            sharedData[index] += sharedData[index + numberOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

/*

Memory banks are divisions of the shared memory in CUDA.
Memory bank conflicts happen when two threads in the same warp try to access shared memory addresses mapping to the same memory bank.

When memory bank conflicts occur, the memory accesses are serialized.
This, of course, slows performance.

The memory bank conflicts are produced in the line when indexing with (int index = 2 * numberOfElementsReduced * threadIndex).
Accessing shared memory in indexes (2 * numberOfElementsReduced * threadIndex) and that plus (numberOfElementsReduced) may lead to conflicts.
This is because numberOfElementsReduced is growing each iteration, which can cause a thread to access another thread's memory bank (even in the same warp).
The solution would be to make the addressing sequential instead of interleaved, reducing the distance between both memory accesses in each iteration.
This helps reduce the likelihood of accessing a memory bank belonging to another thread in the same warp.

*/
