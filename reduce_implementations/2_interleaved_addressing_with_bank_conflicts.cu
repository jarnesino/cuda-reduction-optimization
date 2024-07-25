#include "reduce_implementations.cuh"

__global__ void reduce_using_2_interleaved_addressing_with_bank_conflicts(
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
    for (unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockSize; amountOfElementsReduced <<= 1) {
        unsigned int index = (amountOfElementsReduced * threadBlockIndex) << 1;  // This may produce memory bank conflicts.
        if (index < blockSize) {
            sharedData[index] += sharedData[index + amountOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

/*

Memory banks are divisions of the shared memory in CUDA.
Memory bank conflicts happen when two threads in the same warp try to access shared memory addresses mapping to the same memory bank.

When memory bank conflicts occur, the memory accesses are serialized.
This, of course, slows performance.

The memory bank conflicts are produced in the line when indexing with (int index = 2 * amountOfElementsReduced * threadIndex).
Accessing shared memory in indexes (2 * amountOfElementsReduced * threadIndex) and that plus (amountOfElementsReduced) may lead to conflicts.
This is because amountOfElementsReduced is growing each iteration, which can cause a thread to access another thread's memory bank (even in the same warp).
The solution would be to make the addressing sequential instead of interleaved, reducing the distance between both memory accesses in each iteration.
This helps reduce the likelihood of accessing a memory bank belonging to another thread in the same warp.

*/
