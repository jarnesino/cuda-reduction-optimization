#include "reduce_implementations.cuh"

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *inputData, int *outputData, unsigned int n) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for(unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockDim.x; amountOfElementsReduced *= 2) {
        if (threadBlockIndex % (2 * amountOfElementsReduced) == 0) {  // This instruction produces divergent branching.
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + amountOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}

/*

A warp is a set of 32 threads inside a block, and they all share the program counter. They all execute the same instruction at the same time.
Divergent branching happens when threads in the same warp follow different execution paths.
The divergent branching is produced when the if statment checks for (threadBlockIndex % (2 * amountOfElementsReduced) == 0).

When threads in a warp diverge, the warp serializes the execution of different paths.
Some threads in the warp are executing while others aren't.
It introduces additional overhead.

*/
