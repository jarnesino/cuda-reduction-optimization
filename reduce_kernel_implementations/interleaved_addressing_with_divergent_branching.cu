#include "reduce_kernels.cuh"

__global__ void interleavedAddressingWithDivergentBranching(
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
        if (threadBlockIndex % (numberOfElementsReduced << 1) == 0) {  // This instruction produces divergent branching.
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + numberOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

int reduceWithInterleavedAddressingWithDivergentBranching(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            interleavedAddressingWithDivergentBranching, numberOfBlocksForStandardReduction
    };
    return reduceWithKernel(kernel, data, dataSize);
}

/*

A warp is a set of 32 threads inside a block, and they all share the program counter. They all execute the same instruction at the same time.
Divergent branching happens when threads in the same warp follow different execution paths.

When threads in a warp diverge, the warp serializes the execution of different paths.
Some threads in the warp are executing while others aren't.
It introduces additional overhead.

The divergent branching here is produced when the if statement checks for (threadBlockIndex % (2 * numberOfElementsReduced) == 0).
Only one in every (2 * numberOfElementsReduced) consecutive threads is running the instructions inside the if statement.
Therefore, not all threads inside the warp are running that instruction, leading to divergent branching.

*/
