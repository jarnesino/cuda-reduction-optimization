#include "reduce_kernels.cuh"

__global__ void first_add_during_load_with_loop_overhead(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize * 2 + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + blockSize];
    __syncthreads();

    // Do reduction in shared memory.
    for (
            unsigned int numberOfElementsToReduce = blockSize >> 1;
            numberOfElementsToReduce > 0;
            numberOfElementsToReduce >>= 1
            ) {  // This loop produces instruction overhead.
        if (threadBlockIndex < numberOfElementsToReduce) {
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + numberOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

int reduceWithFirstAddDuringLoadWithLoopOverhead(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            first_add_during_load_with_loop_overhead, numberOfBlocksForReductionWithExtraStep
    };
    return reduceWithKernel(kernel, data, dataSize);
}

/*

Loops like this one produce instruction overhead.

Unrolling the loop could be a good solution.
We know all threads in a warp are SIMD-synchronous. Then, we could safely unroll the last warp, when (numberOfElementsToReduce <= 32).
The if statement is also unnecessary, because it doesn't save work between threads in a warp.

*/
