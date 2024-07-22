#include "reduce_implementations.cuh"
#include "../reduction.cuh"

__global__ void reduce_using_4_first_add_during_load_with_loop_overhead(int *inputData, int *outputData, unsigned int dataSize) {
    extern __shared__ int sharedData[];

    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * BLOCK_SIZE * 2 + threadBlockIndex;
    sharedData[threadBlockIndex] = inputData[threadIndex] + inputData[threadIndex + BLOCK_SIZE];
    __syncthreads();

    // Do reduction in shared memory.
    for (unsigned int amountOfElementsToReduce = BLOCK_SIZE / 2; amountOfElementsToReduce > 0; amountOfElementsToReduce >>= 1) {  // This loop produces instruction overhead.
        if (threadBlockIndex < amountOfElementsToReduce) {
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + amountOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0];
}

/*

Loops like this one produce instruction overhead.

Unrolling the loop could be a good solution.
We know all threads in a warp are SIMD-synchronous. Then, we could safely unroll the last warp, when (amountOfElementsToReduce <= 32).
The if statement is also unnecessary, because it doesn't save work between threads in a warp.

*/
