#include <cuda_runtime.h>

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *inputData, int *outputData, unsigned int n) {
    extern __shared__ int sharedData[];

    // Load one element from global to shared memory in each thread.
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[threadBlockIndex] = inputData[threadIndex];
    __syncthreads();

    // Do reduction in shared memory.
    for(unsigned int amountOfElementsReduced = 1; amountOfElementsReduced < blockDim.x; amountOfElementsReduced *= 2) {
        if (threadBlockIndex % (2 * amountOfElementsReduced) == 0) {
            sharedData[threadBlockIndex] += sharedData[threadBlockIndex + amountOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result in shared memory.
    if (threadBlockIndex == 0) outputData[blockIdx.x] = sharedData[0];
}
