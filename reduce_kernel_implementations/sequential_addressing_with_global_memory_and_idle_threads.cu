#include "reduce_kernels.cuh"

__global__ void sequentialAddressingWithGlobalMemoryAndIdleThreads(
        int *inputData, int *outputData, unsigned int dataSize
) {
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    __syncthreads();

    // Do reduction in global memory. Causes slow access speeds.
    for (
            unsigned int numberOfElementsToReduce = blockSize >> 1;
            numberOfElementsToReduce > 0;
            numberOfElementsToReduce >>= 1
            ) {
        if (threadBlockIndex <
            numberOfElementsToReduce) {  // This if statement makes many threads idle threads in each iteration.
            inputData[threadIndex] += inputData[threadIndex + numberOfElementsToReduce];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = inputData[threadIndex];
}

int reduceWithSequentialAddressingWithGlobalMemoryAndIdleThreads(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {sequentialAddressingWithGlobalMemoryAndIdleThreads, numberOfBlocksForStandardReduction};
    return reduceWithKernel(kernel, data, dataSize);
}
