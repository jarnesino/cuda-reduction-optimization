#include "reduce_kernels.cuh"

__global__ void interleavedAddressingWithLocalMemoryAndBankConflicts(
        int *inputData, int *outputData, unsigned int dataSize
) {
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;
    __syncthreads();

    // Do reduction.
    for (unsigned int numberOfElementsReduced = 1; numberOfElementsReduced < blockSize; numberOfElementsReduced <<= 1) {
        unsigned int index = (numberOfElementsReduced * threadBlockIndex) << 1;
        if (index < blockSize) {
            inputData[blockSize * blockIndex + index] += inputData[blockSize * blockIndex + index + numberOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = inputData[threadIndex];
}

int reduceWithInterleavedAddressingWithLocalMemoryAndBankConflicts(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            interleavedAddressingWithLocalMemoryAndBankConflicts, numberOfBlocksForStandardReduction
    };
    return reduceWithKernel(kernel, data, dataSize);
}
