#include "reduce_kernels.cuh"

__global__ void interleavedAddressingWithGlobalMemoryAndDivergentBranching(
        int *inputData, int *outputData, unsigned int dataSize
) {
    unsigned int blockSize = blockDim.x;
    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    unsigned int threadIndex = blockIndex * blockSize + threadBlockIndex;

    // Do reduction in global memory. Causes slow access speeds.
    for (unsigned int numberOfElementsReduced = 1; numberOfElementsReduced < blockSize; numberOfElementsReduced <<= 1) {
        if (threadBlockIndex % (numberOfElementsReduced << 1) == 0) {
            inputData[threadIndex] += inputData[threadIndex + numberOfElementsReduced];
        }
        __syncthreads();
    }

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = inputData[threadIndex];
}

int reduceWithInterleavedAddressingWithGlobalMemoryAndDivergentBranching(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            interleavedAddressingWithGlobalMemoryAndDivergentBranching, numberOfBlocksForStandardReduction
    };
    return reduceWithKernel(kernel, data, dataSize);
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
