#include <cassert>
#include "reduce_kernels.cuh"

__global__ void operationsForConsecutiveMemoryAddressing2(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    unsigned int blockIndex = blockIdx.x;
    unsigned int threadBlockIndex = threadIdx.x;
    int4 *inputDataForConsecutiveAccessing = (int4 *) inputData;
    unsigned int elementsReducedByBlock = BLOCK_SIZE;
    unsigned int index = blockIndex * elementsReducedByBlock + threadBlockIndex;
    unsigned int elementsReducedByGrid = elementsReducedByBlock * gridDim.x;
    sharedData[threadBlockIndex] = 0;
    while (index < (dataSize >> 2)) {
        int4 input = inputDataForConsecutiveAccessing[index];
        sharedData[threadBlockIndex] += input.x + input.y + input.z + input.w;
        index += elementsReducedByGrid;
    }
    __syncthreads();

    // Do reduction in shared memory.
    assert(BLOCK_SIZE == 1024);  // Assuming BLOCK_SIZE == 1024.

    int4 *sharedDataAsInt4 = (int4 *) sharedData;
    int4 &sharedData1 = sharedDataAsInt4[threadBlockIndex];
    int4 &sharedData2 = sharedDataAsInt4[threadBlockIndex + 128];

    if (threadBlockIndex < 128) {
        sharedData[threadBlockIndex] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    if (threadBlockIndex < 16) {
        sharedData1 = sharedDataAsInt4[threadBlockIndex];
        sharedData2 = sharedDataAsInt4[threadBlockIndex + 16];

        sharedData[threadBlockIndex] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    if (threadBlockIndex < 2) {
        sharedData1 = sharedDataAsInt4[threadBlockIndex];
        sharedData2 = sharedDataAsInt4[threadBlockIndex + 2];

        sharedData[threadBlockIndex] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    // Write this block's result.
    if (threadBlockIndex == 0) outputData[blockIndex] = sharedData[0] + sharedData[1];
}

int reduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            operationsForConsecutiveMemoryAddressing2, numberOfBlocksForReductionWithConsecutiveMemoryAddressing
    };
    return reduceWithKernel(kernel, data, dataSize);
}
