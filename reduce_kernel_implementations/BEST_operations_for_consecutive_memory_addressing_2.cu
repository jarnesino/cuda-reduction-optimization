#include <cassert>
#include "reduce_kernels.cuh"

__global__ void BESTOperationsForConsecutiveMemoryAddressing2(
        int *inputData, int *outputData, unsigned int dataSize
) {
    extern __shared__ int sharedData[];

    int4 *inputDataForConsecutiveAccessing = (int4 *) inputData;
    unsigned int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int elementsReducedByGrid = BLOCK_SIZE * gridDim.x;
    sharedData[threadIdx.x] = 0;
    while (index < (dataSize >> 2)) {
        int4 input = inputDataForConsecutiveAccessing[index];
        sharedData[threadIdx.x] += input.x + input.y + input.z + input.w;
        index += elementsReducedByGrid;
    }
    __syncthreads();

    // Do reduction in shared memory.
    int4 *sharedDataAsInt4 = (int4 *) sharedData;
    int4 &sharedData1 = sharedDataAsInt4[threadIdx.x];
    int4 &sharedData2 = sharedDataAsInt4[threadIdx.x + 128];

    if (threadIdx.x < 128) {
        sharedData[threadIdx.x] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    if (threadIdx.x < 16) {
        sharedData1 = sharedDataAsInt4[threadIdx.x];
        sharedData2 = sharedDataAsInt4[threadIdx.x + 16];

        sharedData[threadIdx.x] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    if (threadIdx.x < 2) {
        sharedData1 = sharedDataAsInt4[threadIdx.x];
        sharedData2 = sharedDataAsInt4[threadIdx.x + 2];

        sharedData[threadIdx.x] =
                sharedData1.x + sharedData1.y + sharedData1.z + sharedData1.w +
                sharedData2.x + sharedData2.y + sharedData2.z + sharedData2.w;
    }
    __syncthreads();

    // Write this block's result.
    if (threadIdx.x == 0) outputData[blockIdx.x] = sharedData[0] + sharedData[1];
}

int BESTReduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            BESTOperationsForConsecutiveMemoryAddressing2, numberOfBlocksForReductionWithConsecutiveMemoryAddressing
    };
    return reduceWithKernel(kernel, data, dataSize);
}
