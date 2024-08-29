#include <cassert>
#include "reduce_kernels.cuh"

__global__ void BEST2OperationsForConsecutiveMemoryAddressing2(
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

    int val = sharedData[threadIdx.x];
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    // Write this block's result.
    if (threadIdx.x == 0) outputData[blockIdx.x] = val;
}

int BEST2ReduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize) {
    ReduceImplementationKernel kernel = {
            BEST2OperationsForConsecutiveMemoryAddressing2, numberOfBlocksForReductionWithConsecutiveMemoryAddressing
    };
    return reduceWithKernel(kernel, data, dataSize);
}
