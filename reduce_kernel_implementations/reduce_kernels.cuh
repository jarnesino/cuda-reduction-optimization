#ifndef REDUCE_KERNELS
#define REDUCE_KERNELS

#include <iostream>
#include <string>
#include <cuda_runtime.h>

const unsigned int NUMBER_OF_KERNEL_IMPLEMENTATIONS = 16;

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef void (*reduceKernelFunction)(int *inputData, int *outputData, unsigned int dataSize);

typedef unsigned int (*numberOfBlocksFunction)(unsigned int dataSize);

struct ReduceImplementationKernel {
    reduceKernelFunction function;
    numberOfBlocksFunction numberOfBlocksFunction;
};

unsigned int unsignedMin(unsigned int a, unsigned int b);

int reduceWithInterleavedAddressingWithGlobalMemoryAndDivergentBranching(int *data, unsigned int dataSize);

int reduceWithInterleavedAddressingWithDivergentBranching(int *data, unsigned int dataSize);

int reduceWithInterleavedAddressingWithGlobalMemoryAndGoodBranching(int *data, unsigned int dataSize);

int reduceWithInterleavedAddressingWithBankConflicts(int *data, unsigned int dataSize);

int reduceWithSequentialAddressingWithGlobalMemoryAndIdleThreads(int *data, unsigned int dataSize);

int reduceWithSequentialAddressingWithIdleThreads(int *data, unsigned int dataSize);

int reduceWithFirstAddDuringLoadWithLoopOverhead(int *data, unsigned int dataSize);

int reduceWithLoopUnrollingOnlyAtWarpLevelIterations(int *data, unsigned int dataSize);

int reduceWithCompleteLoopUnrollingWithOneReduction(int *data, unsigned int dataSize);

int reduceWithMultipleReduceOperationsPerThreadIteration(int *data, unsigned int dataSize);

int reduceWithOperationsForConsecutiveMemoryAddressing(int *data, unsigned int dataSize);

int reduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize);

int reduceWithShuffleDown(int *data, unsigned int dataSize);

int reduceWithShuffleDownWithLoopUnrolling(int *data, unsigned int dataSize);

int BESTReduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize);

int BEST2ReduceWithOperationsForConsecutiveMemoryAddressing2(int *data, unsigned int dataSize);

unsigned int numberOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

int reduceWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
);

#endif  // REDUCE_KERNELS
