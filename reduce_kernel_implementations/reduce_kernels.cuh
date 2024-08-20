#ifndef REDUCE_KERNELS
#define REDUCE_KERNELS

#include <iostream>
#include <string>
#include <cuda_runtime.h>

const unsigned int NUMBER_OF_KERNEL_IMPLEMENTATIONS = 10;

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef void (*reduceKernelFunction)(int *inputData, int *outputData, unsigned int dataSize);

typedef unsigned int (*numberOfBlocksFunction)(unsigned int dataSize);

struct ReduceImplementationKernel {
    reduceKernelFunction function;
    numberOfBlocksFunction numberOfBlocksFunction;
};

unsigned int unsignedMin(unsigned int a, unsigned int b);

int reduceWithInterleavedAddressingWithLocalMemory(int *data, unsigned int dataSize);

int reduceWithInterleavedAddressingWithDivergentBranching(int *data, unsigned int dataSize);

int reduceWithInterleavedAddressingWithBankConflicts(int *data, unsigned int dataSize);

int reduceWithSequentialAddressingWithIdleThreads(int *data, unsigned int dataSize);

int reduceWithFirstAddDuringLoadWithLoopOverhead(int *data, unsigned int dataSize);

int reduceWithLoopUnrollingOnlyAtWarpLevelIterations(int *data, unsigned int dataSize);

int reduceWithCompleteLoopUnrollingWithOneReduction(int *data, unsigned int dataSize);

int reduceWithMultipleReduceOperationsPerThreadIteration(int *data, unsigned int dataSize);

int reduceWithOperationsForConsecutiveMemoryAddressing(int *data, unsigned int dataSize);

int reduceWithShuffleDown(int *data, unsigned int dataSize);

unsigned int numberOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

int reduceWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
);

#endif  // REDUCE_KERNELS
