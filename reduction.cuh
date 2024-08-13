#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_kernel_implementations/reduce_kernels.cuh"

struct ReductionResult {
    int value;
    float elapsedMilliseconds;
};

typedef unsigned int (*numberOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        const ReduceImplementationKernel& reduceKernel, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

#endif  // REDUCTION
