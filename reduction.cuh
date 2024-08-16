#ifndef REDUCTION
#define REDUCTION

#include "reduce_kernel_implementations/reduce_kernels.cuh"

struct ReductionResult {
    int value;
    float elapsedMilliseconds;
};

ReductionResult reduceAndMeasureTime(
        const ReduceImplementationKernel& reduceKernel, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

#endif  // REDUCTION
