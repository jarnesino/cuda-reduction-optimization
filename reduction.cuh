#ifndef REDUCTION
#define REDUCTION

#include "reduce_kernel_implementations/reduce_kernels.cuh"
#include "reduce_non_kernel_implementations/reduce_non_kernel_implementations.cuh"

struct TimedReductionResult {
    int value;
    float elapsedMilliseconds;
};

TimedReductionResult reduceAndMeasureTimeWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
);

TimedReductionResult reduceAndMeasureTimeWithNonKernel(
        const ReduceNonKernelImplementation &implementation, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

#endif  // REDUCTION
