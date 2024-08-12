#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/custom_reduce_implementations.cuh"

struct ReductionResult {
    int value;
    float elapsedMilliseconds;
};

typedef unsigned int (*numberOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        const ReduceImplementationKernel& reduceImplementation, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

#endif  // REDUCTION
