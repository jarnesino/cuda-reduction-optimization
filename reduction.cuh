#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

struct ReductionResult {
    int value;
    float elapsedTimeInMilliseconds;
};

typedef unsigned int (*numberOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        reduceImplementationFunction implementation,
        numberOfBlocksFunction numberOfBlocksFor, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

void initializeTestingDataIn(int *data, int size);

void fillDataWith1s(int *data, int size);

#endif  // REDUCTION
