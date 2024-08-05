#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

struct ReductionResult {
    int value;
    float elapsedMilliseconds;
};

typedef unsigned int (*numberOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        ReduceImplementation reduceImplementation, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

void initializeTestingDataIn(int *data, int size);

void fillDataWith1s(int *data, unsigned int size);

#endif  // REDUCTION
