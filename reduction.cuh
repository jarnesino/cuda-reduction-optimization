#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

struct ReductionResult {
    int value;
    float elapsedTimeInMilliseconds;
};

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef unsigned int (*numberOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        reduceImplementationFunction implementation,
        numberOfBlocksFunction numberOfBlocksFor, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

unsigned int numberOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

void initializeTestingDataIn(int *data, int size);

void fillDataWith1s(int *data, int size);

unsigned int unsignedMin(unsigned int a, unsigned int b);

#endif  // REDUCTION
