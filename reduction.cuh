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

typedef unsigned int (*amountOfBlocksFunction)(const unsigned int dataSize);

ReductionResult reduceAndMeasureTime(
        reduceImplementationFunction implementation,
        amountOfBlocksFunction amountOfBlocksFor, int *inputData, unsigned int dataSize
);

void checkForCUDAErrors();

unsigned int amountOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

void initializeTestingDataIn(int *data, int size);

void fillDataWith1s(int *data, int size);

unsigned int unsignedMin(unsigned int a, unsigned int b);

#endif  // REDUCTION
