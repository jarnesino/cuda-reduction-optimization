#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef unsigned int (*amountOfBlocksFunction)(const unsigned int dataSize);

void reduceAndMeasureTime(
        unsigned int implementationNumber, reduceImplementationFunction implementation,
        amountOfBlocksFunction amountOfBlocksFor, int *inputData, unsigned int dataSize,
        cudaEvent_t startEvent, cudaEvent_t stopEvent
);

void checkForCUDAErrors();

unsigned int amountOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int amountOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result);

void initializeTestingDataIn(int *data, int size);

unsigned int unsignedMin(unsigned int a, unsigned int b);

#endif  // REDUCTION
