#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include <iomanip>
#include "reduce_implementations/reduce_implementations.cuh"

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef int (*amountOfBlocksFunction)(const int dataSize);

void reduce(const int implementationNumber, reduceImplementationFunction implementation, amountOfBlocksFunction amountOfBlocksFor, int *inputData, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent);
int amountOfBlocksForStandardReduction(const int dataSize);
int amountOfBlocksForReductionWithExtraStep(const int dataSize);
int amountOfBlocksForReductionWithMultipleSteps(const int dataSize);
void printImplementationData(const int implementationNumber, float elapsedTimeInMilliseconds, int result);
void initializeTestingDataIn(int *data, int size);

#endif // REDUCTION
