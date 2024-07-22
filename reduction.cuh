#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.

void reduce(const int implementationNumber, reduceImplementationFunction implementation, const int reductionsPerIteration, int *inputData, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent);
int amountOfBlocksForReduction(const int dataSize, const int threadsPerBlock, const int blockSizedChunksReducedPerBlock);
void printImplementationData(const int implementationNumber, float elapsedTimeInMilliseconds, int result);
void initializeTestingDataIn(int *data, int size);

#endif // REDUCTION
