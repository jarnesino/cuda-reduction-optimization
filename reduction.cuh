#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

void reduce(const int implementationNumber, reduceImplementationFunction implementation, int* inputData, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent);
void initializeTestingDataIn(int *data, int size);

#endif // REDUCTION
