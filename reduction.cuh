#ifndef REDUCTION
#define REDUCTION

#include <iostream>
#include "reduce_implementations/reduce_implementations.cuh"

void reduce(const int implementationNumber, reduce_implementation_function implementation, const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent);
void initializeRandomTestingDataIn(int *data, int size);

#endif // REDUCTION
