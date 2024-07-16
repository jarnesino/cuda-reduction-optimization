#ifndef REDUCE_IMPLEMENTATIONS
#define REDUCE_IMPLEMENTATIONS

#include <cuda_runtime.h>

typedef void (*reduce_implementation_function)(int *inputData, int *outputData, unsigned int dataSize);

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *inputData, int *outputData, unsigned int dataSize);
__global__ void reduce_using_2_interleaved_addressing_with_bank_conflicts(int *inputData, int *outputData, unsigned int dataSize);

#endif // REDUCE_IMPLEMENTATIONS
