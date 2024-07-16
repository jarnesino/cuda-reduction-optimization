#ifndef REDUCE_IMPLEMENTATIONS
#define REDUCE_IMPLEMENTATIONS

#include <cuda_runtime.h>

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *input_data, int *output_data, unsigned int n);

#endif // REDUCE_IMPLEMENTATIONS
