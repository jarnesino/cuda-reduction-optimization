#ifndef REDUCE_IMPLEMENTATIONS
#define REDUCE_IMPLEMENTATIONS

#include <cuda_runtime.h>

typedef void (*reduceImplementationFunction)(int *inputData, int *outputData, unsigned int dataSize);

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *inputData, int *outputData, unsigned int dataSize);
__global__ void reduce_using_2_interleaved_addressing_with_bank_conflicts(int *inputData, int *outputData, unsigned int dataSize);
__global__ void reduce_using_3_sequential_addressing_with_idle_threads(int *inputData, int *outputData, unsigned int dataSize);
__global__ void reduce_using_4_first_add_during_load_with_loop_overhead(int *inputData, int *outputData, unsigned int dataSize);
__global__ void reduce_using_5_loop_unrolling_only_at_warp_level_iterations(int *inputData, int *outputData, unsigned int dataSize);

#endif // REDUCE_IMPLEMENTATIONS
