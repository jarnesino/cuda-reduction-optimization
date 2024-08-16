#ifndef REDUCE_KERNELS
#define REDUCE_KERNELS

#include <iostream>
#include <string>
#include <cuda_runtime.h>

const unsigned int NUMBER_OF_KERNEL_IMPLEMENTATIONS = 10;

const unsigned int BLOCK_SIZE = 1024;  // Hardcoded for simplicity.
const unsigned int GRID_SIZE = 16;  // Hardcoded for simplicity.

typedef void (*reduceKernelFunction)(int *inputData, int *outputData, unsigned int dataSize);

typedef unsigned int (*numberOfBlocksFunction)(unsigned int dataSize);

struct ReduceImplementationKernel {
    const int number;
    std::string name;
    reduceKernelFunction function;
    numberOfBlocksFunction numberOfBlocksFunction;
};

unsigned int unsignedMin(unsigned int a, unsigned int b);

__global__ void interleaved_addressing_with_local_memory(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void interleaved_addressing_with_divergent_branching(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void interleaved_addressing_with_bank_conflicts(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void sequential_addressing_with_idle_threads(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void first_add_during_load_with_loop_overhead(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void loop_unrolling_only_at_warp_level_iterations(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void complete_loop_unrolling_with_one_reduction(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void multiple_reduce_operations_per_thread_iteration(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void operations_for_consecutive_memory_addressing(
        int *inputData, int *outputData, unsigned int dataSize
);

__global__ void shuffle_down(
        int *inputData, int *outputData, unsigned int dataSize
);

unsigned int numberOfBlocksForStandardReduction(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithExtraStep(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithMultipleSteps(unsigned int dataSize);

unsigned int numberOfBlocksForReductionWithConsecutiveMemoryAddressing(unsigned int dataSize);

extern ReduceImplementationKernel reduceImplementationKernels[NUMBER_OF_KERNEL_IMPLEMENTATIONS];

int reduceWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
);

#endif  // REDUCE_KERNELS
