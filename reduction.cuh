#ifndef REDUCTION
#define REDUCTION

#include <string>
#include "reduce_kernel_implementations/reduce_kernels.cuh"
#include "reduce_non_kernel_implementations/reduce_non_kernel_implementations.cuh"

const unsigned int NUMBER_OF_IMPLEMENTATIONS = NUMBER_OF_KERNEL_IMPLEMENTATIONS + NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS;

typedef int (*reduceFunction)(int *inputData, unsigned int size);

struct ReduceImplementation {
    const int number;
    std::string name;
    reduceFunction function;
};

extern ReduceImplementation reduceImplementations[NUMBER_OF_IMPLEMENTATIONS];

#endif
