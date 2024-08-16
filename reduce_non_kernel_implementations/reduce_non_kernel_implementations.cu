#include "reduce_non_kernel_implementations.cuh"


ReduceNonKernelImplementation reduceNonKernelImplementations[NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS] = {
        {1, "CPU sequential", reduceWithCPU},
        {2, "CUDA Thrust",    reduceWithThrust}
};

