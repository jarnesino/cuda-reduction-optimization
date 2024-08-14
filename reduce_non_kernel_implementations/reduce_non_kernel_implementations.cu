#include "reduce_non_kernel_implementations.cuh"


ReduceImplementation reduceNonKernelImplementations[NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS] = {
        {1, "CPU sequential", reduceAndMeasureTimeWithCPU},
        {2, "CUDA Thrust", reduceAndMeasureTimeWithThrust}
};

