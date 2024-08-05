#include "reduction.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

ReductionResult reduceWithCudaThrustLibrary(int *inputData, unsigned int size);
