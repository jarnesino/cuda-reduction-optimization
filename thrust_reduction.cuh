#include "reduction.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

ReductionResult reduceAndMeasureTimeWithThrust(int *inputData, unsigned int size);
