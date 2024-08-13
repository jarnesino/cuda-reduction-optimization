#include "../reduction.cuh"

ReductionResult reduceAndMeasureTimeWithThrust(int *inputData, unsigned int size);
ReductionResult reduceAndMeasureTimeWithCPU(int *inputData, unsigned int size);
