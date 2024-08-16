#include <string>
#include "../reduction.cuh"

const unsigned int NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS = 2;

typedef ReductionResult (*reduceFunction)(int *inputData, unsigned int size);

struct ReduceImplementation {
    const int number;
    std::string name;
    reduceFunction function;
};

ReductionResult reduceAndMeasureTimeWithThrust(int *inputData, unsigned int size);
ReductionResult reduceAndMeasureTimeWithCPU(int *inputData, unsigned int size);

extern ReduceImplementation reduceNonKernelImplementations[NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS];
