#include <string>

const unsigned int NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS = 2;

typedef int (*reduceNonKernelFunction)(int *inputData, unsigned int size);

struct ReduceNonKernelImplementation {
    const int number;
    std::string name;
    reduceNonKernelFunction function;
};

int reduceWithThrust(int *inputData, unsigned int size);
int reduceWithCPU(int *inputData, unsigned int size);

extern ReduceNonKernelImplementation reduceNonKernelImplementations[NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS];
