#ifndef NON_KERNEL_IMPLEMENTATIONS
#define NON_KERNEL_IMPLEMENTATIONS

#include <string>

const unsigned int NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS = 2;

int reduceWithThrust(int *inputData, unsigned int size);
int reduceWithCPU(int *inputData, unsigned int size);

#endif
