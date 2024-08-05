#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;
    int *testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    for (int implementationIndex = 0; implementationIndex < 9; implementationIndex++) {
        ReductionResult reductionResult = reduceAndMeasureTime(
                reduceImplementations[implementationIndex], testingData, dataSize
        );
        printImplementationData(implementationIndex, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    }

    return EXIT_SUCCESS;
}

void printImplementationData(const unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Reduction result: %d\n", result);
}