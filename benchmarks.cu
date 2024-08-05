#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;
    int *testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    float elapsedTimeInReduction[AMOUNT_OF_IMPLEMENTATIONS];

    for (unsigned int implementationIndex = 0; implementationIndex < AMOUNT_OF_IMPLEMENTATIONS; implementationIndex++) {
        const unsigned int SAMPLE_SIZE = 20;
        float sumOfElapsedTimesInMilliseconds = 0;

        for (unsigned int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
            ReductionResult reductionResultForImplementation = reduceAndMeasureTime(
                    reduceImplementations[implementationIndex], testingData, dataSize
            );
            sumOfElapsedTimesInMilliseconds += reductionResultForImplementation.elapsedTimeInMilliseconds;
        }

        elapsedTimeInReduction[implementationIndex] = sumOfElapsedTimesInMilliseconds / SAMPLE_SIZE;
        printImplementationData(implementationIndex, elapsedTimeInReduction[implementationIndex]);
    }

    return EXIT_SUCCESS;
}

void printImplementationData(const unsigned int implementationNumber, float elapsedTimeInMilliseconds) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms\n", elapsedTimeInMilliseconds);
}