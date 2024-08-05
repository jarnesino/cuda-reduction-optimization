#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;

    const unsigned int SAMPLE_SIZE = 20;
    float elapsedTimesInMilliseconds[NUMBER_OF_IMPLEMENTATIONS] = {};  // Default constructor fills array with zeroes.

    for (unsigned int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
        printf("Generating data for sample %d\n", sampleIndex);
        int *testingData = new int[dataSize];
        initializeTestingDataIn(testingData, dataSize);

        for (int implementationIndex = 0; implementationIndex < NUMBER_OF_IMPLEMENTATIONS; implementationIndex++) {
            ReductionResult reductionResultForImplementation = reduceAndMeasureTime(
                    reduceImplementations[implementationIndex], testingData, dataSize
            );
            elapsedTimesInMilliseconds[implementationIndex] += reductionResultForImplementation.elapsedMilliseconds;

            printf("Completed sample %d for implementation %d\n", sampleIndex, implementationIndex);
        }
    }

    for (int implementationIndex = 0; implementationIndex < NUMBER_OF_IMPLEMENTATIONS; implementationIndex++) {
        elapsedTimesInMilliseconds[implementationIndex] /= SAMPLE_SIZE;
        printImplementationData(implementationIndex, elapsedTimesInMilliseconds[implementationIndex]);
    }

    return EXIT_SUCCESS;
}

void printImplementationData(const unsigned int implementationNumber, float elapsedTimeInMilliseconds) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms\n", elapsedTimeInMilliseconds);
}