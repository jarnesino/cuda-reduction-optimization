#include "reduction.cuh"
#include <random>

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void initializeRandomBenchmarkingDataIn(int *data, int size);

void printImplementationData(
        unsigned int implementationNumber,
        float elapsedTimeInMilliseconds,
        float timesFaster,
        float percentageOfTimeSaved
);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;

    const unsigned int SAMPLE_SIZE = 20;
    float elapsedTimesInMilliseconds[NUMBER_OF_IMPLEMENTATIONS] = {};  // Default constructor fills array with zeroes.

    int *testingData = new int[dataSize];

    for (unsigned int sampleIndex = 0; sampleIndex < SAMPLE_SIZE; sampleIndex++) {
        printf("Generating data for sample %d\n", sampleIndex);
        initializeRandomBenchmarkingDataIn(testingData, dataSize);

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

        float timesFaster = elapsedTimesInMilliseconds[0] / elapsedTimesInMilliseconds[implementationIndex];
        float percentageOfTimeSaved = (
                100.0f
                * (elapsedTimesInMilliseconds[0] - elapsedTimesInMilliseconds[implementationIndex])
                / elapsedTimesInMilliseconds[0]
        );

        printImplementationData(
                implementationIndex, elapsedTimesInMilliseconds[implementationIndex], timesFaster, percentageOfTimeSaved
        );
    }

    return EXIT_SUCCESS;
}

void printImplementationData(
        const unsigned int implementationNumber,
        float elapsedTimeInMilliseconds,
        float timesFaster,
        float percentageOfTimeSaved
) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Times faster than base implementation: %f\n", timesFaster);
    printf("\t Time saved compared with base implementation: %f %%\n", percentageOfTimeSaved);
}

void initializeRandomBenchmarkingDataIn(int *data, int size) {
    std::random_device randomSeed;
    std::mt19937 gen(randomSeed());
    std::uniform_int_distribution<> uniformDistribution(-2000, 2000);

    for (unsigned int index = 0; index < size; ++index) {
        int randomNumber = uniformDistribution(gen);
        data[index] = randomNumber;
    }
}