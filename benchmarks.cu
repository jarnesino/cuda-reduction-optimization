#include "reduction.cuh"
#include <random>

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void measureAndSumElapsedTimes(unsigned int dataSize, unsigned int SAMPLE_SIZE, float *elapsedTimesInMilliseconds);

void printBenchmarkStats(unsigned int SAMPLE_SIZE, float *sumOfElapsedTimesInMilliseconds);

void initializeRandomBenchmarkingDataIn(int *data, unsigned int size);

void printImplementationData(
        unsigned int implementationNumber,
        float elapsedTimeInMilliseconds,
        float timesFaster,
        float percentageOfTimeSaved
);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;

    const unsigned int SAMPLE_SIZE = 5;
    float sumOfElapsedTimesInMilliseconds[NUMBER_OF_IMPLEMENTATIONS] = {};  // Default constructor fills array with zeroes.

    measureAndSumElapsedTimes(dataSize, SAMPLE_SIZE, sumOfElapsedTimesInMilliseconds);
    printBenchmarkStats(SAMPLE_SIZE, sumOfElapsedTimesInMilliseconds);

    return EXIT_SUCCESS;
}

/* *************** AUXILIARY *************** */

void measureAndSumElapsedTimes(
        const unsigned int dataSize, const unsigned int SAMPLE_SIZE, float *elapsedTimesInMilliseconds
) {
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
}

void printBenchmarkStats(const unsigned int SAMPLE_SIZE, float *sumOfElapsedTimesInMilliseconds) {
    for (int implementationIndex = 0; implementationIndex < NUMBER_OF_IMPLEMENTATIONS; implementationIndex++) {
        sumOfElapsedTimesInMilliseconds[implementationIndex] /= (float)SAMPLE_SIZE;

        float timesFaster = sumOfElapsedTimesInMilliseconds[0] / sumOfElapsedTimesInMilliseconds[implementationIndex];
        float percentageOfTimeSaved = (
                100.0f
                * (sumOfElapsedTimesInMilliseconds[0] - sumOfElapsedTimesInMilliseconds[implementationIndex])
                / sumOfElapsedTimesInMilliseconds[0]
        );

        printImplementationData(
                implementationIndex,
                sumOfElapsedTimesInMilliseconds[implementationIndex],
                timesFaster,
                percentageOfTimeSaved
        );
    }
}

void initializeRandomBenchmarkingDataIn(int *data, unsigned int size) {
    std::random_device randomSeed;
    std::mt19937 gen(randomSeed());
    std::uniform_int_distribution<> uniformDistribution(-2000, 2000);

    for (unsigned int index = 0; index < size; ++index) {
        int randomNumber = uniformDistribution(gen);
        data[index] = randomNumber;
    }
}

void printImplementationData(
        const unsigned int implementationNumber,
        float elapsedTimeInMilliseconds,
        float timesFaster,
        float percentageOfTimeSaved
) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Times faster than base implementation: %f", timesFaster);
    printf("\t Time saved compared with base implementation: %f %%\n", percentageOfTimeSaved);
}
