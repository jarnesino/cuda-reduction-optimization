#include "time.cuh"
#include "../data/data.cuh"
#include <string>

void measureElapsedTimes(
        unsigned int dataSize,
        unsigned int SAMPLE_SIZE,
        float *elapsedTimesInMilliseconds
);

void printBenchmarkStats(
        unsigned int logDataSize,
        unsigned int SAMPLE_SIZE,
        const float *elapsedTimesInMilliseconds
);

void printImplementationData(
        unsigned int implementationNumber,
        const std::string &implementationName,
        float elapsedTimeInMilliseconds,
        float timesFasterVsCPU,
        float percentageOfTimeSavedVsCPU,
        float timesFasterVsBaseGPU,
        float percentageOfTimeSavedVsBaseGPU
);

int main() {
    const unsigned int SAMPLE_SIZE = 50;

    // Constructor fills array with zeroes.
    float elapsedTimesInMilliseconds[NUMBER_OF_IMPLEMENTATIONS] = {};

    const unsigned int logDataSizes[3] = {10, 20, 30};
    for (unsigned int logDataSize: logDataSizes) {
        const unsigned int dataSize = 1 << logDataSize;

        measureElapsedTimes(dataSize, SAMPLE_SIZE, elapsedTimesInMilliseconds);
        printBenchmarkStats(logDataSize, SAMPLE_SIZE, elapsedTimesInMilliseconds);
    }

    return EXIT_SUCCESS;
}

/* *************** AUXILIARY *************** */

void measureElapsedTimes(
        const unsigned int dataSize,
        const unsigned int SAMPLE_SIZE,
        float *elapsedTimesInMilliseconds
) {
    int *testingData = new int[dataSize];

    for (unsigned int sampleNumber = 1; sampleNumber <= SAMPLE_SIZE; sampleNumber++) {
        printf("Generating data for sample %d\n", sampleNumber);
        initializeRandomDataAndGetSumIn(testingData, dataSize);

        for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++) {
            TimedReductionResult reductionResultForImplementation = reduceAndMeasureTime(
                    reduceImplementations[index], testingData, dataSize
            );
            elapsedTimesInMilliseconds[index] += reductionResultForImplementation.elapsedMilliseconds;

            printf(
                    "Completed sample %d for implementation %d of %d\n",
                    sampleNumber,
                    reduceImplementations[index].number,
                    NUMBER_OF_IMPLEMENTATIONS
            );
        }
    }

    for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++)
        elapsedTimesInMilliseconds[index] /= (float) SAMPLE_SIZE;
}

void printBenchmarkStats(
        const unsigned int logDataSize,
        const unsigned int SAMPLE_SIZE,
        const float *elapsedTimesInMilliseconds
) {
    printf(
            "****************** LOG DATA SIZE: %d ****************** SAMPLE SIZE: %d ******************\n",
            logDataSize, SAMPLE_SIZE
    );

    float timesFasterVsCPU;
    float percentageOfTimeSavedVsCPU;
    float timesFasterVsBaseGPU;
    float percentageOfTimeSavedVsBaseGPU;

    const float elapsedTimeForSequentialCPUImplementation = elapsedTimesInMilliseconds[0];
    const float elapsedTimeForBaseGPUImplementation = elapsedTimesInMilliseconds[1];
    for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++) {
        timesFasterVsCPU = elapsedTimeForSequentialCPUImplementation / elapsedTimesInMilliseconds[index];
        percentageOfTimeSavedVsCPU = (
                100.0f
                * (elapsedTimeForSequentialCPUImplementation - elapsedTimesInMilliseconds[index])
                / elapsedTimeForSequentialCPUImplementation
        );
        timesFasterVsBaseGPU = elapsedTimeForBaseGPUImplementation / elapsedTimesInMilliseconds[index];
        percentageOfTimeSavedVsBaseGPU = (
                100.0f
                * (elapsedTimeForBaseGPUImplementation - elapsedTimesInMilliseconds[index])
                / elapsedTimeForBaseGPUImplementation
        );

        printImplementationData(
                reduceImplementations[index].number,
                reduceImplementations[index].name,
                elapsedTimesInMilliseconds[index],
                timesFasterVsCPU,
                percentageOfTimeSavedVsCPU,
                timesFasterVsBaseGPU,
                percentageOfTimeSavedVsBaseGPU
        );
    }

    printf("*****************************************************************************************\n");
}

void printImplementationData(
        const unsigned int implementationNumber,
        const std::string &implementationName,
        float elapsedTimeInMilliseconds,
        float timesFasterVsCPU,
        float percentageOfTimeSavedVsCPU,
        float timesFasterVsBaseGPU,
        float percentageOfTimeSavedVsBaseGPU
) {
    printf("Implementation: %d - ", implementationNumber);
    std::cout << implementationName << "\n";
    printf("\t Time: %f ms\n", elapsedTimeInMilliseconds);
    printf(
            "\t Against CPU implementation: %f times as fast | %f%% time saved\n",
            timesFasterVsCPU,
            percentageOfTimeSavedVsCPU
    );
    printf(
            "\t Against base GPU implementation: %f times as fast | %f%% time saved\n",
            timesFasterVsBaseGPU,
            percentageOfTimeSavedVsBaseGPU
    );
}
