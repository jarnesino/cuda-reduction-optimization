#include "time.cuh"
#include "../data/data.cuh"
#include <string>

void measureElapsedTimes(
        unsigned int dataSize,
        unsigned int SAMPLE_SIZE,
        float *averageElapsedTimesInMilliseconds
);

void printBenchmarkStats(
        unsigned int logDataSize,
        unsigned int SAMPLE_SIZE,
        const float *averageElapsedTimesInMilliseconds
);

void printImplementationData(
        unsigned int implementationNumber,
        const std::string &implementationName,
        float averageElapsedTimeInMilliseconds,
        float timesFasterVsCPU,
        float percentageOfTimeSavedVsCPU,
        float timesFasterVsBaseGPU,
        float percentageOfTimeSavedVsBaseGPU
);

int main() {
    const unsigned int SAMPLE_SIZE = 50;

    // Constructor fills array with zeroes.
    float averageElapsedTimesInMilliseconds[NUMBER_OF_IMPLEMENTATIONS] = {};

    const unsigned int logDataSizes[3] = {10, 20, 30};
    for (unsigned int logDataSize: logDataSizes) {
        const unsigned int dataSize = 1 << logDataSize;

        measureElapsedTimes(dataSize, SAMPLE_SIZE, averageElapsedTimesInMilliseconds);
        printBenchmarkStats(logDataSize, SAMPLE_SIZE, averageElapsedTimesInMilliseconds);
    }

    return EXIT_SUCCESS;
}

/* *************** AUXILIARY *************** */

void measureElapsedTimes(
        const unsigned int dataSize,
        const unsigned int SAMPLE_SIZE,
        float *averageElapsedTimesInMilliseconds
) {
    int *testingData = new int[dataSize];

    for (unsigned int sampleNumber = 1; sampleNumber <= SAMPLE_SIZE; sampleNumber++) {
        printf("Generating data for sample %d\n", sampleNumber);
        initializeRandomDataAndGetSumIn(testingData, dataSize);

        for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++) {
            TimedReductionResult reductionResultForImplementation = reduceAndMeasureTime(
                    reduceImplementations[index], testingData, dataSize
            );
            averageElapsedTimesInMilliseconds[index] += reductionResultForImplementation.elapsedMilliseconds;

            printf(
                    "Completed sample %d for implementation %d of %d\n",
                    sampleNumber,
                    reduceImplementations[index].number,
                    NUMBER_OF_IMPLEMENTATIONS
            );
        }
    }

    for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++)
        averageElapsedTimesInMilliseconds[index] /= (float) SAMPLE_SIZE;
}

void printBenchmarkStats(
        const unsigned int logDataSize,
        const unsigned int SAMPLE_SIZE,
        const float *averageElapsedTimesInMilliseconds
) {
    printf(
            "****************** LOG DATA SIZE: %d ****************** SAMPLE SIZE: %d ******************\n",
            logDataSize, SAMPLE_SIZE
    );

    float timesFasterVsCPU;
    float percentageOfTimeSavedVsCPU;
    float timesFasterVsBaseGPU;
    float percentageOfTimeSavedVsBaseGPU;

    const float averageElapsedTimeForSequentialCPUImplementation = averageElapsedTimesInMilliseconds[0];
    const float averageElapsedTimeForBaseGPUImplementation = averageElapsedTimesInMilliseconds[1];
    for (int index = 0; index < NUMBER_OF_IMPLEMENTATIONS; index++) {
        timesFasterVsCPU = averageElapsedTimeForSequentialCPUImplementation / averageElapsedTimesInMilliseconds[index];
        percentageOfTimeSavedVsCPU = (
                100.0f
                * (averageElapsedTimeForSequentialCPUImplementation - averageElapsedTimesInMilliseconds[index])
                / averageElapsedTimeForSequentialCPUImplementation
        );
        timesFasterVsBaseGPU = averageElapsedTimeForBaseGPUImplementation / averageElapsedTimesInMilliseconds[index];
        percentageOfTimeSavedVsBaseGPU = (
                100.0f
                * (averageElapsedTimeForBaseGPUImplementation - averageElapsedTimesInMilliseconds[index])
                / averageElapsedTimeForBaseGPUImplementation
        );

        printImplementationData(
                reduceImplementations[index].number,
                reduceImplementations[index].name,
                averageElapsedTimesInMilliseconds[index],
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
        float averageElapsedTimeInMilliseconds,
        float timesFasterVsCPU,
        float percentageOfTimeSavedVsCPU,
        float timesFasterVsBaseGPU,
        float percentageOfTimeSavedVsBaseGPU
) {
    printf("Implementation: %d - ", implementationNumber);
    std::cout << implementationName << "\n";
    printf("\t Time: %f ms\n", averageElapsedTimeInMilliseconds);
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
