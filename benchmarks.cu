#include "reduction.cuh"
#include "data.cuh"
#include "reduce_non_kernel_implementations/reduce_non_kernel_implementations.cuh"
#include <string>

void measureElapsedTimes(
        unsigned int dataSize,
        unsigned int SAMPLE_SIZE,
        float *elapsedTimesInMillisecondsForKernels,
        float *elapsedTimesInMillisecondsForNonKernels
);

void printBenchmarkStats(
        unsigned int logDataSize,
        unsigned int SAMPLE_SIZE,
        const float *elapsedTimesInMillisecondsForKernels,
        const float *elapsedTimesInMillisecondsForNonKernels
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
    const unsigned int SAMPLE_SIZE = 5;

    // Constructor fills array with zeroes.
    float elapsedTimesInMillisecondsForKernels[NUMBER_OF_KERNEL_IMPLEMENTATIONS] = {};
    float elapsedTimesInMillisecondsForNonKernels[NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS] = {};

    const unsigned int logDataSizes[3] = {10, 20, 30};
    for (unsigned int logDataSize: logDataSizes) {
        const unsigned int dataSize = 1 << logDataSize;

        measureElapsedTimes(
                dataSize, SAMPLE_SIZE, elapsedTimesInMillisecondsForKernels, elapsedTimesInMillisecondsForNonKernels
        );
        printBenchmarkStats(
                logDataSize, SAMPLE_SIZE, elapsedTimesInMillisecondsForKernels, elapsedTimesInMillisecondsForNonKernels
        );
    }

    return EXIT_SUCCESS;
}

/* *************** AUXILIARY *************** */

void measureElapsedTimes(
        const unsigned int dataSize,
        const unsigned int SAMPLE_SIZE,
        float *elapsedTimesInMillisecondsForKernels,
        float *elapsedTimesInMillisecondsForNonKernels
) {
    int *testingData = new int[dataSize];

    for (unsigned int sampleNumber = 1; sampleNumber <= SAMPLE_SIZE; sampleNumber++) {
        printf("Generating data for sample %d\n", sampleNumber);
        initializeRandomDataAndGetSumIn(testingData, dataSize);

        for (int index = 0; index < NUMBER_OF_KERNEL_IMPLEMENTATIONS; index++) {
            ReductionResult reductionResultForImplementation = reduceAndMeasureTime(
                    reduceImplementationKernels[index], testingData, dataSize
            );
            elapsedTimesInMillisecondsForKernels[index] += reductionResultForImplementation.elapsedMilliseconds;

            printf(
                    "Completed sample %d for (kernel) implementation %d\n",
                    sampleNumber,
                    reduceImplementationKernels[index].number
            );
        }

        for (unsigned int index = 0; index < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; index++) {
            ReductionResult reductionResultForImplementation = reduceNonKernelImplementations[index].function(
                    testingData, dataSize
            );
            elapsedTimesInMillisecondsForNonKernels[index] += reductionResultForImplementation.elapsedMilliseconds;
            printf(
                    "Completed sample %d for (non-kernel) implementation %d\n",
                    sampleNumber,
                    reduceNonKernelImplementations[index].number
            );
        }
    }

    for (int index = 0; index < NUMBER_OF_KERNEL_IMPLEMENTATIONS; index++)
        elapsedTimesInMillisecondsForKernels[index] /= (float) SAMPLE_SIZE;

    for (int index = 0; index < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; index++)
        elapsedTimesInMillisecondsForNonKernels[index] /= (float) SAMPLE_SIZE;
}

void printBenchmarkStats(
        const unsigned int logDataSize,
        const unsigned int SAMPLE_SIZE,
        const float *elapsedTimesInMillisecondsForKernels,
        const float *elapsedTimesInMillisecondsForNonKernels
) {
    printf(
            "****************** LOG DATA SIZE: %d ****************** SAMPLE SIZE: %d ******************\n",
            logDataSize, SAMPLE_SIZE
    );

    float timesFasterVsCPU;
    float percentageOfTimeSavedVsCPU;
    float timesFasterVsBaseGPU;
    float percentageOfTimeSavedVsBaseGPU;

    const float elapsedTimeForSequentialCPUImplementation = elapsedTimesInMillisecondsForNonKernels[0];
    const float elapsedTimeForBaseGPUImplementation = elapsedTimesInMillisecondsForKernels[0];
    for (int index = 0; index < NUMBER_OF_KERNEL_IMPLEMENTATIONS; index++) {
        timesFasterVsCPU = elapsedTimeForSequentialCPUImplementation / elapsedTimesInMillisecondsForKernels[index];
        percentageOfTimeSavedVsCPU = (
                100.0f
                * (elapsedTimeForSequentialCPUImplementation - elapsedTimesInMillisecondsForKernels[index])
                / elapsedTimeForSequentialCPUImplementation
        );
        timesFasterVsBaseGPU = elapsedTimeForBaseGPUImplementation / elapsedTimesInMillisecondsForKernels[index];
        percentageOfTimeSavedVsBaseGPU = (
                100.0f
                * (elapsedTimeForBaseGPUImplementation - elapsedTimesInMillisecondsForKernels[index])
                / elapsedTimeForBaseGPUImplementation
        );

        printImplementationData(
                reduceImplementationKernels[index].number,
                reduceImplementationKernels[index].name,
                elapsedTimesInMillisecondsForKernels[index],
                timesFasterVsCPU,
                percentageOfTimeSavedVsCPU,
                timesFasterVsBaseGPU,
                percentageOfTimeSavedVsBaseGPU
        );
    }

    for (int index = 0; index < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; index++) {
        timesFasterVsCPU = elapsedTimeForSequentialCPUImplementation / elapsedTimesInMillisecondsForNonKernels[index];
        percentageOfTimeSavedVsCPU = (
                100.0f
                * (elapsedTimeForSequentialCPUImplementation - elapsedTimesInMillisecondsForNonKernels[index])
                / elapsedTimeForSequentialCPUImplementation
        );
        timesFasterVsBaseGPU = elapsedTimeForBaseGPUImplementation / elapsedTimesInMillisecondsForNonKernels[index];
        percentageOfTimeSavedVsBaseGPU = (
                100.0f
                * (elapsedTimeForBaseGPUImplementation - elapsedTimesInMillisecondsForNonKernels[index])
                / elapsedTimeForBaseGPUImplementation
        );

        printImplementationData(
                reduceNonKernelImplementations[index].number,
                reduceNonKernelImplementations[index].name,
                elapsedTimesInMillisecondsForNonKernels[index],
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
