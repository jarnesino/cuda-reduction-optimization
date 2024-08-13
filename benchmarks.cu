#include "reduction.cuh"
#include "reduce_non_kernel_implementations/thrust_reduction.cuh"
#include "reduce_non_kernel_implementations/sequential_cpu_reduction.cuh"
#include "data.cuh"
#include <string>

const unsigned int NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS = 2;

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

        ReductionResult reductionResultForThrust = reduceAndMeasureTimeWithThrust(testingData, dataSize);
        elapsedTimesInMillisecondsForNonKernels[0] += reductionResultForThrust.elapsedMilliseconds;
        printf(
                "Completed sample %d for (non-kernel) implementation %d\n",
                sampleNumber,
                NUMBER_OF_KERNEL_IMPLEMENTATIONS + 1
        );

        ReductionResult reductionResultForCPU = reduceAndMeasureTimeWithCPU(testingData, dataSize);
        elapsedTimesInMillisecondsForNonKernels[1] += reductionResultForCPU.elapsedMilliseconds;
        printf(
                "Completed sample %d for (non-kernel) implementation %d\n",
                sampleNumber,
                NUMBER_OF_KERNEL_IMPLEMENTATIONS + 2
        );
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

    const float elapsedTimeForSequentialCPUImplementation = elapsedTimesInMillisecondsForNonKernels[1];
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

    timesFasterVsCPU = elapsedTimeForSequentialCPUImplementation / elapsedTimesInMillisecondsForNonKernels[0];
    percentageOfTimeSavedVsCPU = (
            100.0f
            * (elapsedTimeForSequentialCPUImplementation - elapsedTimesInMillisecondsForNonKernels[0])
            / elapsedTimeForSequentialCPUImplementation
    );
    timesFasterVsBaseGPU = elapsedTimeForBaseGPUImplementation / elapsedTimesInMillisecondsForNonKernels[0];
    percentageOfTimeSavedVsBaseGPU = (
            100.0f
            * (elapsedTimeForBaseGPUImplementation - elapsedTimesInMillisecondsForNonKernels[0])
            / elapsedTimeForBaseGPUImplementation
    );

    printImplementationData(
            NUMBER_OF_KERNEL_IMPLEMENTATIONS + 1,
            "CUDA Thrust",
            elapsedTimesInMillisecondsForNonKernels[0],
            timesFasterVsCPU,
            percentageOfTimeSavedVsCPU,
            timesFasterVsBaseGPU,
            percentageOfTimeSavedVsBaseGPU
    );

    timesFasterVsCPU = 1;
    percentageOfTimeSavedVsCPU = 0;
    timesFasterVsBaseGPU = elapsedTimeForBaseGPUImplementation / elapsedTimesInMillisecondsForNonKernels[1];
    percentageOfTimeSavedVsBaseGPU = (
            100.0f
            * (elapsedTimeForBaseGPUImplementation - elapsedTimesInMillisecondsForNonKernels[1])
            / elapsedTimeForBaseGPUImplementation
    );

    printImplementationData(
            NUMBER_OF_KERNEL_IMPLEMENTATIONS + 2,
            "CPU sequential",
            elapsedTimesInMillisecondsForNonKernels[1],
            timesFasterVsCPU,
            percentageOfTimeSavedVsCPU,
            timesFasterVsBaseGPU,
            percentageOfTimeSavedVsBaseGPU
    );

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
            "\t Against CPU implementation: %f faster | %f%% time saved\n",
            timesFasterVsCPU,
            percentageOfTimeSavedVsCPU
    );
    printf(
            "\t Against base GPU implementation: %f faster | %f%% time saved\n",
            timesFasterVsBaseGPU,
            percentageOfTimeSavedVsBaseGPU
    );
}
