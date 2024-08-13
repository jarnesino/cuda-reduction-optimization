#include "reduction.cuh"
#include "reduce_non_kernel_implementations/thrust_reduction.cuh"
#include "data.cuh"
#include <string>

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

const unsigned int NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS = 1;

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
        float timesFaster,
        float percentageOfTimeSaved
);

int main() {
    const unsigned int SAMPLE_SIZE = 5;

    // Constructor fills array with zeroes.
    float elapsedTimesInMillisecondsForKernels[NUMBER_OF_KERNEL_IMPLEMENTATIONS + 1] = {};
    float elapsedTimesInMillisecondsForNonKernels[1] = {};

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

        for (int index = 0; index < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; index++) {
            ReductionResult reductionResultForImplementation = reduceAndMeasureTimeWithThrust(testingData, dataSize);
            elapsedTimesInMillisecondsForNonKernels[index] += reductionResultForImplementation.elapsedMilliseconds;

            printf(
                    "Completed sample %d for (non-kernel) implementation %d\n",
                    sampleNumber,
                    NUMBER_OF_KERNEL_IMPLEMENTATIONS + index + 1
            );
        }
    }

    for (int implementationIndex = 0;
         implementationIndex < NUMBER_OF_KERNEL_IMPLEMENTATIONS + 1; implementationIndex++) {
        elapsedTimesInMillisecondsForKernels[implementationIndex] /= (float) SAMPLE_SIZE;
    }

    for (int implementationIndex = 0;
         implementationIndex < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; implementationIndex++) {
        elapsedTimesInMillisecondsForNonKernels[implementationIndex] /= (float) SAMPLE_SIZE;
    }
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

    for (int index = 0; index < NUMBER_OF_KERNEL_IMPLEMENTATIONS; index++) {
        float timesFaster = elapsedTimesInMillisecondsForKernels[0] / elapsedTimesInMillisecondsForKernels[index];
        float percentageOfTimeSaved = (
                100.0f
                * (elapsedTimesInMillisecondsForKernels[0] - elapsedTimesInMillisecondsForKernels[index])
                / elapsedTimesInMillisecondsForKernels[0]
        );

        printImplementationData(
                reduceImplementationKernels[index].number,
                reduceImplementationKernels[index].name,
                elapsedTimesInMillisecondsForKernels[index],
                timesFaster,
                percentageOfTimeSaved
        );
    }

    for (int index = 0; index < NUMBER_OF_NON_KERNEL_IMPLEMENTATIONS; index++) {
        float timesFaster = elapsedTimesInMillisecondsForKernels[0] / elapsedTimesInMillisecondsForNonKernels[index];
        float percentageOfTimeSaved = (
                100.0f
                * (elapsedTimesInMillisecondsForKernels[0] - elapsedTimesInMillisecondsForNonKernels[index])
                / elapsedTimesInMillisecondsForKernels[0]
        );

        printImplementationData(
                NUMBER_OF_KERNEL_IMPLEMENTATIONS + index + 1,
                "CUDA Thrust",
                elapsedTimesInMillisecondsForKernels[index],
                timesFaster,
                percentageOfTimeSaved
        );
    }

    printf("*****************************************************************************************\n");
}

void printImplementationData(
        const unsigned int implementationNumber,
        const std::string &implementationName,
        float elapsedTimeInMilliseconds,
        float timesFaster,
        float percentageOfTimeSaved
) {
    printf("Implementation: %d - ", implementationNumber);
    std::cout << implementationName << " ->";
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Times faster than base implementation: %f", timesFaster);
    printf("\t Time saved compared with base implementation: %f %%\n", percentageOfTimeSaved);
}
