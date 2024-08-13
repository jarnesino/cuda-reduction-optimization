#define DOCTEST_CONFIG_IMPLEMENT

#include "doctest.h"
#include "../data.cuh"
#include "../reduction.cuh"
#include "../reduce_non_kernel_implementations/thrust_reduction.cuh"
#include <random>

TEST_SUITE("reduction of arrays with different sizes") {
    int initializeTestingDataAndGetSum(int *data, unsigned int size) {
        for (unsigned int index1 = 0; index1 < size; ++index1) {
            data[index1] = 1;
        }

        int sum = (int) size;
        return sum;
    }

    void testReductionsWithArrayOfLogSize(unsigned int logDataSize) {
        const unsigned int dataSize = 1 << logDataSize;
        int *testingData = new int[dataSize];
        int expectedSum = initializeTestingDataAndGetSum(testingData, dataSize);

        for (const auto &reduceKernel: reduceImplementationKernels) {
            ReductionResult reductionResult = reduceAndMeasureTime(
                    reduceKernel, testingData, dataSize
            );
            CHECK_EQ(reductionResult.value, expectedSum);
        }

        ReductionResult reductionResultForThrust = reduceAndMeasureTimeWithThrust(testingData, dataSize);
        CHECK_EQ(reductionResultForThrust.value, expectedSum);
    }

    TEST_CASE("reduce small arrays") {
        const unsigned int logDataSize = 8;
        testReductionsWithArrayOfLogSize(logDataSize);
    }

    TEST_CASE("reduce medium arrays") {
        const unsigned int logDataSize = 17;
        testReductionsWithArrayOfLogSize(logDataSize);
    }

    TEST_CASE("reduce big arrays") {
        const unsigned int logDataSize = 30;
        testReductionsWithArrayOfLogSize(logDataSize);
    }
}

TEST_SUITE("reduction of arrays with random data") {
    TEST_CASE("reduce arrays with random positive and negative integers") {
        const unsigned int logDataSize = 11;
        const unsigned int dataSize = 1 << logDataSize;
        int *testingData = new int[dataSize];
        int expectedSum = initializeRandomDataAndGetSumIn(testingData, dataSize);

        for (const auto &reduceKernel: reduceImplementationKernels) {
            ReductionResult reductionResult = reduceAndMeasureTime(
                    reduceKernel, testingData, dataSize
            );

            CHECK_EQ(reductionResult.value, expectedSum);
        }

        ReductionResult reductionResultForThrust = reduceAndMeasureTimeWithThrust(testingData, dataSize);
        CHECK_EQ(reductionResultForThrust.value, expectedSum);
    }
}

int main() {
    doctest::Context context;

    int res = context.run();
    if (context.shouldExit())
        return res;  // Propagate test results.

    int clientStuffReturnCode = 0;
    return res + clientStuffReturnCode;  // Propagate test results.
}