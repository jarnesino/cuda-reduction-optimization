#define DOCTEST_CONFIG_IMPLEMENT

#include "doctest.h"
#include "../reduction.cuh"
#include <random>

TEST_SUITE("reduction of arrays with different sizes") {
    int initializeTestingDataAndGetSum(int *data, unsigned int size) {
        fillDataWith1s(data, size);
        int sum = (int) size;
        return sum;
    }

    void testReductionsWithArrayOfLogSize(unsigned int logDataSize) {
        const unsigned int dataSize = 1 << logDataSize;
        int *testingData = new int[dataSize];
        int expectedSum = initializeTestingDataAndGetSum(testingData, dataSize);

        for (const auto &reduceImplementation: reduceImplementations) {
            ReductionResult reductionResult = reduceAndMeasureTime(
                    reduceImplementation, testingData, dataSize
            );

            CHECK_EQ(reductionResult.value, expectedSum);
        }
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
    int initializeRandomTestingDataAndGetSum(int *data, unsigned int size) {
        std::random_device randomSeed;
        std::mt19937 gen(randomSeed());
        std::uniform_int_distribution<> uniformDistribution(-2000, 2000);

        int sum = 0;
        for (unsigned int index = 0; index < size; ++index) {
            int randomNumber = uniformDistribution(gen);
            data[index] = randomNumber;
            sum += randomNumber;
        }
        return sum;
    }

    TEST_CASE("reduce arrays with random positive and negative integers") {
        const unsigned int logDataSize = 11;
        const unsigned int dataSize = 1 << logDataSize;
        int *testingData = new int[dataSize];
        int expectedSum = initializeRandomTestingDataAndGetSum(testingData, dataSize);

        for (const auto &reduceImplementation: reduceImplementations) {
            ReductionResult reductionResult = reduceAndMeasureTime(
                    reduceImplementation, testingData, dataSize
            );

            CHECK_EQ(reductionResult.value, expectedSum);
        }
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