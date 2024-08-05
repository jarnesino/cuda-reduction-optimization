#define DOCTEST_CONFIG_IMPLEMENT

#include "doctest.h"
#include "../reduction.cuh"

TEST_SUITE("reduction of arrays with different sizes") {
    int initializeTestingDataAndGetSum(int *data, int size) {
        fillDataWith1s(data, size);
        int sum = size;
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

int main() {
    doctest::Context context;

    int res = context.run();
    if (context.shouldExit())
        return res;  // Propagate test results.

    int clientStuffReturnCode = 0;
    return res + clientStuffReturnCode;  // Propagate test results.
}