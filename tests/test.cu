#define DOCTEST_CONFIG_IMPLEMENT

#include "doctest.h"
#include "../reduction.cuh"

TEST_SUITE("reduction of arrays with different sizes") {
    int initializeTestingDataAndGetSum(int *data, int size) {
        fillDataWith1s(data, size);
        int sum = size;
        return sum;
    }

    TEST_CASE("reduce big arrays") {
        const unsigned int logDataSize = 30;
        const unsigned int dataSize = 1 << logDataSize;
        int *testingData = new int[dataSize];
        int expectedSum = initializeTestingDataAndGetSum(testingData, dataSize);

        ReductionResult reductionResult = reduceAndMeasureTime(
                reduceImplementations[8].function,
                reduceImplementations[8].numberOfBlocksFunction, testingData, dataSize
        );

        printf("%d", testingData[0]);
        CHECK_EQ(reductionResult.value, expectedSum);
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