#ifndef TIME
#define TIME

#include "../reduction.cuh"

struct TimedReductionResult {
    int value;
    float elapsedMilliseconds;
};

TimedReductionResult reduceAndMeasureTime(
        const ReduceImplementation &reduceImplementation, int *inputData, const unsigned int dataSize
);

#endif  // TIME
