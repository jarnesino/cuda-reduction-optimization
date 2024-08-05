#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result);

int main() {
    const unsigned int logDataSize = 30;
    const unsigned int dataSize = 1 << logDataSize;
    int *testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    ReductionResult reductionResult{};
    reductionResult = reduceAndMeasureTime(
            reduce_using_0_interleaved_addressing_with_local_memory,
            numberOfBlocksForStandardReduction, testingData, dataSize
    );
    printImplementationData(0, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_1_interleaved_addressing_with_divergent_branching,
            numberOfBlocksForStandardReduction, testingData, dataSize
    );
    printImplementationData(1, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_2_interleaved_addressing_with_bank_conflicts,
            numberOfBlocksForStandardReduction, testingData, dataSize
    );
    printImplementationData(2, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_3_sequential_addressing_with_idle_threads,
            numberOfBlocksForStandardReduction, testingData, dataSize
    );
    printImplementationData(3, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_4_first_add_during_load_with_loop_overhead,
            numberOfBlocksForReductionWithExtraStep, testingData, dataSize
    );
    printImplementationData(4, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_5_loop_unrolling_only_at_warp_level_iterations,
            numberOfBlocksForReductionWithExtraStep, testingData, dataSize
    );
    printImplementationData(5, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_6_complete_loop_unrolling_with_one_reduction,
            numberOfBlocksForReductionWithExtraStep, testingData, dataSize
    );
    printImplementationData(6, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_7_multiple_reduce_operations_per_thread_iteration,
            numberOfBlocksForReductionWithMultipleSteps, testingData, dataSize
    );
    printImplementationData(7, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            reduce_using_8_operations_for_consecutive_memory_addressing,
            numberOfBlocksForReductionWithConsecutiveMemoryAddressing, testingData, dataSize
    );
    printImplementationData(8, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);

    return EXIT_SUCCESS;
}

void printImplementationData(const unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Reduction result: %d\n", result);
}