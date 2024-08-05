#include "reduction.cuh"

/*

Playing around with CUDA optimizations.
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

*/

void printImplementationData(unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result);

int main() {
    // Create CUDA events for timing.
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    const unsigned int logDataSize = 30;  // At least one element in the testing data
    const unsigned int dataSize = 1 << logDataSize;
    int *testingData = new int[dataSize];
    initializeTestingDataIn(testingData, dataSize);

    auto reductionResult = ReductionResult{0, 0};
    reductionResult = reduceAndMeasureTime(
            0, reduce_using_0_interleaved_addressing_with_local_memory,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(0, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            1, reduce_using_1_interleaved_addressing_with_divergent_branching,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(1, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            2, reduce_using_2_interleaved_addressing_with_bank_conflicts,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(2, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            3, reduce_using_3_sequential_addressing_with_idle_threads,
            amountOfBlocksForStandardReduction, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(3, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            4, reduce_using_4_first_add_during_load_with_loop_overhead,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(4, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            5, reduce_using_5_loop_unrolling_only_at_warp_level_iterations,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(5, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            6, reduce_using_6_complete_loop_unrolling_with_one_reduction,
            amountOfBlocksForReductionWithExtraStep, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(6, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            7, reduce_using_7_multiple_reduce_operations_per_thread_iteration,
            amountOfBlocksForReductionWithMultipleSteps, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(7, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);
    reductionResult = reduceAndMeasureTime(
            8, reduce_using_8_operations_for_consecutive_memory_addressing,
            amountOfBlocksForReductionWithConsecutiveMemoryAddressing, testingData, dataSize, startEvent, stopEvent
    );
    printImplementationData(8, reductionResult.elapsedTimeInMilliseconds, reductionResult.value);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return EXIT_SUCCESS;
}

void printImplementationData(const unsigned int implementationNumber, float elapsedTimeInMilliseconds, int result) {
    printf("*** Implementation number: %d", implementationNumber);
    printf("\t Elapsed time: %f ms", elapsedTimeInMilliseconds);
    printf("\t Reduction result: %d\n", result);
}