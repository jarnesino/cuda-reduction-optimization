#include "reduce_implementations.cuh"

unsigned int numberOfBlocksForStandardReduction(const unsigned int dataSize) {
    return (dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}


unsigned int numberOfBlocksForReductionWithExtraStep(const unsigned int dataSize) {
    const int blockSizedChunksReducedPerBlock = 2;
    return (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
           (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
}


unsigned int numberOfBlocksForReductionWithMultipleSteps(const unsigned int dataSize) {
    return unsignedMin(GRID_SIZE, numberOfBlocksForReductionWithExtraStep(dataSize));
}


unsigned int numberOfBlocksForReductionWithConsecutiveMemoryAddressing(const unsigned int dataSize) {
    const unsigned int blockSizedChunksReducedPerBlock = 4;
    const unsigned int blocks = (dataSize + BLOCK_SIZE * blockSizedChunksReducedPerBlock - 1) /
                                (BLOCK_SIZE * blockSizedChunksReducedPerBlock);
    return unsignedMin(GRID_SIZE, blocks);
}

unsigned int unsignedMin(unsigned int a, unsigned int b) {
    return a < b ? a : b;
}

ReduceImplementationKernel reduceImplementations[9] = {
    {0, reduce_using_0_interleaved_addressing_with_local_memory, numberOfBlocksForStandardReduction},
    {1, reduce_using_1_interleaved_addressing_with_divergent_branching, numberOfBlocksForStandardReduction},
    {2, reduce_using_2_interleaved_addressing_with_bank_conflicts, numberOfBlocksForStandardReduction},
    {3, reduce_using_3_sequential_addressing_with_idle_threads, numberOfBlocksForStandardReduction},
    {4, reduce_using_4_first_add_during_load_with_loop_overhead, numberOfBlocksForReductionWithExtraStep},
    {5, reduce_using_5_loop_unrolling_only_at_warp_level_iterations, numberOfBlocksForReductionWithExtraStep},
    {6, reduce_using_6_complete_loop_unrolling_with_one_reduction, numberOfBlocksForReductionWithExtraStep},
    {7, reduce_using_7_multiple_reduce_operations_per_thread_iteration, numberOfBlocksForReductionWithMultipleSteps},
    {8, reduce_using_8_operations_for_consecutive_memory_addressing, numberOfBlocksForReductionWithConsecutiveMemoryAddressing}
};