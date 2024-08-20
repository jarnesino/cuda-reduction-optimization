#include "reduction.cuh"

ReduceImplementation reduceImplementations[NUMBER_OF_IMPLEMENTATIONS] = {
        {1, "CPU sequential", reduceWithCPU},
        {2, "Interleaved addressing with local memory", reduceWithInterleavedAddressingWithLocalMemory},
        {3, "Interleaved addressing with divergent branching", reduceWithInterleavedAddressingWithDivergentBranching},
        {4, "Interleaved addressing with bank conflicts", reduceWithInterleavedAddressingWithBankConflicts},
        {5, "Sequential addressing with idle threads", reduceWithSequentialAddressingWithIdleThreads},
        {6, "First add during load with loop overhead", reduceWithFirstAddDuringLoadWithLoopOverhead},
        {7, "Loop unrolling only at warp level iterations", reduceWithLoopUnrollingOnlyAtWarpLevelIterations},
        {8, "Complete loop unrolling with one reduction", reduceWithCompleteLoopUnrollingWithOneReduction},
        {9, "Multiple reduce operations per thread iteration", reduceWithMultipleReduceOperationsPerThreadIteration},
        {10, "Operations for consecutive memory addressing", reduceWithOperationsForConsecutiveMemoryAddressing},
        {11, "Shuffle down", reduceWithShuffleDown},
        {12, "CUDA Thrust", reduceWithThrust}
};
