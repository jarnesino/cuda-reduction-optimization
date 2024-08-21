#include "reduction.cuh"

ReduceImplementation reduceImplementations[NUMBER_OF_IMPLEMENTATIONS] = {
        {1,  "CPU sequential",                                                   reduceWithCPU},
        {2,  "Interleaved addressing with local memory and divergent branching", reduceWithInterleavedAddressingWithLocalMemoryAndDivergentBranching},
        {3,  "Interleaved addressing with divergent branching",                  reduceWithInterleavedAddressingWithDivergentBranching},
        {4,  "Interleaved addressing with local memory and bank conflicts",      reduceWithInterleavedAddressingWithLocalMemoryAndBankConflicts},
        {5,  "Interleaved addressing with bank conflicts",                       reduceWithInterleavedAddressingWithBankConflicts},
        {6,  "Sequential addressing with local memory and idle threads",         reduceWithSequentialAddressingWithLocalMemoryAndIdleThreads},
        {7,  "Sequential addressing with idle threads",                          reduceWithSequentialAddressingWithIdleThreads},
        {8,  "First add during load with loop overhead",                         reduceWithFirstAddDuringLoadWithLoopOverhead},
        {9,  "Loop unrolling only at warp level iterations",                     reduceWithLoopUnrollingOnlyAtWarpLevelIterations},
        {10, "Complete loop unrolling with one reduction",                       reduceWithCompleteLoopUnrollingWithOneReduction},
        {11, "Multiple reduce operations per thread iteration",                  reduceWithMultipleReduceOperationsPerThreadIteration},
        {12, "Operations for consecutive memory addressing",                     reduceWithOperationsForConsecutiveMemoryAddressing},
        {13, "Shuffle down",                                                     reduceWithShuffleDown},
        {14, "CUDA Thrust",                                                      reduceWithThrust}
};
