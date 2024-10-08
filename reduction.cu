#include "reduction.cuh"

ReduceImplementation reduceImplementations[NUMBER_OF_IMPLEMENTATIONS] = {
        {1,  "CPU sequential",                                                    reduceWithCPU},
        {2,  "Interleaved addressing with global memory and divergent branching", reduceWithInterleavedAddressingWithGlobalMemoryAndDivergentBranching},
        {3,  "Interleaved addressing with divergent branching",                   reduceWithInterleavedAddressingWithDivergentBranching},
        {4,  "Interleaved addressing with global memory and good branching",      reduceWithInterleavedAddressingWithGlobalMemoryAndGoodBranching},
        {5,  "Interleaved addressing with bank conflicts",                        reduceWithInterleavedAddressingWithBankConflicts},
        {6,  "Sequential addressing with global memory and idle threads",         reduceWithSequentialAddressingWithGlobalMemoryAndIdleThreads},
        {7,  "Sequential addressing with idle threads",                           reduceWithSequentialAddressingWithIdleThreads},
        {8,  "First add during load with loop overhead",                          reduceWithFirstAddDuringLoadWithLoopOverhead},
        {9,  "Loop unrolling only at warp level iterations",                      reduceWithLoopUnrollingOnlyAtWarpLevelIterations},
        {10, "Complete loop unrolling with one reduction",                        reduceWithCompleteLoopUnrollingWithOneReduction},
        {11, "Multiple reduce operations per thread iteration",                   reduceWithMultipleReduceOperationsPerThreadIteration},
        {12, "Operations for consecutive memory addressing",                      reduceWithOperationsForConsecutiveMemoryAddressing},
        {13, "Operations for consecutive memory addressing 2",                    reduceWithOperationsForConsecutiveMemoryAddressing2},
        {14, "Shuffle down",                                                      reduceWithShuffleDown},
        {15, "Shuffle down with loop unrolling",                                  reduceWithShuffleDownWithLoopUnrolling},
        {16, "BEST",                                                              BESTReduceWithOperationsForConsecutiveMemoryAddressing2},
        {17, "BEST2",                                                             BEST2ReduceWithOperationsForConsecutiveMemoryAddressing2},
        {18, "CUDA Thrust",                                                       reduceWithThrust}
};
