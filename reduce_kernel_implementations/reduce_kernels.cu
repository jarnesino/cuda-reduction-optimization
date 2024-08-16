#include "reduce_kernels.cuh"

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

ReduceImplementationKernel reduceImplementationKernels[NUMBER_OF_KERNEL_IMPLEMENTATIONS] = {
        {1,  "interleaved addressing with local memory",        interleaved_addressing_with_local_memory,        numberOfBlocksForStandardReduction},
        {2,  "interleaved addressing with divergent branching", interleaved_addressing_with_divergent_branching, numberOfBlocksForStandardReduction},
        {3,  "interleaved addressing with bank conflicts",      interleaved_addressing_with_bank_conflicts,      numberOfBlocksForStandardReduction},
        {4,  "sequential addressing with idle threads",         sequential_addressing_with_idle_threads,         numberOfBlocksForStandardReduction},
        {5,  "first add during load with loop overhead",        first_add_during_load_with_loop_overhead,        numberOfBlocksForReductionWithExtraStep},
        {6,  "loop unrolling only at warp level iterations",    loop_unrolling_only_at_warp_level_iterations,    numberOfBlocksForReductionWithExtraStep},
        {7,  "complete loop unrolling with one reduction",      complete_loop_unrolling_with_one_reduction,      numberOfBlocksForReductionWithExtraStep},
        {8,  "multiple reduce operations per thread iteration", multiple_reduce_operations_per_thread_iteration, numberOfBlocksForReductionWithMultipleSteps},
        {9,  "operations for consecutive memory addressing",    operations_for_consecutive_memory_addressing,    numberOfBlocksForReductionWithConsecutiveMemoryAddressing},
        {10, "shuffle down",                                    shuffle_down,                                    numberOfBlocksForReductionWithConsecutiveMemoryAddressing}
};

void checkForCUDAErrors() {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: ";
        std::cerr << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int reduceWithKernelInDevice(
        const ReduceImplementationKernel &reduceImplementationKernel,
        unsigned int remainingElements,
        unsigned int numberOfBlocks,
        const size_t sharedMemSize,
        int *inputPointer,
        int *outputPointer
) {
    // Launch kernel for each block.
    while (remainingElements > 1) {
        numberOfBlocks = reduceImplementationKernel.numberOfBlocksFunction(remainingElements);
        reduceImplementationKernel.function<<<numberOfBlocks, BLOCK_SIZE, sharedMemSize>>>(
                inputPointer, outputPointer, remainingElements
        );
        cudaDeviceSynchronize();
        checkForCUDAErrors();

        remainingElements = numberOfBlocks;
        inputPointer = outputPointer;
        outputPointer += remainingElements;
    }

    int value;
    cudaMemcpy(&value, inputPointer, sizeof(int), cudaMemcpyDeviceToHost);
    return value;
}

int reduceWithKernel(
        const ReduceImplementationKernel &reduceKernel, int *inputData, unsigned int dataSize
) {
    const size_t dataSizeInBytes = dataSize * sizeof(int);
    unsigned int remainingElements = dataSize;
    unsigned int numberOfBlocks = reduceKernel.numberOfBlocksFunction(remainingElements);

    int *deviceInputData, *deviceOutputData;
    cudaMalloc((void **) &deviceInputData, dataSizeInBytes);
    cudaMalloc(
            (void **) &deviceOutputData,
            numberOfBlocks * sizeof(int) * 2
    );  // Allocate double the memory for use in subsequent layers.
    cudaMemcpy(deviceInputData, inputData, dataSizeInBytes, cudaMemcpyHostToDevice);
    const size_t sharedMemSize = BLOCK_SIZE * sizeof(int);

    int *inputPointer = deviceInputData;
    int *outputPointer = deviceOutputData;

    int value = reduceWithKernelInDevice(
            reduceKernel, remainingElements, numberOfBlocks, sharedMemSize, inputPointer, outputPointer
    );

    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    return value;
}
