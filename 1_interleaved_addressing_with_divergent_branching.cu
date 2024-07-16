#include <cuda_runtime.h>

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *input_data, int *output_data, unsigned int n) {
    extern __shared__ int shared_data[];

    // each thread loads one element from global to shared mem
    unsigned int thread_block_index = threadIdx.x;
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    shared_data[thread_block_index] = input_data[thread_index];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int amount_of_elements_reduced = 1; amount_of_elements_reduced < blockDim.x; amount_of_elements_reduced *= 2) {
        if (thread_block_index % (2 * amount_of_elements_reduced) == 0) {
            shared_data[thread_block_index] += shared_data[thread_block_index + amount_of_elements_reduced];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (thread_block_index == 0) output_data[blockIdx.x] = shared_data[0];
}
