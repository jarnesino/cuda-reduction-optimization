#include <cuda_runtime.h>

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *input_data, int *output_data, unsigned int n) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int thread_block_index = threadIdx.x;
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[thread_block_index] = input_data[thread_index];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (thread_block_index % (2*s) == 0) {
            sdata[thread_block_index] += sdata[thread_block_index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (thread_block_index == 0) output_data[blockIdx.x] = sdata[0];
}
