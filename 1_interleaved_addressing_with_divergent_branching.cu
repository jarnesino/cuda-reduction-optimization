#include <cuda_runtime.h>

__global__ void reduce_using_1_interleaved_addressing_with_divergent_branching(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];

    // Load shared mem from global mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // Do reduction in shared memory
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
