#include <iostream>
#include <cuda_runtime.h>

__global__ void reduce(int *g_idata, int *g_odata, unsigned int n) {
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

void initializeData(int *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % 100; // Random data for testing
    }
}

int main() {
    const int size = 1024;
    const int bytes = size * sizeof(int);

    int h_idata[size];
    int h_odata[size];

    initializeData(h_idata, size);

    int *d_idata, *d_odata;
    cudaMalloc((void **)&d_idata, bytes);
    cudaMalloc((void **)&d_odata, bytes);

    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    size_t sharedMemSize = threads * sizeof(int);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    reduce<<<blocks, threads, sharedMemSize>>>(d_idata, d_odata, size);

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    std::cout << "*****************************************************" << std::endl;

    std::cout << "Elapsed time: " << elapsedTime << " ms" << std::endl;

    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Reduction result: " << h_odata[0] << std::endl;

    std::cout << "*****************************************************" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_idata);
    cudaFree(d_odata);

    return 0;
}
