#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>

void reduceWithThrust(int *inputData, int *result, int size) {
    thrust::device_vector<int> deviceInputData(inputData, inputData + size);


    int sum = thrust::reduce(deviceInputData.begin(), deviceInputData.end(), 0, thrust::plus<int>());


    cudaMemcpy(result, &sum, sizeof(int), cudaMemcpyHostToDevice);
}
