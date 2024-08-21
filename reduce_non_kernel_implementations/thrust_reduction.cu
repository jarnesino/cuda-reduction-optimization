#include <thrust/device_vector.h>
#include <thrust/reduce.h>

int reduceWithThrust(int *inputData, unsigned int size) {
    thrust::device_vector<int> deviceInputData(inputData, inputData + size);

    return thrust::reduce(deviceInputData.begin(), deviceInputData.end(), 0, thrust::plus<int>());
}
