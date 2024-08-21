int reduceWithCPU(int *inputData, unsigned int size) {
    int sum = 0;

    for (unsigned int index = 0; index < size; index++) {
        sum += inputData[index];
    }

    return sum;
}
