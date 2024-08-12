#include "data.cuh"

int initializeRandomDataAndGetSumIn(int *data, unsigned int size) {
    std::random_device randomSeed;
    std::mt19937 gen(randomSeed());
    std::uniform_int_distribution<> uniformDistribution(-2000, 2000);

    int sum = 0;
    for (unsigned int index = 0; index < size; ++index) {
        int randomNumber = uniformDistribution(gen);
        data[index] = randomNumber;
        sum += randomNumber;
    }
    return sum;
}
