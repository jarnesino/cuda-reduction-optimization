#ifndef REDUCTION
#define REDUCTION

void reduce(const int dataSize, cudaEvent_t startEvent, cudaEvent_t stopEvent);
void initializeRandomTestingDataIn(int *data, int size);

#endif // REDUCTION
