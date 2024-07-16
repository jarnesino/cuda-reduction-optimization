CC = nvcc
REDUCTION_TARGET = reduction

.PHONY: build run clean

build: $(REDUCTION_TARGET)

$(REDUCTION_TARGET): reduction.cu
	$(CC) reduction.cu -o $(REDUCTION_TARGET)

run: build
	./$(REDUCTION_TARGET)
	$(MAKE) clean

clean:
	rm -f $(REDUCTION_TARGET)
