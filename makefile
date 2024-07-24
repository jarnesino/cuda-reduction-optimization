CC = nvcc
REDUCTION_TARGET = reduction

REDUCE_IMPLEMENTATIONS_DIRECTORY = reduce_implementations
REDUCE_IMPLEMENTATIONS = $(wildcard $(REDUCE_IMPLEMENTATIONS_DIRECTORY)/*.cu)


.PHONY: build run clean

build: $(REDUCTION_TARGET)

$(REDUCTION_TARGET): reduction.cu $(REDUCE_IMPLEMENTATIONS)
	$(CC) reduction.cu $(REDUCE_IMPLEMENTATIONS) -o $(REDUCTION_TARGET)

run: build
	./$(REDUCTION_TARGET)
	$(MAKE) clean

all: run

clean:
	rm -f $(REDUCTION_TARGET)
