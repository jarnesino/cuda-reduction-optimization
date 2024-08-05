CC = nvcc

BENCHMARKS_TARGET = benchmarks
REDUCE_IMPLEMENTATIONS_DIRECTORY = reduce_implementations
REDUCE_IMPLEMENTATIONS = $(wildcard $(REDUCE_IMPLEMENTATIONS_DIRECTORY)/*.cu)
REDUCTION_FILES = reduction.cu $(REDUCE_IMPLEMENTATIONS)
BENCHMARK_FILES = benchmarks.cu thrust_reduction.cu $(REDUCTION_FILES)

TEST_TARGET = test

.PHONY: build run clean

build: $(BENCHMARKS_TARGET)

build_for_testing: $(TEST_TARGET)

$(BENCHMARKS_TARGET): $(BENCHMARK_FILES)
	$(CC) $(BENCHMARK_FILES) -o $(BENCHMARKS_TARGET)

$(TEST_TARGET): tests/test.cu $(REDUCTION_FILES)
	$(CC) tests/test.cu $(REDUCTION_FILES) -o $(TEST_TARGET)

run: build
	$(MAKE) run-benchmarks

run-benchmarks:
	./$(BENCHMARKS_TARGET)
	$(MAKE) clean

test-reduction: build_for_testing
	./$(TEST_TARGET)
	$(MAKE) clean

all: run test-reduction

clean:
	rm -f $(BENCHMARKS_TARGET) $(TEST_TARGET)
