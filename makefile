CC = nvcc

REDUCTION_TARGET = reduction
REDUCE_IMPLEMENTATIONS_DIRECTORY = reduce_implementations
REDUCE_IMPLEMENTATIONS = $(wildcard $(REDUCE_IMPLEMENTATIONS_DIRECTORY)/*.cu)
REDUCTION_FILES = run.cu reduction.cu $(REDUCE_IMPLEMENTATIONS)

TEST_TARGET = test
TEST_FILES = tests/test.cu

.PHONY: build run clean

build: $(REDUCTION_TARGET)
build_for_testing: $(TEST_TARGET)

$(REDUCTION_TARGET): $(REDUCTION_FILES)
	$(CC) $(REDUCTION_FILES) -o $(REDUCTION_TARGET)

$(TEST_TARGET): $(TEST_FILES)
	$(CC) $(TEST_FILES) -o $(TEST_TARGET)

run: build
	./$(REDUCTION_TARGET)
	$(MAKE) clean

test-reduction: build_for_testing
	./$(TEST_TARGET)
	$(MAKE) clean

all: run test-reduction

clean:
	rm -f $(REDUCTION_TARGET) $(TEST_TARGET)
