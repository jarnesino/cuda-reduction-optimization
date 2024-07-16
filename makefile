CC = nvcc
TARGET = reduction

all: $(TARGET)

$(TARGET): reduction.cu
	$(CC) reduction.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
