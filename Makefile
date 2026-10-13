# Makefile for High-Performance Multi-Threaded Parallel K-Means
# Compiler configuration
CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c11 -Iinclude
PTHREAD_FLAGS ?= -pthread

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests

# Sources & Objects for Modular Architecture
QUEUE_POOL_SRCS = $(SRC_DIR)/task_queue.c $(SRC_DIR)/thread_pool.c
QUEUE_POOL_OBJS = $(OBJ_DIR)/task_queue.o $(OBJ_DIR)/thread_pool.o

MEM_ACC_SRCS = $(SRC_DIR)/aligned_memory.c $(SRC_DIR)/accumulator.c
MEM_ACC_OBJS = $(OBJ_DIR)/aligned_memory.o $(OBJ_DIR)/accumulator.o

CORE_SRCS = $(QUEUE_POOL_SRCS) $(MEM_ACC_SRCS) $(SRC_DIR)/kmeans.c
CORE_OBJS = $(QUEUE_POOL_OBJS) $(MEM_ACC_OBJS) $(OBJ_DIR)/kmeans.o

# Default target builds core test and threshold benchmark
all: test_engine benchmark_threshold

# Build all test and benchmark binaries
build_all: test_engine benchmark_threshold benchmark_false_sharing benchmark_scheduler hard_comparison_benchmark

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -c $< -o $@

# Test Suite: Unit Tests (Queue, Thread Pool, Telemetry)
test_engine: $(QUEUE_POOL_OBJS) $(TEST_DIR)/test_thread_pool.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) $(TEST_DIR)/test_thread_pool.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/test_thread_pool.exe

test: test_engine
	./$(BIN_DIR)/test_thread_pool.exe

# Benchmark 1: Parallel Overhead Break-Even Cut-Off Threshold
benchmark_threshold: $(CORE_OBJS) benchmarks/threshold_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/threshold_benchmark.c $(CORE_OBJS) -o $(BIN_DIR)/threshold_benchmark.exe -lm

run_threshold: benchmark_threshold
	./$(BIN_DIR)/threshold_benchmark.exe

# Benchmark 2: False Sharing & Cache-Line Alignment (MESI Protocol)
benchmark_false_sharing: $(MEM_ACC_OBJS) benchmarks/false_sharing_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/false_sharing_benchmark.c $(MEM_ACC_OBJS) -o $(BIN_DIR)/false_sharing_benchmark.exe -lm

run_false_sharing: benchmark_false_sharing
	./$(BIN_DIR)/false_sharing_benchmark.exe

# Benchmark 3: Dynamic Work-Pulling vs Straggler Jitter
benchmark_scheduler: $(QUEUE_POOL_OBJS) benchmarks/benchmark_scheduler.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/benchmark_scheduler.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/benchmark_scheduler.exe

run_scheduler: benchmark_scheduler
	./$(BIN_DIR)/benchmark_scheduler.exe

# Benchmark 4: Hardest Workload 3-Way Architecture Stress Comparison
hard_comparison_benchmark: $(QUEUE_POOL_OBJS) benchmarks/hard_comparison_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/hard_comparison_benchmark.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/hard_comparison_benchmark.exe -lm

run_hard_comparison: hard_comparison_benchmark
	./$(BIN_DIR)/hard_comparison_benchmark.exe

# Run every single test and benchmark sequentially
run_all: test run_false_sharing run_scheduler run_threshold run_hard_comparison

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all build_all clean test test_engine benchmark_threshold run_threshold \
        benchmark_false_sharing run_false_sharing benchmark_scheduler run_scheduler \
        hard_comparison_benchmark run_hard_comparison run_all
