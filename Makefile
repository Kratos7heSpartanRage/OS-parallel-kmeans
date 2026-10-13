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

# Modular Sources & Objects
QUEUE_POOL_SRCS = $(SRC_DIR)/task_queue.c $(SRC_DIR)/thread_pool.c
QUEUE_POOL_OBJS = $(OBJ_DIR)/task_queue.o $(OBJ_DIR)/thread_pool.o

MEM_ACC_SRCS = $(SRC_DIR)/aligned_memory.c $(SRC_DIR)/accumulator.c
MEM_ACC_OBJS = $(OBJ_DIR)/aligned_memory.o $(OBJ_DIR)/accumulator.o

SYNC_SRCS = $(SRC_DIR)/synchronization.c
SYNC_OBJS = $(OBJ_DIR)/synchronization.o

CORE_SRCS = $(QUEUE_POOL_SRCS) $(MEM_ACC_SRCS) $(SRC_DIR)/kmeans.c
CORE_OBJS = $(QUEUE_POOL_OBJS) $(MEM_ACC_OBJS) $(OBJ_DIR)/kmeans.o

BASELINES_OBJS = $(OBJ_DIR)/naive_kmeans.o $(OBJ_DIR)/static_kmeans.o
ALL_KMEANS_OBJS = $(CORE_OBJS) $(BASELINES_OBJS)

# Default targets
all: test_engine test_synchronization test_naive test_static benchmark_threshold

build_all: test_engine test_synchronization test_naive test_static \
           benchmark_threshold benchmark_false_sharing benchmark_scheduler \
           hard_comparison_benchmark benchmark_synchronization benchmark_suite

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -c $< -o $@

# Unit Tests
test_engine: $(QUEUE_POOL_OBJS) $(TEST_DIR)/test_thread_pool.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) $(TEST_DIR)/test_thread_pool.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/test_thread_pool.exe

test_synchronization: $(MEM_ACC_OBJS) $(SYNC_OBJS) $(TEST_DIR)/test_synchronization.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) $(TEST_DIR)/test_synchronization.c $(MEM_ACC_OBJS) $(SYNC_OBJS) -o $(BIN_DIR)/test_synchronization.exe -lm

test_naive: $(ALL_KMEANS_OBJS) test_naive.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) test_naive.c $(ALL_KMEANS_OBJS) -o $(BIN_DIR)/test_naive.exe -lm

test_static: $(ALL_KMEANS_OBJS) test_static.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) test_static.c $(ALL_KMEANS_OBJS) -o $(BIN_DIR)/test_static.exe -lm

test: test_engine test_synchronization test_naive test_static
	./$(BIN_DIR)/test_thread_pool.exe
	./$(BIN_DIR)/test_synchronization.exe
	./$(BIN_DIR)/test_naive.exe
	./$(BIN_DIR)/test_static.exe

# Benchmarks
benchmark_threshold: $(CORE_OBJS) benchmarks/threshold_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/threshold_benchmark.c $(CORE_OBJS) -o $(BIN_DIR)/threshold_benchmark.exe -lm

run_threshold: benchmark_threshold
	./$(BIN_DIR)/threshold_benchmark.exe

benchmark_false_sharing: $(MEM_ACC_OBJS) benchmarks/false_sharing_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/false_sharing_benchmark.c $(MEM_ACC_OBJS) -o $(BIN_DIR)/false_sharing_benchmark.exe -lm

run_false_sharing: benchmark_false_sharing
	./$(BIN_DIR)/false_sharing_benchmark.exe

benchmark_scheduler: $(QUEUE_POOL_OBJS) benchmarks/benchmark_scheduler.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/benchmark_scheduler.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/benchmark_scheduler.exe

run_scheduler: benchmark_scheduler
	./$(BIN_DIR)/benchmark_scheduler.exe

hard_comparison_benchmark: $(QUEUE_POOL_OBJS) benchmarks/hard_comparison_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/hard_comparison_benchmark.c $(QUEUE_POOL_OBJS) -o $(BIN_DIR)/hard_comparison_benchmark.exe -lm

run_hard_comparison: hard_comparison_benchmark
	./$(BIN_DIR)/hard_comparison_benchmark.exe

benchmark_synchronization: $(MEM_ACC_OBJS) $(SYNC_OBJS) benchmarks/synchronization_benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/synchronization_benchmark.c $(MEM_ACC_OBJS) $(SYNC_OBJS) -o $(BIN_DIR)/benchmark_synchronization.exe -lm

benchmark_suite: $(ALL_KMEANS_OBJS) benchmarks/benchmark_suite.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/benchmark_suite.c $(ALL_KMEANS_OBJS) -o $(BIN_DIR)/benchmark_suite.exe -lm

benchmark_baselines: $(ALL_KMEANS_OBJS) benchmarks/benchmark.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) benchmarks/benchmark.c $(ALL_KMEANS_OBJS) -o $(BIN_DIR)/benchmark_baselines.exe -lm

run_all: test run_false_sharing run_scheduler run_threshold run_hard_comparison

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all build_all clean test test_engine test_synchronization test_naive test_static \
        benchmark_threshold run_threshold benchmark_false_sharing run_false_sharing \
        benchmark_scheduler run_scheduler hard_comparison_benchmark run_hard_comparison \
        benchmark_synchronization benchmark_suite benchmark_baselines run_all
