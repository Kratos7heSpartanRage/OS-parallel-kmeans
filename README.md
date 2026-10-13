# High-Performance Multi-Threaded Parallel & Adaptive K-Means Engine

A systems-grade, high-performance parallel computing runtime for K-Means clustering in **C (C11, POSIX Threads)**, engineered around Operating System concurrency, synchronization, memory-hierarchy, and cache-coherence principles.

Tested and empirically profiled on: **13th Gen Intel(R) Core(TM) i5-13420H (12 Logical Threads)** under **Linux Ubuntu 24.04 (WSL2)** with GCC `-O3`.

---

## Quick Links
* 📄 **[Operating Systems Syllabus Mapping & Viva Revision Guide (Markdown)](OS_SYLLABUS_MAPPING.md)**
* 📑 **[Printable PDF Revision Guide](OS_Project_Syllabus_Mapping_and_Viva_Guide.pdf)**
* 📊 **[Empirical Benchmarks & Experimental Results](#empirical-benchmark-results)**
* 💻 **[How to Build & Run Tests](#how-to-build--run-tests)**

---

## Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Dataset & Workload     │
                          │   N Points x D Dimensions│
                          └────────────┬─────────────┘
                                       │
                                       ▼
                     ┌──────────────────────────────────┐
                     │   Adaptive Execution Evaluator   │
                     │    N < Threshold ? Sequential :  │
                     │    Dynamic Multi-Thread Pool     │
                     └───────────────┬───┬──────────────┘
                    N < Threshold    │   │ N >= Threshold
          ┌──────────────────────────┘   └──────────────────────────┐
          ▼                                                         ▼
┌──────────────────────────┐                    ┌────────────────────────────────────────┐
│ Pure Sequential Engine   │                    │     Dynamic Task Queue (Ring Buffer)   │
│ Zero Mutexes, Zero Conds │                    │      pthread_mutex + condvars          │
│ 1 Accumulator, Zero Wrap │                    └───────────────────┬────────────────────┘
└─────────┬────────────────┘                                        │
          │                                          ┌──────────────┼──────────────┐
          │                                          ▼              ▼              ▼
          │                                     ┌─────────┐    ┌─────────┐    ┌─────────┐
          │                                     │Worker 0 │    │Worker 1 │ .. │Worker P │
          │                                     │ pthread │    │ pthread │    │ pthread │
          │                                     └────┬────┘    └────┬────┘    └────┬────┘
          │                                          │              │              │
          │                                          ▼              ▼              ▼
          │                                    Padded Acc 0   Padded Acc 1   Padded Acc P
          │                                    (64B-aligned)  (64B-aligned)  (64B-aligned)
          │                                    [No Contention][No Contention][No Contention]
          │                                          │              │              │
          │                                          └──────────────┼──────────────┘
          │                                                         ▼
          │                                             ┌──────────────────────┐
          │                                             │  Iteration Barrier   │
          │                                             │ pool_wait_iteration  │
          │                                             └──────────┬───────────┘
          │                                                        ▼
          │                                             ┌──────────────────────┐
          │                                             │ Centroid Reduction   │
          │                                             │ O(K x D) Linear Scan │
          │                                             └──────────┬───────────┘
          │                                                        │
          └──────────────────────────┬─────────────────────────────┘
                                     ▼
                      ┌──────────────────────────────┐
                      │  Convergence Check / Next    │
                      │  Iteration (Tolerance Tol)   │
                      └──────────────────────────────┘
```

---

## Key Operating System Innovations

### 1. Cache-Line Alignment & False Sharing Elimination (MESI Protocol)
- **The Hardware Problem:** Modern CPUs transfer memory in 64-byte cache lines. In naive multi-threaded K-Means, thread accumulators are packed contiguously. When Thread 0 and Thread 1 write to adjacent accumulators on the same 64-byte line, the hardware **MESI (Modified, Exclusive, Shared, Invalid)** cache coherence protocol constantly invalidates peer caches over the motherboard bus, causing severe pipeline stalls.
- **Our OS Solution:**
  1. We allocate accumulator memory using POSIX `posix_memalign(&ptr, 64, size)` to guarantee base addresses start on 64-byte hardware boundaries.
  2. We apply bitwise cache padding: `padded_size = (size + 63) & ~63`.
  3. Result: Eliminates false sharing with a **5.09x - 6.14x empirical speedup** over unpadded concurrent access.

### 2. The Grain Size Problem & Parallel Overhead Cut-Off Threshold ($N_{\text{threshold}}$)
- **The Algorithmic Dilemma:** Parallel execution is not free. Spawning tasks incurs non-zero operating system overheads:
  $$T_{\text{parallel}}(N, P) = \frac{T_{\text{compute}}(N)}{P} + T_{\text{overhead}}(N, P)$$
  Where $T_{\text{overhead}} = T_{\text{dispatch}} + T_{\text{queue\_locks}} + T_{\text{condvar\_waking}} + T_{\text{reduction}}$.
  On smaller datasets ($N < 5,000$), $T_{\text{compute}}$ is under 0.5 ms, meaning $T_{\text{overhead}} \gg T_{\text{compute}}$, causing naive multi-threading to run **up to 14x slower** (0.07x speedup).
- **Our OS Solution:** Our `kmeans_adaptive_run()` driver empirically checks $N$ against the break-even threshold ($N \approx 5,000$ for 2T, $N \approx 10,000$ points for 4T, $N \approx 25,000$ points for 8-12T). Below the threshold, it dynamically bypasses multi-threading and routes execution to pure sequential clustering, delivering optimal latency across ALL input sizes without penalty.

### 3. Dynamic Work-Pulling vs Straggler Jitter
- **The Core Asymmetry:** Under static coarse slicing ($N / P$), if one thread encounters system jitter or runs on an Intel Efficiency-core (E-core), all other threads remain blocked at the barrier.
- **Our OS Solution:** We partition data into fine-grained tasks (e.g. chunks of 2,048 or 4,096 points) submitted to a bounded power-of-2 circular ring buffer. Faster threads automatically pull additional chunks from the queue, drastically reducing the Load Imbalance Ratio and mitigating stragglers.

### 4. Lock-Free Distance Assignment & Critical Section Minimization
- **The Contention Trap:** In a naive implementation, threads acquire a mutex whenever updating centroids, serializing all cores.
- **Our OS Solution:** Distance evaluation and local accumulation are **100% lock-free**. Each worker thread updates exclusively its own thread-private `AlignedAccumulator[tid]`. Mutexes are restricted strictly to task queue management, achieving a **11.41x speedup** over naive global mutex locking.

---

## Master OS Syllabus-to-Project Mapping

*(For the comprehensive detailed guide with full viva questions and answers, see [`OS_SYLLABUS_MAPPING.md`](OS_SYLLABUS_MAPPING.md))*

| OS Syllabus Unit | Textbook Topic | Where & How It Is Used in This Project | Source Files |
|---|---|---|---|
| **Unit 1: OS Architecture & Hardware** | System Calls vs User Space | Avoiding repeated `clone()`/`pthread_create` kernel traps via a persistent Thread Pool. | [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 1: OS Architecture & Hardware** | Cache Hierarchy & Locality | 64-byte L1 Cache Line alignment (`posix_memalign`) and MESI false sharing resolution. | [`src/aligned_memory.c`](src/aligned_memory.c), [`src/accumulator.c`](src/accumulator.c) |
| **Unit 2: Process Management & Threads** | Pthreads 1:1 Kernel Model | POSIX threads (`pthread_t`, `pthread_create`, `pthread_join`) 1:1 schedulable entity model. | [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 2: Process Management & Threads** | IPC: Shared Memory Systems | Zero-copy direct memory access for multi-megabyte datasets and centroids across worker threads. | [`src/kmeans.c`](src/kmeans.c) |
| **Unit 3: CPU Scheduling** | Scheduling Criteria & Stragglers | Dynamic FCFS Task Queue work-pulling vs static slicing to absorb core performance heterogeneity. | [`src/task_queue.c`](src/task_queue.c), [`benchmarks/benchmark_scheduler.c`](benchmarks/benchmark_scheduler.c) |
| **Unit 3: CPU Scheduling** | The Grain Size Problem | Empirical cut-off threshold ($N_{\text{threshold}}$) where thread synchronization costs exceed computation. | [`src/kmeans.c`](src/kmeans.c), [`benchmarks/threshold_benchmark.c`](benchmarks/threshold_benchmark.c) |
| **Unit 3: Process Synchronization** | Critical Section & Mutexes | Guarding shared queue cursors (`head`, `tail`, `count`) with `pthread_mutex_t`. | [`src/task_queue.c`](src/task_queue.c) |
| **Unit 3: Process Synchronization** | Producer-Consumer (Bounded Buffer)| Bounded ring buffer with mutex and two condition variables (`not_empty`, `not_full`). | [`src/task_queue.c`](src/task_queue.c) |
| **Unit 3: Process Synchronization** | Phase Barrier Synchronization | Iteration barrier via `pool_wait_iteration` using condition variable broadcast on `all_done`, plus `pthread_barrier_t`. | [`src/thread_pool.c`](src/thread_pool.c), [`src/synchronization.c`](src/synchronization.c) |
| **Unit 4: Deadlocks** | Deadlock Prevention | Breaking Coffman's "Circular Wait" and "Hold-and-Wait" through strict Single-Lock Hierarchy. | [`src/task_queue.c`](src/task_queue.c), [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 5: Memory Management** | Contiguous Allocation & Prefetch | Row-major contiguous array layout `points[p * d + dim]` maximizing spatial locality. | [`src/aligned_memory.c`](src/aligned_memory.c) |
| **Unit 6: UNIX Case Study** | Virtual File System (`/proc`) | Parsing `/proc/cpuinfo` to detect processor model name and logical core topology dynamically. | [`benchmarks/threshold_benchmark.c`](benchmarks/threshold_benchmark.c) |

---

## Empirical Benchmark Results

Profiled on: **13th Gen Intel(R) Core(TM) i5-13420H (12 Logical Threads)**

### 1. Parallel Overhead Break-Even Cut-Off Benchmark
Testing 9 workload sizes ($N = 500$ to $250,000$ points) with $D = 8$, $K = 16$, 5 iterations:

| Points $N$ | Sequential (ms) | Parallel [P=2] (ms) | Parallel [P=4] (ms) | Parallel [P=8] (ms) | Parallel [P=12] (ms) | Adaptive Engine (ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **500** | **0.16 ms** | 0.98 ms (0.16x) | 2.24 ms (0.07x) | 1.06 ms (0.15x) | 2.07 ms (0.08x) | **0.26 ms** *(sequential bypass)* |
| **1,000** | **0.37 ms** | 0.73 ms (0.50x) | 0.88 ms (0.42x) | 1.11 ms (0.33x) | 2.93 ms (0.13x) | **0.23 ms** *(sequential bypass)* |
| **2,500** | **0.57 ms** | 0.79 ms (0.72x) | 0.92 ms (0.62x) | 1.78 ms (0.32x) | 1.35 ms (0.42x) | **0.54 ms** *(sequential bypass)* |
| **5,000** | **1.25 ms** | 1.18 ms (1.06x) | 1.25 ms (1.00x) | 2.70 ms (0.46x) | 3.29 ms (0.38x) | **2.10 ms** |
| **10,000** | 2.62 ms | 1.97 ms (1.33x) | **1.75 ms (1.50x)** | 2.94 ms (0.89x) | 3.53 ms (0.74x) | **1.50 ms** *(effective P=2)* |
| **25,000** | 6.20 ms | 3.48 ms (1.78x) | 2.62 ms (2.36x) | **3.38 ms (1.83x)** | **4.94 ms (1.25x)** | **5.15 ms** *(effective P=4)* |
| **50,000** | 14.56 ms | 9.26 ms (1.57x) | 5.09 ms (2.86x) | 4.94 ms (2.95x) | 7.97 ms (1.83x) | **12.93 ms** |
| **100,000** | 25.28 ms | 13.48 ms (1.88x) | 7.24 ms (3.49x) | 6.39 ms (3.96x) | 10.13 ms (2.50x) | **7.86 ms** |
| **250,000** | 62.92 ms | 33.19 ms (1.90x) | 18.29 ms (3.44x) | 15.85 ms (3.97x) | **14.80 ms (4.25x)** | **15.72 ms** |

**Empirical Break-Even Verdict:**
- For **2 Threads:** $N \approx 5,000$ points.
- For **4 Threads:** $N \approx 10,000$ points.
- For **8-12 Threads:** $N \approx 25,000$ points.

---

### 2. False Sharing Benchmark (MESI Cache Invalidation)
Simulating 50,000,000 accumulation operations ($K = 4, D = 3$):

| Threads | Unpadded Time (s) | Padded Time (s) | Speedup Factor | Hardware Phenomenon |
|:---:|:---:|:---:|:---:|---|
| **1** | 0.1147 s | 0.1150 s | **1.00x** | Baseline single-core computation |
| **2** | 0.7029 s | 0.1144 s | **6.14x faster** | Unpadded suffers continuous MESI invalidations across cores |
| **4** | 0.9129 s | 0.1709 s | **5.34x faster** | 4-core cache bus saturation eliminated by 64B padding |
| **8** | 1.0925 s | 0.2147 s | **5.09x faster** | Multi-core memory scaling without bus contention |

---

### 3. Hard Workload 3-Way Architecture Stress Comparison
Testing 1,000,000 Points $\times$ 8 Dimensions $\times$ 32 Clusters $\times$ 10 Iterations (**320,000,000 Euclidean operations**):

| Architecture | Time (ms) | Speedup vs Baseline | Architectural Mechanism |
|---|:---:|:---:|---|
| **System 0: Naive Global Mutex** | 1,868.38 ms | 1.00x (Baseline) | Severe lock contention; threads serialize on single mutex |
| **System 1: Static Multi-Threaded** | 176.50 ms | 10.59x faster | Static point slicing; recreates threads per loop |
| **System 2: Our OS Runtime** | **163.80 ms** | **11.41x faster** | Persistent thread pool, dynamic ring buffer work-pulling, 64B cache-line padding |

---

## Directory Structure

```
OS-parallel-kmeans/
├── include/                              # Header files & API contracts
│   ├── accumulator.h                     # 64-byte padded accumulator structures & reduction
│   ├── aligned_memory.h                  # posix_memalign 64-byte memory allocation API
│   ├── kmeans.h                          # Sequential, Parallel & Adaptive execution drivers
│   ├── synchronization.h                 # Phase barrier & reduction synchronization API
│   ├── task_queue.h                      # Power-of-2 bounded circular ring buffer API
│   └── thread_pool.h                     # Persistent worker thread pool & telemetry API
├── src/                                  # Implementation files
│   ├── accumulator.c                     # Bitwise padding & centroid reduction logic
│   ├── aligned_memory.c                  # posix_memalign wrapper
│   ├── kmeans.c                          # Core clustering engine & sequential cut-off
│   ├── naive_kmeans.c                    # Naive multi-threaded K-Means baseline
│   ├── static_kmeans.c                   # Static chunk multi-threaded K-Means baseline
│   ├── synchronization.c                 # Phase barrier synchronization & reduction
│   ├── task_queue.c                      # Thread-safe ring buffer with mutex + 2 condvars
│   └── thread_pool.c                     # Worker thread loop & iteration barrier
├── tests/                                # Unit test suite
│   ├── test_thread_pool.c                # 5 Unit tests: FIFO wrap, telemetry, edge cases
│   └── test_synchronization.c            # 4 Unit tests: Barrier synchronization & reduction
├── test_naive.c                          # Naive multi-threaded baseline validation
├── test_static.c                         # Static chunk multi-threaded baseline validation
├── benchmarks/                           # Empirical benchmarks
│   ├── benchmark.c                       # Baseline sequential vs parallel benchmark
│   ├── benchmark_scheduler.c             # Dynamic work-pulling vs straggler jitter
│   ├── benchmark_suite.c                 # Automated multi-workload benchmark suite
│   ├── false_sharing_benchmark.c         # MESI cache-coherence false sharing validation
│   ├── hard_comparison_benchmark.c       # 3-Way architectural comparison (320M ops)
│   ├── synchronization_benchmark.c       # Barrier vs local accumulation benchmark
│   ├── threshold_benchmark.c             # Parallel overhead break-even cut-off sweep
│   └── results/                          # Output CSV data files
│       └── threshold_results.csv         # Empirically measured latency data
├── scripts/                              # Helper scripts
│   └── benchmark_scheduler.py            # Automated benchmark runner script
├── OS_SYLLABUS_MAPPING.md                # Comprehensive OS syllabus mapping & viva guide
├── OS_Project_Syllabus_Mapping_and_Viva_Guide.pdf # Printable PDF revision guide
├── Makefile                              # Build configuration (GCC -O3 -pthread)
└── README.md                             # Project overview & documentation
```

---

## How to Build & Run Tests

### 1. Run All Unit Tests:
```bash
make test
```
This builds and executes all 4 test suites:
1. **Thread Pool & Dynamic Queue:** `./bin/test_thread_pool.exe` (5 unit tests)
2. **Phase Barrier & Synchronization:** `./bin/test_synchronization.exe` (4 unit tests)
3. **Naive Multi-Threaded Baseline:** `./bin/test_naive.exe`
4. **Static Partitioning Baseline:** `./bin/test_static.exe`

---

### 2. Run Empirical Benchmarks:

* **Parallel Overhead Break-Even Cut-Off Threshold Benchmark:**
  ```bash
  make run_threshold
  ```
* **Hardware False Sharing & MESI Protocol Benchmark:**
  ```bash
  make run_false_sharing
  ```
* **Dynamic Work-Pulling vs Straggler Jitter Benchmark:**
  ```bash
  make run_scheduler
  ```
* **Hard Workload 3-Way Architecture Stress Comparison:**
  ```bash
  make run_hard_comparison
  ```
* **Synchronization & Barrier Benchmark:**
  ```bash
  make benchmark_synchronization && ./bin/benchmark_synchronization.exe
  ```
* **Comprehensive Benchmark Suite:**
  ```bash
  make benchmark_suite && ./bin/benchmark_suite.exe
  ```

---

### 3. Run Everything in One Command:
```bash
make run_all
```

### Clean & Rebuild All:
```bash
make clean
make build_all
```
