# Operating Systems Course Syllabus Mapping & Viva Revision Guide

**Project:** High-Performance Multi-Threaded Parallel & Adaptive K-Means Engine  
**Course:** Operating Systems (Sem 5)  
**Hardware Profiled:** 13th Gen Intel(R) Core(TM) i5-13420H (12 Logical Threads) under Linux/WSL2  
**Standards:** C11, POSIX Threads (`pthreads`), GCC `-O3`  
**PDF Printable Guide:** [`OS_Project_Syllabus_Mapping_and_Viva_Guide.pdf`](OS_Project_Syllabus_Mapping_and_Viva_Guide.pdf)

---

## Master Syllabus-to-Project Mapping Table

| OS Unit | Textbook Topic | Where & How It Is Used in This Project | Relevant Files |
|---|---|---|---|
| **Unit 1: OS & Hardware Interaction** | System Calls vs User Space | Avoiding repeated `clone()`/`pthread_create` kernel traps by keeping a persistent Thread Pool alive in user space across iterations. | [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 1: OS & Hardware Interaction** | Cache & Memory Hierarchy | 64-byte L1 Cache Line alignment (`posix_memalign`) and MESI cache coherence false sharing resolution. | [`src/accumulator.c`](src/accumulator.c), [`src/aligned_memory.c`](src/aligned_memory.c) |
| **Unit 2: Process & Threads** | Thread Libraries (Pthreads) | POSIX threads (`pthread_t`, `pthread_create`, `pthread_join`) 1:1 kernel thread execution model. | [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 2: Process & Threads** | IPC: Shared Memory Systems | All threads share contiguous heap arrays (dataset points and centroids) with zero memory copy overhead. | [`src/kmeans.c`](src/kmeans.c) |
| **Unit 3: CPU Scheduling** | Scheduling Criteria & Stragglers | Static chunk distribution vs. Dynamic FCFS Task Queue work-pulling to balance varying CPU core latencies (P-cores vs E-cores). | [`src/task_queue.c`](src/task_queue.c), [`benchmarks/benchmark_scheduler.c`](benchmarks/benchmark_scheduler.c) |
| **Unit 3: CPU Scheduling** | Granularity & Parallel Cut-off | The Grain Size Problem: Task queue & thread wakeup overhead vs compute time (Empirical Sequential Cut-off). | [`src/kmeans.c`](src/kmeans.c), [`benchmarks/threshold_benchmark.c`](benchmarks/threshold_benchmark.c) |
| **Unit 3: Process Synchronization** | Critical-Section & Race Conditions | Guarding shared queue cursors (`head`, `tail`, `count`) using `pthread_mutex_t` mutual exclusion. | [`src/task_queue.c`](src/task_queue.c) |
| **Unit 3: Process Synchronization** | Producer-Consumer (Bounded Buffer) | Bounded circular ring buffer pattern with Mutex and 2 Condition Variables (`not_empty`, `not_full`). | [`src/task_queue.c`](src/task_queue.c) |
| **Unit 3: Process Synchronization** | Phase Barrier Synchronization | Iteration barrier implemented via `pool_wait_iteration` using `pthread_cond_broadcast` on `all_done`, plus `pthread_barrier_t` in [`src/synchronization.c`](src/synchronization.c). | [`src/thread_pool.c`](src/thread_pool.c), [`src/synchronization.c`](src/synchronization.c) |
| **Unit 4: Deadlocks** | Deadlock Prevention | Breaking Coffman's "Circular Wait" and "Hold-and-Wait" through a strict Single-Lock Hierarchy. | [`src/task_queue.c`](src/task_queue.c), [`src/thread_pool.c`](src/thread_pool.c) |
| **Unit 5: Memory Management** | Contiguous Allocation & Prefetch | Row-major contiguous dataset storage for spatial locality prefetching; 64-byte aligned allocation via `posix_memalign`. | [`src/aligned_memory.c`](src/aligned_memory.c) |
| **Unit 6: UNIX Case Study** | Virtual File System (`/proc`) | Parsing `/proc/cpuinfo` in the UNIX virtual pseudo-filesystem to detect CPU hardware model and logical threads dynamically. | [`benchmarks/threshold_benchmark.c`](benchmarks/threshold_benchmark.c) |

---

## Detailed Breakdown & Viva Voce Answers by Unit

### UNIT 1: Introduction, OS Architecture & Hardware Interaction
**Concept:** How the Operating System abstracts hardware and mediates between CPU registers, Caches, RAM, and system calls.

#### In Our Project:
1. **User Mode vs. Kernel Mode Overhead:** Creating threads repeatedly inside each clustering iteration invokes `clone()` system calls, forcing expensive hardware traps from User Mode into Kernel Mode and context switching. Our Thread Pool initializes worker threads once at program start and keeps them running in user space, saving thousands of context switches.
2. **Hardware Cache Lines & Spatial Locality:** L1/L2 caches transfer memory in 64-byte blocks. When two threads write to adjacent accumulators residing on the same 64-byte cache line, the CPU's hardware **MESI (Modified, Exclusive, Shared, Invalid)** protocol continually invalidates peer caches over the motherboard interconnect bus, destroying multi-core scalability (False Sharing). We resolved this via `posix_memalign` and bitwise padding `(size + 63) & ~63`.

#### Viva Voce Questions & Ideal Answers:
* **Q1: "What hardware-level property did you optimize for in your accumulators?"**  
  * **Answer:** *"The CPU L1 Cache Line size, which is 64 bytes on modern x86-64 and ARM processors. We padded each thread's local accumulator to a multiple of 64 bytes using bitwise math `(size + 63) & ~63` and aligned memory addresses with `posix_memalign`. This completely eliminates False Sharing and MESI invalidation bus traffic, yielding an empirical **5x to 6x speedup** on multi-threaded runs."*
* **Q2: "Why is creating threads per iteration an OS design flaw?"**  
  * **Answer:** *"Spawning threads repeatedly invokes the `clone()` system call in Linux. Every system call triggers a trap from User Mode to Kernel Mode, kernel page table allocations for thread stacks, and scheduler book-keeping. By using a persistent Thread Pool, threads stay warm in user space, eliminating per-iteration context switch penalties."*

---

### UNIT 2: Process Management, Threads & Inter-Process Communication
**Concept:** Process vs. Thread, Shared Memory IPC, and the POSIX Threads library.

#### In Our Project:
1. **1:1 Threading Model:** We implement POSIX Threads (`pthread_t`) under Linux's 1:1 kernel thread model. Each worker thread is an independent kernel schedulable entity with its own CPU registers and private stack, while sharing the process heap.
2. **Shared Memory Architecture:** The dataset point array and centroid coordinates live in shared heap memory. All worker threads read directly from these buffers with zero memory duplication or IPC messaging overhead.

#### Viva Voce Questions & Ideal Answers:
* **Q3: "Why choose multi-threading over multi-processing (`fork()`) for this clustering engine?"**  
  * **Answer:** *"Threads share the virtual address space (heap). Because K-Means requires all workers to read the multi-megabyte dataset and share centroid positions on every iteration, threads allow zero-copy direct memory access. `fork()` would spawn separate processes with isolated address spaces, requiring either copy-on-write page fault overhead or explicit POSIX shared memory segments (`shmget`/`mmap`)."*

---

### UNIT 3: CPU Scheduling & Process Synchronization (Core Viva Topic)

#### Part A: CPU Scheduling
1. **Dynamic Work-Pulling vs. Static Partitioning:** Under static scheduling ($N/P$), if one thread gets stalled by operating system background tasks or runs on an Intel Efficiency-core (E-core), all other threads sit idle at the barrier. With our Dynamic Ring Buffer Queue, points are divided into smaller chunks (e.g. 2,048 or 4,096 points); faster threads automatically pull subsequent chunks, minimizing the Load Imbalance Ratio.
2. **The Grain Size Problem & Parallel Overhead Cut-Off Threshold ($N_{\text{threshold}}$):** Parallel computing is not free. When the dataset size $N$ is small ($N < 5,000$), the computation time is under 0.5 ms. The time spent locking queue mutexes, waking threads with condition variables, and reducing accumulators exceeds the calculation time itself:
   $$T_{\text{parallel}}(N, P) = \frac{T_{\text{compute}}(N)}{P} + T_{\text{overhead}}(N, P)$$
   Making multi-threading **up to 14x slower** (0.07x speedup).
3. **Adaptive Fallback Engine:** Our `kmeans_adaptive_run()` driver checks $N$ against the break-even threshold ($N \approx 5,000$ for 2T, $N \approx 10,000$ for 4T, $N \approx 25,000$ for 12T). Below the threshold, it dynamically bypasses multi-threading and runs sequential clustering with zero threading overhead.

#### Part B: Process Synchronization
1. **The Critical-Section Problem:** Task queue indices (`head`, `tail`, `count`) are shared mutable state, guarded by `pthread_mutex_t lock`.
2. **Producer-Consumer Bounded Buffer Pattern:** Implemented in [`src/task_queue.c`](src/task_queue.c). The main thread acts as the Producer (`queue_push`), and worker threads act as Consumers (`queue_pop`). When the queue is empty, workers sleep on `pthread_cond_wait(&not_empty, &lock)` without burning CPU cycles.
3. **Lock-Free Inner Compute Loop:** Distance evaluations and point accumulations invoke **zero mutex locks**. Each thread accumulates into its own thread-private 64-byte padded accumulator, synchronizing only once per iteration at the iteration barrier (`pool_wait_iteration`).

#### Viva Voce Questions & Ideal Answers:
* **Q4: "How did you solve the Producer-Consumer problem in your Task Queue?"**  
  * **Answer:** *"We implemented a bounded circular ring buffer in `task_queue.c` guarded by a mutex and two condition variables: `not_empty` and `not_full`. When the queue is empty, worker threads sleep on `pthread_cond_wait(&not_empty, &lock)` without burning CPU cycles, and wake up as soon as tasks are submitted."*
* **Q5: "What causes parallel computing to become slower on small inputs, and how did you fix it?"**  
  * **Answer:** *"This is the **Grain Size Problem** in CPU scheduling. When dataset size $N$ is small ($N < 5,000$), arithmetic compute time is fractions of a millisecond. The overhead of task queue mutex locks, condition variable wakeups, and accumulator reduction dominates the runtime, making multi-threading up to 14x slower than single-threaded code. We empirically measured crossover break-even points on our Intel Core i5 processor ($N \approx 10,000$ for 4 threads, $N \approx 25,000$ for 12 threads) and implemented `kmeans_adaptive_run()`, which automatically routes small workloads to single-threaded sequential execution, eliminating all synchronization penalties."*
* **Q6: "Why don't your worker threads lock a mutex when updating cluster centroids?"**  
  * **Answer:** *"Because acquiring a mutex inside the inner distance calculation loop (evaluated 320,000,000 times) creates severe lock contention. In our benchmark, a naive global mutex baseline took **1,868 ms**, while our lock-free thread-local accumulator architecture took **163 ms (11.41x faster)**. Threads accumulate into private padded buffers and reduce partial sums once at the barrier."*

---

### UNIT 4: Deadlocks
**Concept:** System Model, Coffman Conditions (Mutual Exclusion, Hold and Wait, No Preemption, Circular Wait), and Deadlock Prevention.

#### In Our Project:
1. **Deadlock Prevention via Single-Lock Hierarchy:** Our architecture enforces a strict lock ordering discipline: **no thread ever holds more than one lock simultaneously**. Workers acquire `q->lock` while popping tasks, release it, execute compute unlocked, and acquire `pool->done_lock` only when decrementing the completion counter. This mathematically breaks Coffman's **Hold-and-Wait** and **Circular Wait** conditions!
2. **Termination Liveness:** In `queue_close()`, setting `q->closed = true` is immediately paired with `pthread_cond_broadcast(&q->not_empty)`. This wakes all sleeping workers so they exit cleanly without hanging indefinitely.

#### Viva Voce Questions & Ideal Answers:
* **Q7: "How does your thread pool design guarantee that worker threads will never deadlock?"**  
  * **Answer:** *"We prevent deadlock by breaking the Hold-and-Wait and Circular Wait conditions through a Single-Lock Hierarchy. No function ever holds multiple mutexes simultaneously. Furthermore, `queue_close()` broadcasts to all waiting condition variables upon shutdown so no thread blocks indefinitely."*

---

### UNIT 5: Memory Management & Virtual Memory
**Concept:** Contiguous Allocation, Spatial Locality, Paging, Alignment, and Virtual Memory Thrashing.

#### In Our Project:
1. **Contiguous Row-Major Allocation:** Points are laid out contiguously: `points[p * d + dim]`. This maximizes **spatial locality**—when core 0 loads dimension 0, the hardware prefetcher pulls subsequent dimensions into the L1 cache in a single 64-byte burst.
2. **64-Byte Hardware Alignment:** We invoke `posix_memalign` to allocate memory at addresses that are exact multiples of 64 bytes.
3. **Stress Testing Without Thrashing:** Our benchmark tests up to 1,000,000 8-D points (64 MB). This exceeds the CPU's 16 MB L3 cache to stress RAM bandwidth while remaining well within physical RAM limits, preventing page faults and disk thrashing.

#### Viva Voce Questions & Ideal Answers:
* **Q8: "How does your memory layout exploit spatial locality and hardware prefetching?"**  
  * **Answer:** *"Points are stored contiguously in row-major order. When a core evaluates Euclidean distance, the CPU hardware prefetcher recognizes the sequential memory access pattern and loads adjacent dimensions into L1 cache in a single 64-byte burst. Using `posix_memalign` ensures memory allocations align with hardware cache line boundaries, preventing cache line split penalties."*

---

### UNIT 6: UNIX Case Study & File Systems
**Concept:** UNIX Virtual File System (VFS), `/proc` Pseudo-Filesystem, and POSIX I/O System Calls.

#### In Our Project:
1. **UNIX `/proc` Pseudo-Filesystem:** In [`benchmarks/threshold_benchmark.c`](benchmarks/threshold_benchmark.c), we open and parse `/proc/cpuinfo` to inspect hardware topology (extracting the CPU model name and logical core count dynamically via the UNIX kernel interface).
2. **POSIX File I/O:** Standard POSIX file streams (`fopen`, `fprintf`, `fclose`) record empirical measurements directly to CSV files (`benchmarks/results/threshold_results.csv`).

#### Viva Voce Questions & Ideal Answers:
* **Q9: "How did your project interact with the UNIX operating system interface?"**  
  * **Answer:** *"We interacted with the UNIX kernel through the `/proc` virtual pseudo-filesystem by parsing `/proc/cpuinfo` to dynamically detect processor model and core topology, and used POSIX system calls (`sysconf(_SC_NPROCESSORS_ONLN)`, `clock_gettime(CLOCK_MONOTONIC)`) and standard file I/O to log benchmarking telemetry."*

---

## 30-Second Viva Voce "Cheat Sheet"

* **False Sharing Solution:** 64-byte cache line alignment via `posix_memalign` + bitwise padding `(size + 63) & ~63`. (5.09x–6.14x speedup).
* **Small Workload Overhead:** The Grain Size Problem. Mutex queue locks and thread wakeups exceed compute time when $N < 5,000$. Solved with `kmeans_adaptive_run()`.
* **Lock Contention Elimination:** Thread-Local Padded Accumulators; zero mutexes inside distance calculation loops. (11.41x faster than global mutex).
* **Straggler Mitigation:** Dynamic work-pulling ring buffer queue absorbs latency fluctuations between fast/slow cores.
* **Deadlock Prevention:** Single-Lock Hierarchy; workers never hold more than one mutex at any time (breaks Coffman's Circular Wait).
* **Thread Model:** POSIX Threads 1:1 kernel model; persistent thread pool avoids repeated `clone()` system calls.
