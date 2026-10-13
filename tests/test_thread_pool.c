/**
 * @file test_thread_pool.c
 * @brief Comprehensive test harness for Task Queue & Thread Pool runtime.
 */

#include "task_queue.h"
#include "thread_pool.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Global test counters */
static uint64_t g_tasks_executed = 0;
static pthread_mutex_t g_test_lock = PTHREAD_MUTEX_INITIALIZER;

/* Dummy task function */
static void dummy_task_fn(const Task *task, int tid, void *ctx) {
  (void)tid;
  (void)ctx;
  // Simulate minor work
  volatile int dummy = 0;
  for (size_t i = 0; i < (task->end_idx - task->start_idx) * 100; i++) {
    dummy += (int)i;
  }
  (void)dummy;

  pthread_mutex_lock(&g_test_lock);
  g_tasks_executed++;
  pthread_mutex_unlock(&g_test_lock);
}

/* ========================================================================= */
/* Test 1: Single-Threaded Ring-Buffer Queue Correctness                     */
/* ========================================================================= */
static void test_queue_single_threaded(void) {
  printf("[TEST 1] Single-Threaded Ring Buffer Queue... ");
  TaskQueue q;
  int rc = queue_init(&q, 4); // Will round up to 4
  assert(rc == QUEUE_OK);
  assert(q.cap == 4);
  assert(queue_is_empty(&q));

  // Push 4 elements
  for (size_t i = 0; i < 4; i++) {
    Task t = {.start_idx = i * 10,
              .end_idx = (i + 1) * 10,
              .task_id = (int)i,
              .ctx = NULL};
    assert(queue_push(&q, t) == QUEUE_OK);
  }
  assert(queue_size(&q) == 4);

  // Pop and verify FIFO ordering
  for (size_t i = 0; i < 4; i++) {
    Task out;
    assert(queue_pop(&q, &out) == true);
    assert(out.task_id == (int)i);
    assert(out.start_idx == i * 10);
  }
  assert(queue_is_empty(&q));

  // Test wrap-around past capacity boundary
  for (size_t i = 0; i < 10; i++) {
    Task t = {.start_idx = i, .end_idx = i + 1, .task_id = (int)i, .ctx = NULL};
    assert(queue_push(&q, t) == QUEUE_OK);
    Task out;
    assert(queue_pop(&q, &out) == true);
    assert(out.task_id == (int)i);
  }

  queue_destroy(&q);
  printf("PASSED\n");
}

/* ========================================================================= */
/* Test 2: Thread Pool Basic Execution & Dynamic Distribution               */
/* ========================================================================= */
static void test_pool_basic_execution(void) {
  printf("[TEST 2] Thread Pool Basic Execution & Distribution... ");
  pthread_mutex_lock(&g_test_lock);
  g_tasks_executed = 0;
  pthread_mutex_unlock(&g_test_lock);

  int num_threads = 4;
  size_t num_tasks = 400;
  ThreadPool *pool = pool_create(num_threads, 64, dummy_task_fn);
  assert(pool != NULL);

  for (size_t i = 0; i < num_tasks; i++) {
    Task t = {.start_idx = i * 10,
              .end_idx = (i + 1) * 10,
              .task_id = (int)i,
              .ctx = NULL};
    assert(pool_submit(pool, t) == QUEUE_OK);
  }

  pool_wait_iteration(pool);

  pthread_mutex_lock(&g_test_lock);
  assert(g_tasks_executed == num_tasks);
  pthread_mutex_unlock(&g_test_lock);

  // Verify all threads received work
  for (int t = 0; t < num_threads; t++) {
    const WorkerStats *st = pool_get_stats(pool, t);
    assert(st != NULL);
    assert(st->tasks_done > 0);
  }

  pool_shutdown(pool);
  printf("PASSED\n");
}

/* ========================================================================= */
/* Test 3: Multi-Iteration Reuse (Simulating K-Means Iteration Loop)         */
/* ========================================================================= */
static void test_pool_multi_iteration_reuse(void) {
  printf("[TEST 3] Multi-Iteration Reuse across 50 iterations... ");
  int num_threads = 4;
  int num_iterations = 50;
  size_t tasks_per_iter = 20;

  pthread_mutex_lock(&g_test_lock);
  g_tasks_executed = 0;
  pthread_mutex_unlock(&g_test_lock);

  ThreadPool *pool = pool_create(num_threads, 32, dummy_task_fn);
  assert(pool != NULL);

  for (int iter = 0; iter < num_iterations; iter++) {
    for (size_t i = 0; i < tasks_per_iter; i++) {
      Task t = {.start_idx = i * 5,
                .end_idx = (i + 1) * 5,
                .task_id = (int)i,
                .ctx = NULL};
      assert(pool_submit(pool, t) == QUEUE_OK);
    }
    pool_wait_iteration(pool);
  }

  pthread_mutex_lock(&g_test_lock);
  assert(g_tasks_executed == (uint64_t)num_iterations * tasks_per_iter);
  pthread_mutex_unlock(&g_test_lock);

  pool_print_stats(pool);
  pool_shutdown(pool);
  printf("PASSED\n");
}

/* ========================================================================= */
/* Test 4: Edge Cases (0 Tasks, 1 Task, Rapid Shutdown)                      */
/* ========================================================================= */
static void test_pool_edge_cases(void) {
  printf("[TEST 4] Edge Cases (0 tasks, 1 task, rapid shutdown)... ");

  // Case A: 0 tasks submitted
  ThreadPool *pool = pool_create(4, 16, dummy_task_fn);
  assert(pool != NULL);
  pool_wait_iteration(pool); // Must return immediately without hanging
  pool_shutdown(pool);

  // Case B: Exactly 1 task submitted with 8 threads
  pthread_mutex_lock(&g_test_lock);
  g_tasks_executed = 0;
  pthread_mutex_unlock(&g_test_lock);

  pool = pool_create(8, 16, dummy_task_fn);
  assert(pool != NULL);
  Task single = {.start_idx = 0, .end_idx = 10, .task_id = 1, .ctx = NULL};
  assert(pool_submit(pool, single) == QUEUE_OK);
  pool_wait_iteration(pool);

  pthread_mutex_lock(&g_test_lock);
  assert(g_tasks_executed == 1);
  pthread_mutex_unlock(&g_test_lock);
  pool_shutdown(pool);

  // Case C: Immediate creation and shutdown
  pool = pool_create(8, 16, dummy_task_fn);
  assert(pool != NULL);
  pool_shutdown(pool);

  printf("PASSED\n");
}

/* ========================================================================= */
/* Test 5: Batch Submission                                                  */
/* ========================================================================= */
static void test_pool_batch_submission(void) {
  printf("[TEST 5] Batch Submission Performance... ");
  pthread_mutex_lock(&g_test_lock);
  g_tasks_executed = 0;
  pthread_mutex_unlock(&g_test_lock);

  int num_threads = 4;
  size_t batch_size = 50;
  ThreadPool *pool = pool_create(num_threads, 64, dummy_task_fn);
  assert(pool != NULL);

  Task batch[50];
  for (size_t i = 0; i < batch_size; i++) {
    batch[i].start_idx = i * 10;
    batch[i].end_idx = (i + 1) * 10;
    batch[i].task_id = (int)i;
    batch[i].ctx = NULL;
  }

  assert(pool_submit_batch(pool, batch, batch_size) == QUEUE_OK);
  pool_wait_iteration(pool);

  pthread_mutex_lock(&g_test_lock);
  assert(g_tasks_executed == batch_size);
  pthread_mutex_unlock(&g_test_lock);

  pool_shutdown(pool);
  printf("PASSED\n");
}

int main(void) {
  printf("====================================================\n");
  printf(" Running Unit 1 Tests: Task Queue & Thread Pool\n");
  printf("====================================================\n");

  test_queue_single_threaded();
  test_pool_basic_execution();
  test_pool_multi_iteration_reuse();
  test_pool_edge_cases();
  test_pool_batch_submission();

  printf("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n");
  return 0;
}
