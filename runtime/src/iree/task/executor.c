// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/executor.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/internal/debugging.h"
#include "iree/base/internal/math.h"
#include "iree/task/affinity_set.h"
#include "iree/task/executor_impl.h"
#include "iree/task/tuning.h"
#include "iree/task/worker.h"

static void iree_task_executor_destroy(iree_task_executor_t* executor);

void iree_task_executor_options_initialize(
    iree_task_executor_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
}

// Returns the size of the worker local memory required by |group| in bytes.
// We don't want destructive sharing between workers so ensure we are aligned to
// at least the destructive interference size, even if a bit larger than what
// the user asked for or the device supports.
static iree_host_size_t iree_task_topology_group_local_memory_size(
    iree_task_executor_options_t options,
    const iree_task_topology_group_t* group) {
  iree_host_size_t worker_local_memory_size = options.worker_local_memory_size;
  if (!worker_local_memory_size) {
    worker_local_memory_size = group->caches.l2_data;
  }
  if (!worker_local_memory_size) {
    worker_local_memory_size = group->caches.l1_data;
  }
  return iree_host_align(worker_local_memory_size,
                         iree_hardware_destructive_interference_size);
}

iree_status_t iree_task_executor_create(iree_task_executor_options_t options,
                                        const iree_task_topology_t* topology,
                                        iree_allocator_t allocator,
                                        iree_task_executor_t** out_executor) {
  iree_host_size_t worker_count = iree_task_topology_group_count(topology);
  if (worker_count > IREE_TASK_EXECUTOR_MAX_WORKER_COUNT) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "requested %" PRIhsz
                            " workers but must be in [1, %d)",
                            worker_count, IREE_TASK_EXECUTOR_MAX_WORKER_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_executor);
  *out_executor = NULL;

  // The executor is followed in memory by worker[] + worker_local_memory[].
  iree_host_size_t total_worker_local_memory_size = 0;
  for (iree_host_size_t i = 0; i < worker_count; ++i) {
    total_worker_local_memory_size +=
        iree_task_topology_group_local_memory_size(
            options, iree_task_topology_get_group(topology, i));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)total_worker_local_memory_size);

  iree_host_size_t executor_base_size =
      iree_host_align(sizeof(iree_task_executor_t),
                      iree_hardware_destructive_interference_size);
  iree_host_size_t worker_list_size =
      iree_host_align(worker_count * sizeof(iree_task_worker_t),
                      iree_hardware_destructive_interference_size);
  iree_host_size_t executor_size =
      executor_base_size + worker_list_size + total_worker_local_memory_size;

  iree_task_executor_t* executor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, executor_size, (void**)&executor));
  memset(executor, 0, executor_size);
  iree_atomic_ref_count_init(&executor->ref_count);
  executor->allocator = allocator;
  executor->node_id = topology->node_id;
  executor->scheduling_mode = options.scheduling_mode;
  executor->worker_spin_ns = options.worker_spin_ns;
  iree_task_process_slist_initialize(&executor->immediate_list);
  iree_task_process_slist_initialize(&executor->compute_overflow);

  IREE_TRACE({
    static iree_atomic_int32_t executor_id = IREE_ATOMIC_VAR_INIT(0);
    char trace_name[32];
    int trace_name_length = iree_snprintf(
        trace_name, sizeof(trace_name), "iree-executor-%d",
        iree_atomic_fetch_add(&executor_id, 1, iree_memory_order_seq_cst));
    IREE_LEAK_CHECK_DISABLE_PUSH();
    executor->trace_name = malloc(trace_name_length + 1);
    memcpy((void*)executor->trace_name, trace_name, trace_name_length + 1);
    IREE_LEAK_CHECK_DISABLE_POP();
    IREE_TRACE_SET_PLOT_TYPE(executor->trace_name,
                             IREE_TRACING_PLOT_TYPE_PERCENTAGE, /*step=*/true,
                             /*fill=*/true, /*color=*/0xFF1F883Du);
    IREE_TRACE_PLOT_VALUE_F32(executor->trace_name, 0.0f);
  });

  // Simple PRNG used to generate seeds for the per-worker PRNGs used to
  // distribute work. This isn't strong (and doesn't need to be); it's just
  // enough to ensure each worker gets a sufficiently random seed for itself to
  // then generate entropy with. As a hack we use out_executor's address, as
  // that should live on the caller stack and with ASLR that's likely pretty
  // random itself. I'm sure somewhere a mathematician just cringed :)
  iree_prng_splitmix64_state_t seed_prng;
  iree_prng_splitmix64_initialize(/*seed=*/(uint64_t)(out_executor),
                                  &seed_prng);

  iree_status_t status = iree_ok_status();

  // Bring up the workers; the threads will be created here but be suspended
  // (if the platform supports it) awaiting the first tasks getting scheduled.
  if (iree_status_is_ok(status)) {
    executor->worker_base_index = options.worker_base_index;
    executor->worker_count = worker_count;
    executor->workers =
        (iree_task_worker_t*)((uint8_t*)executor + executor_base_size);
    uint8_t* worker_local_memory =
        (uint8_t*)executor->workers + worker_list_size;

    iree_task_affinity_set_t worker_mask =
        iree_task_affinity_set_ones(worker_count);

    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      const iree_task_topology_group_t* group =
          iree_task_topology_get_group(topology, i);
      iree_host_size_t worker_local_memory_size =
          iree_task_topology_group_local_memory_size(options, group);
      iree_task_worker_t* worker = &executor->workers[i];
      status = iree_task_worker_initialize(
          executor, i, group, options.worker_stack_size,
          iree_make_byte_span(worker_local_memory, worker_local_memory_size),
          &seed_prng, worker);
      worker_local_memory += worker_local_memory_size;
      if (!iree_status_is_ok(status)) break;
    }

    iree_atomic_task_affinity_set_store(&executor->worker_idle_mask,
                                        worker_mask, iree_memory_order_release);
    iree_atomic_task_affinity_set_store(&executor->worker_live_mask,
                                        worker_mask, iree_memory_order_release);
    iree_atomic_store(&executor->worker_idle_count, (int32_t)worker_count,
                      iree_memory_order_release);
  }

  if (!iree_status_is_ok(status)) {
    // NOTE: destroy will ensure that any workers we have initialized are
    // properly cleaned up.
    iree_task_executor_destroy(executor);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_executor = executor;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_task_executor_destroy(iree_task_executor_t* executor) {
  if (!executor) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // First ask all workers to exit. We do this prior to waiting on them to exit
  // so that we parallelize the shutdown logic (which may flush pending tasks).
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_request_exit(worker);
  }

  // Now that all workers should be in the process of exiting we can join with
  // them. Some may take longer than others to exit but that's fine as we can't
  // return from here until they exit anyway.
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_await_exit(worker);
  }

  // Tear down all workers now that no more threads are live. Any live threads
  // may still be touching their own data structures or those of others (for
  // example when trying to steal work).
  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    iree_task_worker_deinitialize(worker);
  }

  iree_task_process_slist_deinitialize(&executor->immediate_list);
  iree_task_process_slist_deinitialize(&executor->compute_overflow);
  iree_allocator_free(executor->allocator, executor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_executor_retain(iree_task_executor_t* executor) {
  if (executor) {
    iree_atomic_ref_count_inc(&executor->ref_count);
  }
}

void iree_task_executor_release(iree_task_executor_t* executor) {
  if (executor && iree_atomic_ref_count_dec(&executor->ref_count) == 1) {
    iree_task_executor_destroy(executor);
  }
}

iree_task_topology_node_id_t iree_task_executor_node_id(
    iree_task_executor_t* executor) {
  return executor->node_id;
}

void iree_task_executor_trim(iree_task_executor_t* executor) {
  // Placeholder for future cache/memory trimming.
}

iree_host_size_t iree_task_executor_worker_count(
    iree_task_executor_t* executor) {
  return executor->worker_count;
}

// Seeds the wake tree by adding |count| to the desired_wake counter and
// waking one idle worker. The woken worker will claim a share of
// desired_wake and propagate wakes to additional workers (see
// iree_task_worker_relay_wake in worker.c), forming a tree that fills in
// log2(N) rounds with IREE_TASK_WAKE_FANOUT.
//
// If no idle workers are found, posts to one live worker so it loops back
// and picks up the desired_wake on its next pump iteration.
void iree_task_executor_wake_workers(iree_task_executor_t* executor,
                                     int32_t count) {
  if (count <= 0) return;

  // Add to the desired wake counter. Workers claim from this in relay_wake.
  iree_atomic_fetch_add(&executor->desired_wake, count,
                        iree_memory_order_release);

  // Seed the tree: wake one idle worker to start the cascade.
  iree_task_affinity_set_t idle_mask = iree_atomic_task_affinity_set_load(
      &executor->worker_idle_mask, iree_memory_order_relaxed);
  int idle_target = iree_task_affinity_set_find_first(idle_mask);
  if (idle_target >= 0 && idle_target < (int)executor->worker_count) {
    iree_notification_post(&executor->workers[idle_target].wake_notification,
                           1);
    return;
  }

  // No idle workers found. Post to any live worker so it loops back and
  // picks up desired_wake on its next iteration.
  iree_task_affinity_set_t live_mask = iree_atomic_task_affinity_set_load(
      &executor->worker_live_mask, iree_memory_order_relaxed);
  int live_target = iree_task_affinity_set_find_first(live_mask);
  if (live_target < 0) return;  // Shutdown.
  if (live_target < (int)executor->worker_count) {
    iree_notification_post(&executor->workers[live_target].wake_notification,
                           1);
  }
}

// Tries to place a process into the first available compute slot. Returns true
// if placed, false if all slots are occupied.
bool iree_task_executor_try_place_in_compute_slot(
    iree_task_executor_t* executor, iree_task_process_t* process) {
  for (iree_host_size_t i = 0; i < IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS; ++i) {
    intptr_t expected = 0;
    if (iree_atomic_compare_exchange_strong(
            &executor->compute_slots[i].process, &expected, (intptr_t)process,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      return true;
    }
  }
  return false;
}

// Places a process into a compute slot, or pushes it to the overflow list if
// all slots are occupied. Overflow processes are promoted into slots as slots
// are released by workers (see release_compute_process in worker.c).
static void iree_task_executor_place_in_compute_slot(
    iree_task_executor_t* executor, iree_task_process_t* process) {
  if (IREE_LIKELY(
          iree_task_executor_try_place_in_compute_slot(executor, process))) {
    return;
  }
  // All slots occupied. Push to overflow list — a releasing worker will
  // promote this process into a slot when one becomes available.
  iree_task_process_slist_push(&executor->compute_overflow, process);
}

void iree_task_executor_schedule_process(iree_task_executor_t* executor,
                                         iree_task_process_t* process) {
  IREE_ASSERT(!iree_task_process_is_terminal(process),
              "cannot schedule a completed or cancelled process");
  IREE_TRACE_ZONE_BEGIN(z0);

  int32_t budget = iree_task_process_worker_budget(process);

  // Signal that new work is available. The draining worker checks this
  // before transitioning to idle, closing the sleep/wake race.
  //
  // seq_cst is required here because this is one half of a Dekker-style
  // protocol with drain_process (worker.c):
  //   Scheduler: store(needs_drain=1)        then CAS(schedule_state)
  //   Worker:    store(schedule_state=IDLE)   then load(needs_drain)
  // Both threads store one variable and load the other. Release/acquire
  // on different variables does not prevent StoreLoad reordering — on ARM
  // the CAS below could read schedule_state before this store is globally
  // visible, causing the scheduler to miss the worker's IDLE transition
  // while the worker misses our needs_drain=1 signal. seq_cst provides
  // the required StoreLoad barrier (matching the worker's seq_cst store
  // to schedule_state).
  iree_atomic_store(&process->needs_drain, 1, iree_memory_order_seq_cst);

  if (budget <= 1) {
    // Sequential process: immediate list with Dekker sleeping protocol.
    // Try to enqueue if idle. If already QUEUED or DRAINING, the worker
    // will see our needs_drain signal before going idle.
    int32_t expected = IREE_TASK_PROCESS_SCHEDULE_IDLE;
    if (iree_atomic_compare_exchange_strong(
            &process->schedule_state, &expected,
            (int32_t)IREE_TASK_PROCESS_SCHEDULE_QUEUED,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_task_process_slist_push(&executor->immediate_list, process);
      iree_task_executor_wake_workers(executor, 1);
    }
  } else {
    // Compute process: place in a compute slot on first activation.
    // Workers scan these slots round-robin and drain cooperatively.
    int32_t expected = IREE_TASK_PROCESS_SCHEDULE_IDLE;
    if (iree_atomic_compare_exchange_strong(
            &process->schedule_state, &expected,
            (int32_t)IREE_TASK_PROCESS_SCHEDULE_DRAINING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_task_executor_place_in_compute_slot(executor, process);
    }
    // Wake workers up to the budget (whether first activation or re-wake).
    iree_task_executor_wake_workers(executor, budget);
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_task_executor_dump_wake_state(iree_task_executor_t* executor,
                                        FILE* file) {
  fprintf(file, "\n=== EXECUTOR WAKE STATE ===\n");
  fprintf(file, "desired_wake: %d\n",
          iree_atomic_load(&executor->desired_wake, iree_memory_order_relaxed));
  iree_task_affinity_set_t idle_mask = iree_atomic_task_affinity_set_load(
      &executor->worker_idle_mask, iree_memory_order_relaxed);
  iree_task_affinity_set_t live_mask = iree_atomic_task_affinity_set_load(
      &executor->worker_live_mask, iree_memory_order_relaxed);
  fprintf(file, "idle_count: %d  idle_mask:",
          iree_atomic_load(&executor->worker_idle_count,
                           iree_memory_order_relaxed));
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    fprintf(file, " 0x%016llx", (unsigned long long)idle_mask.words[i]);
  }
  fprintf(file, "  live_mask:");
  for (iree_host_size_t i = 0; i < IREE_TASK_AFFINITY_SET_WORD_COUNT; ++i) {
    fprintf(file, " 0x%016llx", (unsigned long long)live_mask.words[i]);
  }
  fprintf(file, "\n");

  for (iree_host_size_t i = 0; i < executor->worker_count; ++i) {
    iree_task_worker_t* worker = &executor->workers[i];
    int32_t state = iree_atomic_load(&worker->state, iree_memory_order_relaxed);
    bool is_idle = iree_task_affinity_set_test(idle_mask, worker->worker_bit);
    fprintf(file, "  worker[%zu]: state=%d  idle=%d\n", i, state, (int)is_idle);
  }

  fprintf(file, "compute_slots:\n");
  for (iree_host_size_t i = 0; i < IREE_TASK_EXECUTOR_MAX_COMPUTE_SLOTS; ++i) {
    intptr_t process = iree_atomic_load(&executor->compute_slots[i].process,
                                        iree_memory_order_relaxed);
    if (!process) continue;
    int64_t active_drainers = iree_atomic_load(
        &executor->compute_slots[i].active_drainers, iree_memory_order_relaxed);
    int32_t completion_claimed =
        iree_atomic_load(&executor->compute_slots[i].completion_claimed,
                         iree_memory_order_relaxed);
    fprintf(file,
            "  slot[%zu]: process=%p  active_drainers=gen:%d|count:%d  "
            "completion_claimed=%d\n",
            i, (void*)process, (int32_t)(active_drainers >> 32),
            (int32_t)active_drainers, completion_claimed);
  }
  fprintf(file, "=== END EXECUTOR WAKE STATE ===\n\n");
  fflush(file);
}
