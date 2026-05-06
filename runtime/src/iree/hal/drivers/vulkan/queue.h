// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_QUEUE_H_
#define IREE_HAL_DRIVERS_VULKAN_QUEUE_H_

#include "iree/async/frontier.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/allocator.h"
#include "iree/hal/drivers/vulkan/builtins.h"
#include "iree/hal/drivers/vulkan/semaphore.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/hal/local/profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY 64
#define IREE_HAL_VULKAN_QUEUE_DESCRIPTOR_BLOCK_CAPACITY 4096
#define IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_COUNT 16
#define IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE (1ull << 20)

IREE_ASYNC_FIXED_FRONTIER_TYPE(iree_hal_vulkan_queue_frontier_t,
                               IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY);

typedef struct iree_hal_vulkan_queue_pending_submission_t
    iree_hal_vulkan_queue_pending_submission_t;
typedef struct iree_hal_vulkan_queue_command_buffer_block_t
    iree_hal_vulkan_queue_command_buffer_block_t;
typedef struct iree_hal_vulkan_queue_descriptor_block_t
    iree_hal_vulkan_queue_descriptor_block_t;
typedef struct iree_hal_vulkan_queue_native_descriptor_block_t
    iree_hal_vulkan_queue_native_descriptor_block_t;
typedef struct iree_hal_vulkan_queue_staging_ring_t
    iree_hal_vulkan_queue_staging_ring_t;

typedef enum iree_hal_vulkan_queue_role_e {
  IREE_HAL_VULKAN_QUEUE_ROLE_COMPUTE = 0,
  IREE_HAL_VULKAN_QUEUE_ROLE_TRANSFER = 1,
  IREE_HAL_VULKAN_QUEUE_ROLE_SPARSE_BINDING = 2,
} iree_hal_vulkan_queue_role_t;

// Queue construction parameters.
typedef struct iree_hal_vulkan_queue_params_t {
  // Logical device owning this queue. Borrowed.
  iree_hal_vulkan_logical_device_t* device;

  // Device-level Vulkan dispatch table. Borrowed and copied into the queue.
  const iree_hal_vulkan_device_syms_t* syms;

  // Vulkan logical device that owns all queue handles and semaphores.
  VkDevice logical_device;

  // Device-owned built-in pipelines. Borrowed.
  const iree_hal_vulkan_builtins_t* builtins;

  // Vulkan queue handle borrowed from the logical device.
  VkQueue queue;

  // Vulkan queue family capability flags for |queue|.
  VkQueueFlags queue_flags;

  // Valid timestamp bits reported by the queue family.
  uint32_t timestamp_valid_bits;

  // Mutex serializing host access to |queue|. Borrowed.
  iree_slim_mutex_t* queue_handle_mutex;

  // Proactor used for cold queue-side waits such as pool notifications.
  iree_async_proactor_t* proactor;

  // Queue family index used to acquire the queue handle.
  uint32_t queue_family_index;

  // Queue index within the selected family.
  uint32_t queue_index;

  // HAL-visible queue affinity represented by this queue.
  iree_hal_queue_affinity_t queue_affinity;

  // Queue role used only for diagnostics and profiling labels.
  iree_hal_vulkan_queue_role_t role;

  // Host allocator used for queue-owned allocations.
  iree_allocator_t host_allocator;
} iree_hal_vulkan_queue_params_t;

// Host-driven Vulkan queue lane.
typedef struct iree_hal_vulkan_queue_t {
  // Logical device owning this queue. Borrowed.
  iree_hal_vulkan_logical_device_t* device;

  // Device-level Vulkan dispatch table copied at initialization.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns queue resources.
  VkDevice logical_device;

  // Device-owned built-in pipelines. Borrowed.
  const iree_hal_vulkan_builtins_t* builtins;

  // Device allocator used for queue-owned staging resources. Borrowed.
  iree_hal_allocator_t* device_allocator;

  // Vulkan queue handle borrowed from the logical device.
  VkQueue queue;

  // Vulkan queue family capability flags for |queue|.
  VkQueueFlags queue_flags;

  // Valid timestamp bits reported by the queue family.
  uint32_t timestamp_valid_bits;

  // Mutex serializing host access to queue. Borrowed.
  iree_slim_mutex_t* queue_handle_mutex;

  // Proactor used for cold queue-side waits such as pool notifications.
  iree_async_proactor_t* proactor;

  // Queue family index used to acquire the queue handle.
  uint32_t queue_family_index;

  // Queue index within the selected family.
  uint32_t queue_index;

  // HAL-visible queue affinity represented by this queue.
  iree_hal_queue_affinity_t queue_affinity;

  // Queue role used only for diagnostics and profiling labels.
  iree_hal_vulkan_queue_role_t role;

  // Host allocator used for queue-owned allocations.
  iree_allocator_t host_allocator;

  // Mutex serializing queue epoch assignment and lane-local submission state.
  iree_slim_mutex_t submission_mutex;

  // Sticky failure status for this queue. Owned by the queue.
  iree_atomic_intptr_t failure_status;

  // Set to non-zero when queue teardown should stop the completion thread.
  iree_atomic_int32_t stop_requested;

  // Queue axis assigned by the logical device topology.
  iree_async_axis_t axis;

  // Shared frontier tracker for queue epoch advancement. Borrowed.
  iree_async_frontier_tracker_t* frontier_tracker;

  // Queue-owned timeline semaphore signaled once per accepted submission.
  VkSemaphore epoch_semaphore;

  // Queue-owned command buffer cache for one-shot native submissions.
  struct {
    // First command buffer block owned by this queue.
    iree_hal_vulkan_queue_command_buffer_block_t* head;

    // Last command buffer block owned by this queue.
    iree_hal_vulkan_queue_command_buffer_block_t* tail;

    // Next command buffer block considered for acquisition.
    iree_hal_vulkan_queue_command_buffer_block_t* cursor;

    // Number of command buffer blocks currently owned by this queue.
    uint32_t block_count;
  } command_buffer_cache;

  // Host-signaled timeline semaphore used to wake the completion thread.
  VkSemaphore wakeup_semaphore;

  // Monotonic host-side wakeup timeline value.
  iree_atomic_int64_t wakeup_value;

  // Next queue epoch value to assign to an accepted submission.
  uint64_t next_epoch_value;

  // Last queue epoch whose completion side effects have fully retired.
  iree_atomic_int64_t last_drained_epoch;

  // Accumulated causal frontier for work submitted through this queue.
  iree_hal_vulkan_queue_frontier_t frontier;

  // Head of pending submissions ordered by epoch.
  iree_hal_vulkan_queue_pending_submission_t* pending_head;

  // Tail of pending submissions ordered by epoch.
  iree_hal_vulkan_queue_pending_submission_t* pending_tail;

  // Head of submissions waiting for software-resolved semaphore dependencies.
  iree_hal_vulkan_queue_pending_submission_t* deferred_head;

  // Head of deferred submissions whose software waits have resolved.
  iree_hal_vulkan_queue_pending_submission_t* ready_head;

  // Tail of deferred submissions whose software waits have resolved.
  iree_hal_vulkan_queue_pending_submission_t* ready_tail;

  // Queue-owned descriptor cache for built-in command packets.
  struct {
    // First descriptor block owned by this queue.
    iree_hal_vulkan_queue_descriptor_block_t* head;

    // Last descriptor block owned by this queue.
    iree_hal_vulkan_queue_descriptor_block_t* tail;

    // Next descriptor block considered for acquisition.
    iree_hal_vulkan_queue_descriptor_block_t* cursor;

    // Number of descriptor blocks currently owned by this queue.
    uint32_t block_count;
  } descriptor_cache;

  // Queue-owned descriptor pool cache for native command recording.
  struct {
    // First native descriptor block owned by this queue.
    iree_hal_vulkan_queue_native_descriptor_block_t* head;

    // Last native descriptor block owned by this queue.
    iree_hal_vulkan_queue_native_descriptor_block_t* tail;

    // Next native descriptor block considered for acquisition.
    iree_hal_vulkan_queue_native_descriptor_block_t* cursor;

    // Number of native descriptor blocks currently owned by this queue.
    uint32_t block_count;
  } native_descriptor_cache;

  // Queue-owned host-to-device staging ring for uploads.
  iree_hal_vulkan_queue_staging_ring_t* upload_staging_ring;

  // Queue-owned device-to-host staging ring for downloads.
  iree_hal_vulkan_queue_staging_ring_t* download_staging_ring;

  // Queue-owned completion thread waiting on epoch_semaphore.
  iree_thread_t* completion_thread;

  // Active HAL-native profile recorder, when profiling is enabled. Borrowed.
  iree_hal_local_profile_recorder_t* profile_recorder;

  // Queue identity emitted with HAL-native profile records.
  iree_hal_local_profile_queue_scope_t profile_scope;

  // Shared submission id source for HAL-native profile records. Borrowed.
  iree_atomic_int64_t* profile_submission_counter;
} iree_hal_vulkan_queue_t;

// Initializes a queue lane around a borrowed VkQueue.
iree_status_t iree_hal_vulkan_queue_initialize(
    const iree_hal_vulkan_queue_params_t* params,
    iree_hal_vulkan_queue_t* out_queue);

// Initializes queue-owned staging resources once the device allocator exists.
iree_status_t iree_hal_vulkan_queue_initialize_staging(
    iree_hal_vulkan_queue_t* queue, iree_hal_allocator_t* device_allocator);

// Deinitializes a queue lane and releases all queue-owned resources.
void iree_hal_vulkan_queue_deinitialize(iree_hal_vulkan_queue_t* queue);

// Assigns this queue's causal frontier axis and starts completion processing.
iree_status_t iree_hal_vulkan_queue_assign_frontier(
    iree_hal_vulkan_queue_t* queue,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis);

// Retires this queue's causal frontier axis and stops completion processing.
void iree_hal_vulkan_queue_retire_frontier(iree_hal_vulkan_queue_t* queue);

// Assigns the active HAL-native profile recorder for this queue.
void iree_hal_vulkan_queue_set_profile_recorder(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t profile_scope,
    iree_atomic_int64_t* submission_counter);

// Submits a queue-ordered semaphore barrier.
iree_status_t iree_hal_vulkan_queue_submit_barrier(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list);

// Submits a queue-ordered transient allocation.
iree_status_t iree_hal_vulkan_queue_submit_alloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_alloca_plan_t allocation_plan,
    iree_hal_buffer_params_t params, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer);

// Submits a queue-ordered transient deallocation.
iree_status_t iree_hal_vulkan_queue_submit_dealloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags);

// Submits sparse buffer memory binds ordered by queue semaphores.
//
// |binds| is copied into queue-owned storage and may be released by the caller
// after this returns. The sparse VkBuffer and VkDeviceMemory handles referenced
// by |binds| must stay live until |signal_semaphore_list| has retired or
// failed.
iree_status_t iree_hal_vulkan_queue_submit_sparse_bind(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list, VkBuffer buffer,
    iree_host_size_t bind_count, const VkSparseMemoryBind* binds);

// Submits a buffer fill ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_fill(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags);

// Submits a buffer update ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_update(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags);

// Submits a buffer copy ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_copy(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags);

// Submits a direct dispatch ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_dispatch(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags);

// Submits a file write ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_write(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags);

// Submits a file read ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_read(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags);

// Submits a recorded command buffer ordered by queue semaphores.
iree_status_t iree_hal_vulkan_queue_submit_execute(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags,
    iree_hal_profile_queue_event_type_t queue_event_type);

// Submits a queue-ordered host call.
iree_status_t iree_hal_vulkan_queue_submit_host_call(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags);

// Drains any completed epochs without blocking for future GPU progress.
iree_host_size_t iree_hal_vulkan_queue_drain_completions(
    iree_hal_vulkan_queue_t* queue);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_QUEUE_H_
