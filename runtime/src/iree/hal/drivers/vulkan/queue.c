// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/queue.h"

#include <stdio.h>
#include <string.h>

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/local/transient_buffer.h"

typedef enum iree_hal_vulkan_queue_submission_kind_e {
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER = 0,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL = 1,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL = 2,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE = 3,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY = 4,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE = 5,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA = 6,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA = 7,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_WRITE = 8,
  IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_READ = 9,
} iree_hal_vulkan_queue_submission_kind_t;

typedef enum iree_hal_vulkan_queue_deferred_state_e {
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING = 0,
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_READY = 1,
  IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING = 2,
} iree_hal_vulkan_queue_deferred_state_t;

typedef enum iree_hal_vulkan_queue_alloca_memory_wait_kind_e {
  // No active queue-owned memory readiness wait.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE = 0,

  // Staged backing is reserved but cannot be used until a pool death frontier
  // retires.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER = 1,

  // No reservation is currently available; retry after the pool notification
  // advances.
  IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_vulkan_queue_alloca_memory_wait_kind_t;

typedef struct iree_hal_vulkan_queue_wait_entry_t {
  // Async semaphore timepoint registered for one software-resolved wait.
  iree_async_semaphore_timepoint_t timepoint;

  // Deferred submission owning this wait entry.
  iree_hal_vulkan_queue_pending_submission_t* submission;

  // Set to non-zero after the timepoint callback's final access completes.
  iree_atomic_int32_t callback_complete;
} iree_hal_vulkan_queue_wait_entry_t;

typedef struct iree_hal_vulkan_queue_wait_resolution_t {
  // Native Vulkan wait semaphores resolved from queue last-signal metadata.
  VkSemaphoreSubmitInfo* wait_infos;

  // Number of populated wait_infos entries.
  uint32_t wait_info_count;

  // Capacity of the wait_infos storage.
  uint32_t wait_info_capacity;

  // Whether unresolved waits require software timepoint deferral.
  bool needs_deferral;
} iree_hal_vulkan_queue_wait_resolution_t;

typedef struct iree_hal_vulkan_queue_submission_result_t {
  // Whether ownership moved to the native pending queue.
  bool submitted;

  // Submission parked on a queue-owned memory readiness wait.
  iree_hal_vulkan_queue_pending_submission_t* memory_wait_submission;

  // Sticky queue failure status recorded during native submission.
  iree_status_t queue_failure_status;
} iree_hal_vulkan_queue_submission_result_t;

struct iree_hal_vulkan_queue_pending_submission_t {
  // Next submission in the queue-owned list currently holding this object.
  iree_hal_vulkan_queue_pending_submission_t* next;

  // Queue owning this submission.
  iree_hal_vulkan_queue_t* queue;

  // Queue epoch signaled by the native Vulkan submit.
  uint64_t epoch;

  // Queue frontier published by this submission.
  iree_hal_vulkan_queue_frontier_t frontier;

  // Wait semaphores retained until the native submit retires.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphores retained until completion side effects retire.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Submission kind controlling completion-side actions.
  iree_hal_vulkan_queue_submission_kind_t kind;

  // Native Vulkan command buffer submitted for GPU-encoded work.
  VkCommandBuffer native_command_buffer;

  // Wait entries registered for software-resolved wait-before-signal edges.
  iree_hal_vulkan_queue_wait_entry_t* wait_entries;

  // Number of entries in wait_entries.
  iree_host_size_t wait_entry_count;

  // Outstanding software wait callbacks before the submission becomes ready.
  iree_atomic_int32_t wait_count;

  // Lifecycle owner for software-deferred submission promotion/cancellation.
  iree_atomic_int32_t deferred_state;

  // First software wait failure status. Owned by this submission.
  iree_atomic_intptr_t wait_failure_status;

  // Wakes cancellation or promotion owners waiting for callbacks to quiesce.
  iree_notification_t callback_notification;

  // Queue-ordered host action payload.
  struct {
    // Host callback and user data captured from queue_host_call.
    iree_hal_host_call_t call;

    // User arguments copied from queue_host_call.
    uint64_t args[4];

    // HAL host-call flags captured from queue_host_call.
    iree_hal_host_call_flags_t flags;
  } host_call;

  // Host-mediated buffer fill payload.
  struct {
    // Target buffer retained until the fill retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_fill.
    iree_device_size_t target_offset;

    // Number of bytes to fill in the target buffer.
    iree_device_size_t length;

    // Fill pattern bytes captured from queue_fill.
    uint8_t pattern[4];

    // Number of bytes in pattern.
    iree_host_size_t pattern_length;

    // HAL fill flags captured from queue_fill.
    iree_hal_fill_flags_t flags;
  } fill;

  // Host-mediated buffer update payload.
  struct {
    // Source bytes copied from the queue_update caller.
    void* source_data;

    // Target buffer retained until the update retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_update.
    iree_device_size_t target_offset;

    // Number of bytes to copy into the target buffer.
    iree_device_size_t length;

    // HAL update flags captured from queue_update.
    iree_hal_update_flags_t flags;
  } update;

  // Host-mediated buffer copy payload.
  struct {
    // Source buffer retained until the copy retires.
    iree_hal_buffer_t* source_buffer;

    // Source byte offset captured from queue_copy.
    iree_device_size_t source_offset;

    // Target buffer retained until the copy retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_copy.
    iree_device_size_t target_offset;

    // Number of bytes to copy into the target buffer.
    iree_device_size_t length;

    // HAL copy flags captured from queue_copy.
    iree_hal_copy_flags_t flags;
  } copy;

  // Queue-ordered file write payload.
  struct {
    // Source buffer retained until the write retires.
    iree_hal_buffer_t* source_buffer;

    // Source byte offset captured from queue_write.
    iree_device_size_t source_offset;

    // Target file retained until the write retires.
    iree_hal_file_t* target_file;

    // Target file offset captured from queue_write.
    uint64_t target_offset;

    // Staging buffer receiving native GPU copies before host file writes.
    iree_hal_buffer_t* staging_buffer;

    // Number of bytes to write into the target file.
    iree_device_size_t length;

    // HAL write flags captured from queue_write.
    iree_hal_write_flags_t flags;
  } write;

  // Queue-ordered file read payload.
  struct {
    // Source file retained until the read retires.
    iree_hal_file_t* source_file;

    // Source file offset captured from queue_read.
    uint64_t source_offset;

    // Target buffer retained until the read retires.
    iree_hal_buffer_t* target_buffer;

    // Target byte offset captured from queue_read.
    iree_device_size_t target_offset;

    // Number of bytes to read into the target buffer.
    iree_device_size_t length;

    // HAL read flags captured from queue_read.
    iree_hal_read_flags_t flags;
  } read;

  // Queue-ordered allocation payload.
  struct {
    // Transient buffer retained until the alloca retires.
    iree_hal_buffer_t* buffer;

    // Borrowed pool used to acquire backing for buffer.
    iree_hal_pool_t* pool;

    // Buffer parameters captured from queue_alloca after normalization.
    iree_hal_buffer_params_t params;

    // Physical allocation size requested from pool.
    iree_device_size_t allocation_size;

    // HAL allocation flags captured from queue_alloca.
    iree_hal_alloca_flags_t flags;

    // Pool reservation flags used when probing pool.
    iree_hal_pool_reserve_flags_t reserve_flags;

    // Active memory-readiness wait kind.
    iree_hal_vulkan_queue_alloca_memory_wait_kind_t memory_wait_kind;

    // Set to non-zero after the memory wait callback's final access completes.
    iree_atomic_int32_t memory_wait_callback_complete;

    // Pool death frontier borrowed while buffer keeps the reservation armed.
    const iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for wait_frontier.
    iree_async_frontier_waiter_t frontier_waiter;

    // Pool notification borrowed while a notification wait is active.
    iree_async_notification_t* pool_notification;

    // Notification epoch observed before retrying pool reservation.
    uint32_t pool_notification_wait_token;

    // Whether an observe scope is held until the wait operation is submitted.
    bool pool_notification_observation_held;

    // Async wait operations rotated so callbacks can arm another wait.
    iree_async_notification_wait_operation_t pool_notification_wait_ops[2];

    // Active slot in pool_notification_wait_ops.
    uint8_t pool_notification_wait_slot;
  } alloca;

  // Queue-ordered deallocation payload.
  struct {
    // Transient buffer retained until the dealloca retires.
    iree_hal_buffer_t* buffer;

    // HAL dealloca flags captured from queue_dealloca.
    iree_hal_dealloca_flags_t flags;
  } dealloca;

  // Recorded command-buffer execution payload.
  struct {
    // Command buffer retained until execution retires.
    iree_hal_command_buffer_t* command_buffer;

    // Descriptor pool backing native command buffer descriptor sets.
    VkDescriptorPool native_descriptor_pool;

    // Captured binding table entries used by indirect command references.
    iree_hal_buffer_binding_t* binding_table_bindings;

    // Number of entries populated in binding_table_bindings.
    iree_host_size_t binding_table_count;

    // HAL execute flags captured from queue_execute.
    iree_hal_execute_flags_t flags;
  } execute;
};

static iree_status_t iree_hal_vulkan_queue_create_timeline_semaphore(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    uint64_t initial_value, VkSemaphore* out_semaphore) {
  *out_semaphore = VK_NULL_HANDLE;
  VkSemaphoreTypeCreateInfo timeline_create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = initial_value,
  };
  VkSemaphoreCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &timeline_create_info,
  };
  return iree_vkCreateSemaphore(IREE_VULKAN_DEVICE(syms), logical_device,
                                &create_info, /*pAllocator=*/NULL,
                                out_semaphore);
}

static iree_status_t iree_hal_vulkan_queue_create_command_pool(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    uint32_t queue_family_index, VkCommandPool* out_command_pool) {
  *out_command_pool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queue_family_index,
  };
  return iree_vkCreateCommandPool(IREE_VULKAN_DEVICE(syms), logical_device,
                                  &create_info, /*pAllocator=*/NULL,
                                  out_command_pool);
}

static iree_status_t iree_hal_vulkan_queue_allocate_native_command_buffer(
    iree_hal_vulkan_queue_t* queue, VkCommandBuffer* out_command_buffer) {
  *out_command_buffer = VK_NULL_HANDLE;
  if (!queue->command_pool) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue has no command pool");
  }

  VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = queue->command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_status_t status = iree_vkAllocateCommandBuffers(
      IREE_VULKAN_DEVICE(&queue->syms), queue->logical_device, &allocate_info,
      out_command_buffer);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return status;
}

static bool iree_hal_vulkan_queue_has_pending(iree_hal_vulkan_queue_t* queue) {
  bool has_pending = false;
  iree_slim_mutex_lock(&queue->submission_mutex);
  has_pending = queue->pending_head != NULL || queue->deferred_head != NULL ||
                queue->ready_head != NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  return has_pending;
}

static void iree_hal_vulkan_queue_append_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = queue->deferred_head;
  queue->deferred_head = submission;
}

static void iree_hal_vulkan_queue_unlink_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_pending_submission_t** link = &queue->deferred_head;
  while (*link) {
    if (*link == submission) {
      *link = submission->next;
      submission->next = NULL;
      return;
    }
    link = &(*link)->next;
  }
}

static void iree_hal_vulkan_queue_append_ready_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = NULL;
  if (queue->ready_tail) {
    queue->ready_tail->next = submission;
  } else {
    queue->ready_head = submission;
  }
  queue->ready_tail = submission;
}

static iree_hal_vulkan_queue_pending_submission_t*
iree_hal_vulkan_queue_pop_ready_submission(iree_hal_vulkan_queue_t* queue) {
  iree_hal_vulkan_queue_pending_submission_t* submission = queue->ready_head;
  if (!submission) return NULL;
  queue->ready_head = submission->next;
  if (!queue->ready_head) queue->ready_tail = NULL;
  submission->next = NULL;
  return submission;
}

static iree_status_t iree_hal_vulkan_queue_store_error(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status),
              "queue failure status must be non-OK");
  intptr_t expected = 0;
  if (iree_atomic_compare_exchange_strong(
          &queue->failure_status, &expected, (intptr_t)status,
          iree_memory_order_release, iree_memory_order_acquire)) {
    return iree_status_clone(status);
  }
  iree_status_free(status);
  return iree_status_clone((iree_status_t)expected);
}

static iree_status_t iree_hal_vulkan_queue_check_error(
    iree_hal_vulkan_queue_t* queue) {
  iree_status_t status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  return iree_status_is_ok(status) ? iree_ok_status()
                                   : iree_status_clone(status);
}

static iree_status_t iree_hal_vulkan_queue_signal_wakeup(
    iree_hal_vulkan_queue_t* queue) {
  const uint64_t new_value =
      (uint64_t)iree_atomic_fetch_add(&queue->wakeup_value, 1,
                                      iree_memory_order_acq_rel) +
      1;
  VkSemaphoreSignalInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = queue->wakeup_semaphore,
      .value = new_value,
  };
  return iree_vkSignalSemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                                queue->logical_device, &signal_info);
}

static iree_status_t iree_hal_vulkan_queue_validate_host_call(
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  const iree_hal_host_call_flags_t known_flags =
      IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING |
      IREE_HAL_HOST_CALL_FLAG_WAIT_ACTIVE | IREE_HAL_HOST_CALL_FLAG_RELAXED;
  if (!call.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call callback must be non-null");
  }
  if (!args) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "host_call args must be non-null");
  }
  if (iree_any_bit_set(flags, ~known_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported host_call flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_validate_semaphore_list(
    iree_hal_vulkan_queue_t* queue, iree_hal_semaphore_list_t semaphore_list,
    iree_string_view_t usage) {
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = semaphore_list.semaphores[i];
    if (!iree_hal_vulkan_semaphore_is_local(semaphore, queue->device)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Vulkan queue %.*s semaphore %" PRIhsz
                              " is not a local Vulkan semaphore",
                              (int)usage.size, usage.data, i);
    }
    if (semaphore_list.payload_values[i] > IREE_HAL_SEMAPHORE_MAX_VALUE) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan queue %.*s semaphore value %" PRIu64
          " exceeds the maximum HAL semaphore value %" PRIu64,
          (int)usage.size, usage.data, semaphore_list.payload_values[i],
          (uint64_t)IREE_HAL_SEMAPHORE_MAX_VALUE);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_pending_submission_create(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_vulkan_queue_submission_kind_t kind, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags,
    iree_hal_vulkan_queue_pending_submission_t** out_submission) {
  *out_submission = NULL;
  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*submission), (void**)&submission));
  memset(submission, 0, sizeof(*submission));
  submission->queue = queue;
  submission->kind = kind;
  iree_notification_initialize(&submission->callback_notification);
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                    iree_memory_order_relaxed);
  if (kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL) {
    submission->host_call.call = call;
    memcpy(submission->host_call.args, args,
           sizeof(submission->host_call.args));
    submission->host_call.flags = flags;
  }

  iree_status_t status =
      iree_hal_semaphore_list_clone(&wait_semaphore_list, queue->host_allocator,
                                    &submission->wait_semaphore_list);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_semaphore_list,
                                           queue->host_allocator,
                                           &submission->signal_semaphore_list);
  }
  if (iree_status_is_ok(status)) {
    *out_submission = submission;
  } else {
    if (submission->wait_semaphore_list.semaphores) {
      iree_hal_semaphore_list_free(submission->wait_semaphore_list,
                                   queue->host_allocator);
    }
    iree_notification_deinitialize(&submission->callback_notification);
    iree_allocator_free(queue->host_allocator, submission);
  }
  return status;
}

static void iree_hal_vulkan_queue_pending_submission_destroy(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->wait_semaphore_list.semaphores) {
    iree_hal_semaphore_list_free(submission->wait_semaphore_list,
                                 queue->host_allocator);
  }
  if (submission->signal_semaphore_list.semaphores) {
    iree_hal_semaphore_list_free(submission->signal_semaphore_list,
                                 queue->host_allocator);
  }
  iree_status_t wait_failure_status = (iree_status_t)iree_atomic_exchange(
      &submission->wait_failure_status, 0, iree_memory_order_acquire);
  iree_status_free(wait_failure_status);
  if (submission->wait_entries) {
    iree_allocator_free(queue->host_allocator, submission->wait_entries);
  }
  if (submission->fill.target_buffer) {
    iree_hal_buffer_release(submission->fill.target_buffer);
  }
  if (submission->update.target_buffer) {
    iree_hal_buffer_release(submission->update.target_buffer);
  }
  if (submission->update.source_data) {
    iree_allocator_free(queue->host_allocator, submission->update.source_data);
  }
  if (submission->copy.source_buffer) {
    iree_hal_buffer_release(submission->copy.source_buffer);
  }
  if (submission->copy.target_buffer) {
    iree_hal_buffer_release(submission->copy.target_buffer);
  }
  if (submission->write.source_buffer) {
    iree_hal_buffer_release(submission->write.source_buffer);
  }
  if (submission->write.target_file) {
    iree_hal_file_release(submission->write.target_file);
  }
  if (submission->write.staging_buffer) {
    iree_hal_buffer_release(submission->write.staging_buffer);
  }
  if (submission->read.source_file) {
    iree_hal_file_release(submission->read.source_file);
  }
  if (submission->read.target_buffer) {
    iree_hal_buffer_release(submission->read.target_buffer);
  }
  if (submission->alloca.buffer) {
    if (submission->alloca.pool_notification_observation_held) {
      submission->alloca.pool_notification_observation_held = false;
      iree_async_notification_end_observe(submission->alloca.pool_notification);
    }
    iree_hal_buffer_release(submission->alloca.buffer);
  }
  if (submission->dealloca.buffer) {
    iree_hal_buffer_release(submission->dealloca.buffer);
  }
  if (submission->execute.native_descriptor_pool) {
    iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(&queue->syms),
                                 queue->logical_device,
                                 submission->execute.native_descriptor_pool,
                                 /*pAllocator=*/NULL);
  }
  if (submission->native_command_buffer) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_vkFreeCommandBuffers(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, queue->command_pool,
                              /*commandBufferCount=*/1,
                              &submission->native_command_buffer);
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }
  if (submission->execute.command_buffer) {
    iree_hal_command_buffer_release(submission->execute.command_buffer);
  }
  if (submission->execute.binding_table_bindings) {
    if (!iree_any_bit_set(
            submission->execute.flags,
            IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
      for (iree_host_size_t i = 0; i < submission->execute.binding_table_count;
           ++i) {
        iree_hal_buffer_release(
            submission->execute.binding_table_bindings[i].buffer);
      }
    }
    iree_allocator_free(queue->host_allocator,
                        submission->execute.binding_table_bindings);
  }
  iree_notification_deinitialize(&submission->callback_notification);
  iree_allocator_free(queue->host_allocator, submission);
}

static void iree_hal_vulkan_queue_append_pending_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  submission->next = NULL;
  if (queue->pending_tail) {
    queue->pending_tail->next = submission;
  } else {
    queue->pending_head = submission;
  }
  queue->pending_tail = submission;
}

static iree_hal_vulkan_queue_pending_submission_t*
iree_hal_vulkan_queue_pop_completed_submissions(iree_hal_vulkan_queue_t* queue,
                                                uint64_t completed_epoch) {
  iree_hal_vulkan_queue_pending_submission_t* completed_head =
      queue->pending_head;
  iree_hal_vulkan_queue_pending_submission_t* completed_tail = NULL;
  while (queue->pending_head && queue->pending_head->epoch <= completed_epoch) {
    completed_tail = queue->pending_head;
    queue->pending_head = queue->pending_head->next;
  }
  if (!completed_tail) return NULL;

  if (!queue->pending_head) queue->pending_tail = NULL;
  completed_tail->next = NULL;
  return completed_head;
}

static void iree_hal_vulkan_queue_fail_signal_list(
    iree_hal_semaphore_list_t signal_semaphore_list, iree_status_t status) {
  if (signal_semaphore_list.count == 0) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(signal_semaphore_list, status);
}

static void iree_hal_vulkan_queue_signal_list_or_fail(
    iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_async_frontier_t* frontier) {
  iree_status_t status =
      iree_hal_semaphore_list_signal(signal_semaphore_list, frontier);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_execute_host_call(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);

  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }

  const bool is_nonblocking = iree_any_bit_set(
      submission->host_call.flags, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);
  if (is_nonblocking) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  }

  iree_hal_host_call_context_t context = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_affinity = queue->queue_affinity,
      .signal_semaphore_list = is_nonblocking ? iree_hal_semaphore_list_empty()
                                              : signal_semaphore_list,
  };
  iree_status_t call_status =
      submission->host_call.call.fn(submission->host_call.call.user_data,
                                    submission->host_call.args, &context);

  if (is_nonblocking || iree_status_is_deferred(call_status)) {
    iree_status_free(call_status);
  } else if (iree_status_is_ok(call_status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, call_status);
  }
}

static bool iree_hal_vulkan_queue_can_update_native(
    iree_device_size_t target_offset, iree_device_size_t length) {
  return length != 0 && target_offset % sizeof(uint32_t) == 0 &&
         length % sizeof(uint32_t) == 0 && length <= 65536;
}

static bool iree_hal_vulkan_queue_can_fill_native(
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_host_size_t pattern_length) {
  return length != 0 &&
         (pattern_length == sizeof(uint8_t) ||
          pattern_length == sizeof(uint16_t) ||
          pattern_length == sizeof(uint32_t)) &&
         target_offset % sizeof(uint32_t) == 0 &&
         length % sizeof(uint32_t) == 0;
}

static bool iree_hal_vulkan_queue_buffer_has_recordable_backing(
    iree_hal_buffer_t* buffer) {
  return !iree_hal_local_transient_buffer_isa(buffer) ||
         iree_hal_local_transient_buffer_backing_buffer(buffer) != NULL;
}

static iree_status_t iree_hal_vulkan_queue_buffer_is_native(
    iree_hal_buffer_t* buffer, bool* out_is_native) {
  *out_is_native = false;
  if (!iree_hal_vulkan_queue_buffer_has_recordable_backing(buffer)) {
    return iree_ok_status();
  }
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(buffer, &backing_buffer));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(backing_buffer);
  *out_is_native = iree_hal_vulkan_buffer_isa(allocated_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_expand_fill_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_expanded_pattern) {
  *out_expanded_pattern = 0;
  switch (pattern_length) {
    case sizeof(uint8_t):
      *out_expanded_pattern = pattern[0];
      *out_expanded_pattern |= *out_expanded_pattern << 8;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    case sizeof(uint16_t): {
      uint16_t pattern16 = 0;
      memcpy(&pattern16, pattern, sizeof(pattern16));
      *out_expanded_pattern = pattern16;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    }
    case sizeof(uint32_t):
      memcpy(out_expanded_pattern, pattern, sizeof(*out_expanded_pattern));
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue fill pattern length must be 1, 2, "
                              "or 4 bytes (got %" PRIhsz ")",
                              pattern_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_record_fill_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(submission->fill.target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      submission->fill.target_buffer, &target_backing));
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t target_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      submission->fill.target_buffer, target_backing,
      submission->fill.target_offset, &target_offset));
  if (target_offset % sizeof(uint32_t) != 0 ||
      submission->fill.length % sizeof(uint32_t) != 0 ||
      !iree_hal_vulkan_queue_can_fill_native(target_offset,
                                             submission->fill.length,
                                             submission->fill.pattern_length)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native queue fill requires 4-byte target alignment, 4-byte "
        "length, and a 1-, 2-, or 4-byte pattern");
  }

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      &begin_info));
  uint32_t pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_expand_fill_pattern(
      submission->fill.pattern, submission->fill.pattern_length, &pattern));
  iree_vkCmdFillBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                       submission->native_command_buffer, target_handle,
                       (VkDeviceSize)target_offset,
                       (VkDeviceSize)submission->fill.length, pattern);
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 submission->native_command_buffer);
}

static void iree_hal_vulkan_queue_execute_fill(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->fill.length == 0 || submission->native_command_buffer) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_status_t status = iree_hal_buffer_map_fill(
      submission->fill.target_buffer, submission->fill.target_offset,
      submission->fill.length, submission->fill.pattern,
      submission->fill.pattern_length);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static iree_status_t iree_hal_vulkan_queue_record_update_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(submission->update.target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      submission->update.target_buffer, &target_backing));
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t target_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      submission->update.target_buffer, target_backing,
      submission->update.target_offset, &target_offset));
  if (target_offset % sizeof(uint32_t) != 0 ||
      submission->update.length % sizeof(uint32_t) != 0 ||
      submission->update.length > 65536) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native queue update requires 4-byte target alignment, 4-byte "
        "length, and at most 65536 bytes");
  }

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      &begin_info));
  iree_vkCmdUpdateBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), submission->native_command_buffer,
      target_handle, (VkDeviceSize)target_offset,
      (VkDeviceSize)submission->update.length, submission->update.source_data);
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 submission->native_command_buffer);
}

static void iree_hal_vulkan_queue_execute_update(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->update.length == 0) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  if (submission->native_command_buffer) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_status_t status = iree_hal_buffer_map_write(
      submission->update.target_buffer, submission->update.target_offset,
      submission->update.source_data, submission->update.length);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static iree_status_t iree_hal_vulkan_queue_record_copy_native_buffers(
    iree_hal_vulkan_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    VkCommandBuffer command_buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  iree_hal_buffer_t* source_backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(source_buffer, &source_backing));
  iree_hal_buffer_t* target_backing = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_resolve_backing(target_buffer, &target_backing));
  VkDeviceMemory source_memory = VK_NULL_HANDLE;
  VkBuffer source_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      source_backing, &source_memory, &source_handle));
  (void)source_memory;
  VkDeviceMemory target_memory = VK_NULL_HANDLE;
  VkBuffer target_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_handle(
      target_backing, &target_memory, &target_handle));
  (void)target_memory;

  iree_device_size_t source_backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      source_buffer, source_backing, source_offset, &source_backing_offset));
  iree_device_size_t target_backing_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      target_buffer, target_backing, target_offset, &target_backing_offset));

  VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  IREE_RETURN_IF_ERROR(iree_vkBeginCommandBuffer(
      IREE_VULKAN_DEVICE(&queue->syms), command_buffer, &begin_info));
  VkBufferCopy copy_region = {
      .srcOffset = (VkDeviceSize)source_backing_offset,
      .dstOffset = (VkDeviceSize)target_backing_offset,
      .size = (VkDeviceSize)length,
  };
  iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(&queue->syms), command_buffer,
                       source_handle, target_handle, /*regionCount=*/1,
                       &copy_region);
  return iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(&queue->syms),
                                 command_buffer);
}

static iree_status_t iree_hal_vulkan_queue_record_copy_native(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_vulkan_queue_record_copy_native_buffers(
      queue, submission->copy.source_buffer, submission->copy.source_offset,
      submission->copy.target_buffer, submission->copy.target_offset,
      submission->copy.length, submission->native_command_buffer);
}

static void iree_hal_vulkan_queue_execute_copy(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->copy.length == 0) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }
  if (submission->native_command_buffer) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_status_t status = iree_hal_buffer_map_copy(
      submission->copy.source_buffer, submission->copy.source_offset,
      submission->copy.target_buffer, submission->copy.target_offset,
      submission->copy.length);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_execute_write(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->write.length == 0) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_hal_buffer_t* source_buffer = submission->write.staging_buffer
                                         ? submission->write.staging_buffer
                                         : submission->write.source_buffer;
  const iree_device_size_t source_offset =
      submission->write.staging_buffer ? 0 : submission->write.source_offset;
  iree_status_t status = iree_hal_file_write(
      submission->write.target_file, submission->write.target_offset,
      source_buffer, source_offset, submission->write.length);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_execute_read(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->read.length == 0) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_status_t status = iree_hal_file_read(
      submission->read.source_file, submission->read.source_offset,
      submission->read.target_buffer, submission->read.target_offset,
      submission->read.length);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_execute_command_buffer(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (!iree_status_is_ok(completion_status)) {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
    return;
  }
  if (submission->native_command_buffer) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
    return;
  }

  iree_hal_buffer_binding_table_t binding_table = {
      .count = submission->execute.binding_table_count,
      .bindings = submission->execute.binding_table_bindings,
  };
  iree_status_t status = iree_hal_vulkan_command_buffer_replay_host(
      submission->execute.command_buffer, binding_table);
  if (iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(signal_semaphore_list, status);
  }
}

static void iree_hal_vulkan_queue_complete_alloca(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_local_transient_buffer_commit(submission->alloca.buffer);
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_local_transient_buffer_decommit(submission->alloca.buffer);
    iree_hal_local_transient_buffer_release_reservation(
        submission->alloca.buffer, submission->alloca.wait_frontier);
    submission->alloca.wait_frontier = NULL;
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_complete_dealloca(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  iree_hal_semaphore_list_t signal_semaphore_list =
      submission->signal_semaphore_list;
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  if (iree_status_is_ok(completion_status)) {
    iree_hal_local_transient_buffer_decommit(submission->dealloca.buffer);
    iree_hal_local_transient_buffer_release_reservation(
        submission->dealloca.buffer, frontier);
    iree_hal_vulkan_queue_signal_list_or_fail(signal_semaphore_list, frontier);
  } else {
    iree_hal_vulkan_queue_fail_signal_list(
        signal_semaphore_list, iree_status_clone(completion_status));
  }
}

static void iree_hal_vulkan_queue_complete_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t completion_status) {
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER:
      if (iree_status_is_ok(completion_status)) {
        iree_hal_vulkan_queue_signal_list_or_fail(
            submission->signal_semaphore_list,
            iree_async_fixed_frontier_as_const_frontier(&submission->frontier));
      } else {
        iree_hal_vulkan_queue_fail_signal_list(
            submission->signal_semaphore_list,
            iree_status_clone(completion_status));
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL:
      iree_hal_vulkan_queue_execute_host_call(queue, submission,
                                              completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL:
      iree_hal_vulkan_queue_execute_fill(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE:
      iree_hal_vulkan_queue_execute_update(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY:
      iree_hal_vulkan_queue_execute_copy(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_WRITE:
      iree_hal_vulkan_queue_execute_write(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_READ:
      iree_hal_vulkan_queue_execute_read(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE:
      iree_hal_vulkan_queue_execute_command_buffer(submission,
                                                   completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      iree_hal_vulkan_queue_complete_alloca(submission, completion_status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      iree_hal_vulkan_queue_complete_dealloca(submission, completion_status);
      break;
    default:
      iree_hal_vulkan_queue_fail_signal_list(
          submission->signal_semaphore_list,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "unknown Vulkan queue submission kind %u",
                           (uint32_t)submission->kind));
      break;
  }

  if (iree_status_is_ok(completion_status) && queue->frontier_tracker) {
    iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                        submission->epoch);
  }
  iree_atomic_store(&queue->last_drained_epoch, (int64_t)submission->epoch,
                    iree_memory_order_release);
}

static void iree_hal_vulkan_queue_fail_unsubmitted_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  IREE_ASSERT(!iree_status_is_ok(status),
              "unsubmitted queue failure status must be non-OK");
  switch (submission->kind) {
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA:
      iree_hal_local_transient_buffer_decommit(submission->alloca.buffer);
      iree_hal_local_transient_buffer_release_reservation(
          submission->alloca.buffer, submission->alloca.wait_frontier);
      submission->alloca.wait_frontier = NULL;
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
    case IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA:
      iree_hal_local_transient_buffer_abort_dealloca(
          submission->dealloca.buffer);
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
    default:
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             status);
      break;
  }
  iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
}

static iree_status_t iree_hal_vulkan_queue_resolve_waits(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_vulkan_queue_frontier_t* pending_frontier,
    bool allow_software_deferral,
    iree_hal_vulkan_queue_wait_resolution_t* resolution) {
  resolution->wait_info_count = 0;
  resolution->needs_deferral = false;
  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* semaphore = wait_semaphore_list.semaphores[i];
    const uint64_t minimum_value = wait_semaphore_list.payload_values[i];

    uint64_t current_value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(semaphore, &current_value));
    if (current_value >= minimum_value) continue;

    iree_hal_vulkan_last_signal_flags_t signal_flags = 0;
    iree_async_axis_t producer_axis = 0;
    uint64_t producer_epoch = 0;
    uint64_t producer_value = 0;
    const bool has_last_signal = iree_hal_vulkan_last_signal_load(
        iree_hal_vulkan_semaphore_last_signal(semaphore), &signal_flags,
        &producer_axis, &producer_epoch, &producer_value);
    (void)signal_flags;
    if (!has_last_signal || producer_value < minimum_value) {
      if (allow_software_deferral) {
        resolution->needs_deferral = true;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan queue wait on semaphore %" PRIhsz
          " requires software deferral; only already-signaled waits and waits "
          "backed by a published producer epoch are accepted",
          i);
    }

    if (producer_axis == queue->axis &&
        producer_epoch >= queue->next_epoch_value) {
      if (allow_software_deferral) {
        resolution->needs_deferral = true;
        return iree_ok_status();
      }
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan queue wait on same-queue future signal requires software "
          "deferral; only prior accepted epochs are accepted");
    }

    iree_async_single_frontier_t producer_frontier;
    iree_async_single_frontier_initialize(&producer_frontier, producer_axis,
                                          producer_epoch);
    if (!iree_async_frontier_merge(
            iree_async_fixed_frontier_as_frontier(pending_frontier),
            IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY,
            iree_async_single_frontier_as_const_frontier(&producer_frontier))) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "Vulkan queue frontier capacity exhausted while resolving waits");
    }

    VkSemaphore wait_handle = VK_NULL_HANDLE;
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_semaphore_handle(semaphore, &wait_handle));
    if (resolution->wait_info_count >= resolution->wait_info_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "too many native Vulkan wait semaphores for queue submission");
    }
    resolution->wait_infos[resolution->wait_info_count] =
        (VkSemaphoreSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = wait_handle,
            .value = minimum_value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };
    resolution->wait_info_count = resolution->wait_info_count + 1;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_publish_signals(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  const iree_async_frontier_t* frontier =
      iree_async_fixed_frontier_as_const_frontier(&submission->frontier);
  for (iree_host_size_t i = 0; i < submission->signal_semaphore_list.count;
       ++i) {
    iree_hal_vulkan_semaphore_publish_signal(
        submission->signal_semaphore_list.semaphores[i], queue->axis, frontier,
        submission->epoch, submission->signal_semaphore_list.payload_values[i]);
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_queue_alloca_pool_notification_end_observe(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->alloca.pool_notification_observation_held) {
    submission->alloca.pool_notification_observation_held = false;
    iree_async_notification_end_observe(submission->alloca.pool_notification);
  }
}

static bool iree_hal_vulkan_queue_alloca_has_staged_backing(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_hal_local_transient_buffer_backing_buffer(
             submission->alloca.buffer) != NULL;
}

static iree_status_t iree_hal_vulkan_queue_stage_alloca_reservation(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_pool_acquire_result_t acquire_result) {
  if (acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT &&
      !iree_all_bits_set(submission->alloca.flags,
                         IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
    iree_hal_pool_release_reservation(submission->alloca.pool, reservation,
                                      wait_frontier);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue_alloca recycled pool memory requires "
                            "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
  }

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_status_t status = iree_hal_pool_materialize_reservation(
      submission->alloca.pool, submission->alloca.params, reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_local_transient_buffer_attach_reservation(
        submission->alloca.buffer, submission->alloca.pool, reservation);
    iree_hal_local_transient_buffer_stage_backing(submission->alloca.buffer,
                                                  backing_buffer);
    submission->alloca.wait_frontier =
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ? wait_frontier
                                                              : NULL;
    submission->alloca.memory_wait_kind =
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
            ? IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER
            : IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  } else {
    iree_hal_pool_release_reservation(
        submission->alloca.pool, reservation,
        acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT ? wait_frontier
                                                              : NULL);
  }
  iree_hal_buffer_release(backing_buffer);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_alloca_pool_notification(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_async_frontier_t* requester_frontier) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(submission->alloca.pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca exhausted pool did not provide a notification");
  }

  submission->alloca.pool_notification = notification;
  submission->alloca.pool_notification_wait_token =
      iree_async_notification_begin_observe(notification);
  submission->alloca.pool_notification_observation_held = true;

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  iree_status_t status = iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result);

  if (iree_status_is_ok(status)) {
    switch (acquire_result) {
      case IREE_HAL_POOL_ACQUIRE_OK:
      case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
        iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
        status = iree_hal_vulkan_queue_stage_alloca_reservation(
            submission, &reservation, acquire_info.wait_frontier,
            acquire_result);
        break;
      case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
      case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION;
        break;
      default:
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "unrecognized pool acquire result %u",
                                  acquire_result);
        break;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_alloca_backing(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    const iree_async_frontier_t* requester_frontier,
    bool* out_needs_memory_wait) {
  *out_needs_memory_wait = false;
  if (iree_hal_vulkan_queue_alloca_has_staged_backing(submission)) {
    *out_needs_memory_wait = submission->alloca.memory_wait_kind !=
                             IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    return iree_ok_status();
  }
  if (submission->alloca.memory_wait_kind !=
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE) {
    *out_needs_memory_wait = true;
    return iree_ok_status();
  }

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result));

  iree_status_t status = iree_ok_status();
  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      status = iree_hal_vulkan_queue_stage_alloca_reservation(
          submission, &reservation, acquire_info.wait_frontier, acquire_result);
      break;
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      status = iree_hal_vulkan_queue_prepare_alloca_pool_notification(
          submission, requester_frontier);
      break;
    default:
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "unrecognized pool acquire result %u",
                                acquire_result);
      break;
  }
  if (iree_status_is_ok(status)) {
    *out_needs_memory_wait = submission->alloca.memory_wait_kind !=
                             IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
  }
  return status;
}

static iree_status_t iree_hal_vulkan_queue_try_stage_alloca_backing_now(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_frontier_t requester_frontier_storage;
  iree_slim_mutex_lock(&queue->submission_mutex);
  requester_frontier_storage = queue->frontier;
  iree_slim_mutex_unlock(&queue->submission_mutex);
  const iree_async_frontier_t* requester_frontier =
      iree_async_fixed_frontier_as_const_frontier(&requester_frontier_storage);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t acquire_result =
      IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      submission->alloca.pool, submission->alloca.allocation_size,
      submission->alloca.params.min_alignment
          ? submission->alloca.params.min_alignment
          : 1,
      requester_frontier, submission->alloca.reserve_flags, &reservation,
      &acquire_info, &acquire_result));

  switch (acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      return iree_hal_vulkan_queue_stage_alloca_reservation(
          submission, &reservation, acquire_info.wait_frontier, acquire_result);
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              acquire_result);
  }
}

static iree_status_t iree_hal_vulkan_queue_submit_native_under_lock(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    bool allow_software_deferral,
    iree_hal_vulkan_queue_wait_resolution_t* resolution,
    iree_hal_vulkan_queue_submission_result_t* out_result) {
  out_result->submitted = false;
  out_result->memory_wait_submission = NULL;
  out_result->queue_failure_status = iree_ok_status();
  resolution->wait_info_count = 0;
  resolution->needs_deferral = false;

  iree_status_t status = iree_hal_vulkan_queue_check_error(queue);
  if (iree_status_is_ok(status) && !queue->frontier_tracker) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "Vulkan queue frontier is not assigned");
  }
  if (iree_status_is_ok(status) &&
      iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire)) {
    status = iree_make_status(IREE_STATUS_CANCELLED,
                              "Vulkan queue is shutting down");
  }
  if (iree_status_is_ok(status)) {
    submission->epoch = queue->next_epoch_value;
    submission->frontier = queue->frontier;
    status = iree_hal_vulkan_queue_resolve_waits(
        queue, submission->wait_semaphore_list, &submission->frontier,
        allow_software_deferral, resolution);
  }
  if (iree_status_is_ok(status) && resolution->needs_deferral) {
    return iree_ok_status();
  }
  if (iree_status_is_ok(status) &&
      submission->kind == IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA) {
    bool needs_memory_wait = false;
    status = iree_hal_vulkan_queue_prepare_alloca_backing(
        submission,
        iree_async_fixed_frontier_as_const_frontier(&submission->frontier),
        &needs_memory_wait);
    if (iree_status_is_ok(status) && needs_memory_wait) {
      iree_atomic_store(&submission->deferred_state,
                        IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING,
                        iree_memory_order_release);
      iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                        iree_memory_order_relaxed);
      iree_hal_vulkan_queue_append_deferred_submission(queue, submission);
      out_result->memory_wait_submission = submission;
      return iree_ok_status();
    }
  }
  if (iree_status_is_ok(status)) {
    iree_async_single_frontier_t self_frontier;
    iree_async_single_frontier_initialize(&self_frontier, queue->axis,
                                          submission->epoch);
    if (!iree_async_frontier_merge(
            iree_async_fixed_frontier_as_frontier(&submission->frontier),
            IREE_HAL_VULKAN_QUEUE_FRONTIER_CAPACITY,
            iree_async_single_frontier_as_const_frontier(&self_frontier))) {
      status = iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "Vulkan queue frontier capacity exhausted while publishing epoch");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_publish_signals(queue, submission);
  }
  if (iree_status_is_ok(status)) {
    VkSemaphoreSubmitInfo epoch_signal_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = queue->epoch_semaphore,
        .value = submission->epoch,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };
    VkCommandBufferSubmitInfo command_buffer_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = submission->native_command_buffer,
        .deviceMask = 0,
    };
    VkSubmitInfo2 submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = resolution->wait_info_count,
        .pWaitSemaphoreInfos = resolution->wait_infos,
        .commandBufferInfoCount = submission->native_command_buffer ? 1u : 0u,
        .pCommandBufferInfos =
            submission->native_command_buffer ? &command_buffer_info : NULL,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &epoch_signal_info,
    };
    iree_slim_mutex_lock(queue->queue_handle_mutex);
    status = iree_vkQueueSubmit2(IREE_VULKAN_DEVICE(&queue->syms), queue->queue,
                                 1, &submit_info, VK_NULL_HANDLE);
    iree_slim_mutex_unlock(queue->queue_handle_mutex);
    if (iree_status_is_ok(status)) {
      queue->frontier = submission->frontier;
      queue->next_epoch_value = queue->next_epoch_value + 1;
      iree_hal_vulkan_queue_append_pending_submission(queue, submission);
      out_result->submitted = true;
    } else if (iree_status_code(status) == IREE_STATUS_DATA_LOSS) {
      out_result->queue_failure_status =
          iree_hal_vulkan_queue_store_error(queue, iree_status_clone(status));
    }
  }
  return status;
}

static bool iree_hal_vulkan_queue_wait_entry_callback_is_complete(
    const iree_hal_vulkan_queue_wait_entry_t* entry) {
  return iree_atomic_load(&entry->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_vulkan_queue_wait_entry_publish_callback_complete(
    iree_hal_vulkan_queue_wait_entry_t* entry) {
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&entry->submission->callback_notification,
                         IREE_ALL_WAITERS);
}

static bool iree_hal_vulkan_queue_wait_callbacks_are_complete(void* user_data) {
  iree_hal_vulkan_queue_pending_submission_t* submission =
      (iree_hal_vulkan_queue_pending_submission_t*)user_data;
  for (iree_host_size_t i = 0; i < submission->wait_entry_count; ++i) {
    if (!iree_hal_vulkan_queue_wait_entry_callback_is_complete(
            &submission->wait_entries[i])) {
      return false;
    }
  }
  return true;
}

static void iree_hal_vulkan_queue_submission_record_wait_status(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &submission->wait_failure_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static bool iree_hal_vulkan_queue_submission_mark_ready(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING;
  return iree_atomic_compare_exchange_strong(
      &submission->deferred_state, &expected_state,
      IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_READY, iree_memory_order_acq_rel,
      iree_memory_order_acquire);
}

static void iree_hal_vulkan_queue_deferred_submission_ready(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_t* queue = submission->queue;
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_vulkan_queue_unlink_deferred_submission(queue, submission);
  iree_hal_vulkan_queue_append_ready_submission(queue, submission);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  iree_status_t wake_status = iree_hal_vulkan_queue_signal_wakeup(queue);
  if (!iree_status_is_ok(wake_status)) {
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, wake_status);
    iree_status_free(stored_status);
  }
}

static void iree_hal_vulkan_queue_deferred_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_vulkan_queue_wait_entry_t* entry =
      (iree_hal_vulkan_queue_wait_entry_t*)user_data;
  iree_hal_vulkan_queue_pending_submission_t* submission = entry->submission;
  iree_hal_vulkan_queue_submission_record_wait_status(submission, status);

  const int32_t previous_count = iree_atomic_fetch_sub(
      &submission->wait_count, 1, iree_memory_order_acq_rel);
  const bool owns_ready =
      previous_count == 1 &&
      iree_hal_vulkan_queue_submission_mark_ready(submission);

  iree_hal_vulkan_queue_wait_entry_publish_callback_complete(entry);
  if (!owns_ready) return;

  iree_notification_await(&submission->callback_notification,
                          iree_hal_vulkan_queue_wait_callbacks_are_complete,
                          submission, iree_infinite_timeout());
  iree_hal_vulkan_queue_deferred_submission_ready(submission);
}

static iree_status_t iree_hal_vulkan_queue_prepare_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  const iree_host_size_t wait_count = submission->wait_semaphore_list.count;
  if (wait_count > INT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many software-deferred Vulkan queue waits");
  }
  iree_host_size_t wait_entries_size = 0;
  if (!iree_host_size_checked_mul(wait_count, sizeof(*submission->wait_entries),
                                  &wait_entries_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "software-deferred Vulkan wait list is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, wait_entries_size,
                            (void**)&submission->wait_entries));
  memset(submission->wait_entries, 0, wait_entries_size);
  submission->wait_entry_count = wait_count;
  iree_atomic_store(&submission->wait_count, (int32_t)wait_count,
                    iree_memory_order_release);
  iree_atomic_store(&submission->deferred_state,
                    IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING,
                    iree_memory_order_release);
  for (iree_host_size_t i = 0; i < wait_count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    entry->submission = submission;
    iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_relaxed);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_start_deferred_submission(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_semaphore_list_t wait_semaphores = submission->wait_semaphore_list;
  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_vulkan_queue_deferred_wait_resolved;
    entry->timepoint.user_data = entry;
    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_submission_record_wait_status(submission, status);
      iree_atomic_store(&entry->callback_complete, 1,
                        iree_memory_order_release);
      const int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      const int32_t previous_count = iree_atomic_fetch_sub(
          &submission->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered &&
          iree_hal_vulkan_queue_submission_mark_ready(submission)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_wait_callbacks_are_complete, submission,
            iree_infinite_timeout());
        iree_hal_vulkan_queue_deferred_submission_ready(submission);
      }
      return iree_ok_status();
    }
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
    const iree_hal_vulkan_queue_pending_submission_t* submission) {
  return iree_atomic_load(&submission->alloca.memory_wait_callback_complete,
                          iree_memory_order_acquire) != 0;
}

static bool iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk(
    void* user_data) {
  return iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
      (const iree_hal_vulkan_queue_pending_submission_t*)user_data);
}

static void iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 1,
                    iree_memory_order_release);
  iree_notification_post(&submission->callback_notification, IREE_ALL_WAITERS);
}

static void iree_hal_vulkan_queue_alloca_memory_wait_resolved(
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  if (iree_status_is_ok(status)) {
    switch (submission->alloca.memory_wait_kind) {
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
        submission->alloca.memory_wait_kind =
            IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
        submission->alloca.pool_notification = NULL;
        break;
      case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
        break;
    }
  }

  iree_hal_vulkan_queue_submission_record_wait_status(submission, status);
  const bool owns_ready =
      iree_hal_vulkan_queue_submission_mark_ready(submission);
  iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(submission);
  if (owns_ready) {
    iree_hal_vulkan_queue_deferred_submission_ready(submission);
  }
}

static void iree_hal_vulkan_queue_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_vulkan_queue_alloca_memory_wait_resolved(
      (iree_hal_vulkan_queue_pending_submission_t*)user_data, status);
}

static void iree_hal_vulkan_queue_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_vulkan_queue_alloca_memory_wait_resolved(
      (iree_hal_vulkan_queue_pending_submission_t*)user_data, status);
}

static void iree_hal_vulkan_queue_start_alloca_frontier_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (IREE_UNLIKELY(!submission->alloca.wait_frontier)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(
        submission,
        iree_make_status(
            IREE_STATUS_INTERNAL,
            "queue_alloca waitable pool reservation did not provide a "
            "frontier"));
    return;
  }
  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 0,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      submission->queue->frontier_tracker, submission->alloca.wait_frontier,
      iree_hal_vulkan_queue_alloca_frontier_wait_resolved, submission,
      &submission->alloca.frontier_waiter);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
  }
}

static void iree_hal_vulkan_queue_start_alloca_pool_notification_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_async_notification_wait_operation_t* wait_op =
      &submission->alloca.pool_notification_wait_ops
           [submission->alloca.pool_notification_wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_vulkan_queue_alloca_pool_notification_wait_resolved, submission);
  wait_op->notification = submission->alloca.pool_notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = submission->alloca.pool_notification_wait_token;

  iree_atomic_store(&submission->alloca.memory_wait_callback_complete, 0,
                    iree_memory_order_relaxed);
  iree_status_t status = iree_async_proactor_submit_one(
      submission->queue->proactor, &wait_op->base);
  iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_alloca_memory_wait_resolved(submission, status);
  }
}

static iree_status_t iree_hal_vulkan_queue_start_alloca_memory_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  switch (submission->alloca.memory_wait_kind) {
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_vulkan_queue_start_alloca_frontier_wait(submission);
      return iree_ok_status();
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      submission->alloca.pool_notification_wait_slot =
          (uint8_t)((submission->alloca.pool_notification_wait_slot + 1u) & 1u);
      iree_hal_vulkan_queue_start_alloca_pool_notification_wait(submission);
      return iree_ok_status();
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan alloca submission has no memory wait to start");
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "unrecognized Vulkan alloca memory wait kind %u",
                          submission->alloca.memory_wait_kind);
}

static iree_status_t iree_hal_vulkan_queue_allocate_wait_infos(
    iree_hal_vulkan_queue_t* queue, iree_host_size_t wait_count,
    VkSemaphoreSubmitInfo** out_wait_infos, uint32_t* out_wait_info_capacity) {
  *out_wait_infos = NULL;
  *out_wait_info_capacity = 0;
  if (wait_count == 0) return iree_ok_status();
  if (wait_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "too many Vulkan queue wait semaphores");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, wait_count * sizeof(**out_wait_infos),
      (void**)out_wait_infos));
  *out_wait_info_capacity = (uint32_t)wait_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_queue_submission_take_wait_failure(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  return (iree_status_t)iree_atomic_exchange(&submission->wait_failure_status,
                                             0, iree_memory_order_acquire);
}

static void iree_hal_vulkan_queue_cancel_alloca_memory_wait(
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  if (submission->kind != IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA) return;
  if (submission->alloca.memory_wait_kind ==
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE) {
    return;
  }

  switch (submission->alloca.memory_wait_kind) {
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_FRONTIER:
      if (iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
              submission)) {
        break;
      }
      if (!iree_async_frontier_tracker_cancel_wait(
              submission->queue->frontier_tracker,
              &submission->alloca.frontier_waiter)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk,
            submission, iree_infinite_timeout());
      } else {
        iree_hal_vulkan_queue_alloca_memory_wait_publish_complete(submission);
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_vulkan_queue_alloca_pool_notification_end_observe(submission);
      if (submission->alloca.pool_notification) {
        iree_async_notification_signal(submission->alloca.pool_notification,
                                       INT32_MAX);
      }
      if (!iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete(
              submission)) {
        iree_notification_await(
            &submission->callback_notification,
            iree_hal_vulkan_queue_alloca_memory_wait_callback_is_complete_thunk,
            submission, iree_infinite_timeout());
      }
      break;
    case IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
  submission->alloca.memory_wait_kind =
      IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
}

static void iree_hal_vulkan_queue_cancel_deferred_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_status_t status) {
  int32_t expected_state = IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_PENDING;
  if (!iree_atomic_compare_exchange_strong(
          &submission->deferred_state, &expected_state,
          IREE_HAL_VULKAN_QUEUE_DEFERRED_STATE_CANCELLING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_status_free(status);
    return;
  }

  for (iree_host_size_t i = 0; i < submission->wait_entry_count; ++i) {
    iree_hal_vulkan_queue_wait_entry_t* entry = &submission->wait_entries[i];
    if (iree_hal_vulkan_queue_wait_entry_callback_is_complete(entry)) continue;
    if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                              &entry->timepoint)) {
      iree_hal_vulkan_queue_wait_entry_publish_callback_complete(entry);
    }
  }
  iree_notification_await(&submission->callback_notification,
                          iree_hal_vulkan_queue_wait_callbacks_are_complete,
                          submission, iree_infinite_timeout());
  iree_hal_vulkan_queue_cancel_alloca_memory_wait(submission);
  iree_hal_vulkan_queue_fail_unsubmitted_submission(queue, submission, status);
}

static void iree_hal_vulkan_queue_cancel_deferred_submissions(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  iree_hal_vulkan_queue_pending_submission_t* deferred_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  deferred_head = queue->deferred_head;
  queue->deferred_head = NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  while (deferred_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = deferred_head->next;
    deferred_head->next = NULL;
    iree_hal_vulkan_queue_cancel_deferred_submission(queue, deferred_head,
                                                     iree_status_clone(status));
    deferred_head = next;
  }
  iree_status_free(status);
}

static void iree_hal_vulkan_queue_fail_pending_submissions(
    iree_hal_vulkan_queue_t* queue, iree_status_t status) {
  iree_hal_vulkan_queue_pending_submission_t* pending_head = NULL;
  iree_hal_vulkan_queue_pending_submission_t* ready_head = NULL;
  iree_hal_vulkan_queue_pending_submission_t* deferred_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  pending_head = queue->pending_head;
  ready_head = queue->ready_head;
  deferred_head = queue->deferred_head;
  queue->pending_head = NULL;
  queue->pending_tail = NULL;
  queue->ready_head = NULL;
  queue->ready_tail = NULL;
  queue->deferred_head = NULL;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_fail_axis(queue->frontier_tracker, queue->axis,
                                          iree_status_clone(status));
  }

  while (pending_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = pending_head->next;
    pending_head->next = NULL;
    iree_hal_vulkan_queue_complete_submission(queue, pending_head, status);
    iree_hal_vulkan_queue_pending_submission_destroy(queue, pending_head);
    pending_head = next;
  }
  while (ready_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = ready_head->next;
    ready_head->next = NULL;
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, ready_head, iree_status_clone(status));
    ready_head = next;
  }
  while (deferred_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = deferred_head->next;
    deferred_head->next = NULL;
    iree_hal_vulkan_queue_cancel_deferred_submission(queue, deferred_head,
                                                     iree_status_clone(status));
    deferred_head = next;
  }
  iree_status_free(status);
}

static void iree_hal_vulkan_queue_drain_ready_submissions(
    iree_hal_vulkan_queue_t* queue) {
  while (true) {
    iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
    iree_hal_vulkan_queue_submission_result_t submit_result = {
        .submitted = false,
        .queue_failure_status = iree_ok_status(),
    };
    iree_status_t status = iree_ok_status();

    iree_slim_mutex_lock(&queue->submission_mutex);
    submission = iree_hal_vulkan_queue_pop_ready_submission(queue);
    iree_slim_mutex_unlock(&queue->submission_mutex);
    if (!submission) return;

    VkSemaphoreSubmitInfo* wait_infos = NULL;
    uint32_t wait_info_capacity = 0;
    status = iree_hal_vulkan_queue_allocate_wait_infos(
        queue, submission->wait_semaphore_list.count, &wait_infos,
        &wait_info_capacity);
    if (iree_status_is_ok(status)) {
      iree_slim_mutex_lock(&queue->submission_mutex);
      status = iree_hal_vulkan_queue_submission_take_wait_failure(submission);
      if (iree_status_is_ok(status)) {
        if (wait_info_capacity < submission->wait_semaphore_list.count) {
          status = iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "too many Vulkan queue wait semaphores for ready submission");
        } else {
          iree_hal_vulkan_queue_wait_resolution_t resolution = {
              .wait_infos = wait_infos,
              .wait_info_capacity = wait_info_capacity,
          };
          status = iree_hal_vulkan_queue_submit_native_under_lock(
              queue, submission, /*allow_software_deferral=*/false, &resolution,
              &submit_result);
        }
      }
      iree_slim_mutex_unlock(&queue->submission_mutex);
    }

    if (wait_infos) {
      iree_allocator_free(queue->host_allocator, wait_infos);
    }

    if (!iree_status_is_ok(submit_result.queue_failure_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(
          queue, submit_result.queue_failure_status);
    }
    if (submit_result.memory_wait_submission) {
      iree_status_t memory_wait_status =
          iree_hal_vulkan_queue_start_alloca_memory_wait(
              submit_result.memory_wait_submission);
      if (!iree_status_is_ok(memory_wait_status)) {
        iree_hal_vulkan_queue_fail_pending_submissions(queue,
                                                       memory_wait_status);
      }
      iree_status_free(status);
      continue;
    }
    if (submit_result.submitted) {
      iree_status_free(status);
      continue;
    }

    if (iree_status_is_ok(status)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan ready submission was neither submitted nor failed");
    }
    iree_hal_vulkan_queue_fail_unsubmitted_submission(queue, submission,
                                                      status);
  }
}

static int iree_hal_vulkan_queue_completion_thread_main(void* entry_arg) {
  iree_hal_vulkan_queue_t* queue = (iree_hal_vulkan_queue_t*)entry_arg;
  uint64_t observed_wakeup_value = (uint64_t)iree_atomic_load(
      &queue->wakeup_value, iree_memory_order_acquire);

  while (true) {
    iree_hal_vulkan_queue_drain_ready_submissions(queue);
    iree_hal_vulkan_queue_drain_completions(queue);
    const bool stop_requested =
        iree_atomic_load(&queue->stop_requested, iree_memory_order_acquire) !=
        0;
    if (stop_requested) {
      iree_hal_vulkan_queue_cancel_deferred_submissions(
          queue, iree_status_from_code(IREE_STATUS_CANCELLED));
      iree_hal_vulkan_queue_drain_ready_submissions(queue);
      iree_hal_vulkan_queue_drain_completions(queue);
    }
    if (stop_requested && !iree_hal_vulkan_queue_has_pending(queue)) break;

    VkSemaphore wait_semaphores[2] = {
        queue->epoch_semaphore,
        queue->wakeup_semaphore,
    };
    uint64_t wait_values[2] = {
        (uint64_t)iree_atomic_load(&queue->last_drained_epoch,
                                   iree_memory_order_acquire) +
            1,
        observed_wakeup_value + 1,
    };
    VkSemaphoreWaitInfo wait_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .flags = VK_SEMAPHORE_WAIT_ANY_BIT,
        .semaphoreCount = IREE_ARRAYSIZE(wait_semaphores),
        .pSemaphores = wait_semaphores,
        .pValues = wait_values,
    };
    VkResult result = iree_vkWaitSemaphores_raw(
        &queue->syms, queue->logical_device, &wait_info, UINT64_MAX);
    if (result == VK_SUCCESS) {
      observed_wakeup_value = (uint64_t)iree_atomic_load(
          &queue->wakeup_value, iree_memory_order_acquire);
      continue;
    }

    iree_status_t status = iree_status_from_vk_result(
        __FILE__, __LINE__, result, "vkWaitSemaphores");
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, status);
    iree_hal_vulkan_queue_fail_pending_submissions(queue, stored_status);
    break;
  }

  return 0;
}

iree_status_t iree_hal_vulkan_queue_initialize(
    const iree_hal_vulkan_queue_params_t* params,
    iree_hal_vulkan_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(params->device);
  IREE_ASSERT_ARGUMENT(params->syms);
  IREE_ASSERT_ARGUMENT(params->logical_device);
  IREE_ASSERT_ARGUMENT(params->queue);
  IREE_ASSERT_ARGUMENT(params->queue_handle_mutex);
  IREE_ASSERT_ARGUMENT(params->proactor);
  IREE_ASSERT_ARGUMENT(out_queue);
  memset(out_queue, 0, sizeof(*out_queue));

  out_queue->device = params->device;
  out_queue->syms = *params->syms;
  out_queue->logical_device = params->logical_device;
  out_queue->queue = params->queue;
  out_queue->queue_handle_mutex = params->queue_handle_mutex;
  out_queue->proactor = params->proactor;
  out_queue->queue_family_index = params->queue_family_index;
  out_queue->queue_index = params->queue_index;
  out_queue->queue_affinity = params->queue_affinity;
  out_queue->role = params->role;
  out_queue->host_allocator = params->host_allocator;
  out_queue->next_epoch_value = 1;
  iree_slim_mutex_initialize(&out_queue->submission_mutex);
  iree_async_frontier_initialize(
      iree_async_fixed_frontier_as_frontier(&out_queue->frontier),
      /*entry_count=*/0);
  iree_atomic_store(&out_queue->last_drained_epoch, 0,
                    iree_memory_order_release);
  iree_atomic_store(&out_queue->wakeup_value, 0, iree_memory_order_release);

  iree_status_t status = iree_hal_vulkan_queue_create_timeline_semaphore(
      &out_queue->syms, out_queue->logical_device, /*initial_value=*/0,
      &out_queue->epoch_semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_timeline_semaphore(
        &out_queue->syms, out_queue->logical_device, /*initial_value=*/0,
        &out_queue->wakeup_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_command_pool(
        &out_queue->syms, out_queue->logical_device,
        out_queue->queue_family_index, &out_queue->command_pool);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_queue_deinitialize(out_queue);
  }
  return status;
}

void iree_hal_vulkan_queue_deinitialize(iree_hal_vulkan_queue_t* queue) {
  if (!queue->logical_device) return;
  iree_hal_vulkan_queue_retire_frontier(queue);
  if (queue->command_pool) {
    iree_vkDestroyCommandPool(IREE_VULKAN_DEVICE(&queue->syms),
                              queue->logical_device, queue->command_pool,
                              /*pAllocator=*/NULL);
    queue->command_pool = VK_NULL_HANDLE;
  }
  if (queue->epoch_semaphore) {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, queue->epoch_semaphore,
                            /*pAllocator=*/NULL);
    queue->epoch_semaphore = VK_NULL_HANDLE;
  }
  if (queue->wakeup_semaphore) {
    iree_vkDestroySemaphore(IREE_VULKAN_DEVICE(&queue->syms),
                            queue->logical_device, queue->wakeup_semaphore,
                            /*pAllocator=*/NULL);
    queue->wakeup_semaphore = VK_NULL_HANDLE;
  }
  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  iree_status_free(failure_status);
  iree_slim_mutex_deinitialize(&queue->submission_mutex);
  memset(queue, 0, sizeof(*queue));
}

iree_status_t iree_hal_vulkan_queue_assign_frontier(
    iree_hal_vulkan_queue_t* queue,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  if (queue->frontier_tracker) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan queue frontier already assigned");
  }

  IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL));
  queue->frontier_tracker = frontier_tracker;
  queue->axis = axis;
  iree_atomic_store(&queue->stop_requested, 0, iree_memory_order_release);

  iree_thread_create_params_t thread_params;
  memset(&thread_params, 0, sizeof(thread_params));
  char thread_name[48] = {0};
  snprintf(thread_name, IREE_ARRAYSIZE(thread_name),
           "iree-hal-vulkan-d%uq%u-complete",
           (unsigned)iree_async_axis_device_index(axis),
           (unsigned)iree_async_axis_queue_index(axis));
  thread_params.name = iree_make_cstring_view(thread_name);
  iree_status_t status = iree_thread_create(
      iree_hal_vulkan_queue_completion_thread_main, queue, thread_params,
      queue->host_allocator, &queue->completion_thread);
  if (!iree_status_is_ok(status)) {
    iree_async_frontier_tracker_retire_axis(
        frontier_tracker, axis, iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }
  return status;
}

void iree_hal_vulkan_queue_retire_frontier(iree_hal_vulkan_queue_t* queue) {
  if (!queue->frontier_tracker) return;
  iree_atomic_store(&queue->stop_requested, 1, iree_memory_order_release);
  iree_status_t wake_status = iree_hal_vulkan_queue_signal_wakeup(queue);
  IREE_ASSERT(iree_status_is_ok(wake_status) ||
                  iree_status_code(wake_status) == IREE_STATUS_DATA_LOSS,
              "Vulkan queue completion wakeup failed during teardown");
  iree_status_free(wake_status);
  if (queue->completion_thread) {
    iree_thread_release(queue->completion_thread);
    queue->completion_thread = NULL;
  }
  iree_async_frontier_tracker_retire_axis(
      queue->frontier_tracker, queue->axis,
      iree_status_from_code(IREE_STATUS_CANCELLED));
  queue->frontier_tracker = NULL;
  queue->axis = 0;
}

static iree_status_t iree_hal_vulkan_queue_submit_captured_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission) {
  iree_hal_vulkan_queue_pending_submission_t* deferred_submission = NULL;
  iree_hal_vulkan_queue_pending_submission_t* memory_wait_submission = NULL;
  iree_hal_vulkan_queue_submission_result_t submit_result = {
      .submitted = false,
      .queue_failure_status = iree_ok_status(),
  };

  VkSemaphoreSubmitInfo* wait_infos = NULL;
  uint32_t wait_info_capacity = 0;
  iree_status_t status = iree_hal_vulkan_queue_allocate_wait_infos(
      queue, submission->wait_semaphore_list.count, &wait_infos,
      &wait_info_capacity);
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&queue->submission_mutex);
    iree_hal_vulkan_queue_wait_resolution_t resolution = {
        .wait_infos = wait_infos,
        .wait_info_capacity = wait_info_capacity,
    };
    status = iree_hal_vulkan_queue_submit_native_under_lock(
        queue, submission, /*allow_software_deferral=*/true, &resolution,
        &submit_result);
    if (iree_status_is_ok(status) && resolution.needs_deferral) {
      status =
          iree_hal_vulkan_queue_prepare_deferred_submission(queue, submission);
      if (iree_status_is_ok(status)) {
        iree_hal_vulkan_queue_append_deferred_submission(queue, submission);
        deferred_submission = submission;
        submission = NULL;
      }
    } else if (submit_result.memory_wait_submission) {
      memory_wait_submission = submit_result.memory_wait_submission;
      submission = NULL;
    } else if (submit_result.submitted) {
      submission = NULL;
    }
    iree_slim_mutex_unlock(&queue->submission_mutex);
  }

  if (wait_infos) {
    iree_allocator_free(queue->host_allocator, wait_infos);
  }
  if (!iree_status_is_ok(submit_result.queue_failure_status)) {
    iree_hal_vulkan_queue_fail_pending_submissions(
        queue, submit_result.queue_failure_status);
  }
  if (deferred_submission) {
    iree_status_t deferred_status =
        iree_hal_vulkan_queue_start_deferred_submission(deferred_submission);
    if (!iree_status_is_ok(deferred_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(queue, deferred_status);
    }
  }
  if (memory_wait_submission) {
    iree_status_t memory_wait_status =
        iree_hal_vulkan_queue_start_alloca_memory_wait(memory_wait_submission);
    if (!iree_status_is_ok(memory_wait_status)) {
      iree_hal_vulkan_queue_fail_pending_submissions(queue, memory_wait_status);
    }
  }
  if (submission) {
    if (iree_status_is_ok(status)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "Vulkan captured submission was neither submitted nor deferred");
    }
    iree_hal_vulkan_queue_fail_unsubmitted_submission(
        queue, submission, iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_barrier(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_vulkan_queue_validate_semaphore_list(
      queue, wait_semaphore_list, IREE_SV("wait"));
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_BARRIER,
        (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_create_transient_buffer(
    iree_hal_vulkan_queue_t* queue, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_alloca_flags_t flags, iree_hal_buffer_t** out_buffer) {
  iree_hal_buffer_placement_t placement = {
      .device = (iree_hal_device_t*)queue->device,
      .queue_affinity = queue->queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  if (iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    placement.flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME;
  }
  return iree_hal_local_transient_buffer_create(
      placement, params, allocation_size, byte_length, queue->host_allocator,
      out_buffer);
}

iree_status_t iree_hal_vulkan_queue_submit_alloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_device_size_t byte_length,
    iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_alloca_flags_t known_flags =
      IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME |
      IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue alloca flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) && allocation_size == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_alloca size must be non-zero");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_create_transient_buffer(
        queue, params, allocation_size, byte_length, flags, &buffer);
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_ALLOCA, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->alloca.buffer = buffer;
    iree_hal_buffer_retain(buffer);
    submission->alloca.pool = pool;
    submission->alloca.params = params;
    submission->alloca.allocation_size = allocation_size;
    submission->alloca.flags = flags;
    submission->alloca.reserve_flags =
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER;
    submission->alloca.memory_wait_kind =
        IREE_HAL_VULKAN_QUEUE_ALLOCA_MEMORY_WAIT_NONE;
    if (wait_semaphore_list.count > 0) {
      status =
          iree_hal_vulkan_queue_try_stage_alloca_backing_now(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
    buffer = NULL;
  }
  iree_hal_buffer_release(buffer);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_dealloca(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_dealloca_flags_t known_flags =
      IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue dealloca flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_local_transient_buffer_isa(buffer)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue_dealloca buffer was not returned "
                              "by Vulkan queue_alloca");
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_local_transient_buffer_begin_dealloca(buffer)) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan transient buffer has already been queued for deallocation");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_DEALLOCA,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->dealloca.buffer = buffer;
    iree_hal_buffer_retain(buffer);
    submission->dealloca.flags = flags;
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    iree_hal_local_transient_buffer_abort_dealloca(buffer);
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  } else if (!iree_status_is_ok(status) &&
             iree_hal_local_transient_buffer_isa(buffer)) {
    iree_hal_local_transient_buffer_abort_dealloca(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_fill(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue fill flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status) && !pattern) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Vulkan queue fill pattern is NULL");
  }
  if (iree_status_is_ok(status) && pattern_length != 1 && pattern_length != 2 &&
      pattern_length != 4) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan queue fill pattern length must be 1, 2, or 4 bytes "
        "(got %" PRIhsz ")",
        pattern_length);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_FILL, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->fill.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->fill.target_offset = target_offset;
    submission->fill.length = length;
    memset(submission->fill.pattern, 0, sizeof(submission->fill.pattern));
    memcpy(submission->fill.pattern, pattern, pattern_length);
    submission->fill.pattern_length = pattern_length;
    submission->fill.flags = flags;
  }
  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_queue_can_fill_native(target_offset, length,
                                            pattern_length) &&
      iree_hal_vulkan_queue_buffer_has_recordable_backing(target_buffer)) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_fill_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_update(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue update flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status) && length > IREE_HOST_SIZE_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan queue update length exceeds host size");
  }
  if (iree_status_is_ok(status) &&
      source_offset > IREE_HOST_SIZE_MAX - (iree_host_size_t)length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan queue update source range exceeds host "
                              "size");
  }
  if (iree_status_is_ok(status) && length > 0 && !source_buffer) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue update source buffer must be non-null");
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_UPDATE, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->update.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->update.target_offset = target_offset;
    submission->update.length = length;
    submission->update.flags = flags;
    if (length > 0) {
      status =
          iree_allocator_malloc(queue->host_allocator, (iree_host_size_t)length,
                                &submission->update.source_data);
      if (iree_status_is_ok(status)) {
        memcpy(submission->update.source_data,
               (const uint8_t*)source_buffer + source_offset,
               (iree_host_size_t)length);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      iree_hal_vulkan_queue_can_update_native(target_offset, length) &&
      iree_hal_vulkan_queue_buffer_has_recordable_backing(target_buffer)) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_update_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_copy(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue copy flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_COPY, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->copy.source_buffer = source_buffer;
    iree_hal_buffer_retain(source_buffer);
    submission->copy.source_offset = source_offset;
    submission->copy.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->copy.target_offset = target_offset;
    submission->copy.length = length;
    submission->copy.flags = flags;
  }
  if (iree_status_is_ok(status) && length != 0) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_copy_native(queue, submission);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_allocate_write_staging_buffer(
    iree_hal_vulkan_queue_t* queue, iree_device_size_t length,
    iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;
  iree_hal_buffer_params_t params = {
      .access =
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
      .queue_affinity = queue->queue_affinity,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST |
              IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
               IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM,
  };
  return iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator((iree_hal_device_t*)queue->device), params,
      length, out_buffer);
}

static iree_status_t iree_hal_vulkan_queue_validate_file_range(
    iree_hal_file_t* file, uint64_t offset, iree_device_size_t length,
    iree_string_view_t operation) {
  const uint64_t file_length = iree_hal_file_length(file);
  if (file_length == 0) return iree_ok_status();
  if (length > UINT64_MAX - offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%.*s range [%" PRIu64 ", overflow) exceeds file length %" PRIu64,
        (int)operation.size, operation.data, offset, file_length);
  }
  const uint64_t end_offset = offset + (uint64_t)length;
  if (end_offset > file_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "%.*s range [%" PRIu64 ", %" PRIu64 ") exceeds file length %" PRIu64,
        (int)operation.size, operation.data, offset, end_offset, file_length);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_queue_submit_read(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_file);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_READ_FLAG_NONE) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported Vulkan queue read flags: 0x%" PRIx64,
                              flags);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(target_buffer, target_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_file_range(
        source_file, source_offset, length, IREE_SV("read"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  iree_hal_buffer_t* source_storage_buffer =
      iree_status_is_ok(status) ? iree_hal_file_storage_buffer(source_file)
                                : NULL;
  bool source_storage_is_native = false;
  if (iree_status_is_ok(status) && source_storage_buffer) {
    status = iree_hal_vulkan_queue_buffer_is_native(source_storage_buffer,
                                                    &source_storage_is_native);
  }
  bool target_is_native = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_buffer_is_native(target_buffer,
                                                    &target_is_native);
  }
  if (iree_status_is_ok(status) && source_storage_is_native &&
      target_is_native) {
    if (source_offset > IREE_DEVICE_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "read source offset %" PRIu64 " exceeds device size", source_offset);
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_hal_vulkan_queue_submit_copy(
          queue, wait_semaphore_list, signal_semaphore_list,
          source_storage_buffer, (iree_device_size_t)source_offset,
          target_buffer, target_offset, length, IREE_HAL_COPY_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_file_supports_synchronous_io(source_file)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan queue read without a direct storage-buffer copy requires "
        "synchronous file I/O until proactor staging is implemented");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
        iree_hal_buffer_memory_type(target_buffer),
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_usage(
        iree_hal_buffer_allowed_usage(target_buffer),
        IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED);
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_READ, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->read.source_file = source_file;
    iree_hal_file_retain(source_file);
    submission->read.source_offset = source_offset;
    submission->read.target_buffer = target_buffer;
    iree_hal_buffer_retain(target_buffer);
    submission->read.target_offset = target_offset;
    submission->read.length = length;
    submission->read.flags = flags;
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_write(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_file);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (flags != IREE_HAL_WRITE_FLAG_NONE) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue write flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_validate_access(target_file,
                                           IREE_HAL_MEMORY_ACCESS_WRITE);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_validate_range(source_buffer, source_offset, length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_file_range(
        target_file, target_offset, length, IREE_SV("write"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && length == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_hal_vulkan_queue_submit_barrier(queue, wait_semaphore_list,
                                                signal_semaphore_list);
  }

  bool source_is_native = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_buffer_is_native(source_buffer,
                                                    &source_is_native);
  }

  iree_hal_buffer_t* target_storage_buffer =
      iree_status_is_ok(status) ? iree_hal_file_storage_buffer(target_file)
                                : NULL;
  bool target_storage_is_native = false;
  if (iree_status_is_ok(status) && target_storage_buffer) {
    status = iree_hal_vulkan_queue_buffer_is_native(target_storage_buffer,
                                                    &target_storage_is_native);
  }
  if (iree_status_is_ok(status) && source_is_native &&
      target_storage_is_native) {
    if (target_offset > IREE_DEVICE_SIZE_MAX) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "write target offset %" PRIu64 " exceeds device size", target_offset);
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_hal_vulkan_queue_submit_copy(
          queue, wait_semaphore_list, signal_semaphore_list, source_buffer,
          source_offset, target_storage_buffer,
          (iree_device_size_t)target_offset, length, IREE_HAL_COPY_FLAG_NONE);
    }
  }
  if (iree_status_is_ok(status) &&
      !iree_hal_file_supports_synchronous_io(target_file)) {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan queue write without a direct storage-buffer copy requires "
        "synchronous file I/O until proactor staging is implemented");
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_WRITE, (iree_hal_host_call_t){0},
        /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE, &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->write.source_buffer = source_buffer;
    iree_hal_buffer_retain(source_buffer);
    submission->write.source_offset = source_offset;
    submission->write.target_file = target_file;
    iree_hal_file_retain(target_file);
    submission->write.target_offset = target_offset;
    submission->write.length = length;
    submission->write.flags = flags;
  }
  if (iree_status_is_ok(status) && source_is_native) {
    status = iree_hal_vulkan_queue_allocate_write_staging_buffer(
        queue, length, &submission->write.staging_buffer);
  }
  if (iree_status_is_ok(status) && submission->write.staging_buffer) {
    status = iree_hal_vulkan_queue_allocate_native_command_buffer(
        queue, &submission->native_command_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_queue_record_copy_native_buffers(
          queue, source_buffer, source_offset, submission->write.staging_buffer,
          /*target_offset=*/0, length, submission->native_command_buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_vulkan_queue_prepare_native_execute_submission(
    iree_hal_vulkan_queue_t* queue,
    iree_hal_vulkan_queue_pending_submission_t* submission,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!iree_hal_vulkan_command_buffer_has_native_commands(
          submission->execute.command_buffer)) {
    return iree_ok_status();
  }
  if (iree_hal_vulkan_command_buffer_has_host_commands(
          submission->execute.command_buffer)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "mixed host-replayed and native Vulkan command "
                            "buffers are unsupported");
  }
  iree_status_t status = iree_hal_vulkan_queue_allocate_native_command_buffer(
      queue, &submission->native_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_command_buffer_record_native(
        submission->execute.command_buffer, &queue->syms, queue->logical_device,
        submission->native_command_buffer, binding_table, queue->host_allocator,
        &submission->execute.native_descriptor_pool);
  }
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_execute(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_execute_flags_t known_flags =
      IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(flags, ~known_flags)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan queue execute flags: 0x%" PRIx64, flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) &&
      binding_table.count < command_buffer->binding_count) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect Vulkan command buffer requires at least %u bindings but only "
        "%" PRIhsz " were provided",
        command_buffer->binding_count, binding_table.count);
  }
  if (iree_status_is_ok(status) && command_buffer->binding_count != 0 &&
      !binding_table.bindings) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect Vulkan command buffer binding table storage is NULL for %u "
        "bindings",
        command_buffer->binding_count);
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_EXECUTE,
        (iree_hal_host_call_t){0}, /*args=*/NULL, IREE_HAL_HOST_CALL_FLAG_NONE,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    submission->execute.command_buffer = command_buffer;
    iree_hal_command_buffer_retain(command_buffer);
    submission->execute.flags = flags;
    submission->execute.binding_table_count = command_buffer->binding_count;
    if (command_buffer->binding_count != 0) {
      iree_host_size_t binding_table_size = 0;
      if (!iree_host_size_checked_mul(command_buffer->binding_count,
                                      sizeof(*binding_table.bindings),
                                      &binding_table_size)) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "Vulkan queue execute binding table capture is too large");
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc(
            queue->host_allocator, binding_table_size,
            (void**)&submission->execute.binding_table_bindings);
      }
      if (iree_status_is_ok(status)) {
        memcpy(submission->execute.binding_table_bindings,
               binding_table.bindings, binding_table_size);
        if (!iree_any_bit_set(
                flags, IREE_HAL_EXECUTE_FLAG_BORROW_BINDING_TABLE_LIFETIME)) {
          for (iree_host_size_t i = 0; i < command_buffer->binding_count; ++i) {
            iree_hal_buffer_retain(
                submission->execute.binding_table_bindings[i].buffer);
          }
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t captured_binding_table = {
        .count = submission->execute.binding_table_count,
        .bindings = submission->execute.binding_table_bindings,
    };
    status = iree_hal_vulkan_queue_prepare_native_execute_submission(
        queue, submission, captured_binding_table);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
    submission = NULL;
  }
  if (submission) {
    if (!iree_status_is_ok(status)) {
      iree_hal_vulkan_queue_fail_signal_list(submission->signal_semaphore_list,
                                             iree_status_clone(status));
    }
    iree_hal_vulkan_queue_pending_submission_destroy(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_queue_submit_host_call(
    iree_hal_vulkan_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_hal_vulkan_queue_validate_host_call(call, args, flags);
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, wait_semaphore_list, IREE_SV("wait"));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_validate_semaphore_list(
        queue, signal_semaphore_list, IREE_SV("signal"));
  }
  if (iree_status_is_ok(status) && wait_semaphore_list.count > UINT32_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "too many Vulkan queue wait semaphores");
  }

  iree_hal_vulkan_queue_pending_submission_t* submission = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_pending_submission_create(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_VULKAN_QUEUE_SUBMISSION_KIND_HOST_CALL, call, args, flags,
        &submission);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_vulkan_queue_submit_captured_submission(queue, submission);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_host_size_t iree_hal_vulkan_queue_drain_completions(
    iree_hal_vulkan_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t failure_status = (iree_status_t)iree_atomic_load(
      &queue->failure_status, iree_memory_order_acquire);
  if (!iree_status_is_ok(failure_status)) {
    iree_hal_vulkan_queue_fail_pending_submissions(
        queue, iree_status_clone(failure_status));
    IREE_TRACE_ZONE_END(z0);
    return 0;
  }

  uint64_t completed_epoch = 0;
  VkResult result = iree_vkGetSemaphoreCounterValue_raw(
      &queue->syms, queue->logical_device, queue->epoch_semaphore,
      &completed_epoch);
  if (result != VK_SUCCESS) {
    iree_status_t status = iree_status_from_vk_result(
        __FILE__, __LINE__, result, "vkGetSemaphoreCounterValue");
    iree_status_t stored_status =
        iree_hal_vulkan_queue_store_error(queue, status);
    iree_hal_vulkan_queue_fail_pending_submissions(queue, stored_status);
    IREE_TRACE_ZONE_END(z0);
    return 0;
  }

  iree_hal_vulkan_queue_pending_submission_t* completed_head = NULL;
  iree_slim_mutex_lock(&queue->submission_mutex);
  completed_head =
      iree_hal_vulkan_queue_pop_completed_submissions(queue, completed_epoch);
  iree_slim_mutex_unlock(&queue->submission_mutex);

  iree_host_size_t completed_count = 0;
  while (completed_head) {
    iree_hal_vulkan_queue_pending_submission_t* next = completed_head->next;
    completed_head->next = NULL;
    iree_hal_vulkan_queue_complete_submission(queue, completed_head,
                                              iree_ok_status());
    iree_hal_vulkan_queue_pending_submission_destroy(queue, completed_head);
    completed_head = next;
    ++completed_count;
  }

  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)completed_count);
  IREE_TRACE_ZONE_END(z0);
  return completed_count;
}
