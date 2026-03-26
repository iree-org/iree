// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

// Proactor progress callback. Polls the epoch signal and drains completed
// notification entries on each proactor iteration.
static iree_host_size_t iree_hal_amdgpu_host_queue_progress_fn(
    void* user_data) {
  return iree_hal_amdgpu_host_queue_drain(
      (iree_hal_amdgpu_host_queue_t*)user_data);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    iree_hal_amdgpu_logical_device_t* device, iree_async_proactor_t* proactor,
    const iree_hal_amdgpu_libhsa_t* libhsa, uint32_t notification_capacity,
    iree_allocator_t host_allocator, iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(out_queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate before touching out_queue.
  if (!iree_host_size_is_power_of_two(notification_capacity)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "notification_capacity must be a power of two");
  }

  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  out_queue->device = device;
  out_queue->proactor = proactor;
  out_queue->libhsa = libhsa;
  out_queue->host_allocator = host_allocator;

  // Create the epoch signal with full interrupt capability (mailbox event,
  // eventfd bridge for proactor integration).
  hsa_signal_t epoch_signal = {0};
  iree_status_t status = iree_hsa_amd_signal_create(
      IREE_LIBHSA(libhsa), IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE,
      /*num_consumers=*/0, /*consumers=*/NULL, /*attributes=*/0, &epoch_signal);

  iree_hal_amdgpu_notification_entry_t* notification_ring = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(
        host_allocator, notification_capacity,
        sizeof(iree_hal_amdgpu_notification_entry_t),
        (void**)&notification_ring);
  }

  if (iree_status_is_ok(status)) {
    out_queue->epoch.signal = epoch_signal;
    out_queue->epoch.next_submission = 0;
    out_queue->epoch.last_drained = 0;
    out_queue->notification.ring = notification_ring;
    out_queue->notification.write = 0;
    out_queue->notification.read = 0;
    out_queue->notification.capacity = notification_capacity;

    memset(&out_queue->progress_entry, 0, sizeof(out_queue->progress_entry));
    out_queue->progress_entry.fn = iree_hal_amdgpu_host_queue_progress_fn;
    out_queue->progress_entry.user_data = out_queue;
    iree_async_proactor_register_progress(proactor, &out_queue->progress_entry);
  } else {
    iree_allocator_free(host_allocator, notification_ring);
    if (epoch_signal.handle) {
      IREE_IGNORE_ERROR(
          iree_hsa_signal_destroy(IREE_LIBHSA(libhsa), epoch_signal));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_unregister_progress(queue->proactor,
                                          &queue->progress_entry);

  // Drain any remaining entries (should be empty if the caller waited for
  // all in-flight work to complete).
  iree_hal_amdgpu_host_queue_drain(queue);

  if (queue->epoch.signal.handle) {
    IREE_IGNORE_ERROR(iree_hsa_signal_destroy(IREE_LIBHSA(queue->libhsa),
                                              queue->epoch.signal));
    queue->epoch.signal.handle = 0;
  }

  iree_allocator_free(queue->host_allocator, queue->notification.ring);
  queue->notification.ring = NULL;

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Notification ring operations
//===----------------------------------------------------------------------===//

void iree_hal_amdgpu_host_queue_push_notification(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t submission_epoch,
    iree_async_semaphore_t* semaphore, uint64_t timeline_value,
    const iree_async_frontier_t* frontier) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT(queue->notification.write - queue->notification.read <
                  queue->notification.capacity,
              "notification ring overflow");

  uint32_t index = (uint32_t)(queue->notification.write &
                              (queue->notification.capacity - 1));
  iree_hal_amdgpu_notification_entry_t* entry =
      &queue->notification.ring[index];

  entry->semaphore = semaphore;
  entry->timeline_value = timeline_value;
  entry->submission_epoch = submission_epoch;

  if (frontier && frontier->entry_count > 0) {
    entry->frontier.entry_count = frontier->entry_count;
    memset(entry->frontier.reserved, 0, sizeof(entry->frontier.reserved));
    memcpy(entry->frontier_entries, frontier->entries,
           frontier->entry_count * sizeof(iree_async_frontier_entry_t));
  } else {
    entry->frontier.entry_count = 0;
    memset(entry->frontier.reserved, 0, sizeof(entry->frontier.reserved));
  }

  ++queue->notification.write;
}

iree_host_size_t iree_hal_amdgpu_host_queue_drain(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);

  hsa_signal_value_t signal_value = iree_hsa_signal_load_scacquire(
      IREE_LIBHSA(queue->libhsa), queue->epoch.signal);
  uint64_t current_epoch =
      (uint64_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE - signal_value);

  if (current_epoch <= queue->epoch.last_drained) return 0;
  queue->epoch.last_drained = current_epoch;

  iree_host_size_t drained_count = 0;
  while (queue->notification.read < queue->notification.write) {
    uint32_t index = (uint32_t)(queue->notification.read &
                                (queue->notification.capacity - 1));
    iree_hal_amdgpu_notification_entry_t* entry =
        &queue->notification.ring[index];
    if (entry->submission_epoch >= current_epoch) break;

    const iree_async_frontier_t* entry_frontier =
        entry->frontier.entry_count > 0 ? &entry->frontier : NULL;
    iree_async_semaphore_signal(entry->semaphore, entry->timeline_value,
                                entry_frontier);

    ++queue->notification.read;
    ++drained_count;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Virtual queue operation stubs (implemented by submission path)
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue alloca not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue dealloca not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue fill not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue update not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue copy not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue host_call not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue dispatch not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue execute not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Virtual queue vtable
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_deinitialize_vtable(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  iree_hal_amdgpu_host_queue_deinitialize(
      (iree_hal_amdgpu_host_queue_t*)base_queue);
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize_vtable,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .alloca = iree_hal_amdgpu_host_queue_alloca,
        .dealloca = iree_hal_amdgpu_host_queue_dealloca,
        .fill = iree_hal_amdgpu_host_queue_fill,
        .update = iree_hal_amdgpu_host_queue_update,
        .copy = iree_hal_amdgpu_host_queue_copy,
        .read = NULL,   // emulation
        .write = NULL,  // emulation
        .host_call = iree_hal_amdgpu_host_queue_host_call,
        .dispatch = iree_hal_amdgpu_host_queue_dispatch,
        .execute = iree_hal_amdgpu_host_queue_execute,
        .flush = iree_hal_amdgpu_host_queue_flush,
};
