// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"

static_assert(sizeof(iree_hal_amdgpu_profile_dispatch_event_t) ==
                  sizeof(iree_hal_profile_dispatch_event_t),
              "AMDGPU dispatch events must convert without layout growth");
static_assert(sizeof(iree_hal_amdgpu_profile_queue_device_event_t) ==
                  sizeof(iree_hal_profile_queue_device_event_t),
              "AMDGPU queue device events must convert without layout growth");
static_assert(
    (uint32_t)IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER ==
        (uint32_t)IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER,
    "AMDGPU command-buffer dispatch event flag must match HAL");
static_assert(
    (uint32_t)IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS ==
        (uint32_t)IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS,
    "AMDGPU indirect dispatch event flag must match HAL");

// Maximum dispatch events buffered per queue between profiling flushes.
#define IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_DISPATCH_EVENT_CAPACITY (64 * 1024)

// Maximum queue device events buffered per queue between profiling flushes.
#define IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_QUEUE_DEVICE_EVENT_CAPACITY \
  (64 * 1024)

static void iree_hal_amdgpu_host_queue_initialize_profiling_signal(
    iree_amd_signal_t* signal) {
  memset(signal, 0, sizeof(*signal));
  signal->kind = IREE_AMD_SIGNAL_KIND_USER;
  // Profiling completion signals are never waited on. Keep the value at
  // all-bits-set so packet completion decrements never require host/device
  // reset traffic; consumers read start_ts/end_ts after queue ordering proves
  // the profiled packet completed.
  signal->value = (iree_hsa_signal_value_t)-1;
}

static iree_status_t
iree_hal_amdgpu_host_queue_allocate_profiling_completion_signals(
    iree_hal_amdgpu_block_pool_t* signal_block_pool, uint32_t signal_count,
    iree_allocator_t host_allocator, iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_count);

  if (IREE_UNLIKELY(signal_block_pool->block_size < sizeof(iree_amd_signal_t) ||
                    signal_block_pool->block_size % sizeof(iree_amd_signal_t) !=
                        0)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "profiling signal block size %" PRIdsz
                             " must hold whole iree_amd_signal_t records",
                             signal_block_pool->block_size));
  }
  const iree_host_size_t signals_per_block =
      signal_block_pool->block_size / sizeof(iree_amd_signal_t);
  if (IREE_UNLIKELY(signals_per_block == 0 || signals_per_block > UINT32_MAX)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "profiling signal block size %" PRIu64
                             " cannot hold a valid signal count",
                             (uint64_t)signal_block_pool->block_size));
  }
  const iree_host_size_t signal_block_count =
      iree_host_size_ceil_div(signal_count, signals_per_block);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, signal_block_count);

  iree_host_size_t signal_block_table_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      IREE_STRUCT_LAYOUT(0, &signal_block_table_size,
                         IREE_STRUCT_FIELD(signal_block_count,
                                           iree_hal_amdgpu_block_t*, NULL)));
  iree_hal_amdgpu_block_t** signal_blocks = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, signal_block_table_size, (void**)&signal_blocks);
  iree_host_size_t acquired_block_count = 0;
  for (iree_host_size_t block_index = 0;
       block_index < signal_block_count && iree_status_is_ok(status);
       ++block_index) {
    iree_hal_amdgpu_block_t* block = NULL;
    status = iree_hal_amdgpu_block_pool_acquire(signal_block_pool, &block);
    if (iree_status_is_ok(status)) {
      signal_blocks[block_index] = block;
      ++acquired_block_count;
      if (IREE_UNLIKELY(
              (uintptr_t)block->ptr % iree_alignof(iree_amd_signal_t) != 0)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "profiling signal block is not aligned to %" PRIhsz " bytes",
            (iree_host_size_t)iree_alignof(iree_amd_signal_t));
      } else {
        for (iree_host_size_t signal_index = 0;
             signal_index < signals_per_block; ++signal_index) {
          uint8_t* signal_ptr =
              (uint8_t*)block->ptr + signal_index * sizeof(iree_amd_signal_t);
          iree_hal_amdgpu_host_queue_initialize_profiling_signal(
              (iree_amd_signal_t*)signal_ptr);
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    out_queue->profiling.signals.block_pool = signal_block_pool;
    out_queue->profiling.signals.blocks = signal_blocks;
    out_queue->profiling.signals.block_count = (uint32_t)signal_block_count;
    out_queue->profiling.signals.signals_per_block =
        (uint32_t)signals_per_block;
  } else {
    for (iree_host_size_t i = 0; i < acquired_block_count; ++i) {
      iree_hal_amdgpu_block_pool_release(signal_block_pool, signal_blocks[i]);
    }
    iree_allocator_free(host_allocator, signal_blocks);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_ensure_profiling_completion_signals(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->profiling.signals.blocks) return iree_ok_status();
  return iree_hal_amdgpu_host_queue_allocate_profiling_completion_signals(
      queue->profiling.signals.block_pool,
      queue->profiling.dispatch_events.capacity, queue->host_allocator, queue);
}

void iree_hal_amdgpu_host_queue_deallocate_profiling_completion_signals(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->profiling.signals.blocks) {
    return;
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  for (uint32_t i = 0; i < queue->profiling.signals.block_count; ++i) {
    iree_hal_amdgpu_block_pool_release(queue->profiling.signals.block_pool,
                                       queue->profiling.signals.blocks[i]);
  }
  iree_allocator_free(queue->host_allocator, queue->profiling.signals.blocks);
  queue->profiling.signals.block_pool = NULL;
  queue->profiling.signals.blocks = NULL;
  queue->profiling.signals.block_count = 0;
  queue->profiling.signals.signals_per_block = 0;

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_queue_ensure_profile_event_storage(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->profiling.event_allocation.base) return iree_ok_status();

  IREE_TRACE_ZONE_BEGIN(z0);
  const uint32_t dispatch_event_capacity =
      IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_DISPATCH_EVENT_CAPACITY;
  const uint32_t queue_device_event_capacity =
      IREE_HAL_AMDGPU_HOST_QUEUE_PROFILE_QUEUE_DEVICE_EVENT_CAPACITY;
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dispatch_event_capacity);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, queue_device_event_capacity);

  iree_host_size_t dispatch_events_offset = 0;
  iree_host_size_t queue_device_events_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              0, &total_size,
              IREE_STRUCT_FIELD(dispatch_event_capacity,
                                iree_hal_amdgpu_profile_dispatch_event_t,
                                &dispatch_events_offset),
              IREE_STRUCT_FIELD(queue_device_event_capacity,
                                iree_hal_amdgpu_profile_queue_device_event_t,
                                &queue_device_events_offset)));
  void* event_storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hsa_amd_memory_pool_allocate(
          IREE_LIBHSA(queue->libhsa),
          queue->profiling.signals.block_pool->memory_pool, total_size,
          HSA_AMD_MEMORY_POOL_STANDARD_FLAG, &event_storage),
      "allocating profile event rings of %" PRIhsz " bytes", total_size);
  memset(event_storage, 0, total_size);

  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.event_allocation.base = event_storage;
  queue->profiling.event_allocation.size = total_size;
  queue->profiling.dispatch_events.values =
      (iree_hal_amdgpu_profile_dispatch_event_t*)((uint8_t*)event_storage +
                                                  dispatch_events_offset);
  queue->profiling.dispatch_events.capacity = dispatch_event_capacity;
  queue->profiling.dispatch_events.mask = dispatch_event_capacity - 1;
  queue->profiling.dispatch_events.read_position = 0;
  queue->profiling.dispatch_events.ready_position = 0;
  queue->profiling.dispatch_events.write_position = 0;
  queue->profiling.dispatch_events.next_event_id = 1;
  queue->profiling.queue_device_events.values =
      (iree_hal_amdgpu_profile_queue_device_event_t*)((uint8_t*)event_storage +
                                                      queue_device_events_offset);
  queue->profiling.queue_device_events.capacity = queue_device_event_capacity;
  queue->profiling.queue_device_events.mask = queue_device_event_capacity - 1;
  queue->profiling.queue_device_events.read_position = 0;
  queue->profiling.queue_device_events.ready_position = 0;
  queue->profiling.queue_device_events.write_position = 0;
  queue->profiling.queue_device_events.next_event_id = 1;
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_clear_profile_events(
    iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.dispatch_events.read_position = 0;
  queue->profiling.dispatch_events.ready_position = 0;
  queue->profiling.dispatch_events.write_position = 0;
  queue->profiling.dispatch_events.next_event_id = 1;
  queue->profiling.queue_device_events.read_position = 0;
  queue->profiling.queue_device_events.ready_position = 0;
  queue->profiling.queue_device_events.write_position = 0;
  queue->profiling.queue_device_events.next_event_id = 1;
  if (queue->profiling.event_allocation.base) {
    memset(queue->profiling.event_allocation.base, 0,
           queue->profiling.event_allocation.size);
  }
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
}

void iree_hal_amdgpu_host_queue_deallocate_profile_events(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (!queue->profiling.event_allocation.base) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdgpu_hsa_cleanup_assert_success(iree_hsa_amd_memory_pool_free_raw(
      queue->libhsa, queue->profiling.event_allocation.base));
  queue->profiling.event_allocation.base = NULL;
  queue->profiling.event_allocation.size = 0;
  queue->profiling.dispatch_events.values = NULL;
  queue->profiling.dispatch_events.capacity = 0;
  queue->profiling.dispatch_events.mask = 0;
  queue->profiling.dispatch_events.read_position = 0;
  queue->profiling.dispatch_events.ready_position = 0;
  queue->profiling.dispatch_events.write_position = 0;
  queue->profiling.dispatch_events.next_event_id = 0;
  queue->profiling.queue_device_events.values = NULL;
  queue->profiling.queue_device_events.capacity = 0;
  queue->profiling.queue_device_events.mask = 0;
  queue->profiling.queue_device_events.read_position = 0;
  queue->profiling.queue_device_events.ready_position = 0;
  queue->profiling.queue_device_events.write_position = 0;
  queue->profiling.queue_device_events.next_event_id = 0;
  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t event_count,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t* out_reservation) {
  *out_reservation = (iree_hal_amdgpu_profile_dispatch_event_reservation_t){0};
  if (event_count == 0 || !queue->profiling.hsa_queue_timestamps_enabled ||
      !queue->profiling.dispatch_events.values) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(event_count > queue->profiling.dispatch_events.capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dispatch profiling reservation of %" PRIu32
                            " events exceeds queue capacity %" PRIu32,
                            event_count,
                            queue->profiling.dispatch_events.capacity);
  }

  bool is_exhausted = false;
  uint64_t exhausted_available_count = 0;
  uint64_t exhausted_ready_count = 0;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  const uint64_t read_position = queue->profiling.dispatch_events.read_position;
  const uint64_t ready_position =
      queue->profiling.dispatch_events.ready_position;
  const uint64_t write_position =
      queue->profiling.dispatch_events.write_position;
  const uint64_t occupied_count = write_position - read_position;
  const uint64_t available_count =
      queue->profiling.dispatch_events.capacity - occupied_count;
  if (event_count <= available_count) {
    out_reservation->first_event_position = write_position;
    out_reservation->event_count = event_count;
    queue->profiling.dispatch_events.write_position =
        write_position + event_count;
    for (uint32_t i = 0; i < event_count; ++i) {
      iree_hal_amdgpu_profile_dispatch_event_t* event =
          iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
              queue, write_position + i);
      memset(event, 0, sizeof(*event));
      event->record_length = sizeof(*event);
      event->event_id = queue->profiling.dispatch_events.next_event_id++;
      event->command_index = UINT32_MAX;
      event->export_ordinal = UINT32_MAX;
    }
  } else {
    is_exhausted = true;
    exhausted_available_count = available_count;
    exhausted_ready_count = ready_position - read_position;
  }
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
  if (IREE_UNLIKELY(is_exhausted)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "dispatch profiling event ring exhausted: requested %" PRIu32
        " events, available %" PRIu64 ", ready %" PRIu64 ", capacity %" PRIu32,
        event_count, exhausted_available_count, exhausted_ready_count,
        queue->profiling.dispatch_events.capacity);
  }
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count) return;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.dispatch_events.write_position =
      reservation.first_event_position;
  queue->profiling.dispatch_events.next_event_id -= reservation.event_count;
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
}

iree_hal_amdgpu_profile_dispatch_event_t*
iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  const uint32_t event_index =
      (uint32_t)(event_position & queue->profiling.dispatch_events.mask);
  return &queue->profiling.dispatch_events.values[event_index];
}

void iree_hal_amdgpu_host_queue_retire_profile_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t reservation) {
  if (!reservation.event_count) return;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.dispatch_events.ready_position =
      reservation.first_event_position + reservation.event_count;
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
}

bool iree_hal_amdgpu_host_queue_should_profile_queue_device_events(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return queue->profiling.queue_device_events_enabled &&
         queue->profiling.queue_device_events.values;
}

iree_status_t iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t event_count,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t* out_reservation) {
  *out_reservation =
      (iree_hal_amdgpu_profile_queue_device_event_reservation_t){0};
  if (event_count == 0 ||
      !iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!queue->pm4_ib_slots)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue device profiling requires queue-local PM4 IB slots");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_pm4_timestamp_strategy_supports_ranges(
          queue->pm4_timestamp_strategy))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "queue device profiling requires PM4 timestamp range support");
  }
  if (IREE_UNLIKELY(event_count >
                    queue->profiling.queue_device_events.capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "queue device profiling reservation of %" PRIu32
                            " events exceeds queue capacity %" PRIu32,
                            event_count,
                            queue->profiling.queue_device_events.capacity);
  }

  bool is_exhausted = false;
  uint64_t exhausted_available_count = 0;
  uint64_t exhausted_ready_count = 0;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  const uint64_t read_position =
      queue->profiling.queue_device_events.read_position;
  const uint64_t ready_position =
      queue->profiling.queue_device_events.ready_position;
  const uint64_t write_position =
      queue->profiling.queue_device_events.write_position;
  const uint64_t occupied_count = write_position - read_position;
  const uint64_t available_count =
      queue->profiling.queue_device_events.capacity - occupied_count;
  if (event_count <= available_count) {
    out_reservation->first_event_position = write_position;
    out_reservation->event_count = event_count;
    queue->profiling.queue_device_events.write_position =
        write_position + event_count;
    for (uint32_t i = 0; i < event_count; ++i) {
      iree_hal_amdgpu_profile_queue_device_event_t* event =
          iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
              queue, write_position + i);
      memset(event, 0, sizeof(*event));
      event->record_length = sizeof(*event);
      event->event_id = queue->profiling.queue_device_events.next_event_id++;
      event->physical_device_ordinal =
          iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
      event->queue_ordinal =
          iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
      event->stream_id = iree_hal_amdgpu_host_queue_profile_stream_id(queue);
    }
  } else {
    is_exhausted = true;
    exhausted_available_count = available_count;
    exhausted_ready_count = ready_position - read_position;
  }
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
  if (IREE_UNLIKELY(is_exhausted)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "queue device profiling event ring exhausted: requested %" PRIu32
        " events, available %" PRIu64 ", ready %" PRIu64 ", capacity %" PRIu32,
        event_count, exhausted_available_count, exhausted_ready_count,
        queue->profiling.queue_device_events.capacity);
  }
  return iree_ok_status();
}

void iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation) {
  if (!reservation.event_count) return;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.queue_device_events.write_position =
      reservation.first_event_position;
  queue->profiling.queue_device_events.next_event_id -= reservation.event_count;
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
}

iree_hal_amdgpu_profile_queue_device_event_t*
iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
    const iree_hal_amdgpu_host_queue_t* queue, uint64_t event_position) {
  const uint32_t event_index =
      (uint32_t)(event_position & queue->profiling.queue_device_events.mask);
  return &queue->profiling.queue_device_events.values[event_index];
}

void iree_hal_amdgpu_host_queue_retire_profile_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_profile_queue_device_event_reservation_t reservation) {
  if (!reservation.event_count) return;
  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  queue->profiling.queue_device_events.ready_position =
      reservation.first_event_position + reservation.event_count;
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);
}

static iree_status_t iree_hal_amdgpu_host_queue_copy_dispatch_events(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t read_position,
    iree_host_size_t event_count, iree_host_size_t* out_storage_size,
    iree_hal_profile_dispatch_event_t** out_events) {
  *out_storage_size = 0;
  *out_events = NULL;
  if (event_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, out_storage_size,
      IREE_STRUCT_FIELD(event_count, iree_hal_profile_dispatch_event_t, NULL)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, *out_storage_size, (void**)out_events));
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    const iree_hal_amdgpu_profile_dispatch_event_t* source =
        iree_hal_amdgpu_host_queue_profile_dispatch_event_at(queue,
                                                             read_position + i);
    iree_hal_profile_dispatch_event_t* target = &(*out_events)[i];
    target->record_length = source->record_length;
    target->flags = source->flags;
    target->event_id = source->event_id;
    target->submission_id = source->submission_id;
    target->command_buffer_id = source->command_buffer_id;
    target->executable_id = source->executable_id;
    target->command_index = source->command_index;
    target->export_ordinal = source->export_ordinal;
    memcpy(target->workgroup_count, source->workgroup_count,
           sizeof(target->workgroup_count));
    memcpy(target->workgroup_size, source->workgroup_size,
           sizeof(target->workgroup_size));
    target->start_tick = source->start_tick;
    target->end_tick = source->end_tick;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_copy_queue_device_events(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t read_position,
    iree_host_size_t event_count, iree_host_size_t* out_storage_size,
    iree_hal_profile_queue_device_event_t** out_events) {
  *out_storage_size = 0;
  *out_events = NULL;
  if (event_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, out_storage_size,
      IREE_STRUCT_FIELD(event_count, iree_hal_profile_queue_device_event_t,
                        NULL)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, *out_storage_size, (void**)out_events));
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    const iree_hal_amdgpu_profile_queue_device_event_t* source =
        iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
            queue, read_position + i);
    memcpy(&(*out_events)[i], source, sizeof((*out_events)[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_build_event_relationships(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_profile_dispatch_event_t* dispatch_events,
    iree_host_size_t dispatch_event_count,
    const iree_hal_profile_queue_device_event_t* queue_device_events,
    iree_host_size_t queue_device_event_count, uint64_t stream_id,
    uint32_t physical_device_ordinal, uint32_t queue_ordinal,
    iree_host_size_t* out_storage_size,
    iree_hal_profile_event_relationship_record_t** out_relationships) {
  *out_storage_size = 0;
  *out_relationships = NULL;

  const iree_host_size_t max_relationship_count =
      dispatch_event_count + queue_device_event_count;
  if (max_relationship_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, out_storage_size,
      IREE_STRUCT_FIELD(max_relationship_count,
                        iree_hal_profile_event_relationship_record_t, NULL)));
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, *out_storage_size, (void**)out_relationships));

  iree_host_size_t relationship_count = 0;
  for (iree_host_size_t i = 0; i < dispatch_event_count; ++i) {
    const iree_hal_profile_dispatch_event_t* event = &dispatch_events[i];
    if (event->submission_id == 0) continue;
    iree_hal_profile_event_relationship_record_t* relationship =
        &(*out_relationships)[relationship_count++];
    *relationship = iree_hal_profile_event_relationship_record_default();
    relationship->type =
        IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_DISPATCH;
    relationship->relationship_id = relationship_count;
    relationship->source_type =
        IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION;
    relationship->target_type =
        IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_DISPATCH_EVENT;
    relationship->physical_device_ordinal = physical_device_ordinal;
    relationship->queue_ordinal = queue_ordinal;
    relationship->stream_id = stream_id;
    relationship->source_id = event->submission_id;
    relationship->target_id = event->event_id;
  }

  for (iree_host_size_t i = 0; i < queue_device_event_count; ++i) {
    const iree_hal_profile_queue_device_event_t* event =
        &queue_device_events[i];
    if (event->submission_id == 0) continue;
    iree_hal_profile_event_relationship_record_t* relationship =
        &(*out_relationships)[relationship_count++];
    *relationship = iree_hal_profile_event_relationship_record_default();
    relationship->type =
        IREE_HAL_PROFILE_EVENT_RELATIONSHIP_TYPE_QUEUE_SUBMISSION_QUEUE_DEVICE_EVENT;
    relationship->relationship_id = relationship_count;
    relationship->source_type =
        IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_SUBMISSION;
    relationship->target_type =
        IREE_HAL_PROFILE_EVENT_ENDPOINT_TYPE_QUEUE_DEVICE_EVENT;
    relationship->physical_device_ordinal = physical_device_ordinal;
    relationship->queue_ordinal = queue_ordinal;
    relationship->stream_id = stream_id;
    relationship->source_id = event->submission_id;
    relationship->target_id = event->event_id;
  }

  *out_storage_size =
      relationship_count * sizeof(iree_hal_profile_event_relationship_record_t);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_write_profile_events(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_profile_sink_t* sink,
    uint64_t session_id) {
  if (!sink) return iree_ok_status();
  if (IREE_UNLIKELY(queue->device_ordinal > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile event physical device ordinal out of "
                            "range: %" PRIhsz,
                            queue->device_ordinal);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_profile_dispatch_event_t* dispatch_events = NULL;
  iree_host_size_t dispatch_event_storage_size = 0;
  iree_hal_profile_queue_device_event_t* queue_device_events = NULL;
  iree_host_size_t queue_device_event_storage_size = 0;
  iree_hal_profile_event_relationship_record_t* relationships = NULL;
  iree_host_size_t relationship_storage_size = 0;
  iree_host_size_t dispatch_event_count = 0;
  iree_host_size_t queue_device_event_count = 0;

  iree_slim_mutex_lock(&queue->profiling.event_mutex);
  const uint64_t dispatch_read_position =
      queue->profiling.dispatch_events.read_position;
  const uint64_t dispatch_ready_position =
      queue->profiling.dispatch_events.ready_position;
  const uint64_t queue_device_read_position =
      queue->profiling.queue_device_events.read_position;
  const uint64_t queue_device_ready_position =
      queue->profiling.queue_device_events.ready_position;
  dispatch_event_count =
      (iree_host_size_t)(dispatch_ready_position - dispatch_read_position);
  queue_device_event_count = (iree_host_size_t)(queue_device_ready_position -
                                                queue_device_read_position);
  iree_status_t status = iree_hal_amdgpu_host_queue_copy_dispatch_events(
      queue, dispatch_read_position, dispatch_event_count,
      &dispatch_event_storage_size, &dispatch_events);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_copy_queue_device_events(
        queue, queue_device_read_position, queue_device_event_count,
        &queue_device_event_storage_size, &queue_device_events);
  }
  iree_slim_mutex_unlock(&queue->profiling.event_mutex);

  const bool has_events =
      dispatch_event_count != 0 || queue_device_event_count != 0;
  if (iree_status_is_ok(status) && has_events) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    const uint32_t physical_device_ordinal = (uint32_t)queue->device_ordinal;
    const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DISPATCH_EVENTS;
    metadata.name = iree_make_cstring_view("amdgpu.dispatch");
    metadata.session_id = session_id;
    metadata.stream_id =
        ((uint64_t)physical_device_ordinal << 32) | (uint64_t)queue_ordinal;
    metadata.physical_device_ordinal = physical_device_ordinal;
    metadata.queue_ordinal = queue_ordinal;

    status = iree_hal_amdgpu_host_queue_build_event_relationships(
        queue, dispatch_events, dispatch_event_count, queue_device_events,
        queue_device_event_count, metadata.stream_id, physical_device_ordinal,
        queue_ordinal, &relationship_storage_size, &relationships);

    if (iree_status_is_ok(status) && dispatch_event_count != 0) {
      iree_const_byte_span_t iovec = iree_make_const_byte_span(
          dispatch_events, dispatch_event_storage_size);
      status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
    }
    if (iree_status_is_ok(status) && queue_device_event_count != 0) {
      metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_DEVICE_EVENTS;
      metadata.name = iree_make_cstring_view("amdgpu.queue_device");
      iree_const_byte_span_t iovec = iree_make_const_byte_span(
          queue_device_events, queue_device_event_storage_size);
      status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
    }
    if (iree_status_is_ok(status) && relationship_storage_size != 0) {
      metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EVENT_RELATIONSHIPS;
      metadata.name = iree_make_cstring_view("amdgpu.relationships");
      iree_const_byte_span_t relationship_iovec =
          iree_make_const_byte_span(relationships, relationship_storage_size);
      status =
          iree_hal_profile_sink_write(sink, &metadata, 1, &relationship_iovec);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_queue_write_profile_counter_samples(
          queue, sink, session_id, dispatch_read_position, dispatch_event_count,
          dispatch_events);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_queue_write_profile_traces(
          queue, sink, session_id, dispatch_read_position, dispatch_event_count,
          dispatch_events);
    }
  }

  if (iree_status_is_ok(status) && has_events) {
    iree_hal_amdgpu_host_queue_release_profile_trace_slots(
        queue, dispatch_read_position, dispatch_event_count);
    iree_slim_mutex_lock(&queue->profiling.event_mutex);
    queue->profiling.dispatch_events.read_position =
        dispatch_read_position + dispatch_event_count;
    queue->profiling.queue_device_events.read_position =
        queue_device_read_position + queue_device_event_count;
    iree_slim_mutex_unlock(&queue->profiling.event_mutex);
  }

  iree_allocator_free(queue->host_allocator, dispatch_events);
  iree_allocator_free(queue->host_allocator, queue_device_events);
  iree_allocator_free(queue->host_allocator, relationships);
  IREE_TRACE_ZONE_END(z0);
  return status;
}
