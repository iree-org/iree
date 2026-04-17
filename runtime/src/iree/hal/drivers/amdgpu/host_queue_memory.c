// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_memory.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"

static void iree_hal_amdgpu_host_queue_record_memory_event(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_profile_memory_event_type_t type,
    iree_hal_profile_memory_event_flags_t flags, uint32_t result,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer, const iree_hal_pool_reservation_t* reservation,
    iree_device_size_t length, uint64_t submission_id,
    uint32_t frontier_entry_count) {
  if (!iree_hal_amdgpu_logical_device_should_record_profile_memory_events(
          queue->logical_device)) {
    return;
  }

  iree_hal_profile_memory_event_t event =
      iree_hal_profile_memory_event_default();
  event.type = type;
  event.flags = flags;
  event.result = result;
  event.allocation_id = (uint64_t)(uintptr_t)buffer;
  event.pool_id = (uint64_t)(uintptr_t)pool;
  event.submission_id = submission_id;
  event.physical_device_ordinal =
      iree_hal_amdgpu_host_queue_profile_device_ordinal(queue);
  event.queue_ordinal = iree_hal_amdgpu_host_queue_profile_queue_ordinal(queue);
  event.frontier_entry_count = frontier_entry_count;
  event.memory_type = params.type;
  event.buffer_usage = params.usage;
  event.length = length;
  event.alignment = params.min_alignment ? params.min_alignment : 1;
  if (reservation) {
    event.flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_POOL_RESERVATION;
    event.backing_id = reservation->block_handle;
    event.offset = reservation->offset;
    if (type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA &&
        type != IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA) {
      event.length = reservation->length;
    }
  }
  iree_hal_amdgpu_logical_device_record_profile_memory_event(
      queue->logical_device, &event);
}

typedef struct iree_hal_amdgpu_host_queue_release_reservation_state_t {
  // Queue whose frontier owns the release.
  iree_hal_amdgpu_host_queue_t* queue;
  // Transient buffer whose reservation is released.
  iree_hal_buffer_t* buffer;
} iree_hal_amdgpu_host_queue_release_reservation_state_t;

static void iree_hal_amdgpu_host_queue_commit_transient_buffer(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  if (!iree_status_is_ok(status)) return;
  iree_hal_amdgpu_transient_buffer_commit((iree_hal_buffer_t*)user_data);
}

static void iree_hal_amdgpu_host_queue_decommit_transient_buffer(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  if (!iree_status_is_ok(status)) return;
  iree_hal_amdgpu_transient_buffer_decommit((iree_hal_buffer_t*)user_data);
}

static void iree_hal_amdgpu_host_queue_release_transient_buffer_reservation(
    void* user_data, const iree_async_frontier_t* queue_frontier) {
  iree_hal_amdgpu_host_queue_release_reservation_state_t* state =
      (iree_hal_amdgpu_host_queue_release_reservation_state_t*)user_data;
  iree_hal_pool_t* pool = NULL;
  iree_hal_pool_reservation_t reservation;
  if (iree_hal_amdgpu_transient_buffer_query_reservation(state->buffer, &pool,
                                                         &reservation)) {
    iree_hal_buffer_params_t params = {
        .type = iree_hal_buffer_memory_type(state->buffer),
        .access = iree_hal_buffer_allowed_access(state->buffer),
        .usage = iree_hal_buffer_allowed_usage(state->buffer),
    };
    iree_hal_amdgpu_host_queue_record_memory_event(
        state->queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX, pool,
        params, state->buffer, &reservation,
        iree_hal_buffer_byte_length(state->buffer),
        /*submission_id=*/0, queue_frontier ? queue_frontier->entry_count : 0);
  }
  iree_hal_amdgpu_transient_buffer_release_reservation(state->buffer,
                                                       queue_frontier);
}

static iree_hal_pool_t* iree_hal_amdgpu_host_queue_resolve_pool(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool) {
  return pool ? pool : queue->default_pool;
}

iree_status_t iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_pool_t** out_allocation_pool,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_allocation_pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_allocation_pool = NULL;
  *out_buffer = NULL;

  if (IREE_UNLIKELY(iree_any_bit_set(
          flags, ~(IREE_HAL_ALLOCA_FLAG_NONE |
                   IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME |
                   IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported alloca flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(allocation_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_alloca allocation_size must be non-zero");
  }

  iree_hal_pool_t* allocation_pool =
      iree_hal_amdgpu_host_queue_resolve_pool(queue, pool);

  iree_hal_buffer_params_canonicalize(params);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(allocation_pool, &capabilities);
  if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    params->type |= capabilities.memory_type;
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(capabilities.memory_type, params->type))) {
    iree_bitfield_string_temp_t requested_type_string;
    iree_bitfield_string_temp_t pool_type_string;
    iree_string_view_t requested_type =
        iree_hal_memory_type_format(params->type, &requested_type_string);
    iree_string_view_t pool_type = iree_hal_memory_type_format(
        capabilities.memory_type, &pool_type_string);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation pool does not support requested memory type %.*s "
        "(pool memory type %.*s)",
        (int)requested_type.size, requested_type.data, (int)pool_type.size,
        pool_type.data);
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(capabilities.supported_usage, params->usage))) {
    iree_bitfield_string_temp_t requested_usage_string;
    iree_bitfield_string_temp_t pool_usage_string;
    iree_string_view_t requested_usage =
        iree_hal_buffer_usage_format(params->usage, &requested_usage_string);
    iree_string_view_t pool_usage = iree_hal_buffer_usage_format(
        capabilities.supported_usage, &pool_usage_string);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation pool does not support requested buffer usage %.*s "
        "(pool usage %.*s)",
        (int)requested_usage.size, requested_usage.data, (int)pool_usage.size,
        pool_usage.data);
  }
  if (IREE_UNLIKELY(capabilities.max_allocation_size != 0 &&
                    allocation_size > capabilities.max_allocation_size)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "queue_alloca allocation_size %" PRIdsz
        " exceeds allocation pool max_allocation_size %" PRIdsz,
        allocation_size, capabilities.max_allocation_size);
  }

  iree_hal_buffer_placement_t placement = {
      .device = queue->logical_device,
      .queue_affinity = queue->queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  if (iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    placement.flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME;
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_create(
      placement, *params, allocation_size, allocation_size,
      queue->transient_buffer_pool, out_buffer));
  *out_allocation_pool = allocation_pool;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_alloca_reservation_t* out_reservation) {
  IREE_ASSERT_ARGUMENT(out_reservation);
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  out_reservation->acquire_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  out_reservation->wait_resolution = *resolution;

  iree_hal_amdgpu_fixed_frontier_t requester_frontier_storage;
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_host_queue_pool_requester_frontier(
          queue, resolution, &requester_frontier_storage);

  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      allocation_pool, allocation_size,
      params.min_alignment ? params.min_alignment : 1, requester_frontier,
      reserve_flags, &out_reservation->reservation,
      &out_reservation->acquire_info, &out_reservation->acquire_result));

  iree_hal_profile_memory_event_flags_t reserve_event_flags =
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION;
  if (out_reservation->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT) {
    reserve_event_flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER;
  } else if (out_reservation->acquire_result ==
                 IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
             out_reservation->acquire_result ==
                 IREE_HAL_POOL_ACQUIRE_OVER_BUDGET) {
    reserve_event_flags |= IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION;
  }
  iree_hal_amdgpu_host_queue_record_memory_event(
      queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RESERVE,
      reserve_event_flags, out_reservation->acquire_result, allocation_pool,
      params, buffer,
      out_reservation->acquire_result == IREE_HAL_POOL_ACQUIRE_EXHAUSTED ||
              out_reservation->acquire_result ==
                  IREE_HAL_POOL_ACQUIRE_OVER_BUDGET
          ? NULL
          : &out_reservation->reservation,
      allocation_size, /*submission_id=*/0,
      out_reservation->acquire_info.wait_frontier
          ? out_reservation->acquire_info.wait_frontier->entry_count
          : 0);

  switch (out_reservation->acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_ok_status();
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      if (!iree_all_bits_set(flags,
                             IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)) {
        iree_hal_amdgpu_host_queue_record_memory_event(
            queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
                IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
            out_reservation->acquire_result, allocation_pool, params, buffer,
            &out_reservation->reservation, allocation_size,
            /*submission_id=*/0,
            out_reservation->acquire_info.wait_frontier
                ? out_reservation->acquire_info.wait_frontier->entry_count
                : 0);
        iree_hal_pool_release_reservation(
            allocation_pool, &out_reservation->reservation,
            out_reservation->acquire_info.wait_frontier);
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "queue_alloca recycled pool memory requires "
            "IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER");
      }
      // A waitable pool reservation is legal whenever the HAL alloca flag
      // permits one. Appending device-side barriers is only one representation;
      // non-local, over-capacity, or forced-DEFER frontiers must route to the
      // cold host-gated memory-readiness path.
      if (iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
              queue, requester_frontier,
              out_reservation->acquire_info.wait_frontier,
              &out_reservation->wait_resolution)) {
        out_reservation->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
      } else {
        out_reservation->readiness =
            IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT;
        iree_hal_amdgpu_host_queue_record_memory_event(
            queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
            IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
                IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_FRONTIER,
            out_reservation->acquire_result, allocation_pool, params, buffer,
            &out_reservation->reservation, allocation_size,
            /*submission_id=*/0,
            out_reservation->acquire_info.wait_frontier
                ? out_reservation->acquire_info.wait_frontier->entry_count
                : 0);
      }
      return iree_ok_status();
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      out_reservation->readiness =
          IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION;
      iree_hal_amdgpu_host_queue_record_memory_event(
          queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_WAIT,
          IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION |
              IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_WAIT_NOTIFICATION,
          out_reservation->acquire_result, allocation_pool, params, buffer,
          /*reservation=*/NULL, allocation_size, /*submission_id=*/0,
          /*frontier_entry_count=*/0);
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              out_reservation->acquire_result);
  }
}

iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  const iree_async_frontier_t* reservation_failure_frontier =
      alloca_reservation->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
          ? alloca_reservation->acquire_info.wait_frontier
          : NULL;

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)buffer,
  };
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  iree_status_t status =
      iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
          queue, &alloca_reservation->wait_resolution, signal_semaphore_list,
          IREE_ARRAYSIZE(operation_resources), out_ready, &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_record_memory_event(
        queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        alloca_reservation->acquire_result, allocation_pool, params, buffer,
        &alloca_reservation->reservation, iree_hal_buffer_byte_length(buffer),
        /*submission_id=*/0,
        alloca_reservation->acquire_info.wait_frontier
            ? alloca_reservation->acquire_info.wait_frontier->entry_count
            : 0);
    iree_hal_pool_release_reservation(allocation_pool,
                                      &alloca_reservation->reservation,
                                      reservation_failure_frontier);
    return status;
  }

  iree_hal_buffer_t* backing_buffer = NULL;
  status = iree_hal_pool_materialize_reservation(
      allocation_pool, params, &alloca_reservation->reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_attach_reservation(
        buffer, allocation_pool, &alloca_reservation->reservation);
    iree_hal_amdgpu_transient_buffer_stage_backing(buffer, backing_buffer);
    iree_hal_amdgpu_host_queue_record_memory_event(
        queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_MATERIALIZE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        alloca_reservation->acquire_result, allocation_pool, params, buffer,
        &alloca_reservation->reservation, iree_hal_buffer_byte_length(buffer),
        /*submission_id=*/0,
        alloca_reservation->acquire_info.wait_frontier
            ? alloca_reservation->acquire_info.wait_frontier->entry_count
            : 0);
    backing_buffer = NULL;
  }
  iree_hal_buffer_release(backing_buffer);

  uint64_t submission_epoch = 0;
  if (iree_status_is_ok(status)) {
    submission_epoch = iree_hal_amdgpu_host_queue_finish_barrier_submission(
        queue, &alloca_reservation->wait_resolution, signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){
            .fn = iree_hal_amdgpu_host_queue_commit_transient_buffer,
            .user_data = buffer,
        },
        operation_resources, IREE_ARRAYSIZE(operation_resources),
        iree_hal_amdgpu_host_queue_post_commit_callback_null(),
        /*resource_set=*/NULL, submission_flags, &submission);
    iree_hal_amdgpu_host_queue_record_memory_event(
        queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_ALLOCA,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        alloca_reservation->acquire_result, allocation_pool, params, buffer,
        &alloca_reservation->reservation, iree_hal_buffer_byte_length(buffer),
        submission_epoch,
        alloca_reservation->acquire_info.wait_frontier
            ? alloca_reservation->acquire_info.wait_frontier->entry_count
            : 0);
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        queue, &alloca_reservation->wait_resolution, signal_semaphore_list,
        &(iree_hal_amdgpu_host_queue_profile_event_info_t){
            .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA,
            .submission_id = submission_epoch,
            .allocation_id = (uint64_t)(uintptr_t)buffer,
            .payload_length = iree_hal_buffer_byte_length(buffer),
            .operation_count = 1,
        });
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_fail_barrier_submission(queue, &submission);
    iree_hal_amdgpu_host_queue_record_memory_event(
        queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_POOL_RELEASE,
        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION,
        alloca_reservation->acquire_result, allocation_pool, params, buffer,
        &alloca_reservation->reservation, iree_hal_buffer_byte_length(buffer),
        /*submission_id=*/0,
        alloca_reservation->acquire_info.wait_frontier
            ? alloca_reservation->acquire_info.wait_frontier->entry_count
            : 0);
    iree_hal_pool_release_reservation(allocation_pool,
                                      &alloca_reservation->reservation,
                                      reservation_failure_frontier);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)buffer,
  };
  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  iree_hal_amdgpu_host_queue_release_reservation_state_t release_state = {
      .queue = queue,
      .buffer = buffer,
  };
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_barrier_submission(
          queue, resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){
              .fn = iree_hal_amdgpu_host_queue_decommit_transient_buffer,
              .user_data = buffer,
          },
          operation_resources, IREE_ARRAYSIZE(operation_resources),
          (iree_hal_amdgpu_host_queue_post_commit_callback_t){
              .fn =
                  iree_hal_amdgpu_host_queue_release_transient_buffer_reservation,
              .user_data = &release_state,
          },
          /*resource_set=*/NULL, submission_flags, &submission);
  iree_hal_buffer_params_t params = {
      .type = iree_hal_buffer_memory_type(buffer),
      .access = iree_hal_buffer_allowed_access(buffer),
      .usage = iree_hal_buffer_allowed_usage(buffer),
  };
  iree_hal_amdgpu_host_queue_record_memory_event(
      queue, IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_QUEUE_DEALLOCA,
      IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION, UINT32_MAX,
      /*pool=*/NULL, params, buffer, /*reservation=*/NULL,
      iree_hal_buffer_byte_length(buffer), submission_epoch,
      /*frontier_entry_count=*/0);
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list,
      &(iree_hal_amdgpu_host_queue_profile_event_info_t){
          .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA,
          .submission_id = submission_epoch,
          .allocation_id = (uint64_t)(uintptr_t)buffer,
          .payload_length = iree_hal_buffer_byte_length(buffer),
          .operation_count = 1,
      });
  return iree_ok_status();
}
