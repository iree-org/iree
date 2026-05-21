// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_replay.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/aql_program_validation.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_block.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_amdgpu_command_buffer_replay_t {
  // Reference-counted replay continuation resource.
  iree_hal_resource_t resource;
  // Host queue borrowed for the replay lifetime.
  iree_hal_amdgpu_host_queue_t* queue;
  // Host allocator used for this continuation allocation.
  iree_allocator_t host_allocator;
  // Command buffer retained for the replay lifetime.
  iree_hal_command_buffer_t* command_buffer;
  // Final user-visible signal semaphore list retained for the replay lifetime.
  iree_hal_semaphore_list_t signal_semaphore_list;
  // Binding table snapshot used after queue_execute returns.
  iree_hal_buffer_binding_table_t binding_table;
  // Resolved binding table base pointers indexed by original binding slot.
  const uint64_t* binding_ptrs;
  // Resource set retaining binding-table buffers for the replay lifetime.
  iree_hal_resource_set_t* binding_resource_set;
  // Immutable recorded AQL program borrowed from |command_buffer|.
  const iree_hal_amdgpu_aql_program_t* program;
  // Next command-buffer block to evaluate or submit.
  const iree_hal_amdgpu_command_buffer_block_header_t* current_block;
  // Wait resolution that prefixes the next packet submission.
  iree_hal_amdgpu_wait_resolution_t wait_resolution;
  // Intrusive continuation used to retry replay after notification drain.
  iree_hal_amdgpu_host_queue_post_drain_action_t post_drain_action;
} iree_hal_amdgpu_command_buffer_replay_t;

static void iree_hal_amdgpu_command_buffer_replay_consume_wait_resolution(
    iree_hal_amdgpu_command_buffer_replay_t* replay) {
  memset(&replay->wait_resolution, 0, sizeof(replay->wait_resolution));
}

static void iree_hal_amdgpu_command_buffer_replay_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_command_buffer_replay_t* replay =
      (iree_hal_amdgpu_command_buffer_replay_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_resource_set_free(replay->binding_resource_set);
  iree_hal_semaphore_list_release(replay->signal_semaphore_list);
  iree_hal_command_buffer_release(replay->command_buffer);
  iree_allocator_free(replay->host_allocator, replay);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_command_buffer_replay_vtable = {
        .destroy = iree_hal_amdgpu_command_buffer_replay_destroy,
};

static iree_status_t iree_hal_amdgpu_command_buffer_replay_create(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_command_buffer_replay_t** out_replay) {
  *out_replay = NULL;

  if (!*inout_binding_resource_set) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
            queue, command_buffer, binding_table, execute_flags,
            inout_binding_resource_set));
  }

  const iree_host_size_t signal_count = signal_semaphore_list.count;
  const iree_host_size_t binding_count = command_buffer->binding_count;

  iree_host_size_t semaphore_offset = 0;
  iree_host_size_t payload_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_ptr_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_command_buffer_replay_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(signal_count, iree_hal_semaphore_t*, 1,
                                &semaphore_offset),
      IREE_STRUCT_FIELD_ALIGNED(signal_count, uint64_t, 1, &payload_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_count, iree_hal_buffer_binding_t, 1,
                                &binding_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_count, uint64_t, 1,
                                &binding_ptr_offset)));

  iree_hal_amdgpu_command_buffer_replay_t* replay = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator, total_size,
                                             (void**)&replay));
  memset(replay, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_amdgpu_command_buffer_replay_vtable,
                               &replay->resource);
  replay->queue = queue;
  replay->host_allocator = queue->host_allocator;
  replay->command_buffer = command_buffer;
  replay->binding_resource_set = *inout_binding_resource_set;
  *inout_binding_resource_set = NULL;
  replay->program = iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  replay->current_block = replay->program->first_block;
  replay->wait_resolution = *resolution;
  iree_hal_command_buffer_retain(command_buffer);

  uint8_t* storage = (uint8_t*)replay;
  if (signal_count > 0) {
    replay->signal_semaphore_list.count = signal_count;
    replay->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)(storage + semaphore_offset);
    replay->signal_semaphore_list.payload_values =
        (uint64_t*)(storage + payload_offset);
    memcpy(replay->signal_semaphore_list.semaphores,
           signal_semaphore_list.semaphores,
           signal_count * sizeof(*signal_semaphore_list.semaphores));
    memcpy(replay->signal_semaphore_list.payload_values,
           signal_semaphore_list.payload_values,
           signal_count * sizeof(*signal_semaphore_list.payload_values));
  }
  iree_hal_semaphore_list_retain(replay->signal_semaphore_list);
  if (binding_count > 0) {
    iree_hal_buffer_binding_t* binding_storage =
        (iree_hal_buffer_binding_t*)(storage + binding_offset);
    replay->binding_table.count = binding_count;
    replay->binding_table.bindings = binding_storage;
    memcpy(binding_storage, binding_table.bindings,
           binding_count * sizeof(*binding_table.bindings));
    uint64_t* binding_ptrs = (uint64_t*)(storage + binding_ptr_offset);
    iree_status_t status =
        iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
            command_buffer, replay->binding_table, binding_ptrs);
    if (!iree_status_is_ok(status)) {
      iree_hal_resource_release(&replay->resource);
      return status;
    }
    replay->binding_ptrs = binding_ptrs;
  }

  *out_replay = replay;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_command_buffer_replay_clone_queue_error(
    iree_hal_amdgpu_command_buffer_replay_t* replay) {
  if (IREE_UNLIKELY(replay->queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &replay->queue->error_status, iree_memory_order_acquire);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

static void iree_hal_amdgpu_command_buffer_replay_fail_signals(
    iree_hal_amdgpu_command_buffer_replay_t* replay, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  if (iree_hal_semaphore_list_is_empty(replay->signal_semaphore_list)) {
    iree_status_free(status);
    return;
  }
  iree_hal_semaphore_list_fail(replay->signal_semaphore_list, status);
}

static iree_status_t iree_hal_amdgpu_command_buffer_replay_park(
    iree_hal_amdgpu_command_buffer_replay_t* replay,
    iree_hal_amdgpu_host_queue_post_drain_fn_t post_drain_fn) {
  iree_hal_resource_retain(&replay->resource);
  iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
      replay->queue, &replay->post_drain_action, post_drain_fn, replay);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_command_buffer_replay_submit_completion_packet(
    iree_hal_amdgpu_command_buffer_replay_t* replay,
    const iree_hal_amdgpu_wait_resolution_t* resolution, bool* out_ready) {
  *out_ready = false;

  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info = {
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE,
      .command_buffer_id =
          iree_hal_amdgpu_aql_command_buffer_profile_id(replay->command_buffer),
      .operation_count = 0,
  };
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (iree_hal_amdgpu_host_queue_should_profile_queue_device_events(
          replay->queue)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            replay->queue, /*event_count=*/1, &profile_queue_device_events));
  }

  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      replay->queue, resolution, replay->signal_semaphore_list,
      /*operation_resource_count=*/1, /*payload_packet_count=*/1,
      /*kernarg_block_count=*/0, out_ready, &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        replay->queue, profile_queue_device_events);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(
        replay->queue, resolution, &submission);
    iree_hal_resource_t* replay_resource = &replay->resource;
    const uint64_t submission_id =
        iree_hal_amdgpu_host_queue_finish_kernel_submission(
            replay->queue, resolution, replay->signal_semaphore_list,
            &replay_resource, /*operation_resource_count=*/1,
            /*inout_resource_set=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
            &submission);

    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
        iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
            replay->queue, profile_queue_device_events, &profile_event_info);
    if (queue_device_event) {
      submission.reclaim_entry->queue_device_event_first_position =
          profile_queue_device_events.first_event_position;
      submission.reclaim_entry->queue_device_event_count =
          profile_queue_device_events.event_count;
      queue_device_event->submission_id = submission_id;
    }

    if (queue_device_event) {
      const uint64_t timestamp_packet_id =
          submission.first_packet_id + submission.packet_count - 1;
      iree_hal_amdgpu_host_queue_commit_timestamp_range(
          replay->queue, timestamp_packet_id,
          iree_hal_amdgpu_host_queue_command_buffer_packet_control(
              replay->queue, resolution, replay->signal_semaphore_list,
              /*packet_index=*/0, IREE_HSA_FENCE_SCOPE_NONE,
              IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL),
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &replay->queue->notification_ring),
          &queue_device_event->start_tick, &queue_device_event->end_tick);
    } else {
      iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
          &replay->queue->aql_ring,
          submission.first_packet_id + submission.packet_count - 1);
      const uint16_t header = iree_hal_amdgpu_aql_emit_nop(
          &packet->barrier_and,
          iree_hal_amdgpu_aql_packet_control_barrier(
              resolution->inline_acquire_scope,
              iree_hal_amdgpu_host_queue_signal_list_release_scope(
                  replay->queue, replay->signal_semaphore_list)),
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &replay->queue->notification_ring));
      iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
    }
    iree_hal_amdgpu_aql_ring_doorbell(
        &replay->queue->aql_ring,
        submission.first_packet_id + submission.packet_count - 1);
    profile_event_info.submission_id = submission_id;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        replay->queue, resolution, replay->signal_semaphore_list,
        &profile_event_info);
    memset(&submission, 0, sizeof(submission));
    replay->current_block = NULL;
    iree_hal_amdgpu_command_buffer_replay_consume_wait_resolution(replay);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_command_buffer_replay_resume_under_lock(
    iree_hal_amdgpu_command_buffer_replay_t* replay,
    iree_hal_amdgpu_host_queue_post_drain_fn_t post_drain_fn) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_command_buffer_replay_clone_queue_error(replay));

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && replay->current_block) {
    status = iree_hal_amdgpu_aql_program_validate_block_terminator(
        replay->current_block);
    if (!iree_status_is_ok(status)) break;
    const uint8_t terminator_opcode = replay->current_block->terminator_opcode;

    if (replay->current_block->aql_packet_count == 0) {
      if (terminator_opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN) {
        const iree_hal_amdgpu_wait_resolution_t* current_resolution =
            &replay->wait_resolution;
        bool ready = false;
        status = iree_hal_amdgpu_command_buffer_replay_submit_completion_packet(
            replay, current_resolution, &ready);
        if (iree_status_is_ok(status) && !ready) {
          status =
              iree_hal_amdgpu_command_buffer_replay_park(replay, post_drain_fn);
        }
        break;
      }

      const iree_hal_amdgpu_command_buffer_block_header_t* next_block = NULL;
      status = iree_hal_amdgpu_aql_program_next_linear_block(
          replay->program, replay->current_block,
          replay->current_block->terminator_target_block_ordinal, &next_block);
      if (iree_status_is_ok(status)) {
        replay->current_block = next_block;
      }
      continue;
    }

    const iree_hal_amdgpu_wait_resolution_t* current_resolution =
        &replay->wait_resolution;

    if (terminator_opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN) {
      iree_hal_resource_t* replay_resource = &replay->resource;
      bool ready = false;
      status = iree_hal_amdgpu_host_queue_submit_command_buffer_block(
          replay->queue, current_resolution, replay->signal_semaphore_list,
          replay->command_buffer, replay->binding_table, replay->binding_ptrs,
          replay->current_block, /*inout_binding_resource_set=*/NULL,
          (iree_hal_amdgpu_reclaim_action_t){0}, &replay_resource,
          /*operation_resource_count=*/1,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready);
      if (iree_status_is_ok(status) && ready) {
        replay->current_block = NULL;
        iree_hal_amdgpu_command_buffer_replay_consume_wait_resolution(replay);
      } else if (iree_status_is_ok(status)) {
        status =
            iree_hal_amdgpu_command_buffer_replay_park(replay, post_drain_fn);
      }
      break;
    }

    const iree_hal_amdgpu_command_buffer_block_header_t* next_block = NULL;
    status = iree_hal_amdgpu_aql_program_next_linear_block(
        replay->program, replay->current_block,
        replay->current_block->terminator_target_block_ordinal, &next_block);
    if (!iree_status_is_ok(status)) break;

    iree_hal_resource_t* replay_resource = &replay->resource;
    bool ready = false;
    status = iree_hal_amdgpu_host_queue_submit_command_buffer_block(
        replay->queue, current_resolution, iree_hal_semaphore_list_empty(),
        replay->command_buffer, replay->binding_table, replay->binding_ptrs,
        replay->current_block, /*inout_binding_resource_set=*/NULL,
        (iree_hal_amdgpu_reclaim_action_t){0}, &replay_resource,
        /*operation_resource_count=*/1,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready);
    if (iree_status_is_ok(status) && ready) {
      replay->current_block = next_block;
      iree_hal_amdgpu_command_buffer_replay_consume_wait_resolution(replay);
      continue;
    } else if (iree_status_is_ok(status)) {
      status =
          iree_hal_amdgpu_command_buffer_replay_park(replay, post_drain_fn);
    }
    break;
  }
  return status;
}

static void iree_hal_amdgpu_command_buffer_replay_post_drain(void* user_data) {
  iree_hal_amdgpu_command_buffer_replay_t* replay =
      (iree_hal_amdgpu_command_buffer_replay_t*)user_data;
  iree_slim_mutex_lock(&replay->queue->locks.submission_mutex);
  iree_status_t status =
      iree_hal_amdgpu_command_buffer_replay_resume_under_lock(
          replay, iree_hal_amdgpu_command_buffer_replay_post_drain);
  iree_slim_mutex_unlock(&replay->queue->locks.submission_mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_command_buffer_replay_fail_signals(replay, status);
  }
  iree_hal_resource_release(&replay->resource);
}

iree_status_t iree_hal_amdgpu_command_buffer_replay_start_under_lock(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t execute_flags,
    iree_hal_resource_set_t** inout_binding_resource_set) {
  iree_hal_amdgpu_command_buffer_replay_t* replay = NULL;
  iree_status_t status = iree_hal_amdgpu_command_buffer_replay_create(
      queue, resolution, signal_semaphore_list, command_buffer, binding_table,
      execute_flags, inout_binding_resource_set, &replay);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_command_buffer_replay_resume_under_lock(
        replay, iree_hal_amdgpu_command_buffer_replay_post_drain);
    iree_hal_resource_release(&replay->resource);
  } else {
    iree_hal_resource_set_free(*inout_binding_resource_set);
    *inout_binding_resource_set = NULL;
  }
  return status;
}
