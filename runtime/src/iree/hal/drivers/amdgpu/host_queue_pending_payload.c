// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue_blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer.h"
#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"
#include "iree/hal/drivers/amdgpu/host_queue_host_call.h"
#include "iree/hal/drivers/amdgpu/host_queue_memory.h"
#include "iree/hal/drivers/amdgpu/host_queue_pending_operation.h"

//===----------------------------------------------------------------------===//
// Pending operation payload issue
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_pending_op_issue_fill(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_fill(
      op->queue, resolution, op->signal_semaphore_list, op->fill.target_buffer,
      op->fill.target_offset, op->fill.length, op->fill.pattern_bits,
      op->fill.pattern_length, op->fill.flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_copy(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_copy(
      op->queue, resolution, op->signal_semaphore_list, op->copy.source_buffer,
      op->copy.source_offset, op->copy.target_buffer, op->copy.target_offset,
      op->copy.length, op->copy.flags, op->copy.profile_event_type,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_update(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_update(
      op->queue, resolution, op->signal_semaphore_list, op->update.source_data,
      /*source_offset=*/0, op->update.target_buffer, op->update.target_offset,
      op->update.length, op->update.flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_dispatch(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_dispatch(
      op->queue, resolution, op->signal_semaphore_list, op->dispatch.executable,
      op->dispatch.export_ordinal, op->dispatch.config, op->dispatch.constants,
      op->dispatch.bindings, op->dispatch.flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_execute(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  if (op->execute.command_buffer) {
    iree_status_t status = iree_hal_amdgpu_host_queue_submit_command_buffer(
        op->queue, resolution, op->signal_semaphore_list,
        op->execute.command_buffer, op->execute.binding_table,
        op->execute.flags, &op->execute.binding_resource_set, &issue->ready);
    if (iree_status_is_ok(status) && issue->ready) {
      iree_hal_amdgpu_pending_op_release_retained(op);
    }
    return status;
  }

  iree_status_t status = iree_hal_amdgpu_host_queue_try_submit_barrier(
      op->queue, resolution, op->signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){0},
      /*operation_resources=*/NULL,
      /*operation_resource_count=*/0,
      /*profile_event_info=*/NULL,
      iree_hal_amdgpu_host_queue_post_commit_callback_null(),
      /*resource_set=*/NULL, IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE,
      &issue->ready,
      /*out_submission_id=*/NULL);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_alloca(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_alloca(
      op->queue, resolution, op->signal_semaphore_list, op->alloca_op.pool,
      op->alloca_op.params, op->alloca_op.allocation_size, op->alloca_op.flags,
      op->alloca_op.reserve_flags, op->alloca_op.buffer,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, op,
      &issue->memory_wait_op, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_dealloca(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_dealloca(
      op->queue, resolution, op->signal_semaphore_list, op->dealloca.buffer,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_host_call(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_host_call(
      op->queue, resolution, op->signal_semaphore_list, op->host_call.call,
      op->host_call.args, op->host_call.flags, &issue->ready);
  if (iree_status_is_ok(status) && issue->ready) {
    iree_hal_amdgpu_pending_op_release_retained(op);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_pending_op_issue_host_action(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  iree_status_t status = iree_hal_amdgpu_host_queue_try_submit_barrier(
      op->queue, resolution, iree_hal_semaphore_list_empty(),
      op->host_action.action, op->retained_resources,
      op->retained_resource_count,
      /*profile_event_info=*/NULL,
      iree_hal_amdgpu_host_queue_post_commit_callback_null(),
      /*resource_set=*/NULL, IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE,
      &issue->ready,
      /*out_submission_id=*/NULL);
  if (iree_status_is_ok(status) && issue->ready) {
    op->retained_resource_count = 0;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_pending_op_issue_payload(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_pending_op_payload_issue_t* issue) {
  switch (op->type) {
    case IREE_HAL_AMDGPU_PENDING_OP_FILL:
      return iree_hal_amdgpu_pending_op_issue_fill(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_COPY:
      return iree_hal_amdgpu_pending_op_issue_copy(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_UPDATE:
      return iree_hal_amdgpu_pending_op_issue_update(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_DISPATCH:
      return iree_hal_amdgpu_pending_op_issue_dispatch(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_EXECUTE:
      return iree_hal_amdgpu_pending_op_issue_execute(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_ALLOCA:
      return iree_hal_amdgpu_pending_op_issue_alloca(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA:
      return iree_hal_amdgpu_pending_op_issue_dealloca(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL:
      return iree_hal_amdgpu_pending_op_issue_host_call(op, resolution, issue);
    case IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION:
      return iree_hal_amdgpu_pending_op_issue_host_action(op, resolution,
                                                          issue);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pending op type %d", op->type);
  }
}

//===----------------------------------------------------------------------===//
// Pending operation payload capture
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_host_queue_defer_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_ALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->alloca_op.pool = pool;
  op->alloca_op.params = params;
  op->alloca_op.allocation_size = allocation_size;
  op->alloca_op.flags = flags;
  op->alloca_op.reserve_flags = reserve_flags;
  op->alloca_op.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->dealloca.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_FILL, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->fill.target_buffer = target_buffer;
  op->fill.target_offset = target_offset;
  op->fill.length = length;
  op->fill.pattern_bits = pattern_bits;
  op->fill.pattern_length = pattern_length;
  op->fill.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_profile_queue_event_type_t profile_event_type,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/2, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_COPY, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)source_buffer);
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->copy.source_buffer = source_buffer;
  op->copy.source_offset = source_offset;
  op->copy.target_buffer = target_buffer;
  op->copy.target_offset = target_offset;
  op->copy.length = length;
  op->copy.flags = flags;
  op->copy.profile_event_type = profile_event_type;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));
  (void)target_device_ptr;

  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_UPDATE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);

  void* source_copy = NULL;
  iree_status_t status =
      iree_arena_allocate(&op->arena, source_length, &source_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_update");
  }
  memcpy(source_copy, source_bytes, source_length);
  op->update.source_data = source_copy;
  op->update.target_buffer = target_buffer;
  op->update.target_offset = target_offset;
  op->update.length = length;
  op->update.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_execute_flags(flags));
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }
  const iree_host_size_t binding_count =
      command_buffer ? command_buffer->binding_count : 0;
  if (command_buffer && command_buffer->binding_count == 0) {
    binding_table = iree_hal_buffer_binding_table_empty();
  }

  iree_hal_resource_set_t* binding_resource_set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_create_binding_table_resource_set(
          queue, command_buffer, binding_table, flags, &binding_resource_set));

  const iree_host_size_t operation_resource_count = command_buffer ? 1 : 0;
  uint16_t max_resources = 0;
  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_allocate(
        queue, wait_semaphore_list, signal_semaphore_list,
        IREE_HAL_AMDGPU_PENDING_OP_EXECUTE, max_resources, &op);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)command_buffer);
    op->execute.command_buffer = command_buffer;
    op->execute.binding_resource_set = binding_resource_set;
    binding_resource_set = NULL;
    op->execute.flags = flags;
  }

  if (iree_status_is_ok(status) && binding_count > 0) {
    iree_hal_buffer_binding_t* bindings_copy = NULL;
    iree_host_size_t binding_table_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &binding_table_size,
        IREE_STRUCT_FIELD(binding_count, iree_hal_buffer_binding_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, binding_table_size,
                                   (void**)&bindings_copy);
    }
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, binding_table.bindings, binding_table_size);
      op->execute.binding_table.count = binding_count;
      op->execute.binding_table.bindings = bindings_copy;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_resource_set_free(binding_resource_set);
    if (op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                    iree_status_clone(status));
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_defer_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  iree_host_size_t operation_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &operation_resource_count));
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DISPATCH, max_resources, &op));
  const bool borrow_resource_lifetimes =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES);
  if (!borrow_resource_lifetimes) {
    iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)executable);
  }
  op->dispatch.executable = executable;
  op->dispatch.export_ordinal = export_ordinal;
  op->dispatch.config = config;
  op->dispatch.flags = flags;

  iree_status_t status = iree_ok_status();
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    status =
        iree_arena_allocate(&op->arena, constants.data_length, &constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      op->dispatch.constants.data = (const uint8_t*)constants_copy;
      op->dispatch.constants.data_length = constants.data_length;
    }
  }

  if (iree_status_is_ok(status) && bindings.count > 0 &&
      !iree_any_bit_set(flags,
                        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    iree_host_size_t binding_ref_size = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &binding_ref_size,
        IREE_STRUCT_FIELD(bindings.count, iree_hal_buffer_ref_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_arena_allocate(&op->arena, binding_ref_size,
                                   (void**)&bindings_copy);
    }
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings.values, binding_ref_size);
      if (!borrow_resource_lifetimes) {
        for (iree_host_size_t i = 0; i < bindings.count; ++i) {
          iree_hal_amdgpu_pending_op_retain(
              op, (iree_hal_resource_t*)bindings_copy[i].buffer);
        }
      }
      op->dispatch.bindings.count = bindings.count;
      op->dispatch.bindings.values = bindings_copy;
    }
  }
  if (iree_status_is_ok(status) && !borrow_resource_lifetimes &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    iree_hal_amdgpu_pending_op_retain(
        op, (iree_hal_resource_t*)config.workgroup_count_ref.buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_op = op;
  } else {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op,
                                                  iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_defer_host_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      /*signal_semaphore_count=*/0,
      /*operation_resource_count=*/operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  const iree_hal_semaphore_list_t empty_signal_list =
      iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, &empty_signal_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_ACTION, max_resources, &op));
  for (iree_host_size_t i = 0; i < operation_resource_count; ++i) {
    iree_hal_amdgpu_pending_op_retain(op, operation_resources[i]);
  }
  op->host_action.action = action;
  *out_op = op;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_defer_host_call(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/0, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL, max_resources, &op));
  op->host_call.call = call;
  memcpy(op->host_call.args, args, sizeof(op->host_call.args));
  op->host_call.flags = flags;
  *out_op = op;
  return iree_ok_status();
}
