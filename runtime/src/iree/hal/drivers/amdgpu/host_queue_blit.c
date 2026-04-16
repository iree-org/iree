// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_blit.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "fill kernargs must fit in one kernarg ring block");

// PM4 WRITE_DATA payload for tiny queue_fill/queue_update operations.
typedef struct iree_hal_amdgpu_host_queue_pm4_write_data_t {
  // Device-visible target pointer written by PM4 WRITE_DATA.
  void* target_device_ptr;
  // Immediate value written to |target_device_ptr|.
  uint64_t value;
  // Byte length written from |value|; currently 4 or 8.
  uint8_t length;
} iree_hal_amdgpu_host_queue_pm4_write_data_t;

// Validates a queue_fill target and resolves the target device pointer.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_fill_target(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags, uint8_t** out_target_device_ptr) {
  *out_target_device_ptr = NULL;

  if (IREE_UNLIKELY(!target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must be non-null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill patterns must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_FILL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill flags: 0x%" PRIx64, flags);
  }

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

// Returns the low-byte fill pattern extended to a full 64-bit repetition.
static uint64_t iree_hal_amdgpu_host_queue_extend_fill_pattern_x8(
    uint64_t pattern_bits, iree_host_size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      const uint64_t pattern = pattern_bits & 0xFFu;
      return pattern * 0x0101010101010101ull;
    }
    case 2: {
      const uint64_t pattern = pattern_bits & 0xFFFFu;
      return pattern | (pattern << 16) | (pattern << 32) | (pattern << 48);
    }
    default: {
      const uint64_t pattern = pattern_bits & 0xFFFFFFFFull;
      return pattern | (pattern << 32);
    }
  }
}

static bool iree_hal_amdgpu_host_queue_can_use_pm4_write_data(
    const iree_hal_amdgpu_host_queue_t* queue, const void* target_device_ptr,
    iree_host_size_t length) {
  return queue->pm4_ib_slots && (length == 4 || length == 8) &&
         iree_host_ptr_has_alignment(target_device_ptr, sizeof(uint32_t));
}

static bool iree_hal_amdgpu_host_queue_prepare_pm4_fill_write_data(
    const iree_hal_amdgpu_host_queue_t* queue, void* target_device_ptr,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length,
    iree_hal_amdgpu_host_queue_pm4_write_data_t* out_write_data) {
  if (length != 4 && length != 8) {
    return false;
  }
  if (!iree_hal_amdgpu_host_queue_can_use_pm4_write_data(
          queue, target_device_ptr, (iree_host_size_t)length)) {
    return false;
  }
  if (!iree_host_ptr_has_alignment(target_device_ptr, pattern_length) ||
      !iree_device_size_has_alignment(length, pattern_length)) {
    return false;
  }

  out_write_data->target_device_ptr = target_device_ptr;
  out_write_data->value = iree_hal_amdgpu_host_queue_extend_fill_pattern_x8(
      pattern_bits, pattern_length);
  out_write_data->length = (uint8_t)length;
  return true;
}

static bool iree_hal_amdgpu_host_queue_prepare_pm4_update_write_data(
    const iree_hal_amdgpu_host_queue_t* queue, const uint8_t* source_bytes,
    iree_host_size_t source_length, void* target_device_ptr,
    iree_hal_amdgpu_host_queue_pm4_write_data_t* out_write_data) {
  if (!iree_hal_amdgpu_host_queue_can_use_pm4_write_data(
          queue, target_device_ptr, source_length)) {
    return false;
  }

  out_write_data->target_device_ptr = target_device_ptr;
  out_write_data->value = 0;
  out_write_data->length = (uint8_t)source_length;
  memcpy(&out_write_data->value, source_bytes, source_length);
  return true;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_pm4_write_data(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer,
    const iree_hal_amdgpu_host_queue_pm4_write_data_t* write_data,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };

  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  if (write_data->length == 4) {
    uint32_t value = 0;
    memcpy(&value, &write_data->value, sizeof(value));
    submission.ib_dword_count = iree_hal_amdgpu_pm4_emit_write_data32(
        submission.pm4_ib_slot, write_data->target_device_ptr, value);
  } else {
    submission.ib_dword_count = iree_hal_amdgpu_pm4_emit_write_data64(
        submission.pm4_ib_slot, write_data->target_device_ptr,
        write_data->value);
  }
  iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags, &submission);
  return iree_ok_status();
}

// Prepares a fill dispatch packet and kernargs in stack-local storage without
// touching queue rings. All user-input validation must happen before this so
// the caller can avoid reserving AQL slots before the packet shape is known.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_fill_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, uint8_t* target_device_ptr,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_device_buffer_fill_kernargs_t* out_kernargs) {
  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          queue->transfer_context, &dispatch_packet, target_device_ptr, length,
          pattern_bits, (uint8_t)pattern_length, &kernargs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill dispatch shape (length=%" PRIdsz
                            ", pattern_length=%" PRIhsz ")",
                            length, pattern_length);
  }
  dispatch_packet.kernarg_address = NULL;

  *out_dispatch_packet = dispatch_packet;
  *out_kernargs = kernargs;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_fill_target(
      target_buffer, target_offset, length, pattern_length, flags,
      &target_device_ptr));

  iree_hal_amdgpu_host_queue_pm4_write_data_t pm4_write_data;
  if (iree_hal_amdgpu_host_queue_prepare_pm4_fill_write_data(
          queue, target_device_ptr, length, pattern_bits, pattern_length,
          &pm4_write_data)) {
    return iree_hal_amdgpu_host_queue_submit_pm4_write_data(
        queue, resolution, signal_semaphore_list, target_buffer,
        &pm4_write_data, submission_flags, out_ready);
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_fill_dispatch(
      queue, target_device_ptr, length, pattern_bits, pattern_length,
      &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags, out_ready);
}

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "copy kernargs must fit in one kernarg ring block");

// Prepares a copy dispatch packet and kernargs in stack-local storage without
// touching queue rings. Overlapping ranges within the same buffer are rejected
// here because the builtin copy kernels implement memcpy semantics, not
// memmove.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_copy_flags_t flags,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_device_buffer_copy_kernargs_t* out_kernargs) {
  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source buffer must be non-null");
  }
  if (IREE_UNLIKELY(!target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must be non-null");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(iree_hal_buffer_test_overlap(source_buffer, source_offset,
                                                 length, target_buffer,
                                                 target_offset, length) !=
                    IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  iree_hal_buffer_t* allocated_source_buffer =
      iree_hal_buffer_allocated_buffer(source_buffer);
  const uint8_t* source_device_ptr =
      (const uint8_t*)iree_hal_amdgpu_buffer_device_pointer(
          allocated_source_buffer);
  if (IREE_UNLIKELY(!source_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source buffer must be backed by an AMDGPU allocation");
  }
  source_device_ptr +=
      iree_hal_buffer_byte_offset(source_buffer) + source_offset;

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &dispatch_packet, source_device_ptr,
          target_device_ptr, length, &kernargs))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported copy dispatch shape (source_offset=%" PRIdsz
        ", target_offset=%" PRIdsz ", length=%" PRIdsz ")",
        source_offset, target_offset, length);
  }
  dispatch_packet.kernarg_address = NULL;

  *out_dispatch_packet = dispatch_packet;
  *out_kernargs = kernargs;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
      queue, source_buffer, source_offset, target_buffer, target_offset, length,
      flags, &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags, out_ready);
}

iree_status_t iree_hal_amdgpu_host_queue_submit_copy_with_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hsa_fence_scope_t minimum_release_scope,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* extra_operation_resources,
    iree_host_size_t extra_operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  if (IREE_UNLIKELY(extra_operation_resource_count > 0 &&
                    !extra_operation_resources)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "extra operation resources must be non-null");
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
      queue, source_buffer, source_offset, target_buffer, target_offset, length,
      flags, &dispatch_packet, &kernargs));

  iree_host_size_t operation_resource_count = 0;
  if (!iree_host_size_checked_add(2, extra_operation_resource_count,
                                  &operation_resource_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "copy operation resource count overflows");
  }
  iree_host_size_t operation_resources_size = 0;
  if (!iree_host_size_checked_mul(operation_resource_count,
                                  sizeof(iree_hal_resource_t*),
                                  &operation_resources_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "copy operation resource table size overflows");
  }
  iree_hal_resource_t** operation_resources =
      (iree_hal_resource_t**)iree_alloca(operation_resources_size);
  operation_resources[0] = (iree_hal_resource_t*)source_buffer;
  operation_resources[1] = (iree_hal_resource_t*)target_buffer;
  for (iree_host_size_t i = 0; i < extra_operation_resource_count; ++i) {
    operation_resources[2 + i] = extra_operation_resources[i];
  }

  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      /*kernarg_block_count=*/1, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  memcpy(submission.kernel.kernarg_blocks->data, &kernargs, sizeof(kernargs));
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, &dispatch_packet,
          submission.kernel.kernarg_blocks->data,
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring));
  submission.minimum_acquire_scope = minimum_acquire_scope;
  submission.minimum_release_scope = minimum_release_scope;
  submission.kernel.pre_signal_action = pre_signal_action;
  iree_hal_amdgpu_host_queue_finish_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      operation_resource_count, submission_flags, &submission);
  return iree_ok_status();
}

// Validates a queue_update request and resolves the source host span and target
// device pointer. The source host pointer is captured by the caller either into
// the pending-op arena or into the queue-owned kernarg ring.
iree_status_t iree_hal_amdgpu_host_queue_prepare_update_copy(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    const uint8_t** out_source_bytes, iree_host_size_t* out_source_length,
    uint8_t** out_target_device_ptr) {
  *out_source_bytes = NULL;
  *out_source_length = 0;
  *out_target_device_ptr = NULL;

  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source buffer must be non-null");
  }
  if (IREE_UNLIKELY(!target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must be non-null");
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(flags != IREE_HAL_UPDATE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported update flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "update length %" PRIdsz
                            " exceeds host addressable size %" PRIhsz,
                            length, IREE_HOST_SIZE_MAX);
  }
  const iree_host_size_t source_length = (iree_host_size_t)length;
  iree_host_size_t source_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(source_offset, source_length,
                                                &source_end))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update source span overflows host size (offset=%" PRIhsz
        ", length=%" PRIhsz ")",
        source_offset, source_length);
  }
  (void)source_end;

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  *out_source_bytes = (const uint8_t*)source_buffer + source_offset;
  *out_source_length = source_length;
  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));

  iree_hal_amdgpu_host_queue_pm4_write_data_t pm4_write_data;
  if (iree_hal_amdgpu_host_queue_prepare_pm4_update_write_data(
          queue, source_bytes, source_length, target_device_ptr,
          &pm4_write_data)) {
    return iree_hal_amdgpu_host_queue_submit_pm4_write_data(
        queue, resolution, signal_semaphore_list, target_buffer,
        &pm4_write_data, submission_flags, out_ready);
  }

  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  iree_host_size_t kernarg_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          source_payload_offset, source_length, &kernarg_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update staging payload overflows host size (offset=%" PRIhsz
        ", source_length=%" PRIhsz ")",
        source_payload_offset, source_length);
  }
  const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (IREE_UNLIKELY(kernarg_block_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update staging payload requires too many kernarg blocks (%" PRIhsz
        ", max=%u)",
        kernarg_block_count, UINT32_MAX);
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  // The eventual staged source pointer is 16-byte aligned by construction. Use
  // a synthetic aligned pointer for pre-reservation packet-shape selection,
  // then patch source_ptr to the real ring address after allocation succeeds.
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &dispatch_packet,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_device_ptr, length, &kernargs))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported update dispatch shape (target_offset=%" PRIdsz
        ", length=%" PRIdsz ", source_payload_alignment=%d)",
        target_offset, length,
        IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT);
  }

  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list,
      /*operation_resource_count=*/1, (uint32_t)kernarg_block_count, out_ready,
      &submission));
  if (!*out_ready) return iree_ok_status();

  uint8_t* staged_source_bytes =
      (uint8_t*)submission.kernel.kernarg_blocks + source_payload_offset;
  memcpy(submission.kernel.kernarg_blocks->data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)
       submission.kernel.kernarg_blocks->data)
      ->source_ptr = staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, source_length);
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, &dispatch_packet,
          submission.kernel.kernarg_blocks->data,
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring));

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  iree_hal_amdgpu_host_queue_finish_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags, &submission);
  return iree_ok_status();
}
