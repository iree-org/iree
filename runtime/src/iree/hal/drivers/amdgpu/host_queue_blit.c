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
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "fill kernargs must fit in one kernarg ring block");

static iree_hal_amdgpu_host_queue_profile_event_info_t
iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
    iree_hal_profile_queue_event_type_t type, uint64_t payload_length) {
  iree_hal_amdgpu_host_queue_profile_event_info_t info = {
      .type = type,
      .payload_length = payload_length,
      .operation_count = 1,
  };
  return info;
}

static void iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint64_t submission_id,
    iree_hal_amdgpu_host_queue_profile_event_info_t* info) {
  info->submission_id = submission_id;
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list, info);
}

// PM4 WRITE_DATA payload for tiny queue_fill/queue_update operations.
typedef struct iree_hal_amdgpu_host_queue_pm4_write_data_t {
  // Device-visible target pointer written by PM4 WRITE_DATA.
  void* target_device_ptr;
  // Immediate value written to |target_device_ptr|.
  uint64_t value;
  // Byte length written from |value|; currently 4 or 8.
  uint8_t length;
} iree_hal_amdgpu_host_queue_pm4_write_data_t;

// PM4 COPY_DATA payload for tiny queue_copy operations.
typedef struct iree_hal_amdgpu_host_queue_pm4_copy_data_t {
  // Device-visible source pointer read by PM4 COPY_DATA.
  const void* source_device_ptr;
  // Device-visible target pointer written by PM4 COPY_DATA.
  void* target_device_ptr;
  // Byte length copied from source to target; currently 4 or 8.
  uint8_t length;
} iree_hal_amdgpu_host_queue_pm4_copy_data_t;

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
  return queue->pm4_ib_slots &&
         iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_write_data(
             queue->vendor_packet_capabilities) &&
         (length == 4 || length == 8) &&
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
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id) {
  if (out_submission_id) *out_submission_id = 0;
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };

  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), profile_event_info, out_ready,
      &submission));
  if (!*out_ready) return iree_ok_status();

  bool did_emit = false;
  if (write_data->length == 4) {
    uint32_t value = 0;
    memcpy(&value, &write_data->value, sizeof(value));
    did_emit = iree_hal_amdgpu_pm4_ib_builder_emit_write_data32(
        &submission.pm4_ib_builder, write_data->target_device_ptr, value);
  } else {
    did_emit = iree_hal_amdgpu_pm4_ib_builder_emit_write_data64(
        &submission.pm4_ib_builder, write_data->target_device_ptr,
        write_data->value);
  }
  if (IREE_UNLIKELY(!did_emit)) {
    iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, &submission);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 WRITE_DATA payload does not fit IB slot");
  }
  uint64_t submission_epoch = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), profile_event_info, submission_flags,
      &submission, &submission_epoch));
  if (out_submission_id) *out_submission_id = submission_epoch;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_host_queue_prepare_pm4_copy_data(
    const iree_hal_amdgpu_host_queue_t* queue, const void* source_device_ptr,
    void* target_device_ptr, iree_device_size_t length,
    iree_hal_amdgpu_host_queue_pm4_copy_data_t* out_copy_data) {
  if (!queue->pm4_ib_slots ||
      !iree_hal_amdgpu_vendor_packet_capabilities_support_pm4_memory_copy_data(
          queue->vendor_packet_capabilities)) {
    return false;
  }
  switch (length) {
    case 4:
      if (!iree_host_ptr_has_alignment(source_device_ptr, sizeof(uint32_t)) ||
          !iree_host_ptr_has_alignment(target_device_ptr, sizeof(uint32_t))) {
        return false;
      }
      break;
    case 8:
      if (!iree_host_ptr_has_alignment(source_device_ptr, sizeof(uint64_t)) ||
          !iree_host_ptr_has_alignment(target_device_ptr, sizeof(uint64_t))) {
        return false;
      }
      break;
    default:
      return false;
  }

  out_copy_data->source_device_ptr = source_device_ptr;
  out_copy_data->target_device_ptr = target_device_ptr;
  out_copy_data->length = (uint8_t)length;
  return true;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_pm4_copy_data(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_hal_buffer_t* target_buffer,
    const iree_hal_amdgpu_host_queue_pm4_copy_data_t* copy_data,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id) {
  if (out_submission_id) *out_submission_id = 0;
  iree_hal_resource_t* operation_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };

  iree_hal_amdgpu_host_queue_pm4_ib_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
      queue, resolution, signal_semaphore_list,
      IREE_ARRAYSIZE(operation_resources), profile_event_info, out_ready,
      &submission));
  if (!*out_ready) return iree_ok_status();

  bool did_emit = false;
  if (copy_data->length == 4) {
    did_emit = iree_hal_amdgpu_pm4_ib_builder_emit_copy_data32(
        &submission.pm4_ib_builder, copy_data->source_device_ptr,
        copy_data->target_device_ptr);
  } else {
    did_emit = iree_hal_amdgpu_pm4_ib_builder_emit_copy_data64(
        &submission.pm4_ib_builder, copy_data->source_device_ptr,
        copy_data->target_device_ptr);
  }
  if (IREE_UNLIKELY(!did_emit)) {
    iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, &submission);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 COPY_DATA payload does not fit IB slot");
  }
  uint64_t submission_epoch = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), profile_event_info, submission_flags,
      &submission, &submission_epoch));
  if (out_submission_id) *out_submission_id = submission_epoch;
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
    iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
        iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
            IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL, length);
    uint64_t submission_id = 0;
    iree_status_t status = iree_hal_amdgpu_host_queue_submit_pm4_write_data(
        queue, resolution, signal_semaphore_list, target_buffer,
        &pm4_write_data, &profile_event_info, submission_flags, out_ready,
        &submission_id);
    if (iree_status_is_ok(status) && *out_ready) {
      iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
          queue, resolution, signal_semaphore_list, submission_id,
          &profile_event_info);
    }
    return status;
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_fill_dispatch(
      queue, target_device_ptr, length, pattern_bits, pattern_length,
      &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  uint64_t submission_id = 0;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
          IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL, length);
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), &profile_event_info,
      submission_flags, out_ready, &submission_id);
  if (iree_status_is_ok(status) && *out_ready) {
    iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
        queue, resolution, signal_semaphore_list, submission_id,
        &profile_event_info);
  }
  return status;
}

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "copy kernargs must fit in one kernarg ring block");

// Validates a queue_copy request and resolves device pointers. Overlapping
// ranges within the same buffer are rejected here because both PM4 COPY_DATA
// and the builtin copy kernels implement memcpy semantics, not memmove.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_copy_ranges(
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    const uint8_t** out_source_device_ptr, uint8_t** out_target_device_ptr) {
  *out_source_device_ptr = NULL;
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

  *out_source_device_ptr = source_device_ptr;
  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

// Prepares a copy dispatch packet and kernargs in stack-local storage without
// touching queue rings. All user-input validation must happen before this so
// the caller can avoid reserving AQL slots before the packet shape is known.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, const uint8_t* source_device_ptr,
    iree_device_size_t source_offset, uint8_t* target_device_ptr,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_device_buffer_copy_kernargs_t* out_kernargs) {
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
    iree_hal_profile_queue_event_type_t profile_event_type,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  const uint8_t* source_device_ptr = NULL;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_ranges(
      source_buffer, source_offset, target_buffer, target_offset, length, flags,
      &source_device_ptr, &target_device_ptr));

  iree_hal_amdgpu_host_queue_pm4_copy_data_t pm4_copy_data;
  if (iree_hal_amdgpu_host_queue_prepare_pm4_copy_data(
          queue, source_device_ptr, target_device_ptr, length,
          &pm4_copy_data)) {
    iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
        iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
            profile_event_type, length);
    uint64_t submission_id = 0;
    iree_status_t status = iree_hal_amdgpu_host_queue_submit_pm4_copy_data(
        queue, resolution, signal_semaphore_list, source_buffer, target_buffer,
        &pm4_copy_data, &profile_event_info, submission_flags, out_ready,
        &submission_id);
    if (iree_status_is_ok(status) && *out_ready) {
      iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
          queue, resolution, signal_semaphore_list, submission_id,
          &profile_event_info);
    }
    return status;
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
      queue, source_device_ptr, source_offset, target_device_ptr, target_offset,
      length, &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };
  uint64_t submission_id = 0;
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
          profile_event_type, length);
  iree_status_t status = iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), &profile_event_info,
      submission_flags, out_ready, &submission_id);
  if (iree_status_is_ok(status) && *out_ready) {
    iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
        queue, resolution, signal_semaphore_list, submission_id,
        &profile_event_info);
  }
  return status;
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

  const uint8_t* source_device_ptr = NULL;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_ranges(
      source_buffer, source_offset, target_buffer, target_offset, length, flags,
      &source_device_ptr, &target_device_ptr));

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
      queue, source_device_ptr, source_offset, target_device_ptr, target_offset,
      length, &dispatch_packet, &kernargs));

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
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
          IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY, length);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      /*kernarg_block_count=*/1,
      (iree_hal_amdgpu_profile_dispatch_event_reservation_t){0},
      &profile_event_info, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  memcpy(submission.kernel.kernargs.blocks->data, &kernargs, sizeof(kernargs));
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, &dispatch_packet,
          submission.kernel.kernargs.blocks->data,
          submission.dispatch_completion_signal);
  submission.minimum_acquire_scope = minimum_acquire_scope;
  submission.minimum_release_scope = minimum_release_scope;
  submission.kernel.pre_signal_action = pre_signal_action;
  const uint64_t submission_id =
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, &profile_event_info, submission_flags,
          &submission);
  iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
      queue, resolution, signal_semaphore_list, submission_id,
      &profile_event_info);
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
    iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
        iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
            IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE, length);
    uint64_t submission_id = 0;
    iree_status_t status = iree_hal_amdgpu_host_queue_submit_pm4_write_data(
        queue, resolution, signal_semaphore_list, target_buffer,
        &pm4_write_data, &profile_event_info, submission_flags, out_ready,
        &submission_id);
    if (iree_status_is_ok(status) && *out_ready) {
      iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
          queue, resolution, signal_semaphore_list, submission_id,
          &profile_event_info);
    }
    return status;
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
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_event_info =
      iree_hal_amdgpu_host_queue_make_blit_profile_event_info(
          IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE, length);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list,
      /*operation_resource_count=*/1, (uint32_t)kernarg_block_count,
      (iree_hal_amdgpu_profile_dispatch_event_reservation_t){0},
      &profile_event_info, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  uint8_t* staged_source_bytes =
      (uint8_t*)submission.kernel.kernargs.blocks + source_payload_offset;
  memcpy(submission.kernel.kernargs.blocks->data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)
       submission.kernel.kernargs.blocks->data)
      ->source_ptr = staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, source_length);
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, &dispatch_packet,
          submission.kernel.kernargs.blocks->data,
          submission.dispatch_completion_signal);

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  const uint64_t submission_id =
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          IREE_ARRAYSIZE(operation_resources), &profile_event_info,
          submission_flags, &submission);
  iree_hal_amdgpu_host_queue_record_submitted_blit_profile_event(
      queue, resolution, signal_semaphore_list, submission_id,
      &profile_event_info);
  return iree_ok_status();
}
