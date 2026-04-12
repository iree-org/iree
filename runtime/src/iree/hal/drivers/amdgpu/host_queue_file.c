// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_file.h"

static iree_status_t iree_hal_amdgpu_host_queue_file_barrier(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  return queue->vtable->execute(
      queue, wait_semaphore_list, signal_semaphore_list,
      /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_amdgpu_host_queue_cast_file_offset(
    uint64_t file_offset, iree_device_size_t length,
    iree_device_size_t* out_device_offset) {
  *out_device_offset = 0;
  if (IREE_UNLIKELY(file_offset > IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file offset %" PRIu64
                            " exceeds device size max %" PRIdsz,
                            file_offset, IREE_DEVICE_SIZE_MAX);
  }
  iree_device_size_t device_offset = (iree_device_size_t)file_offset;
  iree_device_size_t device_end = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_add(device_offset, length, &device_end))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file range overflows device size (offset=%" PRIdsz
                            ", length=%" PRIdsz ")",
                            device_offset, length);
  }
  *out_device_offset = device_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_file_range(
    iree_hal_file_t* file, const char* operation_name, uint64_t file_offset,
    iree_device_size_t length, iree_device_size_t* out_device_offset) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_cast_file_offset(
      file_offset, length, out_device_offset));
  const uint64_t file_length = iree_hal_file_length(file);
  if (IREE_UNLIKELY(file_offset > file_length ||
                    (uint64_t)length > file_length - file_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s range [%" PRIu64 ", %" PRIu64
                            ") exceeds file length %" PRIu64,
                            operation_name, file_offset,
                            file_offset + (uint64_t)length, file_length);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_read_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));
  if (IREE_UNLIKELY(flags != IREE_HAL_READ_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported read flags: 0x%" PRIx64, flags);
  }
  if (length == 0) {
    return iree_hal_amdgpu_host_queue_file_barrier(queue, wait_semaphore_list,
                                                   signal_semaphore_list);
  }

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  if (IREE_UNLIKELY(!storage_buffer)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_read currently requires a device-accessible file storage "
        "buffer");
  }
  iree_device_size_t source_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_file_range(
      source_file, "read", source_offset, length, &source_device_offset));
  return queue->vtable->copy(queue, wait_semaphore_list, signal_semaphore_list,
                             storage_buffer, source_device_offset,
                             target_buffer, target_offset, length,
                             IREE_HAL_COPY_FLAG_NONE);
}

iree_status_t iree_hal_amdgpu_host_queue_write_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));
  if (IREE_UNLIKELY(flags != IREE_HAL_WRITE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported write flags: 0x%" PRIx64, flags);
  }
  if (length == 0) {
    return iree_hal_amdgpu_host_queue_file_barrier(queue, wait_semaphore_list,
                                                   signal_semaphore_list);
  }

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  if (IREE_UNLIKELY(!storage_buffer)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_write currently requires a device-accessible file "
        "storage buffer");
  }
  iree_device_size_t target_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_file_range(
      target_file, "write", target_offset, length, &target_device_offset));
  return queue->vtable->copy(queue, wait_semaphore_list, signal_semaphore_list,
                             source_buffer, source_offset, storage_buffer,
                             target_device_offset, length,
                             IREE_HAL_COPY_FLAG_NONE);
}
