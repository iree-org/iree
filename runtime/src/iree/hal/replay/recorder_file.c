// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_file.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
#include <sys/stat.h>
#include <unistd.h>
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

//===----------------------------------------------------------------------===//
// File reference capture
//===----------------------------------------------------------------------===//

static void iree_hal_replay_recorder_file_capture_fd_identity(
    int fd, iree_hal_replay_file_object_payload_t* out_payload) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  struct stat file_stat;
  if (fstat(fd, &file_stat) == 0) {
    out_payload->file_device = (uint64_t)file_stat.st_dev;
    out_payload->file_inode = (uint64_t)file_stat.st_ino;
    out_payload->file_mtime_ns =
        ((uint64_t)file_stat.st_mtim.tv_sec * 1000000000ull) +
        (uint64_t)file_stat.st_mtim.tv_nsec;
  }
#else
  (void)fd;
  (void)out_payload;
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

static iree_string_view_t iree_hal_replay_recorder_file_capture_fd_path(
    int fd, iree_byte_span_t storage) {
#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  if (storage.data_length == 0) return iree_string_view_empty();
  char link_path[64];
  if (snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd) <= 0) {
    return iree_string_view_empty();
  }
  ssize_t length =
      readlink(link_path, (char*)storage.data, storage.data_length);
  if (length <= 0 || (iree_host_size_t)length >= storage.data_length) {
    return iree_string_view_empty();
  }
  return iree_make_string_view((const char*)storage.data,
                               (iree_host_size_t)length);
#else
  (void)fd;
  (void)storage;
  return iree_string_view_empty();
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
}

iree_status_t iree_hal_replay_recorder_file_make_object_payload(
    iree_io_file_handle_t* handle, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_hal_external_file_flags_t flags,
    iree_hal_file_t* base_file,
    iree_hal_replay_recorder_external_file_policy_t external_file_policy,
    iree_byte_span_t reference_storage,
    iree_hal_replay_file_object_payload_t* out_payload,
    iree_string_view_t* out_reference) {
  memset(out_payload, 0, sizeof(*out_payload));
  *out_reference = iree_string_view_empty();
  out_payload->queue_affinity = queue_affinity;
  out_payload->file_length = base_file ? iree_hal_file_length(base_file) : 0;
  out_payload->access = access;
  out_payload->flags = flags;
  out_payload->handle_type = iree_io_file_handle_type(handle);

  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  switch (primitive.type) {
    case IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION: {
      iree_byte_span_t host_allocation = primitive.value.host_allocation;
      out_payload->reference_type =
          IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES;
      out_payload->reference_length = host_allocation.data_length;
      if (!base_file) {
        out_payload->file_length = host_allocation.data_length;
      }
      *out_reference = iree_make_string_view((const char*)host_allocation.data,
                                             host_allocation.data_length);
      break;
    }
    case IREE_IO_FILE_HANDLE_TYPE_FD: {
      if (external_file_policy ==
          IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_FAIL) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "HAL replay recorder external file policy forbids fd-backed file "
            "references");
      } else if (external_file_policy !=
                 IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_REFERENCE) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HAL replay recorder external file policy %" PRIu32 " is unknown",
            external_file_policy);
      }
      iree_hal_replay_recorder_file_capture_fd_identity(primitive.value.fd,
                                                        out_payload);
      *out_reference = iree_hal_replay_recorder_file_capture_fd_path(
          primitive.value.fd, reference_storage);
      if (iree_string_view_is_empty(*out_reference)) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "HAL replay recorder could not capture a path for an fd-backed "
            "file");
      }
      out_payload->reference_type =
          IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH;
      out_payload->reference_length = out_reference->size;
      break;
    }
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "HAL replay recorder cannot capture imported file handles of type "
          "%" PRIu32,
          (uint32_t)primitive.type);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_file_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_file_t {
  // HAL file resource header for the recording wrapper file.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying file receiving forwarded HAL calls.
  iree_hal_file_t* base_file;
  // Session-local device object id associated with this file.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this file.
  iree_hal_replay_object_id_t file_id;
} iree_hal_replay_recorder_file_t;

static const iree_hal_file_vtable_t iree_hal_replay_recorder_file_vtable;

static bool iree_hal_replay_recorder_file_isa(iree_hal_file_t* base_file) {
  return iree_hal_resource_is(base_file, &iree_hal_replay_recorder_file_vtable);
}

static iree_hal_replay_recorder_file_t* iree_hal_replay_recorder_file_cast(
    iree_hal_file_t* base_file) {
  IREE_HAL_ASSERT_TYPE(base_file, &iree_hal_replay_recorder_file_vtable);
  return (iree_hal_replay_recorder_file_t*)base_file;
}

iree_status_t iree_hal_replay_recorder_file_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t file_id, iree_hal_file_t* base_file,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_file);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  if (iree_hal_replay_recorder_file_isa(base_file)) {
    iree_hal_file_retain(base_file);
    *out_file = base_file;
    return iree_ok_status();
  }

  iree_hal_replay_recorder_file_t* file = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file));
  memset(file, 0, sizeof(*file));

  iree_hal_resource_initialize(&iree_hal_replay_recorder_file_vtable,
                               &file->resource);
  file->host_allocator = host_allocator;
  file->recorder = recorder;
  iree_hal_replay_recorder_retain(file->recorder);
  file->base_file = base_file;
  iree_hal_file_retain(file->base_file);
  file->device_id = device_id;
  file->file_id = file_id;

  *out_file = (iree_hal_file_t*)file;
  return iree_ok_status();
}

iree_hal_file_t* iree_hal_replay_recorder_file_base_or_self(
    iree_hal_file_t* file) {
  return iree_hal_replay_recorder_file_isa(file)
             ? iree_hal_replay_recorder_file_cast(file)->base_file
             : file;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_file_id_or_none(
    iree_hal_file_t* file) {
  return file && iree_hal_replay_recorder_file_isa(file)
             ? iree_hal_replay_recorder_file_cast(file)->file_id
             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

static void iree_hal_replay_recorder_file_destroy(iree_hal_file_t* base_file) {
  iree_hal_replay_recorder_file_t* file =
      iree_hal_replay_recorder_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_release(file->base_file);
  iree_hal_replay_recorder_release(file->recorder);
  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_memory_access_t iree_hal_replay_recorder_file_allowed_access(
    iree_hal_file_t* base_file) {
  iree_hal_replay_recorder_file_t* file =
      iree_hal_replay_recorder_file_cast(base_file);
  return iree_hal_file_allowed_access(file->base_file);
}

static uint64_t iree_hal_replay_recorder_file_length(
    iree_hal_file_t* base_file) {
  iree_hal_replay_recorder_file_t* file =
      iree_hal_replay_recorder_file_cast(base_file);
  return iree_hal_file_length(file->base_file);
}

static iree_hal_buffer_t* iree_hal_replay_recorder_file_storage_buffer(
    iree_hal_file_t* base_file) {
  (void)base_file;
  return NULL;
}

static iree_async_file_t* iree_hal_replay_recorder_file_async_handle(
    iree_hal_file_t* base_file) {
  (void)base_file;
  return NULL;
}

static bool iree_hal_replay_recorder_file_supports_synchronous_io(
    iree_hal_file_t* base_file) {
  (void)base_file;
  return false;
}

static iree_status_t iree_hal_replay_recorder_file_read(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  (void)base_file;
  (void)file_offset;
  (void)buffer;
  (void)buffer_offset;
  (void)length;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "direct replay file reads are not recorded");
}

static iree_status_t iree_hal_replay_recorder_file_write(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  (void)base_file;
  (void)file_offset;
  (void)buffer;
  (void)buffer_offset;
  (void)length;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "direct replay file writes are not recorded");
}

static const iree_hal_file_vtable_t iree_hal_replay_recorder_file_vtable = {
    .destroy = iree_hal_replay_recorder_file_destroy,
    .allowed_access = iree_hal_replay_recorder_file_allowed_access,
    .length = iree_hal_replay_recorder_file_length,
    .storage_buffer = iree_hal_replay_recorder_file_storage_buffer,
    .async_handle = iree_hal_replay_recorder_file_async_handle,
    .supports_synchronous_io =
        iree_hal_replay_recorder_file_supports_synchronous_io,
    .read = iree_hal_replay_recorder_file_read,
    .write = iree_hal_replay_recorder_file_write,
};
