// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// TODO(benvanik): make these either compile-time configuration options so we
// can prune code paths or flags (somehow).

// When 1 a fast-path for importable memory will be used to avoid staging.
#if !defined(IREE_HAL_MEMORY_FILE_CAN_IMPORT)
#define IREE_HAL_MEMORY_FILE_CAN_IMPORT 1
#endif  // !IREE_HAL_MEMORY_FILE_CAN_IMPORT

//===----------------------------------------------------------------------===//
// iree_hal_memory_file_storage_t
//===----------------------------------------------------------------------===//

// Reference-counted storage for memory file contents.
// This allows both the memory file and any intermediate/staging buffers that
// may reference it to keep the underlying storage live and not create cycles.
typedef struct iree_hal_memory_file_storage_t {
  // Reference count for this storage instance.
  iree_atomic_ref_count_t ref_count;
  // Used to allocate this structure.
  iree_allocator_t host_allocator;
  // Base file handle, retained.
  iree_io_file_handle_t* handle;
  // Host memory contents, unowned.
  iree_byte_span_t contents;
} iree_hal_memory_file_storage_t;

static iree_status_t iree_hal_memory_file_storage_create(
    iree_io_file_handle_t* handle, iree_byte_span_t contents,
    iree_allocator_t host_allocator,
    iree_hal_memory_file_storage_t** out_storage) {
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_storage);
  *out_storage = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_memory_file_storage_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*storage),
                                (void**)&storage));
  iree_atomic_ref_count_init(&storage->ref_count);
  storage->host_allocator = host_allocator;
  storage->handle = handle;
  iree_io_file_handle_retain(storage->handle);
  storage->contents = contents;

  *out_storage = storage;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_memory_file_storage_destroy(
    iree_hal_memory_file_storage_t* storage) {
  IREE_ASSERT_ARGUMENT(storage);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = storage->host_allocator;

  iree_io_file_handle_release(storage->handle);

  iree_allocator_free(host_allocator, storage);

  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_memory_file_storage_retain(
    iree_hal_memory_file_storage_t* storage) {
  if (IREE_LIKELY(storage)) {
    iree_atomic_ref_count_inc(&storage->ref_count);
  }
}

static void iree_hal_memory_file_storage_release(
    iree_hal_memory_file_storage_t* storage) {
  if (IREE_LIKELY(storage) &&
      iree_atomic_ref_count_dec(&storage->ref_count) == 1) {
    iree_hal_memory_file_storage_destroy(storage);
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_memory_file_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_memory_file_t {
  iree_hal_resource_t resource;
  // Used to allocate this structure.
  iree_allocator_t host_allocator;
  // Allowed access bits.
  iree_hal_memory_access_t access;
  // Underlying storage container, retained.
  iree_hal_memory_file_storage_t* storage;
  // Optional imported buffer if it was possible to do so.
  // Not all implementations and not all buffers can be imported.
  iree_hal_buffer_t* imported_buffer;
} iree_hal_memory_file_t;

static const iree_hal_file_vtable_t iree_hal_memory_file_vtable;

static iree_hal_memory_file_t* iree_hal_memory_file_cast(
    iree_hal_file_t* IREE_RESTRICT base_value) {
  return (iree_hal_memory_file_t*)base_value;
}

static void iree_hal_memory_file_try_import_buffer(
    iree_hal_memory_file_t* file, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_byte_span_t contents,
    iree_hal_allocator_t* device_allocator);

IREE_API_EXPORT iree_status_t iree_hal_memory_file_wrap(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_allocator_t* device_allocator,
    iree_allocator_t host_allocator, iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // For now we only support host allocations but could open other types that
  // may be backed by memory if desired.
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "support for wrapping non-host-allocation file "
                            "handles with memory files is not yet implemented");
  }
  iree_byte_span_t contents = iree_io_file_handle_value(handle).host_allocation;

  // Note that iree_device_size_t (for device offsets/sizes) may be smaller than
  // iree_host_size_t (for host offsets/sizes) - if so we need to ensure the
  // bytes passed in will still fit in iree_device_size_t.
  if (contents.data_length > IREE_DEVICE_SIZE_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "device size too small to represent host contents");
  }

  // Allocate file handle; this just holds a reference to the storage and
  // (optionally) the imported buffer.
  iree_hal_memory_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*file), (void**)&file));
  iree_hal_resource_initialize(&iree_hal_memory_file_vtable, &file->resource);
  file->host_allocator = host_allocator;
  file->access = access;

  // Create the underlying storage container that we use to manage the storage
  // lifetime independently from the file lifetime.
  iree_status_t status = iree_hal_memory_file_storage_create(
      handle, contents, host_allocator, &file->storage);

#if !IREE_HAL_MEMORY_FILE_CAN_IMPORT
  // Importing disabled; useful for testing the slow path.
  device_allocator = NULL;
#endif  // IREE_HAL_MEMORY_FILE_CAN_IMPORT

  // Try importing the buffer as a host-local staging buffer.
  // This won't always succeed due to device, platform, HAL implementation, or
  // buffer limitations but if it does we can avoid staging ourselves during
  // streaming and directly read/write the memory via transfer commands.
  if (iree_status_is_ok(status) && device_allocator) {
    iree_hal_memory_file_try_import_buffer(file, queue_affinity, access,
                                           contents, device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_file = (iree_hal_file_t*)file;
  } else {
    iree_hal_file_release((iree_hal_file_t*)file);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_memory_file_destroy(
    iree_hal_file_t* IREE_RESTRICT base_file) {
  iree_hal_memory_file_t* file = iree_hal_memory_file_cast(base_file);
  iree_allocator_t host_allocator = file->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (file->imported_buffer) {
    iree_hal_buffer_release(file->imported_buffer);
    file->imported_buffer = NULL;
  }

  iree_hal_memory_file_storage_release(file->storage);

  iree_allocator_free(host_allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

// Releases the underlying file storage after the buffer using it is released.
static void iree_hal_memory_file_buffer_release(void* user_data,
                                                iree_hal_buffer_t* buffer) {
  iree_hal_memory_file_storage_release(
      (iree_hal_memory_file_storage_t*)user_data);
}

// Tries to import |contents| as a device-accessible HAL buffer.
// If this succeeds we can fast-path copies without needing to allocate any
// staging buffers and directly make use of DMA resources. If it fails we fall
// back to staging from host memory ourselves.
static void iree_hal_memory_file_try_import_buffer(
    iree_hal_memory_file_t* file, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_byte_span_t contents,
    iree_hal_allocator_t* device_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_params_t staging_buffer_params = {
      .access = access,
      .queue_affinity = queue_affinity,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST |
              IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      .usage = IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
               IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE |
               (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)
                    ? IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE
                    : 0) |
               (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)
                    ? IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET
                    : 0),
  };

  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = 0,
      .size = (iree_device_size_t)contents.data_length,
      .handle =
          {
              .host_allocation =
                  {
                      .ptr = contents.data,
                  },
          },
  };

  // NOTE: we make the buffer retain the underlying storage.
  // We have to handle the case where the import fails and we need to balance
  // the retain we did below.
  iree_hal_buffer_release_callback_t imported_release_callback = {
      .fn = iree_hal_memory_file_buffer_release,
      .user_data = file->storage,
  };
  iree_hal_memory_file_storage_retain(file->storage);
  iree_status_t status = iree_hal_allocator_import_buffer(
      device_allocator, staging_buffer_params, &external_buffer,
      imported_release_callback, &file->imported_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_memory_file_storage_release(file->storage);
  }

  IREE_TRACE({
    if (iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "import success");
    } else {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "import failure");
      IREE_TRACE_ZONE_APPEND_TEXT(
          z0, iree_status_code_string(iree_status_code(status)));
    }
  });

  IREE_TRACE_ZONE_END(z0);
  iree_status_ignore(status);
}

static const iree_hal_file_vtable_t iree_hal_memory_file_vtable = {
    .destroy = iree_hal_memory_file_destroy,
};

//===----------------------------------------------------------------------===//
// EXPERIMENTAL: synchronous file read/write API
//===----------------------------------------------------------------------===//
// This is incomplete and may not appear like this on the iree_hal_file_t
// vtable; this does work for memory files though.

IREE_API_EXPORT iree_hal_memory_access_t
iree_hal_file_allowed_access(iree_hal_file_t* base_file) {
  IREE_ASSERT_ARGUMENT(base_file);

  // EXPERIMENTAL: today only memory files. This should be on the file vtable
  // (if supported - not all implementations need to support it).
  iree_hal_memory_file_t* file = (iree_hal_memory_file_t*)base_file;

  return file->access;
}

IREE_API_EXPORT uint64_t iree_hal_file_length(iree_hal_file_t* base_file) {
  IREE_ASSERT_ARGUMENT(base_file);

  // EXPERIMENTAL: today only memory files. This should be on the file vtable
  // (if supported - not all implementations need to support it).
  iree_hal_memory_file_t* file = (iree_hal_memory_file_t*)base_file;

  return file->storage->contents.data_length;
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_file_storage_buffer(
    iree_hal_file_t* base_file) {
  IREE_ASSERT_ARGUMENT(base_file);

  // EXPERIMENTAL: today only memory files. This should be on the file vtable
  // (if supported - not all implementations need to support it).
  iree_hal_memory_file_t* file = (iree_hal_memory_file_t*)base_file;

  return file->imported_buffer;
}

IREE_API_EXPORT iree_status_t iree_hal_file_read(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(base_file);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, file_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  // EXPERIMENTAL: today only memory files. This should be on the file vtable
  // (if supported - not all implementations need to support it).
  iree_hal_memory_file_t* file = (iree_hal_memory_file_t*)base_file;

  // Copy from the file contents to the staging buffer.
  iree_byte_span_t file_contents = file->storage->contents;
  iree_status_t status = iree_hal_buffer_map_write(
      buffer, buffer_offset, file_contents.data + file_offset, length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_file_write(
    iree_hal_file_t* base_file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_ASSERT_ARGUMENT(base_file);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, file_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)buffer_offset);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)length);

  // EXPERIMENTAL: today only memory files. This should be on the file vtable
  // (if supported - not all implementations need to support it).
  iree_hal_memory_file_t* file = (iree_hal_memory_file_t*)base_file;

  // Copy from the staging buffer to the file contents.
  iree_byte_span_t file_contents = file->storage->contents;
  iree_status_t status = iree_hal_buffer_map_read(
      buffer, buffer_offset, file_contents.data + file_offset, length);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
