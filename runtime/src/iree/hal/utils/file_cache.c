// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Excepthalns.
// See https://llvm.org/LICENSE.txt for license informathaln.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-excepthaln

#include "iree/hal/utils/file_cache.h"

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"

typedef struct iree_hal_file_cache_entry_t {
  iree_io_file_handle_t* handle;
  iree_hal_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
  iree_hal_memory_access_t access;
  iree_hal_file_t* file;
} iree_hal_file_cache_entry_t;

struct iree_hal_file_cache_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Guards mutation of the entries list.
  // NOTE: this does not guard the entries themselves as we assume they are
  // immutable (today).
  iree_slim_mutex_t mutex;

  // Total capacity of the entries list in elements.
  iree_host_size_t entry_capacity;
  // Currently used entry count in elements.
  iree_host_size_t entry_count;
  // Dense list of entries in the cache. Grows as needed.
  iree_hal_file_cache_entry_t** entries;
};

IREE_API_EXPORT iree_status_t iree_hal_file_cache_create(
    iree_allocator_t host_allocator, iree_hal_file_cache_t** out_file_cache) {
  IREE_ASSERT_ARGUMENT(out_file_cache);
  *out_file_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_cache_t* file_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*file_cache),
                                (void**)&file_cache));
  iree_atomic_ref_count_init(&file_cache->ref_count);
  file_cache->host_allocator = host_allocator;

  iree_slim_mutex_initialize(&file_cache->mutex);

  // Grown on first use. We could allocate a bit of inline storage or take an
  // optional initial capacity for callers that know.
  file_cache->entry_capacity = 0;
  file_cache->entry_count = 0;
  file_cache->entries = NULL;

  *out_file_cache = file_cache;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_file_cache_destroy(iree_hal_file_cache_t* file_cache) {
  IREE_ASSERT_ARGUMENT(file_cache);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = file_cache->host_allocator;

  iree_hal_file_cache_trim(file_cache);

  iree_slim_mutex_deinitialize(&file_cache->mutex);

  iree_allocator_free(host_allocator, file_cache);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_hal_file_cache_retain(
    iree_hal_file_cache_t* file_cache) {
  if (IREE_LIKELY(file_cache)) {
    iree_atomic_ref_count_inc(&file_cache->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_file_cache_release(
    iree_hal_file_cache_t* file_cache) {
  if (IREE_LIKELY(file_cache) &&
      iree_atomic_ref_count_dec(&file_cache->ref_count) == 1) {
    iree_hal_file_cache_destroy(file_cache);
  }
}

IREE_API_EXPORT void iree_hal_file_cache_trim(
    iree_hal_file_cache_t* file_cache) {
  IREE_ASSERT_ARGUMENT(file_cache);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t host_allocator = file_cache->host_allocator;

  iree_slim_mutex_lock(&file_cache->mutex);

  for (iree_host_size_t i = 0; i < file_cache->entry_count; ++i) {
    iree_hal_file_cache_entry_t* entry = file_cache->entries[i];
    iree_hal_file_release(entry->file);
    iree_hal_device_release(entry->device);
    iree_io_file_handle_release(entry->handle);
    iree_allocator_free(host_allocator, entry);
  }
  file_cache->entry_count = 0;
  if (file_cache->entries) {
    iree_allocator_free(host_allocator, file_cache->entries);
    file_cache->entries = NULL;
    file_cache->entry_capacity = 0;
  }

  iree_slim_mutex_unlock(&file_cache->mutex);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_file_cache_reserve_unsafe(
    iree_hal_file_cache_t* file_cache, iree_host_size_t new_capacity) {
  IREE_ASSERT_ARGUMENT(file_cache);
  if (new_capacity < file_cache->entry_capacity) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, new_capacity);

  iree_hal_file_cache_entry_t** new_entries = file_cache->entries;
  iree_status_t status = iree_allocator_realloc(
      file_cache->host_allocator, new_capacity * sizeof(file_cache->entries[0]),
      (void**)&new_entries);
  if (iree_status_is_ok(status)) {
    file_cache->entry_capacity = new_capacity;
    file_cache->entries = new_entries;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_file_cache_insert_unsafe(
    iree_hal_file_cache_t* file_cache, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_file_t* file) {
  // Ensure there's space to grow the cache table.
  if (file_cache->entry_count == file_cache->entry_capacity) {
    IREE_RETURN_IF_ERROR(iree_hal_file_cache_reserve_unsafe(
        file_cache, iree_max(16u, file_cache->entry_capacity * 2)));
  }

  // Allocate the cache entry and retain all resources.
  iree_hal_file_cache_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(file_cache->host_allocator,
                                             sizeof(*entry), (void**)&entry));
  entry->handle = handle;
  iree_io_file_handle_retain(entry->handle);
  entry->device = device;
  iree_hal_device_retain(entry->device);
  entry->queue_affinity = queue_affinity;
  entry->access = access;
  entry->file = file;
  iree_hal_file_retain(entry->file);

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_file_cache_lookup(
    iree_hal_file_cache_t* file_cache, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access,
    iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(file_cache);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&file_cache->mutex);

  // Scan the cache to see if we have an already imported file we can use.
  for (iree_host_size_t i = 0; i < file_cache->entry_count; ++i) {
    iree_hal_file_cache_entry_t* entry = file_cache->entries[i];
    if (entry->device == device &&
        iree_all_bits_set(entry->queue_affinity, queue_affinity) &&
        iree_all_bits_set(entry->access, access) && entry->handle == handle) {
      iree_hal_file_t* file = entry->file;
      iree_hal_file_retain(file);
      iree_slim_mutex_unlock(&file_cache->mutex);
      *out_file = file;
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  // Import the file. This could be slow and ideally we'd not hold the mutex
  // such that other files can still be accessed through the cache but (today)
  // it's unexpected that file I/O initialization is a hot path.
  iree_hal_file_t* file = NULL;
  iree_status_t status = iree_hal_file_import(device, queue_affinity, access,
                                              handle, flags, &file);

  // Insert into cache as an append.
  // If we support removal we'll compact the entries list at the time an entry
  // is removed to keep this simple.
  if (iree_status_is_ok(status)) {
    status = iree_hal_file_cache_insert_unsafe(
        file_cache, device, queue_affinity, access, handle, file);
  }

  iree_slim_mutex_unlock(&file_cache->mutex);
  if (iree_status_is_ok(status)) {
    *out_file = file;
  } else {
    iree_hal_file_release(file);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
