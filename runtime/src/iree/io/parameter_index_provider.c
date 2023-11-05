// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include "iree/hal/utils/file_cache.h"

// Limit concurrent operations to avoid blowing the stack. This is arbitrary and
// if we wanted to support more we could switch to using heap allocations or
// a growable stack scratchpad.
#define IREE_IO_PARAMETER_INDEX_PROVIDER_CONCURRENT_OPERATION_LIMIT 128

typedef struct iree_io_parameter_index_provider_t {
  iree_io_parameter_provider_t base;
  iree_allocator_t host_allocator;
  iree_host_size_t max_concurrent_operations;
  iree_string_view_t scope;
  iree_io_parameter_index_t* index;
  iree_hal_file_cache_t* file_cache;
} iree_io_parameter_index_provider_t;

static const iree_io_parameter_provider_vtable_t
    iree_io_parameter_index_provider_vtable;

static iree_io_parameter_index_provider_t*
iree_io_parameter_index_provider_cast(
    iree_io_parameter_provider_t* IREE_RESTRICT base_provider) {
  return (iree_io_parameter_index_provider_t*)base_provider;
}

IREE_API_EXPORT iree_status_t iree_io_parameter_index_provider_create(
    iree_string_view_t scope, iree_io_parameter_index_t* index,
    iree_host_size_t max_concurrent_operations, iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, scope.data, scope.size);

  max_concurrent_operations =
      iree_min(max_concurrent_operations,
               IREE_IO_PARAMETER_INDEX_PROVIDER_CONCURRENT_OPERATION_LIMIT);

  iree_io_parameter_index_provider_t* provider = NULL;
  iree_host_size_t total_size = sizeof(*provider) + scope.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&provider));
  iree_atomic_ref_count_init(&provider->base.ref_count);
  provider->base.vtable = &iree_io_parameter_index_provider_vtable;
  provider->host_allocator = host_allocator;
  provider->max_concurrent_operations = max_concurrent_operations;

  provider->scope = iree_make_string_view(
      (const char*)provider + sizeof(*provider), scope.size);
  memcpy((void*)provider->scope.data, scope.data, scope.size);

  provider->index = index;
  iree_io_parameter_index_retain(index);

  iree_status_t status =
      iree_hal_file_cache_create(host_allocator, &provider->file_cache);

  if (iree_status_is_ok(status)) {
    *out_provider = (iree_io_parameter_provider_t*)provider;
  } else {
    iree_io_parameter_provider_release((iree_io_parameter_provider_t*)provider);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_io_parameter_index_provider_destroy(
    iree_io_parameter_provider_t* IREE_RESTRICT base_provider) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  iree_allocator_t host_allocator = provider->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_file_cache_release(provider->file_cache);
  iree_io_parameter_index_release(provider->index);

  iree_allocator_free(host_allocator, provider);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_io_parameter_index_provider_notify(
    iree_io_parameter_provider_t* base_provider,
    iree_io_parameter_provider_signal_t signal) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  switch (signal) {
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_SUSPEND:
    case IREE_IO_PARAMETER_PROVIDER_SIGNAL_LOW_MEMORY:
      iree_hal_file_cache_trim(provider->file_cache);
      break;
    default:
      break;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static bool iree_io_parameter_index_provider_query_support(
    iree_io_parameter_provider_t* base_provider, iree_string_view_t scope) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  return iree_string_view_equal(scope, provider->scope);
}

// Resolves a parameter with |key| for use on the given |device|.
// Returns the entry containing the parameter metadata and a retained
// HAL file that stores it (must be released by the caller).
static iree_status_t iree_io_parameter_index_provider_resolve(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity, iree_string_view_t scope,
    iree_string_view_t key, iree_hal_memory_access_t access,
    const iree_io_parameter_index_entry_t** out_entry,
    iree_hal_file_t** out_file) {
  IREE_ASSERT_ARGUMENT(out_entry);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_entry = NULL;
  *out_file = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup the parameter in the index.
  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_lookup(provider->index, key, &entry));

  // Get (or import) the HAL file backing the entry.
  // NOTE: file is retained!
  iree_hal_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_file_cache_lookup(
              provider->file_cache, device, queue_affinity, access,
              entry->file_handle, IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &file));

  *out_entry = entry;
  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Validates that the range specified by [offset, offset+length) is in bounds.
static iree_status_t iree_io_validate_parameter_range(
    iree_hal_memory_access_t required_access,
    const iree_io_parameter_index_entry_t* entry, uint64_t offset,
    uint64_t length) {
  iree_hal_memory_access_t allowed_access = IREE_HAL_MEMORY_ACCESS_NONE;
  if (iree_all_bits_set(iree_io_file_handle_access(entry->file_handle),
                        IREE_IO_FILE_ACCESS_READ)) {
    allowed_access |= IREE_HAL_MEMORY_ACCESS_READ;
  }
  if (iree_all_bits_set(iree_io_file_handle_access(entry->file_handle),
                        IREE_IO_FILE_ACCESS_WRITE)) {
    allowed_access |=
        IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD;
  }
  if (!iree_all_bits_set(allowed_access, required_access)) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "access denied to parameter backing file");
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_memory_access_str =
        iree_hal_memory_access_format(allowed_access, &temp0);
    iree_string_view_t required_memory_access_str =
        iree_hal_memory_access_format(required_access, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "parameter storage does not support the requested access "
        "type; parameter allows %.*s, operation requires %.*s",
        (int)allowed_memory_access_str.size, allowed_memory_access_str.data,
        (int)required_memory_access_str.size, required_memory_access_str.data);
#else
    return iree_status_from_code(IREE_STATUS_PERMISSION_DENIED);
#endif  // IREE_STATUS_MODE
  }

  if (offset + length > entry->length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter range out of bounds (offset=%" PRIu64
                            ", length=%" PRIu64 ", size=%" PRIu64 ")",
                            offset, length, entry->length);
  }

  return iree_ok_status();
}

static void iree_io_file_handle_buffer_release(void* user_data,
                                               iree_hal_buffer_t* buffer) {
  iree_io_file_handle_release((iree_io_file_handle_t*)user_data);
}

static iree_status_t iree_io_parameter_index_provider_load(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_params_t target_params,
    iree_device_size_t length,
    iree_hal_buffer_t** IREE_RESTRICT out_target_buffer) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup the parameter metadata and get its backing file.
  const iree_io_parameter_index_entry_t* source_entry = NULL;
  iree_hal_file_t* source_file = NULL;  // retained
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_provider_resolve(
              provider, device, queue_affinity, source_scope, source_key,
              target_params.access, &source_entry, &source_file));

  // Validate the parameter range is in-bounds.
  iree_status_t status = iree_io_validate_parameter_range(
      target_params.access, source_entry, source_offset, length);

  // Try first to reuse the file backing store directly as a buffer. This only
  // works with specific file types and with specific target usage. The most
  // common cases for this are when using parameters as staging sources (so host
  // memory is ok) or on unified memory systems (where host memory is device
  // memory) and the file was originally mapped. We could extend the conditions
  // in which we use this with some better file handle helpers that allow us to
  // map files that we already have open via other mechanisms (FILE, fd, etc).
  iree_hal_buffer_t* target_buffer = NULL;
  if (iree_status_is_ok(status) &&
      iree_io_file_handle_type(source_entry->file_handle) ==
          IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    iree_byte_span_t host_allocation =
        iree_io_file_handle_primitive(source_entry->file_handle)
            .value.host_allocation;
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = host_allocation.data_length,
        .handle =
            {
                .host_allocation =
                    {
                        .ptr = host_allocation.data,
                    },
            },
    };
    iree_hal_buffer_release_callback_t release_callback = {
        .fn = iree_io_file_handle_buffer_release,
        .user_data = source_entry->file_handle,
    };
    iree_io_file_handle_retain(source_entry->file_handle);
    iree_status_t import_status = iree_hal_allocator_import_buffer(
        iree_hal_device_allocator(device), target_params, &external_buffer,
        release_callback, &target_buffer);
    if (iree_status_is_ok(import_status)) {
      // Import succeeded - issue a barrier to preserve the async timeline.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "import succeeded");
      status = iree_hal_device_queue_barrier(
          device, queue_affinity, wait_semaphore_list, signal_semaphore_list);
    } else {
      // Failed to import - that's ok as we'll just do the full allocate + read.
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "import failed");
      import_status = iree_status_ignore(import_status);
      iree_io_file_handle_release(source_entry->file_handle);
    }
  }

  if (!target_buffer) {
    // Temporary semaphore for chaining the allocation and read.
    iree_hal_semaphore_t* temporary_semaphore = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_create(device, 0ull, &temporary_semaphore);
    }
    uint64_t temporary_semaphore_value = 1ull;
    const iree_hal_semaphore_list_t alloca_semaphore_list = {
        .count = 1,
        .semaphores = &temporary_semaphore,
        .payload_values = &temporary_semaphore_value,
    };

    // Allocate the target buffer.
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_alloca(
          device, queue_affinity, wait_semaphore_list, alloca_semaphore_list,
          IREE_HAL_ALLOCATOR_POOL_DEFAULT, target_params, length,
          &target_buffer);
    }

    // Queue the file read into the target buffer.
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_read(
          device, queue_affinity, alloca_semaphore_list, signal_semaphore_list,
          source_file, source_entry->offset + source_offset, target_buffer, 0,
          length, 0);
    }

    iree_hal_semaphore_release(temporary_semaphore);
  }

  iree_hal_file_release(source_file);
  if (iree_status_is_ok(status)) {
    IREE_ASSERT_NE(target_buffer, NULL);
    *out_target_buffer = target_buffer;
  } else {
    iree_hal_buffer_release(target_buffer);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_read(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup the parameter metadata and get its backing file.
  const iree_io_parameter_index_entry_t* source_entry = NULL;
  iree_hal_file_t* source_file = NULL;  // retained
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_provider_resolve(
              provider, device, queue_affinity, source_scope, source_key,
              IREE_HAL_MEMORY_ACCESS_READ, &source_entry, &source_file));

  // Validate the parameter range is in-bounds.
  iree_status_t status = iree_io_validate_parameter_range(
      IREE_HAL_MEMORY_ACCESS_READ, source_entry, source_offset, length);

  // Queue the file read into the target buffer.
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_file, source_entry->offset + source_offset, target_buffer,
        target_offset, length, 0);
  }

  iree_hal_file_release(source_file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_provider_write(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_string_view_t target_scope, iree_string_view_t target_key,
    uint64_t target_offset, iree_device_size_t length) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup the parameter metadata and get its backing file.
  const iree_io_parameter_index_entry_t* target_entry = NULL;
  iree_hal_file_t* target_file = NULL;  // retained
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_io_parameter_index_provider_resolve(
              provider, device, queue_affinity, target_scope, target_key,
              IREE_HAL_MEMORY_ACCESS_READ, &target_entry, &target_file));

  // Validate the parameter range is in-bounds.
  iree_status_t status = iree_io_validate_parameter_range(
      IREE_HAL_MEMORY_ACCESS_WRITE, target_entry, target_offset, length);

  // Queue the file write from the source buffer.
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        source_buffer, source_offset, target_file,
        target_entry->offset + target_offset, length, 0);
  }

  iree_hal_file_release(target_file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef iree_status_t(IREE_API_PTR* iree_io_parameter_index_file_operation_t)(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t flags);

// Returns the index of the smallest value in |values|.
// Linear scan as the number of values is expected to be small.
static iree_host_size_t iree_io_select_timeline_bucket(iree_host_size_t count,
                                                       const uint64_t* values) {
  IREE_ASSERT_GT(count, 0);
  uint64_t smallest_value = values[0];
  iree_host_size_t smallest_index = 0;
  for (iree_host_size_t i = 1; i < count; ++i) {
    if (values[i] < smallest_value) {
      smallest_value = values[i];
      smallest_index = i;
    }
  }
  return smallest_index;
}

static iree_status_t iree_io_parameter_index_provider_gather_scatter(
    iree_io_parameter_index_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t scope, iree_hal_buffer_t* buffer, iree_host_size_t count,
    iree_io_parameter_enumerator_t enumerator, iree_hal_memory_access_t access,
    iree_io_parameter_index_file_operation_t operation) {
  // Decide how many operations we'll keep in-flight at a time. Each concurrent
  // stream of operations requires its own semaphore.
  //
  // NOTE: we expect count == 0 and count == 1 to have been handled by callers
  // and assume that if we've hit this method we're doing something significant
  // and it's worth it to do all this.
  const iree_host_size_t concurrency =
      iree_min(count, provider->max_concurrent_operations);

  // Distribute operations over each timeline based on how much
  // I/O is done. It's possible for pathologically bad latency if there are
  // large and small operations interleaved as all large operations may end up
  // serialized on one timeline and all small ones on the other.
  // We distribute by tracking the total bytes outstanding on each timeline and
  // always placing the next operation on the one with the fewest. This assumes
  // that all I/O happens at roughly the same speed but if parameters come from
  // different files on different devices that may not be the case. It's better
  // than doing nothing, though.
  uint64_t* timeline_bytes_outstanding =
      (uint64_t*)iree_alloca(concurrency * sizeof(uint64_t));
  memset(timeline_bytes_outstanding, 0,
         concurrency * sizeof(*timeline_bytes_outstanding));

  // Allocate one semaphore per concurrent timeline.
  IREE_TRACE_ZONE_BEGIN_NAMED(
      z_init, "iree_io_parameter_index_provider_semaphore_pool_initialize");
  iree_hal_semaphore_t** timeline_semaphores =
      (iree_hal_semaphore_t**)iree_alloca(concurrency *
                                          sizeof(iree_hal_semaphore_t*));
  uint64_t* timeline_values =
      (uint64_t*)iree_alloca(concurrency * sizeof(uint64_t));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < concurrency; ++i) {
    timeline_values[i] = 0ull;
    status = iree_hal_semaphore_create(device, timeline_values[i],
                                       &timeline_semaphores[i]);
    if (!iree_status_is_ok(status)) break;
  }
  IREE_TRACE_ZONE_END(z_init);

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < count; ++i) {
      IREE_TRACE_ZONE_BEGIN(z_entry);
      IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, i);

      // Fetch the next parameter to copy and its buffer range.
      iree_string_view_t key;
      iree_io_parameter_span_t span;
      status = enumerator.fn(enumerator.user_data, i, &key, &span);

      // Lookup the parameter metadata and get its backing file.
      const iree_io_parameter_index_entry_t* entry = NULL;
      iree_hal_file_t* file = NULL;  // retained
      if (iree_status_is_ok(status)) {
        IREE_TRACE_ZONE_APPEND_TEXT(z_entry, key.data, key.size);
        status = iree_io_parameter_index_provider_resolve(
            provider, device, queue_affinity, scope, key, access, &entry,
            &file);
      }

      // Validate the parameter range is in-bounds.
      if (iree_status_is_ok(status)) {
        status = iree_io_validate_parameter_range(
            access, entry, span.parameter_offset, span.length);
      }

      // Queue the file operation.
      if (iree_status_is_ok(status)) {
        // Operations are tracked on as many timelines as there is concurrency.
        // We distribute operations onto timelines based on which has the fewest
        // outstanding I/O bytes.
        const iree_host_size_t timeline_index = iree_io_select_timeline_bucket(
            concurrency, timeline_bytes_outstanding);
        timeline_bytes_outstanding[timeline_index] += span.length;
        iree_hal_semaphore_t* timeline_semaphore =
            timeline_semaphores[timeline_index];
        uint64_t previous_timeline_value = timeline_values[timeline_index];
        uint64_t next_timeline_value = ++timeline_values[timeline_index];
        IREE_TRACE_ZONE_APPEND_VALUE_I64(z_entry, (uint64_t)timeline_index);

        // The first wave of operations all wait on the provided wait
        // semaphores. All others wait on their own internal concurrent
        // timelines.
        iree_hal_semaphore_list_t entry_wait_semaphore_list;
        if (i < concurrency) {
          entry_wait_semaphore_list = wait_semaphore_list;
        } else {
          entry_wait_semaphore_list = (iree_hal_semaphore_list_t){
              .count = 1,
              .semaphores = &timeline_semaphore,
              .payload_values = &previous_timeline_value,
          };
        }

        // All operations signal their concurrency timelines and we'll put a
        // barrier at the end so that we can join them all.
        iree_hal_semaphore_list_t entry_signal_semaphore_list = {
            .count = 1,
            .semaphores = &timeline_semaphore,
            .payload_values = &next_timeline_value,
        };

        // Perform the operation.
        status = operation(device, queue_affinity, entry_wait_semaphore_list,
                           entry_signal_semaphore_list, file,
                           entry->offset + span.parameter_offset, buffer,
                           span.buffer_offset, span.length, 0);
      }

      iree_hal_file_release(file);

      IREE_TRACE_ZONE_END(z_entry);
      if (!iree_status_is_ok(status)) break;
    }
  }

  // Join all concurrent timelines and continue the user-provided timeline.
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_list_t join_semaphore_list = {
        .count = concurrency,
        .semaphores = timeline_semaphores,
        .payload_values = timeline_values,
    };
    status = iree_hal_device_queue_barrier(
        device, queue_affinity, join_semaphore_list, signal_semaphore_list);
  }

  // Release temporary semaphores.
  IREE_TRACE_ZONE_BEGIN_NAMED(
      z_deinit, "iree_io_parameter_index_provider_semaphore_pool_deinitialize");
  for (iree_host_size_t i = 0; i < concurrency; ++i) {
    iree_hal_semaphore_release(timeline_semaphores[i]);
  }
  IREE_TRACE_ZONE_END(z_deinit);

  return status;
}

static iree_status_t iree_io_parameter_index_file_read(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t flags) {
  return iree_hal_device_queue_read(device, queue_affinity, wait_semaphore_list,
                                    signal_semaphore_list, file, file_offset,
                                    buffer, buffer_offset, length, flags);
}

static iree_status_t iree_io_parameter_index_provider_gather(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_io_parameter_index_provider_gather_scatter(
      provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, source_scope, target_buffer, count, enumerator,
      IREE_HAL_MEMORY_ACCESS_READ, iree_io_parameter_index_file_read);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_io_parameter_index_file_write(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* file, uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t flags) {
  return iree_hal_device_queue_write(
      device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      buffer, buffer_offset, file, file_offset, length, flags);
}

static iree_status_t iree_io_parameter_index_provider_scatter(
    iree_io_parameter_provider_t* base_provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  iree_io_parameter_index_provider_t* provider =
      iree_io_parameter_index_provider_cast(base_provider);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_io_parameter_index_provider_gather_scatter(
      provider, device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, target_scope, source_buffer, count, enumerator,
      IREE_HAL_MEMORY_ACCESS_WRITE, iree_io_parameter_index_file_write);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_io_parameter_provider_vtable_t
    iree_io_parameter_index_provider_vtable = {
        .destroy = iree_io_parameter_index_provider_destroy,
        .notify = iree_io_parameter_index_provider_notify,
        .query_support = iree_io_parameter_index_provider_query_support,
        .load = iree_io_parameter_index_provider_load,
        .read = iree_io_parameter_index_provider_read,
        .write = iree_io_parameter_index_provider_write,
        .gather = iree_io_parameter_index_provider_gather,
        .scatter = iree_io_parameter_index_provider_scatter,
};
