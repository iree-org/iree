// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/proactor_pool.h"

#include <stdio.h>

#include "iree/async/proactor_platform.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

//===----------------------------------------------------------------------===//
// iree_async_proactor_pool_t
//===----------------------------------------------------------------------===//

// Per-node entry in the pool.
typedef struct iree_async_proactor_pool_entry_t {
  // NUMA node ID for this entry, or UINT32_MAX if unspecified.
  uint32_t node_id;
  // Proactor instance, created on first access (retained by the pool).
  iree_async_proactor_t* proactor;
  // Thread driving the proactor's poll loop, created with proactor (retained).
  iree_async_proactor_thread_t* thread;
} iree_async_proactor_pool_entry_t;

struct iree_async_proactor_pool_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t allocator;
  // Options stored for deferred proactor/thread creation.
  iree_async_proactor_pool_options_t options;
  // Mutex protecting lazy initialization of entries.
  iree_slim_mutex_t mutex;
  iree_host_size_t count;
  iree_async_proactor_pool_entry_t entries[];
};

static void iree_async_proactor_pool_destroy(iree_async_proactor_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)pool->count);

  // Request all threads to stop first (non-blocking), then join them.
  // Requesting all stops before joining any avoids serializing the shutdown
  // latency across N threads.
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    if (pool->entries[i].thread) {
      iree_async_proactor_thread_request_stop(pool->entries[i].thread);
    }
  }
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    if (pool->entries[i].thread) {
      iree_status_ignore(iree_async_proactor_thread_join(
          pool->entries[i].thread, IREE_DURATION_INFINITE));
      iree_async_proactor_thread_release(pool->entries[i].thread);
      pool->entries[i].thread = NULL;
    }
    if (pool->entries[i].proactor) {
      iree_async_proactor_release(pool->entries[i].proactor);
      pool->entries[i].proactor = NULL;
    }
  }

  iree_slim_mutex_deinitialize(&pool->mutex);
  iree_allocator_t allocator = pool->allocator;
  iree_allocator_free(allocator, pool);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_async_proactor_pool_create(
    iree_host_size_t node_count, const uint32_t* node_ids,
    iree_async_proactor_pool_options_t options, iree_allocator_t allocator,
    iree_async_proactor_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)node_count);
  *out_pool = NULL;

  if (node_count == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "node_count must be >= 1");
  }

  // Allocate pool with trailing entry array.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              iree_sizeof_struct(iree_async_proactor_pool_t), &total_size,
              IREE_STRUCT_FIELD(node_count, iree_async_proactor_pool_entry_t,
                                /*out_offset=*/NULL)));
  iree_async_proactor_pool_t* pool = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&pool));
  memset(pool, 0, total_size);

  iree_atomic_ref_count_init(&pool->ref_count);
  pool->allocator = allocator;
  pool->options = options;
  iree_slim_mutex_initialize(&pool->mutex);
  pool->count = node_count;

  // Initialize node IDs. Proactors and threads are created on-demand when
  // pool_get or pool_get_for_node is first called for each entry.
  for (iree_host_size_t i = 0; i < node_count; ++i) {
    pool->entries[i].node_id = node_ids ? node_ids[i] : UINT32_MAX;
  }

  *out_pool = pool;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_async_proactor_pool_retain(iree_async_proactor_pool_t* pool) {
  if (IREE_LIKELY(pool)) {
    iree_atomic_ref_count_inc(&pool->ref_count);
  }
}

void iree_async_proactor_pool_release(iree_async_proactor_pool_t* pool) {
  if (IREE_LIKELY(pool) && iree_atomic_ref_count_dec(&pool->ref_count) == 1) {
    iree_async_proactor_pool_destroy(pool);
  }
}

iree_host_size_t iree_async_proactor_pool_count(
    const iree_async_proactor_pool_t* pool) {
  IREE_ASSERT_ARGUMENT(pool);
  return pool->count;
}

// Creates the proactor and thread for |entry| if not already initialized.
// Must be called with pool->mutex held.
static iree_status_t iree_async_proactor_pool_ensure_entry_locked(
    iree_async_proactor_pool_t* pool, iree_host_size_t index) {
  iree_async_proactor_pool_entry_t* entry = &pool->entries[index];
  if (entry->proactor) return iree_ok_status();

  // Set a per-node debug name on the proactor.
  iree_async_proactor_options_t proactor_options =
      pool->options.proactor_options;
  char name_buffer[32];
  if (iree_string_view_is_empty(proactor_options.debug_name)) {
    snprintf(name_buffer, sizeof(name_buffer), "proactor-%zu", index);
    proactor_options.debug_name =
        iree_make_string_view(name_buffer, strlen(name_buffer));
  }

  IREE_RETURN_IF_ERROR(iree_async_proactor_create_platform(
      proactor_options, pool->allocator, &entry->proactor));

  // Configure thread with NUMA affinity if node ID is specified.
  iree_async_proactor_thread_options_t thread_options =
      iree_async_proactor_thread_options_default();
  if (entry->node_id != UINT32_MAX) {
    iree_thread_affinity_set_group_any(entry->node_id,
                                       &thread_options.affinity);
  }
  thread_options.error_callback = pool->options.error_callback;

  char thread_name_buffer[16];
  if (entry->node_id != UINT32_MAX) {
    snprintf(thread_name_buffer, sizeof(thread_name_buffer), "iree-pro-%u",
             entry->node_id);
  } else {
    snprintf(thread_name_buffer, sizeof(thread_name_buffer), "iree-pro-%zu",
             index);
  }
  thread_options.debug_name =
      iree_make_string_view(thread_name_buffer, strlen(thread_name_buffer));

  iree_status_t status = iree_async_proactor_thread_create(
      entry->proactor, thread_options, pool->allocator, &entry->thread);
  if (!iree_status_is_ok(status)) {
    iree_async_proactor_release(entry->proactor);
    entry->proactor = NULL;
  }
  return status;
}

iree_status_t iree_async_proactor_pool_get(
    iree_async_proactor_pool_t* pool, iree_host_size_t index,
    iree_async_proactor_t** out_proactor) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;
  if (IREE_UNLIKELY(index >= pool->count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "proactor pool index %" PRIhsz
                            " out of range (pool has %" PRIhsz " entries)",
                            index, pool->count);
  }
  iree_slim_mutex_lock(&pool->mutex);
  iree_status_t status =
      iree_async_proactor_pool_ensure_entry_locked(pool, index);
  if (iree_status_is_ok(status)) {
    *out_proactor = pool->entries[index].proactor;
  }
  iree_slim_mutex_unlock(&pool->mutex);
  return status;
}

uint32_t iree_async_proactor_pool_node_id(
    const iree_async_proactor_pool_t* pool, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(pool);
  if (IREE_UNLIKELY(index >= pool->count)) return UINT32_MAX;
  return pool->entries[index].node_id;
}

iree_status_t iree_async_proactor_pool_get_for_node(
    iree_async_proactor_pool_t* pool, uint32_t node_id,
    iree_async_proactor_t** out_proactor) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;
  // Linear scan for the matching node. N is small (1-8 NUMA nodes).
  iree_host_size_t index = 0;  // Fallback to first entry if no match.
  for (iree_host_size_t i = 0; i < pool->count; ++i) {
    if (pool->entries[i].node_id == node_id) {
      index = i;
      break;
    }
  }
  return iree_async_proactor_pool_get(pool, index, out_proactor);
}
