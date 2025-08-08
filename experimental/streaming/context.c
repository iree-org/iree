// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "experimental/streaming/internal.h"
#include "experimental/streaming/util/buffer_table.h"

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Thread-local current context.
static iree_thread_local iree_hal_streaming_context_t*
    iree_hal_streaming_current_context = NULL;

typedef struct iree_hal_streaming_context_stack_t {
  iree_hal_streaming_context_t** contexts;
  iree_host_size_t depth;
  iree_host_size_t capacity;
} iree_hal_streaming_context_stack_t;

// Thread-local context stack for push/pop.
static iree_thread_local iree_hal_streaming_context_stack_t
    iree_hal_streaming_context_stack = {
        .contexts = NULL,
        .depth = 0,
        .capacity = 0,
};

//===----------------------------------------------------------------------===//
// Context management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context);

iree_status_t iree_hal_streaming_context_create(
    iree_hal_streaming_device_t* device_entry,
    iree_hal_streaming_context_flags_t flags, iree_allocator_t host_allocator,
    iree_hal_streaming_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device_entry);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_context_t* context = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*context),
                                (void**)&context));

  // Initialize context.
  iree_atomic_ref_count_init(&context->ref_count);
  context->device = device_entry->hal_device;
  context->device_ordinal = device_entry->ordinal;
  context->device_entry = device_entry;
  context->device_allocator =
      iree_hal_device_allocator(device_entry->hal_device);
  context->executable_cache = NULL;
  context->flags = flags;
  context->default_stream = NULL;
  context->peer_contexts = NULL;
  context->peer_count = 0;
  context->peer_capacity = 0;
  context->buffer_table = NULL;
  context->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&context->mutex);

  // Initialize global list pointers.
  context->context_list_entry.next = NULL;
  context->context_list_entry.prev = NULL;

  // Initialize stream tracking BEFORE creating default stream.
  iree_slim_mutex_initialize(&context->stream_list_mutex);
  context->stream_count = 0;
  context->stream_capacity =
      8;  // Pre-allocate for default stream + user streams.
  context->streams = NULL;

  // Initialize default limits.
  // These are typical defaults matching CUDA/HIP behavior.
  context->limits.stack_size = 1024;                        // 1KB default
  context->limits.printf_fifo_size = 1024 * 1024;           // 1MB
  context->limits.malloc_heap_size = 8 * 1024 * 1024;       // 8MB
  context->limits.dev_runtime_sync_depth = 128;             // 128 levels
  context->limits.dev_runtime_pending_launch_count = 2048;  // 2048 launches
  context->limits.max_l2_fetch_granularity = 128;           // 128 bytes
  context->limits.persisting_l2_cache_size = 0;             // 0 = default

  // Retain the HAL device.
  iree_hal_device_retain(context->device);
  iree_hal_allocator_retain(context->device_allocator);

  // Create executable cache for this context.
  iree_status_t status = iree_hal_executable_cache_create(
      context->device, IREE_SV("stream_hal_cache"),
      iree_loop_inline(&context->loop_status), &context->executable_cache);

  // Create buffer mapping table.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_buffer_table_allocate(host_allocator,
                                                      &context->buffer_table);
  }

  // Allocate stream tracking array.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        host_allocator,
        sizeof(iree_hal_streaming_stream_t*) * context->stream_capacity,
        (void**)&context->streams);
  }

  // Create default stream.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_stream_create(context, /*flags=*/0,
                                              /*priority=*/0, host_allocator,
                                              &context->default_stream);
  }

  if (iree_status_is_ok(status)) {
    // Register with global list.
    iree_hal_streaming_register_context(context);
    *out_context = context;
  } else {
    iree_hal_streaming_context_destroy(context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_context_destroy(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Unregister from global list.
  iree_hal_streaming_unregister_context(context);

  // Clean up peer contexts array.
  if (context->peer_contexts) {
    for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
      if (context->peer_contexts[i]) {
        iree_hal_streaming_context_release(context->peer_contexts[i]);
      }
    }
    iree_allocator_free(context->host_allocator, context->peer_contexts);
  }

  // Synchronize all streams before cleanup to ensure all operations complete.
  // This is particularly important for the default stream which may have
  // pending command buffers with allocated arena blocks.
  if (context->default_stream) {
    iree_status_ignore(
        iree_hal_streaming_stream_synchronize(context->default_stream));
  }

  // Free buffer mapping table.
  iree_hal_streaming_buffer_table_free(context->buffer_table);

  // Release default stream FIRST before cleaning up stream list.
  if (context->default_stream) {
    iree_hal_streaming_stream_release(context->default_stream);
    context->default_stream = NULL;
  }

  // Assert all streams have been unregistered.
  IREE_ASSERT_EQ(context->stream_count, 0,
                 "context destroyed with %u streams still registered",
                 context->stream_count);

  // Free stream tracking resources.
  if (context->streams) {
    iree_allocator_free(context->host_allocator, context->streams);
  }
  iree_slim_mutex_deinitialize(&context->stream_list_mutex);

  iree_status_ignore(context->loop_status);
  iree_hal_executable_cache_release(context->executable_cache);
  iree_hal_allocator_release(context->device_allocator);
  iree_hal_device_release(context->device);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&context->mutex);

  // Free context memory.
  const iree_allocator_t host_allocator = context->host_allocator;
  iree_allocator_free(host_allocator, context);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_context_retain(iree_hal_streaming_context_t* context) {
  if (context) {
    iree_atomic_ref_count_inc(&context->ref_count);
  }
}

void iree_hal_streaming_context_release(iree_hal_streaming_context_t* context) {
  if (context && iree_atomic_ref_count_dec(&context->ref_count) == 1) {
    iree_hal_streaming_context_destroy(context);
  }
}

iree_hal_streaming_context_flags_t iree_hal_streaming_context_flags(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  return context->flags;
}

iree_hal_streaming_context_t* iree_hal_streaming_context_current(void) {
  iree_hal_streaming_context_t* context = iree_hal_streaming_current_context;
  return context;
}

iree_status_t iree_hal_streaming_context_set_current(
    iree_hal_streaming_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Retain new context and release old one.
  iree_hal_streaming_context_t* old_context =
      iree_hal_streaming_current_context;
  if (context) {
    iree_hal_streaming_context_retain(context);
  }
  iree_hal_streaming_current_context = context;
  if (old_context) {
    iree_hal_streaming_context_release(old_context);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_push(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Grow stack if needed.
  if (iree_hal_streaming_context_stack.depth >=
      iree_hal_streaming_context_stack.capacity) {
    iree_host_size_t new_capacity =
        iree_hal_streaming_context_stack.capacity
            ? iree_hal_streaming_context_stack.capacity * 2
            : 8;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_realloc(
                context->host_allocator,
                new_capacity * sizeof(iree_hal_streaming_context_t*),
                (void**)&iree_hal_streaming_context_stack.contexts));
    iree_hal_streaming_context_stack.capacity = new_capacity;
  }

  // Push current context onto stack.
  if (iree_hal_streaming_current_context) {
    iree_hal_streaming_context_stack
        .contexts[iree_hal_streaming_context_stack.depth++] =
        iree_hal_streaming_current_context;
  }

  // Set new current context.
  iree_hal_streaming_context_retain(context);
  iree_hal_streaming_current_context = context;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_pop(
    iree_hal_streaming_context_t** out_context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (out_context) *out_context = NULL;

  // Release current context.
  if (iree_hal_streaming_current_context) {
    if (out_context) {
      *out_context = iree_hal_streaming_current_context;
    } else {
      iree_hal_streaming_context_release(iree_hal_streaming_current_context);
    }
  }

  // Pop from stack.
  if (iree_hal_streaming_context_stack.depth > 0) {
    iree_hal_streaming_current_context =
        iree_hal_streaming_context_stack
            .contexts[--iree_hal_streaming_context_stack.depth];
  } else {
    iree_hal_streaming_current_context = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t* out_value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  // Return the limit value from context.
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      *out_value = context->limits.stack_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      *out_value = context->limits.printf_fifo_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      *out_value = context->limits.malloc_heap_size;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      *out_value = context->limits.dev_runtime_sync_depth;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      *out_value = context->limits.dev_runtime_pending_launch_count;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      *out_value = context->limits.max_l2_fetch_granularity;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      *out_value = context->limits.persisting_l2_cache_size;
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid limit type %d", limit);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_set_limit(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_limit_t limit, size_t value) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Validate the limit value first without holding the lock.
  iree_status_t status = iree_ok_status();
  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      // Stack size must be at least 512 bytes.
      if (value < 512) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "stack size must be at least 512 bytes");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      // Printf FIFO must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "printf FIFO size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      // Heap size must be at least 4KB.
      if (value < 4096) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malloc heap size must be at least 4KB");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      // Must be at least 1.
      if (value < 1) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "runtime sync depth must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      // Must be at least 1.
      if (value < 1) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "runtime pending launch count must be at least 1");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      // Must be 0, 32, 64, or 128 bytes.
      if (value != 0 && value != 32 && value != 64 && value != 128) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "L2 fetch granularity must be 0, 32, 64, or 128");
      }
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      // No specific validation for cache size.
      break;
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid limit type %d", limit);
  }
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  // Now take the lock and set the value.
  iree_slim_mutex_lock(&context->mutex);

  switch (limit) {
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_STACK_SIZE:
      context->limits.stack_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PRINTF_FIFO_SIZE:
      context->limits.printf_fifo_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MALLOC_HEAP_SIZE:
      context->limits.malloc_heap_size = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_SYNC_DEPTH:
      context->limits.dev_runtime_sync_depth = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT:
      context->limits.dev_runtime_pending_launch_count = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_MAX_L2_FETCH_GRANULARITY:
      context->limits.max_l2_fetch_granularity = value;
      break;
    case IREE_HAL_STREAMING_CONTEXT_LIMIT_PERSISTING_L2_CACHE_SIZE:
      context->limits.persisting_l2_cache_size = value;
      break;
    default:
      // Already validated above, should not reach here.
      break;
  }

  iree_slim_mutex_unlock(&context->mutex);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_enable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Check if already enabled.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();  // Already enabled.
    }
  }

  // Grow peer array if needed.
  if (context->peer_count >= context->peer_capacity) {
    const iree_host_size_t new_capacity =
        context->peer_capacity ? context->peer_capacity * 2 : 4;
    iree_status_t status = iree_allocator_realloc(
        context->host_allocator,
        new_capacity * sizeof(iree_hal_streaming_context_t*),
        (void**)&context->peer_contexts);
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_unlock(&context->mutex);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
    }
    context->peer_capacity = new_capacity;
  }

  // Add peer context.
  iree_hal_streaming_context_retain(peer_context);
  context->peer_contexts[context->peer_count++] = peer_context;

  // Update P2P topology if we have the registry.
  iree_hal_streaming_device_registry_t* device_registry =
      iree_hal_streaming_device_registry();
  if (device_registry && device_registry->p2p_topology) {
    const iree_host_size_t src_ordinal = context->device_ordinal;
    const iree_host_size_t dst_ordinal = peer_context->device_ordinal;
    const iree_host_size_t device_count = device_registry->device_count;
    if (src_ordinal < device_count && dst_ordinal < device_count) {
      // Find the link in topology.
      const iree_host_size_t link_index =
          src_ordinal * device_count + dst_ordinal;
      iree_hal_streaming_p2p_link_t* link =
          &device_registry->p2p_topology[link_index];
      // Enable P2P access.
      link->access_supported = true;
      // TODO: Query actual P2P capabilities.
    }
  }

  iree_slim_mutex_unlock(&context->mutex);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_context_disable_peer_access(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_context_t* peer_context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(peer_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->mutex);

  // Find and remove peer.
  for (iree_host_size_t i = 0; i < context->peer_count; ++i) {
    if (context->peer_contexts[i] == peer_context) {
      // Release peer context.
      iree_hal_streaming_context_release(peer_context);

      // Shift remaining peers.
      for (iree_host_size_t j = i + 1; j < context->peer_count; ++j) {
        context->peer_contexts[j - 1] = context->peer_contexts[j];
      }
      context->peer_count--;

      // Update P2P topology.
      iree_hal_streaming_device_registry_t* device_registry =
          iree_hal_streaming_device_registry();
      if (device_registry && device_registry->p2p_topology) {
        const iree_host_size_t src_ordinal = context->device_ordinal;
        const iree_host_size_t dst_ordinal = peer_context->device_ordinal;
        const iree_host_size_t device_count = device_registry->device_count;
        if (src_ordinal < device_count && dst_ordinal < device_count) {
          // Find the link in topology.
          const iree_host_size_t link_index =
              src_ordinal * device_count + dst_ordinal;
          iree_hal_streaming_p2p_link_t* link =
              &device_registry->p2p_topology[link_index];
          // Disable P2P access.
          link->access_supported = false;
        }
      }

      iree_slim_mutex_unlock(&context->mutex);
      IREE_TRACE_ZONE_END(z0);
      return iree_ok_status();
    }
  }

  iree_slim_mutex_unlock(&context->mutex);

  // Peer not found.
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_NOT_FOUND, "peer context not found");
}

iree_status_t iree_hal_streaming_context_register_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();

  iree_slim_mutex_lock(&context->stream_list_mutex);

  // Grow array if needed (double capacity).
  if (context->stream_count >= context->stream_capacity) {
    iree_host_size_t new_capacity = context->stream_capacity * 2;
    status = iree_allocator_realloc(
        context->host_allocator,
        sizeof(iree_hal_streaming_stream_t*) * new_capacity,
        (void**)&context->streams);
    if (iree_status_is_ok(status)) {
      context->stream_capacity = new_capacity;
    }
  }

  if (iree_status_is_ok(status)) {
    context->streams[context->stream_count++] = stream;  // Non-owning.
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_streaming_context_unregister_stream(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_t* stream) {
  if (!context || !stream) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&context->stream_list_mutex);

  for (iree_host_size_t i = 0; i < context->stream_count; ++i) {
    if (context->streams[i] == stream) {
      // Swap with last and remove.
      context->streams[i] = context->streams[context->stream_count - 1];
      --context->stream_count;
      break;
    }
  }

  iree_slim_mutex_unlock(&context->stream_list_mutex);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_streaming_context_wait_idle(
    iree_hal_streaming_context_t* context, iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Make temporary retained copy of streams to avoid use-after-free if another
  // thread comes in and tries to delete the stream.
  iree_slim_mutex_lock(&context->stream_list_mutex);
  const iree_host_size_t count = context->stream_count;
  iree_hal_streaming_stream_t** temp_streams =
      (iree_hal_streaming_stream_t**)iree_alloca(sizeof(temp_streams[0]) *
                                                 count);
  for (iree_host_size_t i = 0; i < count; ++i) {
    temp_streams[i] = context->streams[i];
    iree_hal_streaming_stream_retain(temp_streams[i]);
  }
  iree_slim_mutex_unlock(&context->stream_list_mutex);

  // Synchronize all streams. Bail on the first failure.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < count; ++i) {
    status = iree_hal_streaming_stream_synchronize(temp_streams[i]);
  }

  // Release temporary references.
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_streaming_stream_release(temp_streams[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_context_synchronize(
    iree_hal_streaming_context_t* context) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize default stream.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_stream_synchronize(context->default_stream));

  // Could also synchronize any other active streams if we track them?

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
