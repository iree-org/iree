// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"

//===----------------------------------------------------------------------===//
// Stream management
//===----------------------------------------------------------------------===//

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream);

iree_status_t iree_hal_streaming_stream_create(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_stream_flags_t flags, int priority,
    iree_allocator_t host_allocator, iree_hal_streaming_stream_t** out_stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_stream);
  *out_stream = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_stream_t* stream = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*stream), (void**)&stream));

  // Initialize stream.
  iree_atomic_ref_count_init(&stream->ref_count);
  stream->context = context;
  stream->flags = flags;
  stream->priority = priority;
  stream->command_buffer = NULL;
  stream->timeline_semaphore = NULL;
  stream->pending_value = 0;
  stream->completed_value = 0;
  stream->queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  stream->recorded_events = NULL;
  stream->event_count = 0;
  stream->event_capacity = 0;

  // Initialize capture state.
  stream->capture_status = IREE_HAL_STREAMING_CAPTURE_STATUS_NONE;
  stream->capture_mode = IREE_HAL_STREAMING_CAPTURE_MODE_GLOBAL;
  stream->capture_graph = NULL;
  stream->capture_id = 0;
  stream->capture_dependencies = NULL;
  stream->capture_dependency_count = 0;
  stream->capture_dependency_capacity = 0;

  stream->host_allocator = host_allocator;
  iree_slim_mutex_initialize(&stream->mutex);

  // Create timeline semaphore for synchronization.
  iree_status_t status = iree_hal_semaphore_create(context->device, 0ULL,
                                                   IREE_HAL_SEMAPHORE_FLAG_NONE,
                                                   &stream->timeline_semaphore);

  // Register stream with context.
  if (iree_status_is_ok(status)) {
    status = iree_hal_streaming_context_register_stream(context, stream);
  }

  if (iree_status_is_ok(status)) {
    *out_stream = stream;
  } else {
    iree_hal_streaming_stream_destroy(stream);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_streaming_stream_destroy(
    iree_hal_streaming_stream_t* stream) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize stream before cleanup to ensure all operations complete.
  // This is important to avoid leaking resources from pending operations.
  iree_status_ignore(iree_hal_streaming_stream_synchronize(stream));

  // Unregister from context before cleanup.
  if (stream->context) {
    iree_hal_streaming_context_unregister_stream(stream->context, stream);
  }

  // Clean up recorded events.
  if (stream->recorded_events) {
    for (iree_host_size_t i = 0; i < stream->event_count; ++i) {
      if (stream->recorded_events[i]) {
        iree_hal_streaming_event_release(stream->recorded_events[i]);
      }
    }
    iree_allocator_free(stream->host_allocator, stream->recorded_events);
  }

  // Release command buffer.
  iree_hal_command_buffer_release(stream->command_buffer);

  // Release timeline semaphore.
  iree_hal_semaphore_release(stream->timeline_semaphore);

  // Deinitialize synchronization.
  iree_slim_mutex_deinitialize(&stream->mutex);

  // Free stream memory.
  const iree_allocator_t host_allocator = stream->host_allocator;
  iree_allocator_free(host_allocator, stream);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_streaming_stream_retain(iree_hal_streaming_stream_t* stream) {
  if (stream) {
    iree_atomic_ref_count_inc(&stream->ref_count);
  }
}

void iree_hal_streaming_stream_release(iree_hal_streaming_stream_t* stream) {
  if (stream && iree_atomic_ref_count_dec(&stream->ref_count) == 1) {
    iree_hal_streaming_stream_destroy(stream);
  }
}

iree_status_t iree_hal_streaming_stream_begin(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);

  // Create command buffer if not already created.
  // Note that we set UNRETAINED as we ensure the resources we have to track are
  // retained at the graph exec level and CUDA/HIP don't make any statements
  // about resource lifetime.
  //
  // TODO: if we are beginning with an idle stream or no waits we _could_
  // ALLOW_INLINE_EXECUTION. I'm not exactly sure how to maintain that, though,
  // so for now we err on the side of deferring.
  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_create(
                stream->context->device,
                IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
                    IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED,
                IREE_HAL_COMMAND_CATEGORY_ANY, stream->queue_affinity,
                /*binding_capacity=*/0, &stream->command_buffer));
  }

  // Begin recording.
  iree_status_t status = iree_hal_command_buffer_begin(stream->command_buffer);

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_flush(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_slim_mutex_lock(&stream->mutex);

  iree_status_t status = iree_ok_status();
  if (stream->command_buffer) {
    // End recording and submit command buffer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_command_buffer_end(stream->command_buffer));

    // Use the completed value as wait to ensure sequential execution.
    uint64_t wait_value = stream->completed_value;

    // Increment pending value for this submission.
    stream->pending_value++;

    // Submit to device queue with timeline semaphore.
    // Wait for the previous submission to complete before executing.
    iree_hal_queue_affinity_t queue_affinity = stream->queue_affinity;
    iree_hal_semaphore_list_t wait_semaphores = {
        .count = wait_value > 0
                     ? 1
                     : 0,  // Only wait if there was a previous submission.
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &wait_value,
    };
    iree_hal_semaphore_list_t signal_semaphores = {
        .count = 1,
        .semaphores = &stream->timeline_semaphore,
        .payload_values = &stream->pending_value,
    };

    status = iree_hal_device_queue_execute(
        stream->context->device, queue_affinity, wait_semaphores,
        signal_semaphores, stream->command_buffer,
        iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE);

    // Update completed value to track last submission.
    if (iree_status_is_ok(status)) {
      stream->completed_value = stream->pending_value;
    }

    // Release command buffer (we're done with it).
    iree_hal_command_buffer_release(stream->command_buffer);
    stream->command_buffer = NULL;
  }

  iree_slim_mutex_unlock(&stream->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_stream_query(
    iree_hal_streaming_stream_t* stream, int* status) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(status);

  uint64_t current_value = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_query(stream->timeline_semaphore, &current_value));

  if (current_value >= stream->pending_value) {
    *status = 0;  // Complete
    stream->completed_value = current_value;
  } else {
    *status = 1;  // Not complete
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_stream_synchronize(
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Wait for timeline semaphore to reach pending value.
  if (stream->pending_value > stream->completed_value) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_semaphore_wait(stream->timeline_semaphore,
                                    stream->pending_value,
                                    iree_infinite_timeout()));
    stream->completed_value = stream->pending_value;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_stream_wait_event(
    iree_hal_streaming_stream_t* stream, iree_hal_streaming_event_t* event) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(event);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush the stream to ensure all prior operations are submitted.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_streaming_stream_flush(stream));

  // Get the current stream pending value to signal after waiting for the event.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;

  // Create a queue barrier that waits for the event and signals the stream.
  // This ensures the stream continues only after the event is signaled.
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = 1,
      .semaphores = &event->semaphore,
      .payload_values = &event->signal_value,
  };
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_barrier(
              stream->context->device, stream->queue_affinity, wait_semaphores,
              signal_semaphores, IREE_HAL_EXECUTE_FLAG_NONE));

  // Update completed value to track this barrier.
  stream->completed_value = signal_value;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Execution control
//===----------------------------------------------------------------------===//

// DO NOT SUBMIT This needs to be implemented based on the actual argument
// format. For now, we'll just copy the pointers as-is, but this needs proper
// packing.
iree_status_t iree_hal_streaming_pack_kernel_args(
    const iree_hal_streaming_dispatch_params_t* params, void* packed_buffer,
    iree_host_size_t buffer_size, iree_host_size_t* out_packed_size) {
  // DO NOT SUBMIT
  // use extra

  // Calculate size needed.
  // For now, assume each kernel_param is a pointer (8 bytes).
  // This is a placeholder - actual implementation needs to know the real sizes.
  // DO NOT SUBMIT need to use reflection to detect param count
  iree_host_size_t num_params = 0;
  if (params->kernel_params) {
    // Count params until NULL.
    while (params->kernel_params[num_params]) {
      num_params++;
    }
  }

  iree_host_size_t packed_size = num_params * sizeof(void*);
  if (out_packed_size) {
    *out_packed_size = packed_size;
  }

  if (packed_buffer && buffer_size >= packed_size) {
    // Copy the parameter pointers.
    // DO NOT SUBMIT
    // memcpy(packed_buffer, params->kernel_params, packed_size);
    uint64_t* dst = (uint64_t*)packed_buffer;
    for (iree_host_size_t i = 0; i < num_params; ++i) {
      dst[i] = (uint64_t)params->kernel_params[i];
    }
  }

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_launch_kernel(
    iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(symbol);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Verify the symbol is a function.
  if (symbol->type != IREE_HAL_STREAMING_SYMBOL_TYPE_FUNCTION) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol is not a function (type=%d)", symbol->type);
  }

  // Check if cooperative launch is requested.
  if (params->flags & IREE_HAL_STREAMING_DISPATCH_FLAG_COOPERATIVE) {
    // TODO: Add HAL dispatch flag for cooperative kernel support and pass
    // through to the backend. For now, return unimplemented.
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "cooperative kernel launch not yet implemented in HAL layer");
  }

  // Ensure command buffer is recording.
  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Check if we're capturing to a graph.
  if (stream->capture_status == IREE_HAL_STREAMING_CAPTURE_STATUS_ACTIVE) {
    // Add kernel node to the graph instead of recording to command buffer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_graph_add_kernel_node(
                stream->capture_graph, stream->capture_dependencies,
                stream->capture_dependency_count, symbol, params, NULL));
    // Clear dependencies after adding the node.
    stream->capture_dependency_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Calculate the size needed for packed arguments.
  iree_host_size_t packed_args_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_streaming_pack_kernel_args(params, NULL, 0, &packed_args_size));

  // Allocate space for packed arguments.
  // Use stack allocation if under 4KB, otherwise heap allocate.
  void* packed_args = NULL;
  void* heap_allocation = NULL;
  const iree_host_size_t STACK_LIMIT = 4096;

  // DO NOT SUBMIT clean this up
  if (packed_args_size > 0) {
    if (packed_args_size <= STACK_LIMIT) {
      // Stack allocate for small buffers.
      packed_args = iree_alloca(packed_args_size);
    } else {
      // Heap allocate for large buffers.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(stream->host_allocator, packed_args_size,
                                    &heap_allocation));
      packed_args = heap_allocation;
    }

    // Pack the arguments.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_pack_kernel_args(params, packed_args,
                                                packed_args_size, NULL));
  }

  const iree_hal_dispatch_config_t config = {
      .workgroup_size =
          {
              params->block_dim[0],
              params->block_dim[1],
              params->block_dim[2],
          },
      .workgroup_count =
          {
              params->grid_dim[0],
              params->grid_dim[1],
              params->grid_dim[2],
          },
      .dynamic_workgroup_local_memory = params->shared_memory_bytes,
  };
  iree_hal_dispatch_flags_t flags =
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;

  // Create constants span from packed arguments.
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(packed_args, packed_args_size);

  iree_status_t status = iree_hal_command_buffer_dispatch(
      stream->command_buffer, symbol->module->executable, symbol->entry_point,
      config, constants, iree_hal_buffer_ref_list_empty(), flags);

  // Free heap allocation if we used one.
  if (heap_allocation) {
    iree_allocator_free(stream->host_allocator, heap_allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Host callback wrapper structure to adapt CUDA/HIP callbacks to HAL callbacks.
typedef struct iree_hal_streaming_host_callback_t {
  void (*fn)(void* user_data);
  void* user_data;
} iree_hal_streaming_host_callback_t;

// HAL host call function that invokes the CUDA/HIP style callback.
static iree_status_t iree_hal_streaming_host_callback_thunk(
    void* user_data, const uint64_t args[4],
    iree_hal_host_call_context_t* context) {
  iree_hal_streaming_host_callback_t* callback =
      (iree_hal_streaming_host_callback_t*)user_data;
  callback->fn(callback->user_data);
  iree_allocator_free(iree_allocator_system(), callback);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_launch_host_function(
    iree_hal_streaming_stream_t* stream, void (*fn)(void*), void* user_data) {
  IREE_ASSERT_ARGUMENT(stream);
  IREE_ASSERT_ARGUMENT(fn);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Flush any pending operations in the stream's command buffer.
  if (stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_flush(stream));
  }

  // Allocate a wrapper structure to hold the callback and user data.
  iree_hal_streaming_host_callback_t* callback = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(iree_allocator_system(), sizeof(*callback),
                                (void**)&callback));
  callback->fn = fn;
  callback->user_data = user_data;

  // Set up semaphores for the host call.
  // Wait for the current stream position.
  uint64_t wait_value = stream->pending_value;
  iree_hal_semaphore_list_t wait_semaphores = {
      .count = wait_value > 0 ? 1 : 0,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &wait_value,
  };

  // Signal the next value after the host call completes.
  uint64_t signal_value = stream->pending_value + 1;
  stream->pending_value = signal_value;
  iree_hal_semaphore_list_t signal_semaphores = {
      .count = 1,
      .semaphores = &stream->timeline_semaphore,
      .payload_values = &signal_value,
  };

  // Create the host call with our wrapper function.
  iree_hal_host_call_t call =
      iree_hal_make_host_call(iree_hal_streaming_host_callback_thunk, callback);

  // Empty args array (not used by CUDA/HIP callbacks).
  uint64_t args[4] = {0, 0, 0, 0};

  // Enqueue the host call on the device queue.
  // Use NON_BLOCKING flag as CUDA/HIP host functions don't block the stream.
  iree_status_t status = iree_hal_device_queue_host_call(
      stream->context->device, stream->queue_affinity, wait_semaphores,
      signal_semaphores, call, args, IREE_HAL_HOST_CALL_FLAG_NON_BLOCKING);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), callback);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
