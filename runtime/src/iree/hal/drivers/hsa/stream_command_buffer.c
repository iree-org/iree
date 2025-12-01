// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/stream_command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hsa/hsa_buffer.h"
#include "iree/hal/drivers/hsa/native_executable.h"
#include "iree/hal/drivers/hsa/per_device_information.h"
#include "iree/hal/drivers/hsa/status_util.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_hsa_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;

  // Per-device information for the target device.
  iree_hal_hsa_per_device_info_t* device_info;

  // Arena used for all allocations; references the block pool.
  iree_arena_allocator_t arena;

  // Maintains a reference to all resources used within the command buffer.
  iree_hal_resource_set_t* resource_set;

  // Device allocator for transient allocations.
  iree_hal_allocator_t* device_allocator;

  // Tracing context for this command buffer.
  iree_hal_stream_tracing_context_t* tracing_context;
} iree_hal_hsa_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable;

static iree_hal_hsa_stream_command_buffer_t*
iree_hal_hsa_stream_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_stream_command_buffer_vtable);
  return (iree_hal_hsa_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_hsa_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_stream_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_hsa_per_device_info_t* device_info,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(device_info);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hsa_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->hsa_symbols = hsa_symbols;
  command_buffer->device_info = device_info;
  command_buffer->device_allocator = device_allocator;
  command_buffer->tracing_context = tracing_context;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_hsa_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(command_buffer,
                              &iree_hal_hsa_stream_command_buffer_vtable);
}

static void iree_hal_hsa_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // Nothing to do - commands are executed immediately.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // No-op for now.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // No-op for now.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }

  // HSA uses AQL packets with barriers - for now we use signal-based sync.
  // Wait for all previous work to complete.
  command_buffer->hsa_symbols->hsa_signal_store_screlease(
      command_buffer->device_info->completion_signal, 1);

  // In a full implementation, we would use a barrier packet here.
  // For simplicity, we do a blocking wait.
  command_buffer->hsa_symbols->hsa_signal_wait_scacquire(
      command_buffer->device_info->completion_signal, HSA_SIGNAL_CONDITION_EQ,
      0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  // We could mark the memory as invalidated so that if managed HSA does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Use hsa_amd_memory_fill for 32-bit patterns.
  if (pattern_length == 4) {
    uint32_t pattern32 = *(const uint32_t*)pattern;
    size_t count = target_ref.length / sizeof(uint32_t);
    return IREE_HSA_CALL_TO_STATUS(
        command_buffer->hsa_symbols,
        hsa_amd_memory_fill(target_ptr, pattern32, count),
        "hsa_amd_memory_fill");
  }

  // For other pattern sizes, we need to do a manual fill.
  // This is a simple synchronous implementation.
  uint8_t* target_bytes = (uint8_t*)target_ptr;
  for (iree_device_size_t i = 0; i < target_ref.length; i += pattern_length) {
    memcpy(target_bytes + i, pattern,
           iree_min(pattern_length, target_ref.length - i));
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_ref.buffer));

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Simple memcpy for now - assumes host-visible memory.
  memcpy(target_ptr, (const uint8_t*)source_buffer + source_offset,
         target_ref.length);

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);

  iree_hal_buffer_t* buffers[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  void* source_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  const void* source_ptr = (const uint8_t*)source_device_buffer + source_offset;

  void* target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  void* target_ptr = (uint8_t*)target_device_buffer + target_offset;

  // Use async copy with completion signal.
  hsa_signal_t completion_signal =
      command_buffer->device_info->completion_signal;
  command_buffer->hsa_symbols->hsa_signal_store_screlease(completion_signal, 1);

  iree_status_t status = IREE_HSA_CALL_TO_STATUS(
      command_buffer->hsa_symbols,
      hsa_amd_memory_async_copy(target_ptr, command_buffer->device_info->agent,
                                source_ptr, command_buffer->device_info->agent,
                                source_ref.length, 0, NULL, completion_signal),
      "hsa_amd_memory_async_copy");

  if (iree_status_is_ok(status)) {
    // Wait for copy to complete (synchronous for simplicity).
    command_buffer->hsa_symbols->hsa_signal_wait_scacquire(
        completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
        HSA_WAIT_STATE_BLOCKED);
  }

  return status;
}

static iree_status_t iree_hal_hsa_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref,
    iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: we can support CUSTOM_DIRECT_ARGUMENTS quite easily here.
  if (iree_hal_dispatch_uses_custom_arguments(flags)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "direct/indirect arguments are not supported in HSA streams");
  } else if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect parameters are not supported in HSA streams");
  }

  // If any of the workgroup counts are zero, we can skip execution.
  if (config.workgroup_count[0] == 0 || config.workgroup_count[1] == 0 ||
      config.workgroup_count[2] == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Get kernel params.
  const iree_hal_hsa_kernel_params_t* kernel_params = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_lookup_kernel_params(
              executable, export_ordinal, command_buffer->base.queue_affinity,
              &kernel_params));

  IREE_TRACE({
    if (kernel_params->debug_info.function_name.size > 0) {
      IREE_TRACE_ZONE_APPEND_TEXT(z0,
                                  kernel_params->debug_info.function_name.data,
                                  kernel_params->debug_info.function_name.size);
    }
  });

  // Track resources.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (bindings.values[i].buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &bindings.values[i].buffer));
    }
  }

  // Allocate kernarg memory.
  void* kernarg_address = NULL;
  if (kernel_params->kernarg_segment_size > 0 &&
      command_buffer->device_info->kernarg_memory_pool_valid) {
    iree_status_t status = IREE_HSA_CALL_TO_STATUS(
        command_buffer->hsa_symbols,
        hsa_amd_memory_pool_allocate(
            command_buffer->device_info->kernarg_memory_pool,
            kernel_params->kernarg_segment_size, 0, &kernarg_address),
        "hsa_amd_memory_pool_allocate (kernarg)");
    if (!iree_status_is_ok(status)) {
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  // Set up kernarg: buffer pointers followed by constants.
  if (kernarg_address) {
    uint8_t* kernarg_ptr = (uint8_t*)kernarg_address;

    // Set up buffer pointers.
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      void* buffer_ptr = NULL;
      if (bindings.values[i].buffer) {
        void* device_buffer = iree_hal_hsa_buffer_device_pointer(
            iree_hal_buffer_allocated_buffer(bindings.values[i].buffer));
        buffer_ptr = (uint8_t*)device_buffer +
                     iree_hal_buffer_byte_offset(bindings.values[i].buffer) +
                     bindings.values[i].offset;
      }
      memcpy(kernarg_ptr, &buffer_ptr, sizeof(buffer_ptr));
      kernarg_ptr += sizeof(buffer_ptr);
    }

    // Copy constants.
    if (constants.data_length > 0) {
      memcpy(kernarg_ptr, constants.data, constants.data_length);
    }
  }

  // Create and submit AQL dispatch packet.
  hsa_queue_t* queue = command_buffer->device_info->queue;

  // Get write index for the queue.
  uint64_t write_index =
      command_buffer->hsa_symbols->hsa_queue_add_write_index_relaxed(queue, 1);

  // Wait for queue space.
  while (write_index -
             command_buffer->hsa_symbols->hsa_queue_load_read_index_relaxed(
                 queue) >=
         queue->size) {
    // Busy wait for space.
  }

  // Get packet address.
  hsa_kernel_dispatch_packet_t* packet =
      (hsa_kernel_dispatch_packet_t*)queue->base_address +
      (write_index & (queue->size - 1));

  // Initialize packet.
  memset(packet, 0, sizeof(*packet));
  packet->setup = 3;  // 3 dimensions
  packet->workgroup_size_x = config.workgroup_size[0]
                                 ? config.workgroup_size[0]
                                 : kernel_params->block_dims[0];
  packet->workgroup_size_y = config.workgroup_size[1]
                                 ? config.workgroup_size[1]
                                 : kernel_params->block_dims[1];
  packet->workgroup_size_z = config.workgroup_size[2]
                                 ? config.workgroup_size[2]
                                 : kernel_params->block_dims[2];
  packet->grid_size_x = config.workgroup_count[0] * packet->workgroup_size_x;
  packet->grid_size_y = config.workgroup_count[1] * packet->workgroup_size_y;
  packet->grid_size_z = config.workgroup_count[2] * packet->workgroup_size_z;
  packet->group_segment_size = kernel_params->group_segment_size;
  packet->private_segment_size = kernel_params->private_segment_size;
  packet->kernel_object = kernel_params->kernel_object;
  packet->kernarg_address = kernarg_address;

  // Set up completion signal.
  hsa_signal_t completion_signal =
      command_buffer->device_info->completion_signal;
  command_buffer->hsa_symbols->hsa_signal_store_screlease(completion_signal, 1);
  packet->completion_signal = completion_signal;

  // Set header and launch.
  uint16_t header =
      HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE |
      HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE |
      HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);

  // Ring doorbell.
  command_buffer->hsa_symbols->hsa_signal_store_screlease(queue->doorbell_signal,
                                                          write_index);

  // Wait for completion (synchronous for simplicity).
  command_buffer->hsa_symbols->hsa_signal_wait_scacquire(
      completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
      HSA_WAIT_STATE_BLOCKED);

  // Free kernarg memory.
  if (kernarg_address) {
    IREE_HSA_IGNORE_ERROR(command_buffer->hsa_symbols,
                          hsa_amd_memory_pool_free(kernarg_address));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable = {
        .destroy = iree_hal_hsa_stream_command_buffer_destroy,
        .begin = iree_hal_hsa_stream_command_buffer_begin,
        .end = iree_hal_hsa_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_hsa_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hsa_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hsa_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_hsa_stream_command_buffer_signal_event,
        .reset_event = iree_hal_hsa_stream_command_buffer_reset_event,
        .wait_events = iree_hal_hsa_stream_command_buffer_wait_events,
        .advise_buffer = iree_hal_hsa_stream_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_hsa_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hsa_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hsa_stream_command_buffer_copy_buffer,
        .collective = iree_hal_hsa_stream_command_buffer_collective,
        .dispatch = iree_hal_hsa_stream_command_buffer_dispatch,
};
