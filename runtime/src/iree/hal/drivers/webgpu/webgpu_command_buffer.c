// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/webgpu/webgpu_command_buffer.h"

#include "iree/hal/drivers/webgpu/webgpu_buffer.h"
#include "iree/hal/drivers/webgpu/webgpu_executable.h"
#include "iree/hal/drivers/webgpu/webgpu_imports.h"

//===----------------------------------------------------------------------===//
// iree_hal_webgpu_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_webgpu_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_webgpu_handle_t device_handle;
  iree_hal_webgpu_handle_t queue_handle;

  // Pre-created compute pipelines for fill/copy. Borrowed from the device.
  const iree_hal_webgpu_builtins_t* builtins;

  // Instruction stream builder.
  iree_hal_webgpu_builder_t builder;

  // JS-side Recording handle for reusable command buffers. Created at end() for
  // reusable mode. 0 for ONE_SHOT.
  iree_hal_webgpu_handle_t recording_handle;
} iree_hal_webgpu_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_webgpu_command_buffer_vtable;

static iree_hal_webgpu_command_buffer_t* iree_hal_webgpu_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_webgpu_command_buffer_vtable);
  return (iree_hal_webgpu_command_buffer_t*)base_value;
}

iree_status_t iree_hal_webgpu_command_buffer_create(
    iree_hal_webgpu_handle_t device_handle,
    iree_hal_webgpu_handle_t queue_handle,
    const iree_hal_webgpu_builtins_t* builtins,
    iree_arena_block_pool_t* block_pool, iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(builtins);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  iree_hal_webgpu_command_buffer_t* command_buffer = NULL;
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
      &iree_hal_webgpu_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device_handle = device_handle;
  command_buffer->queue_handle = queue_handle;
  command_buffer->builtins = builtins;
  command_buffer->recording_handle = 0;

  // ONE_SHOT command buffers have no dynamic slots (all buffers are direct).
  // Reusable command buffers reserve binding_capacity dynamic slots.
  uint32_t dynamic_count = (mode & IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)
                               ? 0
                               : (uint32_t)binding_capacity;

  iree_status_t status = iree_hal_webgpu_builder_initialize(
      block_pool, dynamic_count, host_allocator, &command_buffer->builder);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, command_buffer);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_webgpu_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_webgpu_command_buffer_vtable);
}

iree_hal_webgpu_builder_t* iree_hal_webgpu_command_buffer_builder(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return &command_buffer->builder;
}

iree_hal_webgpu_handle_t iree_hal_webgpu_command_buffer_recording_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return command_buffer->recording_handle;
}

static void iree_hal_webgpu_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release the JS-side recording if present.
  if (command_buffer->recording_handle) {
    iree_hal_webgpu_import_handle_release(command_buffer->recording_handle);
  }

  iree_hal_webgpu_builder_deinitialize(&command_buffer->builder);
  iree_allocator_free(host_allocator, command_buffer);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_webgpu_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  // Release any previous recording (re-recording a reusable command buffer).
  if (command_buffer->recording_handle) {
    iree_hal_webgpu_import_handle_release(command_buffer->recording_handle);
    command_buffer->recording_handle = 0;
  }

  return iree_hal_webgpu_builder_reset(&command_buffer->builder);
}

static iree_status_t iree_hal_webgpu_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(
      iree_hal_webgpu_builder_finalize(&command_buffer->builder));

  // For reusable command buffers, create a JS-side Recording that caches the
  // instruction stream and static bindings.
  bool reusable =
      !(command_buffer->base.mode & IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT);
  if (reusable) {
    iree_hal_webgpu_builder_t* builder = &command_buffer->builder;

    // Build the static binding table in wire format.
    uint32_t static_count = iree_hal_webgpu_builder_static_slot_count(builder);
    iree_hal_webgpu_isa_binding_table_entry_t* static_entries = NULL;
    if (static_count > 0) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
          command_buffer->host_allocator, static_count,
          sizeof(iree_hal_webgpu_isa_binding_table_entry_t),
          (void**)&static_entries));
      const iree_hal_webgpu_builder_slot_entry_t* slot_entries =
          iree_hal_webgpu_builder_static_slot_entries(builder);
      for (uint32_t i = 0; i < static_count; ++i) {
        static_entries[i].gpu_buffer_handle = slot_entries[i].gpu_buffer_handle;
        static_entries[i].base_offset =
            0;  // Static: offset is in instructions.
      }
    }

    iree_hal_webgpu_isa_builtins_descriptor_t builtins_descriptor;
    iree_hal_webgpu_builtins_get_descriptor(command_buffer->builtins,
                                            &builtins_descriptor);

    command_buffer->recording_handle = iree_hal_webgpu_import_create_recording(
        command_buffer->device_handle,
        (uint32_t)(uintptr_t)iree_hal_webgpu_builder_block_table(builder),
        iree_hal_webgpu_builder_block_count(builder),
        iree_hal_webgpu_builder_block_word_capacity(builder),
        iree_hal_webgpu_builder_last_block_word_count(builder),
        (uint32_t)(uintptr_t)static_entries, static_count,
        builder->dynamic_count, (uint32_t)(uintptr_t)&builtins_descriptor);

    iree_allocator_free(command_buffer->host_allocator, static_entries);

    if (!command_buffer->recording_handle) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "failed to create JS-side recording for reusable command buffer");
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return iree_hal_webgpu_builder_execution_barrier(&command_buffer->builder);
}

static iree_status_t iree_hal_webgpu_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support command buffer events");
}

static iree_status_t iree_hal_webgpu_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support command buffer events");
}

static iree_status_t iree_hal_webgpu_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support command buffer events");
}

static iree_status_t iree_hal_webgpu_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return iree_hal_webgpu_builder_fill_buffer(
      &command_buffer->builder, target_ref, pattern, pattern_length);
}

//===----------------------------------------------------------------------===//
// update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return iree_hal_webgpu_builder_update_buffer(
      &command_buffer->builder, source_buffer, source_offset, target_ref);
}

//===----------------------------------------------------------------------===//
// copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);
  return iree_hal_webgpu_builder_copy_buffer(&command_buffer->builder,
                                             source_ref, target_ref);
}

//===----------------------------------------------------------------------===//
// Remaining vtable methods
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_webgpu_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "WebGPU does not support collective operations");
}

static iree_status_t iree_hal_webgpu_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_webgpu_command_buffer_t* command_buffer =
      iree_hal_webgpu_command_buffer_cast(base_command_buffer);

  iree_hal_webgpu_handle_t pipeline_handle =
      iree_hal_webgpu_executable_pipeline_handle(executable, export_ordinal);
  iree_hal_webgpu_handle_t bind_group_layout_handle =
      iree_hal_webgpu_executable_bind_group_layout_handle(executable,
                                                          export_ordinal);

  return iree_hal_webgpu_builder_dispatch(
      &command_buffer->builder, pipeline_handle, bind_group_layout_handle,
      config.workgroup_count, bindings);
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_webgpu_command_buffer_vtable = {
        .destroy = iree_hal_webgpu_command_buffer_destroy,
        .begin = iree_hal_webgpu_command_buffer_begin,
        .end = iree_hal_webgpu_command_buffer_end,
        .begin_debug_group = iree_hal_webgpu_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_webgpu_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_webgpu_command_buffer_execution_barrier,
        .signal_event = iree_hal_webgpu_command_buffer_signal_event,
        .reset_event = iree_hal_webgpu_command_buffer_reset_event,
        .wait_events = iree_hal_webgpu_command_buffer_wait_events,
        .advise_buffer = iree_hal_webgpu_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_webgpu_command_buffer_fill_buffer,
        .update_buffer = iree_hal_webgpu_command_buffer_update_buffer,
        .copy_buffer = iree_hal_webgpu_command_buffer_copy_buffer,
        .collective = iree_hal_webgpu_command_buffer_collective,
        .dispatch = iree_hal_webgpu_command_buffer_dispatch,
};
