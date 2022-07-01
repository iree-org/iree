// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/direct_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_buffer.h"
#include "experimental/level_zero/level_zero_event.h"
#include "experimental/level_zero/native_executable.h"
#include "experimental/level_zero/pipeline_layout.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/tracing.h"

// Command buffer implementation that directly maps to level_zero direct.
// This records the commands on the calling thread without additional threading
// indirection.

typedef struct {
  iree_hal_command_buffer_t base;
  iree_hal_level_zero_context_wrapper_t* context;
  iree_arena_block_pool_t* block_pool;
  ze_command_list_handle_t command_list;

  // Keep track of the current set of kernel arguments.
  int32_t push_constant[IREE_HAL_LEVEL_ZERO_MAX_PUSH_CONSTANT_COUNT];
  void* current_descriptor[];
} iree_hal_level_zero_direct_command_buffer_t;

#define IREE_HAL_LEVEL_ZERO_MAX_BINDING_COUNT 64
// Kernel arguments contains binding and push constants.
#define IREE_HAL_LEVEL_ZERO_MAX_KERNEL_ARG 128

static const iree_hal_command_buffer_vtable_t
    iree_hal_level_zero_direct_command_buffer_vtable;

static iree_hal_level_zero_direct_command_buffer_t*
iree_hal_level_zero_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_level_zero_direct_command_buffer_vtable);
  return (iree_hal_level_zero_direct_command_buffer_t*)base_value;
}
// TODO: Create helper function to get cmdlist out to device for submissions.
iree_status_t iree_hal_level_zero_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, ze_device_handle_t level_zero_device,
    uint32_t command_queue_ordinal,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_level_zero_direct_command_buffer_t* command_buffer = NULL;
  size_t total_size = sizeof(*command_buffer) +
                      IREE_HAL_LEVEL_ZERO_MAX_KERNEL_ARG * sizeof(void*) +
                      IREE_HAL_LEVEL_ZERO_MAX_KERNEL_ARG *
                          sizeof(iree_hal_level_zero_device_ptr_t);
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, total_size, (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, queue_affinity, binding_capacity,
        &iree_hal_level_zero_direct_command_buffer_vtable,
        &command_buffer->base);
    command_buffer->context = context;
    command_buffer->block_pool = block_pool;
    iree_hal_level_zero_device_ptr_t* device_ptrs =
        (iree_hal_level_zero_device_ptr_t*)(command_buffer->current_descriptor +
                                            IREE_HAL_LEVEL_ZERO_MAX_KERNEL_ARG);
    for (size_t i = 0; i < IREE_HAL_LEVEL_ZERO_MAX_KERNEL_ARG; i++) {
      command_buffer->current_descriptor[i] = &device_ptrs[i];
    }
    // Create a command list
    ze_command_list_handle_t command_list;
    ze_command_list_desc_t command_list_desc = {};
    command_list_desc.commandQueueGroupOrdinal = command_queue_ordinal;
    LEVEL_ZERO_RETURN_IF_ERROR(
        command_buffer->context->syms,
        zeCommandListCreate(command_buffer->context->level_zero_context,
                            level_zero_device, &command_list_desc,
                            &command_list),
        "zeCommandListCreate");
    command_buffer->command_list = command_list;

    *out_command_buffer = &command_buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_level_zero_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(command_buffer->context->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_level_zero_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_command_buffer_dyn_cast(
      command_buffer, &iree_hal_level_zero_direct_command_buffer_vtable);
}

static void* iree_hal_level_zero_direct_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable) {
  if (vtable == &iree_hal_level_zero_direct_command_buffer_vtable) {
    IREE_HAL_ASSERT_TYPE(command_buffer, vtable);
    return command_buffer;
  }
  return NULL;
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static void iree_hal_level_zero_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_level_zero_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

static iree_status_t
iree_hal_level_zero_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendBarrier(command_buffer->command_list, NULL, 0, NULL),
      "zeCommandListAppendMemoryFill");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO: Implement barrier
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendSignalEvent(command_buffer->command_list, iree_hal_level_zero_event_handle(event)),
      "zeCommandListAppendSignalEvent");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendEventReset(command_buffer->command_list, iree_hal_level_zero_event_handle(event)),
      "zeCommandListAppendEventReset");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_inline_array(ze_event_handle_t, event_handles, event_count, command_buffer->context->host_allocator);
  for (int i = 0; i < event_count; ++i) {
    *iree_inline_array_at(event_handles, i) =
        iree_hal_level_zero_event_handle(events[i]);
  }
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendWaitOnEvents(command_buffer->command_list, event_count, iree_inline_array_data(event_handles)),
      "zeCommandListAppendWaitOnEvents");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);

  iree_hal_level_zero_device_ptr_t target_device_buffer =
      iree_hal_level_zero_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  iree_hal_level_zero_device_ptr_t dst = target_device_buffer + target_offset;
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendMemoryFill(command_buffer->command_list, dst, pattern,
                                    pattern_length, length, NULL, 0, NULL),
      "zeCommandListAppendMemoryFill");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need level_zero implementation");
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);

  iree_hal_level_zero_device_ptr_t target_device_buffer =
      iree_hal_level_zero_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  iree_hal_level_zero_device_ptr_t source_device_buffer =
      iree_hal_level_zero_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  iree_hal_level_zero_device_ptr_t dst = target_device_buffer + target_offset;
  iree_hal_level_zero_device_ptr_t src = source_device_buffer + source_offset;
  // TODO(raikonenfnu): Currently using NULL stream, need to figure out way to
  // access proper stream from command buffer
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendMemoryCopy(command_buffer->command_list, dst, src,
                                    length, NULL, 0, NULL),
      "zeCommandListAppendMemoryCopy");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constant[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }
  return iree_ok_status();
}

// Tie together the binding index and its index in |bindings| array.
typedef struct {
  uint32_t index;
  uint32_t binding;
} iree_hal_level_zero_binding_mapping_t;

// Helper to sort the binding based on their binding index.
static int compare_binding_index(const void* a, const void* b) {
  const iree_hal_level_zero_binding_mapping_t buffer_a =
      *(const iree_hal_level_zero_binding_mapping_t*)a;
  const iree_hal_level_zero_binding_mapping_t buffer_b =
      *(const iree_hal_level_zero_binding_mapping_t*)b;
  return buffer_a.binding < buffer_b.binding ? -1 : 1;
}

static iree_status_t
iree_hal_level_zero_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_host_size_t base_binding =
      iree_hal_level_zero_base_binding_index(pipeline_layout, set);
  // Convention with the compiler side. We map bindings to kernel argument.
  // We compact the bindings to get a dense set of arguments and keep them order
  // based on the binding index.
  // Sort the binding based on the binding index and map the array index to the
  // argument index.
  iree_hal_level_zero_binding_mapping_t
      binding_used[IREE_HAL_LEVEL_ZERO_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_level_zero_binding_mapping_t buffer = {i, bindings[i].binding};
    binding_used[i] = buffer;
  }
  qsort(binding_used, binding_count,
        sizeof(iree_hal_level_zero_binding_mapping_t), compare_binding_index);
  assert(binding_count < IREE_HAL_LEVEL_ZERO_MAX_BINDING_COUNT &&
         "binding count larger than the max expected.");
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_descriptor_set_binding_t binding = bindings[binding_used[i].index];
    iree_hal_level_zero_device_ptr_t device_ptr =
        binding.buffer
            ? iree_hal_level_zero_buffer_device_pointer(
                  iree_hal_buffer_allocated_buffer(binding.buffer)) +
                  iree_hal_buffer_byte_offset(binding.buffer) + binding.offset
            : 0;
    *((iree_hal_level_zero_device_ptr_t*)
          command_buffer->current_descriptor[i + base_binding]) = device_ptr;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_hal_level_zero_direct_command_buffer_cast(base_command_buffer);
  iree_hal_pipeline_layout_t* layout =
      iree_hal_level_zero_executable_get_layout(executable, entry_point);
  iree_host_size_t num_constants =
      iree_hal_level_zero_pipeline_layout_num_constants(layout);
  iree_host_size_t constant_base_index =
      iree_hal_level_zero_push_constant_index(layout);

  int32_t block_size_x, block_size_y, block_size_z;
  IREE_RETURN_IF_ERROR(iree_hal_level_zero_native_executable_block_size(
      executable, entry_point, &block_size_x, &block_size_y, &block_size_z));
  ze_kernel_handle_t func =
      iree_hal_level_zero_native_executable_for_entry_point(executable,
                                                            entry_point);
  // TODO(raikonenfnu): Currently using NULL stream, need to figure out way to
  // access proper stream from command buffer
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeKernelSetGroupSize(func, block_size_x, block_size_y, block_size_z),
      "zeKernelSetGroupSize");

  // Patch the push constants in the kernel arguments.
  for (iree_host_size_t i = 0; i < num_constants; i++) {
    *((uint32_t*)command_buffer->current_descriptor[i + constant_base_index]) =
        command_buffer->push_constant[i];
  }
  iree_host_size_t num_kernel_args = constant_base_index + num_constants;
  for (iree_host_size_t i = 0; i < num_kernel_args; i++) {
    LEVEL_ZERO_RETURN_IF_ERROR(
        command_buffer->context->syms,
        zeKernelSetArgumentValue(func, i,
                                 sizeof(command_buffer->current_descriptor[i]),
                                 command_buffer->current_descriptor[i]),
        "zeKernelSetArgumentValue");
  }

  // Kernel thread-dispatch
  ze_group_count_t dispatch;
  dispatch.groupCountX = workgroup_x;
  dispatch.groupCountY = workgroup_y;
  dispatch.groupCountZ = workgroup_z;

  // Launch kernel on the GPU
  LEVEL_ZERO_RETURN_IF_ERROR(
      command_buffer->context->syms,
      zeCommandListAppendLaunchKernel(command_buffer->command_list, func,
                                      &dispatch, NULL, 0, NULL),
      "zeCommandListAppendLaunchKernel");
  return iree_ok_status();
}

static iree_status_t
iree_hal_level_zero_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need level_zero implementation");
}

ze_command_list_handle_t iree_hal_level_zero_direct_command_buffer_exec(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_level_zero_direct_command_buffer_t* command_buffer =
      (iree_hal_level_zero_direct_command_buffer_t*)
          iree_hal_command_buffer_dyn_cast(
              base_command_buffer,
              &iree_hal_level_zero_direct_command_buffer_vtable);
  IREE_ASSERT_TRUE(command_buffer);
  return command_buffer->command_list;
}

static iree_status_t iree_hal_rocm_direct_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_level_zero_direct_command_buffer_vtable = {
        .destroy = iree_hal_level_zero_direct_command_buffer_destroy,
        .dyn_cast = iree_hal_level_zero_direct_command_buffer_dyn_cast,
        .begin = iree_hal_level_zero_direct_command_buffer_begin,
        .end = iree_hal_level_zero_direct_command_buffer_end,
        .begin_debug_group =
            iree_hal_level_zero_direct_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_level_zero_direct_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_level_zero_direct_command_buffer_execution_barrier,
        .signal_event = iree_hal_level_zero_direct_command_buffer_signal_event,
        .reset_event = iree_hal_level_zero_direct_command_buffer_reset_event,
        .wait_events = iree_hal_level_zero_direct_command_buffer_wait_events,
        .discard_buffer =
            iree_hal_level_zero_direct_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_level_zero_direct_command_buffer_fill_buffer,
        .update_buffer =
            iree_hal_level_zero_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_level_zero_direct_command_buffer_copy_buffer,
        .push_constants =
            iree_hal_level_zero_direct_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_level_zero_direct_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_level_zero_direct_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_level_zero_direct_command_buffer_dispatch_indirect,
        .execute_commands =
            iree_hal_rocm_direct_command_buffer_execute_commands,
};
