// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "experimental/rocm/direct_command_buffer.h"

#include "experimental/rocm/native_executable.h"
#include "experimental/rocm/rocm_buffer.h"
#include "experimental/rocm/rocm_event.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/tracing.h"

// Command buffer implementation that directly maps to rocm direct.
// This records the commands on the calling thread without additional threading
// indirection.

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t *context;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;
  iree_hal_queue_affinity_t queue_affinity;
  size_t total_size;
  // Keep track of the current set of kernel arguments.
  void *current_descriptor[];
} iree_hal_rocm_direct_command_buffer_t;

static const size_t max_binding_count = 64;

extern const iree_hal_command_buffer_vtable_t
    iree_hal_rocm_direct_command_buffer_vtable;

static iree_hal_rocm_direct_command_buffer_t *
iree_hal_rocm_direct_command_buffer_cast(
    iree_hal_command_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_direct_command_buffer_vtable);
  return (iree_hal_rocm_direct_command_buffer_t *)base_value;
}

iree_status_t iree_hal_rocm_direct_command_buffer_allocate(
    iree_hal_rocm_context_wrapper_t *context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t **out_command_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_direct_command_buffer_t *command_buffer = NULL;
  size_t total_size = sizeof(*command_buffer) +
                      max_binding_count * sizeof(void *) +
                      max_binding_count * sizeof(hipDeviceptr_t);
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, total_size, (void **)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_rocm_direct_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->context = context;
    command_buffer->mode = mode;
    command_buffer->allowed_categories = command_categories;
    command_buffer->queue_affinity = queue_affinity;
    hipDeviceptr_t *device_ptrs =
        (hipDeviceptr_t *)(command_buffer->current_descriptor +
                           max_binding_count);
    for (size_t i = 0; i < max_binding_count; i++) {
      command_buffer->current_descriptor[i] = &device_ptrs[i];
    }
    command_buffer->total_size = total_size;

    *out_command_buffer = (iree_hal_command_buffer_t *)command_buffer;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_rocm_direct_command_buffer_destroy(
    iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_rocm_direct_command_buffer_t *command_buffer =
      iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(command_buffer->context->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_hal_command_buffer_mode_t iree_hal_rocm_direct_command_buffer_mode(
    const iree_hal_command_buffer_t *base_command_buffer) {
  const iree_hal_rocm_direct_command_buffer_t *command_buffer =
      (const iree_hal_rocm_direct_command_buffer_t *)(base_command_buffer);
  return command_buffer->mode;
}

static iree_hal_command_category_t
iree_hal_rocm_direct_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t *base_command_buffer) {
  const iree_hal_rocm_direct_command_buffer_t *command_buffer =
      (const iree_hal_rocm_direct_command_buffer_t *)(base_command_buffer);
  return command_buffer->allowed_categories;
}

static iree_status_t iree_hal_rocm_direct_command_buffer_begin(
    iree_hal_command_buffer_t *base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_end(
    iree_hal_command_buffer_t *base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  // TODO: Implement barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO: Implement barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO: Implement barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t **events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  // TODO: Implement barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_discard_buffer(
    iree_hal_command_buffer_t *base_command_buffer, iree_hal_buffer_t *buffer) {
  // nothing to do.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_rocm_splat_pattern(const void *pattern,
                                            size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t *)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t *)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t *)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_rocm_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length) {
  iree_hal_rocm_direct_command_buffer_t *command_buffer =
      iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);

  hipDeviceptr_t target_device_buffer = iree_hal_rocm_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  uint32_t dword_pattern = iree_hal_rocm_splat_pattern(pattern, pattern_length);
  hipDeviceptr_t dst = target_device_buffer + target_offset;
  int value = dword_pattern;
  size_t sizeBytes = length;
  // TODO(raikonenfnu): Currently using NULL stream, need to figure out way to
  // access proper stream from command buffer
  ROCM_RETURN_IF_ERROR(command_buffer->context->syms,
                       hipMemsetAsync(dst, value, sizeBytes, 0),
                       "hipMemsetAsync");
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t *base_command_buffer, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t *target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need rocm implementation");
}

static iree_status_t iree_hal_rocm_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_rocm_direct_command_buffer_t *command_buffer =
      iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);

  hipDeviceptr_t target_device_buffer = iree_hal_rocm_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  hipDeviceptr_t source_device_buffer = iree_hal_rocm_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  // TODO(raikonenfnu): Currently using NULL stream, need to figure out way to
  // access proper stream from command buffer
  ROCM_RETURN_IF_ERROR(
      command_buffer->context->syms,
      hipMemcpyAsync(target_device_buffer, source_device_buffer, length,
                     hipMemcpyDeviceToDevice, 0),
      "hipMemcpyAsync");
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_layout_t *executable_layout, iree_host_size_t offset,
    const void *values, iree_host_size_t values_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need rocm implementation");
}

static iree_status_t iree_hal_rocm_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_layout_t *executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t *bindings) {
  iree_hal_rocm_direct_command_buffer_t *command_buffer =
      iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    uint32_t arg_index = bindings[i].binding;
    assert(arg_index < max_binding_count &&
           "binding index larger than the max expected.");
    hipDeviceptr_t device_ptr =
        iree_hal_rocm_buffer_device_pointer(
            iree_hal_buffer_allocated_buffer(bindings[i].buffer)) +
        iree_hal_buffer_byte_offset(bindings[i].buffer) + bindings[i].offset;
    *((hipDeviceptr_t *)command_buffer->current_descriptor[arg_index]) =
        device_ptr;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_layout_t *executable_layout, uint32_t set,
    iree_hal_descriptor_set_t *descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t *dynamic_offsets) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need rocm implementation");
}

static iree_status_t iree_hal_rocm_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_t *executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_rocm_direct_command_buffer_t *command_buffer =
      iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);
  iree_hal_rocm_direct_command_buffer_cast(base_command_buffer);

  int32_t block_size_x, block_size_y, block_size_z;
  IREE_RETURN_IF_ERROR(iree_hal_rocm_native_executable_block_size(
      executable, entry_point, &block_size_x, &block_size_y, &block_size_z));
  int size = command_buffer->total_size;
  hipFunction_t func =
      iree_hal_rocm_native_executable_for_entry_point(executable, entry_point);
  // TODO(raikonenfnu): Currently using NULL stream, need to figure out way to
  // access proper stream from command buffer
  ROCM_RETURN_IF_ERROR(
      command_buffer->context->syms,
      hipModuleLaunchKernel(func, workgroup_x, workgroup_y, workgroup_z,
                            block_size_x, block_size_y, block_size_z, 0, 0,
                            command_buffer->current_descriptor, NULL),
      "hipModuleLaunchKernel");
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t *base_command_buffer,
    iree_hal_executable_t *executable, int32_t entry_point,
    iree_hal_buffer_t *workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need rocm implementation");
}

const iree_hal_command_buffer_vtable_t
    iree_hal_rocm_direct_command_buffer_vtable = {
        .destroy = iree_hal_rocm_direct_command_buffer_destroy,
        .mode = iree_hal_rocm_direct_command_buffer_mode,
        .allowed_categories =
            iree_hal_rocm_direct_command_buffer_allowed_categories,
        .begin = iree_hal_rocm_direct_command_buffer_begin,
        .end = iree_hal_rocm_direct_command_buffer_end,
        .execution_barrier =
            iree_hal_rocm_direct_command_buffer_execution_barrier,
        .signal_event = iree_hal_rocm_direct_command_buffer_signal_event,
        .reset_event = iree_hal_rocm_direct_command_buffer_reset_event,
        .wait_events = iree_hal_rocm_direct_command_buffer_wait_events,
        .discard_buffer = iree_hal_rocm_direct_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_rocm_direct_command_buffer_fill_buffer,
        .update_buffer = iree_hal_rocm_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_rocm_direct_command_buffer_copy_buffer,
        .push_constants = iree_hal_rocm_direct_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_rocm_direct_command_buffer_push_descriptor_set,
        .bind_descriptor_set =
            iree_hal_rocm_direct_command_buffer_bind_descriptor_set,
        .dispatch = iree_hal_rocm_direct_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_rocm_direct_command_buffer_dispatch_indirect,
};
