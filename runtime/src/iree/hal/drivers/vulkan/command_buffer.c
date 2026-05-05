// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/command_buffer.h"

#include <string.h>

#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/executable.h"

typedef enum iree_hal_vulkan_command_type_e {
  IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER = 0,
  IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER = 1,
  IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER = 2,
  IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH = 3,
  IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER = 4,
} iree_hal_vulkan_command_type_t;

typedef struct iree_hal_vulkan_command_t {
  // Recorded operation selector.
  iree_hal_vulkan_command_type_t type;

  // Recorded buffer-fill command payload.
  struct {
    // Target buffer reference captured during recording.
    iree_hal_buffer_ref_t target_ref;

    // Fill pattern bytes captured during recording.
    uint8_t pattern[4];

    // Number of bytes in pattern.
    iree_host_size_t pattern_length;

    // HAL fill flags captured during recording.
    iree_hal_fill_flags_t flags;
  } fill_buffer;

  // Recorded buffer-update command payload.
  struct {
    // Source bytes copied from the recording caller.
    void* source_data;

    // Target buffer reference captured during recording.
    iree_hal_buffer_ref_t target_ref;

    // Number of bytes copied into source_data.
    iree_host_size_t source_data_length;

    // HAL update flags captured during recording.
    iree_hal_update_flags_t flags;
  } update_buffer;

  // Recorded buffer-copy command payload.
  struct {
    // Source buffer reference captured during recording.
    iree_hal_buffer_ref_t source_ref;

    // Target buffer reference captured during recording.
    iree_hal_buffer_ref_t target_ref;

    // HAL copy flags captured during recording.
    iree_hal_copy_flags_t flags;
  } copy_buffer;

  // Recorded dispatch command payload.
  struct {
    // Executable retained until the command buffer is destroyed.
    iree_hal_executable_t* executable;

    // Executable export ordinal captured during recording.
    iree_hal_executable_export_ordinal_t export_ordinal;

    // Dispatch workgroup configuration captured during recording.
    iree_hal_dispatch_config_t config;

    // Push constant bytes copied from the recording caller.
    void* constants_data;

    // Number of bytes in constants_data.
    iree_host_size_t constants_data_length;

    // Buffer references copied from the recording caller.
    iree_hal_buffer_ref_t* bindings;

    // Number of entries in bindings.
    iree_host_size_t binding_count;

    // HAL dispatch flags captured during recording.
    iree_hal_dispatch_flags_t flags;
  } dispatch;

  // Recorded execution-barrier command payload.
  struct {
    // Number of memory barriers collapsed into the conservative native barrier.
    iree_host_size_t memory_barrier_count;

    // Number of buffer barriers collapsed into the conservative native barrier.
    iree_host_size_t buffer_barrier_count;
  } execution_barrier;
} iree_hal_vulkan_command_t;

typedef enum iree_hal_vulkan_command_buffer_state_e {
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL = 0,
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING = 1,
  IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED = 2,
} iree_hal_vulkan_command_buffer_state_t;

typedef struct iree_hal_vulkan_command_buffer_t {
  iree_hal_command_buffer_t base;

  // Host allocator used for command-buffer storage.
  iree_allocator_t host_allocator;

  // Current recording lifecycle state.
  iree_hal_vulkan_command_buffer_state_t state;

  // Whether a supported device command has been recorded.
  bool has_commands;

  // Whether any command must be replayed by the queue completion thread.
  bool has_host_commands;

  // Whether any command must be recorded into a native VkCommandBuffer.
  bool has_native_commands;

  // Recorded commands in submission order.
  iree_hal_vulkan_command_t* commands;

  // Number of commands populated in commands.
  iree_host_size_t command_count;

  // Capacity of commands.
  iree_host_size_t command_capacity;

  // Direct buffers retained by recorded commands.
  iree_hal_buffer_t** retained_buffers;

  // Number of entries populated in retained_buffers.
  iree_host_size_t retained_buffer_count;

  // Capacity of retained_buffers.
  iree_host_size_t retained_buffer_capacity;
} iree_hal_vulkan_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_command_buffer_vtable;

static iree_hal_vulkan_command_buffer_t* iree_hal_vulkan_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_command_buffer_vtable);
  return (iree_hal_vulkan_command_buffer_t*)base_value;
}

static iree_host_size_t iree_hal_vulkan_command_buffer_grow_capacity(
    iree_host_size_t current_capacity, iree_host_size_t minimum_capacity) {
  iree_host_size_t new_capacity = current_capacity ? current_capacity : 8;
  while (new_capacity < minimum_capacity) {
    if (new_capacity > IREE_HOST_SIZE_MAX / 2) {
      return minimum_capacity;
    }
    new_capacity = new_capacity * 2;
  }
  return new_capacity;
}

static iree_status_t iree_hal_vulkan_command_buffer_reserve_commands(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_host_size_t minimum_capacity) {
  if (minimum_capacity <= command_buffer->command_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t new_capacity =
      iree_hal_vulkan_command_buffer_grow_capacity(
          command_buffer->command_capacity, minimum_capacity);
  iree_host_size_t new_size = 0;
  if (!iree_host_size_checked_mul(
          new_capacity, sizeof(*command_buffer->commands), &new_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan command buffer command list is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc(command_buffer->host_allocator, new_size,
                             (void**)&command_buffer->commands));
  command_buffer->command_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_reserve_retained_buffers(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_host_size_t additional_count) {
  iree_host_size_t minimum_capacity = 0;
  if (!iree_host_size_checked_add(command_buffer->retained_buffer_count,
                                  additional_count, &minimum_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer retained-buffer list is too large");
  }
  if (minimum_capacity <= command_buffer->retained_buffer_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t new_capacity =
      iree_hal_vulkan_command_buffer_grow_capacity(
          command_buffer->retained_buffer_capacity, minimum_capacity);
  iree_host_size_t new_size = 0;
  if (!iree_host_size_checked_mul(
          new_capacity, sizeof(*command_buffer->retained_buffers), &new_size)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer retained-buffer list is too large");
  }
  IREE_RETURN_IF_ERROR(
      iree_allocator_realloc(command_buffer->host_allocator, new_size,
                             (void**)&command_buffer->retained_buffers));
  command_buffer->retained_buffer_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_validate_recording_state(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_string_view_t command_name) {
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer %.*s requires recording state",
        (int)command_name.size, command_name.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_validate_buffer_ref(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  if (buffer_ref.buffer) {
    return iree_ok_status();
  }
  if (buffer_ref.buffer_slot >= command_buffer->base.binding_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "indirect buffer reference slot %u is out range of the declared "
        "binding capacity of the Vulkan command buffer %u",
        buffer_ref.buffer_slot, command_buffer->base.binding_capacity);
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_command_buffer_record_buffer_ref(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  if (buffer_ref.buffer) return;
  command_buffer->base.binding_count =
      iree_max(command_buffer->base.binding_count, buffer_ref.buffer_slot + 1);
}

static void iree_hal_vulkan_command_buffer_append_retained_buffer(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_t* buffer) {
  if (!buffer) return;
  iree_hal_buffer_retain(buffer);
  command_buffer->retained_buffers[command_buffer->retained_buffer_count++] =
      buffer;
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_string_view_t usage,
    iree_hal_buffer_ref_t* out_resolved_ref) {
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, buffer_ref, out_resolved_ref));
  if (!out_resolved_ref->buffer) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer %.*s buffer reference resolved to NULL",
        (int)usage.size, usage.data);
  }
  return iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(out_resolved_ref->buffer),
      out_resolved_ref->offset, out_resolved_ref->length,
      &out_resolved_ref->offset, &out_resolved_ref->length);
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_string_view_t usage,
    VkBuffer* out_handle, VkDeviceSize* out_offset, VkDeviceSize* out_length) {
  iree_hal_buffer_ref_t resolved_ref;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
      binding_table, buffer_ref, usage, &resolved_ref));
  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      resolved_ref.buffer, &backing_buffer));
  if (!iree_hal_vulkan_buffer_isa(
          iree_hal_buffer_allocated_buffer(backing_buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer %.*s buffer reference is not backed by the "
        "Vulkan HAL rewrite",
        (int)usage.size, usage.data);
  }

  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(backing_buffer, &memory, &handle));
  (void)memory;

  iree_device_size_t absolute_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      resolved_ref.buffer, backing_buffer, resolved_ref.offset,
      &absolute_offset));

  *out_handle = handle;
  *out_offset = (VkDeviceSize)absolute_offset;
  *out_length = (VkDeviceSize)resolved_ref.length;
  return iree_ok_status();
}

static bool iree_hal_vulkan_command_buffer_can_fill_native(
    iree_hal_buffer_ref_t target_ref, iree_host_size_t pattern_length) {
  return (pattern_length == sizeof(uint8_t) ||
          pattern_length == sizeof(uint16_t) ||
          pattern_length == sizeof(uint32_t)) &&
         target_ref.length != IREE_HAL_WHOLE_BUFFER &&
         target_ref.offset % sizeof(uint32_t) == 0 &&
         target_ref.length % sizeof(uint32_t) == 0;
}

static iree_status_t iree_hal_vulkan_expand_fill_pattern(
    const uint8_t* pattern, iree_host_size_t pattern_length,
    uint32_t* out_expanded_pattern) {
  *out_expanded_pattern = 0;
  switch (pattern_length) {
    case sizeof(uint8_t):
      *out_expanded_pattern = pattern[0];
      *out_expanded_pattern |= *out_expanded_pattern << 8;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    case sizeof(uint16_t): {
      uint16_t pattern16 = 0;
      memcpy(&pattern16, pattern, sizeof(pattern16));
      *out_expanded_pattern = pattern16;
      *out_expanded_pattern |= *out_expanded_pattern << 16;
      break;
    }
    case sizeof(uint32_t):
      memcpy(out_expanded_pattern, pattern, sizeof(*out_expanded_pattern));
      break;
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Vulkan fill pattern length must be 1, 2, or 4 bytes (got %" PRIhsz
          ")",
          pattern_length);
  }
  return iree_ok_status();
}

static bool iree_hal_vulkan_command_buffer_can_update_native(
    iree_hal_buffer_ref_t target_ref) {
  return target_ref.length != IREE_HAL_WHOLE_BUFFER &&
         target_ref.offset % sizeof(uint32_t) == 0 &&
         target_ref.length % sizeof(uint32_t) == 0 &&
         target_ref.length <= 65536;
}

static iree_status_t
iree_hal_vulkan_command_buffer_validate_indirect_parameters_ref(
    iree_hal_vulkan_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t workgroup_count_ref) {
  const iree_device_size_t workgroup_count_length = sizeof(uint32_t[3]);
  if ((workgroup_count_ref.offset % sizeof(uint32_t)) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan indirect workgroup parameter offset must be 4-byte aligned");
  }
  if (workgroup_count_ref.length != IREE_HAL_WHOLE_BUFFER &&
      workgroup_count_ref.length < workgroup_count_length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan indirect workgroup parameter buffer must contain at least "
        "uint32_t[3]");
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
      command_buffer, workgroup_count_ref));
  if (!workgroup_count_ref.buffer) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroup_count_ref.buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroup_count_ref.buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroup_count_ref.buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  return iree_hal_buffer_validate_range(workgroup_count_ref.buffer,
                                        workgroup_count_ref.offset,
                                        workgroup_count_length);
}

iree_status_t iree_hal_vulkan_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t validation_state_size =
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  iree_hal_vulkan_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator,
                                sizeof(*command_buffer) + validation_state_size,
                                (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer) + validation_state_size);

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_vulkan_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL;
  *out_command_buffer = &command_buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

bool iree_hal_vulkan_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_vulkan_command_buffer_vtable);
}

bool iree_hal_vulkan_command_buffer_is_empty(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return !command_buffer->has_commands;
}

bool iree_hal_vulkan_command_buffer_has_host_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->has_host_commands;
}

bool iree_hal_vulkan_command_buffer_has_native_commands(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  return command_buffer->has_native_commands;
}

iree_status_t iree_hal_vulkan_command_buffer_replay_host(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan command buffer replay requires ended "
                            "state");
  }
  if (command_buffer->has_native_commands) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan command buffer contains native commands and cannot be "
        "host-replayed");
  }

  for (iree_host_size_t i = 0; i < command_buffer->command_count; ++i) {
    const iree_hal_vulkan_command_t* command = &command_buffer->commands[i];
    switch (command->type) {
      case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER: {
        if (command->fill_buffer.flags != IREE_HAL_FILL_FLAG_NONE) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "unsupported Vulkan command buffer fill flags: 0x%" PRIx64,
              command->fill_buffer.flags);
        }
        iree_hal_buffer_ref_t target_ref;
        IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
            binding_table, command->fill_buffer.target_ref, IREE_SV("target"),
            &target_ref));
        IREE_RETURN_IF_ERROR(iree_hal_buffer_map_fill(
            target_ref.buffer, target_ref.offset, target_ref.length,
            command->fill_buffer.pattern, command->fill_buffer.pattern_length));
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER: {
        if (command->update_buffer.flags != IREE_HAL_UPDATE_FLAG_NONE) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "unsupported Vulkan command buffer update flags: 0x%" PRIx64,
              command->update_buffer.flags);
        }
        iree_hal_buffer_ref_t target_ref;
        IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
            binding_table, command->update_buffer.target_ref, IREE_SV("target"),
            &target_ref));
        if (target_ref.length != command->update_buffer.source_data_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "resolved Vulkan command buffer update span differs from "
              "captured source data (target_length=%" PRIdsz
              ", source_length=%" PRIhsz ")",
              target_ref.length, command->update_buffer.source_data_length);
        }
        IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
            target_ref.buffer, target_ref.offset,
            command->update_buffer.source_data, target_ref.length));
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER: {
        if (command->copy_buffer.flags != IREE_HAL_COPY_FLAG_NONE) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "unsupported Vulkan command buffer copy flags: 0x%" PRIx64,
              command->copy_buffer.flags);
        }
        iree_hal_buffer_ref_t source_ref;
        IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
            binding_table, command->copy_buffer.source_ref, IREE_SV("source"),
            &source_ref));
        iree_hal_buffer_ref_t target_ref;
        IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_buffer_ref(
            binding_table, command->copy_buffer.target_ref, IREE_SV("target"),
            &target_ref));
        if (source_ref.length != target_ref.length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "resolved Vulkan command buffer copy spans differ "
              "(source_length=%" PRIdsz ", target_length=%" PRIdsz ")",
              source_ref.length, target_ref.length);
        }
        IREE_RETURN_IF_ERROR(iree_hal_buffer_map_copy(
            source_ref.buffer, source_ref.offset, target_ref.buffer,
            target_ref.offset, target_ref.length));
        break;
      }
      case IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER:
        break;
      default:
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "unknown Vulkan command buffer command kind %u",
                                (uint32_t)command->type);
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_vulkan_command_buffer_count_native_descriptor_pool_requirements(
    iree_hal_vulkan_command_buffer_t* command_buffer, uint32_t* out_max_sets,
    uint32_t* out_sampler_count, uint32_t* out_uniform_buffer_count,
    uint32_t* out_storage_buffer_count) {
  *out_max_sets = 0;
  *out_sampler_count = 0;
  *out_uniform_buffer_count = 0;
  *out_storage_buffer_count = 0;

  uint64_t max_set_count = 0;
  uint64_t sampler_count = 0;
  uint64_t uniform_buffer_count = 0;
  uint64_t storage_buffer_count = 0;
  for (iree_host_size_t i = 0; i < command_buffer->command_count; ++i) {
    const iree_hal_vulkan_command_t* command = &command_buffer->commands[i];
    if (command->type != IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH) continue;

    const iree_hal_vulkan_pipeline_t* pipeline = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
        command->dispatch.executable, command->dispatch.export_ordinal,
        &pipeline));
    max_set_count += pipeline->descriptor_set_layout_count;
    for (iree_host_size_t j = 0; j < pipeline->descriptor_binding_count; ++j) {
      switch (pipeline->descriptor_bindings[j].descriptor_type) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
          sampler_count += 1;
          break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
          uniform_buffer_count += 1;
          break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
          storage_buffer_count += 1;
          break;
        default:
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "unsupported Vulkan dispatch descriptor type %u",
              (uint32_t)pipeline->descriptor_bindings[j].descriptor_type);
      }
    }
  }

  if (max_set_count > UINT32_MAX || sampler_count > UINT32_MAX ||
      uniform_buffer_count > UINT32_MAX || storage_buffer_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer descriptor pool requirements exceed Vulkan "
        "limits");
  }

  *out_max_sets = (uint32_t)max_set_count;
  *out_sampler_count = (uint32_t)sampler_count;
  *out_uniform_buffer_count = (uint32_t)uniform_buffer_count;
  *out_storage_buffer_count = (uint32_t)storage_buffer_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_create_descriptor_pool(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_vulkan_command_buffer_t* command_buffer,
    VkDescriptorPool* out_descriptor_pool) {
  *out_descriptor_pool = VK_NULL_HANDLE;

  uint32_t max_set_count = 0;
  uint32_t sampler_count = 0;
  uint32_t uniform_buffer_count = 0;
  uint32_t storage_buffer_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_command_buffer_count_native_descriptor_pool_requirements(
          command_buffer, &max_set_count, &sampler_count, &uniform_buffer_count,
          &storage_buffer_count));
  if (max_set_count == 0) return iree_ok_status();

  VkDescriptorPoolSize pool_sizes[3];
  uint32_t pool_size_count = 0;
  if (sampler_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = sampler_count,
    };
  }
  if (uniform_buffer_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = uniform_buffer_count,
    };
  }
  if (storage_buffer_count != 0) {
    pool_sizes[pool_size_count++] = (VkDescriptorPoolSize){
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = storage_buffer_count,
    };
  }

  VkDescriptorPoolCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = max_set_count,
      .poolSizeCount = pool_size_count,
      .pPoolSizes = pool_sizes,
  };
  return iree_vkCreateDescriptorPool(IREE_VULKAN_DEVICE(syms), logical_device,
                                     &create_info, /*pAllocator=*/NULL,
                                     out_descriptor_pool);
}

static iree_status_t iree_hal_vulkan_command_buffer_resolve_descriptor_binding(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t buffer_ref, iree_host_size_t binding_ordinal,
    VkDescriptorType descriptor_type, VkDescriptorBufferInfo* out_buffer_info) {
  if (descriptor_type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
      descriptor_type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan dispatch binding %" PRIhsz
        " uses descriptor type %u that cannot be populated from a HAL buffer",
        binding_ordinal, (uint32_t)descriptor_type);
  }

  iree_hal_buffer_ref_t resolved_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, buffer_ref, &resolved_ref));
  if (!resolved_ref.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan dispatch binding %" PRIhsz
                            " buffer reference resolved to NULL",
                            binding_ordinal);
  }

  iree_hal_buffer_t* backing_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing(
      resolved_ref.buffer, &backing_buffer));

  iree_device_size_t buffer_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_backing_offset(
      resolved_ref.buffer, backing_buffer, /*local_byte_offset=*/0,
      &buffer_byte_offset));
  const iree_device_size_t buffer_byte_length =
      iree_hal_buffer_byte_length(resolved_ref.buffer);
  iree_device_size_t absolute_buffer_end = 0;
  if (!iree_device_size_checked_add(buffer_byte_offset, buffer_byte_length,
                                    &absolute_buffer_end)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "Vulkan dispatch binding %" PRIhsz
                            " absolute buffer range overflows",
                            binding_ordinal);
  }

  iree_device_size_t descriptor_offset = 0;
  iree_device_size_t descriptor_length = 0;
  // Some callers pass allocation-absolute offsets through binding tables. Keep
  // ordinary view-relative offsets on the documented path and only accept
  // absolute offsets after they have clearly escaped the view range.
  const bool offset_in_absolute_range =
      resolved_ref.offset > buffer_byte_length &&
      resolved_ref.offset >= buffer_byte_offset &&
      resolved_ref.offset <= absolute_buffer_end;
  if (offset_in_absolute_range) {
    descriptor_offset = resolved_ref.offset;
    if (resolved_ref.length == IREE_HAL_WHOLE_BUFFER) {
      descriptor_length = absolute_buffer_end - resolved_ref.offset;
    } else if (resolved_ref.length <=
               absolute_buffer_end - resolved_ref.offset) {
      descriptor_length = resolved_ref.length;
    } else {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Vulkan dispatch binding %" PRIhsz
          " absolute descriptor range exceeds the valid buffer range",
          binding_ordinal);
    }
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
        /*base_offset=*/0, buffer_byte_length, resolved_ref.offset,
        resolved_ref.length, &descriptor_offset, &descriptor_length));
    if (!iree_device_size_checked_add(buffer_byte_offset, descriptor_offset,
                                      &descriptor_offset)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "Vulkan dispatch binding %" PRIhsz
                              " descriptor offset overflows",
                              binding_ordinal);
    }
  }
  if (descriptor_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan dispatch binding %" PRIhsz
                            " resolved to an empty buffer range",
                            binding_ordinal);
  }

  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer buffer = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_buffer_handle(backing_buffer, &memory, &buffer));
  (void)memory;

  *out_buffer_info = (VkDescriptorBufferInfo){
      .buffer = buffer,
      .offset = (VkDeviceSize)descriptor_offset,
      .range = (VkDeviceSize)descriptor_length,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_fill_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command) {
  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, command->fill_buffer.target_ref, IREE_SV("fill target"),
      &target_handle, &target_offset, &target_length));
  if (target_length == 0) return iree_ok_status();

  iree_hal_buffer_ref_t resolved_target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, command->fill_buffer.target_ref, &resolved_target_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_target_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  if (target_offset % sizeof(uint32_t) != 0 ||
      target_length % sizeof(uint32_t) != 0 ||
      !iree_hal_vulkan_command_buffer_can_fill_native(
          (iree_hal_buffer_ref_t){
              .buffer = resolved_target_ref.buffer,
              .offset = target_offset,
              .length = target_length,
          },
          command->fill_buffer.pattern_length)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native fill requires 4-byte target alignment, 4-byte length, "
        "and a 1-, 2-, or 4-byte pattern");
  }

  uint32_t pattern = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_expand_fill_pattern(
      command->fill_buffer.pattern, command->fill_buffer.pattern_length,
      &pattern));
  iree_vkCmdFillBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       target_handle, target_offset, target_length, pattern);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_update_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command) {
  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, command->update_buffer.target_ref,
      IREE_SV("update target"), &target_handle, &target_offset,
      &target_length));
  if (target_length == 0) return iree_ok_status();
  if (target_length != command->update_buffer.source_data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resolved Vulkan command buffer update span differs from captured "
        "source data (target_length=%" PRIu64 ", source_length=%" PRIhsz ")",
        (uint64_t)target_length, command->update_buffer.source_data_length);
  }
  if (target_offset % sizeof(uint32_t) != 0 ||
      target_length % sizeof(uint32_t) != 0 || target_length > 65536) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Vulkan native update requires 4-byte target alignment, 4-byte "
        "length, and at most 65536 bytes");
  }

  iree_vkCmdUpdateBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                         target_handle, target_offset, target_length,
                         command->update_buffer.source_data);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_copy_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command) {
  VkBuffer source_handle = VK_NULL_HANDLE;
  VkDeviceSize source_offset = 0;
  VkDeviceSize source_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, command->copy_buffer.source_ref, IREE_SV("copy source"),
      &source_handle, &source_offset, &source_length));
  VkBuffer target_handle = VK_NULL_HANDLE;
  VkDeviceSize target_offset = 0;
  VkDeviceSize target_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, command->copy_buffer.target_ref, IREE_SV("copy target"),
      &target_handle, &target_offset, &target_length));
  if (source_length != target_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved Vulkan command buffer copy spans differ "
                            "(source_length=%" PRIu64 ", target_length=%" PRIu64
                            ")",
                            (uint64_t)source_length, (uint64_t)target_length);
  }
  if (source_length == 0) return iree_ok_status();

  iree_hal_buffer_ref_t resolved_source_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, command->copy_buffer.source_ref, &resolved_source_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_source_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  iree_hal_buffer_ref_t resolved_target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, command->copy_buffer.target_ref, &resolved_target_ref));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(resolved_target_ref.buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));

  VkBufferCopy copy_region = {
      .srcOffset = source_offset,
      .dstOffset = target_offset,
      .size = source_length,
  };
  iree_vkCmdCopyBuffer(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       source_handle, target_handle, /*regionCount=*/1,
                       &copy_region);
  return iree_ok_status();
}

static void iree_hal_vulkan_command_buffer_record_execution_barrier_native(
    const iree_hal_vulkan_device_syms_t* syms,
    VkCommandBuffer native_command_buffer) {
  VkMemoryBarrier2 memory_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask =
          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
  };
  VkDependencyInfo dependency_info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory_barrier,
  };
  iree_vkCmdPipelineBarrier2(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                             &dependency_info);
}

static iree_status_t
iree_hal_vulkan_command_buffer_resolve_indirect_parameters_buffer(
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t workgroup_count_ref, VkBuffer* out_handle,
    VkDeviceSize* out_offset) {
  VkBuffer parameter_handle = VK_NULL_HANDLE;
  VkDeviceSize parameter_offset = 0;
  VkDeviceSize parameter_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_resolve_native_buffer_ref(
      binding_table, workgroup_count_ref, IREE_SV("indirect parameters"),
      &parameter_handle, &parameter_offset, &parameter_length));
  if (parameter_offset % sizeof(uint32_t) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan indirect workgroup parameter offset must be 4-byte aligned");
  }
  if (parameter_length < sizeof(uint32_t[3])) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan indirect workgroup parameter buffer must contain at least "
        "uint32_t[3]");
  }

  *out_handle = parameter_handle;
  *out_offset = parameter_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_record_dispatch_native(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_t* command, VkDescriptorPool descriptor_pool,
    iree_allocator_t host_allocator) {
  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      command->dispatch.executable, command->dispatch.export_ordinal,
      &pipeline));
  if (command->dispatch.binding_count != pipeline->descriptor_binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan dispatch recorded %" PRIhsz
                            " bindings but pipeline expects %" PRIhsz,
                            command->dispatch.binding_count,
                            pipeline->descriptor_binding_count);
  }

  VkDescriptorSet* descriptor_sets = NULL;
  if (pipeline->descriptor_set_layout_count != 0) {
    if (!descriptor_pool) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "Vulkan dispatch requires descriptor sets but no pool is available");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
        host_allocator, pipeline->descriptor_set_layout_count,
        sizeof(descriptor_sets[0]), (void**)&descriptor_sets));
    VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = (uint32_t)pipeline->descriptor_set_layout_count,
        .pSetLayouts = pipeline->descriptor_set_layouts,
    };
    iree_status_t status =
        iree_vkAllocateDescriptorSets(IREE_VULKAN_DEVICE(syms), logical_device,
                                      &allocate_info, descriptor_sets);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, descriptor_sets);
      return status;
    }
  }

  VkDescriptorBufferInfo* buffer_infos = NULL;
  VkWriteDescriptorSet* write_infos = NULL;
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status) && pipeline->descriptor_binding_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, pipeline->descriptor_binding_count,
        sizeof(buffer_infos[0]), (void**)&buffer_infos);
  }
  if (iree_status_is_ok(status) && pipeline->descriptor_binding_count != 0) {
    status = iree_allocator_malloc_array(
        host_allocator, pipeline->descriptor_binding_count,
        sizeof(write_infos[0]), (void**)&write_infos);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < pipeline->descriptor_binding_count;
       ++i) {
    const iree_hal_vulkan_descriptor_binding_t* descriptor_binding =
        &pipeline->descriptor_bindings[i];
    status = iree_hal_vulkan_command_buffer_resolve_descriptor_binding(
        binding_table, command->dispatch.bindings[i], i,
        descriptor_binding->descriptor_type, &buffer_infos[i]);
    if (iree_status_is_ok(status)) {
      write_infos[i] = (VkWriteDescriptorSet){
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = descriptor_sets[descriptor_binding->set_ordinal],
          .dstBinding = descriptor_binding->binding,
          .dstArrayElement = descriptor_binding->array_element,
          .descriptorCount = 1,
          .descriptorType = descriptor_binding->descriptor_type,
          .pBufferInfo = &buffer_infos[i],
      };
    }
  }
  if (iree_status_is_ok(status) && pipeline->descriptor_binding_count != 0) {
    iree_vkUpdateDescriptorSets(
        IREE_VULKAN_DEVICE(syms), logical_device,
        (uint32_t)pipeline->descriptor_binding_count, write_infos,
        /*descriptorCopyCount=*/0, /*pDescriptorCopies=*/NULL);
  }

  iree_allocator_free(host_allocator, write_infos);
  iree_allocator_free(host_allocator, buffer_infos);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, descriptor_sets);
    return status;
  }

  if (pipeline->descriptor_set_layout_count != 0) {
    iree_vkCmdBindDescriptorSets(
        IREE_VULKAN_DEVICE(syms), native_command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout, /*firstSet=*/0,
        (uint32_t)pipeline->descriptor_set_layout_count, descriptor_sets,
        /*dynamicOffsetCount=*/0, /*pDynamicOffsets=*/NULL);
  }
  if (command->dispatch.constants_data_length != 0) {
    iree_vkCmdPushConstants(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                            pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                            /*offset=*/0,
                            (uint32_t)command->dispatch.constants_data_length,
                            command->dispatch.constants_data);
  }
  iree_vkCmdBindPipeline(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                         VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
  if (iree_hal_dispatch_uses_indirect_parameters(command->dispatch.flags)) {
    VkBuffer parameter_handle = VK_NULL_HANDLE;
    VkDeviceSize parameter_offset = 0;
    status = iree_hal_vulkan_command_buffer_resolve_indirect_parameters_buffer(
        binding_table, command->dispatch.config.workgroup_count_ref,
        &parameter_handle, &parameter_offset);
    if (iree_status_is_ok(status)) {
      iree_vkCmdDispatchIndirect(IREE_VULKAN_DEVICE(syms),
                                 native_command_buffer, parameter_handle,
                                 parameter_offset);
    }
  } else {
    iree_vkCmdDispatch(IREE_VULKAN_DEVICE(syms), native_command_buffer,
                       command->dispatch.config.workgroup_count[0],
                       command->dispatch.config.workgroup_count[1],
                       command->dispatch.config.workgroup_count[2]);
  }

  iree_allocator_free(host_allocator, descriptor_sets);
  return status;
}

iree_status_t iree_hal_vulkan_command_buffer_record_native(
    iree_hal_command_buffer_t* base_command_buffer,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_allocator_t host_allocator, VkDescriptorPool* out_descriptor_pool) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(native_command_buffer);
  IREE_ASSERT_ARGUMENT(out_descriptor_pool);
  *out_descriptor_pool = VK_NULL_HANDLE;

  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan native command buffer recording requires "
                            "ended state");
  }
  if (!command_buffer->has_native_commands) return iree_ok_status();
  if (command_buffer->has_host_commands) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "mixed host-replayed and native Vulkan command "
                            "buffers are unsupported");
  }

  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  iree_status_t status = iree_hal_vulkan_command_buffer_create_descriptor_pool(
      syms, logical_device, command_buffer, &descriptor_pool);

  if (iree_status_is_ok(status)) {
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    status = iree_vkBeginCommandBuffer(IREE_VULKAN_DEVICE(syms),
                                       native_command_buffer, &begin_info);
  }
  for (iree_host_size_t i = 0;
       iree_status_is_ok(status) && i < command_buffer->command_count; ++i) {
    const iree_hal_vulkan_command_t* command = &command_buffer->commands[i];
    switch (command->type) {
      case IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_fill_native(
            syms, native_command_buffer, binding_table, command);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_update_native(
            syms, native_command_buffer, binding_table, command);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER:
        status = iree_hal_vulkan_command_buffer_record_copy_native(
            syms, native_command_buffer, binding_table, command);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH:
        status = iree_hal_vulkan_command_buffer_record_dispatch_native(
            syms, logical_device, native_command_buffer, binding_table, command,
            descriptor_pool, host_allocator);
        break;
      case IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER:
        iree_hal_vulkan_command_buffer_record_execution_barrier_native(
            syms, native_command_buffer);
        break;
      default:
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "Vulkan command buffer contains non-native "
                                  "command kind %u",
                                  (uint32_t)command->type);
        break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_vkEndCommandBuffer(IREE_VULKAN_DEVICE(syms),
                                     native_command_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_descriptor_pool = descriptor_pool;
  } else if (descriptor_pool) {
    iree_vkDestroyDescriptorPool(IREE_VULKAN_DEVICE(syms), logical_device,
                                 descriptor_pool, /*pAllocator=*/NULL);
  }
  return status;
}

static void iree_hal_vulkan_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  for (iree_host_size_t i = 0; i < command_buffer->command_count; ++i) {
    iree_allocator_free(host_allocator,
                        command_buffer->commands[i].update_buffer.source_data);
    iree_allocator_free(host_allocator,
                        command_buffer->commands[i].dispatch.constants_data);
    iree_allocator_free(host_allocator,
                        command_buffer->commands[i].dispatch.bindings);
    if (command_buffer->commands[i].dispatch.executable) {
      iree_hal_executable_release(
          command_buffer->commands[i].dispatch.executable);
    }
  }
  for (iree_host_size_t i = 0; i < command_buffer->retained_buffer_count; ++i) {
    iree_hal_buffer_release(command_buffer->retained_buffers[i]);
  }
  iree_allocator_free(host_allocator, command_buffer->retained_buffers);
  iree_allocator_free(host_allocator, command_buffer->commands);
  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t iree_hal_vulkan_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_INITIAL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan command buffer begin requires initial "
                            "state");
  }
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  if (command_buffer->state != IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_RECORDING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan command buffer end requires recording "
                            "state");
  }
  command_buffer->state = IREE_HAL_VULKAN_COMMAND_BUFFER_STATE_ENDED;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  (void)base_command_buffer;
  (void)label;
  (void)label_color;
  (void)location;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barriers;
  (void)buffer_barriers;
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("execution_barrier")));
  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer execution barrier flags: 0x%" PRIx64,
        flags);
  }
  if (memory_barrier_count == 0 && buffer_barrier_count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_commands(
      command_buffer, command_buffer->command_count + 1));
  iree_hal_vulkan_command_t* command =
      &command_buffer->commands[command_buffer->command_count++];
  memset(command, 0, sizeof(*command));
  command->type = IREE_HAL_VULKAN_COMMAND_TYPE_EXECUTION_BARRIER;
  command->execution_barrier.memory_barrier_count = memory_barrier_count;
  command->execution_barrier.buffer_barrier_count = buffer_barrier_count;
  command_buffer->has_commands = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "Vulkan command buffer event signals are unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer event resets are unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)base_command_buffer;
  (void)event_count;
  (void)events;
  (void)source_stage_mask;
  (void)target_stage_mask;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer event waits are unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  (void)base_command_buffer;
  (void)buffer_ref;
  (void)flags;
  (void)arg0;
  (void)arg1;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer memory advice is unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("fill_buffer")));
  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer fill flags: 0x%" PRIx64, flags);
  }
  if (!pattern) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command buffer fill pattern is NULL");
  }
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer fill pattern length must be 1, 2, or 4 bytes "
        "(got %" PRIhsz ")",
        pattern_length);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
      command_buffer, target_ref));

  const bool retain_resources = !iree_all_bits_set(
      command_buffer->base.mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
  const iree_host_size_t retained_buffer_count =
      retain_resources && target_ref.buffer ? 1 : 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_retained_buffers(
      command_buffer, retained_buffer_count));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_commands(
      command_buffer, command_buffer->command_count + 1));

  iree_hal_vulkan_command_buffer_record_buffer_ref(command_buffer, target_ref);
  if (retain_resources) {
    iree_hal_vulkan_command_buffer_append_retained_buffer(command_buffer,
                                                          target_ref.buffer);
  }

  iree_hal_vulkan_command_t* command =
      &command_buffer->commands[command_buffer->command_count++];
  memset(command, 0, sizeof(*command));
  command->type = IREE_HAL_VULKAN_COMMAND_TYPE_FILL_BUFFER;
  command->fill_buffer.target_ref = target_ref;
  memset(command->fill_buffer.pattern, 0, sizeof(command->fill_buffer.pattern));
  memcpy(command->fill_buffer.pattern, pattern, pattern_length);
  command->fill_buffer.pattern_length = pattern_length;
  command->fill_buffer.flags = flags;
  command_buffer->has_commands = true;
  if (iree_hal_vulkan_command_buffer_can_fill_native(target_ref,
                                                     pattern_length)) {
    command_buffer->has_native_commands = true;
  } else {
    command_buffer->has_host_commands = true;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("update_buffer")));
  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer update flags: 0x%" PRIx64, flags);
  }
  if (!source_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command buffer update source is NULL");
  }
  if (target_ref.length == IREE_HAL_WHOLE_BUFFER) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer updates require an explicit target length");
  }
  if (target_ref.length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer update target length exceeds host size");
  }
  if (source_offset >
      IREE_HOST_SIZE_MAX - (iree_host_size_t)target_ref.length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer update source range exceeds host size");
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
      command_buffer, target_ref));

  const bool retain_resources = !iree_all_bits_set(
      command_buffer->base.mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
  const iree_host_size_t retained_buffer_count =
      retain_resources && target_ref.buffer ? 1 : 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_retained_buffers(
      command_buffer, retained_buffer_count));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_commands(
      command_buffer, command_buffer->command_count + 1));

  iree_hal_vulkan_command_t* command =
      &command_buffer->commands[command_buffer->command_count];
  memset(command, 0, sizeof(*command));
  if (target_ref.length > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        command_buffer->host_allocator, (iree_host_size_t)target_ref.length,
        &command->update_buffer.source_data));
    memcpy(command->update_buffer.source_data,
           (const uint8_t*)source_buffer + source_offset,
           (iree_host_size_t)target_ref.length);
  }

  iree_hal_vulkan_command_buffer_record_buffer_ref(command_buffer, target_ref);
  if (retain_resources) {
    iree_hal_vulkan_command_buffer_append_retained_buffer(command_buffer,
                                                          target_ref.buffer);
  }

  command->type = IREE_HAL_VULKAN_COMMAND_TYPE_UPDATE_BUFFER;
  command->update_buffer.target_ref = target_ref;
  command->update_buffer.source_data_length =
      (iree_host_size_t)target_ref.length;
  command->update_buffer.flags = flags;
  command_buffer->command_count = command_buffer->command_count + 1;
  command_buffer->has_commands = true;
  if (iree_hal_vulkan_command_buffer_can_update_native(target_ref)) {
    command_buffer->has_native_commands = true;
  } else {
    command_buffer->has_host_commands = true;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("copy_buffer")));
  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer copy flags: 0x%" PRIx64, flags);
  }
  if (source_ref.length != target_ref.length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy spans between source and target must match "
                            "(source_length=%" PRIdsz ", target_length=%" PRIdsz
                            ")",
                            source_ref.length, target_ref.length);
  }
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
      command_buffer, source_ref));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
      command_buffer, target_ref));

  const bool retain_resources = !iree_all_bits_set(
      command_buffer->base.mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
  const iree_host_size_t retained_buffer_count =
      retain_resources
          ? (source_ref.buffer ? 1 : 0) + (target_ref.buffer ? 1 : 0)
          : 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_retained_buffers(
      command_buffer, retained_buffer_count));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_commands(
      command_buffer, command_buffer->command_count + 1));

  iree_hal_vulkan_command_buffer_record_buffer_ref(command_buffer, source_ref);
  iree_hal_vulkan_command_buffer_record_buffer_ref(command_buffer, target_ref);
  if (retain_resources) {
    iree_hal_vulkan_command_buffer_append_retained_buffer(command_buffer,
                                                          source_ref.buffer);
    iree_hal_vulkan_command_buffer_append_retained_buffer(command_buffer,
                                                          target_ref.buffer);
  }

  iree_hal_vulkan_command_t* command =
      &command_buffer->commands[command_buffer->command_count++];
  memset(command, 0, sizeof(*command));
  command->type = IREE_HAL_VULKAN_COMMAND_TYPE_COPY_BUFFER;
  command->copy_buffer.source_ref = source_ref;
  command->copy_buffer.target_ref = target_ref;
  command->copy_buffer.flags = flags;
  command_buffer->has_commands = true;
  command_buffer->has_native_commands = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  (void)base_command_buffer;
  (void)channel;
  (void)op;
  (void)param;
  (void)send_ref;
  (void)recv_ref;
  (void)element_count;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "Vulkan command buffer collectives are unsupported");
}

static iree_status_t iree_hal_vulkan_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_vulkan_command_buffer_t* command_buffer =
      iree_hal_vulkan_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_recording_state(
      command_buffer, IREE_SV("dispatch")));
  if (iree_hal_dispatch_uses_indirect_arguments(flags) ||
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer custom dispatch arguments are unsupported");
  }
  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (iree_any_bit_set(flags, ~supported_flags)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported Vulkan command buffer dispatch flags: 0x%" PRIx64, flags);
  }
  const iree_hal_dispatch_flags_t indirect_parameter_flags =
      flags & (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
               IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS);
  if (indirect_parameter_flags ==
      (IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
       IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan dispatch cannot use both static and dynamic indirect "
        "workgroup parameters");
  }
  if (config.workgroup_size[0] != 0 || config.workgroup_size[1] != 0 ||
      config.workgroup_size[2] != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer dispatch workgroup size overrides are "
        "unsupported");
  }
  if (config.dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Vulkan command buffer dispatch dynamic workgroup local memory is "
        "unsupported");
  }
  if (constants.data_length % sizeof(uint32_t) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer dispatch constants must be 4-byte aligned");
  }
  if (constants.data_length > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer dispatch constants exceed Vulkan limit %u",
        UINT32_MAX);
  }
  if (bindings.count != 0 && !bindings.values) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer dispatch binding storage is NULL");
  }
  if (!iree_hal_vulkan_executable_isa(executable)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Vulkan command buffer dispatch executable is not a Vulkan executable");
  }

  const iree_hal_vulkan_pipeline_t* pipeline = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_executable_lookup_pipeline(
      executable, export_ordinal, &pipeline));
  if (constants.data_length >
      (iree_host_size_t)pipeline->constant_count * sizeof(uint32_t)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "Vulkan command buffer dispatch provides %" PRIhsz
        " constant bytes but pipeline accepts at most %u",
        constants.data_length,
        (uint32_t)pipeline->constant_count * (uint32_t)sizeof(uint32_t));
  }
  if (bindings.count != pipeline->descriptor_binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Vulkan command buffer dispatch provides %" PRIhsz
                            " bindings but pipeline expects %" PRIhsz,
                            bindings.count, pipeline->descriptor_binding_count);
  }
  for (iree_host_size_t i = 0; i < pipeline->descriptor_binding_count; ++i) {
    if (pipeline->descriptor_bindings[i].descriptor_type !=
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
        pipeline->descriptor_bindings[i].descriptor_type !=
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "Vulkan command buffer dispatch descriptor type %u is unsupported",
          (uint32_t)pipeline->descriptor_bindings[i].descriptor_type);
    }
  }

  const bool retain_resources = !iree_all_bits_set(
      command_buffer->base.mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED);
  iree_host_size_t retained_buffer_count = 0;
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(
        iree_hal_vulkan_command_buffer_validate_indirect_parameters_ref(
            command_buffer, config.workgroup_count_ref));
    if (retain_resources && config.workgroup_count_ref.buffer) {
      retained_buffer_count = retained_buffer_count + 1;
    }
  }
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_validate_buffer_ref(
        command_buffer, bindings.values[i]));
    if (retain_resources && bindings.values[i].buffer) {
      retained_buffer_count = retained_buffer_count + 1;
    }
  }

  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_retained_buffers(
      command_buffer, retained_buffer_count));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_command_buffer_reserve_commands(
      command_buffer, command_buffer->command_count + 1));

  void* constants_data = NULL;
  iree_hal_buffer_ref_t* binding_refs = NULL;
  iree_status_t status = iree_ok_status();
  if (constants.data_length != 0) {
    status = iree_allocator_malloc(command_buffer->host_allocator,
                                   constants.data_length, &constants_data);
    if (iree_status_is_ok(status)) {
      memcpy(constants_data, constants.data, constants.data_length);
    }
  }
  if (iree_status_is_ok(status) && bindings.count != 0) {
    status = iree_allocator_malloc_array(
        command_buffer->host_allocator, bindings.count, sizeof(binding_refs[0]),
        (void**)&binding_refs);
    if (iree_status_is_ok(status)) {
      memcpy(binding_refs, bindings.values,
             bindings.count * sizeof(binding_refs[0]));
    }
  }

  if (iree_status_is_ok(status)) {
    if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
      iree_hal_vulkan_command_buffer_record_buffer_ref(
          command_buffer, config.workgroup_count_ref);
      if (retain_resources) {
        iree_hal_vulkan_command_buffer_append_retained_buffer(
            command_buffer, config.workgroup_count_ref.buffer);
      }
    }
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_vulkan_command_buffer_record_buffer_ref(command_buffer,
                                                       bindings.values[i]);
      if (retain_resources) {
        iree_hal_vulkan_command_buffer_append_retained_buffer(
            command_buffer, bindings.values[i].buffer);
      }
    }

    iree_hal_vulkan_command_t* command =
        &command_buffer->commands[command_buffer->command_count++];
    memset(command, 0, sizeof(*command));
    command->type = IREE_HAL_VULKAN_COMMAND_TYPE_DISPATCH;
    command->dispatch.executable = executable;
    iree_hal_executable_retain(executable);
    command->dispatch.export_ordinal = export_ordinal;
    command->dispatch.config = config;
    command->dispatch.constants_data = constants_data;
    command->dispatch.constants_data_length = constants.data_length;
    command->dispatch.bindings = binding_refs;
    command->dispatch.binding_count = bindings.count;
    command->dispatch.flags = flags;
    command_buffer->has_commands = true;
    command_buffer->has_native_commands = true;
    constants_data = NULL;
    binding_refs = NULL;
  }

  iree_allocator_free(command_buffer->host_allocator, binding_refs);
  iree_allocator_free(command_buffer->host_allocator, constants_data);
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_command_buffer_vtable = {
        .destroy = iree_hal_vulkan_command_buffer_destroy,
        .begin = iree_hal_vulkan_command_buffer_begin,
        .end = iree_hal_vulkan_command_buffer_end,
        .begin_debug_group = iree_hal_vulkan_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_vulkan_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_vulkan_command_buffer_execution_barrier,
        .signal_event = iree_hal_vulkan_command_buffer_signal_event,
        .reset_event = iree_hal_vulkan_command_buffer_reset_event,
        .wait_events = iree_hal_vulkan_command_buffer_wait_events,
        .advise_buffer = iree_hal_vulkan_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_vulkan_command_buffer_fill_buffer,
        .update_buffer = iree_hal_vulkan_command_buffer_update_buffer,
        .copy_buffer = iree_hal_vulkan_command_buffer_copy_buffer,
        .collective = iree_hal_vulkan_command_buffer_collective,
        .dispatch = iree_hal_vulkan_command_buffer_dispatch,
};
