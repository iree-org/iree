// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"

#define IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_STACK_BINDING_CAPACITY 16
#define IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_STACK_RESOURCE_CAPACITY 17

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect workgroup parameters are not supported by AMDGPU "
        "queue_dispatch yet");
  }
  if (iree_hal_dispatch_uses_indirect_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect dispatch arguments are not supported by AMDGPU "
        "queue_dispatch yet");
  }

  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~supported_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported dispatch flags: 0x%" PRIx64, flags);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_dispatch_kernel_args(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config,
    iree_hal_amdgpu_device_kernel_args_t* out_kernel_args) {
  const iree_hal_amdgpu_device_kernel_args_t* device_kernel_args = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_executable_lookup_kernel_args_for_device(
      executable, export_ordinal, queue->device_ordinal, &device_kernel_args));
  if (IREE_UNLIKELY(!device_kernel_args)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable has no kernel arguments for device ordinal %" PRIhsz,
        queue->device_ordinal);
  }

  *out_kernel_args = *device_kernel_args;
  const bool has_workgroup_size_override = config.workgroup_size[0] ||
                                           config.workgroup_size[1] ||
                                           config.workgroup_size[2];
  if (has_workgroup_size_override) {
    if (IREE_UNLIKELY(!config.workgroup_size[0] || !config.workgroup_size[1] ||
                      !config.workgroup_size[2])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch workgroup size override must specify all dimensions");
    }
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(config.workgroup_size[i] > UINT16_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch workgroup size override dimension %" PRIhsz
            " value %u exceeds %u",
            i, config.workgroup_size[i], UINT16_MAX);
      }
      out_kernel_args->workgroup_size[i] = (uint16_t)config.workgroup_size[i];
    }
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_shape(
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const uint32_t workgroup_count[3],
    uint32_t dynamic_workgroup_local_memory) {
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(!kernel_args->workgroup_size[i])) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable kernel workgroup size dimension %" PRIhsz " is zero", i);
    }
    const uint64_t grid_size =
        (uint64_t)workgroup_count[i] * kernel_args->workgroup_size[i];
    if (IREE_UNLIKELY(grid_size > UINT32_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch grid dimension %" PRIhsz
          " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
          i, workgroup_count[i], kernel_args->workgroup_size[i]);
    }
  }
  const uint64_t group_segment_size =
      (uint64_t)kernel_args->group_segment_size +
      dynamic_workgroup_local_memory;
  if (IREE_UNLIKELY(group_segment_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch group segment size overflows uint32_t "
                            "(static=%u, dynamic=%u)",
                            kernel_args->group_segment_size,
                            dynamic_workgroup_local_memory);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_binding(
    const iree_hal_buffer_ref_t* binding) {
  if (IREE_UNLIKELY(binding->reserved != 0 || binding->buffer_slot != 0 ||
                    !binding->buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_dispatch bindings must be direct non-null buffer references");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(binding->buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(binding->buffer),
      IREE_HAL_MEMORY_ACCESS_ANY));
  return iree_hal_buffer_validate_range(binding->buffer, binding->offset,
                                        binding->length);
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_kernargs(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    iree_const_byte_span_t constants, const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_device_dispatch_kernarg_layout_t* out_layout,
    uint32_t* out_kernarg_block_count,
    iree_host_size_t* out_operation_resource_count) {
  *out_kernarg_block_count = 0;
  *out_operation_resource_count = 0;

  if (IREE_UNLIKELY(
          !iree_host_size_is_power_of_two(kernel_args->kernarg_alignment))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment must be a power of two (got %u)",
        kernel_args->kernarg_alignment);
  }
  if (IREE_UNLIKELY(kernel_args->kernarg_alignment >
                    iree_alignof(iree_hal_amdgpu_kernarg_block_t))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "executable kernel kernarg alignment %u exceeds queue kernarg ring "
        "alignment %" PRIhsz,
        kernel_args->kernarg_alignment,
        (iree_host_size_t)iree_alignof(iree_hal_amdgpu_kernarg_block_t));
  }

  iree_host_size_t operation_resource_count = 1;
  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    if (IREE_UNLIKELY(constants.data_length != kernel_args->kernarg_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument length mismatch; expected %u but got "
          "%" PRIhsz,
          kernel_args->kernarg_size, constants.data_length);
    }
    if (IREE_UNLIKELY(constants.data_length > 0 && !constants.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument data must be non-null when length is "
          "non-zero");
    }
    *out_layout = iree_hal_amdgpu_device_dispatch_make_custom_kernarg_layout(
        constants.data_length);
  } else {
    if (IREE_UNLIKELY((constants.data_length % sizeof(uint32_t)) != 0)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch constants must be 4-byte aligned");
    }
    if (IREE_UNLIKELY(constants.data_length > 0 && !constants.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant data must be non-null when length is non-zero");
    }
    const iree_host_size_t expected_constant_length =
        (iree_host_size_t)kernel_args->constant_count * sizeof(uint32_t);
    if (IREE_UNLIKELY(constants.data_length != expected_constant_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)kernel_args->constant_count,
          constants.data_length / sizeof(uint32_t));
    }
    if (IREE_UNLIKELY(bindings.count != kernel_args->binding_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)kernel_args->binding_count, bindings.count);
    }
    if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch bindings must be non-null when count is non-zero");
    }
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch_binding(
                               &bindings.values[i]),
                           "binding[%" PRIhsz "]", i);
    }
    if (IREE_UNLIKELY(!iree_host_size_checked_add(1, bindings.count,
                                                  &operation_resource_count))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch retains too many resources (bindings=%" PRIhsz ")",
          bindings.count);
    }

    *out_layout =
        iree_hal_amdgpu_device_dispatch_make_hal_kernarg_layout(kernel_args);
    if (IREE_UNLIKELY(kernel_args->kernarg_size <
                      out_layout->explicit_kernarg_size)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "executable kernel kernarg size %u is smaller than HAL ABI "
          "explicit argument size %" PRIhsz,
          kernel_args->kernarg_size, out_layout->explicit_kernarg_size);
    }
  }

  iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      out_layout->total_kernarg_size, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (kernarg_block_count == 0) {
    kernarg_block_count = 1;
  }
  if (IREE_UNLIKELY(kernarg_block_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch kernargs require too many blocks (%" PRIhsz ", max=%u)",
        kernarg_block_count, UINT32_MAX);
  }
  if (IREE_UNLIKELY(kernarg_block_count > queue->kernarg_ring.capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch kernargs require %" PRIhsz
                            " blocks but the queue kernarg ring capacity is %u",
                            kernarg_block_count, queue->kernarg_ring.capacity);
  }

  *out_kernarg_block_count = (uint32_t)kernarg_block_count;
  *out_operation_resource_count = operation_resource_count;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_host_size_t* out_operation_resource_count) {
  *out_operation_resource_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_dispatch_flags(flags));

  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_dispatch_kernel_args(
      queue, executable, export_ordinal, config, &kernel_args));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch_shape(
      &kernel_args, config.workgroup_count,
      config.dynamic_workgroup_local_memory));

  iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout;
  uint32_t kernarg_block_count = 0;
  return iree_hal_amdgpu_host_queue_validate_dispatch_kernargs(
      queue, &kernel_args, constants, bindings, flags, &layout,
      &kernarg_block_count, out_operation_resource_count);
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_binding_ptr(
    const iree_hal_buffer_ref_t* binding, uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_dispatch_binding(binding));
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(binding->buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch binding buffer must be backed by an AMDGPU allocation");
  }

  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(binding->buffer), binding->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding device pointer offset overflows device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding device pointer offset exceeds host pointer size");
  }
  *out_binding_ptr = (uint64_t)((uintptr_t)device_ptr + device_offset);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_dispatch_flags(flags));

  iree_hal_amdgpu_device_kernel_args_t kernel_args;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_dispatch_kernel_args(
      queue, executable, export_ordinal, config, &kernel_args));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch_shape(
      &kernel_args, config.workgroup_count,
      config.dynamic_workgroup_local_memory));

  iree_hal_amdgpu_device_dispatch_kernarg_layout_t layout;
  uint32_t kernarg_block_count = 0;
  iree_host_size_t operation_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch_kernargs(
      queue, &kernel_args, constants, bindings, flags, &layout,
      &kernarg_block_count, &operation_resource_count));

  iree_hal_resource_t* stack_operation_resources
      [IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_STACK_RESOURCE_CAPACITY];
  iree_hal_resource_t** operation_resources = stack_operation_resources;
  uint64_t stack_binding_ptrs
      [IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_STACK_BINDING_CAPACITY];
  uint64_t* binding_ptrs = stack_binding_ptrs;

  iree_status_t status = iree_ok_status();
  if (operation_resource_count > IREE_ARRAYSIZE(stack_operation_resources)) {
    iree_host_size_t byte_length = 0;
    if (iree_host_size_checked_mul(operation_resource_count,
                                   sizeof(*operation_resources),
                                   &byte_length)) {
      status = iree_allocator_malloc(queue->host_allocator, byte_length,
                                     (void**)&operation_resources);
    } else {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch operation resource list overflows host size");
    }
  }

  const bool uses_custom_direct_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  if (iree_status_is_ok(status) && !uses_custom_direct_arguments &&
      bindings.count > IREE_ARRAYSIZE(stack_binding_ptrs)) {
    iree_host_size_t byte_length = 0;
    if (iree_host_size_checked_mul(bindings.count, sizeof(*binding_ptrs),
                                   &byte_length)) {
      status = iree_allocator_malloc(queue->host_allocator, byte_length,
                                     (void**)&binding_ptrs);
    } else {
      status =
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "dispatch binding pointer list overflows host size");
    }
  }

  if (iree_status_is_ok(status)) {
    operation_resources[0] = (iree_hal_resource_t*)executable;
    if (!uses_custom_direct_arguments) {
      for (iree_host_size_t i = 0;
           i < bindings.count && iree_status_is_ok(status); ++i) {
        status = iree_hal_amdgpu_host_queue_resolve_binding_ptr(
            &bindings.values[i], &binding_ptrs[i]);
        if (iree_status_is_ok(status)) {
          operation_resources[i + 1] =
              (iree_hal_resource_t*)bindings.values[i].buffer;
        }
      }
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
    status = iree_hal_amdgpu_host_queue_begin_dispatch_submission(
        queue, resolution, signal_semaphore_list, operation_resource_count,
        kernarg_block_count, &submission);
    if (iree_status_is_ok(status)) {
      if (uses_custom_direct_arguments) {
        iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
            &layout, constants.data, submission.kernarg_blocks->data);
      } else {
        iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
            &kernel_args, config.workgroup_count,
            config.dynamic_workgroup_local_memory, &layout, binding_ptrs,
            (const uint32_t*)constants.data, submission.kernarg_blocks->data);
      }
      iree_hal_amdgpu_device_dispatch_emplace_packet(
          &kernel_args, config.workgroup_count,
          config.dynamic_workgroup_local_memory,
          &submission.dispatch_slot->dispatch, submission.kernarg_blocks->data);
      submission.dispatch_slot->dispatch.completion_signal =
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring);
      submission.dispatch_setup = submission.dispatch_slot->dispatch.setup;
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, submission_flags, &submission);
    }
  }

  if (binding_ptrs != stack_binding_ptrs) {
    iree_allocator_free(queue->host_allocator, binding_ptrs);
  }
  if (operation_resources != stack_operation_resources) {
    iree_allocator_free(queue->host_allocator, operation_resources);
  }
  return status;
}
