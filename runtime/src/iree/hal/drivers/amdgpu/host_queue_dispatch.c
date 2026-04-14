// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"

typedef struct iree_hal_amdgpu_host_queue_dispatch_plan_t {
  // Executable dispatch descriptor selected for the queue's physical device.
  const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor;
  // Kernel arguments used for packet and kernarg emission.
  const iree_hal_amdgpu_device_kernel_args_t* kernel_args;
  // Storage for a workgroup-size override when dispatch config provides one.
  iree_hal_amdgpu_device_kernel_args_t override_kernel_args;
  // Device ABI layout describing the kernarg bytes to emit.
  const iree_hal_amdgpu_device_dispatch_kernarg_layout_t* layout;
  // Number of queue-owned kernarg blocks required for the dispatch.
  uint32_t kernarg_block_count;
  // Number of operation resources retained until dispatch completion.
  iree_host_size_t operation_resource_count;
} iree_hal_amdgpu_host_queue_dispatch_plan_t;

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

static iree_status_t iree_hal_amdgpu_host_queue_lookup_dispatch_descriptor(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_amdgpu_executable_dispatch_descriptor_t** out_descriptor) {
  return iree_hal_amdgpu_executable_lookup_dispatch_descriptor_for_device(
      executable, export_ordinal, queue->device_ordinal, out_descriptor);
}

static bool iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(
    const iree_hal_dispatch_config_t config) {
  return config.workgroup_size[0] || config.workgroup_size[1] ||
         config.workgroup_size[2];
}

static iree_status_t iree_hal_amdgpu_host_queue_select_dispatch_kernel_args(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_dispatch_config_t config,
    iree_hal_amdgpu_device_kernel_args_t* override_kernel_args,
    const iree_hal_amdgpu_device_kernel_args_t** out_kernel_args) {
  *out_kernel_args = &descriptor->kernel_args;
  if (!iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    return iree_ok_status();
  }

  *override_kernel_args = descriptor->kernel_args;
  for (iree_host_size_t i = 0; i < 3; ++i) {
    if (IREE_UNLIKELY(!config.workgroup_size[i])) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch workgroup size override must specify all dimensions");
    }
    if (IREE_UNLIKELY(config.workgroup_size[i] > UINT16_MAX)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "dispatch workgroup size override dimension %" PRIhsz
          " value %u exceeds %u",
          i, config.workgroup_size[i], UINT16_MAX);
    }
    override_kernel_args->workgroup_size[i] =
        (uint16_t)config.workgroup_size[i];
  }

  *out_kernel_args = override_kernel_args;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_shape(
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    const iree_hal_amdgpu_device_kernel_args_t* kernel_args,
    const iree_hal_dispatch_config_t config) {
  if (iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      const uint64_t grid_size =
          (uint64_t)config.workgroup_count[i] * kernel_args->workgroup_size[i];
      if (IREE_UNLIKELY(grid_size > UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz
            " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
            i, config.workgroup_count[i], kernel_args->workgroup_size[i]);
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      if (IREE_UNLIKELY(config.workgroup_count[i] >
                        descriptor->max_workgroup_count[i])) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz
            " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
            i, config.workgroup_count[i], kernel_args->workgroup_size[i]);
      }
    }
  }
  if (IREE_UNLIKELY(config.dynamic_workgroup_local_memory >
                    descriptor->max_dynamic_workgroup_local_memory)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch group segment size overflows uint32_t "
                            "(static=%u, dynamic=%u)",
                            kernel_args->group_segment_size,
                            config.dynamic_workgroup_local_memory);
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
    const iree_hal_amdgpu_executable_dispatch_descriptor_t* descriptor,
    iree_const_byte_span_t constants, const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags,
    const iree_hal_amdgpu_device_dispatch_kernarg_layout_t** out_layout,
    uint32_t* out_kernarg_block_count,
    iree_host_size_t* out_operation_resource_count) {
  *out_layout = NULL;
  *out_kernarg_block_count = 0;
  *out_operation_resource_count = 0;

  iree_host_size_t operation_resource_count = 1;
  if (iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    if (IREE_UNLIKELY(constants.data_length !=
                      descriptor->kernel_args.kernarg_size)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument length mismatch; expected %u but got "
          "%" PRIhsz,
          descriptor->kernel_args.kernarg_size, constants.data_length);
    }
    if (IREE_UNLIKELY(constants.data_length > 0 && !constants.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "custom dispatch argument data must be non-null when length is "
          "non-zero");
    }
    *out_layout = &descriptor->custom_kernarg_layout;
    *out_kernarg_block_count = descriptor->custom_kernarg_block_count;
  } else {
    if (IREE_UNLIKELY(constants.data_length > 0 && !constants.data)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant data must be non-null when length is non-zero");
    }
    const iree_host_size_t expected_constant_length =
        (iree_host_size_t)descriptor->kernel_args.constant_count *
        sizeof(uint32_t);
    if (IREE_UNLIKELY(constants.data_length != expected_constant_length)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch constant count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.constant_count,
          constants.data_length / sizeof(uint32_t));
    }
    if (IREE_UNLIKELY(bindings.count !=
                      descriptor->kernel_args.binding_count)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding count mismatch; expected %u but got %" PRIhsz,
          (uint32_t)descriptor->kernel_args.binding_count, bindings.count);
    }
    if (IREE_UNLIKELY(
            bindings.count >
            IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_BINDING_CAPACITY)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "queue_dispatch supports at most %u direct buffer bindings but got "
          "%" PRIhsz,
          IREE_HAL_AMDGPU_HOST_QUEUE_DISPATCH_SCRATCH_BINDING_CAPACITY,
          bindings.count);
    }
    if (IREE_UNLIKELY(bindings.count > 0 && !bindings.values)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch bindings must be non-null when count is non-zero");
    }
    operation_resource_count = 1 + bindings.count;
    *out_layout = &descriptor->hal_kernarg_layout;
    *out_kernarg_block_count = descriptor->hal_kernarg_block_count;
  }

  if (IREE_UNLIKELY(*out_kernarg_block_count > queue->kernarg_ring.capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch kernargs require %u"
                            " blocks but the queue kernarg ring capacity is %u",
                            *out_kernarg_block_count,
                            queue->kernarg_ring.capacity);
  }

  *out_operation_resource_count = operation_resource_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_prepare_dispatch_plan(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_host_queue_dispatch_plan_t* out_plan) {
  memset(out_plan, 0, sizeof(*out_plan));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_dispatch_flags(flags));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_lookup_dispatch_descriptor(
      queue, executable, export_ordinal, &out_plan->descriptor));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_select_dispatch_kernel_args(
      out_plan->descriptor, config, &out_plan->override_kernel_args,
      &out_plan->kernel_args));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_dispatch_shape(
      out_plan->descriptor, out_plan->kernel_args, config));

  return iree_hal_amdgpu_host_queue_validate_dispatch_kernargs(
      queue, out_plan->descriptor, constants, bindings, flags,
      &out_plan->layout, &out_plan->kernarg_block_count,
      &out_plan->operation_resource_count);
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_validated_binding_ptr(
    const iree_hal_buffer_ref_t* binding, uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
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

// Validates direct dispatch bindings and optionally fills the submit-time
// channels used to retain resources and write final kernargs. The arrays are
// caller-owned scratch; passing NULL runs only the corresponding validation.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_dispatch_bindings(
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_resource_t** operation_resources, uint64_t* binding_ptrs) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < bindings.count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_buffer_ref_t* binding = &bindings.values[i];
    status = iree_hal_amdgpu_host_queue_validate_dispatch_binding(binding);
    if (iree_status_is_ok(status) && binding_ptrs) {
      status = iree_hal_amdgpu_host_queue_resolve_validated_binding_ptr(
          binding, &binding_ptrs[i]);
    }
    if (iree_status_is_ok(status) && operation_resources) {
      operation_resources[i + 1] = (iree_hal_resource_t*)binding->buffer;
    }
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "binding[%" PRIhsz "]", i);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_host_size_t* out_operation_resource_count) {
  *out_operation_resource_count = 0;
  iree_hal_amdgpu_host_queue_dispatch_plan_t plan;
  iree_status_t status = iree_hal_amdgpu_host_queue_prepare_dispatch_plan(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &plan);
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(flags,
                        IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS)) {
    status = iree_hal_amdgpu_host_queue_prepare_dispatch_bindings(
        bindings, /*operation_resources=*/NULL, /*binding_ptrs=*/NULL);
  }
  if (iree_status_is_ok(status)) {
    *out_operation_resource_count = plan.operation_resource_count;
  }
  return status;
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
  iree_hal_amdgpu_host_queue_dispatch_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_dispatch_plan(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &plan));

  iree_hal_resource_t** operation_resources =
      queue->dispatch_operation_resource_scratch;
  uint64_t* binding_ptrs = queue->dispatch_binding_ptr_scratch;

  const bool uses_custom_direct_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  operation_resources[0] = (iree_hal_resource_t*)executable;

  iree_status_t status = iree_ok_status();
  if (!uses_custom_direct_arguments) {
    status = iree_hal_amdgpu_host_queue_prepare_dispatch_bindings(
        bindings, operation_resources, binding_ptrs);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
    status = iree_hal_amdgpu_host_queue_begin_dispatch_submission(
        queue, resolution, signal_semaphore_list, plan.operation_resource_count,
        plan.kernarg_block_count, &submission);
    if (iree_status_is_ok(status)) {
      if (uses_custom_direct_arguments) {
        iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
            plan.layout, constants.data,
            submission.kernel.kernarg_blocks->data);
      } else {
        iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
            plan.kernel_args, config.workgroup_count,
            config.dynamic_workgroup_local_memory, plan.layout, binding_ptrs,
            (const uint32_t*)constants.data,
            submission.kernel.kernarg_blocks->data);
      }
      iree_hal_amdgpu_device_dispatch_emplace_packet(
          plan.kernel_args, config.workgroup_count,
          config.dynamic_workgroup_local_memory,
          &submission.dispatch_slot->dispatch,
          submission.kernel.kernarg_blocks->data);
      submission.dispatch_slot->dispatch.completion_signal =
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring);
      submission.dispatch_setup = submission.dispatch_slot->dispatch.setup;
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          plan.operation_resource_count, submission_flags, &submission);
    }
  }

  return status;
}
