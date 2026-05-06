// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_dispatch.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/device/timestamp.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

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
  // True when workgroup counts are read from a device buffer before dispatch.
  bool uses_indirect_parameters;
} iree_hal_amdgpu_host_queue_dispatch_plan_t;

static iree_status_t iree_hal_amdgpu_host_queue_validate_dispatch_flags(
    iree_hal_dispatch_flags_t flags) {
  if (iree_hal_dispatch_uses_indirect_arguments(flags)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "indirect dispatch arguments are not supported by AMDGPU "
        "queue_dispatch yet");
  }

  const iree_hal_dispatch_flags_t supported_flags =
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS |
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS |
      IREE_HAL_DISPATCH_FLAG_ALLOW_INLINE_EXECUTION |
      IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES;
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
    const iree_hal_dispatch_config_t config, iree_hal_dispatch_flags_t flags) {
  const bool uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);
  if (iree_hal_amdgpu_dispatch_config_has_workgroup_size_override(config)) {
    for (iree_host_size_t i = 0; i < 3; ++i) {
      const uint64_t grid_size =
          (uint64_t)config.workgroup_count[i] * kernel_args->workgroup_size[i];
      if (!uses_indirect_parameters && IREE_UNLIKELY(grid_size > UINT32_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "dispatch grid dimension %" PRIhsz
            " overflows uint32_t (workgroup_count=%u, workgroup_size=%u)",
            i, config.workgroup_count[i], kernel_args->workgroup_size[i]);
      }
    }
  } else if (!uses_indirect_parameters) {
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

static iree_status_t
iree_hal_amdgpu_host_queue_validate_dispatch_indirect_parameters(
    const iree_hal_buffer_ref_t* workgroup_count_ref) {
  const iree_device_size_t workgroup_count_length = sizeof(uint32_t[3]);
  if (IREE_UNLIKELY(workgroup_count_ref->reserved != 0 ||
                    workgroup_count_ref->buffer_slot != 0 ||
                    !workgroup_count_ref->buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_dispatch indirect workgroup parameters must use a direct "
        "non-null buffer reference");
  }
  if (IREE_UNLIKELY((workgroup_count_ref->offset % sizeof(uint32_t)) != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_dispatch indirect workgroup parameter offset must be 4-byte "
        "aligned");
  }
  if (IREE_UNLIKELY(workgroup_count_ref->length != IREE_HAL_WHOLE_BUFFER &&
                    workgroup_count_ref->length < workgroup_count_length)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue_dispatch indirect workgroup parameter buffer must contain at "
        "least uint32_t[3]");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroup_count_ref->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroup_count_ref->buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroup_count_ref->buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  return iree_hal_buffer_validate_range(workgroup_count_ref->buffer,
                                        workgroup_count_ref->offset,
                                        workgroup_count_length);
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
    *out_kernarg_block_count =
        iree_max(1u, descriptor->custom_kernarg_block_count);
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
    *out_kernarg_block_count =
        iree_max(1u, descriptor->hal_kernarg_block_count);
  }

  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    ++operation_resource_count;
  }
  if (iree_any_bit_set(flags,
                       IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES)) {
    operation_resource_count = 0;
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
      out_plan->descriptor, out_plan->kernel_args, config, flags));
  out_plan->uses_indirect_parameters =
      iree_hal_dispatch_uses_indirect_parameters(flags);

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

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_dispatch_indirect_parameters(
    const iree_hal_dispatch_config_t config,
    iree_hal_resource_t** operation_resources,
    iree_host_size_t operation_resource_index,
    uint64_t* out_workgroup_count_ptr) {
  *out_workgroup_count_ptr = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_validate_dispatch_indirect_parameters(
          &config.workgroup_count_ref));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_validated_binding_ptr(
      &config.workgroup_count_ref, out_workgroup_count_ptr));
  operation_resources[operation_resource_index] =
      (iree_hal_resource_t*)config.workgroup_count_ref.buffer;
  return iree_ok_status();
}

static void iree_hal_amdgpu_host_queue_initialize_dispatch_event(
    iree_hal_amdgpu_profile_dispatch_event_t* event,
    const iree_hal_amdgpu_host_queue_dispatch_plan_t* plan,
    iree_hal_executable_export_ordinal_t export_ordinal, uint64_t executable_id,
    const iree_hal_dispatch_config_t config,
    iree_hal_amdgpu_profile_dispatch_event_flags_t flags) {
  const uint64_t event_id = event->event_id;
  memset(event, 0, sizeof(*event));
  event->record_length = sizeof(*event);
  event->flags = flags;
  event->event_id = event_id;
  event->command_index = UINT32_MAX;
  event->export_ordinal = export_ordinal;
  event->executable_id = executable_id;
  if (!iree_any_bit_set(
          flags,
          IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS)) {
    memcpy(event->workgroup_count, config.workgroup_count,
           sizeof(event->workgroup_count));
  }
  event->workgroup_size[0] = plan->kernel_args->workgroup_size[0];
  event->workgroup_size[1] = plan->kernel_args->workgroup_size[1];
  event->workgroup_size[2] = plan->kernel_args->workgroup_size[2];
}

static bool iree_hal_amdgpu_host_queue_should_profile_dispatch(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t executable_id,
    iree_hal_executable_export_ordinal_t export_ordinal) {
  if (!queue->profiling.dispatch_profiling_enabled) return false;
  if (!queue->profiling.hsa_queue_timestamps_enabled) return false;
  iree_hal_amdgpu_logical_device_t* logical_device =
      (iree_hal_amdgpu_logical_device_t*)queue->logical_device;
  const uint32_t physical_device_ordinal = queue->device_ordinal <= UINT32_MAX
                                               ? (uint32_t)queue->device_ordinal
                                               : UINT32_MAX;
  const uint32_t queue_ordinal = iree_async_axis_queue_index(queue->axis);
  return iree_hal_amdgpu_logical_device_should_profile_dispatch(
      logical_device, executable_id, export_ordinal,
      /*command_buffer_id=*/0, /*command_index=*/UINT32_MAX,
      physical_device_ordinal, queue_ordinal);
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
  if (iree_status_is_ok(status) && plan.uses_indirect_parameters) {
    status = iree_hal_amdgpu_host_queue_validate_dispatch_indirect_parameters(
        &config.workgroup_count_ref);
  }
  if (iree_status_is_ok(status)) {
    *out_operation_resource_count = plan.operation_resource_count;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_direct_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_dispatch_plan_t* plan,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const uint64_t* binding_ptrs,
    iree_hal_resource_t* const* operation_resources,
    bool uses_custom_direct_arguments,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  uint64_t executable_id = 0;
  bool should_profile_dispatch = false;
  if (queue->profiling.dispatch_profiling_enabled) {
    executable_id = iree_hal_amdgpu_executable_profile_id(executable);
    should_profile_dispatch =
        iree_hal_amdgpu_host_queue_should_profile_dispatch(queue, executable_id,
                                                           export_ordinal);
  }
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  iree_status_t status = iree_ok_status();
  if (should_profile_dispatch) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, /*event_count=*/1, &profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
        queue, profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_traces(queue,
                                                               profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
        queue, profile_events.first_event_position, executable_id);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return status;
  }
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_queue_event_info = {
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
      .operation_count = 1,
  };
  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  status = iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list, plan->operation_resource_count,
      plan->kernarg_block_count, profile_events, &profile_queue_event_info,
      out_ready, &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return status;
  }

  if (uses_custom_direct_arguments) {
    iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
        plan->layout, constants.data, submission.kernel.kernargs.blocks->data);
  } else {
    iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
        plan->kernel_args, config.workgroup_count,
        config.dynamic_workgroup_local_memory, plan->layout, binding_ptrs,
        (const uint32_t*)constants.data,
        submission.kernel.kernargs.blocks->data);
  }
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      plan->kernel_args, config.workgroup_count,
      config.dynamic_workgroup_local_memory,
      &submission.dispatch_slot->dispatch,
      submission.kernel.kernargs.blocks->data);
  submission.dispatch_slot->dispatch.completion_signal =
      submission.dispatch_completion_signal;
  submission.dispatch_setup = submission.dispatch_slot->dispatch.setup;
  if (submission.profile_harvest_slot) {
    iree_hal_amdgpu_profile_dispatch_event_t* event =
        iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
            queue, profile_events.first_event_position);
    iree_hal_amdgpu_host_queue_initialize_dispatch_event(
        event, plan, export_ordinal, executable_id, config,
        IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_NONE);
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* sources =
        iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
            &queue->transfer_context->kernels
                 ->iree_hal_amdgpu_device_timestamp_harvest_dispatch_records,
            profile_events.event_count,
            &submission.profile_harvest_slot->dispatch,
            submission.profile_harvest_kernarg_blocks->data);
    sources[0].completion_signal =
        iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
            queue, profile_events.first_event_position);
    sources[0].ticks = iree_hal_amdgpu_profile_dispatch_event_ticks(event);
    submission.profile_harvest_setup =
        submission.profile_harvest_slot->dispatch.setup;
  }
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          plan->operation_resource_count, &profile_queue_event_info,
          submission_flags, &submission);
  profile_queue_event_info.submission_id = submission_epoch;
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list, &profile_queue_event_info);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_indirect_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hal_amdgpu_host_queue_dispatch_plan_t* plan,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const uint64_t* binding_ptrs, uint64_t workgroup_count_ptr,
    iree_hal_resource_t* const* operation_resources,
    bool uses_custom_direct_arguments,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  uint64_t executable_id = 0;
  bool should_profile_dispatch = false;
  if (queue->profiling.dispatch_profiling_enabled) {
    executable_id = iree_hal_amdgpu_executable_profile_id(executable);
    should_profile_dispatch =
        iree_hal_amdgpu_host_queue_should_profile_dispatch(queue, executable_id,
                                                           export_ordinal);
  }
  const uint32_t target_kernarg_block_count = plan->kernarg_block_count;
  const uint32_t patch_kernarg_block_count = 1;
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  iree_status_t status = iree_ok_status();
  if (should_profile_dispatch) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, /*event_count=*/1, &profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
        queue, profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_traces(queue,
                                                               profile_events);
  }
  if (iree_status_is_ok(status) && profile_events.event_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_trace_code_object(
        queue, profile_events.first_event_position, executable_id);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return status;
  }
  iree_hal_amdgpu_host_queue_profile_event_info_t profile_queue_event_info = {
      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH,
      .operation_count = 1,
  };
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue)) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
        queue, /*event_count=*/1, &profile_queue_device_events);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    return status;
  }
  const bool profile_dispatch_packet = profile_events.event_count != 0;
  const bool profile_queue_device_event =
      profile_queue_device_events.event_count != 0;
  uint32_t profile_counter_set_count = 0;
  uint32_t profile_counter_packet_count = 0;
  uint32_t profile_trace_packet_count = 0;
  uint32_t profile_trace_start_packet_count = 0;
  if (profile_dispatch_packet) {
    profile_counter_set_count =
        iree_hal_amdgpu_host_queue_profile_counter_set_count(queue,
                                                             profile_events);
    profile_counter_packet_count =
        iree_hal_amdgpu_host_queue_profile_counter_packet_count(queue,
                                                                profile_events);
    profile_trace_packet_count =
        iree_hal_amdgpu_host_queue_profile_trace_packet_count(queue,
                                                              profile_events);
    profile_trace_start_packet_count =
        iree_hal_amdgpu_host_queue_profile_trace_start_packet_count(
            queue, profile_events);
  }
  const uint32_t profile_trace_stop_packet_count =
      profile_trace_packet_count - profile_trace_start_packet_count;
  const uint32_t profile_queue_device_prefix_packet_count =
      profile_queue_device_event ? 1u : 0u;
  const uint32_t profile_queue_device_suffix_packet_count =
      profile_queue_device_event ? 1u : 0u;
  const uint32_t profile_queue_device_packet_count =
      profile_queue_device_prefix_packet_count +
      profile_queue_device_suffix_packet_count;
  const uint32_t payload_packet_count =
      profile_queue_device_packet_count + 2u + profile_counter_packet_count +
      profile_trace_packet_count + (profile_dispatch_packet ? 1u : 0u);
  const uint32_t profile_harvest_kernarg_block_count =
      profile_dispatch_packet
          ? (uint32_t)iree_host_size_ceil_div(
                iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(
                    profile_events.event_count),
                sizeof(iree_hal_amdgpu_kernarg_block_t))
          : 0u;
  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      queue, resolution, signal_semaphore_list, plan->operation_resource_count,
      payload_packet_count,
      patch_kernarg_block_count + target_kernarg_block_count +
          profile_harvest_kernarg_block_count,
      out_ready, &submission);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
    return status;
  }

  const uint64_t patch_packet_id = submission.first_packet_id +
                                   resolution->barrier_count +
                                   profile_queue_device_prefix_packet_count;
  const uint64_t dispatch_packet_id = patch_packet_id + 1u +
                                      profile_counter_set_count +
                                      profile_trace_start_packet_count;
  const uint64_t profile_harvest_packet_id =
      submission.first_packet_id + resolution->barrier_count +
      payload_packet_count - 1u - profile_queue_device_suffix_packet_count;
  iree_hal_amdgpu_aql_packet_t* patch_packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, patch_packet_id);
  iree_hal_amdgpu_aql_packet_t* dispatch_packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, dispatch_packet_id);
  iree_hal_amdgpu_aql_packet_t* profile_harvest_packet = NULL;
  if (profile_dispatch_packet) {
    profile_harvest_packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, profile_harvest_packet_id);
  }
  iree_hal_amdgpu_kernarg_block_t* kernarg_blocks = submission.kernargs.blocks;
  uint8_t* patch_kernarg_data = kernarg_blocks[0].data;
  uint8_t* dispatch_kernarg_data = kernarg_blocks[1].data;
  uint8_t* profile_harvest_kernarg_data = NULL;
  if (profile_dispatch_packet) {
    iree_hal_amdgpu_kernarg_block_t* profile_harvest_kernarg_blocks =
        &kernarg_blocks[patch_kernarg_block_count + target_kernarg_block_count];
    profile_harvest_kernarg_data = profile_harvest_kernarg_blocks->data;
  }
  const uint32_t placeholder_workgroup_count[3] = {0, 0, 0};
  if (uses_custom_direct_arguments) {
    iree_hal_amdgpu_device_dispatch_emplace_custom_kernargs(
        plan->layout, constants.data, dispatch_kernarg_data);
  } else {
    iree_hal_amdgpu_device_dispatch_emplace_hal_kernargs(
        plan->kernel_args, placeholder_workgroup_count,
        config.dynamic_workgroup_local_memory, plan->layout, binding_ptrs,
        (const uint32_t*)constants.data, dispatch_kernarg_data);
  }
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      plan->kernel_args, placeholder_workgroup_count,
      config.dynamic_workgroup_local_memory, &dispatch_packet->dispatch,
      dispatch_kernarg_data);
  iree_hsa_signal_t dispatch_completion_signal =
      profile_queue_device_event
          ? iree_hsa_signal_null()
          : iree_hal_amdgpu_notification_ring_epoch_signal(
                &queue->notification_ring);
  if (profile_dispatch_packet) {
    dispatch_completion_signal =
        iree_hal_amdgpu_host_queue_profiling_completion_signal(
            queue, profile_events.first_event_position);
  }
  dispatch_packet->dispatch.completion_signal = dispatch_completion_signal;

  iree_amdgpu_kernel_implicit_args_t* implicit_args =
      plan->layout->has_implicit_args
          ? (iree_amdgpu_kernel_implicit_args_t*)(dispatch_kernarg_data +
                                                  plan->layout
                                                      ->implicit_args_offset)
          : NULL;
  const uint16_t dispatch_setup = dispatch_packet->dispatch.setup;
  const iree_hsa_fence_scope_t dispatch_acquire_scope =
      iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
          queue, IREE_HSA_FENCE_SCOPE_AGENT);
  const iree_hal_amdgpu_aql_packet_control_t dispatch_packet_control =
      (profile_dispatch_packet || profile_queue_device_event)
          ? iree_hal_amdgpu_aql_packet_control_barrier(
                iree_hal_amdgpu_host_queue_max_fence_scope(
                    dispatch_acquire_scope, resolution->inline_acquire_scope),
                profile_dispatch_packet ? IREE_HSA_FENCE_SCOPE_AGENT
                                        : IREE_HSA_FENCE_SCOPE_NONE)
          : iree_hal_amdgpu_aql_packet_control_barrier(
                iree_hal_amdgpu_host_queue_max_fence_scope(
                    dispatch_acquire_scope, resolution->inline_acquire_scope),
                iree_hal_amdgpu_host_queue_signal_list_release_scope(
                    queue, signal_semaphore_list));
  const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH, dispatch_packet_control);
  iree_hal_amdgpu_device_dispatch_emplace_indirect_params_patch(
      &queue->transfer_context->kernels
           ->iree_hal_amdgpu_device_dispatch_patch_indirect_params,
      (const uint32_t*)(uintptr_t)workgroup_count_ptr,
      &dispatch_packet->dispatch, dispatch_header, dispatch_setup,
      implicit_args, &patch_packet->dispatch, patch_kernarg_data);
  if (profile_dispatch_packet) {
    iree_hal_amdgpu_profile_dispatch_event_t* event =
        iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
            queue, profile_events.first_event_position);
    iree_hal_amdgpu_host_queue_initialize_dispatch_event(
        event, plan, export_ordinal, executable_id, config,
        IREE_HAL_AMDGPU_PROFILE_DISPATCH_EVENT_FLAG_INDIRECT_PARAMETERS);
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* sources =
        iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
            &queue->transfer_context->kernels
                 ->iree_hal_amdgpu_device_timestamp_harvest_dispatch_records,
            profile_events.event_count, &profile_harvest_packet->dispatch,
            profile_harvest_kernarg_data);
    sources[0].completion_signal =
        iree_hal_amdgpu_host_queue_profiling_completion_signal_ptr(
            queue, profile_events.first_event_position);
    sources[0].ticks = iree_hal_amdgpu_profile_dispatch_event_ticks(event);
  }

  iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, resolution,
                                                           &submission);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_kernel_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          plan->operation_resource_count, /*inout_resource_set=*/NULL,
          submission_flags, &submission);
  profile_queue_event_info.submission_id = submission_epoch;
  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, profile_queue_device_events, &profile_queue_event_info);
  if (queue_device_event) {
    submission.reclaim_entry->queue_device_event_first_position =
        profile_queue_device_events.first_event_position;
    submission.reclaim_entry->queue_device_event_count =
        profile_queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }
  uint16_t profile_harvest_header = 0;
  if (profile_dispatch_packet) {
    submission.reclaim_entry->profile_event_first_position =
        profile_events.first_event_position;
    submission.reclaim_entry->profile_event_count = profile_events.event_count;
    iree_hal_amdgpu_profile_dispatch_event_t* event =
        iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
            queue, profile_events.first_event_position);
    event->submission_id = submission_epoch;
    profile_harvest_packet->dispatch.completion_signal =
        queue_device_event ? iree_hsa_signal_null()
                           : iree_hal_amdgpu_notification_ring_epoch_signal(
                                 &queue->notification_ring);
    const iree_hsa_fence_scope_t profile_harvest_acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            queue, IREE_HSA_FENCE_SCOPE_AGENT);
    profile_harvest_header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                profile_harvest_acquire_scope,
                resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_SYSTEM));
  }
  iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue, &submission);
  if (queue_device_event) {
    iree_hal_amdgpu_host_queue_commit_queue_device_start_packet(
        queue, resolution,
        submission.first_packet_id + resolution->barrier_count,
        queue_device_event);
  }
  if (profile_counter_set_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_counter_start_packets(
        queue, profile_events.first_event_position, profile_counter_set_count,
        patch_packet_id + 1u,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (profile_trace_packet_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_trace_start_packet(
        queue, profile_events.first_event_position,
        patch_packet_id + 1u + profile_counter_set_count,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_AGENT));
    iree_hal_amdgpu_host_queue_commit_profile_trace_code_object_packet(
        queue, profile_events.first_event_position,
        patch_packet_id + 1u + profile_counter_set_count + 1u,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
    iree_hal_amdgpu_host_queue_commit_profile_trace_stop_packet(
        queue, profile_events.first_event_position, dispatch_packet_id + 1u,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (profile_counter_set_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_counter_read_stop_packets(
        queue, profile_events.first_event_position, profile_counter_set_count,
        dispatch_packet_id + 1u + profile_trace_stop_packet_count,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (profile_dispatch_packet) {
    iree_hal_amdgpu_aql_ring_commit(profile_harvest_packet,
                                    profile_harvest_header,
                                    profile_harvest_packet->dispatch.setup);
  }
  if (queue_device_event) {
    iree_hal_amdgpu_host_queue_commit_queue_device_end_packet(
        queue, resolution, signal_semaphore_list,
        submission.first_packet_id + submission.packet_count - 1,
        queue_device_event);
  }
  const uint16_t patch_setup = patch_packet->dispatch.setup;
  const uint16_t patch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_packet_control_barrier(
          iree_hal_amdgpu_host_queue_max_fence_scope(
              IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
          IREE_HSA_FENCE_SCOPE_AGENT));
  iree_hal_amdgpu_aql_ring_commit(patch_packet, patch_header, patch_setup);
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission.first_packet_id + submission.packet_count - 1);
  iree_hal_amdgpu_host_queue_record_profile_queue_event(
      queue, resolution, signal_semaphore_list, &profile_queue_event_info);
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
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  iree_hal_amdgpu_host_queue_dispatch_plan_t plan;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_dispatch_plan(
      queue, executable, export_ordinal, config, constants, bindings, flags,
      &plan));

  iree_hal_resource_t** operation_resources =
      queue->dispatch_scratch.operation_resources;
  uint64_t* binding_ptrs = queue->dispatch_scratch.binding_ptrs;

  const bool uses_custom_direct_arguments =
      iree_any_bit_set(flags, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  operation_resources[0] = (iree_hal_resource_t*)executable;

  iree_status_t status = iree_ok_status();
  if (!uses_custom_direct_arguments) {
    status = iree_hal_amdgpu_host_queue_prepare_dispatch_bindings(
        bindings, operation_resources, binding_ptrs);
  }
  uint64_t workgroup_count_ptr = 0;
  if (iree_status_is_ok(status) && plan.uses_indirect_parameters) {
    const iree_host_size_t resource_index =
        uses_custom_direct_arguments ? 1 : 1 + bindings.count;
    status = iree_hal_amdgpu_host_queue_prepare_dispatch_indirect_parameters(
        config, operation_resources, resource_index, &workgroup_count_ptr);
  }

  if (iree_status_is_ok(status)) {
    if (plan.uses_indirect_parameters) {
      status = iree_hal_amdgpu_host_queue_submit_indirect_dispatch(
          queue, resolution, signal_semaphore_list, &plan, executable,
          export_ordinal, config, constants, binding_ptrs, workgroup_count_ptr,
          operation_resources, uses_custom_direct_arguments, submission_flags,
          out_ready);
    } else {
      status = iree_hal_amdgpu_host_queue_submit_direct_dispatch(
          queue, resolution, signal_semaphore_list, &plan, executable,
          export_ordinal, config, constants, binding_ptrs, operation_resources,
          uses_custom_direct_arguments, submission_flags, out_ready);
    }
  }

  return status;
}
