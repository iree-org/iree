// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/virtual_queue.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_queue_options_t
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_queue_infer_placement(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent, iree_hal_amdgpu_queue_placement_t* out_placement) {
  // TODO(benvanik): implement conditions:
  // * PCIe Atomics
  // * !PCIe Atomics && APU
  // * !PCIe Atomics && gfx90a && xGMI
  *out_placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST;
  return iree_ok_status();
}

void iree_hal_amdgpu_queue_options_initialize(
    iree_hal_amdgpu_queue_options_t* out_options) {
  IREE_ASSERT_ARGUMENT(out_options);
  memset(out_options, 0, sizeof(*out_options));
  out_options->placement = IREE_HAL_AMDGPU_QUEUE_PLACEMENT_HOST;
  out_options->flags = IREE_HAL_AMDGPU_QUEUE_FLAG_NONE;
  out_options->mode = IREE_HAL_AMDGPU_QUEUE_SCHEDULING_MODE_DEFAULT;
  out_options->control_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_CONTROL_QUEUE_CAPACITY;
  out_options->execution_queue_count =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_COUNT;
  out_options->execution_queue_capacity =
      IREE_HAL_AMDGPU_DEFAULT_EXECUTION_QUEUE_CAPACITY;
  out_options->kernarg_ringbuffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_KERNARG_RINGBUFFER_CAPACITY;
  out_options->trace_buffer_capacity =
      IREE_HAL_AMDGPU_DEFAULT_TRACE_BUFFER_CAPACITY;
}

// Verifies that the given |queue_capacity| is between the agent min/max
// requirements and a power-of-two.
static iree_status_t iree_hal_amdgpu_verify_hsa_queue_size(
    iree_string_view_t queue_name, iree_host_size_t queue_size,
    uint32_t queue_min_size, uint32_t queue_max_size) {
  // Queues must meet the min/max size requirements.
  if (queue_size < queue_min_size || queue_size > queue_max_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity on this agent must be between "
        "HSA_AGENT_INFO_QUEUE_MIN_SIZE=%u and HSA_AGENT_INFO_QUEUE_MAX_SIZE=%u "
        "(provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_min_size, queue_max_size,
        queue_size);
  }

  // All queues must be a power-of-two due to ringbuffer masking.
  if (!iree_host_size_is_power_of_two(queue_size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%.*s queue capacity must be a power of two (provided %" PRIhsz ")",
        (int)queue_name.size, queue_name.data, queue_size);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_options_verify(
    const iree_hal_amdgpu_queue_options_t* options,
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t cpu_agent,
    hsa_agent_t gpu_agent) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(libhsa);

  // If the queue is placed on the device it must support PCIe atomics or be
  // connected via xGMI.
  if (options->placement == IREE_HAL_AMDGPU_QUEUE_PLACEMENT_DEVICE) {
    iree_hal_amdgpu_queue_placement_t possible_placement =
        IREE_HAL_AMDGPU_QUEUE_PLACEMENT_ANY;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_infer_placement(
        libhsa, cpu_agent, gpu_agent, &possible_placement));
    if (possible_placement != options->placement) {
      return iree_make_status(
          IREE_STATUS_INCOMPATIBLE,
          "device-side queue placement requested but the device does not meet "
          "the minimum requirements (PCIe atomics, xGMI, or APU)");
    }
  }

  // Query agent min/max queue size.
  uint32_t queue_min_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                                               HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                                               &queue_min_size));
  uint32_t queue_max_size = 0;
  IREE_RETURN_IF_ERROR(iree_hsa_agent_get_info(IREE_LIBHSA(libhsa), gpu_agent,
                                               HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                                               &queue_max_size));

  // Verify HSA queues.
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("control"), options->control_queue_capacity, queue_min_size,
      queue_max_size));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_verify_hsa_queue_size(
      IREE_SV("execution"), options->execution_queue_capacity, queue_min_size,
      queue_max_size));

  // Verify kernarg ringbuffer capacity (our ringbuffer so no HSA min/max
  // required).
  if (!iree_device_size_is_power_of_two(options->kernarg_ringbuffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ringbuffer capacity must be a power of two (provided %" PRIdsz
        ")",
        options->kernarg_ringbuffer_capacity);
  }

  // Verify trace buffer capacity (our ringbuffer so no HSA min/max required).
  if (options->trace_buffer_capacity &&
      !iree_device_size_is_power_of_two(options->trace_buffer_capacity)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "trace buffer capacity must be a power of two (provided %" PRIdsz ")",
        options->trace_buffer_capacity);
  }

  return iree_ok_status();
}
