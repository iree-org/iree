// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/profiling.h"

#include "iree/hal/drivers/amdgpu/device/support/kernel.h"

//===----------------------------------------------------------------------===//
// Dispatch timestamp harvest
//===----------------------------------------------------------------------===//

iree_hal_amdgpu_profile_dispatch_harvest_source_t*
iree_hal_amdgpu_device_profile_emplace_dispatch_harvest(
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        harvest_kernel_args,
    uint32_t source_count,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT dispatch_packet,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  iree_hal_amdgpu_profile_dispatch_harvest_args_t* IREE_AMDGPU_RESTRICT
      kernargs = (iree_hal_amdgpu_profile_dispatch_harvest_args_t*)kernarg_ptr;
  iree_hal_amdgpu_profile_dispatch_harvest_source_t* sources =
      iree_hal_amdgpu_device_profile_dispatch_harvest_sources(kernarg_ptr);
  kernargs->sources = sources;
  kernargs->source_count = source_count;
  kernargs->reserved0 = 0;

  const uint32_t harvest_workgroup_size =
      harvest_kernel_args->workgroup_size[0];
  const uint32_t harvest_workgroup_count[3] = {
      (uint32_t)IREE_AMDGPU_CEIL_DIV(source_count, harvest_workgroup_size), 1,
      1};
  iree_hal_amdgpu_device_dispatch_emplace_packet(
      harvest_kernel_args, harvest_workgroup_count,
      /*dynamic_workgroup_local_memory=*/0, dispatch_packet, kernarg_ptr);
  return sources;
}

#if defined(IREE_AMDGPU_TARGET_DEVICE)

IREE_AMDGPU_ATTRIBUTE_KERNEL void
iree_hal_amdgpu_device_profile_harvest_dispatch_events(
    const iree_hal_amdgpu_profile_dispatch_harvest_source_t*
        IREE_AMDGPU_RESTRICT sources,
    uint32_t source_count) {
  const size_t source_index = iree_hal_amdgpu_device_global_linear_id_1d();
  if (source_index >= source_count) return;

  const iree_hal_amdgpu_profile_dispatch_harvest_source_t source =
      sources[source_index];
  source.event->start_tick = source.completion_signal->start_ts;
  source.event->end_tick = source.completion_signal->end_ts;
}

#endif  // IREE_AMDGPU_TARGET_DEVICE
