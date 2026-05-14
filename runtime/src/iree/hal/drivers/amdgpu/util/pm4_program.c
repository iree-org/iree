// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/pm4_program.h"

#include <stddef.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

iree_status_t iree_hal_amdgpu_pm4_program_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t memory_pool, const uint32_t* source_dwords,
    uint32_t dword_count, iree_hal_amdgpu_pm4_program_t* out_program) {
  return iree_hal_amdgpu_pm4_program_initialize_with_stats(
      libhsa, device_agent, memory_pool, source_dwords, dword_count,
      /*out_stats=*/NULL, out_program);
}

iree_status_t iree_hal_amdgpu_pm4_program_initialize_with_stats(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t device_agent,
    hsa_amd_memory_pool_t memory_pool, const uint32_t* source_dwords,
    uint32_t dword_count,
    iree_hal_amdgpu_pm4_program_publish_stats_t* out_stats,
    iree_hal_amdgpu_pm4_program_t* out_program) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(source_dwords);
  IREE_ASSERT_ARGUMENT(out_program);
  memset(out_program, 0, sizeof(*out_program));
  if (out_stats) memset(out_stats, 0, sizeof(*out_stats));
  if (IREE_UNLIKELY(!memory_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 program memory pool is required");
  }
  if (IREE_UNLIKELY(dword_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 program must contain at least one dword");
  }
  if (IREE_UNLIKELY(dword_count > IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "PM4 program dword count %u exceeds PM4-IB maximum %u", dword_count,
        IREE_HAL_AMDGPU_PM4_IB_MAX_DWORD_COUNT);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, dword_count);

  iree_host_size_t byte_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(0, &byte_length,
                             IREE_STRUCT_FIELD(dword_count, uint32_t, NULL)));
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, byte_length);
  if (out_stats) out_stats->byte_length = byte_length;
  const bool collect_stats = out_stats != NULL;

  IREE_AMDGPU_DEVICE_PTR uint32_t* dwords = NULL;
  iree_time_t time_start = collect_stats ? iree_time_now() : 0;
  iree_status_t status = iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), memory_pool, byte_length,
      HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, (void**)&dwords);
  if (collect_stats) out_stats->allocate_ns += iree_time_now() - time_start;

  if (iree_status_is_ok(status)) {
    time_start = collect_stats ? iree_time_now() : 0;
    status =
        iree_hsa_amd_agents_allow_access(IREE_LIBHSA(libhsa), /*num_agents=*/1,
                                         &device_agent, /*flags=*/NULL, dwords);
    if (collect_stats) {
      out_stats->allow_access_ns += iree_time_now() - time_start;
    }
  }

  if (iree_status_is_ok(status)) {
    time_start = collect_stats ? iree_time_now() : 0;
    status = iree_hsa_memory_copy(IREE_LIBHSA(libhsa), dwords, source_dwords,
                                  byte_length);
    if (collect_stats) out_stats->copy_ns += iree_time_now() - time_start;
  }

  if (iree_status_is_ok(status)) {
    out_program->libhsa = libhsa;
    out_program->memory_pool = memory_pool;
    out_program->dwords = dwords;
    out_program->dword_count = dword_count;
    out_program->byte_length = byte_length;
  } else if (dwords) {
    status = iree_status_join(
        status, iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), dwords));
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_pm4_program_release(
    iree_hal_amdgpu_pm4_program_t* program) {
  if (!program || !program->dwords) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, program->byte_length);
  iree_status_t status = iree_hsa_amd_memory_pool_free(
      IREE_LIBHSA(program->libhsa), program->dwords);
  if (iree_status_is_ok(status)) {
    memset(program, 0, sizeof(*program));
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}
