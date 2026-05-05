// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_INFO_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_INFO_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_system_info_t
//===----------------------------------------------------------------------===//

// Cached information about the system.
typedef struct iree_hal_amdgpu_system_info_t {
  // Timestamp value increase rate in hz.
  // Query of HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY.
  uint64_t timestamp_frequency;
  // HSA SVM/HMM process-wide capability facts.
  struct {
    // Whether the HSA SVM attribute and prefetch APIs are supported.
    // Query of HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED.
    uint32_t supported : 1;
    // Whether all agents have access to system allocated memory by default.
    // Query of HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT.
    uint32_t accessible_by_default : 1;
    // Whether the process is bound to XNACK-enabled execution.
    // Query of HSA_AMD_SYSTEM_INFO_XNACK_ENABLED.
    uint32_t xnack_enabled : 1;
  } svm;
  // Whether the dmabuf APIs are supported by the driver.
  // Query of HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED.
  uint32_t dmabuf_supported : 1;
} iree_hal_amdgpu_system_info_t;

// Queries system information and verifies that the minimum required
// capabilities and versions are available. If this fails it's unlikely that the
// HAL will work and it should be called early on startup.
iree_status_t iree_hal_amdgpu_system_info_query(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_system_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_INFO_H_
