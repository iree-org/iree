// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_headers.h"
#include "iree/hal/drivers/hip/per_device_information.h"

// The max number of per-dispatch bindings allowed in the HIP HAL
// implementation.
#define IREE_HAL_HIP_MAX_DISPATCH_BINDING_COUNT 16

// The max number of per-dispatch constants supported by the HIP HAL
// implementation.
#define IREE_HAL_HIP_MAX_DISPATCH_CONSTANT_COUNT 64

typedef struct iree_hal_hip_kernel_debug_info_t {
  iree_string_view_t function_name;
  iree_string_view_t source_filename;
  uint32_t source_line;
} iree_hal_hip_kernel_debug_info_t;

typedef struct iree_hal_hip_kernel_params_t {
  hipFunction_t function;

  uint32_t constant_count;
  uint32_t binding_count;

  uint32_t block_dims[3];

  IREE_TRACE(iree_hal_hip_kernel_debug_info_t debug_info;)
} iree_hal_hip_kernel_params_t;

// Creates an IREE executable from a HSACO module. The module may contain
// several kernels that can be extracted along with the associated block size.
iree_status_t iree_hal_hip_native_executable_create(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_device_topology_t topology,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the kernel launch parameters for the given |entry_point| in the
// |executable|.
iree_status_t iree_hal_hip_native_executable_lookup_kernel_params(
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_kernel_params_t** out_params);

#endif  // IREE_HAL_DRIVERS_HIP_NATIVE_EXECUTABLE_H_
