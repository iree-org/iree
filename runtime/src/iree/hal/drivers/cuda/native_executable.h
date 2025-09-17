// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The max number of per-dispatch bindings allowed in the CUDA HAL
// implementation.
#define IREE_HAL_CUDA_MAX_DISPATCH_BINDING_COUNT 16

// The max number of per-dispatch constants supported by the CUDA HAL
// implementation.
#define IREE_HAL_CUDA_MAX_DISPATCH_CONSTANT_COUNT 64

typedef struct iree_hal_cuda_kernel_debug_info_t {
  iree_string_view_t function_name;
  iree_string_view_t source_filename;
  uint32_t source_line;
} iree_hal_cuda_kernel_debug_info_t;

typedef struct iree_hal_cuda_kernel_params_t {
  CUfunction function;

  uint32_t constant_count;
  uint32_t binding_count;

  uint32_t block_dims[3];
  uint32_t block_shared_memory_size;

  IREE_TRACE(iree_hal_cuda_kernel_debug_info_t debug_info;)
} iree_hal_cuda_kernel_params_t;

// Infers the format of the executable and calculates its total size.
// If executable_data.data_length is 0 attempts to infer size from the data.
// Returns the canonical format string and total size of the executable data.
iree_status_t iree_hal_cuda_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size);

// Creates an IREE executable from a CUDA PTX module. The module may contain
// several kernels that can be extracted along with the associated block size.
iree_status_t iree_hal_cuda_native_executable_create(
    const iree_hal_cuda_dynamic_symbols_t* symbols, CUdevice device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the kernel launch information for the given |entry_point| in the
// |executable|.
iree_status_t iree_hal_cuda_native_executable_lookup_kernel_params(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_cuda_kernel_params_t** out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_NATIVE_EXECUTABLE_H_
