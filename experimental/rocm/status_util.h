// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_ROCM_STATUS_UTIL_H_
#define IREE_HAL_ROCM_STATUS_UTIL_H_

#include <stdint.h>

#include "experimental/rocm/dynamic_symbols.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a hipError_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = ROCM_RESULT_TO_STATUS(rocmDoThing(...));
#define ROCM_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_rocm_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the hipError_t return value to
// a Status.
//
// Usage:
//   ROCM_RETURN_IF_ERROR(rocmDoThing(...), "message");
#define ROCM_RETURN_IF_ERROR(syms, expr, ...)                                 \
  IREE_RETURN_IF_ERROR(iree_hal_rocm_result_to_status((syms), ((syms)->expr), \
                                                      __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_RETURN_IF_ERROR but ends the current zone and implicitly converts the
// hipError_t return value to an iree_status_t.
//
// Usage:
//   ROCM_RETURN_AND_END_ZONE_IF_ERROR(zone_id, cuda_symbols,
//                                          hipDoThing(...), "message");
#define ROCM_RETURN_AND_END_ZONE_IF_ERROR(zone_id, syms, expr, ...)           \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                          \
      zone_id,                                                                \
      iree_hal_rocm_result_to_status((syms), ((syms)->expr), __FILE__,        \
                                      __LINE__),                              \
      __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the hipError_t return value to a
// Status.
//
// Usage:
//   ROCM_IGNORE_ERROR(rocmDoThing(...));
#define ROCM_IGNORE_ERROR(syms, expr)                                      \
  IREE_IGNORE_ERROR(iree_hal_rocm_result_to_status((syms), ((syms)->expr), \
                                                   __FILE__, __LINE__))

// Converts a hipError_t to a Status object.
iree_status_t iree_hal_rocm_result_to_status(
    iree_hal_rocm_dynamic_symbols_t* syms, hipError_t result, const char* file,
    uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_STATUS_UTIL_H_
