// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_STATUS_UTIL_H_
#define IREE_EXPERIMENTAL_HIP_STATUS_UTIL_H_

#include <stdint.h>

#include "experimental/hip/dynamic_symbols.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a hipError_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = IREE_HIP_RESULT_TO_STATUS(hip_symbols,
//                                                    hipDoThing(...));
#define IREE_HIP_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_hip_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the hipError_t return value to
// an iree_status_t.
//
// Usage:
//   IREE_HIP_RETURN_IF_ERROR(hip_symbols, hipDoThing(...), "message");
#define IREE_HIP_RETURN_IF_ERROR(syms, expr, ...)                            \
  IREE_RETURN_IF_ERROR(iree_hal_hip_result_to_status((syms), ((syms)->expr), \
                                                     __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_RETURN_IF_ERROR but ends the current zone and implicitly converts the
// hipError_t return value to an iree_status_t.
//
// Usage:
//   IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(zone_id, hip_symbols,
//                                         hipDoThing(...), "message");
#define IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(zone_id, syms, expr, ...) \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                    \
      zone_id,                                                          \
      iree_hal_hip_result_to_status((syms), ((syms)->expr), __FILE__,   \
                                    __LINE__),                          \
      __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the hipError_t return value to an
// iree_status_t.
//
// Usage:
//   IREE_HIP_IGNORE_ERROR(hip_symbols, hipDoThing(...));
#define IREE_HIP_IGNORE_ERROR(syms, expr)                                 \
  IREE_IGNORE_ERROR(iree_hal_hip_result_to_status((syms), ((syms)->expr), \
                                                  __FILE__, __LINE__))

// Converts a hipError_t to an iree_status_t object.
iree_status_t iree_hal_hip_result_to_status(
    const iree_hal_hip_dynamic_symbols_t* syms, hipError_t result,
    const char* file, uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_STATUS_UTIL_H_
