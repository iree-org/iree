// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HSA_STATUS_UTIL_H_
#define IREE_EXPERIMENTAL_HSA_STATUS_UTIL_H_

#include <stdint.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a hsa_status_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = IREE_HSA_RESULT_TO_STATUS(hsa_symbols,
//                                                    hsaDoThing(...));
#define IREE_HSA_RESULT_TO_STATUS(syms, expr, ...) \
  iree_hal_hsa_result_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the hsa_status_t return value to
// an iree_status_t.
//
// Usage:
//   IREE_HSA_RETURN_IF_ERROR(hsa_symbols, hsaDoThing(...), "message");
#define IREE_HSA_RETURN_IF_ERROR(syms, expr, ...)                            \
  IREE_RETURN_IF_ERROR(iree_hal_hsa_result_to_status((syms), ((syms)->expr), \
                                                     __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_RETURN_IF_ERROR but ends the current zone and implicitly converts the
// hsa_status_t return value to an iree_status_t.
//
// Usage:
//   IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(zone_id, hsa_symbols,
//                                         hsaDoThing(...), "message");

#define IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(zone_id, syms, expr, ...) \
  IREE_RETURN_AND_END_ZONE_IF_ERROR(                                    \
      zone_id,                                                          \
      iree_hal_hsa_result_to_status((syms), ((syms)->expr), __FILE__,   \
                                    __LINE__),                          \
      __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the hsa_status_t return value to an
// iree_status_t.
//
// Usage:
//   IREE_HSA_IGNORE_ERROR(hsa_symbols, hsaDoThing(...));
#define IREE_HSA_IGNORE_ERROR(syms, expr)                                 \
  IREE_IGNORE_ERROR(iree_hal_hsa_result_to_status((syms), ((syms)->expr), \
                                                  __FILE__, __LINE__))

// Converts a hsa_status_t to an iree_status_t object.
iree_status_t iree_hal_hsa_result_to_status(
    const iree_hal_hsa_dynamic_symbols_t* syms, hsa_status_t result,
    const char* file, uint32_t line);

const char* hsa_status_to_string(hsa_status_t status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HSA_STATUS_UTIL_H_
