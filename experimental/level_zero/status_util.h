// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_STATUS_UTIL_H_
#define IREE_HAL_LEVEL_ZERO_STATUS_UTIL_H_

#include <stdint.h>

#include "experimental/level_zero/dynamic_symbols.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a ze_result_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = LEVEL_ZERO_RESULT_TO_STATUS(levelZeroDoThing(...));
#define LEVEL_ZERO_RESULT_TO_STATUS(syms, expr, ...)                     \
  iree_hal_level_zero_result_to_status((syms), ((syms)->expr), __FILE__, \
                                       __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the ze_result_t return value to
// a Status.
//
// Usage:
//   LEVEL_ZERO_RETURN_IF_ERROR(levelZeroDoThing(...), "message");
#define LEVEL_ZERO_RETURN_IF_ERROR(syms, expr, ...)                     \
  IREE_RETURN_IF_ERROR(iree_hal_level_zero_result_to_status(            \
                           (syms), ((syms)->expr), __FILE__, __LINE__), \
                       __VA_ARGS__)

// IREE_IGNORE_ERROR but implicitly converts the ze_result_t return value to a
// Status.
//
// Usage:
//   LEVEL_ZERO_IGNORE_ERROR(levelZeroDoThing(...));
#define LEVEL_ZERO_IGNORE_ERROR(syms, expr)               \
  IREE_IGNORE_ERROR(iree_hal_level_zero_result_to_status( \
      (syms), ((syms)->expr), __FILE__, __LINE__))

// Converts a ze_result_t to a Status object.
iree_status_t iree_hal_level_zero_result_to_status(
    iree_hal_level_zero_dynamic_symbols_t* syms, ze_result_t result,
    const char* file, uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_STATUS_UTIL_H_
