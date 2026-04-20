// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_EXECUTE_H_
#define IREE_HAL_REPLAY_EXECUTE_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_replay_execute_flags_t;
enum iree_hal_replay_execute_flag_bits_t {
  IREE_HAL_REPLAY_EXECUTE_FLAG_NONE = 0u,
};

typedef struct iree_hal_replay_execute_options_t {
  // Execution flags controlling replay behavior.
  iree_hal_replay_execute_flags_t flags;
  // Reserved for future replay execution options; must be zero.
  uint32_t reserved0;
} iree_hal_replay_execute_options_t;

static inline iree_hal_replay_execute_options_t
iree_hal_replay_execute_options_default(void) {
  iree_hal_replay_execute_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

// Executes the HAL operations in |file_contents| against |device_group|.
//
// Replay execution intentionally serializes the captured stream in file order.
// This preserves the program-order behavior needed for deterministic
// reproducers while leaving parallel replay strategies as an execution policy
// that can be added later without changing the file format.
IREE_API_EXPORT iree_status_t iree_hal_replay_execute_file(
    iree_const_byte_span_t file_contents, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_EXECUTE_H_
