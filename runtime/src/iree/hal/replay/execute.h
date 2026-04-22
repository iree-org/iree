// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_EXECUTE_H_
#define IREE_HAL_REPLAY_EXECUTE_H_

#include <stdbool.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef uint32_t iree_hal_replay_execute_flags_t;
enum iree_hal_replay_execute_flag_bits_t {
  IREE_HAL_REPLAY_EXECUTE_FLAG_NONE = 0u,
};

typedef struct iree_hal_replay_file_path_remap_t {
  // Captured external file path prefix to replace.
  iree_string_view_t captured_prefix;
  // Replay-time path prefix substituted for |captured_prefix|.
  iree_string_view_t replay_prefix;
} iree_hal_replay_file_path_remap_t;

typedef struct iree_hal_replay_executable_substitution_request_t {
  // Sequence ordinal of the executable prepare operation being replayed.
  uint64_t sequence_ordinal;
  // Captured device object id that owns the executable cache.
  iree_hal_replay_object_id_t device_id;
  // Captured executable cache object id receiving the prepare request.
  iree_hal_replay_object_id_t executable_cache_id;
  // Captured executable object id produced by the prepare request.
  iree_hal_replay_object_id_t executable_id;
  // Captured executable parameters borrowed from the replay file.
  const iree_hal_executable_params_t* captured_params;
} iree_hal_replay_executable_substitution_request_t;

typedef struct iree_hal_replay_executable_substitution_t {
  // True when |executable_data| should replace the captured executable bytes.
  bool substitute;
  // Optional diagnostic source for the replacement, such as a file path.
  iree_string_view_t source;
  // Replacement executable format, or empty to infer from |executable_data|.
  iree_string_view_t executable_format;
  // Replacement executable data borrowed for the prepare call.
  iree_const_byte_span_t executable_data;
} iree_hal_replay_executable_substitution_t;

// Callback allowing callers to substitute captured executable payloads.
//
// The callback receives the captured executable ids and parameters. It may
// leave |out_substitution->substitute| false to use the captured payload
// unchanged. Replacement data only needs to remain valid for the callback's
// prepare call; replay clears ALIAS_PROVIDED_DATA on substituted executable
// parameters before handing them to the HAL.
typedef iree_status_t (*iree_hal_replay_executable_substitution_fn_t)(
    void* user_data,
    const iree_hal_replay_executable_substitution_request_t* request,
    iree_hal_replay_executable_substitution_t* out_substitution);

typedef struct iree_hal_replay_executable_substitution_callback_t {
  // Callback invoked for each captured executable prepare operation.
  iree_hal_replay_executable_substitution_fn_t fn;
  // Opaque callback state passed to |fn|.
  void* user_data;
} iree_hal_replay_executable_substitution_callback_t;

typedef struct iree_hal_replay_execute_options_t {
  // Execution flags controlling replay behavior.
  iree_hal_replay_execute_flags_t flags;
  // Number of entries in |file_path_remaps|.
  iree_host_size_t file_path_remap_count;
  // Optional captured-prefix to replay-prefix rewrites for external files.
  const iree_hal_replay_file_path_remap_t* file_path_remaps;
  // Optional callback used to replace captured executable payloads.
  iree_hal_replay_executable_substitution_callback_t
      executable_substitution_callback;
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
