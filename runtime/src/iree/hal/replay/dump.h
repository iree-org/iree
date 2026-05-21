// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_DUMP_H_
#define IREE_HAL_REPLAY_DUMP_H_

#include "iree/base/api.h"
#include "iree/hal/replay/format.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Output projection emitted by iree_hal_replay_dump_file.
typedef enum iree_hal_replay_dump_format_e {
  // Human-oriented text summary of replay records.
  IREE_HAL_REPLAY_DUMP_FORMAT_TEXT = 0,
  // One JSON object per line. Blob payload bytes are represented as file
  // ranges.
  IREE_HAL_REPLAY_DUMP_FORMAT_JSONL = 1,
} iree_hal_replay_dump_format_t;

// Options controlling replay dump output.
typedef struct iree_hal_replay_dump_options_t {
  // Output format to emit.
  iree_hal_replay_dump_format_t format;
} iree_hal_replay_dump_options_t;

// Streaming sink used by the dumper.
typedef iree_status_t (*iree_hal_replay_dump_write_fn_t)(
    void* user_data, iree_string_view_t text);

// Callback invoked for each text fragment emitted by the dumper.
typedef struct iree_hal_replay_dump_write_callback_t {
  // Function receiving the next text fragment.
  iree_hal_replay_dump_write_fn_t fn;
  // Opaque callback state passed to |fn|.
  void* user_data;
} iree_hal_replay_dump_write_callback_t;

// Returns default replay dump options.
static inline iree_hal_replay_dump_options_t
iree_hal_replay_dump_options_default(void) {
  iree_hal_replay_dump_options_t options;
  options.format = IREE_HAL_REPLAY_DUMP_FORMAT_TEXT;
  return options;
}

// Dumps |file_contents| to |write| according to |options|.
//
// The dumper never materializes blob bytes in textual projections. Payload and
// blob data are reported as validated byte ranges in the original replay file
// so large captures can be queried without copying model or activation data.
IREE_API_EXPORT iree_status_t
iree_hal_replay_dump_file(iree_const_byte_span_t file_contents,
                          const iree_hal_replay_dump_options_t* options,
                          iree_hal_replay_dump_write_callback_t write_callback,
                          iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_DUMP_H_
