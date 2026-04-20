// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_H_
#define IREE_HAL_REPLAY_RECORDER_H_

#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/device_group.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque shared recorder for one HAL replay capture stream.
typedef struct iree_hal_replay_recorder_t iree_hal_replay_recorder_t;

// Bitfield specifying replay recorder behavior.
typedef uint32_t iree_hal_replay_recorder_flags_t;
enum iree_hal_replay_recorder_flag_bits_t {
  IREE_HAL_REPLAY_RECORDER_FLAG_NONE = 0u,
};

// Policy for capturing imported fd-backed HAL files.
typedef uint32_t iree_hal_replay_recorder_external_file_policy_t;
enum iree_hal_replay_recorder_external_file_policy_e {
  // Capture external files by path and replay against the original file.
  //
  // Host-allocation-backed files are always embedded inline because they have
  // no durable external identity to reference.
  IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_REFERENCE = 0u,
  // Reject imported fd-backed files instead of producing a non-hermetic replay.
  //
  // Host-allocation-backed files are still embedded inline under this policy.
  IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_FAIL = 1u,
};

// Options controlling HAL replay recording.
typedef struct iree_hal_replay_recorder_options_t {
  // Flags controlling recorder behavior.
  iree_hal_replay_recorder_flags_t flags;
  // Policy used when an imported file is backed by an external file path.
  iree_hal_replay_recorder_external_file_policy_t external_file_policy;
  // Reserved for future recorder options; must be zero.
  uint32_t reserved0;
} iree_hal_replay_recorder_options_t;

// Returns default replay recorder options.
static inline iree_hal_replay_recorder_options_t
iree_hal_replay_recorder_options_default(void) {
  iree_hal_replay_recorder_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

// Creates a recorder that writes one `.ireereplay` stream to |file_handle|.
//
// The recorder immediately writes a session record. Callers must close the
// recorder with iree_hal_replay_recorder_close before releasing their final
// reference when they require a complete replay artifact.
IREE_API_EXPORT iree_status_t iree_hal_replay_recorder_create(
    iree_io_file_handle_t* file_handle,
    const iree_hal_replay_recorder_options_t* options,
    iree_allocator_t host_allocator, iree_hal_replay_recorder_t** out_recorder);

// Retains |recorder| for the caller.
IREE_API_EXPORT void iree_hal_replay_recorder_retain(
    iree_hal_replay_recorder_t* recorder);

// Releases |recorder| from the caller.
IREE_API_EXPORT void iree_hal_replay_recorder_release(
    iree_hal_replay_recorder_t* recorder);

// Closes |recorder| and writes the final file length.
//
// Safe to call multiple times. If any earlier record write failed, close
// returns that terminal recorder status instead of silently finalizing a
// partial stream.
IREE_API_EXPORT iree_status_t
iree_hal_replay_recorder_close(iree_hal_replay_recorder_t* recorder);

// Creates a device group whose devices record operations to |recorder|.
//
// |base_group| is retained by the wrappers so the underlying devices keep their
// original topology assignment. The returned group contains replacement wrapper
// devices in the same topology order, with the source topology copied into the
// returned group. All wrappers share |recorder| so multi-device captures are
// emitted to one ordered stream.
IREE_API_EXPORT iree_status_t iree_hal_replay_wrap_device_group(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_H_
