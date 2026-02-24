// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_SYSFS_H_
#define IREE_BASE_INTERNAL_SYSFS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Linux sysfs file I/O utilities
//===----------------------------------------------------------------------===//
// Low-level primitives for reading and parsing Linux sysfs files.
// These utilities are platform-specific (Linux-only) and are used by both
// the task topology system and HAL platform topology.

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// Returns the root path for sysfs system files.
// Defaults to "/sys/devices/system" but can be overridden at compile time
// by defining IREE_SYSFS_ROOT="/path/to/sysfs" for testing with mock
// filesystem snapshots.
const char* iree_sysfs_get_root_path(void);

//===----------------------------------------------------------------------===//
// File I/O
//===----------------------------------------------------------------------===//

// Reads a small sysfs file into the provided buffer.
// This is intended for small text files like those in /sys/devices/system/cpu/
// that typically contain a single number or short list.
//
// |path| must be a null-terminated filesystem path.
// |buffer| must be at least |buffer_size| bytes.
// |out_length| receives the number of bytes read (excluding any NUL added).
//
// The buffer is always NUL-terminated on success.
// Returns IREE_STATUS_NOT_FOUND if the file doesn't exist.
// Returns IREE_STATUS_OUT_OF_RANGE if the file exceeds buffer_size.
iree_status_t iree_sysfs_read_small_file(const char* path, char* buffer,
                                         size_t buffer_size,
                                         iree_host_size_t* out_length);

//===----------------------------------------------------------------------===//
// Parsing utilities
//===----------------------------------------------------------------------===//

// Callback for CPU list enumeration.
// Called once for each range in a CPU list (e.g., "0-3" calls with (0, 4)).
// |start_cpu| is inclusive, |end_cpu| is exclusive (half-open range).
// Return false to stop iteration.
typedef bool (*iree_sysfs_cpu_list_callback_t)(uint32_t start_cpu,
                                               uint32_t end_cpu,
                                               void* user_data);

// Parses a Linux CPU list format string.
// Examples: "0-191", "0,2-5,8", "0-3,5,7-9"
// Calls |callback| for each contiguous range.
// Single CPUs are reported as [N, N+1) ranges.
iree_status_t iree_sysfs_parse_cpu_list(iree_string_view_t text,
                                        iree_sysfs_cpu_list_callback_t callback,
                                        void* user_data);

// Parses a size string with optional K suffix (case-insensitive).
// Whitespace is trimmed before parsing.
// Examples: "32K" -> 32768, "1024K" -> 1048576, "1024" -> 1024
//
// Linux kernel cache size_show() only outputs K suffix and that's what we use
// most frequently with CPU, but beware that other sysfs nodes are in unsuffixed
// units of KB and we'll need to handle those differently.
iree_status_t iree_sysfs_parse_size_string(iree_string_view_t text,
                                           uint64_t* out_size);

//===----------------------------------------------------------------------===//
// Convenience helpers
//===----------------------------------------------------------------------===//

// Reads a sysfs file and parses it as a uint32.
iree_status_t iree_sysfs_read_uint32(const char* path, uint32_t* out_value);

// Reads a sysfs file and parses it as a size string (e.g., "32K").
iree_status_t iree_sysfs_read_size(const char* path, uint64_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_INTERNAL_SYSFS_H_
