// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_EXECUTABLE_DEBUG_INFO_H_
#define IREE_HAL_UTILS_EXECUTABLE_DEBUG_INFO_H_

#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
#include "iree/schemas/executable_debug_info_reader.h"
#include "iree/schemas/executable_debug_info_verifier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Verifies per-export debug info is valid.
// Executables using debug info must call this as part of their verification.
iree_status_t iree_hal_debug_verify_export_def(
    iree_hal_debug_ExportDef_table_t export_def);

// Basic debug information referencing allocated host memory.
typedef struct iree_hal_debug_export_info_t {
  iree_string_view_t function_name;
  iree_string_view_t source_filename;
  uint32_t source_line;
} iree_hal_debug_export_info_t;

// Returns the size in bytes required to store a copy of the export debug info.
// Callers should allocate this amount of memory to populate with
// iree_hal_debug_copy_export_info.
iree_host_size_t iree_hal_debug_calculate_export_info_size(
    iree_hal_debug_ExportDef_table_t export_def);

// Clones the given export flatbuffer data into a heap structure allocated with
// at least the size as calculated by iree_hal_debug_calculate_export_info_size.
// The storage is valid until freed by the caller and decoupled from the
// Flatbuffer storage. Returns the size copied (matching
// iree_hal_debug_calculate_export_info_size).
iree_host_size_t iree_hal_debug_copy_export_info(
    iree_hal_debug_ExportDef_table_t export_def,
    iree_hal_debug_export_info_t* out_info);

// Publishes the given source files to any attached debug/trace providers.
// This must be called prior to emitting any debug/trace events that reference
// the files that are contained within.
void iree_hal_debug_publish_source_files(
    iree_hal_debug_SourceFileDef_vec_t source_files_vec);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_EXECUTABLE_DEBUG_INFO_H_
