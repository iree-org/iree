// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/executable_debug_info.h"

static iree_status_t iree_hal_debug_verify_string_nonempty(
    const char* field_name, flatbuffers_string_t value) {
  if (flatbuffers_string_len(value) == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected debug info field `%s` to contain a non-empty string value",
        field_name);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_debug_verify_FileLineLocDef(
    iree_hal_debug_FileLineLocDef_table_t def) {
  if (!def) return iree_ok_status();
  return iree_hal_debug_verify_string_nonempty(
      "filename", iree_hal_debug_FileLineLocDef_filename_get(def));
}

iree_status_t iree_hal_debug_verify_export_def(
    iree_hal_debug_ExportDef_table_t export_def) {
  if (!export_def) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_debug_verify_FileLineLocDef(
      iree_hal_debug_ExportDef_location_get(export_def)));

  iree_hal_debug_StageLocationDef_vec_t stage_locations_vec =
      iree_hal_debug_ExportDef_stage_locations_get(export_def);
  for (iree_host_size_t i = 0;
       i < iree_hal_debug_StageLocationDef_vec_len(stage_locations_vec); ++i) {
    iree_hal_debug_StageLocationDef_table_t stage_location_def =
        iree_hal_debug_StageLocationDef_vec_at(stage_locations_vec, i);
    if (!stage_location_def) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage_locations[%" PRIhsz "] has NULL value", i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_debug_verify_string_nonempty(
                             "stage", iree_hal_debug_StageLocationDef_stage_get(
                                          stage_location_def)),
                         "verifying stage_locations[%" PRIhsz "]", i);
    IREE_RETURN_IF_ERROR(
        iree_hal_debug_verify_FileLineLocDef(
            iree_hal_debug_StageLocationDef_location_get(stage_location_def)),
        "verifying stage_locations[%" PRIhsz "]", i);
  }

  return iree_ok_status();
}

void iree_hal_debug_publish_source_files(
    iree_hal_debug_SourceFileDef_vec_t source_files_vec) {
  if (!source_files_vec) return;
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  for (iree_host_size_t i = 0;
       i < iree_hal_debug_SourceFileDef_vec_len(source_files_vec); ++i) {
    iree_hal_debug_SourceFileDef_table_t source_file =
        iree_hal_debug_SourceFileDef_vec_at(source_files_vec, i);
    if (!source_file) continue;
    flatbuffers_string_t path =
        iree_hal_debug_SourceFileDef_path_get(source_file);
    flatbuffers_uint8_vec_t content =
        iree_hal_debug_SourceFileDef_content_get(source_file);
    IREE_TRACE_PUBLISH_SOURCE_FILE(path, flatbuffers_string_len(path), content,
                                   flatbuffers_uint8_vec_len(content));
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
}
