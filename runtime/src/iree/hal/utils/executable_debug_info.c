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

// TODO(benvanik): a way to select what location is chosen. For now we just
// pick the first stage location if present and otherwise use the source
// location.
static iree_hal_debug_FileLineLocDef_table_t
iree_hal_debug_select_source_location(
    iree_hal_debug_ExportDef_table_t export_def) {
  iree_hal_debug_StageLocationDef_vec_t stage_locations_vec =
      iree_hal_debug_ExportDef_stage_locations_get(export_def);
  if (iree_hal_debug_StageLocationDef_vec_len(stage_locations_vec) > 0) {
    iree_hal_debug_StageLocationDef_table_t stage_location_def =
        iree_hal_debug_StageLocationDef_vec_at(stage_locations_vec, 0);
    return iree_hal_debug_StageLocationDef_location_get(stage_location_def);
  }
  return iree_hal_debug_ExportDef_location_get(export_def);
}

iree_host_size_t iree_hal_debug_calculate_export_info_size(
    iree_hal_debug_ExportDef_table_t export_def) {
  if (!export_def) return 0;

  iree_host_size_t total_size = sizeof(iree_hal_debug_export_info_t);
  total_size +=
      flatbuffers_string_len(iree_hal_debug_ExportDef_name_get(export_def));

  iree_hal_debug_FileLineLocDef_table_t location_def =
      iree_hal_debug_select_source_location(export_def);
  if (location_def) {
    total_size += flatbuffers_string_len(
        iree_hal_debug_FileLineLocDef_filename_get(location_def));
  }

  return total_size;
}

iree_host_size_t iree_hal_debug_copy_export_info(
    iree_hal_debug_ExportDef_table_t export_def,
    iree_hal_debug_export_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));
  if (!export_def) return 0;

  iree_host_size_t total_size = sizeof(iree_hal_debug_export_info_t);
  char* ptr = (char*)out_info + sizeof(*out_info);

  flatbuffers_string_t name = iree_hal_debug_ExportDef_name_get(export_def);
  if (name) {
    size_t name_length = flatbuffers_string_len(name);
    total_size += name_length;
    memcpy(ptr, name, name_length);
    out_info->function_name = iree_make_string_view(ptr, name_length);
    ptr += name_length;
  }

  iree_hal_debug_FileLineLocDef_table_t location_def =
      iree_hal_debug_select_source_location(export_def);
  if (location_def) {
    flatbuffers_string_t filename =
        iree_hal_debug_FileLineLocDef_filename_get(location_def);
    size_t filename_length = flatbuffers_string_len(filename);
    total_size += filename_length;
    memcpy(ptr, filename, filename_length);
    out_info->source_filename = iree_make_string_view(ptr, filename_length);
    ptr += filename_length;
  }

  return total_size;
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
