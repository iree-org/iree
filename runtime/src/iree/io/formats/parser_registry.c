// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/formats/parser_registry.h"

#include "iree/base/internal/path.h"
#include "iree/io/formats/gguf/gguf_parser.h"
#include "iree/io/formats/irpa/irpa_parser.h"
#include "iree/io/formats/safetensors/safetensors_parser.h"

IREE_API_EXPORT iree_status_t iree_io_parse_file_index(
    iree_string_view_t path, iree_io_file_handle_t* file_handle,
    iree_io_parameter_index_t* index) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_TEXT(z0, path.data, path.size);

  // Try to get the extension from the path but also support the user passing in
  // just the raw extension as well.
  iree_string_view_t basename = iree_string_view_empty();
  iree_string_view_t extension = iree_string_view_empty();
  iree_file_path_split_basename(path, &basename, &extension);
  if (iree_string_view_is_empty(extension)) {
    extension = basename;
  }

  iree_status_t status = iree_ok_status();
  if (iree_string_view_equal_case(extension, IREE_SV("irpa"))) {
    status = iree_io_parse_irpa_index(file_handle, index);
  } else if (iree_string_view_equal_case(extension, IREE_SV("gguf"))) {
    status = iree_io_parse_gguf_index(file_handle, index);
  } else if (iree_string_view_equal_case(extension, IREE_SV("safetensors"))) {
    status = iree_io_parse_safetensors_index(file_handle, index);
  } else {
    status = iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported file format `%.*s`; ensure the extension matches one of "
        "the supported formats: [.irpa, .gguf, .safetensors]",
        (int)extension.size, extension.data);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
