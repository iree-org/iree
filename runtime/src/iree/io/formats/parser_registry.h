// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FORMATS_PARSER_REGISTRY_H_
#define IREE_IO_FORMATS_PARSER_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses a parameter file index in the opened |file_handle|.
// |path| is used for logging and file format identification. It may either be
// the original file path of |file_handle| or an extension (such as `irpa`).
// Upon return any parameters in the file are appended to the |index|.
IREE_API_EXPORT iree_status_t iree_io_parse_file_index(
    iree_string_view_t path, iree_io_file_handle_t* file_handle,
    iree_io_parameter_index_t* index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FORMATS_PARSER_REGISTRY_H_
