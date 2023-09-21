// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FORMATS_GGUF_GGUF_FORMAT_H_
#define IREE_IO_FORMATS_GGUF_GGUF_FORMAT_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses a .gguf file and merges its contained resources into |index|.
//
// Specification:
// https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md
IREE_API_EXPORT iree_status_t iree_io_parse_gguf_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FORMATS_GGUF_GGUF_FORMAT_H_
