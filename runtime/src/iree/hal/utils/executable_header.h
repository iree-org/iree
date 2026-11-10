// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_EXECUTABLE_HEADER_H_
#define IREE_HAL_UTILS_EXECUTABLE_HEADER_H_

#include "iree/base/api.h"
#include "iree/base/internal/flatcc/parsing.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Reads a iree_flatbuffer_file_header_t from |executable_data| and returns the
// contained flatbuffer data range in |out_flatbuffer_data|. If the total size
// of the executable data is unavailable the |unsafe_infer_size| flag allows for
// parsing without validation of file extents and the returned flatbuffer data
// will be sized to the range of valid bytes.
iree_status_t iree_hal_read_executable_flatbuffer_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    const char file_identifier[4], iree_const_byte_span_t* out_flatbuffer_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_EXECUTABLE_HEADER_H_
