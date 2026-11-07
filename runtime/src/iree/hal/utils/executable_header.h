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

// Reads an iree_flatbuffer_file_header_t from |executable_data| and returns the
// contained flatbuffer data range in |out_flatbuffer_data|. The result borrows
// |executable_data|'s storage. Accepts only payload version 0.
//
// If the total executable size is unavailable, |unsafe_infer_size| permits
// parsing without validating file extents. The caller must ensure that the
// header and its declared content are readable.
iree_status_t iree_hal_read_executable_flatbuffer_header(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    const char file_identifier[4], iree_const_byte_span_t* out_flatbuffer_data);

// As above, accepting payload versions from 0 through |max_supported_version|.
// Callers must implement every accepted version's payload contracts.
iree_status_t iree_hal_read_executable_flatbuffer_header_versioned(
    iree_const_byte_span_t executable_data, bool unsafe_infer_size,
    const char file_identifier[4], uint32_t max_supported_version,
    iree_const_byte_span_t* out_flatbuffer_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_EXECUTABLE_HEADER_H_
