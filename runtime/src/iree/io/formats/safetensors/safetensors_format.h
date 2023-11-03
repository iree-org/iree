// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_IO_FORMATS_SAFETENSORS_SAFETENSORS_FORMAT_H_
#define IREE_IO_FORMATS_SAFETENSORS_SAFETENSORS_FORMAT_H_

#include "iree/base/api.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Parses a .safetensors file and merges its contained resources into |index|.
//
// Documentation: https://github.com/huggingface/safetensors
// This is a very basic archive file with some issues (no alignment, etc) but
// at least doesn't require Python pickle decoding (just JSON). The major reason
// to use this is if sourcing from a Hugging Face model that has its weights
// already in the safetensors format.
//
// WARNING: this implementation has not been thoroughly tested or verified as
// safe or correct. Use with caution only on trusted inputs. Tip: don't embed
// other file formats within your file format and call it "safe" as it's only
// going to be as safe as the implementations of the other file formats you
// embed. In this case a full JSON parser is required and must be safe and we
// don't take that dependency for a testing tool. Users wanting to productionize
// this should implement their own safetensors parser or use the rust one with
// all the fun that entails.
IREE_API_EXPORT iree_status_t iree_io_parse_safetensors_index(
    iree_io_file_handle_t* file_handle, iree_io_parameter_index_t* index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_IO_FORMATS_SAFETENSORS_SAFETENSORS_FORMAT_H_
