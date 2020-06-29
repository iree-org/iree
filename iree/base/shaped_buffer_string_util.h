// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utilities for parsing and printing ShapedBuffer, mostly useful for
// testing. The format is as described in
// https://github.com/google/iree/tree/main/iree/base/buffer_string_util.h

#ifndef IREE_BASE_SHAPED_BUFFER_STRING_UTIL_H_
#define IREE_BASE_SHAPED_BUFFER_STRING_UTIL_H_

#include <stddef.h>

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/shape.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/status.h"

namespace iree {

// Parses a ShapedBuffer encoded in a string.
// The format accepted matches that produced by PrintShapedBufferToString.
StatusOr<ShapedBuffer> ParseShapedBufferFromString(
    absl::string_view shaped_buf_str);

// Prints a ShapedBuffer to a string encoded in the canonical format.
StatusOr<std::string> PrintShapedBufferToString(const ShapedBuffer& shaped_buf,
                                                BufferDataPrintMode print_mode,
                                                size_t max_entries);
Status PrintShapedBufferToString(const ShapedBuffer& shaped_buf,
                                 BufferDataPrintMode print_mode,
                                 size_t max_entries, std::string* out_result);

// Prints a ShapedBuffer to a stream encoded in the canonical format.
Status PrintShapedBufferToStream(const ShapedBuffer& shaped_buf,
                                 BufferDataPrintMode print_mode,
                                 size_t max_entries, std::ostream* stream);

}  // namespace iree

#endif  // IREE_BASE_SHAPED_BUFFER_STRING_UTIL_H_
