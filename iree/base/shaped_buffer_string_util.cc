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

#include "iree/base/shaped_buffer_string_util.h"

#include <stddef.h>
#include <stdint.h>

#include <sstream>
#include <string>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/memory.h"
#include "iree/base/shape.h"
#include "iree/base/shaped_buffer.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"

namespace iree {

StatusOr<ShapedBuffer> ParseShapedBufferFromString(
    absl::string_view shaped_buf_str) {
  // Strip whitespace that may come along (linefeeds/etc).
  shaped_buf_str = absl::StripAsciiWhitespace(shaped_buf_str);
  shaped_buf_str = absl::StripPrefix(shaped_buf_str, "\"");
  shaped_buf_str = absl::StripSuffix(shaped_buf_str, "\"");
  if (shaped_buf_str.empty()) {
    // Empty lines denote empty shaped_buffers.
    return ShapedBuffer{};
  }

  // Split into the components we can work with: shape, type, and data.
  auto str_parts = BufferStringParts::ExtractFrom(shaped_buf_str);

  ASSIGN_OR_RETURN(int element_size,
                   ParseBufferTypeElementSize(str_parts.type_str));
  ASSIGN_OR_RETURN(Shape shape, ParseShape(str_parts.shape_str));

  int buffer_size = element_size * shape.element_count();
  std::vector<uint8_t> contents(buffer_size);

  if (!str_parts.data_str.empty()) {
    RETURN_IF_ERROR(ParseBufferDataAsType(
        str_parts.data_str, str_parts.type_str, absl::MakeSpan(contents)));
  }
  return ShapedBuffer(element_size, shape, std::move(contents));
}

StatusOr<std::string> PrintShapedBufferToString(const ShapedBuffer& shaped_buf,
                                                BufferDataPrintMode print_mode,
                                                size_t max_entries) {
  std::string result;
  RETURN_IF_ERROR(
      PrintShapedBufferToString(shaped_buf, print_mode, max_entries, &result));
  return result;
}

Status PrintShapedBufferToString(const ShapedBuffer& shaped_buf,
                                 BufferDataPrintMode print_mode,
                                 size_t max_entries, std::string* out_result) {
  std::ostringstream stream;
  RETURN_IF_ERROR(
      PrintShapedBufferToStream(shaped_buf, print_mode, max_entries, &stream));
  *out_result = stream.str();
  return OkStatus();
}

Status PrintShapedBufferToStream(const ShapedBuffer& shaped_buffer,
                                 BufferDataPrintMode print_mode,
                                 size_t max_entries, std::ostream* stream) {
  if (shaped_buffer.contents().empty()) {
    // No data means the shaped_buffer is empty. We use the empty string to
    // denote this (as we have no useful information).
    return OkStatus();
  }

  ASSIGN_OR_RETURN(
      std::string type_str,
      MakeBufferTypeString(shaped_buffer.element_size(), print_mode));

  PrintShapedTypeToStream(shaped_buffer.shape(), type_str, stream);
  *stream << "=";

  if (print_mode == BufferDataPrintMode::kBinary) {
    return PrintBinaryDataToStream(shaped_buffer.element_size(),
                                   shaped_buffer.contents(), max_entries,
                                   stream);
  }
  return PrintNumericalDataToStream(shaped_buffer.shape(), type_str,
                                    shaped_buffer.contents(), max_entries,
                                    stream);
}

}  // namespace iree
