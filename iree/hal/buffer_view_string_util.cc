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

#include "iree/hal/buffer_view_string_util.h"

#include <cstdint>
#include <functional>
#include <sstream>
#include <type_traits>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "iree/base/buffer_string_util.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"
#include "iree/hal/heap_buffer.h"

namespace iree {
namespace hal {

StatusOr<BufferView> ParseBufferViewFromString(
    absl::string_view buffer_view_str, hal::Allocator* allocator) {
  // Strip whitespace that may come along (linefeeds/etc).
  buffer_view_str = absl::StripAsciiWhitespace(buffer_view_str);
  if (buffer_view_str.empty()) {
    // Empty lines denote empty buffer_views.
    return BufferView{};
  }

  // Split into the components we can work with: shape, type, and data.
  auto str_parts = BufferStringParts::ExtractFrom(buffer_view_str);

  // Populate BufferView metadata required for allocation.
  BufferView result;
  ASSIGN_OR_RETURN(result.element_size,
                   ParseBufferTypeElementSize(str_parts.type_str));
  ASSIGN_OR_RETURN(result.shape, ParseShape(str_parts.shape_str));

  // Allocate the host buffer.
  size_t allocation_size = result.shape.element_count() * result.element_size;
  if (allocator) {
    ASSIGN_OR_RETURN(
        result.buffer,
        allocator->Allocate(MemoryType::kHostLocal | MemoryType::kDeviceVisible,
                            BufferUsage::kAll | BufferUsage::kConstant,
                            allocation_size));
  } else {
    result.buffer = HeapBuffer::Allocate(
        MemoryType::kHostLocal, BufferUsage::kAll | BufferUsage::kConstant,
        allocation_size);
  }

  if (!str_parts.data_str.empty()) {
    ASSIGN_OR_RETURN(auto mapping, result.buffer.get()->MapMemory<uint8_t>(
                                       MemoryAccess::kDiscardWrite));
    auto contents = mapping.mutable_contents();
    // Parse the data from the string right into the buffer.
    RETURN_IF_ERROR(ParseBufferDataAsType(str_parts.data_str,
                                          str_parts.type_str, contents));
  }

  return result;
}

StatusOr<std::string> PrintBufferViewToString(const BufferView& buffer_view,
                                              BufferDataPrintMode print_mode,
                                              size_t max_entries) {
  std::string result;
  RETURN_IF_ERROR(
      PrintBufferViewToString(buffer_view, print_mode, max_entries, &result));
  return result;
}

Status PrintBufferViewToString(const BufferView& buffer_view,
                               BufferDataPrintMode print_mode,
                               size_t max_entries, std::string* out_result) {
  std::ostringstream stream;
  RETURN_IF_ERROR(
      PrintBufferViewToStream(buffer_view, print_mode, max_entries, &stream));
  *out_result = stream.str();
  return OkStatus();
}

Status PrintBufferViewToStream(const BufferView& buffer_view,
                               BufferDataPrintMode print_mode,
                               size_t max_entries, std::ostream* stream) {
  if (!buffer_view.buffer) {
    // No buffer means the buffer_view is empty. We use the empty string to
    // denote this (as we have no useful information).
    return OkStatus();
  }

  std::string type_str =
      MakeBufferTypeString(buffer_view.element_size, print_mode);
  PrintShapedTypeToStream(buffer_view.shape, type_str, stream);
  *stream << "=";

  ASSIGN_OR_RETURN(auto mapping, buffer_view.buffer.get()->MapMemory<uint8_t>(
                                     MemoryAccess::kRead));
  if (print_mode == BufferDataPrintMode::kBinary) {
    return PrintBinaryDataToStream(buffer_view.element_size, mapping.contents(),
                                   max_entries, stream);
  } else {
    return PrintNumericalDataToStream(buffer_view.shape, type_str,
                                      mapping.contents(), max_entries, stream);
  }
}

}  // namespace hal
}  // namespace iree
