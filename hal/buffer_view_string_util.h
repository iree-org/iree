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

// Utilities for working with BufferView data, mostly useful for testing.
// These functions allow for conversion between types, parsing and printing, and
// basic comparisons.
//
// The canonical BufferView string format is:
//   [shape]x[type]=value,value,...
// For example:
//   2x2xi32=0,1,2,3
// Characters like [] are optional and will be ignored during parsing:
//   2x2xi32=[[0 1][2 3]]
//
// The type may be one of the following:
// * 1/2/4/8 = 1/2/4/8 byte elements in binary hex format.
// * i8/u8 = signed/unsigned 8-bit integers.
// * i16/u16 = signed/unsigned 16-bit integers.
// * i32/u32 = signed/unsigned 32-bit integers.
// * i64/u64 = signed/unsigned 64-bit integers.
// * f32 = 32-bit floating-point number.
// * f64 = 64-bit floating-point number.

#ifndef IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_
#define IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "base/status.h"
#include "hal/allocator.h"
#include "hal/buffer_view.h"

namespace iree {
namespace hal {

// Returns the size, in bytes, of the given type.
StatusOr<int> GetTypeElementSize(absl::string_view type_str);

// Returns a Shape parsed from the given NxMx... string.
StatusOr<Shape> ParseShape(absl::string_view shape_str);

// Parses a BufferView encoded in a string.
// If an |allocator| is provided the buffer will be allocated as host-local and
// device-visible. Otherwise, buffers will be host-local.
// The format accepted matches that produced by PrintBufferViewToString.
StatusOr<BufferView> ParseBufferViewFromString(
    absl::string_view buffer_view_str, hal::Allocator* allocator = nullptr);

// Defines how the elements within a BufferView are interpreted during printing.
enum class BufferViewPrintMode {
  // Interpret the data as if it were serialized bytes.
  // In this mode no conversion is performed and the bytes in memory are printed
  // as hex in groupings based on the element size. Shortened to 'b'.
  kBinary,
  // Interpret elements as signed integers; shortened to 'i'.
  kSignedInteger,
  // Interpret elements as unsigned integers; shortened to 'u'.
  kUnsignedInteger,
  // Interpret elements as floating-point values; shortened to 'f'.
  kFloatingPoint,
};

// Returns the BufferViewPrintMode based on the shortened char in |str|.
StatusOr<BufferViewPrintMode> ParseBufferViewPrintMode(absl::string_view str);

// Prints a BufferView to a string encoded in the canonical format.
StatusOr<std::string> PrintBufferViewToString(const BufferView& buffer_view,
                                              BufferViewPrintMode print_mode,
                                              size_t max_entries);
Status PrintBufferViewToString(const BufferView& buffer_view,
                               BufferViewPrintMode print_mode,
                               size_t max_entries, std::string* out_result);

// Prints a BufferView to a string stream encoded in the canonical format.
Status PrintBufferViewToStream(const BufferView& buffer_view,
                               BufferViewPrintMode print_mode,
                               size_t max_entries, std::ostream* stream);

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_BUFFER_VIEW_STRING_UTIL_H_
