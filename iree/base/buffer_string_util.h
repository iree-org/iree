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

// Utilities for working with strings defining buffers and associated shapes,
// mostly useful for testing.
//
// The canonical shaped buffer string format is:
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

#ifndef IREE_BASE_BUFFER_STRING_UTIL_H_
#define IREE_BASE_BUFFER_STRING_UTIL_H_

#include <stddef.h>
#include <stdint.h>

#include <ostream>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "iree/base/shape.h"
#include "iree/base/status.h"

namespace iree {

// Defines how the elements within a buffer representation are interpreted
// during printing.
enum class BufferDataPrintMode {
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

// Returns the BufferDataPrintMode based on the shortened char in |str|.
StatusOr<BufferDataPrintMode> ParseBufferDataPrintMode(absl::string_view str);

// Returns the size, in bytes, of the given type.
StatusOr<int> ParseBufferTypeElementSize(absl::string_view type_str);

// Returns the canonical representation of a type based on its size in bytes and
// the specified printing mode. For example, with a size of 4 and a printing
// mode of kFloatingPoint it returns "f32".
std::string MakeBufferTypeString(int element_size,
                                 BufferDataPrintMode print_mode);

// Returns a Shape parsed from the given NxMx... string.
StatusOr<Shape> ParseShape(absl::string_view shape_str);

// Prints a shape and element type, e.g. 2x3xf32
std::string PrintShapedTypeToString(const Shape& shape,
                                    absl::string_view type_str);
void PrintShapedTypeToString(const Shape& shape, absl::string_view type_str,
                             std::string* out_result);
void PrintShapedTypeToStream(const Shape& shape, absl::string_view type_str,
                             std::ostream* stream);

// Prints the given bytes as binary hex data.
// Bytes are grouped as elements according to the |element_size| in bytes. If
// the size of |contents| exceeds |max_entries|, the output will be truncated to
// that many entries followed by an ellipses.
StatusOr<std::string> PrintBinaryDataToString(
    int element_size, absl::Span<const uint8_t> contents, size_t max_entries);
Status PrintBinaryDataToString(int element_size,
                               absl::Span<const uint8_t> contents,
                               size_t max_entries, std::string* out_result);
Status PrintBinaryDataToStream(int element_size,
                               absl::Span<const uint8_t> contents,
                               size_t max_entries, std::ostream* stream);

// Prints a list of elements in a format indicated by the given shape.
// For example: [1 2 3][4 5 6] for a shape of 2x3.
// The bytes in contents will be interpreted as the type specified by
// |type_str|. If the size of |contents| exceeds |max_entries|, the output will
// be truncated to that many entries followed by an ellipses.
StatusOr<std::string> PrintNumericalDataToString(
    const Shape& shape, absl::string_view type_str,
    absl::Span<const uint8_t> contents, size_t max_entries);
Status PrintNumericalDataToString(const Shape& shape,
                                  absl::string_view type_str,
                                  absl::Span<const uint8_t> contents,
                                  size_t max_entries, std::string* out_result);
Status PrintNumericalDataToStream(const Shape& shape,
                                  absl::string_view type_str,
                                  absl::Span<const uint8_t> contents,
                                  size_t max_entries, std::ostream* stream);

// Parses |data_str| as elements of the type specified by |type_str| and writes
// them into |output|.
Status ParseBufferDataAsType(absl::string_view data_str,
                             absl::string_view type_str,
                             absl::Span<uint8_t> output);

// A non-owning struct for referencing parts of a string that describes a shaped
// buffer type, e.g. 1x2x3xf32=1 2 3 4 5 6
struct BufferStringParts {
  // The part of the string corresponding to the shape, e.g. 1x2x3.
  absl::string_view shape_str;
  // The part of the string corresponding to the type, e.g. f32
  absl::string_view type_str;
  // The part of the string corresponding to the buffer data, e.g. 1 2 3 4 5 6
  absl::string_view data_str;

  // Extract the corresponding string parts from a string describing the entire
  // buffer.
  static BufferStringParts ExtractFrom(absl::string_view shaped_buf_str);
};

}  // namespace iree

#endif  // IREE_BASE_BUFFER_STRING_UTIL_H_
