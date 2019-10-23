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

#include "hal/buffer_view_string_util.h"

#include <functional>
#include <sstream>
#include <type_traits>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "base/source_location.h"
#include "base/status.h"
#include "hal/heap_buffer.h"

namespace iree {
namespace hal {

namespace {

/* clang-format off */
constexpr char kHexValue[256] = {
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 0, 0, 0, 0, 0, 0,  // '0'..'9'
    0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 'A'..'F'
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 'a'..'f'
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,  0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};
/* clang-format on */

template <typename T>
void HexStringToBytes(const char* from, T to, ptrdiff_t num) {
  for (int i = 0; i < num; i++) {
    to[i] = (kHexValue[from[i * 2] & 0xFF] << 4) +
            (kHexValue[from[i * 2 + 1] & 0xFF]);
  }
}

constexpr char kHexTable[513] =
    "000102030405060708090a0b0c0d0e0f"
    "101112131415161718191a1b1c1d1e1f"
    "202122232425262728292a2b2c2d2e2f"
    "303132333435363738393a3b3c3d3e3f"
    "404142434445464748494a4b4c4d4e4f"
    "505152535455565758595a5b5c5d5e5f"
    "606162636465666768696a6b6c6d6e6f"
    "707172737475767778797a7b7c7d7e7f"
    "808182838485868788898a8b8c8d8e8f"
    "909192939495969798999a9b9c9d9e9f"
    "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf"
    "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf"
    "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf"
    "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf"
    "e0e1e2e3e4e5e6e7e8e9eaebecedeeef"
    "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff";

template <typename T>
void BytesToHexString(const unsigned char* src, T dest, ptrdiff_t num) {
  auto dest_ptr = &dest[0];
  for (auto src_ptr = src; src_ptr != (src + num); ++src_ptr, dest_ptr += 2) {
    const char* hex_p = &kHexTable[*src_ptr * 2];
    std::copy(hex_p, hex_p + 2, dest_ptr);
  }
}

// Returns true if the given type is represented as binary hex data.
bool IsBinaryType(absl::string_view type_str) {
  return !type_str.empty() && absl::ascii_isdigit(type_str[0]);
}

// Parses binary hex data.
Status ParseBinaryData(absl::string_view data_str, Buffer* buffer) {
  data_str = absl::StripAsciiWhitespace(data_str);
  ASSIGN_OR_RETURN(auto mapping,
                   buffer->MapMemory<uint8_t>(MemoryAccess::kDiscardWrite));
  auto contents = mapping.mutable_contents();
  size_t dst_i = 0;
  size_t src_i = 0;
  while (src_i < data_str.size() && dst_i < contents.size()) {
    char c = data_str[src_i];
    if (absl::ascii_isspace(c) || c == ',') {
      ++src_i;
      continue;
    }
    if (src_i + 1 >= data_str.size()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Invalid input hex data (offset=" << src_i << ")";
    }
    HexStringToBytes(data_str.data() + src_i, contents.data() + dst_i, 1);
    src_i += 2;
    ++dst_i;
  }
  if (dst_i < contents.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Too few elements to fill type; expected " << contents.size()
           << " but only read " << dst_i;
  } else if (data_str.size() - src_i > 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains more elements than the underlying "
              "buffer ("
           << contents.size() << ")";
  }
  return OkStatus();
}

// Prints binary hex data.
Status PrintBinaryData(int element_size, Buffer* buffer, size_t max_entries,
                       std::ostream* stream) {
  max_entries *= element_size;  // Counting bytes, but treat them as elements.
  ASSIGN_OR_RETURN(auto mapping,
                   buffer->MapMemory<uint8_t>(MemoryAccess::kRead));
  auto contents = mapping.contents();
  char hex_buffer[8 * 2];
  for (size_t i = 0; i < std::min(max_entries, mapping.size());
       i += element_size) {
    if (i > 0) *stream << " ";
    BytesToHexString(contents.data() + i, hex_buffer, element_size);
    *stream << hex_buffer;
  }
  if (mapping.size() > max_entries) *stream << "...";
  return OkStatus();
}

template <typename ElementType, typename Enabled = void>
struct SimpleStrToValue {
  absl::optional<ElementType> operator()(absl::string_view text) const = delete;
};

template <typename IntegerType>
struct SimpleStrToValue<
    IntegerType,
    typename std::enable_if<(sizeof(IntegerType) < 4), void>::type> {
  absl::optional<IntegerType> operator()(absl::string_view text) const {
    int32_t value;
    return absl::SimpleAtoi(text, &value) ? absl::optional<IntegerType>{value}
                                          : absl::nullopt;
  }
};

template <typename IntegerType>
struct SimpleStrToValue<
    IntegerType,
    typename std::enable_if<(sizeof(IntegerType) >= 4), void>::type> {
  absl::optional<IntegerType> operator()(absl::string_view text) const {
    IntegerType value;
    return absl::SimpleAtoi(text, &value) ? absl::optional<IntegerType>{value}
                                          : absl::nullopt;
  }
};

template <>
struct SimpleStrToValue<float, void> {
  absl::optional<float> operator()(absl::string_view text) const {
    float value;
    return absl::SimpleAtof(text, &value) ? absl::optional<float>{value}
                                          : absl::nullopt;
  }
};

template <>
struct SimpleStrToValue<double, void> {
  absl::optional<double> operator()(absl::string_view text) const {
    double value;
    return absl::SimpleAtod(text, &value) ? absl::optional<double>{value}
                                          : absl::nullopt;
  }
};

template <typename T>
Status ParseNumericalDataElement(absl::string_view data_str, size_t token_start,
                                 size_t token_end, absl::Span<T> contents,
                                 int dst_i) {
  if (dst_i >= contents.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains more elements than the underlying "
              "buffer ("
           << contents.size() << ")";
  }
  auto element_str = data_str.substr(token_start, token_end - token_start + 1);
  auto element = SimpleStrToValue<T>()(element_str);
  if (!element.has_value()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unable to parse element " << dst_i << " = '" << element_str
           << "'";
  }
  contents[dst_i] = element.value();
  return OkStatus();
}

template <typename T>
Status ParseNumericalDataAsType(absl::string_view data_str, Buffer* buffer) {
  ASSIGN_OR_RETURN(auto mapping,
                   buffer->MapMemory<T>(MemoryAccess::kDiscardWrite));
  auto contents = mapping.mutable_contents();
  size_t src_i = 0;
  size_t dst_i = 0;
  size_t token_start = std::string::npos;
  while (src_i < data_str.size()) {
    char c = data_str[src_i++];
    bool is_separator =
        absl::ascii_isspace(c) || c == ',' || c == '[' || c == ']';
    if (token_start == std::string::npos) {
      if (!is_separator) {
        token_start = src_i - 1;
      }
      continue;
    } else if (token_start != std::string::npos && !is_separator) {
      continue;
    }
    RETURN_IF_ERROR(ParseNumericalDataElement<T>(data_str, token_start,
                                                 src_i - 2, contents, dst_i++));
    token_start = std::string::npos;
  }
  if (token_start != std::string::npos) {
    RETURN_IF_ERROR(ParseNumericalDataElement<T>(
        data_str, token_start, data_str.size() - 1, contents, dst_i++));
  }
  if (dst_i < contents.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains fewer elements than the underlying "
              "buffer (expected "
           << contents.size() << ")";
  }
  return OkStatus();
}

// Parses numerical data (ints, floats, etc) in some typed form.
Status ParseNumericalData(absl::string_view type_str,
                          absl::string_view data_str, Buffer* buffer) {
  if (type_str == "i8") {
    return ParseNumericalDataAsType<int8_t>(data_str, buffer);
  } else if (type_str == "u8") {
    return ParseNumericalDataAsType<uint8_t>(data_str, buffer);
  } else if (type_str == "i16") {
    return ParseNumericalDataAsType<int16_t>(data_str, buffer);
  } else if (type_str == "u16") {
    return ParseNumericalDataAsType<uint16_t>(data_str, buffer);
  } else if (type_str == "i32") {
    return ParseNumericalDataAsType<int32_t>(data_str, buffer);
  } else if (type_str == "u32") {
    return ParseNumericalDataAsType<uint32_t>(data_str, buffer);
  } else if (type_str == "i64") {
    return ParseNumericalDataAsType<int64_t>(data_str, buffer);
  } else if (type_str == "u64") {
    return ParseNumericalDataAsType<uint64_t>(data_str, buffer);
  } else if (type_str == "f32") {
    return ParseNumericalDataAsType<float>(data_str, buffer);
  } else if (type_str == "f64") {
    return ParseNumericalDataAsType<double>(data_str, buffer);
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unsupported type: " << type_str;
  }
}

template <typename T>
void PrintElementList(const Shape& shape, absl::Span<const T> data,
                      size_t* max_entries, std::ostream* stream) {
  if (shape.empty()) {
    // Scalar value.
    PrintElementList({1}, data, max_entries, stream);
    return;
  } else if (shape.size() == 1) {
    // Leaf dimension; output data.
    size_t max_count = std::min(*max_entries, static_cast<size_t>(shape[0]));
    *stream << absl::StrJoin(data.subspan(0, max_count), " ");
    if (max_count < shape[0]) {
      *stream << "...";
    }
    *max_entries -= max_count;
  } else {
    // Nested; recurse into next dimension.
    Shape nested_shape = Shape(shape.subspan(1));
    size_t length = nested_shape.element_count();
    size_t offset = 0;
    for (int i = 0; i < shape[0]; ++i) {
      *stream << "[";
      PrintElementList<T>(nested_shape, data.subspan(offset, length),
                          max_entries, stream);
      offset += length;
      *stream << "]";
    }
  }
}

template <typename T>
Status PrintNumericalDataAsType(const Shape& shape, Buffer* buffer,
                                size_t max_entries, std::ostream* stream) {
  ASSIGN_OR_RETURN(auto mapping, buffer->MapMemory<T>(MemoryAccess::kRead));
  PrintElementList(shape, mapping.contents(), &max_entries, stream);
  return OkStatus();
}

// Prints numerical data (ints, floats, etc) from some typed form.
Status PrintNumericalData(const Shape& shape, absl::string_view type_str,
                          Buffer* buffer, size_t max_entries,
                          std::ostream* stream) {
  if (type_str == "i8") {
    return PrintNumericalDataAsType<int8_t>(shape, buffer, max_entries, stream);
  } else if (type_str == "u8") {
    return PrintNumericalDataAsType<uint8_t>(shape, buffer, max_entries,
                                             stream);
  } else if (type_str == "i16") {
    return PrintNumericalDataAsType<int16_t>(shape, buffer, max_entries,
                                             stream);
  } else if (type_str == "u16") {
    return PrintNumericalDataAsType<uint16_t>(shape, buffer, max_entries,
                                              stream);
  } else if (type_str == "i32") {
    return PrintNumericalDataAsType<int32_t>(shape, buffer, max_entries,
                                             stream);
  } else if (type_str == "u32") {
    return PrintNumericalDataAsType<uint32_t>(shape, buffer, max_entries,
                                              stream);
  } else if (type_str == "i64") {
    return PrintNumericalDataAsType<int64_t>(shape, buffer, max_entries,
                                             stream);
  } else if (type_str == "u64") {
    return PrintNumericalDataAsType<uint64_t>(shape, buffer, max_entries,
                                              stream);
  } else if (type_str == "f32") {
    return PrintNumericalDataAsType<float>(shape, buffer, max_entries, stream);
  } else if (type_str == "f64") {
    return PrintNumericalDataAsType<double>(shape, buffer, max_entries, stream);
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unsupported type: " << type_str;
  }
}

}  // namespace

StatusOr<int> GetTypeElementSize(absl::string_view type_str) {
  if (type_str.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC) << "Type is empty";
  } else if (IsBinaryType(type_str)) {
    // If the first character is a digit then we are dealign with binary data.
    // The type is just the number of bytes per element.
    int element_size = 0;
    if (!absl::SimpleAtoi(type_str, &element_size)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unable to parse element size type '" << type_str << "'";
    }
    return element_size;
  }
  // We know that our types are single characters followed by bit counts.
  // If we start to support other types we may need to do something more clever.
  int bit_count = 0;
  if (!absl::SimpleAtoi(type_str.substr(1), &bit_count)) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unable to parse type bit count from '" << type_str
           << "'; expecting something like 'i32'";
  }
  return bit_count / 8;
}

StatusOr<Shape> ParseShape(absl::string_view shape_str) {
  std::vector<int> dims;
  for (auto dim_str : absl::StrSplit(shape_str, 'x', absl::SkipWhitespace())) {
    int dim_value = 0;
    if (!absl::SimpleAtoi(dim_str, &dim_value)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Invalid shape dimension '" << dim_str
             << "' while parsing shape '" << shape_str << "'";
    }
    dims.push_back(dim_value);
  }
  return Shape{dims};
}

StatusOr<BufferView> ParseBufferViewFromString(
    absl::string_view buffer_view_str, hal::Allocator* allocator) {
  // Strip whitespace that may come along (linefeeds/etc).
  buffer_view_str = absl::StripAsciiWhitespace(buffer_view_str);
  if (buffer_view_str.empty()) {
    // Empty lines denote empty buffer_views.
    return BufferView{};
  }

  // Split into the components we can work with: shape, type, and data.
  absl::string_view shape_and_type_str;
  absl::string_view data_str;
  auto equal_index = buffer_view_str.find('=');
  if (equal_index == std::string::npos) {
    // Treat a lack of = as defaulting the data to zeros.
    shape_and_type_str = buffer_view_str;
  } else {
    shape_and_type_str = buffer_view_str.substr(0, equal_index);
    data_str = buffer_view_str.substr(equal_index + 1);
  }
  absl::string_view shape_str;
  absl::string_view type_str;
  auto last_x_index = shape_and_type_str.rfind('x');
  if (last_x_index == std::string::npos) {
    // Scalar.
    type_str = shape_and_type_str;
  } else {
    // Has a shape.
    shape_str = shape_and_type_str.substr(0, last_x_index);
    type_str = shape_and_type_str.substr(last_x_index + 1);
  }

  // Populate BufferView metadata required for allocation.
  BufferView result;
  ASSIGN_OR_RETURN(result.element_size, GetTypeElementSize(type_str));
  ASSIGN_OR_RETURN(result.shape, ParseShape(shape_str));

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

  if (!data_str.empty()) {
    // Parse the data from the string right into the buffer.
    if (IsBinaryType(type_str)) {
      // Parse as binary hex.
      RETURN_IF_ERROR(ParseBinaryData(data_str, result.buffer.get()));
    } else {
      // Parse as some nicely formatted type.
      RETURN_IF_ERROR(
          ParseNumericalData(type_str, data_str, result.buffer.get()));
    }
  }

  return result;
}

StatusOr<BufferViewPrintMode> ParseBufferViewPrintMode(absl::string_view str) {
  char str_char = str.empty() ? '?' : str[0];
  switch (str_char) {
    case 'b':
      return BufferViewPrintMode::kBinary;
    case 'i':
      return BufferViewPrintMode::kSignedInteger;
    case 'u':
      return BufferViewPrintMode::kUnsignedInteger;
    case 'f':
      return BufferViewPrintMode::kFloatingPoint;
    default:
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported output type '" << str << "'";
  }
}

StatusOr<std::string> PrintBufferViewToString(const BufferView& buffer_view,
                                              BufferViewPrintMode print_mode,
                                              size_t max_entries) {
  std::string result;
  RETURN_IF_ERROR(
      PrintBufferViewToString(buffer_view, print_mode, max_entries, &result));
  return result;
}

Status PrintBufferViewToString(const BufferView& buffer_view,
                               BufferViewPrintMode print_mode,
                               size_t max_entries, std::string* out_result) {
  std::ostringstream stream;
  RETURN_IF_ERROR(
      PrintBufferViewToStream(buffer_view, print_mode, max_entries, &stream));
  *out_result = stream.str();
  return OkStatus();
}

Status PrintBufferViewToStream(const BufferView& buffer_view,
                               BufferViewPrintMode print_mode,
                               size_t max_entries, std::ostream* stream) {
  if (!buffer_view.buffer) {
    // No buffer means the buffer_view is empty. We use the empty string to
    // denote this (as we have no useful information).
    return OkStatus();
  }

  // Pick a type based on the element size and the printing mode.
  std::string type_str;
  switch (print_mode) {
    case BufferViewPrintMode::kBinary:
      type_str = std::to_string(buffer_view.element_size);
      break;
    case BufferViewPrintMode::kSignedInteger:
      absl::StrAppend(&type_str, "i", buffer_view.element_size * 8);
      break;
    case BufferViewPrintMode::kUnsignedInteger:
      absl::StrAppend(&type_str, "u", buffer_view.element_size * 8);
      break;
    case BufferViewPrintMode::kFloatingPoint:
      absl::StrAppend(&type_str, "f", buffer_view.element_size * 8);
      break;
  }

  // [shape]x[type]= prefix (taking into account scalar values).
  *stream << absl::StrJoin(buffer_view.shape.begin(), buffer_view.shape.end(),
                           "x");
  if (!buffer_view.shape.empty()) *stream << "x";
  *stream << type_str;
  *stream << "=";

  if (IsBinaryType(type_str)) {
    return PrintBinaryData(buffer_view.element_size, buffer_view.buffer.get(),
                           max_entries, stream);
  } else {
    return PrintNumericalData(buffer_view.shape, type_str,
                              buffer_view.buffer.get(), max_entries, stream);
  }
}

}  // namespace hal
}  // namespace iree
