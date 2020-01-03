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

#include "iree/base/buffer_string_util.h"

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "iree/base/memory.h"
#include "iree/base/source_location.h"
#include "iree/base/status.h"

namespace iree {

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

// Like the absl method, but works in-place.
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
Status ParseBinaryData(absl::string_view data_str, absl::Span<uint8_t> output) {
  data_str = absl::StripAsciiWhitespace(data_str);
  size_t dst_i = 0;
  size_t src_i = 0;
  while (src_i < data_str.size() && dst_i < output.size()) {
    char c = data_str[src_i];
    if (absl::ascii_isspace(c) || c == ',') {
      ++src_i;
      continue;
    }
    if (src_i + 1 >= data_str.size()) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Invalid input hex data (offset=" << src_i << ")";
    }
    HexStringToBytes(data_str.data() + src_i, output.data() + dst_i, 1);
    src_i += 2;
    ++dst_i;
  }
  if (dst_i < output.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Too few elements to fill type; expected " << output.size()
           << " but only read " << dst_i;
  } else if (data_str.size() - src_i > 0) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains more elements than the underlying "
              "buffer ("
           << output.size() << "): " << data_str;
  }
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
                                 size_t token_end, absl::Span<T> output,
                                 int dst_i) {
  if (dst_i >= output.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains more elements than the underlying "
              "buffer ("
           << output.size() << "): " << data_str;
  }
  auto element_str = data_str.substr(token_start, token_end - token_start + 1);
  auto element = SimpleStrToValue<T>()(element_str);
  if (!element.has_value()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unable to parse element " << dst_i << " = '" << element_str
           << "'";
  }
  output[dst_i] = element.value();
  return OkStatus();
}

template <typename T>
Status ParseNumericalDataAsType(absl::string_view data_str,
                                absl::Span<uint8_t> output) {
  auto cast_output = ReinterpretSpan<T>(output);
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
    RETURN_IF_ERROR(ParseNumericalDataElement<T>(
        data_str, token_start, src_i - 2, cast_output, dst_i++));
    token_start = std::string::npos;
  }
  if (token_start != std::string::npos) {
    RETURN_IF_ERROR(ParseNumericalDataElement<T>(
        data_str, token_start, data_str.size() - 1, cast_output, dst_i++));
  }
  if (dst_i == 1 && cast_output.size() > 1) {
    // Splat the single value we got to the entire tensor.
    for (int i = 1; i < cast_output.size(); ++i) {
      cast_output[i] = cast_output[0];
    }
  } else if (dst_i < cast_output.size()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Input data string contains fewer elements than the underlying "
              "buffer (expected "
           << cast_output.size() << ") " << data_str;
  }
  return OkStatus();
}

// Parses numerical data (ints, floats, etc) in some typed form.
Status ParseNumericalData(absl::string_view type_str,
                          absl::string_view data_str,
                          absl::Span<uint8_t> output) {
  if (type_str == "i8") {
    return ParseNumericalDataAsType<int8_t>(data_str, output);
  } else if (type_str == "u8") {
    return ParseNumericalDataAsType<uint8_t>(data_str, output);
  } else if (type_str == "i16") {
    return ParseNumericalDataAsType<int16_t>(data_str, output);
  } else if (type_str == "u16") {
    return ParseNumericalDataAsType<uint16_t>(data_str, output);
  } else if (type_str == "i32") {
    return ParseNumericalDataAsType<int32_t>(data_str, output);
  } else if (type_str == "u32") {
    return ParseNumericalDataAsType<uint32_t>(data_str, output);
  } else if (type_str == "i64") {
    return ParseNumericalDataAsType<int64_t>(data_str, output);
  } else if (type_str == "u64") {
    return ParseNumericalDataAsType<uint64_t>(data_str, output);
  } else if (type_str == "f32") {
    return ParseNumericalDataAsType<float>(data_str, output);
  } else if (type_str == "f64") {
    return ParseNumericalDataAsType<double>(data_str, output);
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
Status PrintNumericalDataToStreamAsType(const Shape& shape,
                                        absl::Span<const uint8_t> contents,
                                        size_t max_entries,
                                        std::ostream* stream) {
  auto cast_contents = ReinterpretSpan<T>(contents);
  PrintElementList(shape, cast_contents, &max_entries, stream);
  return OkStatus();
}

}  // namespace

StatusOr<BufferDataPrintMode> ParseBufferDataPrintMode(absl::string_view str) {
  if (str.empty()) {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Cannot get print mode of empty string";
  }
  switch (str[0]) {
    case 'b':
      return BufferDataPrintMode::kBinary;
    case 'i':
      return BufferDataPrintMode::kSignedInteger;
    case 'u':
      return BufferDataPrintMode::kUnsignedInteger;
    case 'f':
      return BufferDataPrintMode::kFloatingPoint;
    default:
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Unsupported output type '" << str << "'";
  }
}

StatusOr<int> ParseBufferTypeElementSize(absl::string_view type_str) {
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

std::string MakeBufferTypeString(int element_size,
                                 BufferDataPrintMode print_mode) {
  std::string type_str;
  switch (print_mode) {
    case BufferDataPrintMode::kBinary:
      type_str = std::to_string(element_size);
      break;
    case BufferDataPrintMode::kSignedInteger:
      absl::StrAppend(&type_str, "i", element_size * 8);
      break;
    case BufferDataPrintMode::kUnsignedInteger:
      absl::StrAppend(&type_str, "u", element_size * 8);
      break;
    case BufferDataPrintMode::kFloatingPoint:
      absl::StrAppend(&type_str, "f", element_size * 8);
      break;
  }
  return type_str;
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

std::string PrintShapedTypeToString(const Shape& shape,
                                    absl::string_view type_str) {
  std::string result;
  PrintShapedTypeToString(shape, type_str, &result);
  return result;
}

void PrintShapedTypeToString(const Shape& shape, absl::string_view type_str,
                             std::string* out_result) {
  std::ostringstream stream;
  PrintShapedTypeToStream(shape, type_str, &stream);
  *out_result = stream.str();
}

void PrintShapedTypeToStream(const Shape& shape, absl::string_view type_str,
                             std::ostream* stream) {
  *stream << absl::StrJoin(shape.begin(), shape.end(), "x");
  if (!shape.empty()) *stream << "x";
  *stream << type_str;
}

// Prints binary hex data.
StatusOr<std::string> PrintBinaryDataToString(
    int element_size, absl::Span<const uint8_t> contents, size_t max_entries) {
  std::string result;
  RETURN_IF_ERROR(
      PrintBinaryDataToString(element_size, contents, max_entries, &result));
  return result;
}

Status PrintBinaryDataToString(int element_size,
                               absl::Span<const uint8_t> contents,
                               size_t max_entries, std::string* out_result) {
  std::ostringstream stream;
  RETURN_IF_ERROR(
      PrintBinaryDataToStream(element_size, contents, max_entries, &stream));
  *out_result = stream.str();
  return OkStatus();
}

Status PrintBinaryDataToStream(int element_size,
                               absl::Span<const uint8_t> contents,
                               size_t max_entries, std::ostream* stream) {
  // TODO(gcmn) Can we avoid this fiddly byte counting?
  max_entries *= element_size;  // Counting bytes, but treat them as elements.
  constexpr size_t hex_chars_per_byte = 2;
  constexpr size_t max_bytes = sizeof(int64_t);
  CHECK_LE(element_size, max_bytes);
  // Plus one char for the null terminator.
  char hex_buffer[hex_chars_per_byte * max_bytes + 1] = {0};
  for (size_t i = 0; i < std::min(max_entries, contents.size());
       i += element_size) {
    if (i > 0) *stream << " ";
    BytesToHexString(contents.data() + i, hex_buffer, element_size);
    *stream << hex_buffer;
  }
  if (contents.size() > max_entries) *stream << "...";
  return OkStatus();
}

StatusOr<std::string> PrintNumericalDataToString(
    const Shape& shape, absl::string_view type_str,
    absl::Span<const uint8_t> contents, size_t max_entries) {
  std::string result;
  RETURN_IF_ERROR(PrintNumericalDataToString(shape, type_str, contents,
                                             max_entries, &result));
  return result;
}

Status PrintNumericalDataToString(const Shape& shape,
                                  absl::string_view type_str,
                                  absl::Span<const uint8_t> contents,
                                  size_t max_entries, std::string* out_result) {
  std::ostringstream stream;
  RETURN_IF_ERROR(PrintNumericalDataToStream(shape, type_str, contents,
                                             max_entries, &stream));
  *out_result = stream.str();
  return OkStatus();
}

// Prints numerical data (ints, floats, etc) from some typed form.
Status PrintNumericalDataToStream(const Shape& shape,
                                  absl::string_view type_str,
                                  absl::Span<const uint8_t> contents,
                                  size_t max_entries, std::ostream* stream) {
  if (type_str == "i8") {
    return PrintNumericalDataToStreamAsType<int8_t>(shape, contents,
                                                    max_entries, stream);
  } else if (type_str == "u8") {
    return PrintNumericalDataToStreamAsType<uint8_t>(shape, contents,
                                                     max_entries, stream);
  } else if (type_str == "i16") {
    return PrintNumericalDataToStreamAsType<int16_t>(shape, contents,
                                                     max_entries, stream);
  } else if (type_str == "u16") {
    return PrintNumericalDataToStreamAsType<uint16_t>(shape, contents,
                                                      max_entries, stream);
  } else if (type_str == "i32") {
    return PrintNumericalDataToStreamAsType<int32_t>(shape, contents,
                                                     max_entries, stream);
  } else if (type_str == "u32") {
    return PrintNumericalDataToStreamAsType<uint32_t>(shape, contents,
                                                      max_entries, stream);
  } else if (type_str == "i64") {
    return PrintNumericalDataToStreamAsType<int64_t>(shape, contents,
                                                     max_entries, stream);
  } else if (type_str == "u64") {
    return PrintNumericalDataToStreamAsType<uint64_t>(shape, contents,
                                                      max_entries, stream);
  } else if (type_str == "f32") {
    return PrintNumericalDataToStreamAsType<float>(shape, contents, max_entries,
                                                   stream);
  } else if (type_str == "f64") {
    return PrintNumericalDataToStreamAsType<double>(shape, contents,
                                                    max_entries, stream);
  } else {
    return InvalidArgumentErrorBuilder(IREE_LOC)
           << "Unsupported type: " << type_str;
  }
}

Status ParseBufferDataAsType(absl::string_view data_str,
                             absl::string_view type_str,
                             absl::Span<uint8_t> output) {
  // Parse the data from the string right into the buffer.
  if (IsBinaryType(type_str)) {
    // Parse as binary hex.
    return ParseBinaryData(data_str, output);
  }
  // Parse as some nicely formatted type.
  return ParseNumericalData(type_str, data_str, output);
}

// static
BufferStringParts BufferStringParts::ExtractFrom(
    absl::string_view shaped_buf_str) {
  BufferStringParts parts;
  absl::string_view shape_and_type_str;
  auto equal_index = shaped_buf_str.find('=');
  if (equal_index == std::string::npos) {
    // Treat a lack of = as defaulting the data to zeros.
    shape_and_type_str = shaped_buf_str;
  } else {
    shape_and_type_str = shaped_buf_str.substr(0, equal_index);
    parts.data_str = shaped_buf_str.substr(equal_index + 1);
  }
  auto last_x_index = shape_and_type_str.rfind('x');
  if (last_x_index == std::string::npos) {
    // Scalar.
    parts.type_str = shape_and_type_str;
  } else {
    // Has a shape.
    parts.shape_str = shape_and_type_str.substr(0, last_x_index);
    parts.type_str = shape_and_type_str.substr(last_x_index + 1);
  }
  return parts;
}

}  // namespace iree
