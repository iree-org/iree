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

#include "iree/hal/api.h"

#include <cctype>
#include <cinttypes>
#include <cstdio>

#include "absl/container/inlined_vector.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/memory.h"
#include "iree/base/tracing.h"
#include "iree/hal/api_detail.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/device.h"
#include "iree/hal/driver.h"
#include "iree/hal/driver_registry.h"
#include "iree/hal/heap_buffer.h"
#include "iree/hal/host/host_local_allocator.h"
#include "iree/hal/semaphore.h"

namespace iree {
namespace hal {

// Defines the iree_hal_<type_name>_retain/_release methods.
#define IREE_HAL_API_RETAIN_RELEASE(type_name, cc_type)   \
  IREE_API_EXPORT void iree_hal_##type_name##_retain(     \
      iree_hal_##type_name##_t* type_name) {              \
    auto* handle = reinterpret_cast<cc_type*>(type_name); \
    if (handle) handle->AddReference();                   \
  }                                                       \
  IREE_API_EXPORT void iree_hal_##type_name##_release(    \
      iree_hal_##type_name##_t* type_name) {              \
    auto* handle = reinterpret_cast<cc_type*>(type_name); \
    if (handle) handle->ReleaseReference();               \
  }

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_shape(
    iree_string_view_t value, iree_host_size_t shape_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank) {
  IREE_ASSERT_ARGUMENT(out_shape_rank);
  *out_shape_rank = 0;

  auto str_value = absl::string_view(value.data, value.size);
  if (str_value.empty()) {
    return iree_ok_status();  // empty shape
  }

  absl::InlinedVector<iree_hal_dim_t, 6> dims;
  for (auto dim_str : absl::StrSplit(str_value, 'x')) {
    int dim_value = 0;
    if (!absl::SimpleAtoi(dim_str, &dim_value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "shape[%zu] invalid value '%.*s' of '%.*s'",
                              dims.size(), (int)dim_str.size(), dim_str.data(),
                              (int)value.size, value.data);
    }
    if (dim_value < 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "shape[%zu] unsupported value %d of '%.*s'",
                              dims.size(), dim_value, (int)value.size,
                              value.data);
    }
    dims.push_back(dim_value);
  }
  if (out_shape_rank) {
    *out_shape_rank = dims.size();
  }
  if (dims.size() > shape_capacity) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }
  if (out_shape) {
    std::memcpy(out_shape, dims.data(), dims.size() * sizeof(*out_shape));
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_format_shape(const iree_hal_dim_t* shape, iree_host_size_t shape_rank,
                      iree_host_size_t buffer_capacity, char* buffer,
                      iree_host_size_t* out_buffer_length) {
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  iree_host_size_t buffer_length = 0;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    int n = std::snprintf(buffer ? buffer + buffer_length : nullptr,
                          buffer ? buffer_capacity - buffer_length : 0,
                          (i < shape_rank - 1) ? "%dx" : "%d", shape[i]);
    if (n < 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "snprintf failed to write dimension %zu", i);
    } else if (buffer && n >= buffer_capacity - buffer_length) {
      buffer = nullptr;
    }
    buffer_length += n;
  }
  if (out_buffer_length) {
    *out_buffer_length = buffer_length;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_element_type(
    iree_string_view_t value, iree_hal_element_type_t* out_element_type) {
  IREE_ASSERT_ARGUMENT(out_element_type);
  *out_element_type = IREE_HAL_ELEMENT_TYPE_NONE;

  auto str_value = absl::string_view(value.data, value.size);

  iree_hal_numerical_type_t numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
  if (absl::StartsWith(str_value, "i")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "u")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "f")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE;
    str_value.remove_prefix(1);
  } else if (absl::StartsWith(str_value, "x") ||
             absl::StartsWith(str_value, "*")) {
    numerical_type = IREE_HAL_NUMERICAL_TYPE_UNKNOWN;
    str_value.remove_prefix(1);
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unhandled element type prefix in '%.*s'",
                            (int)value.size, value.data);
  }

  uint32_t bit_count = 0;
  if (!absl::SimpleAtoi(str_value, &bit_count) || bit_count > 0xFFu) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out of range bit count in '%.*s'", (int)value.size,
                            value.data);
  }

  *out_element_type = iree_hal_make_element_type(numerical_type, bit_count);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_element_type(
    iree_hal_element_type_t element_type, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  const char* prefix;
  switch (iree_hal_element_numerical_type(element_type)) {
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED:
      prefix = "i";
      break;
    case IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED:
      prefix = "u";
      break;
    case IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE:
      prefix = "f";
      break;
    default:
      prefix = "*";
      break;
  }
  int n = std::snprintf(
      buffer, buffer_capacity, "%s%d", prefix,
      static_cast<int32_t>(iree_hal_element_bit_count(element_type)));
  if (n < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "snprintf failed");
  }
  if (out_buffer_length) {
    *out_buffer_length = n;
  }
  return n >= buffer_capacity ? iree_status_from_code(IREE_STATUS_OUT_OF_RANGE)
                              : iree_ok_status();
}

// Parses a string of two character pairs representing hex numbers into bytes.
static void iree_hal_hex_string_to_bytes(const char* from, uint8_t* to,
                                         ptrdiff_t num) {
  /* clang-format off */
  static constexpr char kHexValue[256] = {
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
  for (int i = 0; i < num; i++) {
    to[i] = (kHexValue[from[i * 2] & 0xFF] << 4) +
            (kHexValue[from[i * 2 + 1] & 0xFF]);
  }
}

// Parses a signal element string, assuming that the caller has validated that
// |out_data| has enough storage space for the parsed element data.
static iree_status_t iree_hal_parse_element_unsafe(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    uint8_t* out_data) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_SINT_8: {
      int32_t temp = 0;
      if (!absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                            &temp) ||
          temp > INT8_MAX) {
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      }
      *reinterpret_cast<int8_t*>(out_data) = static_cast<int8_t>(temp);
      return iree_ok_status();
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_8: {
      uint32_t temp = 0;
      if (!absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                            &temp) ||
          temp > UINT8_MAX) {
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      }
      *reinterpret_cast<uint8_t*>(out_data) = static_cast<uint8_t>(temp);
      return iree_ok_status();
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_16: {
      int32_t temp = 0;
      if (!absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                            &temp) ||
          temp > INT16_MAX) {
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      }
      *reinterpret_cast<int16_t*>(out_data) = static_cast<int16_t>(temp);
      return iree_ok_status();
    }
    case IREE_HAL_ELEMENT_TYPE_UINT_16: {
      uint32_t temp = 0;
      if (!absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                            &temp) ||
          temp > UINT16_MAX) {
        return iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
      }
      *reinterpret_cast<uint16_t*>(out_data) = static_cast<uint16_t>(temp);
      return iree_ok_status();
    }
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      return absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<int32_t*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      return absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<uint32_t*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      return absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<int64_t*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      return absl::SimpleAtoi(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<uint64_t*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "float16 parsing not implemented");
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      return absl::SimpleAtof(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<float*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      return absl::SimpleAtod(absl::string_view(data_str.data, data_str.size),
                              reinterpret_cast<double*>(out_data))
                 ? iree_ok_status()
                 : iree_status_from_code(IREE_STATUS_INVALID_ARGUMENT);
    default: {
      // Treat any unknown format as binary.
      iree_host_size_t element_size = iree_hal_element_byte_count(element_type);
      if (data_str.size != element_size * 2) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "binary hex element count mismatch: buffer "
                                "length=%zu < expected=%zu",
                                data_str.size, element_size * 2);
      }
      iree_hal_hex_string_to_bytes(data_str.data, out_data, element_size);
      return iree_ok_status();
    }
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_element(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr) {
  iree_host_size_t element_size = iree_hal_element_byte_count(element_type);
  if (data_ptr.data_length < element_size) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "output data buffer overflow: data_length=%zu < element_size=%zu",
        data_ptr.data_length, element_size);
  }
  return iree_hal_parse_element_unsafe(data_str, element_type, data_ptr.data);
}

// Converts a sequence of bytes into hex number strings.
static void iree_hal_bytes_to_hex_string(const uint8_t* src, char* dest,
                                         ptrdiff_t num) {
  static constexpr char kHexTable[513] =
      "000102030405060708090A0B0C0D0E0F"
      "101112131415161718191A1B1C1D1E1F"
      "202122232425262728292A2B2C2D2E2F"
      "303132333435363738393A3B3C3D3E3F"
      "404142434445464748494A4B4C4D4E4F"
      "505152535455565758595A5B5C5D5E5F"
      "606162636465666768696A6B6C6D6E6F"
      "707172737475767778797A7B7C7D7E7F"
      "808182838485868788898A8B8C8D8E8F"
      "909192939495969798999A9B9C9D9E9F"
      "A0A1A2A3A4A5A6A7A8A9AAABACADAEAF"
      "B0B1B2B3B4B5B6B7B8B9BABBBCBDBEBF"
      "C0C1C2C3C4C5C6C7C8C9CACBCCCDCECF"
      "D0D1D2D3D4D5D6D7D8D9DADBDCDDDEDF"
      "E0E1E2E3E4E5E6E7E8E9EAEBECEDEEEF"
      "F0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF";
  for (auto src_ptr = src; src_ptr != (src + num); ++src_ptr, dest += 2) {
    const char* hex_p = &kHexTable[*src_ptr * 2];
    std::copy(hex_p, hex_p + 2, dest);
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_element(
    iree_const_byte_span_t data, iree_hal_element_type_t element_type,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length) {
  iree_host_size_t element_size = iree_hal_element_byte_count(element_type);
  if (data.data_length < element_size) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "data buffer underflow: data_length=%zu < element_size=%zu",
        data.data_length, element_size);
  }
  int n = 0;
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIi8,
                        *reinterpret_cast<const int8_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIu8,
                        *reinterpret_cast<const uint8_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIi16,
                        *reinterpret_cast<const int16_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIu16,
                        *reinterpret_cast<const uint16_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIi32,
                        *reinterpret_cast<const int32_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIu32,
                        *reinterpret_cast<const uint32_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIi64,
                        *reinterpret_cast<const int64_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%" PRIu64,
                        *reinterpret_cast<const uint64_t*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "parser for float16 not yet implemented");
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%G",
                        *reinterpret_cast<const float*>(data.data));
      break;
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      n = std::snprintf(buffer, buffer ? buffer_capacity : 0, "%G",
                        *reinterpret_cast<const double*>(data.data));
      break;
    default: {
      // Treat any unknown format as binary.
      n = 2 * element_size;
      if (buffer && buffer_capacity > n) {
        iree_hal_bytes_to_hex_string(data.data, buffer, element_size);
        buffer[n] = 0;
      }
    }
  }
  if (n < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "snprintf failed");
  } else if (buffer && n >= buffer_capacity) {
    buffer = nullptr;
  }
  if (out_buffer_length) {
    *out_buffer_length = n;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_parse_buffer_elements(
    iree_string_view_t data_str, iree_hal_element_type_t element_type,
    iree_byte_span_t data_ptr) {
  IREE_TRACE_SCOPE0("iree_hal_parse_buffer_elements");
  iree_host_size_t element_size = iree_hal_element_byte_count(element_type);
  iree_host_size_t element_capacity = data_ptr.data_length / element_size;
  if (iree_string_view_is_empty(data_str)) {
    memset(data_ptr.data, 0, data_ptr.data_length);
    return iree_ok_status();
  }
  size_t src_i = 0;
  size_t dst_i = 0;
  size_t token_start = std::string::npos;
  while (src_i < data_str.size) {
    char c = data_str.data[src_i++];
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
    if (dst_i >= element_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "output data buffer overflow: element_capacity=%zu < dst_i=%zu+",
          element_capacity, dst_i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_parse_element_unsafe(
        iree_string_view_t{data_str.data + token_start,
                           src_i - 2 - token_start + 1},
        element_type, data_ptr.data + dst_i * element_size));
    ++dst_i;
    token_start = std::string::npos;
  }
  if (token_start != std::string::npos) {
    if (dst_i >= element_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "output data overflow: element_capacity=%zu < dst_i=%zu",
          element_capacity, dst_i);
    }
    IREE_RETURN_IF_ERROR(iree_hal_parse_element_unsafe(
        iree_string_view_t{data_str.data + token_start,
                           data_str.size - token_start},
        element_type, data_ptr.data + dst_i * element_size));
    ++dst_i;
  }
  if (dst_i == 1 && element_capacity > 1) {
    // Splat the single value we got to the entire buffer.
    uint8_t* p = data_ptr.data + element_size;
    for (int i = 1; i < element_capacity; ++i, p += element_size) {
      memcpy(p, data_ptr.data, element_size);
    }
  } else if (dst_i < element_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "input data string underflow: dst_i=%zu < element_capacity=%zu", dst_i,
        element_capacity);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_format_buffer_elements_recursive(
    iree_const_byte_span_t data, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_host_size_t* max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  iree_host_size_t buffer_length = 0;
  auto append_char = [&](char c) {
    if (buffer) {
      if (buffer_length < buffer_capacity - 1) {
        buffer[buffer_length] = c;
        buffer[buffer_length + 1] = '\0';
      } else {
        buffer = nullptr;
      }
    }
    ++buffer_length;
  };

  if (shape_rank == 0) {
    // Scalar value; recurse to get on to the leaf dimension path.
    const iree_hal_dim_t one = 1;
    return iree_hal_format_buffer_elements_recursive(
        data, &one, 1, element_type, max_element_count, buffer_capacity, buffer,
        out_buffer_length);
  } else if (shape_rank > 1) {
    // Nested dimension; recurse into the next innermost dimension.
    iree_hal_dim_t dim_length = 1;
    for (iree_host_size_t i = 1; i < shape_rank; ++i) {
      dim_length *= shape[i];
    }
    iree_device_size_t dim_stride =
        dim_length * iree_hal_element_byte_count(element_type);
    if (data.data_length < dim_stride * shape[0]) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "input data underflow: data_length=%zu < expected=%zu",
          data.data_length,
          static_cast<iree_host_size_t>(dim_stride * shape[0]));
    }
    iree_const_byte_span_t subdata;
    subdata.data = data.data;
    subdata.data_length = dim_stride;
    for (iree_hal_dim_t i = 0; i < shape[0]; ++i) {
      append_char('[');
      iree_host_size_t actual_length = 0;
      iree_status_t status = iree_hal_format_buffer_elements_recursive(
          subdata, shape + 1, shape_rank - 1, element_type, max_element_count,
          buffer ? buffer_capacity - buffer_length : 0,
          buffer ? buffer + buffer_length : nullptr, &actual_length);
      buffer_length += actual_length;
      if (iree_status_is_out_of_range(status)) {
        buffer = nullptr;
      } else if (!iree_status_is_ok(status)) {
        return status;
      }
      subdata.data += dim_stride;
      append_char(']');
    }
  } else {
    // Leaf dimension; output data.
    iree_host_size_t max_count =
        std::min(*max_element_count, static_cast<iree_host_size_t>(shape[0]));
    iree_device_size_t element_stride =
        iree_hal_element_byte_count(element_type);
    if (data.data_length < max_count * element_stride) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "input data underflow; data_length=%zu < expected=%zu",
          data.data_length,
          static_cast<iree_host_size_t>(max_count * element_stride));
    }
    *max_element_count -= max_count;
    iree_const_byte_span_t subdata;
    subdata.data = data.data;
    subdata.data_length = element_stride;
    for (iree_hal_dim_t i = 0; i < max_count; ++i) {
      if (i > 0) append_char(' ');
      iree_host_size_t actual_length = 0;
      iree_status_t status = iree_hal_format_element(
          subdata, element_type, buffer ? buffer_capacity - buffer_length : 0,
          buffer ? buffer + buffer_length : nullptr, &actual_length);
      subdata.data += element_stride;
      buffer_length += actual_length;
      if (iree_status_is_out_of_range(status)) {
        buffer = nullptr;
      } else if (!iree_status_is_ok(status)) {
        return status;
      }
    }
    if (max_count < shape[0]) {
      append_char('.');
      append_char('.');
      append_char('.');
    }
  }
  if (out_buffer_length) {
    *out_buffer_length = buffer_length;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_format_buffer_elements(
    iree_const_byte_span_t data, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  IREE_TRACE_SCOPE0("iree_hal_format_buffer_elements");
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  if (buffer && buffer_capacity) {
    buffer[0] = '\0';
  }
  return iree_hal_format_buffer_elements_recursive(
      data, shape, shape_rank, element_type, &max_element_count,
      buffer_capacity, buffer, out_buffer_length);
}

//===----------------------------------------------------------------------===//
// iree::hal::Allocator
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(allocator, Allocator);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_allocator_create_host_local(iree_allocator_t allocator,
                                     iree_hal_allocator** out_allocator) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_create_host_local");
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator =
      reinterpret_cast<iree_hal_allocator_t*>(new host::HostLocalAllocator());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_size(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_device_size_t* out_allocation_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(out_allocation_size);
  *out_allocation_size = 0;

  // TODO(benvanik): layout/padding.
  iree_device_size_t byte_length = iree_hal_element_byte_count(element_type);
  for (int i = 0; i < shape_rank; ++i) {
    byte_length *= shape[i];
  }
  *out_allocation_size = byte_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_offset(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* indices, iree_host_size_t indices_count,
    iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(indices);
  IREE_ASSERT_ARGUMENT(out_offset);
  *out_offset = 0;
  if (shape_rank != indices_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %zu != %zu",
                            shape_rank, indices_count);
  }

  // TODO(benvanik): layout/padding.
  iree_device_size_t offset = 0;
  for (iree_host_size_t i = 0; i < indices_count; ++i) {
    if (indices[i] >= shape[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "index[%zu] out of bounds: %d >= %d", i,
                              indices[i], shape[i]);
    }
    iree_device_size_t axis_offset = indices[i];
    for (iree_host_size_t j = i + 1; j < shape_rank; ++j) {
      axis_offset *= shape[j];
    }
    offset += axis_offset;
  }
  offset *= iree_hal_element_byte_count(element_type);

  *out_offset = offset;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_compute_range(
    const iree_hal_allocator_t* allocator, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(shape);
  IREE_ASSERT_ARGUMENT(start_indices);
  IREE_ASSERT_ARGUMENT(lengths);
  IREE_ASSERT_ARGUMENT(out_start_offset);
  IREE_ASSERT_ARGUMENT(out_length);
  *out_start_offset = 0;
  *out_length = 0;
  if (indices_count != lengths_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "indices/lengths mismatch: %zu != %zu",
                            indices_count, lengths_count);
  }
  if (shape_rank != indices_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "shape rank/indices mismatch: %zu != %zu",
                            shape_rank, indices_count);
  }

  // TODO(benvanik): layout/padding.
  absl::InlinedVector<iree_hal_dim_t, 6> end_indices(shape_rank);
  iree_device_size_t element_size = iree_hal_element_byte_count(element_type);
  iree_device_size_t subspan_length = element_size;
  for (int i = 0; i < lengths_count; ++i) {
    subspan_length *= lengths[i];
    end_indices[i] = start_indices[i] + lengths[i] - 1;
  }

  iree_device_size_t start_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_offset(
      allocator, shape, shape_rank, element_type, start_indices, indices_count,
      &start_byte_offset));
  iree_device_size_t end_byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_offset(
      allocator, shape, shape_rank, element_type, end_indices.data(),
      end_indices.size(), &end_byte_offset));

  // Non-contiguous regions not yet implemented. Will be easier to detect when
  // we have strides.
  auto offset_length = end_byte_offset - start_byte_offset + element_size;
  if (subspan_length != offset_length) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "non-contiguous range region computation not implemented");
  }

  *out_start_offset = start_byte_offset;
  *out_length = subspan_length;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t buffer_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_allocate_buffer");
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;

  auto* handle = reinterpret_cast<Allocator*>(allocator);
  IREE_ASSIGN_OR_RETURN(
      auto buffer,
      handle->Allocate(static_cast<MemoryTypeBitfield>(memory_type),
                       static_cast<BufferUsageBitfield>(buffer_usage),
                       allocation_size));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t buffer_usage, iree_byte_span_t data,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_allocator_wrap_buffer");
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;

  auto* handle = reinterpret_cast<Allocator*>(allocator);
  IREE_ASSIGN_OR_RETURN(
      auto buffer,
      handle->WrapMutable(static_cast<MemoryTypeBitfield>(memory_type),
                          static_cast<MemoryAccessBitfield>(allowed_access),
                          static_cast<BufferUsageBitfield>(buffer_usage),
                          data.data, data.data_length));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(buffer.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::Buffer
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(buffer, Buffer);

IREE_API_EXPORT iree_status_t iree_hal_buffer_subspan(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_subspan");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;

  auto handle = add_ref(reinterpret_cast<Buffer*>(buffer));
  IREE_ASSIGN_OR_RETURN(auto new_handle,
                        Buffer::Subspan(handle, byte_offset, byte_length));

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(new_handle.release());

  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_buffer_allocator(const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  const auto* handle = reinterpret_cast<const Buffer*>(buffer);
  return reinterpret_cast<iree_hal_allocator_t*>(handle->allocator());
}

IREE_API_EXPORT iree_device_size_t
iree_hal_buffer_byte_length(const iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(buffer);
  const auto* handle = reinterpret_cast<const Buffer*>(buffer);
  return handle->byte_length();
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_zero(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_zero");
  IREE_ASSERT_ARGUMENT(buffer);
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  return handle->Fill8(byte_offset, byte_length, 0);
}

IREE_API_EXPORT iree_status_t
iree_hal_buffer_fill(iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
                     iree_device_size_t byte_length, const void* pattern,
                     iree_host_size_t pattern_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_fill");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(pattern);
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  return handle->Fill(byte_offset, byte_length, pattern, pattern_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_read_data(
    iree_hal_buffer_t* buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_read_data");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  return handle->ReadData(source_offset, target_buffer, data_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_write_data(
    iree_hal_buffer_t* buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_write_data");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  auto* handle = reinterpret_cast<Buffer*>(buffer);
  return handle->WriteData(target_offset, source_buffer, data_length);
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_map(
    iree_hal_buffer_t* buffer, iree_hal_memory_access_t memory_access,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_mapped_memory_t* out_mapped_memory) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_map");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_mapped_memory);

  std::memset(out_mapped_memory, 0, sizeof(*out_mapped_memory));

  auto* buffer_handle = reinterpret_cast<Buffer*>(buffer);
  IREE_ASSIGN_OR_RETURN(auto mapping,
                        buffer_handle->MapMemory<uint8_t>(
                            static_cast<MemoryAccessBitfield>(memory_access),
                            byte_offset, byte_length));

  static_assert(sizeof(iree_hal_mapped_memory_t::reserved) >=
                    sizeof(MappedMemory<uint8_t>),
                "C mapped memory struct must have large enough storage for the "
                "matching C++ struct");
  auto* mapping_storage =
      reinterpret_cast<MappedMemory<uint8_t>*>(out_mapped_memory->reserved);
  *mapping_storage = std::move(mapping);

  out_mapped_memory->contents = {mapping_storage->unsafe_data(),
                                 mapping_storage->size()};

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_buffer_unmap(
    iree_hal_buffer_t* buffer, iree_hal_mapped_memory_t* mapped_memory) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_map");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(mapped_memory);
  auto* mapping =
      reinterpret_cast<MappedMemory<uint8_t>*>(mapped_memory->reserved);
  mapping->reset();
  std::memset(mapped_memory, 0, sizeof(*mapped_memory));
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::HeapBuffer
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_allocate(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage,
    iree_host_size_t allocation_size, iree_allocator_t contents_allocator,
    iree_allocator_t allocator, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_allocate");
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = nullptr;

  if (!allocation_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation size must be >= 0");
  }

  auto handle = HeapBuffer::Allocate(
      static_cast<MemoryTypeBitfield>(memory_type),
      static_cast<BufferUsageBitfield>(usage), allocation_size);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(
      static_cast<Buffer*>(handle.release()));

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_allocate_copy(
    iree_hal_memory_type_t memory_type, iree_hal_buffer_usage_t usage,
    iree_hal_memory_access_t allowed_access, iree_byte_span_t contents,
    iree_allocator_t contents_allocator, iree_allocator_t allocator,
    iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_allocate_copy");
  IREE_ASSERT_ARGUMENT(out_buffer);

  *out_buffer = nullptr;

  if (!contents.data || !contents.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no contents specified (0 length)");
  }

  auto handle = HeapBuffer::AllocateCopy(
      static_cast<BufferUsageBitfield>(usage),
      static_cast<MemoryAccessBitfield>(allowed_access), contents.data,
      contents.data_length);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(handle.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_heap_buffer_wrap(
    iree_hal_memory_type_t memory_type, iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t usage, iree_byte_span_t contents,
    iree_allocator_t allocator, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_heap_buffer_wrap");
  IREE_ASSERT_ARGUMENT(out_buffer);

  *out_buffer = nullptr;

  if (!contents.data || !contents.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no contents specified (0 length)");
  }

  auto handle =
      HeapBuffer::WrapMutable(static_cast<MemoryTypeBitfield>(memory_type),
                              static_cast<MemoryAccessBitfield>(allowed_access),
                              static_cast<BufferUsageBitfield>(usage),
                              contents.data, contents.data_length);

  *out_buffer = reinterpret_cast<iree_hal_buffer_t*>(handle.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::BufferView
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(buffer_view, iree_hal_buffer_view);

IREE_API_EXPORT iree_status_t iree_hal_buffer_view_create(
    iree_hal_buffer_t* buffer, const iree_hal_dim_t* shape,
    iree_host_size_t shape_rank, iree_hal_element_type_t element_type,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_create");
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_buffer_view);

  *out_buffer_view = nullptr;
  if (shape_rank > 0 && !shape) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no shape dimensions specified");
  }

  // Allocate and initialize the iree_hal_buffer_view struct.
  // Note that we have the dynamically-sized shape dimensions on the end.
  iree_hal_buffer_view* buffer_view = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*buffer_view) + sizeof(iree_hal_dim_t) * shape_rank,
      reinterpret_cast<void**>(&buffer_view)));
  new (buffer_view) iree_hal_buffer_view();
  buffer_view->allocator = allocator;
  buffer_view->buffer = buffer;
  iree_hal_buffer_retain(buffer_view->buffer);
  buffer_view->element_type = element_type;
  buffer_view->byte_length =
      iree_hal_element_byte_count(buffer_view->element_type);
  buffer_view->shape_rank = shape_rank;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) {
    buffer_view->shape[i] = shape[i];
    buffer_view->byte_length *= shape[i];
  }

  *out_buffer_view = buffer_view;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_subview(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_ASSERT_ARGUMENT(out_buffer_view);

  // NOTE: we rely on the compute range call to do parameter validation.
  iree_device_size_t start_offset = 0;
  iree_device_size_t subview_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
      buffer_view, start_indices, indices_count, lengths, lengths_count,
      &start_offset, &subview_length));

  iree_hal_buffer_t* subview_buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_subspan(buffer_view->buffer,
                                               start_offset, subview_length,
                                               allocator, &subview_buffer));

  iree_status_t result = iree_hal_buffer_view_create(
      subview_buffer, lengths, lengths_count, buffer_view->element_type,
      allocator, out_buffer_view);
  iree_hal_buffer_release(subview_buffer);
  return result;
}

IREE_API_EXPORT iree_hal_buffer_t* iree_hal_buffer_view_buffer(
    const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->buffer;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL
iree_hal_buffer_view_shape_rank(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->shape_rank;
}

IREE_API_EXPORT iree_host_size_t IREE_API_CALL iree_hal_buffer_view_shape_dim(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t index) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  if (index > buffer_view->shape_rank) {
    return 0;
  }
  return buffer_view->shape[index];
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_count(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  iree_host_size_t element_count = 1;
  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    element_count *= buffer_view->shape[i];
  }
  return element_count;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_shape(
    const iree_hal_buffer_view_t* buffer_view, iree_host_size_t rank_capacity,
    iree_hal_dim_t* out_shape, iree_host_size_t* out_shape_rank) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  IREE_ASSERT_ARGUMENT(out_shape);
  if (out_shape_rank) {
    *out_shape_rank = 0;
  }

  if (out_shape_rank) {
    *out_shape_rank = buffer_view->shape_rank;
  }
  if (rank_capacity < buffer_view->shape_rank) {
    // Not an error; just a size query.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  for (iree_host_size_t i = 0; i < buffer_view->shape_rank; ++i) {
    out_shape[i] = buffer_view->shape[i];
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_hal_element_type_t IREE_API_CALL
iree_hal_buffer_view_element_type(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->element_type;
}

IREE_API_EXPORT iree_host_size_t
iree_hal_buffer_view_element_size(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_element_byte_count(buffer_view->element_type);
}

IREE_API_EXPORT iree_device_size_t IREE_API_CALL
iree_hal_buffer_view_byte_length(const iree_hal_buffer_view_t* buffer_view) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return buffer_view->byte_length;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_offset(
    const iree_hal_buffer_view_t* buffer_view, const iree_hal_dim_t* indices,
    iree_host_size_t indices_count, iree_device_size_t* out_offset) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_allocator_compute_offset(
      iree_hal_buffer_allocator(buffer_view->buffer), buffer_view->shape,
      buffer_view->shape_rank, buffer_view->element_type, indices,
      indices_count, out_offset);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_compute_range(
    const iree_hal_buffer_view_t* buffer_view,
    const iree_hal_dim_t* start_indices, iree_host_size_t indices_count,
    const iree_hal_dim_t* lengths, iree_host_size_t lengths_count,
    iree_device_size_t* out_start_offset, iree_device_size_t* out_length) {
  IREE_ASSERT_ARGUMENT(buffer_view);
  return iree_hal_allocator_compute_range(
      iree_hal_buffer_allocator(buffer_view->buffer), buffer_view->shape,
      buffer_view->shape_rank, buffer_view->element_type, start_indices,
      indices_count, lengths, lengths_count, out_start_offset, out_length);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_parse(
    iree_string_view_t value, iree_hal_allocator_t* buffer_allocator,
    iree_allocator_t allocator, iree_hal_buffer_view_t** out_buffer_view) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_parse");
  IREE_ASSERT_ARGUMENT(buffer_allocator);

  // Strip whitespace that may come along (linefeeds/etc).
  auto string_view =
      absl::StripAsciiWhitespace(absl::string_view(value.data, value.size));
  string_view = absl::StripPrefix(string_view, "\"");
  string_view = absl::StripSuffix(string_view, "\"");
  if (string_view.empty()) {
    // Empty lines are invalid; need at least the shape/type information.
    *out_buffer_view = nullptr;
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty string input");
  }

  // The part of the string corresponding to the shape, e.g. 1x2x3.
  absl::string_view shape_str;
  // The part of the string corresponding to the type, e.g. f32
  absl::string_view type_str;
  // The part of the string corresponding to the buffer data, e.g. 1 2 3 4 5 6
  absl::string_view data_str;

  absl::string_view shape_and_type_str;
  auto equal_index = string_view.find('=');
  if (equal_index == std::string::npos) {
    // Treat a lack of = as defaulting the data to zeros.
    shape_and_type_str = string_view;
  } else {
    shape_and_type_str = string_view.substr(0, equal_index);
    data_str = string_view.substr(equal_index + 1);
  }
  auto last_x_index = shape_and_type_str.rfind('x');
  if (last_x_index == std::string::npos) {
    // Scalar.
    type_str = shape_and_type_str;
  } else {
    // Has a shape.
    shape_str = shape_and_type_str.substr(0, last_x_index);
    type_str = shape_and_type_str.substr(last_x_index + 1);
  }

  // AxBxC...
  absl::InlinedVector<iree_hal_dim_t, 6> shape(6);
  iree_host_size_t shape_rank = 0;
  iree_status_t shape_result =
      iree_hal_parse_shape({shape_str.data(), shape_str.length()}, shape.size(),
                           shape.data(), &shape_rank);
  if (iree_status_is_ok(shape_result)) {
    shape.resize(shape_rank);
  } else if (iree_status_is_out_of_range(shape_result)) {
    shape.resize(shape_rank);
    IREE_RETURN_IF_ERROR(
        iree_hal_parse_shape({shape_str.data(), shape_str.length()},
                             shape.size(), shape.data(), &shape_rank));
  } else {
    return shape_result;
  }

  // f32, i32, etc
  iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_parse_element_type(
      {type_str.data(), type_str.length()}, &element_type));

  // Allocate the buffer we will parse into from the provided allocator.
  iree_device_size_t buffer_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_compute_size(
      buffer_allocator, shape.data(), shape.size(), element_type,
      &buffer_length));
  iree_hal_buffer_t* buffer = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      buffer_allocator,
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING,
      buffer_length, &buffer));

  iree_status_t status;

  // Parse the elements directly into the buffer.
  iree_hal_mapped_memory_t mapped_buffer;
  status = iree_hal_buffer_map(buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, 0,
                               buffer_length, &mapped_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }
  status = iree_hal_parse_buffer_elements({data_str.data(), data_str.length()},
                                          element_type, mapped_buffer.contents);
  iree_hal_buffer_unmap(buffer, &mapped_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Wrap and pass ownership of the buffer to the buffer view.
  status =
      iree_hal_buffer_view_create(buffer, shape.data(), shape.size(),
                                  element_type, allocator, out_buffer_view);
  iree_hal_buffer_release(buffer);
  return status;
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_buffer_view_format(
    const iree_hal_buffer_view_t* buffer_view,
    iree_host_size_t max_element_count, iree_host_size_t buffer_capacity,
    char* buffer, iree_host_size_t* out_buffer_length) {
  IREE_TRACE_SCOPE0("iree_hal_buffer_view_format");
  IREE_ASSERT_ARGUMENT(buffer_view);
  if (out_buffer_length) {
    *out_buffer_length = 0;
  }
  if (buffer && buffer_capacity) {
    buffer[0] = 0;
  }

  iree_status_t status;
  iree_host_size_t buffer_length = 0;
  auto append_char = [&](char c) {
    if (buffer) {
      if (buffer_length < buffer_capacity - 1) {
        buffer[buffer_length] = c;
        buffer[buffer_length + 1] = '\0';
      } else {
        buffer = nullptr;
      }
    }
    ++buffer_length;
  };

  if (buffer_view->shape_rank > 0) {
    // Shape: 1x2x3
    iree_host_size_t shape_length = 0;
    status = iree_hal_format_shape(buffer_view->shape, buffer_view->shape_rank,
                                   buffer ? buffer_capacity - buffer_length : 0,
                                   buffer ? buffer + buffer_length : nullptr,
                                   &shape_length);
    buffer_length += shape_length;
    if (iree_status_is_out_of_range(status)) {
      status = iree_status_ignore(status);
      buffer = nullptr;
    } else if (!iree_status_is_ok(status)) {
      return status;
    }

    // Separator: <shape>x<format>
    append_char('x');
  }

  // Element type: f32
  iree_host_size_t element_type_length = 0;
  status = iree_hal_format_element_type(
      buffer_view->element_type, buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : nullptr, &element_type_length);
  buffer_length += element_type_length;
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = nullptr;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  // Separator: <meta>=<value>
  append_char('=');

  // Buffer contents: 0 1 2 3 ...
  iree_hal_mapped_memory_t mapped_buffer;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map(buffer_view->buffer,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           IREE_WHOLE_BUFFER, &mapped_buffer));
  iree_host_size_t elements_length = 0;
  status = iree_hal_format_buffer_elements(
      iree_const_byte_span_t{mapped_buffer.contents.data,
                             mapped_buffer.contents.data_length},
      buffer_view->shape, buffer_view->shape_rank, buffer_view->element_type,
      max_element_count, buffer ? buffer_capacity - buffer_length : 0,
      buffer ? buffer + buffer_length : nullptr, &elements_length);
  buffer_length += elements_length;
  iree_hal_buffer_unmap(buffer_view->buffer, &mapped_buffer);
  if (iree_status_is_out_of_range(status)) {
    status = iree_status_ignore(status);
    buffer = nullptr;
  } else if (!iree_status_is_ok(status)) {
    return status;
  }

  if (out_buffer_length) {
    *out_buffer_length = buffer_length;
  }
  return buffer ? iree_ok_status()
                : iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
}

//===----------------------------------------------------------------------===//
// iree::hal::CommandBuffer
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(command_buffer, CommandBuffer);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, iree_allocator_t allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = nullptr;
  auto* handle = reinterpret_cast<Device*>(device);

  IREE_ASSIGN_OR_RETURN(
      auto command_buffer,
      handle->CreateCommandBuffer(
          static_cast<CommandBufferModeBitfield>(mode),
          static_cast<CommandCategoryBitfield>(command_categories)));

  *out_command_buffer =
      reinterpret_cast<iree_hal_command_buffer_t*>(command_buffer.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_begin");
  IREE_ASSERT_ARGUMENT(command_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->Begin();
}

IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_end");
  IREE_ASSERT_ARGUMENT(command_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->End();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_execution_barrier");
  IREE_ASSERT_ARGUMENT(command_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(MemoryBarrier) == sizeof(iree_hal_memory_barrier_t),
                "Expecting identical layout");
  static_assert(sizeof(BufferBarrier) == sizeof(iree_hal_buffer_barrier_t),
                "Expecting identical layout");
  return handle->ExecutionBarrier(
      static_cast<ExecutionStageBitfield>(source_stage_mask),
      static_cast<ExecutionStageBitfield>(target_stage_mask),
      absl::MakeConstSpan(
          reinterpret_cast<const MemoryBarrier*>(memory_barriers),
          memory_barrier_count),
      absl::MakeConstSpan(
          reinterpret_cast<const BufferBarrier*>(buffer_barriers),
          buffer_barrier_count));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_fill_buffer");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->FillBuffer(reinterpret_cast<Buffer*>(target_buffer),
                            target_offset, length, pattern, pattern_length);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_update_buffer(iree_hal_command_buffer_t* command_buffer,
                                      const void* source_buffer,
                                      iree_host_size_t source_offset,
                                      iree_hal_buffer_t* target_buffer,
                                      iree_device_size_t target_offset,
                                      iree_device_size_t length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_update_buffer");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->UpdateBuffer(source_buffer, source_offset,
                              reinterpret_cast<Buffer*>(target_buffer),
                              target_offset, length);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_copy_buffer");
  IREE_ASSERT_ARGUMENT(command_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->CopyBuffer(
      reinterpret_cast<Buffer*>(source_buffer), source_offset,
      reinterpret_cast<Buffer*>(target_buffer), target_offset, length);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_constants(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_push_constants");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(values);
  if (values_length == 0) {
    return iree_ok_status();
  }
  if ((values_length % 4) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid alignment %zu, must be 4-byte aligned",
                            values_length);
  }
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->PushConstants(
      reinterpret_cast<ExecutableLayout*>(executable_layout), offset,
      absl::MakeConstSpan(reinterpret_cast<const uint32_t*>(values),
                          values_length / sizeof(uint32_t)));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, int32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_push_descriptor_set");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (binding_count && !bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bindings/binding_count mismatch");
  }

  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(DescriptorSet::Binding) ==
                    sizeof(iree_hal_descriptor_set_binding_t),
                "Expecting identical layout");
  return handle->PushDescriptorSet(
      reinterpret_cast<ExecutableLayout*>(executable_layout), set,
      absl::MakeConstSpan(
          reinterpret_cast<const DescriptorSet::Binding*>(bindings),
          binding_count));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, int32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_bind_descriptor_set");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(descriptor_set);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  if (dynamic_offset_count && !dynamic_offsets) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dynamic_offset_count/dynamic_offsets mismatch");
  }
  static_assert(sizeof(iree_device_size_t) == sizeof(device_size_t),
                "Device sizes must match");
  return handle->BindDescriptorSet(
      reinterpret_cast<ExecutableLayout*>(executable_layout), set,
      reinterpret_cast<DescriptorSet*>(descriptor_set),
      absl::MakeConstSpan(dynamic_offsets, dynamic_offset_count));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_dispatch");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->Dispatch(reinterpret_cast<Executable*>(executable),
                          entry_point, {workgroup_x, workgroup_y, workgroup_z});
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_TRACE_SCOPE0("iree_hal_command_buffer_dispatch_indirect");
  IREE_ASSERT_ARGUMENT(command_buffer);
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(workgroups_buffer);
  auto* handle = reinterpret_cast<CommandBuffer*>(command_buffer);
  return handle->DispatchIndirect(
      reinterpret_cast<Executable*>(executable), entry_point,
      reinterpret_cast<Buffer*>(workgroups_buffer), workgroups_offset);
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSet
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(descriptor_set, DescriptorSet);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_descriptor_set_create(
    iree_hal_device_t* device, iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_allocator_t allocator,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  IREE_TRACE_SCOPE0("iree_hal_descriptor_set_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(set_layout);
  IREE_ASSERT_ARGUMENT(out_descriptor_set);
  *out_descriptor_set = nullptr;
  if (binding_count && !bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding_count/bindings mismatch");
  }
  auto* handle = reinterpret_cast<Device*>(device);

  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(DescriptorSet::Binding) ==
                    sizeof(iree_hal_descriptor_set_binding_t),
                "Expecting identical layout");
  IREE_ASSIGN_OR_RETURN(
      auto descriptor_set,
      handle->CreateDescriptorSet(
          reinterpret_cast<DescriptorSetLayout*>(set_layout),
          absl::MakeConstSpan(
              reinterpret_cast<const DescriptorSet::Binding*>(bindings),
              binding_count)));

  *out_descriptor_set =
      reinterpret_cast<iree_hal_descriptor_set_t*>(descriptor_set.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::DescriptorSetLayout
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(descriptor_set_layout, DescriptorSetLayout);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_descriptor_set_layout_create(
    iree_hal_device_t* device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  IREE_TRACE_SCOPE0("iree_hal_descriptor_set_layout_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_descriptor_set_layout);
  *out_descriptor_set_layout = nullptr;
  if (binding_count && !bindings) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "binding_count/bindings mismatch");
  }
  auto* handle = reinterpret_cast<Device*>(device);

  // TODO(benvanik): refactor the C++ types to use the C types for storage so
  // that we can safely map between the two. For now assume size equality
  // is layout equality (as compilers aren't allowed to reorder structs).
  static_assert(sizeof(DescriptorSetLayout::Binding) ==
                    sizeof(iree_hal_descriptor_set_layout_binding_t),
                "Expecting identical layout");
  IREE_ASSIGN_OR_RETURN(
      auto descriptor_set_layout,
      handle->CreateDescriptorSetLayout(
          static_cast<DescriptorSetLayout::UsageType>(usage_type),
          absl::MakeConstSpan(
              reinterpret_cast<const DescriptorSetLayout::Binding*>(bindings),
              binding_count)));

  *out_descriptor_set_layout =
      reinterpret_cast<iree_hal_descriptor_set_layout_t*>(
          descriptor_set_layout.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::Device
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(device, Device);

IREE_API_EXPORT iree_hal_allocator_t* IREE_API_CALL
iree_hal_device_allocator(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  auto* handle = reinterpret_cast<Device*>(device);
  return reinterpret_cast<iree_hal_allocator_t*>(handle->allocator());
}

IREE_API_EXPORT iree_string_view_t IREE_API_CALL
iree_hal_device_id(iree_hal_device_t* device) {
  IREE_ASSERT_ARGUMENT(device);
  auto* handle = reinterpret_cast<Device*>(device);
  const auto& id = handle->info().id();
  return iree_string_view_t{id.data(), id.size()};
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_device_queue_submit(
    iree_hal_device_t* device, iree_hal_command_category_t command_categories,
    uint64_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  IREE_TRACE_SCOPE0("iree_hal_device_queue_submit");
  IREE_ASSERT_ARGUMENT(device);
  auto* handle = reinterpret_cast<Device*>(device);
  if (batch_count > 0 && !batches) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "batch_count/batches mismatch");
  }

  // We need to allocate storage to marshal in the semaphores. Ideally we'd
  // change the C++ API to make this 1:1 with a reinterpret_cast, however that
  // makes the C API more difficult. Bleh.
  int total_semaphore_count = 0;
  for (int i = 0; i < batch_count; ++i) {
    total_semaphore_count += batches[i].wait_semaphores.count;
    total_semaphore_count += batches[i].signal_semaphores.count;
  }
  absl::InlinedVector<SemaphoreValue, 4> semaphore_values(
      total_semaphore_count);
  absl::InlinedVector<SubmissionBatch, 2> dst_batches(batch_count);
  int base_semaphore_index = 0;
  for (int i = 0; i < batch_count; ++i) {
    const auto& src_batch = batches[i];
    auto& dst_batch = dst_batches[i];
    for (int j = 0; j < src_batch.wait_semaphores.count; ++j) {
      semaphore_values[base_semaphore_index + j] = {
          reinterpret_cast<Semaphore*>(src_batch.wait_semaphores.semaphores[j]),
          src_batch.wait_semaphores.payload_values[j]};
    }
    dst_batch.wait_semaphores =
        absl::MakeConstSpan(&semaphore_values[base_semaphore_index],
                            src_batch.wait_semaphores.count);
    base_semaphore_index += src_batch.wait_semaphores.count;
    dst_batch.command_buffers =
        iree::ReinterpretSpan<CommandBuffer*>(absl::MakeConstSpan(
            src_batch.command_buffers, src_batch.command_buffer_count));
    for (int j = 0; j < src_batch.signal_semaphores.count; ++j) {
      semaphore_values[base_semaphore_index + j] = {
          reinterpret_cast<Semaphore*>(
              src_batch.signal_semaphores.semaphores[j]),
          src_batch.signal_semaphores.payload_values[j]};
    }
    dst_batch.signal_semaphores =
        absl::MakeConstSpan(&semaphore_values[base_semaphore_index],
                            src_batch.signal_semaphores.count);
    base_semaphore_index += src_batch.signal_semaphores.count;
  }

  // For now we always go to the first compute queue. TBD cleanup pending the
  // device modeling in the IR as to how we really want to handle this. We'll
  // want to use queue_affinity in a way that ensures we have some control over
  // things on the compiler side and may require that devices are declared by
  // the number and types of queues they support.
  uint64_t queue_index = queue_affinity % handle->dispatch_queues().size();
  auto* command_queue = handle->dispatch_queues()[queue_index];
  return command_queue->Submit(dst_batches);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_deadline(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_time_t deadline_ns) {
  IREE_TRACE_SCOPE0("iree_hal_device_wait_semaphores_with_deadline");
  IREE_ASSERT_ARGUMENT(device);
  if (!semaphore_list || semaphore_list->count == 0) return iree_ok_status();
  auto* handle = reinterpret_cast<Device*>(device);

  absl::InlinedVector<SemaphoreValue, 4> semaphore_values(
      semaphore_list->count);
  for (int i = 0; i < semaphore_list->count; ++i) {
    semaphore_values[i] = {
        reinterpret_cast<Semaphore*>(semaphore_list->semaphores[i]),
        semaphore_list->payload_values[i]};
  }

  switch (wait_mode) {
    case IREE_HAL_WAIT_MODE_ALL: {
      return handle->WaitAllSemaphores(semaphore_values, Time(deadline_ns));
    }
    case IREE_HAL_WAIT_MODE_ANY: {
      IREE_ASSIGN_OR_RETURN(
          int wake_index,
          handle->WaitAnySemaphore(semaphore_values, Time(deadline_ns)));
      (void)wake_index;
      return iree_ok_status();
    }
    default: {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unhandled wait_mode");
    }
  }
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_device_wait_semaphores_with_timeout(
    iree_hal_device_t* device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list,
    iree_duration_t timeout_ns) {
  iree_time_t deadline_ns = iree_relative_timeout_to_deadline_ns(timeout_ns);
  return iree_hal_device_wait_semaphores_with_deadline(
      device, wait_mode, semaphore_list, deadline_ns);
}

//===----------------------------------------------------------------------===//
// iree::hal::Driver
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(driver, Driver);

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_query_available_devices(
    iree_hal_driver_t* driver, iree_allocator_t allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  IREE_TRACE_SCOPE0("iree_hal_driver_query_available_devices");
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device_infos);
  IREE_ASSERT_ARGUMENT(out_device_info_count);
  *out_device_info_count = 0;
  auto* handle = reinterpret_cast<Driver*>(driver);

  IREE_ASSIGN_OR_RETURN(auto device_infos, handle->EnumerateAvailableDevices());
  size_t total_string_size = 0;
  for (const auto& device_info : device_infos) {
    total_string_size += device_info.name().size();
  }

  *out_device_info_count = device_infos.size();
  iree_hal_device_info_t* device_info_storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator,
      device_infos.size() * sizeof(*device_info_storage) + total_string_size,
      (void**)&device_info_storage));

  char* p = reinterpret_cast<char*>(device_info_storage) +
            device_infos.size() * sizeof(*device_info_storage);
  for (int i = 0; i < device_infos.size(); ++i) {
    const auto& device_info = device_infos[i];
    device_info_storage[i].device_id = device_info.device_id();

    size_t name_size = device_info.name().size();
    std::memcpy(p, device_info.name().c_str(), name_size);
    device_info_storage[i].name = iree_string_view_t{p, name_size};
    p += name_size;
  }

  *out_device_infos = device_info_storage;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_driver_create_device(
    iree_hal_driver_t* driver, iree_hal_device_id_t device_id,
    iree_allocator_t allocator, iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_driver_create_device");
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  auto* handle = reinterpret_cast<Driver*>(driver);

  IREE_ASSIGN_OR_RETURN(auto device, handle->CreateDevice(device_id));

  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_create_default_device(iree_hal_driver_t* driver,
                                      iree_allocator_t allocator,
                                      iree_hal_device_t** out_device) {
  IREE_TRACE_SCOPE0("iree_hal_driver_create_default_device");
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;
  auto* handle = reinterpret_cast<Driver*>(driver);
  IREE_ASSIGN_OR_RETURN(auto device, handle->CreateDefaultDevice());
  *out_device = reinterpret_cast<iree_hal_device_t*>(device.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::DriverRegistry
//===----------------------------------------------------------------------===//

IREE_API_EXPORT bool IREE_API_CALL
iree_hal_driver_registry_has_driver(iree_string_view_t driver_name) {
  return DriverRegistry::shared_registry()->HasDriver(
      absl::string_view{driver_name.data, driver_name.size});
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_query_available_drivers(
    iree_allocator_t allocator, iree_string_view_t** out_driver_names,
    iree_host_size_t* out_driver_count) {
  IREE_TRACE_SCOPE0("iree_hal_driver_registry_query_available_drivers");
  IREE_ASSERT_ARGUMENT(out_driver_names);
  IREE_ASSERT_ARGUMENT(out_driver_count);
  *out_driver_count = 0;

  auto* registry = DriverRegistry::shared_registry();
  auto available_drivers = registry->EnumerateAvailableDrivers();
  size_t total_string_size = 0;
  for (const auto& driver_name : available_drivers) {
    total_string_size += driver_name.size();
  }

  *out_driver_count = available_drivers.size();
  iree_string_view_t* driver_name_storage = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator,
      available_drivers.size() * sizeof(*driver_name_storage) +
          total_string_size,
      (void**)&driver_name_storage));

  char* p = reinterpret_cast<char*>(driver_name_storage) +
            available_drivers.size() * sizeof(*driver_name_storage);
  for (int i = 0; i < available_drivers.size(); ++i) {
    const auto& driver_name = available_drivers[i];
    size_t name_size = driver_name.size();
    std::memcpy(p, driver_name.c_str(), name_size);
    driver_name_storage[i] = iree_string_view_t{p, name_size};
    p += name_size;
  }

  *out_driver_names = driver_name_storage;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_driver_registry_create_driver(iree_string_view_t driver_name,
                                       iree_allocator_t allocator,
                                       iree_hal_driver_t** out_driver) {
  IREE_TRACE_SCOPE0("iree_hal_driver_registry_create_driver");
  IREE_ASSERT_ARGUMENT(out_driver);
  *out_driver = nullptr;

  auto* registry = DriverRegistry::shared_registry();
  IREE_ASSIGN_OR_RETURN(auto driver, registry->Create(absl::string_view(
                                         driver_name.data, driver_name.size)));

  *out_driver = reinterpret_cast<iree_hal_driver_t*>(driver.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::Executable
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(executable, Executable);

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableCache
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(executable_cache, ExecutableCache);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_cache_create(
    iree_hal_device_t* device, iree_string_view_t identifier,
    iree_allocator_t allocator,
    iree_hal_executable_cache_t** out_executable_cache) {
  IREE_TRACE_SCOPE0("iree_hal_executable_cache_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = nullptr;

  auto* handle = reinterpret_cast<Device*>(device);
  auto executable_cache = handle->CreateExecutableCache();
  *out_executable_cache = reinterpret_cast<iree_hal_executable_cache_t*>(
      executable_cache.release());
  return iree_ok_status();
}

IREE_API_EXPORT bool IREE_API_CALL iree_hal_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_format_t format) {
  IREE_ASSERT_ARGUMENT(executable_cache);
  auto* handle = reinterpret_cast<ExecutableCache*>(executable_cache);
  return handle->CanPrepareFormat(format);
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_executable_cache_prepare_executable(
    iree_hal_executable_cache_t* executable_cache,
    iree_hal_executable_layout_t* executable_layout,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data, iree_allocator_t allocator,
    iree_hal_executable_t** out_executable) {
  IREE_TRACE_SCOPE0("iree_hal_executable_cache_prepare_executable");
  IREE_ASSERT_ARGUMENT(executable_cache);
  IREE_ASSERT_ARGUMENT(executable_layout);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = nullptr;
  auto* handle = reinterpret_cast<ExecutableCache*>(executable_cache);

  ExecutableSpec spec;
  spec.executable_data = {executable_data.data, executable_data.data_length};
  IREE_ASSIGN_OR_RETURN(
      auto executable,
      handle->PrepareExecutable(
          reinterpret_cast<ExecutableLayout*>(executable_layout),
          static_cast<ExecutableCachingMode>(caching_mode), spec));

  *out_executable =
      reinterpret_cast<iree_hal_executable_t*>(executable.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::ExecutableLayout
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(executable_layout, ExecutableLayout);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_executable_layout_create(
    iree_hal_device_t* device, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_host_size_t push_constants, iree_allocator_t allocator,
    iree_hal_executable_layout_t** out_executable_layout) {
  IREE_TRACE_SCOPE0("iree_hal_executable_layout_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_executable_layout);
  *out_executable_layout = nullptr;
  if (set_layout_count && !set_layouts) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set_layout_count/set_layouts mismatch");
  }

  auto* handle = reinterpret_cast<Device*>(device);
  IREE_ASSIGN_OR_RETURN(
      auto executable_layout,
      handle->CreateExecutableLayout(
          absl::MakeConstSpan(
              reinterpret_cast<DescriptorSetLayout* const*>(set_layouts),
              set_layout_count),
          push_constants));

  *out_executable_layout = reinterpret_cast<iree_hal_executable_layout_t*>(
      executable_layout.release());
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree::hal::Semaphore
//===----------------------------------------------------------------------===//

IREE_HAL_API_RETAIN_RELEASE(semaphore, Semaphore);

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_semaphore_create(
    iree_hal_device_t* device, uint64_t initial_value,
    iree_allocator_t allocator, iree_hal_semaphore_t** out_semaphore) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_create");
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = nullptr;

  auto* handle = reinterpret_cast<Device*>(device);
  IREE_ASSIGN_OR_RETURN(auto semaphore, handle->CreateSemaphore(initial_value));

  *out_semaphore = reinterpret_cast<iree_hal_semaphore_t*>(semaphore.release());
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_query(iree_hal_semaphore_t* semaphore, uint64_t* out_value) {
  IREE_ASSERT_ARGUMENT(semaphore);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;

  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  IREE_ASSIGN_OR_RETURN(uint64_t value, handle->Query());
  *out_value = value;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_signal(iree_hal_semaphore_t* semaphore, uint64_t new_value) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_signal");
  IREE_ASSERT_ARGUMENT(semaphore);
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  return handle->Signal(new_value);
}

IREE_API_EXPORT void IREE_API_CALL
iree_hal_semaphore_fail(iree_hal_semaphore_t* semaphore, iree_status_t status) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_fail");
  IREE_ASSERT_ARGUMENT(semaphore);
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  handle->Fail(std::move(status));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_wait_with_deadline(iree_hal_semaphore_t* semaphore,
                                      uint64_t value, iree_time_t deadline_ns) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_wait_with_deadline");
  IREE_ASSERT_ARGUMENT(semaphore);
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  return handle->Wait(value, Time(deadline_ns));
}

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_semaphore_wait_with_timeout(iree_hal_semaphore_t* semaphore,
                                     uint64_t value,
                                     iree_duration_t timeout_ns) {
  IREE_TRACE_SCOPE0("iree_hal_semaphore_wait_with_timeout");
  IREE_ASSERT_ARGUMENT(semaphore);
  auto* handle = reinterpret_cast<Semaphore*>(semaphore);
  return handle->Wait(value, Duration(timeout_ns));
}

}  // namespace hal
}  // namespace iree
