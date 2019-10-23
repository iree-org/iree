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

// Utility to enable bit operators on enum classes treated as bitfields.
//
// To use define an enum class with valid bitmask values and an underlying type
// then use the macro to enable support:
//  enum class MyBitfield : uint32_t {
//    kFoo = 1 << 0,
//    kBar = 1 << 1,
//  };
//  IREE_BITFIELD(MyBitfield);
//  MyBitfield value = ~(MyBitfield::kFoo | MyBitfield::kBar);
//
// AnyBitSet is provided as a way to quickly test if any of the given bits are
// set:
//  if (AnyBitSet(value)) { /* one or more bits are set */ }
//
// If testing for equality it's recommended that AllBitsSet is used to ensure
// that combined values are handled properly:
//  if (AllBitsSet(value, MyBitfield::kSomeSetOfFlags)) { /* all bits set */ }

#ifndef IREE_BASE_BITFIELD_H_
#define IREE_BASE_BITFIELD_H_

#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"

namespace iree {

#define IREE_BITFIELD(enum_class)                                      \
  inline enum_class operator|(enum_class lhs, enum_class rhs) {        \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    return static_cast<enum_class>(static_cast<enum_type>(lhs) |       \
                                   static_cast<enum_type>(rhs));       \
  }                                                                    \
  inline enum_class& operator|=(enum_class& lhs, enum_class rhs) {     \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    lhs = static_cast<enum_class>(static_cast<enum_type>(lhs) |        \
                                  static_cast<enum_type>(rhs));        \
    return lhs;                                                        \
  }                                                                    \
  inline enum_class operator&(enum_class lhs, enum_class rhs) {        \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    return static_cast<enum_class>(static_cast<enum_type>(lhs) &       \
                                   static_cast<enum_type>(rhs));       \
  }                                                                    \
  inline enum_class& operator&=(enum_class& lhs, enum_class rhs) {     \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    lhs = static_cast<enum_class>(static_cast<enum_type>(lhs) &        \
                                  static_cast<enum_type>(rhs));        \
    return lhs;                                                        \
  }                                                                    \
  inline enum_class operator^(enum_class lhs, enum_class rhs) {        \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    return static_cast<enum_class>(static_cast<enum_type>(lhs) ^       \
                                   static_cast<enum_type>(rhs));       \
  }                                                                    \
  inline enum_class& operator^=(enum_class& lhs, enum_class rhs) {     \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    lhs = static_cast<enum_class>(static_cast<enum_type>(lhs) ^        \
                                  static_cast<enum_type>(rhs));        \
    return lhs;                                                        \
  }                                                                    \
  inline enum_class operator~(enum_class lhs) {                        \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    return static_cast<enum_class>(~static_cast<enum_type>(lhs));      \
  }                                                                    \
  inline bool AnyBitSet(enum_class lhs) {                              \
    using enum_type = typename std::underlying_type<enum_class>::type; \
    return static_cast<enum_type>(lhs) != 0;                           \
  }                                                                    \
  inline bool AllBitsSet(enum_class lhs, enum_class rhs) {             \
    return (lhs & rhs) == rhs;                                         \
  }

// Appends the formatted contents of the given bitfield value to a stream.
//
// Processes values in the order of the mapping table provided and will only
// use each bit once. Use this to prioritize combined flags over split ones.
template <typename T>
void FormatBitfieldValue(
    std::ostringstream* stream, T value,
    const absl::Span<const std::pair<T, const char*>> mappings) {
  T remaining_bits = value;
  int i = 0;
  for (const auto& mapping : mappings) {
    if ((remaining_bits & mapping.first) == mapping.first) {
      if (i > 0) {
        *stream << "|";
      }
      *stream << mapping.second;
      remaining_bits &= ~mapping.first;
      ++i;
    }
  }
  using enum_type = typename std::underlying_type<T>::type;
  if (remaining_bits != static_cast<T>(0)) {
    if (i > 0) {
      *stream << "|";
    }
    *stream << std::hex << static_cast<enum_type>(remaining_bits) << "h";
  }
}

// Returns a string with the formatted contents of the given bitfield value.
//
// Usage:
//  MyValue my_value = MyValue::kA | MyValue::kB;
//  std::string string_value = FormatBitfieldValue(my_value, {
//    {MyValue::kA, "kA"},
//    {MyValue::kB, "kB"},
//  });
//  // string_value contains 'kA|kB'
template <typename T>
std::string FormatBitfieldValue(
    T value, absl::Span<const std::pair<T, const char*>> mappings) {
  std::ostringstream stream;
  FormatBitfieldValue<T>(&stream, value, mappings);
  return stream.str();
}

}  // namespace iree

#endif  // IREE_BASE_BITFIELD_H_
