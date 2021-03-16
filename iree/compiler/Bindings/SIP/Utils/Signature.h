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

#ifndef IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_H_
#define IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_H_

#include <array>
#include <cstddef>

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

namespace AbiConstants {

// Canonical integer mappings are maintained for core scalar type codes
// since they change infrequently and are used everywhere.
// Generally, favor adding a custom type vs extending this arbitrarily.
enum class ScalarType : unsigned {
  kIeeeFloat32 = 0,
  kIeeeFloat16 = 1,
  kIeeeFloat64 = 2,
  kGoogleBfloat16 = 3,
  kSint8 = 4,
  kSint16 = 5,
  kSint32 = 6,
  kSint64 = 7,
  kUint8 = 8,
  kUint16 = 9,
  kUint32 = 10,
  kUint64 = 11,
  kMaxScalarType = 11,
};

// Array that maps ScalarType codes to the size in bytes.
extern const std::array<size_t,
                        static_cast<unsigned>(ScalarType::kMaxScalarType) + 1>
    kScalarTypeSize;

extern const std::array<const char*,
                        static_cast<unsigned>(ScalarType::kMaxScalarType) + 1>
    kScalarTypeNames;

}  // namespace AbiConstants

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_BINDINGS_SIP_UTILS_SIGNATURE_H_
