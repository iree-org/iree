// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
