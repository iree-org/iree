// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/SIP/Utils/Signature.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace SIP {

// -----------------------------------------------------------------------------
// AbiConstants
// -----------------------------------------------------------------------------

const std::array<size_t, 12> AbiConstants::kScalarTypeSize = {
    4,  // kIeeeFloat32 = 0,
    2,  // kIeeeFloat16 = 1,
    8,  // kIeeeFloat64 = 2,
    2,  // kGoogleBfloat16 = 3,
    1,  // kSint8 = 4,
    2,  // kSint16 = 5,
    4,  // kSint32 = 6,
    8,  // kSint64 = 7,
    1,  // kUint8 = 8,
    2,  // kUint16 = 9,
    4,  // kUint32 = 10,
    8,  // kUint64 = 11,
};

const std::array<const char*, 12> AbiConstants::kScalarTypeNames = {
    "float32", "float16", "float64", "bfloat16", "sint8",  "sint16",
    "sint32",  "sint64",  "uint8",   "uint16",   "uint32", "uint64",
};

}  // namespace SIP
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
