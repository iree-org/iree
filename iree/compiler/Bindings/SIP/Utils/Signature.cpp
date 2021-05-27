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
