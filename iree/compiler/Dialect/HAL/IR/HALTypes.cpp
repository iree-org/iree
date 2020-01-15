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

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "llvm/ADT/StringExtras.h"

// Order matters:
#include "iree/compiler/Dialect/HAL/IR/HALEnums.cpp.inc"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

llvm::Optional<int32_t> getElementTypeValue(Type type) {
  // TODO(benvanik): replace this with the signature mangling stuff for
  // consistency? For now we are just using it for demos, so this is ok:
  static const uint32_t kSignedIntType = 0;
  static const uint32_t kFloatType = 2;
  if (auto intType = type.dyn_cast_or_null<IntegerType>()) {
    return (kSignedIntType << 16) | intType.getWidth();
  } else if (auto floatType = type.dyn_cast_or_null<FloatType>()) {
    // TODO(benvanik): fltSemantics.
    return (kFloatType << 16) | floatType.getWidth();
  }
  return llvm::None;
}

#include "iree/compiler/Dialect/HAL/IR/HALOpInterface.cpp.inc"

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
