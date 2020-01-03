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

#include "iree/compiler/Utils/TypeConversionUtils.h"

#include <cassert>

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

Type legalizeType(Type type) {
  if (type.isIndex()) {
    return IntegerType::get(kIndexBitWidth, type.getContext());
  } else if (type.isInteger(1)) {
    return IntegerType::get(kBoolBitWidth, type.getContext());
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return MemRefType::get(memRefType.getShape(),
                           legalizeType(memRefType.getElementType()));
  } else if (auto functionType = type.dyn_cast<FunctionType>()) {
    llvm::SmallVector<Type, 4> inputs;
    for (const auto &oldType : functionType.getInputs()) {
      inputs.push_back(legalizeType(oldType));
    }
    llvm::SmallVector<Type, 4> results;
    for (const auto &oldType : functionType.getResults()) {
      results.push_back(legalizeType(oldType));
    }
    return FunctionType::get(inputs, results, type.getContext());
  }
  return type;
}

MemRefType convertTypeToMemRef(Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return MemRefType::get({}, type, {}, 0);
  } else if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  } else if (auto memRefType = type.dyn_cast<MemRefType>()) {
    return memRefType;
  } else {
    llvm_unreachable("Unconvertable type");
  }
}

MemRefType convertTypeToMemRef(Value value) {
  return convertTypeToMemRef(value.getType());
}

}  // namespace iree_compiler
}  // namespace mlir
