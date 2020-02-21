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

#include "iree/compiler/Dialect/Flow/Conversion/TypeConverter.h"

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace iree_compiler {

FlowTypeConverter::FlowTypeConverter() {
  // Allow types through by default.
  addConversion([](Type type) { return type; });
  addConversion([](IndexType type) {
    // Always treat as 32-bit.
    return IntegerType::get(32, type.getContext());
  });
  addConversion([](IntegerType integerType) -> Optional<Type> {
    if (integerType.getWidth() > 32) {
      // Don't support 64-bit types in general. Rewrite to i32 (if desired).
      // TODO(benvanik): split to i32+i32? allow and use availability?
      // TODO(benvanik): make an option.
      return IntegerType::get(32, integerType.getContext());
    }
    return llvm::None;
  });
  addConversion([](FloatType floatType) -> Optional<Type> {
    if (floatType.getWidth() > 32) {
      // Don't support 64-bit types in general. Rewrite to f32 (if desired).
      // TODO(benvanik): make an option.
      return FloatType::getF32(floatType.getContext());
    }
    return llvm::None;
  });
  addConversion([this](RankedTensorType tensorType) -> Optional<Type> {
    auto convertedElementType = convertType(tensorType.getElementType());
    if (!convertedElementType) {
      return llvm::None;
    }
    return RankedTensorType::get(tensorType.getShape(), convertedElementType);
  });
  addConversion([](UnrankedTensorType tensorType) {
    // We only support ranked tensors. We could convert unranked to ranked
    // here for certain cases (such as * on the LHS).
    return Type();
  });
}

Operation *FlowTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                    Type resultType,
                                                    ArrayRef<Value> inputs,
                                                    Location loc) {
  // TODO(b/145876978): materialize conversion when this is called.
  llvm_unreachable("unhandled materialization");
  return nullptr;
}

}  // namespace iree_compiler
}  // namespace mlir
