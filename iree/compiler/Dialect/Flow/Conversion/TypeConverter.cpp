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
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {

FlowTypeConverter::FlowTypeConverter() {
  // Allow types through by default.
  addConversion([](Type type) { return type; });

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

  // UNSAFE: narrow 64-bit integers to 32-bit ones.
  // This is a workaround for lower levels of the stack not always supporting
  // 64-bit types natively.
  // TODO(benvanik): make whether to narrow integers an option.
  addConversion([](IntegerType integerType) -> Optional<Type> {
    if (integerType.getWidth() > 32) {
      // Narrow to i32 preserving signedness semantics.
      return IntegerType::get(integerType.getContext(), 32,
                              integerType.getSignedness());
    }
    return llvm::None;
  });

  // UNSAFE: narrow 64-bit floats to 32-bit ones.
  // This is a workaround for lower levels of the stack not always supporting
  // 64-bit types natively.
  // TODO(benvanik): make whether to narrow floats an option.
  addConversion([](FloatType floatType) -> Optional<Type> {
    if (floatType.getWidth() > 32) {
      return FloatType::getF32(floatType.getContext());
    }
    return llvm::None;
  });
}

}  // namespace iree_compiler
}  // namespace mlir
