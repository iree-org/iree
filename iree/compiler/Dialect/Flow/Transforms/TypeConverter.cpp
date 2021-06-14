// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/TypeConverter.h"

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
