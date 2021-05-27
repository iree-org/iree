// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// CallbackCustomOpShapeBuilder
//===----------------------------------------------------------------------===//

void CallbackCustomOpShapeBuilder::insertRankedShapeBuilder(
    llvm::StringRef operationName, RankedShapeBuilder callback) {
  rankedShapeBuilders.insert(
      std::make_pair(operationName, std::move(callback)));
}

Value CallbackCustomOpShapeBuilder::buildRankedShape(
    RankedShapeType resultShape, Operation *inputOperation,
    OpBuilder &builder) {
  auto it = rankedShapeBuilders.find(inputOperation->getName().getStringRef());
  if (it == rankedShapeBuilders.end()) {
    return nullptr;
  }
  return it->second(resultShape, inputOperation, builder);
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
