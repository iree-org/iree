// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_CONVERSION_PASSES_H_
#define IREE_COMPILER_DIALECT_SHAPE_CONVERSION_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Convert `shape` dialect to `shapex` dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertShapeToShapexPass();

inline void registerShapeConversionPasses() {
  createConvertShapeToShapexPass();
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_CONVERSION_PASSES_H_
