// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// Folds tensor.dim/memref.dim ops taking shape carrying ops as operands.
std::unique_ptr<OperationPass<FuncOp>> createFoldDimOverShapeCarryingOpPass();

// Cleans up any unnecessary shape placeholder ops. Can be run after all
// shape calculation code has been lowered.
std::unique_ptr<OperationPass<FuncOp>> createCleanupShapePlaceholdersPass();

// Register all Passes
inline void registerShapePasses() {
  createFoldDimOverShapeCarryingOpPass();
  createCleanupShapePlaceholdersPass();
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
