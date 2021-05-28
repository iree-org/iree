// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include <memory>

#include "iree/compiler/Dialect/Shape/IR/ShapeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OperationPass<FuncOp>> createExpandFunctionDynamicDimsPass();

// For any function which contains ranked_shape argument/result types,
// expands them to individual dynamic dimensions, inserting appropriate casts
// within the function.
std::unique_ptr<OperationPass<FuncOp>>
createExpandFunctionRankedShapeDimsPass();

// For any dynamically shaped edges in a function, introduces an appropriate
// get_ranked_shape and corresponding tie_shape op to make the association.
//
// Any op contained in a region transitively owned by an op with a name in
// `doNotRecurseOpNames` is not tied.
std::unique_ptr<OperationPass<FuncOp>> createTieDynamicShapesPass(
    ArrayRef<std::string> doNotRecurseOpNames = {});

// Materializes shape calculations for any get_ranked_shape ops.
std::unique_ptr<OperationPass<FuncOp>> createMaterializeShapeCalculationsPass();

// Cleans up any unnecessary shape placeholder ops. Can be run after all
// shape calculation code has been lowered.
std::unique_ptr<OperationPass<FuncOp>> createCleanupShapePlaceholdersPass();

// Converts shape-sensitive HLOs to be based on facilities in the shape
// dialect.
std::unique_ptr<OperationPass<FuncOp>> createConvertHLOToShapePass();

// Best-effort hoisting of shape calculations to attempt to establish the
// invariant that shapex.tie_shape second operand dominates the first operand.
std::unique_ptr<OperationPass<FuncOp>> createHoistShapeCalculationsPass();

// Register all Passes
inline void registerShapePasses() {
  createExpandFunctionDynamicDimsPass();
  createExpandFunctionRankedShapeDimsPass();
  createTieDynamicShapesPass();
  createMaterializeShapeCalculationsPass();
  createCleanupShapePlaceholdersPass();
  createConvertHLOToShapePass();
  createHoistShapeCalculationsPass();
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
