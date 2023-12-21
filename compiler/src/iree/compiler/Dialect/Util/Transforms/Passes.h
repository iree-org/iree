// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_

#include <optional>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
class OpBuilder;
class Type;
class Value;
} // namespace mlir

namespace mlir::iree_compiler::IREE::Util {

std::unique_ptr<OperationPass<void>> createApplyPatternsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createCombineInitializersPass();
std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass();
std::unique_ptr<OperationPass<void>>
createFixedPointIteratorPass(OpPassManager pipeline);
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createIPOPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createOutlineConstantsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateSubrangesPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createSimplifyGlobalAccessesPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createStripAndSplatConstantsPass();
std::unique_ptr<OperationPass<void>> createStripDebugOpsPass();

// Expression hoisting.

struct ExprHoistingOptions {
  using RegisterDialectsFn = std::function<void(DialectRegistry &)>;

  // Hook to register extra dependent dialects needed for types implementing
  // the `HoistableTypeInterace`.
  std::optional<RegisterDialectsFn> registerDependentDialectsFn = std::nullopt;

  // Threshold for controlling the maximum allowed increase in the stored size
  // of a single global as a result of hoisting.
  int64_t maxSizeIncreaseThreshold = 2147483647;
};
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createHoistIntoGlobalsPass(const ExprHoistingOptions &options);
std::unique_ptr<OperationPass<mlir::ModuleOp>> createHoistIntoGlobalsPass();

// Resource Management.
std::unique_ptr<OperationPass<void>> createImportResourcesPass();

// Type conversion.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteI64ToI32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF32ToF16Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF64ToF32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPromoteF16ToF32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPromoteBF16ToF32Pass();

// Debug/test passes.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAnnotateOpOrdinalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createTestConversionPass();
std::unique_ptr<OperationPass<void>> createTestFloatRangeAnalysisPass();

// Register all Passes
void registerTransformPasses();

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
