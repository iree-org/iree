// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

std::unique_ptr<OperationPass<void>> createApplyPatternsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createCombineInitializersPass();
std::unique_ptr<OperationPass<void>> createDropCompilerHintsPass();
std::unique_ptr<OperationPass<void>> createFixedPointIteratorPass(
    OpPassManager pipeline);
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFuseGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createHoistIntoGlobalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createIPOPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPropagateSubrangesPass();
std::unique_ptr<OperationPass<void>> createSimplifyGlobalAccessesPass();
std::unique_ptr<OperationPass<void>> createStripDebugOpsPass();

// Type conversion.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteI64ToI32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF32ToF16Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF64ToF32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPromoteF16ToF32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPromoteBF16ToF32Pass();
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createPromoteArithBF16ToF32Pass();

// Debug/test passes.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAnnotateOpOrdinalsPass();
std::unique_ptr<OperationPass<mlir::ModuleOp>> createTestConversionPass();
std::unique_ptr<OperationPass<void>> createTestFloatRangeAnalysisPass();

// Register all Passes
void registerTransformPasses();

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_TRANSFORMS_PASSES_H_
