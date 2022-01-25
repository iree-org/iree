// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {
struct LLVMGPUVectorLoweringPass
    : public LLVMGPUVectorLoweringBase<LLVMGPUVectorLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet vectorToLoopsPatterns(&getContext());
    VectorTransferToSCFOptions vectorToSCFOptions;
    vectorToSCFOptions.enableFullUnroll();
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    memref::populateFoldSubViewOpPatterns(vectorToLoopsPatterns);
    vector::populateVectorTransferLoweringPatterns(vectorToLoopsPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorLoweringPass() {
  return std::make_unique<LLVMGPUVectorLoweringPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
