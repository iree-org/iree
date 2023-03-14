// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

struct SPIRVBreakDownLargeVectorPass final
    : public SPIRVBreakDownLargeVectorBase<SPIRVBreakDownLargeVectorPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Convert vector.extract_strided_slice into a chain of vector.extract and
    // then a chain of vector.insert ops. This helps to cancel with previous
    // vector.insert/extract ops, especially for fP16 cases where we have
    // mismatched vector size for transfer and compute.
    vector::populateVectorExtractStridedSliceToExtractInsertChainPatterns(
        patterns, [](vector::ExtractStridedSliceOp op) {
          return op.getSourceVectorType().getNumElements() > 4;
        });
    vector::InsertOp::getCanonicalizationPatterns(patterns, context);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVBreakDownLargeVectorPass() {
  return std::make_unique<SPIRVBreakDownLargeVectorPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
