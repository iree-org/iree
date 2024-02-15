// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-optimize-vector-shapes"

using namespace mlir::iree_compiler;

static unsigned getNativeVectorSizeInBytes(mlir::FunctionOpInterface funcOp) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  auto nativeVectorSizeAttr =
      getConfigIntegerAttr(targetAttr, "native_vector_size");
  if (nativeVectorSizeAttr) {
    unsigned nativeVectorSizeVal = nativeVectorSizeAttr->getInt();
    if (nativeVectorSizeVal) {
      return nativeVectorSizeVal;
    }
  }

  return 0;
}

namespace mlir::iree_compiler {
namespace {
class LLVMCPUOptimizeVectorShapesPass
    : public LLVMCPUOptimizeVectorShapesBase<LLVMCPUOptimizeVectorShapesPass> {
public:
  using LLVMCPUOptimizeVectorShapesBase::LLVMCPUOptimizeVectorShapesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};

// TODO: Replace/remove the existing OptimizeVectorTransferPass.
void LLVMCPUOptimizeVectorShapesPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // Apply transfer ops write to read forwarding and dead transfer write
  // optimizations.
  // TODO: Remove store-to-load forwarding to a separate pass as it's unrelated
  // to the vector shape optimizations applied later.
  IRRewriter rewriter(ctx);
  vector::transferOpflowOpt(rewriter, funcOp);

  // Remove unit dimensons.
  // TODO: Revisit some of these patterns to make sure they are not redundant.
  // Dropping trailing unit dims from transfer ops is equivalent to apply
  // flattening.
  {
    RewritePatternSet patterns(ctx);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateVectorTransferCollapseInnerMostContiguousDimsPatterns(
        patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);
    vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  unsigned targetVectorBitwidth = getNativeVectorSizeInBytes(funcOp) * 8;
  if (targetVectorBitwidth == 0)
    return;

  // Collapse dimensions from transfer ops that are contiguous in memory.
  // TODO: Extend flattening patterns so that we can provide a target vector
  // length to make a more targeted flattening decisions.
  RewritePatternSet patterns(ctx);
  vector::populateFlattenVectorTransferPatterns(patterns, targetVectorBitwidth);
  memref::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUOptimizeVectorShapesPass() {
  return std::make_unique<LLVMCPUOptimizeVectorShapesPass>();
}

} // namespace mlir::iree_compiler
