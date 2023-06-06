// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/LLVMCPUPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tensor-pad"

namespace mlir {
namespace iree_compiler {
namespace {
class LLVMCPUTensorPadPass : public LLVMCPUTensorPadBase<LLVMCPUTensorPadPass> {
 public:
  using LLVMCPUTensorPadBase::LLVMCPUTensorPadBase;
  explicit LLVMCPUTensorPadPass(LLVMCPUTensorPadOptions options)
      : options(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;

 private:
  LLVMCPUTensorPadOptions options;
};

void LLVMCPUTensorPadPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  // Preserve the innermost tensor.pad ops (i.e., pad for reduction dims), so we
  // can kick canonicalization patterns to fold outer tensor.pad ops away.
  bool nofold;
  utils::IteratorType targetIterType;
  switch (options.padDims) {
    case LLVMCPUTensorPadOptions::LLVMCPUPadDims::ParallelDims:
      LLVM_DEBUG(llvm::dbgs() << "padding parallel dims\n");
      targetIterType = utils::IteratorType::parallel;
      nofold = false;
      break;
    case LLVMCPUTensorPadOptions::LLVMCPUPadDims::ReductionDims:
      LLVM_DEBUG(llvm::dbgs() << "padding reduction dims\n");
      targetIterType = utils::IteratorType::reduction;
      nofold = true;
      break;
  };
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) { candidates.push_back(op); });
  for (auto linalgOp : candidates) {
    IRRewriter rewriter(context);
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << linalgOp);

    // Early exit if there are no target dimensions to pad.
    if (options.padDims ==
            LLVMCPUTensorPadOptions::LLVMCPUPadDims::ParallelDims &&
        linalgOp.getNumParallelLoops() == 0)
      continue;
    if (options.padDims ==
            LLVMCPUTensorPadOptions::LLVMCPUPadDims::ReductionDims &&
        linalgOp.getNumReductionLoops() == 0)
      continue;

    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(linalgOp);

    SmallVector<int64_t> paddingDims;
    SmallVector<int64_t> padToMultipleOf;
    for (auto [index, iterType] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (iterType == targetIterType) {
        paddingDims.push_back(index);
        padToMultipleOf.push_back(options.padToMultipleOf[index]);
      }
    }

    SmallVector<Attribute> paddingValueAttributes;
    OpBuilder builder(context);
    for (auto &operand : linalgOp->getOpOperands()) {
      auto elemType = getElementTypeOrSelf(operand.get().getType());
      paddingValueAttributes.push_back(builder.getZeroAttr(elemType));
    }

    // If nofold is true, we must create pad ops for input operands. The output
    // operands mostly come from scf.for iter_arg. We can not infer the bounding
    // box for such case, so we do not force pad happening.
    SmallVector<bool> noFold(linalgOp.getNumDpsInputs(), nofold);
    noFold.append(linalgOp.getNumDpsInits(), false);

    auto padAndHoistOptions = linalg::LinalgPaddingOptions()
                                  .setPaddingDimensions(paddingDims)
                                  .setPaddingValues(paddingValueAttributes)
                                  .setPackPaddings(noFold)
                                  .setPadToMultipleOf(padToMultipleOf);
    FailureOr<linalg::LinalgOp> maybePaddedLinalgOp =
        linalg::padAndHoistLinalgOp(rewriter, linalgOp, padAndHoistOptions);
    if (failed(maybePaddedLinalgOp)) {
      LLVM_DEBUG(llvm::dbgs() << "failed on padding\n");
      return signalPassFailure();
    }

    // TODO(hanchung): The upstream utils should use OpFoldResult. Then we don't
    // have to clean things up after every padding transform.
    RewritePatternSet patterns(context);
    memref::populateResolveRankedShapeTypeResultDimsPatterns(patterns);
    context->getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    tensor::PadOp::getCanonicalizationPatterns(patterns, context);
    // Fold multiple tensor.extract_slice ops into a single one. They will be
    // unfolded back later in the pipeline.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTensorPadPass() {
  return std::make_unique<LLVMCPUTensorPadPass>();
}
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTensorPadPass(
    const LLVMCPUTensorPadOptions& options) {
  return std::make_unique<LLVMCPUTensorPadPass>(options);
}
}  // namespace iree_compiler
}  // namespace mlir
