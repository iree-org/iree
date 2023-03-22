// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-tensor-pad"

namespace mlir {
namespace iree_compiler {
namespace {
class LLVMCPUTensorPadPass : public LLVMCPUTensorPadBase<LLVMCPUTensorPadPass> {
 private:
  LLVMCPUTensorPadOption option = LLVMCPUTensorPadOption::ParallelDims;

 public:
  explicit LLVMCPUTensorPadPass(LLVMCPUTensorPadOption option)
      : option(option) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUTensorPadPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();
  utils::IteratorType targetIterType;
  switch (option) {
    case LLVMCPUTensorPadOption::ParallelDims:
      LLVM_DEBUG(llvm::dbgs() << "padding parallel dims\n");
      targetIterType = utils::IteratorType::parallel;
      break;
    case LLVMCPUTensorPadOption::ReductionDims:
      LLVM_DEBUG(llvm::dbgs() << "padding reduction dims\n");
      targetIterType = utils::IteratorType::reduction;
      break;
  };
  SmallVector<linalg::LinalgOp> candidates;
  funcOp.walk([&](linalg::LinalgOp op) { candidates.push_back(op); });
  for (auto linalgOp : candidates) {
    IRRewriter rewriter(context);
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << linalgOp);
    IRRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(linalgOp);

    SmallVector<int64_t> paddingDims;
    for (auto [index, iterType] :
         llvm::enumerate(linalgOp.getIteratorTypesArray())) {
      if (iterType == targetIterType) {
        paddingDims.push_back(index);
      }
    }

    SmallVector<Attribute> paddingValueAttributes;
    OpBuilder builder(context);
    for (auto &operand : linalgOp->getOpOperands()) {
      auto elemType = getElementTypeOrSelf(operand.get().getType());
      paddingValueAttributes.push_back(builder.getZeroAttr(elemType));
    }

    auto options = linalg::LinalgPaddingOptions()
                       .setPaddingDimensions(paddingDims)
                       .setPaddingValues(paddingValueAttributes);
    FailureOr<linalg::LinalgOp> maybePaddedLinalgOp =
        linalg::padAndHoistLinalgOp(rewriter, linalgOp, options);
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
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "----- cleanup failed -----\n");
      return signalPassFailure();
    }
  }
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUTensorPadPass(
    LLVMCPUTensorPadOption option) {
  return std::make_unique<LLVMCPUTensorPadPass>(option);
}
}  // namespace iree_compiler
}  // namespace mlir
