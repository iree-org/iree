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
  // Preserve the innermost tensor.pad ops (i.e., pad for reduction dims), so we
  // can kick canonicalization patterns to fold outer tensor.pad ops away.
  bool nofold;
  utils::IteratorType targetIterType;
  switch (option) {
    case LLVMCPUTensorPadOption::ParallelDims:
      LLVM_DEBUG(llvm::dbgs() << "padding parallel dims\n");
      targetIterType = utils::IteratorType::parallel;
      nofold = false;
      break;
    case LLVMCPUTensorPadOption::ReductionDims:
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
    if (option == LLVMCPUTensorPadOption::ParallelDims &&
        linalgOp.getNumParallelLoops() == 0)
      continue;
    if (option == LLVMCPUTensorPadOption::ReductionDims &&
        linalgOp.getNumReductionLoops() == 0)
      continue;

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

    // If nofold is true, we must create pad ops for input operands. The output
    // operands mostly come from scf.for iter_arg. We can not infer the bounding
    // box for such case, so we do not force pad happening.
    SmallVector<bool> noFold(linalgOp.getNumDpsInputs(), nofold);
    noFold.append(linalgOp.getNumDpsInits(), false);

    auto options = linalg::LinalgPaddingOptions()
                       .setPaddingDimensions(paddingDims)
                       .setPaddingValues(paddingValueAttributes)
                       .setPackPaddings(noFold);
    FailureOr<linalg::LinalgOp> maybePaddedLinalgOp =
        linalg::padAndHoistLinalgOp(rewriter, linalgOp, options);
    if (failed(maybePaddedLinalgOp)) {
      LLVM_DEBUG(llvm::dbgs() << "failed on padding\n");
      return signalPassFailure();
    }

    // TODO(hanchung): The upstream utils should use OpFoldResult. Then we don't
    // have to clean things up after every padding transform.
    RewritePatternSet patterns(context);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    context->getLoadedDialect<tensor::TensorDialect>()
        ->getCanonicalizationPatterns(patterns);
    tensor::PadOp::getCanonicalizationPatterns(patterns, context);
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
