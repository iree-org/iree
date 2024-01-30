// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-split-reduction"

namespace mlir::iree_compiler {

namespace {

/// Make sure that
/// - the pass has not been applied before
/// - has tensor semantics
/// - number of reduction loops == 1
/// - has exactly 1 output
/// - index map has only projected permutations
/// - is a linalg generic op
/// - has exactly 1 input
/// - if enableReductionReordering is not set, then operand is an int
/// - innermost dimension of the input operand is reduction
/// TODO: support named ops, numInputs > 1, and modify lastDim check below
/// accordingly. If fpReductionReordering is not enabled by default, it must
/// be an integer or index type to proceed to allow associative reordering.
LogicalResult splitReductionPrecondition(Operation *op,
                                         bool fpReductionReordering) {
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

  if (!linalgOp.hasPureTensorSemantics()) {
    LLVM_DEBUG(llvm::dbgs() << "doesn't have tensor semantics\n");
    return failure();
  }
  if (linalgOp.getNumReductionLoops() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "number of reduction loops != 1\n");
    return failure();
  }
  if (linalgOp.getNumDpsInits() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "doesn't have exactly 1 output\n");
    return failure();
  }
  if (!linalgOp.hasOnlyProjectedPermutations()) {
    LLVM_DEBUG(llvm::dbgs()
               << "index map doesn't have only projected permutations\n");
    return failure();
  }
  if (!isa<linalg::GenericOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "is not a generic op\n");
    return failure();
  }
  if (linalgOp.getNumDpsInputs() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "doesn't have exactly 1 input\n");
    return failure();
  }
  // The `linalg::splitReduction` method does not work for ops with indexing
  // semantics. See https://github.com/openxla/iree/pull/14979
  if (linalgOp.hasIndexSemantics()) {
    LLVM_DEBUG(llvm::dbgs() << "the split method used currently doesnt support "
                               "indexing semantics\n");
    return failure();
  }

  auto elemType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!(fpReductionReordering || elemType.isIntOrIndex())) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "skipped because reduction reordering on FP is not enabled.\n");
    return failure();
  }

  SmallVector<unsigned> dims;
  linalgOp.getReductionDims(dims);
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  unsigned lastDim = map.getDimPosition(lastIdx);
  if (lastDim != dims[0]) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "innermost dimension of the input operand is not reduction\n");
    return failure();
  }

  return success();
}

/// Converts an inner-reduction into outer reduction + inner-parallel dimension,
/// followed by simple inner reduction.
LogicalResult splitReductionImpl(Operation *op, int64_t size,
                                 RewriterBase &rewriter) {
  IRRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(op);
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);

  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  linalg::ControlSplitReductionFn fn = [size, lastIdx](linalg::LinalgOp) {
    return linalg::SplitReductionOptions{size, lastIdx,
                                         /*innerParallel=*/true};
  };

  auto numLoops = linalgOp.getNumLoops();

  // 1) Tile to extract a single vector-length array.
  SmallVector<OpFoldResult> tileSizesSVFirst(numLoops,
                                             rewriter.getIndexAttr(1));
  tileSizesSVFirst[numLoops - 1] = rewriter.getIndexAttr(0);
  auto options = scf::SCFTilingOptions().setTileSizes(tileSizesSVFirst);
  FailureOr<scf::SCFTilingResult> tileResFirst = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(linalgOp.getOperation()), options);
  if (failed(tileResFirst)) {
    LLVM_DEBUG(llvm::dbgs() << "failed on step 1 (SCFTiling)\n");
    return failure();
  }
  rewriter.replaceOp(linalgOp, tileResFirst->replacements);

  // 2) Apply splitReduction on the single vector-length array.
  // splitReduction already replaces the op.
  FailureOr<linalg::SplitReductionResult> splitRes = splitReduction(
      rewriter, cast<linalg::LinalgOp>(tileResFirst->tiledOps.back()), fn);
  if (failed(splitRes)) {
    LLVM_DEBUG(llvm::dbgs() << "failed on step 2 (SplitReduction)\n");
    return success();
  }

  // 3) Tile the first op generated by splitReduction with tile size of 1,
  // to essentially create a reduction loop. Note that
  // splitRes->splitLinalgOp.getNumLoops() = numLoops + 1.
  SmallVector<OpFoldResult> tileSizesSV(splitRes->splitLinalgOp.getNumLoops(),
                                        rewriter.getIndexAttr(0));
  // The reduction happens only in the penultimate dimension, which we now
  // tile.
  tileSizesSV[numLoops - 1] = rewriter.getIndexAttr(1);
  options = scf::SCFTilingOptions().setTileSizes(tileSizesSV);
  FailureOr<scf::SCFTilingResult> tileRes = scf::tileUsingSCF(
      rewriter, cast<TilingInterface>(splitRes->splitLinalgOp.getOperation()),
      options);
  if (failed(tileRes)) {
    LLVM_DEBUG(llvm::dbgs() << "failed on step 3 (SCFTiling)\n");
    return failure();
  }
  rewriter.replaceOp(splitRes->splitLinalgOp, tileRes->replacements);
  return success();
}

/// Pass to splitReduce linalg operations.
class LLVMCPUSplitReductionPass
    : public LLVMCPUSplitReductionBase<LLVMCPUSplitReductionPass> {
public:
  LLVMCPUSplitReductionPass(bool fpReductionReordering) {
    this->enableFpReductionReordering = fpReductionReordering;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect>();
  }
  void runOnOperation() override;
};

void LLVMCPUSplitReductionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  IRRewriter rewriter(context);
  SmallVector<linalg::GenericOp> candidates;
  funcOp.walk([&](linalg::GenericOp op) { candidates.push_back(op); });
  for (auto genericOp : candidates) {
    LLVM_DEBUG(llvm::dbgs() << "candidate: " << genericOp << "\n");
    if (failed(splitReductionPrecondition(genericOp,
                                          enableFpReductionReordering))) {
      continue;
    }

    FailureOr<IREE::Codegen::LoweringConfigAttr> maybeLoweringConfig =
        getLoweringConfig(genericOp);
    if (failed(maybeLoweringConfig)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "can't find lowering_config, skip SplitReduction");
      continue;
    }
    TilingConfig tilingConfig(maybeLoweringConfig.value());
    auto [reductionSizes, scalableDims] =
        tilingConfig.getVectorReductionSizes();
    if (scalableDims.back()) {
      LLVM_DEBUG(llvm::dbgs() << "scalable reduction dimensions not yet "
                                 "supported, skip SplitReduction");
      continue;
    }
    if (reductionSizes.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "the list of reduction tiling sizes is empty, "
                                 "skip SplitReduction");
      continue;
    }
    int64_t size = reductionSizes.back();
    if (failed(splitReductionImpl(genericOp, size, rewriter))) {
      return signalPassFailure();
    }
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(const bool enableFpReductionReordering) {
  return std::make_unique<LLVMCPUSplitReductionPass>(
      enableFpReductionReordering);
}

} // namespace mlir::iree_compiler
