// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmcpu-split-reduction"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMCPUSPLITREDUCTIONPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

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
    LDBG() << "doesn't have tensor semantics";
    return failure();
  }
  if (linalgOp.getNumReductionLoops() != 1) {
    LDBG() << "number of reduction loops != 1";
    return failure();
  }
  if (linalgOp.getNumDpsInits() != 1) {
    LDBG() << "doesn't have exactly 1 output";
    return failure();
  }
  if (!linalgOp.hasOnlyProjectedPermutations()) {
    LDBG() << "index map doesn't have only projected permutations";
    return failure();
  }
  if (!isa<linalg::GenericOp>(op)) {
    LDBG() << "is not a generic op";
    return failure();
  }
  if (linalgOp.getNumDpsInputs() != 1) {
    LDBG() << "doesn't have exactly 1 input";
    return failure();
  }
  // The `linalg::splitReduction` method does not work for ops with indexing
  // semantics. See https://github.com/iree-org/iree/pull/14979
  if (linalgOp.hasIndexSemantics()) {
    LDBG()
        << "the split method used currently doesnt support indexing semantics";
    return failure();
  }

  auto elemType =
      getElementTypeOrSelf(linalgOp.getDpsInitOperand(0)->get().getType());
  if (!(fpReductionReordering || elemType.isIntOrIndex())) {
    LDBG() << "skipped because reduction reordering on FP is not enabled.";
    return failure();
  }

  SmallVector<unsigned> dims;
  linalgOp.getReductionDims(dims);
  AffineMap map =
      linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  unsigned lastIdx = map.getNumResults() - 1;
  unsigned lastDim = map.getDimPosition(lastIdx);
  if (lastDim != dims[0]) {
    LDBG() << "innermost dimension of the input operand is not reduction";
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
    LDBG() << "failed on step 1 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(linalgOp, tileResFirst->replacements);

  // 2) Apply splitReduction on the single vector-length array.
  // splitReduction already replaces the op.
  FailureOr<linalg::SplitReductionResult> splitRes = splitReduction(
      rewriter, cast<linalg::LinalgOp>(tileResFirst->tiledOps.back()), fn);
  if (failed(splitRes)) {
    LDBG() << "failed on step 2 (SplitReduction)";
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
    LDBG() << "failed on step 3 (SCFTiling)";
    return failure();
  }
  rewriter.replaceOp(splitRes->splitLinalgOp, tileRes->replacements);
  return success();
}

/// Pass to splitReduce linalg operations.
class LLVMCPUSplitReductionPass
    : public impl::LLVMCPUSplitReductionPassBase<LLVMCPUSplitReductionPass> {
public:
  using impl::LLVMCPUSplitReductionPassBase<
      LLVMCPUSplitReductionPass>::LLVMCPUSplitReductionPassBase;
  explicit LLVMCPUSplitReductionPass(bool fpReductionReordering) {
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
    LDBG() << "candidate: " << genericOp;
    if (failed(splitReductionPrecondition(genericOp,
                                          enableFpReductionReordering))) {
      continue;
    }

    IREE::Codegen::LoweringConfigAttrInterface maybeLoweringConfig =
        getLoweringConfig(genericOp);
    if (!maybeLoweringConfig) {
      LDBG() << "can't find lowering_config, skip SplitReduction";
      continue;
    }
    auto attr = cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
        maybeLoweringConfig.getTilingLevelAttr(
            IREE::CPU::TilingLevel::VectorReductionTiles));
    ArrayRef<bool> scalableDims = attr.getScalableFlags();
    if (scalableDims.back()) {
      LDBG() << "scalable reduction dimensions not yet supported, skip "
                "SplitReduction";
      continue;
    }
    ArrayRef<int64_t> reductionSizes = attr.getSizes();
    if (reductionSizes.empty()) {
      LDBG()
          << "the list of reduction tiling sizes is empty, skip SplitReduction";
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
