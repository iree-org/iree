// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/Transforms/Passes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::iree_compiler::IREE::TensorExt {

#define GEN_PASS_DEF_TESTSPARSEOPINTERFACEMETHODSPASS
#include "iree/compiler/Dialect/TensorExt/Transforms/Passes.h.inc" // IWYU pragma: keep

namespace {

struct TestSparseOpInterfaceMethodsPass
    : public impl::TestSparseOpInterfaceMethodsPassBase<
          TestSparseOpInterfaceMethodsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::TensorExt::IREETensorExtDialect>();
  }
  void runOnOperation() override;
};

} // namespace

// Test the `lowerLoopRange` method of the SparseOpInterface. Replace the
// `forallOp` with nested `scf.for` operations that iterate over the sparse
// range.
static LogicalResult
testLowerLoopRangeImpl(scf::ForallOp forallOp,
                       IREE::TensorExt::SparseOpInterface sparseOp) {
  SmallVector<OpFoldResult> mixedOffsets = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedSizes = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStrides = forallOp.getMixedStep();
  auto range = llvm::map_to_vector(
      llvm::zip_equal(mixedOffsets, mixedSizes, mixedStrides),
      [](auto it) -> Range {
        auto [offset, size, stride] = it;
        return Range{offset, size, stride};
      });

  auto sparseIterationDimsAttr =
      IREE::TensorExt::getSparseIterationDimsAttr(forallOp);
  if (!sparseIterationDimsAttr) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute on ForallOp");
  }

  ArrayRef<int64_t> sparseIterationDims =
      sparseIterationDimsAttr.getSparseIterationDims();
  if (sparseIterationDims.size() != range.size()) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute to have the same size as "
        "the loop range");
  }

  IRRewriter rewriter(forallOp->getContext());
  rewriter.setInsertionPoint(forallOp);
  FailureOr<SmallVector<Value>> replacementIvs =
      sparseOp.lowerLoopRange(rewriter, sparseIterationDims, range);
  if (failed(replacementIvs)) {
    return forallOp.emitError("lowerLoopRange failed");
  }
  Block *currBlock = rewriter.getInsertionBlock();
  Block *splitBlock = rewriter.splitBlock(
      currBlock, Block::iterator(*currBlock->getTerminator()));
  Block *origBlock = forallOp.getBody();
  rewriter.eraseOp(origBlock->getTerminator());
  rewriter.mergeBlocks(splitBlock, origBlock);
  rewriter.mergeBlocks(origBlock, currBlock, replacementIvs.value());
  rewriter.eraseOp(forallOp);

  return success();
}

// Test the `getEstimatedLoopRange` method of the SparseOpInterface.
// Replace the `forallOp` with a new `scf.forall` with the estimated loop range.
static LogicalResult
testGetEstimatedLoopRangeImpl(scf::ForallOp forallOp,
                              IREE::TensorExt::SparseOpInterface sparseOp) {
  SmallVector<OpFoldResult> mixedOffsets = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedSizes = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStrides = forallOp.getMixedStep();
  auto range = llvm::map_to_vector(
      llvm::zip_equal(mixedOffsets, mixedSizes, mixedStrides),
      [](auto it) -> Range {
        auto [offset, size, stride] = it;
        return Range{offset, size, stride};
      });

  auto sparseIterationDimsAttr =
      IREE::TensorExt::getSparseIterationDimsAttr(forallOp);
  if (!sparseIterationDimsAttr) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute on ForallOp");
  }

  ArrayRef<int64_t> sparseIterationDims =
      sparseIterationDimsAttr.getSparseIterationDims();
  if (sparseIterationDims.size() != range.size()) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute to have the same size as "
        "the loop range");
  }

  IRRewriter rewriter(forallOp->getContext());
  rewriter.setInsertionPoint(forallOp);
  FailureOr<SmallVector<Range>> estimatedRanges =
      sparseOp.getEstimatedLoopRange(rewriter, sparseIterationDims, range);
  if (failed(estimatedRanges)) {
    return forallOp.emitError("getEstimatedLoopRange failed");
  }
  SmallVector<OpFoldResult> estimatedOffsets, estimatedSizes, estimatedStrides;
  estimatedOffsets.reserve(estimatedRanges->size());
  estimatedSizes.reserve(estimatedRanges->size());
  estimatedStrides.reserve(estimatedRanges->size());
  for (const Range &r : *estimatedRanges) {
    estimatedOffsets.push_back(r.offset);
    estimatedSizes.push_back(r.size);
    estimatedStrides.push_back(r.stride);
  }
  auto newForAllOp = scf::ForallOp::create(
      rewriter, forallOp.getLoc(), estimatedOffsets, estimatedSizes,
      estimatedStrides, /*outputs=*/ValueRange{}, /*mapping=*/std::nullopt,
      [&](OpBuilder &, Location, ValueRange) {});
  Region &newRegion = newForAllOp.getRegion();
  newRegion.takeBody(forallOp.getRegion());
  rewriter.eraseOp(forallOp);

  return success();
}

void TestSparseOpInterfaceMethodsPass::runOnOperation() {
  Operation *op = getOperation();

  SmallVector<scf::ForallOp> forallOps;
  SmallVector<IREE::TensorExt::SparseOpInterface> sparseInterfaceOps;
  op->walk([&](Operation *op) {
    if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
      forallOps.push_back(forallOp);
    }
    if (auto sparseOp = dyn_cast<IREE::TensorExt::SparseOpInterface>(op)) {
      sparseInterfaceOps.push_back(sparseOp);
    }
  });

  if (!llvm::hasSingleElement(forallOps)) {
    op->emitError("expected a single ForallOp");
    return signalPassFailure();
  }
  if (!llvm::hasSingleElement(sparseInterfaceOps)) {
    op->emitError("expected a single SparseOpInterface op");
    return signalPassFailure();
  }

  auto forallOp = *forallOps.begin();
  auto sparseOp = *sparseInterfaceOps.begin();

  if (testLowerLoopRange &&
      failed(testLowerLoopRangeImpl(forallOp, sparseOp))) {
    return signalPassFailure();
  } else if (testGetEstimatedLoopRange &&
             failed(testGetEstimatedLoopRangeImpl(forallOp, sparseOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::TensorExt
