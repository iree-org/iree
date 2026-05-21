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
    : impl::TestSparseOpInterfaceMethodsPassBase<
          TestSparseOpInterfaceMethodsPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::TensorExt::IREETensorExtDialect>();
  }
  void runOnOperation() override;
};

} // namespace

/// Inverse of the SparseLoopResolver
using ResolverToLoopInfo =
    llvm::MapVector<int64_t, int64_t>; // Map from resultDim to sparseLoopId
                                       // that dimension helps resolve.
static llvm::MapVector<IREE::TensorExt::SparseCastOpInterface,
                       ResolverToLoopInfo>
invertSparseLoopResolverInfo(ArrayRef<SparseRangeResolver> resolvers,
                             ArrayRef<int64_t> sparseLoops) {
  llvm::MapVector<IREE::TensorExt::SparseCastOpInterface, ResolverToLoopInfo>
      allResolversToLoopsInfo;
  for (auto [sparseLoopId, resolver] :
       llvm::zip_equal(sparseLoops, resolvers)) {
    if (!resolver.sparseOp) {
      continue;
    }
    ResolverToLoopInfo &resolverToLoopInfo =
        allResolversToLoopsInfo[resolver.sparseOp];
    resolverToLoopInfo[resolver.resultDim] = sparseLoopId;
  }
  for (auto [sparseOp, loopInfo] : allResolversToLoopsInfo) {
    llvm::sort(loopInfo);
  }
  return allResolversToLoopsInfo;
}

// Test the `lowerLoopRange` method of the SparseCastOpInterface. Replace the
// `forallOp` with nested `scf.for` operations that iterate over the sparse
// range.
static LogicalResult testLowerLoopRangeImpl(scf::ForallOp forallOp) {
  auto sparseIterationDimsAttr =
      IREE::TensorExt::getSparseIterationDimsAttr(forallOp);
  if (!sparseIterationDimsAttr) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute on ForallOp");
  }
  SetVector<int64_t> sparseIterationDimsSet;
  sparseIterationDimsSet.insert(
      sparseIterationDimsAttr.getSparseIterationDims().begin(),
      sparseIterationDimsAttr.getSparseIterationDims().end());
  SmallVector<Range> sparseRanges;
  sparseRanges.reserve(sparseIterationDimsSet.size());

  SmallVector<OpFoldResult> mixedOffsets = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedSizes = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStrides = forallOp.getMixedStep();
  for (auto [index, offset, size, stride] :
       llvm::enumerate(mixedOffsets, mixedSizes, mixedStrides)) {
    if (sparseIterationDimsSet.count(index)) {
      sparseRanges.emplace_back(Range{offset, size, stride});
    }
  }

  SmallVector<IREE::TensorExt::SparseRangeResolver> resolvers =
      IREE::TensorExt::getSparseRangeResolvers(sparseRanges);
  llvm::MapVector<int64_t, IREE::TensorExt::SparseRangeResolver> resolverMap;
  for (auto [sparseLoopId, resolver] :
       llvm::zip_equal(sparseIterationDimsSet.getArrayRef(), resolvers)) {
    resolverMap[sparseLoopId] = resolver;
  }
  llvm::MapVector<IREE::TensorExt::SparseCastOpInterface, ResolverToLoopInfo>
      allResolversToLoopsInfo = invertSparseLoopResolverInfo(
          resolvers, sparseIterationDimsSet.getArrayRef());

  IRRewriter rewriter(forallOp->getContext());
  rewriter.setInsertionPoint(forallOp);
  llvm::MapVector<int64_t, Value> resolvedLoopIvs;

  for (auto loopId : llvm::seq<int64_t>(0, forallOp.getRank())) {
    if (resolvedLoopIvs.count(loopId)) {
      continue;
    }

    if (!sparseIterationDimsSet.count(loopId)) {
      // Dense dimension, just create a normal scf.for.
      Value lb = getValueOrCreateConstantIndexOp(rewriter, forallOp.getLoc(),
                                                 mixedOffsets[loopId]);
      Value ub = getValueOrCreateConstantIndexOp(rewriter, forallOp.getLoc(),
                                                 mixedSizes[loopId]);
      Value step = getValueOrCreateConstantIndexOp(rewriter, forallOp.getLoc(),
                                                   mixedStrides[loopId]);
      auto newForOp =
          scf::ForOp::create(rewriter, forallOp.getLoc(), lb, ub, step);
      rewriter.setInsertionPointToStart(newForOp.getBody());
      resolvedLoopIvs[loopId] = newForOp.getInductionVar();
      continue;
    }

    // Check if there is a sparse resolver for this loop.
    SparseRangeResolver resolver = resolverMap[loopId];
    if (!resolver.sparseOp) {
      return forallOp.emitError("expected a sparse resolver for loop dim ")
             << loopId;
    }
    IREE::TensorExt::SparseCastOpInterface sparseOp = resolver.sparseOp;
    SmallVector<int64_t> sparseResolverDims = llvm::map_to_vector(
        allResolversToLoopsInfo[sparseOp], [](auto it) { return it.first; });
    SmallVector<int64_t> resolvedLoopIds = llvm::map_to_vector(
        allResolversToLoopsInfo[sparseOp], [](auto it) { return it.second; });
    SmallVector<Range> currSparseRanges =
        llvm::map_to_vector(resolvedLoopIds, [&](int64_t loopId) {
          return Range{mixedOffsets[loopId], mixedSizes[loopId],
                       mixedStrides[loopId]};
        });

    FailureOr<SmallVector<Value>> replacementIvs =
        sparseOp.lowerLoopRange(rewriter, sparseResolverDims, currSparseRanges);
    if (failed(replacementIvs)) {
      return failure();
    }
    for (auto [resolvedLoopId, iv] :
         llvm::zip_equal(resolvedLoopIds, replacementIvs.value())) {
      resolvedLoopIvs[resolvedLoopId] = iv;
    }
  }

  llvm::sort(resolvedLoopIvs, [](auto &a, auto &b) {
    return a.first < b.first; // Sort by loopId.
  });
  SmallVector<Value> replacementIvs =
      llvm::map_to_vector(resolvedLoopIvs, [](auto it) { return it.second; });

  Block *currBlock = rewriter.getInsertionBlock();
  Block *splitBlock = rewriter.splitBlock(
      currBlock, Block::iterator(*currBlock->getTerminator()));
  Block *origBlock = forallOp.getBody();
  rewriter.eraseOp(origBlock->getTerminator());
  rewriter.mergeBlocks(splitBlock, origBlock);
  rewriter.mergeBlocks(origBlock, currBlock, replacementIvs);
  rewriter.eraseOp(forallOp);

  return success();
}

// Test the `getEstimatedLoopRange` method of the SparseCastOpInterface.
// Replace the `forallOp` with a new `scf.forall` with the estimated loop range.
static LogicalResult testGetEstimatedLoopRangeImpl(scf::ForallOp forallOp) {

  auto sparseIterationDimsAttr =
      IREE::TensorExt::getSparseIterationDimsAttr(forallOp);
  if (!sparseIterationDimsAttr) {
    return forallOp.emitError(
        "expected sparse_iteration_dims attribute on ForallOp");
  }
  SetVector<int64_t> sparseIterationDimsSet;
  sparseIterationDimsSet.insert(
      sparseIterationDimsAttr.getSparseIterationDims().begin(),
      sparseIterationDimsAttr.getSparseIterationDims().end());
  SmallVector<Range> sparseRanges;
  sparseRanges.reserve(sparseIterationDimsSet.size());

  SmallVector<OpFoldResult> mixedOffsets = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> mixedSizes = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> mixedStrides = forallOp.getMixedStep();
  for (auto [index, offset, size, stride] :
       llvm::enumerate(mixedOffsets, mixedSizes, mixedStrides)) {
    if (sparseIterationDimsSet.count(index)) {
      sparseRanges.emplace_back(Range{offset, size, stride});
    }
  }

  SmallVector<IREE::TensorExt::SparseRangeResolver> resolvers =
      IREE::TensorExt::getSparseRangeResolvers(sparseRanges);
  llvm::MapVector<int64_t, IREE::TensorExt::SparseRangeResolver> resolverMap;
  for (auto [sparseLoopId, resolver] :
       llvm::zip_equal(sparseIterationDimsSet.getArrayRef(), resolvers)) {
    resolverMap[sparseLoopId] = resolver;
  }
  llvm::MapVector<IREE::TensorExt::SparseCastOpInterface, ResolverToLoopInfo>
      allResolversToLoopsInfo = invertSparseLoopResolverInfo(
          resolvers, sparseIterationDimsSet.getArrayRef());

  IRRewriter rewriter(forallOp->getContext());
  rewriter.setInsertionPoint(forallOp);
  llvm::MapVector<int64_t, Value> resolvedLoopIvs;
  SmallVector<OpFoldResult> estimatedOffsets = mixedOffsets,
                            estimatedSizes = mixedSizes,
                            estimatedStrides = mixedStrides;

  for (auto loopId : llvm::seq<int64_t>(0, forallOp.getRank())) {
    if (!sparseIterationDimsSet.count(loopId)) {
      continue;
    }

    // Check if there is a sparse resolver for this loop.
    SparseRangeResolver resolver = resolverMap[loopId];
    if (!resolver.sparseOp) {
      return forallOp.emitError("expected a sparse resolver for loop dim ")
             << loopId;
    }
    IREE::TensorExt::SparseCastOpInterface sparseOp = resolver.sparseOp;
    SmallVector<int64_t> sparseResolverDims = llvm::map_to_vector(
        allResolversToLoopsInfo[sparseOp], [](auto it) { return it.first; });
    SmallVector<int64_t> resolvedLoopIds = llvm::map_to_vector(
        allResolversToLoopsInfo[sparseOp], [](auto it) { return it.second; });
    SmallVector<Range> currSparseRanges =
        llvm::map_to_vector(resolvedLoopIds, [&](int64_t loopId) {
          return Range{mixedOffsets[loopId], mixedSizes[loopId],
                       mixedStrides[loopId]};
        });

    FailureOr<SmallVector<Range>> estimatedRanges =
        sparseOp.getEstimatedLoopRange(rewriter, sparseResolverDims,
                                       currSparseRanges);
    if (failed(estimatedRanges)) {
      return failure();
    }
    for (auto [resolvedLoopId, range] :
         llvm::zip_equal(resolvedLoopIds, estimatedRanges.value())) {
      estimatedOffsets[resolvedLoopId] = range.offset;
      estimatedSizes[resolvedLoopId] = range.size;
      estimatedStrides[resolvedLoopId] = range.stride;
      sparseIterationDimsSet.remove(resolvedLoopId);
    }
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

// Test the `resolveRange` method of the SparseCastOpInterface. Finds a
// `"test.range"` marker op whose operands are grouped in triples
// (offset, size, stride) — one triple per result dimension. Calls
// `resolveRange` and replaces the marker with `"test.resolved_range"`.
static LogicalResult testResolveRangeImpl(Operation *funcOp) {
  SmallVector<IREE::TensorExt::SparseCastOpInterface> sparseOps;
  SmallVector<Operation *> rangeOps;
  funcOp->walk([&](Operation *op) {
    if (auto sparseOp = dyn_cast<IREE::TensorExt::SparseCastOpInterface>(op)) {
      sparseOps.push_back(sparseOp);
    }
    if (op->getName().getStringRef() == "test.range") {
      rangeOps.push_back(op);
    }
  });
  if (sparseOps.size() != 1) {
    return funcOp->emitError("expected exactly one SparseCastOpInterface op");
  }
  if (rangeOps.size() != 1) {
    return funcOp->emitError("expected exactly one test.range op");
  }
  IREE::TensorExt::SparseCastOpInterface sparseOp = sparseOps[0];
  Operation *rangeOp = rangeOps[0];

  // Operands are grouped as (offset0, size0, stride0, offset1, size1, ...).
  unsigned numOperands = rangeOp->getNumOperands();
  if (numOperands % 3 != 0) {
    return rangeOp->emitError(
        "test.range operands must be a multiple of 3 (offset, size, stride)");
  }

  SmallVector<Range> allRanges;
  for (unsigned i = 0; i < numOperands; i += 3) {
    allRanges.push_back(Range{rangeOp->getOperand(i),
                              rangeOp->getOperand(i + 1),
                              rangeOp->getOperand(i + 2)});
  }

  llvm::BitVector inBounds(allRanges.size(), true);

  IRRewriter rewriter(funcOp->getContext());
  rewriter.setInsertionPoint(rangeOp);
  Location loc = rangeOp->getLoc();

  FailureOr<SmallVector<Range>> resolvedRanges =
      sparseOp.resolveRange(rewriter, allRanges, inBounds, std::nullopt);
  if (failed(resolvedRanges)) {
    return failure();
  }

  // Materialize the resolved ranges as values so FileCheck can verify them.
  SmallVector<Value> allVals;
  for (auto &range : resolvedRanges.value()) {
    allVals.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, range.offset));
    allVals.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
    allVals.push_back(
        getValueOrCreateConstantIndexOp(rewriter, loc, range.stride));
  }
  OperationState state(loc, "test.resolved_range");
  state.addOperands(allVals);
  rewriter.create(state);

  rewriter.eraseOp(rangeOp);
  return success();
}

// Test the `getDistributionInfoForSparseDimensions` method. For each
// SparseCastOpInterface op, emit a `test.distribution_info` marker with the
// result BitVector encoded as a dense bool array attribute.
static LogicalResult testGetDistributionInfoImpl(Operation *funcOp) {
  SmallVector<IREE::TensorExt::SparseCastOpInterface> sparseOps;
  funcOp->walk([&](IREE::TensorExt::SparseCastOpInterface op) {
    sparseOps.push_back(op);
  });
  if (sparseOps.empty()) {
    return funcOp->emitError("expected at least one SparseCastOpInterface op");
  }

  IRRewriter rewriter(funcOp->getContext());
  for (auto sparseOp : sparseOps) {
    llvm::BitVector distInfo =
        sparseOp.getDistributionInfoForSparseDimensions();

    SmallVector<bool> distributable;
    for (unsigned i = 0; i < distInfo.size(); ++i) {
      distributable.push_back(distInfo.test(i));
    }

    rewriter.setInsertionPointAfter(sparseOp);
    OperationState state(sparseOp->getLoc(), "test.distribution_info");
    state.addAttribute("distributable",
                       rewriter.getDenseBoolArrayAttr(distributable));
    rewriter.create(state);
  }
  return success();
}

void TestSparseOpInterfaceMethodsPass::runOnOperation() {
  Operation *op = getOperation();

  // resolveRange and getDistributionInfo use their own marker ops, not forall.
  if (testResolveRange) {
    if (failed(testResolveRangeImpl(op))) {
      return signalPassFailure();
    }
    return;
  }
  if (testGetDistributionInfo) {
    if (failed(testGetDistributionInfoImpl(op))) {
      return signalPassFailure();
    }
    return;
  }

  SmallVector<scf::ForallOp> forallOps;
  op->walk([&](scf::ForallOp forallOp) { forallOps.push_back(forallOp); });

  if (!llvm::hasSingleElement(forallOps)) {
    op->emitError("expected a single ForallOp");
    return signalPassFailure();
  }

  auto forallOp = *forallOps.begin();

  if (testLowerLoopRange && failed(testLowerLoopRangeImpl(forallOp))) {
    return signalPassFailure();
  } else if (testGetEstimatedLoopRange &&
             failed(testGetEstimatedLoopRangeImpl(forallOp))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::IREE::TensorExt
