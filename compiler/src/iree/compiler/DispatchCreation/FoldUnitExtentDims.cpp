// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldUnitExtentDims.cpp - Pass to fold unit extent dims of tensors -===//
//
// Light weight wrapper to call the patterns to fold unit extent dims with
// IREE control.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fold-unit-extent-dims"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Simplify collapse_shape(expand_shape) by removing unneeded unit dimensions
/// that get expanded and subsequently collapsed.
struct DropUnitDimsFromCollapseOfExpand
    : OpRewritePattern<tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    auto expandOp = collapseOp.getSrc().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
      return failure();
    }

    const auto collapseReassoc = collapseOp.getReassociationIndices();
    ArrayRef<int64_t> interShape = expandOp.getType().getShape();
    ArrayRef<int64_t> outShape = collapseOp.getType().getShape();
    SmallVector<int64_t> interToOutMap(expandOp.getType().getRank());
    llvm::SmallDenseSet<int64_t> toDrop;
    for (const auto &[outDim, indicies] : llvm::enumerate(collapseReassoc)) {
      for (auto [innerIdx, inDim] : llvm::enumerate(indicies)) {
        // Can't drop this dim if it isnt statically 1 or if it isn't being
        // combined with any other dimensions.
        if (indicies.size() == 1 || interShape[inDim] != 1) {
          continue;
        }

        // If we are collapsing multiple unit dims together, at least 1 must be
        // kept (prefer the first).
        if (outShape[outDim] == 1 && innerIdx != 0) {
          continue;
        }
        toDrop.insert(inDim);
      }
    }

    const auto expandReassoc = expandOp.getReassociationIndices();
    for (const auto &[inDim, indicies] : llvm::enumerate(expandReassoc)) {
      // Can't drop unit dim if it isn't from an expansion.
      if (indicies.size() == 1) {
        toDrop.erase(indicies[0]);
      }
    }

    if (toDrop.empty()) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "Didn't find any unit dims to drop");
    }

    SmallVector<int64_t> newInterShape;
    newInterShape.reserve(interShape.size() - toDrop.size());
    for (auto [idx, length] : llvm::enumerate(interShape)) {
      if (!toDrop.contains(idx)) {
        newInterShape.push_back(length);
      }
    }

    /// Returns true if new `ReassociationIndices` were appended to `reassoc`.
    auto appendDroppedReassocation =
        [&toDrop](SmallVector<ReassociationIndices, 4> &reassoc, int64_t start,
                  int64_t count, int64_t origStart) {
          reassoc.emplace_back();
          auto &indicies = reassoc.back();
          indicies.reserve(count);
          int64_t dim = start;
          for (int64_t idx : llvm::seq<int64_t>(origStart, origStart + count)) {
            if (!toDrop.contains(idx)) {
              indicies.push_back(dim++);
            }
          }

          // All indicies have been dropped.
          if (indicies.empty()) {
            reassoc.pop_back();
            return false;
          }
          return true;
        };

    auto dropOutputOfr = [&toDrop](const SmallVector<OpFoldResult> &sizes) {
      return llvm::map_to_vector(
          llvm::make_filter_range(
              llvm::enumerate(sizes),
              [&toDrop](auto pair) { return !toDrop.contains(pair.index()); }),
          [](auto pair) -> OpFoldResult { return pair.value(); });
    };

    auto isIdentityReassociation = [](ArrayRef<ReassociationIndices> reassoc) {
      return llvm::all_of(reassoc,
                          [](auto &indices) { return indices.size() == 1; });
    };

    SmallVector<ReassociationIndices, 4> newCollapseReassoc;
    int64_t collapsedDim = 0;
    for (auto dim : llvm::seq<int64_t>(0, outShape.size())) {
      bool changed = appendDroppedReassocation(newCollapseReassoc, collapsedDim,
                                               collapseReassoc[dim].size(),
                                               collapseReassoc[dim].front());
      if (changed) {
        collapsedDim += newCollapseReassoc.back().size();
      }
    }

    SmallVector<ReassociationIndices, 4> newExpandReassoc;
    ArrayRef<int64_t> srcShape = expandOp.getSrcType().getShape();
    int64_t expandedDim = 0;
    for (auto dim : llvm::seq<int64_t>(0, srcShape.size())) {
      bool changed = appendDroppedReassocation(newExpandReassoc, expandedDim,
                                               expandReassoc[dim].size(),
                                               expandReassoc[dim].front());
      if (changed) {
        expandedDim += newExpandReassoc.back().size();
      }
    }

    auto outputSizes = getMixedValues(expandOp.getStaticOutputShape(),
                                      expandOp.getOutputShape(), rewriter);
    Value newExpanded = expandOp.getSrc();
    if (!isIdentityReassociation(newExpandReassoc)) {
      newExpanded = rewriter.create<tensor::ExpandShapeOp>(
          expandOp.getLoc(),
          RankedTensorType::get(newInterShape,
                                expandOp.getType().getElementType()),
          expandOp.getSrc(), newExpandReassoc, dropOutputOfr(outputSizes));
    }

    Value newCollapsed = newExpanded;
    if (!isIdentityReassociation(newCollapseReassoc)) {
      newCollapsed = rewriter.create<tensor::CollapseShapeOp>(
          collapseOp.getLoc(), collapseOp.getType(), newExpanded,
          newCollapseReassoc);
    }
    rewriter.replaceOp(collapseOp, newCollapsed);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

static LogicalResult
foldUnitDimsOnGlobal(IRRewriter &rewriter, IREE::Util::GlobalOpInterface global,
                     SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps,
                     SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps,
                     SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  SmallVector<int64_t> newShape;
  auto globalType = cast<RankedTensorType>(global.getGlobalType());
  for (auto size : globalType.getShape()) {
    if (size != 1) {
      newShape.push_back(size);
    }
  }
  auto newGlobalType = globalType.clone(newShape);
  auto initialValue = global.getGlobalInitialValue();
  if (!initialValue)
    return success();
  // TODO: Handle other cases
  auto newInitialValue =
      llvm::TypeSwitch<Attribute, Attribute>(initialValue)
          .Case<IREE::Util::UninitializedAttr>([&](Attribute) {
            return IREE::Util::UninitializedAttr::get(rewriter.getContext(),
                                                      newGlobalType);
          })
          .Case<IREE::Stream::NamedParameterAttr>(
              [&](IREE::Stream::NamedParameterAttr attr) {
                return IREE::Stream::NamedParameterAttr::get(
                    rewriter.getContext(), newGlobalType, attr.getScope(),
                    attr.getKey(), attr.getConfig());
              })
          .Default([&](Attribute) { return nullptr; });
  if (!newInitialValue) {
    return success();
  }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(global);
  auto newGlobal =
      clone(rewriter, global, global->getResultTypes(), global->getOperands());
  newGlobal.setGlobalType(newGlobalType);
  newGlobal.setGlobalInitialValue(newInitialValue);

  // Rewrite loads and stores to use the new global.
  auto expandShapeReInds =
      getReassociationIndicesForReshape(globalType, newGlobalType);
  if (!expandShapeReInds) {
    return failure();
  }

  for (auto load : loadOps) {
    rewriter.setInsertionPoint(load);
    auto newLoad = clone(rewriter, load, {newGlobalType}, load->getOperands());
    newLoad.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        load, globalType, newLoad->getResult(0), expandShapeReInds.value());
  }
  for (auto store : storeOps) {
    rewriter.setInsertionPoint(store);
    Value collapse = rewriter.create<tensor::CollapseShapeOp>(
        store.getLoc(), newGlobalType, store->getOperand(0),
        expandShapeReInds.value());
    auto newStore =
        clone(rewriter, store, store->getResultTypes(), store->getOperands());
    newStore.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
    newStore.setStoredGlobalValue(collapse);
    rewriter.eraseOp(store);
  }
  rewriter.eraseOp(global);
  return success();
}

namespace {
struct FoldUnitExtentDimsPass final
    : public impl::FoldUnitExtentDimsPassBase<FoldUnitExtentDimsPass> {
  void runOnOperation() override;
};
} // namespace

void FoldUnitExtentDimsPass::runOnOperation() {
  auto moduleOp = getOperation();
  MLIRContext *context = &getContext();

  SymbolTable moduleSymbols(moduleOp);
  Explorer explorer(moduleOp, TraversalAction::RECURSE);
  explorer.initialize();
  IRRewriter rewriter(context);

  // Fold unit dims of GlobalOpInterface ops.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    IREE::Util::GlobalOpInterface global = globalInfo->op;
    auto tensorType = dyn_cast<RankedTensorType>(global.getGlobalType());
    if (!tensorType || !global.isGlobalPrivate()) {
      return;
    }
    if (llvm::none_of(tensorType.getShape(),
                      [](int64_t size) { return size == 1; })) {
      return;
    }
    SmallVector<IREE::Util::GlobalLoadOpInterface> loadOps =
        llvm::to_vector(globalInfo->getLoads());
    SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps =
        llvm::to_vector(globalInfo->getStores());
    if (failed(foldUnitDimsOnGlobal(rewriter, global, loadOps, storeOps,
                                    moduleSymbols))) {
      return signalPassFailure();
    }
  });

  RewritePatternSet foldUnitDimsPatterns(context);
  linalg::ControlDropUnitDims options;
  auto defaultFn = options.controlFn;
  options.controlFn = [&](Operation *op) {
    // Ignore operations already in dispatches.
    if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
      return SmallVector<unsigned>{};
    }
    return defaultFn(op);
  };
  linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
  linalg::populateMoveInitOperandsToInputPattern(foldUnitDimsPatterns);
  foldUnitDimsPatterns.insert<DropUnitDimsFromCollapseOfExpand>(context);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
