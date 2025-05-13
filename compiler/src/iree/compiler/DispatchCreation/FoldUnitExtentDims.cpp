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
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-fold-unit-extent-dims"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSPASS
#define GEN_PASS_DEF_FOLDUNITEXTENTDIMSFORFUNCPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Attention Pattern
//===----------------------------------------------------------------------===//

namespace {
struct FoldAttentionMaskUnitDim final
    : public OpRewritePattern<IREE::LinalgExt::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IREE::LinalgExt::AttentionOp attentionOp,
                                PatternRewriter &rewriter) const override {
    TypedValue<ShapedType> mask = attentionOp.getMask();
    if (!mask) {
      return failure();
    }
    auto detail = IREE::LinalgExt::AttentionOpDetail::get(
                      attentionOp.getQueryMap(), attentionOp.getKeyMap(),
                      attentionOp.getValueMap(), attentionOp.getOutputMap())
                      .value();
    auto mDims = detail.getMDims();
    auto maskShape = mask.getType().getShape();
    AffineMap maskMap = attentionOp.getMaskMap().value();
    if (!mDims.size()) {
      return failure();
    }

    llvm::DenseMap<int64_t, int64_t> loopToMaskDim;
    for (auto [i, result] : llvm::enumerate(maskMap.getResults())) {
      loopToMaskDim[cast<AffineDimExpr>(result).getPosition()] = i;
    }

    // Find unit M dims in the mask map.
    llvm::DenseSet<int64_t> loopsToDrop;
    for (int64_t dim : mDims) {
      auto it = loopToMaskDim.find(dim);
      if (it != loopToMaskDim.end() && maskShape[it->second] == 1) {
        loopsToDrop.insert(dim);
      }
    }
    if (loopsToDrop.size() == 0) {
      return rewriter.notifyMatchFailure(attentionOp, "no unit M dim");
    }

    // Compute the reassociation to remove unit dims and the new shape.
    ReassociationIndices indices;
    SmallVector<ReassociationIndices> reassoc;
    SmallVector<int64_t> resultShape;
    llvm::SmallBitVector resultsToDrop(maskShape.size());
    for (auto [i, size] : llvm::enumerate(maskShape)) {
      int64_t loop = maskMap.getDimPosition(i);
      if (!loopsToDrop.contains(loop)) {
        resultShape.push_back(size);
        if (indices.size())
          reassoc.emplace_back(std::move(indices));
      } else {
        resultsToDrop.set(i);
      }
      indices.push_back(i);
    }
    if (indices.size()) {
      reassoc.emplace_back(std::move(indices));
    }

    auto loc = attentionOp.getLoc();
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, mask.getType().clone(resultShape), mask, reassoc);

    rewriter.modifyOpInPlace(attentionOp, [&]() {
      attentionOp.getMaskMutable().assign(collapseOp);
      AffineMap newMap = maskMap.dropResults(resultsToDrop);
      auto newMaps = attentionOp.getIndexingMapsArray();
      newMaps[attentionOp.getMaskMutable().begin()->getOperandNumber()] =
          newMap;
      attentionOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
    });
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass helpers
//===----------------------------------------------------------------------===//

static void
populatefoldUnitDimsPatterns(RewritePatternSet &foldUnitDimsPatterns) {
  linalg::ControlDropUnitDims options;
  auto defaultFn = options.controlFn;

  options.controlFn = [=](Operation *op) {
    // Ignore operations already in dispatches.
    if (!IREE::Flow::isNonNullAndOutsideDispatch(op)) {
      return SmallVector<unsigned>{};
    }
    if (isa<IREE::LinalgExt::LinalgExtOp>(op)) {
      return IREE::LinalgExt::defaultControlDropUnitDims(op);
    }
    return defaultFn(op);
  };

  linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
  IREE::LinalgExt::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns,
                                                      options);
  linalg::populateMoveInitOperandsToInputPattern(foldUnitDimsPatterns);
  foldUnitDimsPatterns.insert<FoldAttentionMaskUnitDim>(
      foldUnitDimsPatterns.getContext());
}

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

struct FoldUnitExtentDimsForFuncPass final
    : public impl::FoldUnitExtentDimsForFuncPassBase<
          FoldUnitExtentDimsForFuncPass> {
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
  populatefoldUnitDimsPatterns(foldUnitDimsPatterns);
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

void FoldUnitExtentDimsForFuncPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet foldUnitDimsPatterns(context);
  populatefoldUnitDimsPatterns(foldUnitDimsPatterns);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
