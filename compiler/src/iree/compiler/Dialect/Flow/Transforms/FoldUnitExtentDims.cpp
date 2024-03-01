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

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

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
  // TODO: Handle non-uninitialized cases.
  auto uninitializedAttr =
      llvm::dyn_cast_if_present<IREE::Util::UninitializedAttr>(initialValue);
  if (initialValue && !uninitializedAttr)
    return success();
  TypedAttr newInitialValue;
  if (initialValue) {
    newInitialValue = IREE::Util::UninitializedAttr::get(rewriter.getContext(),
                                                         newGlobalType);
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
struct FoldUnitExtentDimsPass
    : public FoldUnitExtentDimsBase<FoldUnitExtentDimsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void FoldUnitExtentDimsPass::runOnOperation() {
  auto moduleOp = getOperation();
  MLIRContext *context = &getContext();
  Explorer explorer(moduleOp, TraversalAction::RECURSE);
  explorer.initialize();
  IRRewriter rewriter(context);
  SymbolTable moduleSymbols(moduleOp);

  // Fold unit dims of GlobalOpInterface ops.
  explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
    IREE::Util::GlobalOpInterface global = globalInfo->op;
    auto tensorType = dyn_cast<RankedTensorType>(global.getGlobalType());
    if (!tensorType || !global.isGlobalPrivate() || !global.isGlobalMutable()) {
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

  // Fold unit dims on other operations.
  RewritePatternSet foldUnitDimsPatterns(context);
  linalg::ControlDropUnitDims options;
  auto defaultFn = options.controlFn;
  options.controlFn = [&](Operation *op) {
    // Ignore operations already in dispatches.
    if (!isNonNullAndOutsideDispatch(op)) {
      return SmallVector<unsigned>{};
    }
    return defaultFn(op);
  };
  linalg::populateFoldUnitExtentDimsPatterns(foldUnitDimsPatterns, options);
  linalg::populateMoveInitOperandsToInputPattern(foldUnitDimsPatterns);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createFoldUnitExtentDimsPass() {
  return std::make_unique<FoldUnitExtentDimsPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
