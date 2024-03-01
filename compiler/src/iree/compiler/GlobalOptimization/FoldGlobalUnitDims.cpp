// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

//===----------------------------------------------------------------------===//
// Util functions
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
  // Just bail in the special case for tensor of all unit dims.
  if (newShape.empty()) {
    return success();
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

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

class FoldGlobalUnitDimsPass
    : public FoldGlobalUnitDimsBase<FoldGlobalUnitDimsPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    explorer.initialize();
    IRRewriter rewriter(context);
    SymbolTable moduleSymbols(moduleOp);

    // Rewrite globals without unit dims.
    explorer.forEachGlobal([&](const Explorer::GlobalInfo *globalInfo) {
      IREE::Util::GlobalOpInterface global = globalInfo->op;
      auto tensorType = dyn_cast<RankedTensorType>(global.getGlobalType());
      if (!tensorType || !global.isGlobalPrivate() ||
          !global.isGlobalMutable()) {
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

    // Try to fold reshapes introduced by folding unit dims.
    {
      RewritePatternSet patterns(context);
      tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
      tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
      if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createFoldGlobalUnitDimsPass() {
  return std::make_unique<FoldGlobalUnitDimsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
