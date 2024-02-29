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
  auto newGlobalType =
      RankedTensorType::get(newShape, globalType.getElementType());
  std::optional<TypedAttr> newInitialValue = std::nullopt;
  if (auto uninitializedAttr =
          llvm::dyn_cast_or_null<IREE::Util::UninitializedAttr>(
              global.getGlobalInitialValue())) {
    newInitialValue = IREE::Util::UninitializedAttr::get(rewriter.getContext(),
                                                         newGlobalType);
  }
  // TODO: Handle non-uninitialized cases.
  else {
    return success();
  }
  rewriter.setInsertionPoint(global);
  auto newGlobalOp = rewriter.create(
      global->getLoc(), global->getName().getIdentifier(),
      global->getOperands(), global->getResultTypes(), global->getAttrs());
  auto newGlobal = cast<IREE::Util::GlobalOpInterface>(newGlobalOp);
  newGlobal.setGlobalType(newGlobalType);
  newGlobal.setGlobalInliningPolicy(global.getGlobalInliningPolicy());
  newGlobal.setGlobalMutable(global.isGlobalMutable());
  if (newInitialValue.has_value())
    newGlobal.setGlobalInitialValue(newInitialValue.value());
  moduleSymbols.insert(newGlobal);
  SymbolTable::setSymbolVisibility(newGlobal,
                                   SymbolTable::getSymbolVisibility(global));

  // Rewrite loads and stores to use the new global.
  auto expandShapeReInds =
      getReassociationIndicesForReshape(globalType, newGlobalType);
  if (!expandShapeReInds) {
    return failure();
  }

  for (auto load : loadOps) {
    rewriter.setInsertionPoint(load);
    auto newOp =
        rewriter.create(load->getLoc(), load->getName().getIdentifier(),
                        load->getOperands(), {newGlobalType}, load->getAttrs());
    auto newLoad = dyn_cast<IREE::Util::GlobalLoadOpInterface>(newOp);
    newLoad.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
    newLoad.setGlobalImmutable(load.isGlobalImmutable());
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        load, globalType, newLoad->getResult(0), expandShapeReInds.value());
  }
  for (auto store : storeOps) {
    rewriter.setInsertionPoint(store);
    Value collapse = rewriter.create<tensor::CollapseShapeOp>(
        store.getLoc(), newGlobalType, store->getOperand(0),
        expandShapeReInds.value());
    auto newOp = rewriter.create(
        store->getLoc(), store->getName().getIdentifier(), store->getOperands(),
        store->getResultTypes(), store->getAttrs());
    auto newStore = dyn_cast<IREE::Util::GlobalStoreOpInterface>(newOp);
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
    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    explorer.initialize();
    IRRewriter rewriter(&getContext());
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
      MLIRContext *context = &getContext();
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
