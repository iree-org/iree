// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

static LogicalResult foldUnitDimsOnGlobal(IRRewriter &rewriter,
                                          IREE::Util::GlobalOp global,
                                          SmallVector<Operation *> loadStoreOps,
                                          SymbolTable moduleSymbols) {
  // Create a new transformed GlobalOp.
  SmallVector<int64_t> newShape;
  auto globalType = cast<RankedTensorType>(global.getType());
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
  if (auto initialValue = global.getInitialValue()) {
    if (auto uninitializedAttr =
            dyn_cast<IREE::Util::UninitializedAttr>(initialValue.value())) {
      newInitialValue = IREE::Util::UninitializedAttr::get(
          rewriter.getContext(), newGlobalType);
    }
    // TODO: Handle non-uninitialized cases.
    else {
      return success();
    }
  }
  StringRef newGlobalName(global.getGlobalName());
  rewriter.setInsertionPoint(global);
  auto newGlobal = rewriter.create<IREE::Util::GlobalOp>(
      global->getLoc(), newGlobalName, global.getIsMutable(), newGlobalType,
      newInitialValue);
  newGlobal.setInliningPolicyAttr(global.getInliningPolicyAttr());
  moduleSymbols.insert(newGlobal);
  SymbolTable::setSymbolVisibility(newGlobal,
                                   SymbolTable::getSymbolVisibility(global));

  // Rewrite loads and stores to use the new global.
  auto expandShapeReInds =
      getReassociationIndicesForReshape(globalType, newGlobalType);
  if (!expandShapeReInds) {
    return failure();
  }
  for (auto loadOrStore : loadStoreOps) {
    rewriter.setInsertionPoint(loadOrStore);
    if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(loadOrStore)) {
      auto newOp = rewriter.create(
          load->getLoc(), load->getName().getIdentifier(), load->getOperands(),
          {newGlobalType}, load->getAttrs());
      auto newLoad = dyn_cast<IREE::Util::GlobalLoadOpInterface>(newOp);
      newLoad.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
      newLoad.setGlobalImmutable(load.isGlobalImmutable());
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          load, globalType, newLoad->getResult(0), expandShapeReInds.value());
    } else if (auto store =
                   dyn_cast<IREE::Util::GlobalStoreOpInterface>(loadOrStore)) {
      Value collapse = rewriter.create<tensor::CollapseShapeOp>(
          store.getLoc(), newGlobalType, store->getOperand(0),
          expandShapeReInds.value());
      auto newOp = rewriter.create(
          store->getLoc(), store->getName().getIdentifier(),
          store->getOperands(), store->getResultTypes(), store->getAttrs());
      auto newStore = dyn_cast<IREE::Util::GlobalStoreOpInterface>(newOp);
      newStore.setGlobalAttr(FlatSymbolRefAttr::get(newGlobal.getGlobalName()));
      newStore.setStoredGlobalValue(collapse);
      rewriter.eraseOp(store);
    } else {
      return failure();
    }
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
    DenseMap<StringRef, SmallVector<Operation *>> loadStoreMap;
    auto addToLoadStoreMap = [&](StringRef name, Operation *loadStoreOp) {
      if (loadStoreMap.contains(name)) {
        loadStoreMap[name].push_back(loadStoreOp);
      } else {
        SmallVector<Operation *> loadStores(1, loadStoreOp);
        loadStoreMap.insert(std::make_pair(name, loadStores));
      }
    };
    moduleOp.walk([&](Operation *op) {
      if (auto load = dyn_cast<IREE::Util::GlobalLoadOpInterface>(op)) {
        addToLoadStoreMap(load.getGlobalName(), op);
      } else if (auto store =
                     dyn_cast<IREE::Util::GlobalStoreOpInterface>(op)) {
        addToLoadStoreMap(store.getGlobalName(), op);
      }
    });
    IRRewriter rewriter(&getContext());
    SmallVector<IREE::Util::GlobalOp> foldableGlobals;
    for (auto global : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      if (!global.getIsMutable())
        continue;
      auto tensorType = dyn_cast<RankedTensorType>(global.getType());
      if (!tensorType)
        continue;
      if (llvm::any_of(tensorType.getShape(),
                       [](int64_t size) { return size == 1; })) {
        foldableGlobals.push_back(global);
      }
    }
    SymbolTable moduleSymbols(moduleOp);
    for (auto global : foldableGlobals) {
      if (failed(foldUnitDimsOnGlobal(rewriter, global,
                                      loadStoreMap[global.getGlobalName()],
                                      moduleSymbols))) {
        return signalPassFailure();
      }
    }

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
