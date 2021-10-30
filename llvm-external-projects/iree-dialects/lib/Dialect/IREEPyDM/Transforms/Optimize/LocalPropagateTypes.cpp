// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../PassDetail.h"
#include "iree-dialects/Dialect/IREEPyDM/IR/Ops.h"
#include "iree-dialects/Dialect/IREEPyDM/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_pydm;

namespace pydm_d = mlir::iree_pydm;

namespace {

struct LocalPropagateTypesPass
    : public LocalPropagateTypesBase<LocalPropagateTypesPass> {
  void runOnOperation() override {
    // Prepare selected canonicalization patterns.
    auto *context = &getContext();
    RewritePatternSet canonicalizePatterns(context);
    ApplyBinaryOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    ApplyCompareOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    AsBoolOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    BoxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    DynamicBinaryPromoteOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                        context);
    PromoteNumericOp::getCanonicalizationPatterns(canonicalizePatterns,
                                                  context);
    UnboxOp::getCanonicalizationPatterns(canonicalizePatterns, context);
    FrozenRewritePatternSet frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    GreedyRewriteConfig rewriterConfig;
    rewriterConfig.enableRegionSimplification = false;

    do {
      DominanceInfo domInfo(getOperation());
      changed = false;
      forwardStoreToLoad();
      simplifyFreeVariables(domInfo);
      applyPatternsAndFoldGreedily(getOperation(), frozenCanonicalizePatterns,
                                   rewriterConfig);
    } while (changed);
  }

  void forwardStoreToLoad() {
    // Really simple: Just forwarding within each block for now.
    for (auto &block : getOperation().getBody().getBlocks()) {
      forwardStoreToLoadInBlock(block);
    }
  }

  void forwardStoreToLoadInBlock(Block &block) {
    DenseMap<Value, Value> freeVariableMap;
    for (Operation &op : block) {
      if (auto storeOp = llvm::dyn_cast<StoreVarOp>(op)) {
        Value &currentValue = freeVariableMap[storeOp.var()];
        currentValue = storeOp.value();
      } else if (auto loadOp = llvm::dyn_cast<LoadVarOp>(op)) {
        Value &currentValue = freeVariableMap[loadOp.var()];
        if (currentValue) {
          loadOp.getResult().replaceAllUsesWith(currentValue);
          changed = true;
        } else {
          currentValue = loadOp.getResult();
        }
      }
    }
  }

  void simplifyFreeVariables(DominanceInfo &domInfo) {
    getOperation()->walk([&](AllocFreeVarOp freeVarOp) {
      specializeFreeVar(freeVarOp);
    });
  }

  void specializeFreeVar(AllocFreeVarOp freeVarOp) {
    OpBuilder builder(freeVarOp);
    // Analyze.
    Type storeType;
    Type objectType = builder.getType<ObjectType>(nullptr);
    SmallVector<LoadVarOp> unspecializedLoadVarOps;
    for (auto &use : freeVarOp->getUses()) {
      auto useOp = use.getOwner();
      if (auto storeOp = dyn_cast<StoreVarOp>(useOp)) {
        Value storeValue = storeOp.value();
        if (storeValue.getType() == objectType) {
          return;
        }
        if (storeType && storeValue.getType() != storeType) {
          return;
        }
        storeType = storeValue.getType();
      } else if (auto loadOp = dyn_cast<LoadVarOp>(useOp)) {
        Type loadType = loadOp.getResult().getType();
        if (loadType == objectType) {
          unspecializedLoadVarOps.push_back(loadOp);
        }
      } else {
        return;
      }
    }
    if (!storeType || unspecializedLoadVarOps.empty()) return;

    // Update.
    for (auto loadVarOp : unspecializedLoadVarOps) {
      auto originalType = loadVarOp.getResult().getType();
      builder.setInsertionPointAfter(loadVarOp);
      refineType(loadVarOp.getLoc(), loadVarOp.getResult(), storeType, builder);
    }
    changed = true;
  }

  void refineType(Location loc, Value value, Type toType, OpBuilder &builder) {
    auto originalType = value.getType();
    value.setType(toType);
    auto infoCast = builder.create<StaticInfoCastOp>(loc, originalType, value);
    value.replaceUsesWithIf(infoCast.getResult(), [&](OpOperand &operand) {
      if (operand.getOwner() == infoCast) return false;
      return !isRefinable(operand);
    });
  }

  bool isRefinable(OpOperand &operand) {
    // This is a hack. Assuming this pass continues to exist, this should
    // minimally be a query on an interface.
    Operation *op = operand.getOwner();
    if (llvm::isa<DynamicBinaryPromoteOp>(op)) {
      return true;
    }
    return false;
  }

  bool changed = false;
};

}  // namespace

std::unique_ptr<OperationPass<pydm_d::FuncOp>>
mlir::iree_pydm::createLocalPropagateTypesPass() {
  return std::make_unique<LocalPropagateTypesPass>();
}
