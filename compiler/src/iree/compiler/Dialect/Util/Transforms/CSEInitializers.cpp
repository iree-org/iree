// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "iree-util-cse-initializer"
namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_CSEINITIALIZERSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"
namespace {
class CSEInitializersPass
    : public impl::CSEInitializersPassBase<CSEInitializersPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    GlobalTable globalTable(moduleOp);
    globalTable.rebuild();

    SmallVector<GlobalStoreOpInterface> candidateGlobalStores;
    for (auto callableOp : moduleOp.getOps<CallableOpInterface>()) {
      auto *region = callableOp.getCallableRegion();
      if (!region) {
        continue;
      }
      for (auto &block : *region) {
        for (auto storeOp :
             block.getOps<IREE::Util::GlobalStoreOpInterface>()) {
          auto &global = globalTable.lookup(storeOp.getGlobalName());
          if (!global.isIndirect && global.onlyInitialized) {
            candidateGlobalStores.push_back(storeOp);
          }
        }
      }
    }

    if (candidateGlobalStores.empty()) {
      return; // nothing to do
    }

    BackwardSliceOptions options;
    options.inclusive = true;
    options.filter = [](Operation *op) {
      if (auto loadOp =
              dyn_cast_or_null<IREE::Util::GlobalLoadOpInterface>(op)) {
        if (loadOp.isGlobalImmutable()) {
          return true;
        }
        return false;
      }
      if (auto memOpInterface = dyn_cast_or_null<MemoryEffectOpInterface>(op)) {
        if (memOpInterface.hasEffect<mlir::MemoryEffects::Read>() ||
            memOpInterface.hasEffect<mlir::MemoryEffects::Write>()) {
          return false;
        }
      }
      return true;
    };

    SmallVector<std::pair<GlobalStoreOpInterface, SetVector<Operation *>>>
        globalToSliceMap;
    for (auto &storeOp : candidateGlobalStores) {
      SetVector<Operation *> slice;

      LogicalResult result = getBackwardSlice(
          storeOp.getStoredGlobalValue().getDefiningOp(), &slice, options);
      if (failed(result)) {
        return signalPassFailure();
      }

      globalToSliceMap.push_back(std::make_pair(storeOp, std::move(slice)));
    }

    BitVector storeOpBitVector(globalToSliceMap.size());
    llvm::EquivalenceClasses<GlobalStoreOpInterface> ec;
    for (int i = 0; i < globalToSliceMap.size(); ++i) {
      if (storeOpBitVector[i]) {
        continue;
      }
      for (int j = i + 1; j < globalToSliceMap.size(); ++j) {
        if (storeOpBitVector[j]) {
          continue;
        }

        if (globalToSliceMap[i].second.size() !=
            globalToSliceMap[j].second.size()) {
          continue;
        }

        OperationEquivalenceCache cache(moduleOp.getContext());
        auto mapping = cache.acquireMapping();
        bool isEquivalent = true;
        for (int k = 0; k < globalToSliceMap[i].second.size(); ++k) {
          auto *op1 = globalToSliceMap[i].second[k];
          auto *op2 = globalToSliceMap[j].second[k];
          if (!isStructurallyEquivalentTo(cache, *op1, *op2, *mapping)) {
            isEquivalent = false;
            break;
          }
          if (!isEquivalent) {
            break;
          }
        }
        if (isEquivalent) {
          storeOpBitVector.set(i);
          storeOpBitVector.set(j);
          ec.unionSets(globalToSliceMap[i].first, globalToSliceMap[j].first);
        }
      }
    }

    for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
      if (!(*it)->isLeader()) {
        continue; // Ignore non-leader sets.
      }
      if (++ec.member_begin(**it) == ec.member_end()) {
        continue; // size 1
      }
      SmallVector<GlobalStoreOpInterface> members;
      for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
        members.push_back(*mi);
      }

      mlir::OpBuilder builder(moduleOp.getContext());
      builder.setInsertionPoint(members.front());
      auto &firstGlobal = globalTable.lookup(members.front().getGlobalName());
      for (int i = 1; i < members.size(); ++i) {
        builder.create<IREE::Util::GlobalStoreOp>(
            members.front().getLoc(), members.front().getStoredGlobalValue(),
            members[i].getGlobalName());

        auto &currentGlobal = globalTable.lookup(members[i].getGlobalName());
        currentGlobal.op->moveAfter(firstGlobal.op);

        members[i].erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
