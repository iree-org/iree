// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <iterator>
#include <optional>
#include <tuple>
#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/EquivalenceUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"

#define DEBUG_TYPE "iree-util-optimize-global-duplicates"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_OPTIMIZEGLOBALDUPLICATESPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {
static llvm::cl::opt<bool>
    clDisableOptimizeGlobalDuplicates("iree-disable-optimize-global-duplicates",
                                      llvm::cl::desc("Disables this pass"),
                                      llvm::cl::init(false));

bool areAttributesEquivalent(Operation &lhs, Operation &rhs) {
  auto storeOpLhs = llvm::dyn_cast_or_null<IREE::Util::GlobalStoreOp>(lhs);
  auto storeOpRhs = llvm::dyn_cast_or_null<IREE::Util::GlobalStoreOp>(rhs);
  if (storeOpLhs && storeOpRhs) {
    return true;
  }
  return false;
}
class OptimizeGlobalDuplicatesPass
    : public impl::OptimizeGlobalDuplicatesPassBase<
          OptimizeGlobalDuplicatesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  void runOnOperation() override {
    if (clDisableOptimizeGlobalDuplicates)
      return;

    // llvm::outs() << "Running OptimizeGlobalDuplicatesPass\n";
    auto moduleOp = getOperation();

    SmallVector<IREE::Util::InitializerOp> initOps;
    llvm::SmallDenseMap<IREE::Util::InitializerOp, IREE::Util::GlobalStoreOp>
        map;
    for (auto callableOp : moduleOp.getOps<IREE::Util::InitializerOp>()) {
      bool allImmutableLoads = true;
      for (auto loadOp : callableOp.getInitializerRegion()
                             .getOps<IREE::Util::GlobalLoadOp>()) {
        if (!loadOp.isGlobalImmutable()) {
          allImmutableLoads = false;
          break;
        }
      }

      auto bg = callableOp.getInitializerRegion()
                    .getOps<IREE::Util::GlobalStoreOp>()
                    .begin();
      auto en = callableOp.getInitializerRegion()
                    .getOps<IREE::Util::GlobalStoreOp>()
                    .end();

      auto tmp = bg;
      if (allImmutableLoads && ++bg == en) {
        initOps.push_back(callableOp);
        map[callableOp] = *tmp;
      }
    }

    BitVector initOpBitVector(initOps.size());
    OperationEquivalenceCache cache(moduleOp.getContext());
    auto mapping = cache.acquireMapping();
    llvm::EquivalenceClasses<IREE::Util::InitializerOp> ec;
    for (int i = 0; i < initOps.size(); ++i) {
      if (initOpBitVector[i]) {
        continue; // Already processed.
      }
      for (int j = i + 1; j < initOps.size(); ++j) {
        if (initOpBitVector[j]) {
          continue; // Already processed.
        }
        if (isStructurallyEquivalentTo(
                cache, initOps[i].getInitializerRegion(),
                initOps[j].getInitializerRegion(), *mapping,
                [](Operation &lhs, Operation &rhs) {
                  return areAttributesEquivalent(lhs, rhs);
                })) {
          // llvm::outs() << "Found structurally equivalent initializers: "
          //  << initOps[i] << " and " << initOps[j] << "\n";
          initOpBitVector.set(i);
          initOpBitVector.set(j);
          ec.unionSets(initOps[i], initOps[j]);
        }
      }
    }

    for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
      // llvm::outs() << "Equivalence class: \n";
      if (!(*it)->isLeader()) {
        continue; // Ignore non-leader sets.
      }
      if (++ec.member_begin(**it) == ec.member_end()) {
        continue; // size 1
      }
      SmallVector<IREE::Util::InitializerOp> members;
      for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
        members.push_back(*mi);
      }

      mlir::OpBuilder builder(moduleOp.getContext());
      builder.setInsertionPoint(map[members.front()]);
      for (int i = 1; i < members.size(); ++i) {
        builder.create<IREE::Util::GlobalStoreOp>(
            map[members.front()].getLoc(),
            map[members.front()].getStoredGlobalValue(),
            map[members[i]].getGlobal());

        map[members[i]].erase();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
