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
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
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

using MapTy =
    llvm::SmallDenseMap<StringRef, std::pair<SymbolRefAttr, StringRef>>;

void processStoreOp(IREE::Util::GlobalLoadOp loadOp,
                    IREE::Stream::TensorDispatchOp dispatchOp,
                    IREE::Util::GlobalStoreOp storeOp, MapTy &m,
                    llvm::SmallDenseMap<StringRef, StringRef> &m1) {
  StringRef loadRef = loadOp.getGlobal();
  StringRef storeRef = storeOp.getGlobal();
  for (auto ref : dispatchOp.getEntryPointRefs()) {
    auto [it, inserted] = m.try_emplace(loadRef, std::make_pair(ref, storeRef));
    if (!inserted && it->second.first == ref) {
      m1.insert({storeRef, it->second.second});
      storeOp.erase();
    }
  }
}

void processDispatchOp(IREE::Util::GlobalLoadOp loadOp, Operation *dispatchUser,
                       MapTy &m,
                       llvm::SmallDenseMap<StringRef, StringRef> &m1) {
  auto dispatchOp =
      llvm::dyn_cast<IREE::Stream::TensorDispatchOp>(dispatchUser);
  if (!dispatchOp)
    return;

  for (auto *u1 : dispatchOp->getUsers()) {
    for (auto *u2 : u1->getUsers()) {
      if (auto storeOp = llvm::dyn_cast<IREE::Util::GlobalStoreOp>(u2)) {
        processStoreOp(loadOp, dispatchOp, storeOp, m, m1);
      }
    }
  }
}

void processLoadOp(IREE::Util::GlobalLoadOp loadOp, MapTy &m,
                   llvm::SmallDenseMap<StringRef, StringRef> &m1) {
  if (!llvm::isa<IREE::Stream::ResourceType>(loadOp.getType()))
    return;

  llvm::outs() << loadOp << '\n';
  for (auto *loadUser : loadOp->getUsers()) {
    llvm::outs() << *loadUser << '\n';
    for (auto *asyncUser : loadUser->getUsers()) {
      llvm::outs() << *asyncUser << '\n';
      processDispatchOp(loadOp, asyncUser, m, m1);
    }
  }
}

void processInitializerOp(IREE::Util::InitializerOp initializerOp, MapTy &m,
                          llvm::SmallDenseMap<StringRef, StringRef> &m1) {
  for (auto loadOp : initializerOp.getOps<IREE::Util::GlobalLoadOp>()) {
    processLoadOp(loadOp, m, m1);
  }
}

void rewriteGlobalLoads(IREE::Util::FuncOp funcOp,
                        llvm::SmallDenseMap<StringRef, StringRef> &m1) {
  funcOp.walk([&m1](IREE::Util::GlobalLoadOp loadOp) {
    StringRef s1 = loadOp.getGlobal();
    if (auto it = m1.find(s1); it != m1.end()) {
      loadOp.setGlobal(it->second);
    }
  });
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

    auto moduleOp = getOperation();
    MapTy m;
    llvm::SmallDenseMap<StringRef, StringRef> m1;

    for (auto initializerOp : moduleOp.getOps<IREE::Util::InitializerOp>()) {
      processInitializerOp(initializerOp, m, m1);
    }

    for (auto funcOp : moduleOp.getOps<IREE::Util::FuncOp>()) {
      rewriteGlobalLoads(funcOp, m1);
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
