// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/CASResources.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

namespace {

class TestCASResourcesPass : public TestCASResourcesBase<TestCASResourcesPass> {
public:
  TestCASResourcesPass() = default;
  TestCASResourcesPass(const TestCASResourcesPass &) = default;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Util::UtilDialect>();
  }

  GlobalOp addGlobalI32x4(CASScopeAttr scopeAttr, StringRef name, int32_t a,
                          int32_t b, int32_t c, int32_t d) {
    MLIRContext *context = &getContext();
    auto *moduleOp = getOperation();
    Location loc = moduleOp->getLoc();
    OpBuilder builder = OpBuilder::atBlockEnd(&moduleOp->getRegion(0).front());
    auto st = RankedTensorType::get({2, 2}, builder.getIntegerType(32));
    auto casb = CASResourceBuilder::allocateHeap(16);
    auto data = casb.getTypedData<int32_t>();
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;

    CASManagerDialectInterface &casManager =
        CASManagerDialectInterface::get(context);
    PopulatedCASResource::Reference ref;
    if (scopeAttr) {
      ref = casManager.internLocalResource(std::move(casb), scopeAttr);
    } else {
      ref = casManager.internGlobalResource(std::move(casb));
    }
    auto value = CASElementsAttr::get(st, ref->getGlobalResource());
    GlobalOp global = builder.create<GlobalOp>(loc, name, false, st, value);
    return global;
  }

  void runOnOperation() override {
    if (testMode == "internGlobalScope") {
      testInternGlobalScope();
    } else if (testMode == "internLocalScope") {
      testInternLocalScope();
    } else if (testMode == "internLocalInvalidate") {
      internLocalInvalidate();
    } else {
      emitError(getOperation()->getLoc())
          << "unknown test-mode '" << testMode << "'";
      signalPassFailure();
    }
  }

  void testInternGlobalScope() {
    llvm::SmallVector<GlobalOp> ops = {
        addGlobalI32x4(nullptr, "test0", 1, 2, 3, 4),
        addGlobalI32x4(nullptr, "test1", 4, 3, 2, 1),
        addGlobalI32x4(nullptr, "test2", 1, 1, 1, 1),
        addGlobalI32x4(nullptr, "test3", 1, 2, 3, 4),
    };
    markAliasedGlobalResources(ops);
  }

  void testInternLocalScope() {
    auto scopeAttr = CASScopeAttr::findOrCreateRootScope(getOperation());
    llvm::SmallVector<GlobalOp> ops = {
        addGlobalI32x4(scopeAttr, "test0", 1, 2, 3, 4),
        addGlobalI32x4(scopeAttr, "test1", 4, 3, 2, 1),
        addGlobalI32x4(nullptr, "test2", 4, 3, 2, 1), // global
        addGlobalI32x4(scopeAttr, "test3", 1, 2, 3, 4),
    };
    markAliasedGlobalResources(ops);
  }

  void internLocalInvalidate() {
    auto scopeAttr = CASScopeAttr::findOrCreateRootScope(getOperation());
    llvm::SmallVector<GlobalOp> ops = {
        addGlobalI32x4(scopeAttr, "test0", 1, 2, 3, 4),
        addGlobalI32x4(scopeAttr, "test1", 4, 3, 2, 1),
        addGlobalI32x4(nullptr, "test2", 4, 3, 2, 1), // global
        addGlobalI32x4(scopeAttr, "test3", 1, 2, 3, 4),
    };
    CASManagerDialectInterface::get(&getContext()).invalidateScope(scopeAttr);
    markDeadGlobalResources(ops);
  }

  void markAliasedGlobalResources(llvm::SmallVectorImpl<GlobalOp> &ops) {
    // Now go annotate an iree.aliases attribute on each where we note
    // indices that the storage key aliases.
    auto getKey = [](GlobalOp globalOp) -> StringRef {
      auto attr = llvm::cast<CASElementsAttr>(*globalOp.getInitialValue());
      return attr.getHandle().getKey();
    };
    for (auto outerOp : ops) {
      StringRef outerKey = getKey(outerOp);
      llvm::SmallVector<int> aliases;
      for (auto innerIt : llvm::enumerate(ops)) {
        if (outerOp == innerIt.value())
          continue;
        StringRef innerKey = getKey(innerIt.value());
        if (outerKey == innerKey)
          aliases.push_back(innerIt.index());
      }
      Builder b(&getContext());
      outerOp->setAttr("iree.aliases", b.getI32ArrayAttr(aliases));
    }
  }

  void markDeadGlobalResources(llvm::SmallVectorImpl<GlobalOp> &ops) {
    for (auto op : ops) {
      auto attr = llvm::cast<CASElementsAttr>(*op.getInitialValue());
      if (attr.isResourceValid()) {
        op->setAttr("iree.resource-live", UnitAttr::get(&getContext()));
      } else {
        op->setAttr("iree.resource-dead", UnitAttr::get(&getContext()));
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<void>> createTestCASResourcesPass() {
  return std::make_unique<TestCASResourcesPass>();
}

} // namespace Util
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir
