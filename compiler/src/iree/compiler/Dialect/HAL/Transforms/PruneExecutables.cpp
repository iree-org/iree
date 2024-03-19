// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_PRUNEEXECUTABLESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

struct SymbolReferences {
  Operation *symbolOp;
  unsigned count = 0;
};
using SymbolReferenceMap = DenseMap<Attribute, SymbolReferences>;

static void markReferenced(SymbolRefAttr symbolRefAttr,
                           SymbolReferenceMap &referenceMap) {
  auto markReferencedNested = [&](StringAttr rootRefAttr,
                                  ArrayRef<FlatSymbolRefAttr> nestedRefAttrs) {
    auto nestedRefAttr = nestedRefAttrs.empty()
                             ? SymbolRefAttr::get(rootRefAttr)
                             : SymbolRefAttr::get(rootRefAttr, nestedRefAttrs);
    auto it = referenceMap.find(nestedRefAttr);
    if (it != referenceMap.end())
      ++it->second.count;
  };
  auto rootRefAttr = symbolRefAttr.getRootReference();
  auto nestedRefAttrs = symbolRefAttr.getNestedReferences();
  for (size_t i = 0; i <= nestedRefAttrs.size(); ++i) {
    markReferencedNested(rootRefAttr, nestedRefAttrs.slice(0, i));
  }
}

static void processOp(Operation *op, SymbolReferenceMap &referenceMap) {
  SmallVector<Attribute> worklist;
  for (auto namedAttr : op->getAttrs())
    worklist.push_back(namedAttr.getValue());
  while (!worklist.empty()) {
    auto attr = worklist.pop_back_val();
    if (auto symbolRefAttr = dyn_cast<SymbolRefAttr>(attr)) {
      markReferenced(symbolRefAttr, referenceMap);
    } else {
      attr.walkImmediateSubElements(
          [&](Attribute attr) { worklist.push_back(attr); }, [](Type) {});
    }
  }
}

static void eraseOps(ArrayRef<Attribute> symbolRefAttrs,
                     SymbolReferenceMap &referenceMap) {
  for (auto symbolRefAttr : symbolRefAttrs) {
    auto &symbolRefs = referenceMap[symbolRefAttr];
    if (symbolRefs.count == 0)
      symbolRefs.symbolOp->erase();
  }
}

//===----------------------------------------------------------------------===//
// --iree-hal-prune-executables
//===----------------------------------------------------------------------===//

struct PruneExecutablesPass
    : public IREE::HAL::impl::PruneExecutablesPassBase<PruneExecutablesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather all executable op symbols into a map that we can quickly check
    // while walking ops.
    DenseSet<Operation *> ignoredOps;
    SymbolReferenceMap referenceMap;
    SmallVector<Attribute> executableRefAttrs;
    SmallVector<Attribute> variantRefAttrs;
    SmallVector<Attribute> exportRefAttrs;
    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
      ignoredOps.insert(executableOp);
      if (!executableOp.isPrivate())
        continue;
      auto executableRefAttr =
          FlatSymbolRefAttr::get(executableOp.getSymNameAttr());
      referenceMap[executableRefAttr].symbolOp = executableOp;
      executableRefAttrs.push_back(executableRefAttr);
      for (auto variantOp :
           executableOp.getOps<IREE::HAL::ExecutableVariantOp>()) {
        ignoredOps.insert(variantOp);
        auto variantRefAttr = SymbolRefAttr::get(
            executableOp.getSymNameAttr(),
            {
                FlatSymbolRefAttr::get(variantOp.getSymNameAttr()),
            });
        referenceMap[variantRefAttr].symbolOp = variantOp;
        variantRefAttrs.push_back(variantRefAttr);
        for (auto exportOp :
             variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
          ignoredOps.insert(exportOp);
          auto exportRefAttr = SymbolRefAttr::get(
              executableOp.getSymNameAttr(),
              {
                  FlatSymbolRefAttr::get(variantOp.getSymNameAttr()),
                  FlatSymbolRefAttr::get(exportOp.getSymNameAttr()),
              });
          referenceMap[exportRefAttr].symbolOp = exportOp;
          exportRefAttrs.push_back(exportRefAttr);
        }
      }
    }

    // Walk all ops in the module that can reference executable symbols and
    // accumulate the usage counts.
    SymbolTable symbolTable(moduleOp);
    moduleOp.walk([&](Operation *op) -> WalkResult {
      if (ignoredOps.contains(op))
        return WalkResult::skip();
      processOp(op, referenceMap);
      return op->hasTrait<OpTrait::IsIsolatedFromAbove>()
                 ? WalkResult::skip()
                 : WalkResult::advance();
    });

    // Erase any executable-related op with no references.
    // We have to start with exports > variants > executables so that we don't
    // erase a container before the nested ops.
    eraseOps(exportRefAttrs, referenceMap);
    eraseOps(variantRefAttrs, referenceMap);
    eraseOps(executableRefAttrs, referenceMap);
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
