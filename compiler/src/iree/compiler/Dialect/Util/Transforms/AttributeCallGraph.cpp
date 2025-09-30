// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_ATTRIBUTECALLGRAPHPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

static void propagateAttribute(Operation *from, Operation *to, StringRef name) {
  if (auto attr = from->getAttr(name)) {
    if (!to->hasAttr(name)) {
      to->setAttr(name, attr);
    }
  }
}

struct AttributeCallGraphPass
    : public impl::AttributeCallGraphPassBase<AttributeCallGraphPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    // Walk all call sites in top-level ops and propagate attributes from
    // callees (we don't want to traverse into object-like ops).
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      funcOp.walk([&](IREE::Util::CallOp callOp) {
        auto callee = callOp.getCalleeAttr();
        if (!callee)
          return;

        // Look up the callee function.
        auto calleeOp =
            symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(callOp,
                                                                     callee);
        if (!calleeOp) {
          return;
        }

        // TODO(benvanik): perform local effects analysis for internal functions
        // without explicit nosideeffects attributes.

        // Propagate attributes to callers when requested.
        for (auto attr : calleeOp->getAttrs()) {
          if (auto annotationAttr =
                  dyn_cast_if_present<IREE::Util::CallAnnotationAttrInterface>(
                      attr.getValue())) {
            if (annotationAttr.shouldPropagateToCallSite()) {
              propagateAttribute(calleeOp, callOp, attr.getName());
            }
          }
        }

        // TODO(benvanik): support #util.effects<...> for fine-grained effects.
        propagateAttribute(calleeOp, callOp, "nosideeffects");

        // Clone all iree.abi.* attributes.
        for (auto attr : calleeOp->getAttrs()) {
          StringAttr attrName = attr.getName();
          if (attrName.strref().starts_with("iree.abi.")) {
            propagateAttribute(calleeOp, callOp, attrName);
          }
        }
      });
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Util
