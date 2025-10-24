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
      // TODO(benvanik): use an interface? Today we only guarantee our
      // attributes work on util.call, but we may want to support a
      // util.call.indirect in the future (it'd need different analysis, but
      // could annotate with these attributes when analysis resolves).
      funcOp.walk([&](IREE::Util::CallOp callOp) {
        auto callee = callOp.getCalleeAttr();
        if (!callee) {
          return;
        }
        auto calleeOp =
            symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(callOp,
                                                                     callee);
        if (!calleeOp) {
          return;
        }

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
        // TODO(benvanik): perform local effects analysis for internal functions
        //     without needing explicit nosideeffects attributes (though we
        //     should always allow overrides by having the attributes already
        //     exist). We'd run that first here and then annotate both the
        //     callee function declaration and the call sites.
        propagateAttribute(calleeOp, callOp, "nosideeffects");
      });
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Util
