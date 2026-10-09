// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "ootex/IR/OotexDialect.h"
#include "ootex/Transforms/Passes.h"

namespace ootex {

#define GEN_PASS_DEF_ANNOTATEMARKEDFUNCTIONS
#include "ootex/Transforms/Passes.h.inc"

namespace {

struct AnnotateMarkedFunctionsPass
    : public impl::AnnotateMarkedFunctionsBase<AnnotateMarkedFunctionsPass> {
  using Base::Base;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::StringAttr tagAttr =
        mlir::StringAttr::get(module.getContext(), this->tag);

    module.walk([&](mlir::iree_compiler::IREE::Util::FuncOp funcOp) {
      llvm::SmallVector<MarkOp> marks(funcOp.getOps<MarkOp>());
      if (marks.empty()) {
        return;
      }
      funcOp->setAttr("ootex.tag", tagAttr);
      for (MarkOp mark : marks) {
        mark.erase();
      }
    });
  }
};

}  // namespace
}  // namespace ootex
