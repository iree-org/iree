// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir::iree_compiler::IREE::Util {

#define GEN_PASS_DEF_STRIPDEBUGOPSPASS
#include "iree/compiler/Dialect/Util/Transforms/Passes.h.inc"

namespace {

class StripDebugOpsPass
    : public impl::StripDebugOpsPassBase<StripDebugOpsPass> {
public:
  void runOnOperation() override {
    getOperation()->walk([](Operation *op) {
      if (isa<mlir::cf::AssertOp>(op) ||
          op->hasTrait<OpTrait::IREE::Util::DebugOnly>()) {
        op->erase();
      }
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Util
