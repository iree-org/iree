// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_VERIFYAFFINITIESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// Verifies that |op| has an affinity assigned on itself or a parent.
static LogicalResult
verifyAffinityAssigned(IREE::Stream::AffinityOpInterface op) {
  if (!op.requiresAffinity()) {
    return success(); // does not require an affinity
  } else if (IREE::Stream::AffinityAttr::lookup(op)) {
    return success(); // has an affinity
  }
  return op->emitOpError()
         << "does not have an affinity assigned; ensure that the op or some "
            "ancestor of it has a valid execution affinity assigned";
}

//===----------------------------------------------------------------------===//
// --iree-stream-verify-affinities
//===----------------------------------------------------------------------===//

struct VerifyAffinitiesPass
    : public IREE::Stream::impl::VerifyAffinitiesPassBase<
          VerifyAffinitiesPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (moduleOp
            .walk<WalkOrder::PreOrder>([&](Operation *op) {
              if (isa<mlir::ModuleOp>(op)) {
                return WalkResult::advance();
              }
              if (auto affinityOp =
                      dyn_cast<IREE::Stream::AffinityOpInterface>(op)) {
                if (failed(verifyAffinityAssigned(affinityOp))) {
                  return WalkResult::interrupt();
                }
              }
              return WalkResult::advance();
            })
            .wasInterrupted())
      return signalPassFailure();
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
