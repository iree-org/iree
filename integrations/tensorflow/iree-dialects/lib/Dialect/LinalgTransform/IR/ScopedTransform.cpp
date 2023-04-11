// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgTransform/ScopedTransform.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;

namespace {
struct Rewriter : public PatternRewriter {
  Rewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
};
} // namespace

linalg::transform::ScopeOp linalg::transform::wrapInScope(Operation *op) {
  Rewriter rewriter(op->getContext());
  rewriter.setInsertionPoint(op);

  auto scope = rewriter.create<linalg::transform::ScopeOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands());
  Region &body = scope.getBody();
  rewriter.setInsertionPointToStart(&body.emplaceBlock());
  IRMapping bv;
  SmallVector<Location> locs(op->getOperandTypes().size(), op->getLoc());
  bv.map(op->getOperands(), body.addArguments(op->getOperandTypes(), locs));

  Operation *cloneInScope = rewriter.clone(*op, bv);
  rewriter.create<ForwardOp>(op->getLoc(), cloneInScope->getResults());

  rewriter.replaceOp(op, scope.getResults());
  return scope;
}

namespace {
/// Instruct the inliner to inline everything. Scopes have no semantic meaning
/// so moving operations in and out of them, regardless of whether their
/// dialects have implemented an inliner interface, is valid.
struct ScopeInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const override {
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }

  /// Don't recursively analyze operations, because they can all be "inlined".
  bool shouldAnalyzeRecursively(Operation *op) const override { return false; }

  /// Replace uses of the results with the `forward` op's operands.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const override {
    assert(isa<linalg::transform::ForwardOp>(op));
    for (auto value : llvm::zip(op->getOperands(), valuesToRepl))
      std::get<1>(value).replaceAllUsesWith(std::get<0>(value));
  }
};
} // namespace

FailureOr<SmallVector<Operation *>>
linalg::transform::unwrapScope(linalg::transform::ScopeOp scope) {
  ScopeInliner interface(scope->getContext());
  SmallVector<Operation *> ops;
  scope.getBody().walk([&](Operation *op) { ops.push_back(op); });
  if (failed(inlineRegion(interface, &scope.getBody(), scope,
                          scope.getOperands(), scope.getResults(),
                          /*inlineLoc=*/{},
                          /*shouldCloneInlinedRegion=*/false)))
    return failure();
  Rewriter(scope->getContext()).eraseOp(scope);
  return ops;
}
