// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ABSORBSWIZZLEHINTTOALLOCPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct AbsorbSwizzleHintToAllocPass final
    : impl::AbsorbSwizzleHintToAllocPassBase<AbsorbSwizzleHintToAllocPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

/// Absorbs `iree_codegen.swizzle_hint` ops into an attribute on the defining
/// `memref.alloc`, then erases the hint. This allows downstream passes
/// (multi-buffering, pipelining) to operate properly.
static LogicalResult
absorbSwizzleHintToAlloc(RewriterBase &rewriter,
                         IREE::Codegen::SwizzleHintOp hintOp) {
  auto allocOp = hintOp.getOperand().getDefiningOp<memref::AllocOp>();
  if (!allocOp) {
    return hintOp.emitError()
           << "expected swizzle_hint operand to be defined by a memref.alloc";
  }

  allocOp->setAttr("iree_codegen.swizzle", hintOp.getSwizzleAttr());
  rewriter.replaceOp(hintOp, hintOp.getOperand());
  return success();
}

void AbsorbSwizzleHintToAllocPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  SmallVector<IREE::Codegen::SwizzleHintOp> hintOps;
  funcOp.walk(
      [&](IREE::Codegen::SwizzleHintOp hint) { hintOps.push_back(hint); });

  IRRewriter rewriter(funcOp->getContext());
  for (IREE::Codegen::SwizzleHintOp hintOp : hintOps) {
    if (failed(absorbSwizzleHintToAlloc(rewriter, hintOp))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
