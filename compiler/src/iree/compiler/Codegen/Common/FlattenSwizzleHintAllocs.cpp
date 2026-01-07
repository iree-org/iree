// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FLATTENSWIZZLEHINTALLOCSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct FlattenSwizzleHintAllocsPass final
    : public impl::FlattenSwizzleHintAllocsPassBase<
          FlattenSwizzleHintAllocsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static void flattenSwizzleHintAllocs(RewriterBase &rewriter, IREE::Codegen::SwizzleHintOp hintOp) {
    auto allocOp = hintOp.getOperand().getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
        return;
    }
    if (!allocOp->hasOneUse()) {
        return;
    }
    auto resultType = allocOp.getType();
    if (resultType.getRank() == 1) {
        return;
    }
    auto newResultShape = SmallVector<int64_t>({resultType.getNumElements()});
    MemRefType newResultType = MemRefType::get(newResultShape, resultType.getElementType(),
    AffineMap(), resultType.getMemorySpace());
    rewriter.setInsertionPoint(hintOp);
    ReassociationIndices reassoc = llvm::to_vector(llvm::seq(resultType.getRank()));
    auto newAllocOp = memref::AllocOp::create(rewriter, hintOp.getLoc(), newResultType);
    auto newSwizzleHintOp = IREE::Codegen::SwizzleHintOp::create(rewriter, hintOp.getLoc(), newAllocOp.getResult(), hintOp.getSwizzle());
    auto expandShape = memref::ExpandShapeOp::create(rewriter, hintOp.getLoc(), resultType.getShape(), newSwizzleHintOp, {reassoc});
    rewriter.replaceAllUsesWith(hintOp, expandShape);
}

void FlattenSwizzleHintAllocsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  // Collect all swizzle hint ops that operate on allocations.
  // Flatten all allocs of rank > 1.
  SmallVector<IREE::Codegen::SwizzleHintOp> hintOps;
  funcOp.walk(
      [&](IREE::Codegen::SwizzleHintOp hint) { hintOps.push_back(hint); });

  IRRewriter rewriter(funcOp->getContext());
  for (IREE::Codegen::SwizzleHintOp hintOp : hintOps) {
    flattenSwizzleHintAllocs(rewriter, hintOp);
  }
}

} // namespace mlir::iree_compiler

