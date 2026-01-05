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
    auto resultType = allocOp.getType();
    if (resultType.getRank() == 1) {
        return;
    }
    // auto resultShape = allocOp.getType().getShape();
    auto newResultShape = SmallVector<int64_t>({resultType.getNumElements()});
    // auto resultElementType = resultType.getElementType();
    MemRefType newResultType = MemRefType::get(newResultShape, resultType.getElementType(),
    AffineMap(), resultType.getMemorySpace());
    
    //   auto flatAllocType = MemRefType::get(ArrayRef<int64_t>{memRefType.getNumElements()}, memRefType.getElementType(), AffineMap(), workgroupSpace);
    //   Value flatAlloc = memref::AllocOp::create(builder, loc, flatAllocType);
    //   Value swizzled = iree_compiler::IREE::Codegen::SwizzleHintOp::create(builder, loc, flatAlloc,
    //                                                                        iree_compiler::IREE::Codegen::XORShuffleAttr::get(builder.getContext(), 8, 4, int64_t(), int64_t()));
    
    rewriter.setInsertionPoint(hintOp);
    ReassociationIndices reassoc = llvm::to_vector(llvm::seq(resultType.getRank()));
    //   Value expanded = memref::ExpandShapeOp::create(builder, loc, allocType.getShape(), swizzled, {reassoc});
    auto newAllocOp = memref::AllocOp::create(rewriter, hintOp.getLoc(), newResultType);
    llvm::errs() << "newAllocOp: " << newAllocOp.getResult() << "\n";
    auto newSwizzleHintOp = IREE::Codegen::SwizzleHintOp::create(rewriter, hintOp.getLoc(), newAllocOp.getResult(), hintOp.getSwizzle());
    llvm::errs() << "newSwizzleHintOp: " << newSwizzleHintOp.getResult() << "\n";
    auto expandShape = memref::ExpandShapeOp::create(rewriter, hintOp.getLoc(), resultType.getShape(), newSwizzleHintOp, {reassoc});
    llvm::errs() << "expandShape: " << expandShape.getResult() << "\n";
    rewriter.replaceAllUsesWith(hintOp, expandShape);
}

void FlattenSwizzleHintAllocsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  // TODO: Implement pass logic here.
  // Collect all swizzle hint ops that operate on allocations.
  SmallVector<IREE::Codegen::SwizzleHintOp> hintOps;
  funcOp.walk(
      [&](IREE::Codegen::SwizzleHintOp hint) { hintOps.push_back(hint); });

  IRRewriter rewriter(funcOp->getContext());

  // TODO: Implement flattening logic for swizzle hint allocations.
  for (IREE::Codegen::SwizzleHintOp hintOp : hintOps) {
    flattenSwizzleHintAllocs(rewriter, hintOp); // Placeholder - implement transformation logic here.
  }
}

} // namespace mlir::iree_compiler

