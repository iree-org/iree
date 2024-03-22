// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmgpu-promote-dense-cst-to-tensor"

namespace mlir::iree_compiler {
namespace {
struct LLVMGPUPromoteDenseCstToTensorPass
    : public LLVMGPUPromoteDenseCstToTensorBase<
          LLVMGPUPromoteDenseCstToTensorPass> {
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();
    SmallVector<arith::ConstantOp> candidates;
    funcOp.walk([&](arith::ConstantOp op) {
      using VectorValue = TypedValue<VectorType>;
      auto constant = dyn_cast<VectorValue>(op.getResult());
      if (!constant)
        return WalkResult::advance();
      if (isa<SplatElementsAttr>(op.getValue()))
        return WalkResult::advance();
      candidates.push_back(op);
      return WalkResult::advance();
    });

    IRRewriter rewriter(ctx);
    for (auto constantOp : candidates) {
      rewriter.setInsertionPoint(constantOp);
      auto attr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
      auto attrType = dyn_cast<ShapedType>(attr.getType());
      auto tensorType =
          RankedTensorType::get(attrType.getShape(), attrType.getElementType());
      auto newAttr =
          DenseElementsAttr::getFromRawBuffer(tensorType, attr.getRawData());
      auto newCst =
          rewriter.create<arith::ConstantOp>(constantOp.getLoc(), newAttr);
      SmallVector<Value> indices(
          tensorType.getRank(),
          rewriter.create<arith::ConstantIndexOp>(constantOp.getLoc(), 0));
      auto readOp = rewriter.create<vector::TransferReadOp>(
          constantOp.getLoc(),
          cast<VectorType>(constantOp.getResult().getType()), newCst, indices);
      rewriter.replaceOp(constantOp, readOp);
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMGPUPromoteDenseCstToTensorPass() {
  return std::make_unique<LLVMGPUPromoteDenseCstToTensorPass>();
}

} // namespace mlir::iree_compiler
