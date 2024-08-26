// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/FormDispatchRegions.h"

#include "iree/compiler/Dialect/Flow/Transforms/ConvertRegionToWorkgroups.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

//===----------------------------------------------------------------------===//
// Definition of TensorDimTrackingRewriter
//===----------------------------------------------------------------------===//

namespace mlir {

TensorDimTrackingRewriter::TensorDimTrackingRewriter(Operation *op)
    : IRRewriter(op->getContext()) {
  setListener(this);
  op->walk([&](tensor::DimOp dimOp) { dimOps.insert(dimOp.getOperation()); });
}
SmallVector<tensor::DimOp> TensorDimTrackingRewriter::getTensorDimOps() {
  SmallVector<tensor::DimOp> result;
  for (Operation *op : dimOps)
    result.push_back(cast<tensor::DimOp>(op));
  return result;
}
void TensorDimTrackingRewriter::notifyOperationErased(Operation *op) {
  IRRewriter::Listener::notifyOperationErased(op);
  if (isa<tensor::DimOp>(op))
    dimOps.erase(op);
}

void TensorDimTrackingRewriter::notifyOperationInserted(Operation *op,
                                                        InsertPoint previous) {
  IRRewriter::Listener::notifyOperationInserted(op, previous);
  if (isa<tensor::DimOp>(op))
    dimOps.insert(op);
}

} // namespace mlir

namespace mlir::iree_compiler::IREE::Flow {

LogicalResult simplifyDimOps(RewriterBase &rewriter,
                             const SmallVector<tensor::DimOp> &dimOps) {
  for (tensor::DimOp dimOp : dimOps) {
    // Only DimOps with static indices are supported.
    std::optional<int64_t> idx = dimOp.getConstantIndex();
    if (!idx.has_value())
      continue;
    // Only DimOps with ranked tensors are supported.
    auto tensorType =
        llvm::dyn_cast<RankedTensorType>(dimOp.getSource().getType());
    if (!tensorType)
      continue;

    if (!tensorType.isDynamicDim(*idx)) {
      // Rewrite static dimension with constant.
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(dimOp);
      int64_t size = tensorType.getShape()[*idx];
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(dimOp, size);
      continue;
    }

    // Try to simplify dynamic dims.
    SmallVector<Value> dynamicDims;
    if (succeeded(IREE::Flow::getOptimizedDynamicResultDims(
            rewriter, dimOp.getSource(), dynamicDims))) {
      unsigned ctr = 0;
      for (int64_t i = 0; i < *dimOp.getConstantIndex(); ++i)
        if (tensorType.isDynamicDim(i))
          ++ctr;
      rewriter.replaceOp(dimOp, dynamicDims[ctr]);
    }
  }

  return success();
}

} // namespace mlir::iree_compiler::IREE::Flow
