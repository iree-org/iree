// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- BufferAllocViewCleanUpPass.cpp -------------------------------------===//
//
// This pass performs canonicalizations/cleanups related to HAL interface/buffer
// allocations and views. We need a dedicated pass because patterns here involve
// multiple dialects.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Folds linalg.tensor_reshape into the source hal.interface.binding.subspan.
///
/// For example, this matches the following pattern:
///
///   %subspan = hal.interface.binding.subspan @... :
///       !flow.dispatch.tensor<readonly:3x3x1x96xf32>
///   %tensor = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:3x3x1x96xf32> -> tensor<3x3x1x96xf32>
///   %0 = linalg.tensor_reshape %tensor [
///         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///       ] : tensor<3x3x1x96xf32> into tensor<864xf32>
///
/// And turns it into:
///
///   %subspan = hal.interface.binding.subspan @... :
///       !flow.dispatch.tensor<readonly:864xf32>
///   %0 = flow.dispatch.tensor.load %subspan :
///       !flow.dispatch.tensor<readonly:864xf32> -> tensor<864xf32>
template <typename TensorReshapeOp>
struct FoldReshapeIntoInterfaceTensorLoad : OpRewritePattern<TensorReshapeOp> {
  using OpRewritePattern<TensorReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    // TODO(antigainst): enable dynamic shape support once they are needed.
    auto reshapeSrcType = reshapeOp.src().getType().template cast<ShapedType>();
    auto reshapeDstType = reshapeOp.getType().template cast<ShapedType>();
    if (!reshapeSrcType.hasStaticShape() || !reshapeDstType.hasStaticShape()) {
      return failure();
    }

    auto loadOp =
        reshapeOp.src()
            .template getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp) return failure();

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!loadOp.offsets().empty() || !loadOp.sizes().empty() ||
        !loadOp.strides().empty())
      return failure();

    auto subspanOp =
        loadOp.source()
            .template getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) return failure();
    assert(subspanOp.dynamic_dims().empty());

    auto tensorAccess = subspanOp.getType()
                            .template cast<IREE::Flow::DispatchTensorType>()
                            .getAccess();
    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        tensorAccess, reshapeOp.getResultType());

    Value newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp.getLoc(), newSubspanType, subspanOp.binding(),
        subspanOp.byte_offset(), subspanOp.byte_length(),
        subspanOp.dynamic_dims(), subspanOp.alignmentAttr());

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp,
        loadOp.source_dims());

    return success();
  }
};

// Removes operations with Allocate MemoryEffects but no uses.
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!memEffect || !memEffect.hasEffect<MemoryEffects::Allocate>()) {
      return failure();
    }
    if (!op->use_empty()) return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

/// Runs canonicalization patterns on interface load/store ops.
struct CleanupBufferAllocViewPass
    : public CleanupBufferAllocViewBase<CleanupBufferAllocViewPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<FoldReshapeIntoInterfaceTensorLoad<tensor::CollapseShapeOp>,
                    FoldReshapeIntoInterfaceTensorLoad<tensor::ExpandShapeOp>,
                    RemoveDeadMemAllocs>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createCleanupBufferAllocViewPass() {
  return std::make_unique<CleanupBufferAllocViewPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
