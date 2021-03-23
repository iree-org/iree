// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- BufferAllocViewCleanUpPass.cpp -------------------------------------===//
//
// This pass performs canonicalizations/cleanups related to HAL interface/buffer
// allocations and views. We need a dedicated pass because patterns here involve
// multiple dialects.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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
struct FoldReshapeIntoInterfaceTensorLoad
    : OpRewritePattern<linalg::TensorReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TensorReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    auto loadOp =
        reshapeOp.src().getDefiningOp<IREE::Flow::DispatchTensorLoadOp>();
    if (!loadOp) return failure();

    // Make sure we are loading the full incoming subspan. Otherwise we cannot
    // simply adjust the subspan's resultant type later.
    if (!loadOp.offsets().empty() || !loadOp.sizes().empty() ||
        !loadOp.strides().empty())
      return failure();

    auto subspanOp =
        loadOp.source().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp) return failure();

    auto newSubspanType = IREE::Flow::DispatchTensorType::get(
        subspanOp.getType().cast<IREE::Flow::DispatchTensorType>().getAccess(),
        reshapeOp.getResultType());

    Value newSubspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
        subspanOp.getLoc(), newSubspanType, subspanOp.binding(),
        subspanOp.byte_offset(), subspanOp.byte_length());

    rewriter.replaceOpWithNewOp<IREE::Flow::DispatchTensorLoadOp>(
        reshapeOp, reshapeOp.getResultType(), newSubspanOp);

    return success();
  }
};

// Removes operations with Allocate MemoryEffects but no uses.
struct RemoveDeadMemAllocs : RewritePattern {
  RemoveDeadMemAllocs(PatternBenefit benefit = 1)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

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
struct BufferAllocViewCleanUpPass
    : public PassWrapper<BufferAllocViewCleanUpPass, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<FoldReshapeIntoInterfaceTensorLoad>(&getContext());
    patterns.insert<RemoveDeadMemAllocs>();
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createBufferAllocViewCleanUpPass() {
  return std::make_unique<BufferAllocViewCleanUpPass>();
}

static PassRegistration<BufferAllocViewCleanUpPass> pass(
    "iree-codegen-cleanup-buffer-alloc-view",
    "Performs cleanups over HAL interface/buffer allocation/view operations",
    [] { return std::make_unique<BufferAllocViewCleanUpPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
