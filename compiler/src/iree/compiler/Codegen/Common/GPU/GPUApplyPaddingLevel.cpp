// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/GPULoweringConfigUtils.h"
#include "iree/compiler/Codegen/Interfaces/TensorMaskingOpInterface.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-apply-padding-level"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUAPPLYPADDINGLEVELPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

struct FoldPadOfFill final : OpRewritePattern<tensor::PadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto fillOp = padOp.getSource().getDefiningOp<linalg::FillOp>();
    if (!fillOp) {
      return failure();
    }

    Value fillValue = fillOp.getInputs().front();
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return failure();
    }

    // Check if fill value and pad value are both constants and the same.
    Attribute fillAttr;
    Attribute padAttr;
    if (!matchPattern(fillValue, m_Constant(&fillAttr)) ||
        !matchPattern(padValue, m_Constant(&padAttr))) {
      return failure();
    }

    if (!isa<ub::PoisonAttr>(padAttr) && fillAttr != padAttr) {
      // We could also match for when fill is poison, but that doesn't really
      // happen in practice.
      return failure();
    }

    // Replace pad(fill) -> fill.
    rewriter.setInsertionPointAfter(padOp);
    SmallVector<OpFoldResult> paddedSizes =
        tensor::getMixedSizes(rewriter, padOp.getLoc(), padOp.getResult());
    auto emptyOp =
        tensor::EmptyOp::create(rewriter, padOp.getLoc(), paddedSizes,
                                getElementTypeOrSelf(padOp.getResultType()));
    auto newFillOp = linalg::FillOp::create(rewriter, padOp.getLoc(),
                                            {fillValue}, {emptyOp});
    rewriter.replaceOp(padOp, newFillOp.getResult(0));

    return success();
  }
};

struct PropagatePadUpward final : OpRewritePattern<tensor::PadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto resultVal = dyn_cast<OpResult>(padOp.getSource());
    if (!resultVal) {
      return failure();
    }
    Operation *producer = resultVal.getOwner();
    auto maskingIface = dyn_cast<TensorMaskingOpInterface>(producer);
    if (!maskingIface) {
      return failure();
    }
    // If the only uses of this pad op are tensor.dim ops, then don't propagate.
    // This prevents recursive padding when tensor.dim operations are left to
    // resolve.
    // TODO: A better solution is to to not get sizes from the tensor.pad
    // operation, but get it in terms of it's operands.
    bool hasNonDimUse = false;
    for (Operation *user : padOp.getResult().getUsers()) {
      if (!isa<tensor::DimOp>(user)) {
        hasNonDimUse = true;
        break;
      }
    }
    if (!hasNonDimUse) {
      return failure();
    }

    rewriter.setInsertionPointAfter(padOp);
    SmallVector<OpFoldResult> padSizes =
        tensor::getMixedSizes(rewriter, padOp.getLoc(), padOp.getResult());
    FailureOr<Value> paddedResult = maskingIface.maskProducer(
        rewriter,
        /*resultNumber=*/resultVal.getResultNumber(), padSizes);
    if (failed(paddedResult)) {
      return failure();
    }
    DominanceInfo domInfo(padOp);
    rewriter.replaceUsesWithIf(
        padOp.getResult(), paddedResult.value(), [&](OpOperand &use) {
          Operation *user = use.getOwner();
          return domInfo.properlyDominates(paddedResult.value(), user);
        });
    return success();
  }
};

struct PadOfUnMask final : OpRewritePattern<tensor::PadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto unmaskOp =
        padOp.getSource().getDefiningOp<IREE::LinalgExt::UnMaskOp>();
    if (!unmaskOp) {
      return failure();
    }

    // Check if the source dimensions and the padded dimensions are the
    // same.
    for (auto i : llvm::seq<int64_t>(padOp.getResultType().getRank())) {
      auto lhs = ValueBoundsConstraintSet::Variable(unmaskOp.getSrc(), i);
      auto rhs = ValueBoundsConstraintSet::Variable(padOp.getResult(), i);
      bool isEq = ValueBoundsConstraintSet::compare(
          lhs, ValueBoundsConstraintSet::ComparisonOperator::EQ, rhs);
      if (!isEq) {
        return failure();
      }
    }

    // TODO: Generate a select instead of only allowing poison.
    // Check if the pad value is poison.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue) {
      return failure();
    }
    Attribute padAttr;
    if (!matchPattern(padValue, m_Constant(&padAttr))) {
      return failure();
    }
    if (!isa<ub::PoisonAttr>(padAttr)) {
      return failure();
    }

    rewriter.replaceOp(padOp, unmaskOp.getSrc());
    return success();
  }
};

} // namespace

namespace {
struct GPUApplyPaddingLevelPass final
    : impl::GPUApplyPaddingLevelPassBase<GPUApplyPaddingLevelPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static std::optional<SmallVector<int64_t>>
getStaticPaddingSizes(TensorMaskingOpInterface tensorMaskingOp) {
  auto loweringConfig =
      getLoweringConfig<IREE::GPU::LoweringConfigAttr>(tensorMaskingOp);
  if (!loweringConfig) {
    return std::nullopt;
  }
  return IREE::GPU::getPaddingList(loweringConfig);
}

static llvm::SmallDenseSet<TensorMaskingOpInterface>
getOpsToPad(Operation *funcOp) {
  llvm::SmallDenseSet<TensorMaskingOpInterface> targets;
  funcOp->walk([&](TensorMaskingOpInterface target) {
    if (getStaticPaddingSizes(target).has_value()) {
      LDBG() << "Found target op for padding: " << *target.getOperation();
      targets.insert(target);
    }
  });
  return targets;
}

static void makePadDPS(RewriterBase &rewriter, tensor::PadOp padOp) {
  Location loc = padOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(padOp);

  // Record users for RAUW before creating new users.
  llvm::SmallDenseSet<Operation *> users(llvm::from_range,
                                         padOp.getResult().getUsers());
  RankedTensorType tensorTy = padOp.getResultType();
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(rewriter, loc, padOp.getResult());
  Value out = tensor::EmptyOp::create(rewriter, loc, sizes,
                                      getElementTypeOrSelf(tensorTy));
  auto copied = linalg::CopyOp::create(rewriter, loc, padOp.getResult(), out);
  rewriter.replaceUsesWithIf(padOp.getResult(), copied.getResult(0),
                             [&](OpOperand &opOperand) {
                               return users.contains(opOperand.getOwner());
                             });
}

static LogicalResult
applyPaddingLevel(RewriterBase &rewriter,
                  TensorMaskingOpInterface tilingInterfaceOp) {
  auto tensorMaskingOp =
      dyn_cast<TensorMaskingOpInterface>(tilingInterfaceOp.getOperation());
  if (!tensorMaskingOp) {
    return failure();
  }

  std::optional<SmallVector<int64_t>> staticPadSizes =
      getStaticPaddingSizes(tensorMaskingOp);
  if (!staticPadSizes.has_value()) {
    return failure();
  }
  SmallVector<OpFoldResult> padSizes =
      getAsIndexOpFoldResult(rewriter.getContext(), staticPadSizes.value());

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(tilingInterfaceOp);
  FailureOr<SmallVector<Value>> result =
      tensorMaskingOp.getMaskedImplementation(rewriter, padSizes);
  if (failed(result)) {
    return failure();
  }

  rewriter.replaceOp(tilingInterfaceOp.getOperation(), result.value());

  return success();
}

// This uses a really simple algorithm right now:
//   - Pad each operation.
//   - Propagate tensor.pad operations up.
//
// Currently, this doesn't work for cases:
//   - Propagating unmask to consumers.
//   - Handling pad(unmask) unless they both are consistent to each other.
//
// The algorithm could use a revamp in future.
void GPUApplyPaddingLevelPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  llvm::SmallDenseSet<TensorMaskingOpInterface> targetOps = getOpsToPad(funcOp);

  IRRewriter rewriter(funcOp);
  for (TensorMaskingOpInterface op : targetOps) {
    LDBG() << "Applying padding level to op: " << *op.getOperation();
    // If some op does not get padded, that is fine for now.
    (void)applyPaddingLevel(rewriter, op);
  }

  // Clean up patterns.
  RewritePatternSet patterns(&getContext());
  patterns.add<FoldPadOfFill, PropagatePadUpward, PadOfUnMask>(&getContext());
  tensor::PadOp::getCanonicalizationPatterns(patterns, &getContext());
  IREE::LinalgExt::UnMaskOp::getCanonicalizationPatterns(patterns,
                                                         &getContext());
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, &getContext());
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, &getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }

  // Make all remaining pads DPS.
  funcOp.walk([&](tensor::PadOp padOp) {
    if (!padOp.use_empty()) {
      makePadDPS(rewriter, padOp);
    }
  });
}

} // namespace mlir::iree_compiler
