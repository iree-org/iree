// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTSTRIDEDCONTRACTIONTOCONTRACTIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

class ConvertStridedContractionToContraction
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the generic op satisfies all other conditions for being a
    // contraction.
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();
    if (op.getNumReductionLoops() == 0)
      return failure();
    if (!mlir::linalg::detail::isContractionBody(
            *op.getBlock(), [](Operation *first, Operation *second) {
              if ((isa<arith::MulFOp>(first) && isa<arith::AddFOp>(second)) ||
                  (isa<arith::MulIOp>(first) && isa<arith::AddIOp>(second)))
                return true;
              return false;
            })) {
      return failure();
    }

    SmallVector<AffineMap> mapRange = op.getIndexingMapsArray();
    unsigned inputPos = op.getDpsInputOperand(0)->getOperandNumber();
    unsigned filterPos = op.getDpsInputOperand(1)->getOperandNumber();
    unsigned resInitPos = op.getDpsInitOperand(0)->getOperandNumber();
    AffineMap inputMap = mapRange[inputPos];
    AffineMap filterMap = mapRange[filterPos];
    AffineMap resultMap = mapRange[resInitPos];
    // For now, we are only handling the case where the first input is the
    // only non-projected permutation.
    if (!filterMap.isProjectedPermutation() ||
        !resultMap.isProjectedPermutation()) {
      return failure();
    }
    if (inputMap.isProjectedPermutation())
      return failure();
    SmallVector<int64_t> staticShape = op.getStaticLoopRanges();

    llvm::SmallDenseMap<unsigned, int64_t> strides;
    SmallVector<AffineExpr> replacementExprs;
    Value input = op.getDpsInputs()[0];
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTy)
      return failure();
    SmallVector<int64_t> inputShape(inputTy.getShape());
    replacementExprs.reserve(inputMap.getNumResults());
    // Walk through input map and look for expressions of the form `dim * cst`.
    for (auto [pos, expr] : llvm::enumerate(inputMap.getResults())) {
      // Skip dim exprs and constant exprs.
      if (isa<AffineDimExpr>(expr) || isa<AffineConstantExpr>(expr)) {
        replacementExprs.push_back(expr);
        continue;
      }
      // Look at binary op expressions.
      auto binexpr = dyn_cast<AffineBinaryOpExpr>(expr);
      // Fail if we see some unexpected kind of expression.
      if (!binexpr)
        return failure();
      auto rhs = dyn_cast<AffineConstantExpr>(binexpr.getRHS());
      auto lhs = dyn_cast<AffineDimExpr>(binexpr.getLHS());
      // Binary expressions must be of the form `dim * cst`.
      if (!rhs || !lhs || binexpr.getKind() != AffineExprKind::Mul) {
        replacementExprs.push_back(expr);
        continue;
      }
      strides.insert(std::pair<unsigned, int64_t>(pos, rhs.getValue()));
      int64_t newSize = staticShape[lhs.getPosition()];
      if (newSize == ShapedType::kDynamic || newSize == 0)
        return failure();
      inputShape[pos] = newSize;
      replacementExprs.push_back(lhs);
    }

    // Fail if we don't have any work to do.
    if (strides.empty())
      return failure();

    mapRange[inputPos] =
        AffineMap::get(inputMap.getNumDims(), inputMap.getNumSymbols(),
                       replacementExprs, op.getContext());
    auto sliceTy = RankedTensorType::get(inputShape, inputTy.getElementType());

    unsigned rank = inputTy.getRank();
    SmallVector<OpFoldResult> vOffset(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> vSizes;
    SmallVector<OpFoldResult> vStride(rank, rewriter.getIndexAttr(1));
    Location loc = op.getLoc();
    for (unsigned i = 0; i < inputTy.getRank(); i++) {
      if (strides.contains(i)) {
        vStride[i] = rewriter.getIndexAttr(strides.at(i));
      }
      if (inputShape[i] != ShapedType::kDynamic) {
        vSizes.push_back(rewriter.getIndexAttr(inputShape[i]));
        continue;
      }
      vSizes.push_back(rewriter.createOrFold<tensor::DimOp>(loc, input, i));
    }
    Value extractedSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, sliceTy, input, vOffset, vSizes, vStride);
    rewriter.startOpModification(op);
    op.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(mapRange));
    op.setOperand(0, extractedSlice);
    rewriter.finalizeOpModification(op);
    return success();
  }
};

struct ConvertStridedContractionToContractionPass
    : public impl::ConvertStridedContractionToContractionPassBase<
          ConvertStridedContractionToContractionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertStridedContractionToContraction>(context);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
