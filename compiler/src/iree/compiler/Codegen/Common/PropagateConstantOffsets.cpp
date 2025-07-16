// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_PROPAGATECONSTANTOFFSETSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct PropagateConstantOffsetsPass final
    : public impl::PropagateConstantOffsetsPassBase<
          PropagateConstantOffsetsPass> {
  using Base::Base;
  void runOnOperation() override;
};

/// Helper to extract constant RHS values from a defining arith op.
template <typename OpTy>
static std::optional<int64_t> getValueConstantRhs(Value v, bool nsw) {
  auto op = v.getDefiningOp<OpTy>();
  if (!op) {
    return std::nullopt;
  }

  // Conditionally require nsw.
  if (nsw && !bitEnumContainsAll(op.getOverflowFlags(),
                                 arith::IntegerOverflowFlags::nsw)) {
    return std::nullopt;
  }

  // Folders move constant operands to the RHS so no need to check both sides.
  APInt constant;
  if (!matchPattern(op.getRhs(), m_ConstantInt(&constant))) {
    return std::nullopt;
  }

  return constant.getSExtValue();
}

/// Converts applies of the form:
///   %x = affine.apply affine_map<expr + C>
/// to
///   %apply = affine.apply affine_map<expr>
///   %x = arith.addi %apply, C overflow<nsw>
struct ExtractConstantApplyOffset final
    : OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp apply,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply.getMap();
    // Simplify the map to move `+ c` terms to the right most (first) expression
    // in the tree.
    map = simplifyAffineMap(map);
    AffineExpr resultExpr = map.getResult(0);
    auto addExpr = dyn_cast<AffineBinaryOpExpr>(resultExpr);

    // After simplification, the add should be the first expression if present.
    if (!addExpr || addExpr.getKind() != AffineExprKind::Add) {
      return rewriter.notifyMatchFailure(apply, "top level expr not an add");
    }

    auto constantRhs = dyn_cast<AffineConstantExpr>(addExpr.getRHS());
    if (!constantRhs) {
      return rewriter.notifyMatchFailure(apply, "non-const rhs");
    }

    int64_t constantOffset = constantRhs.getValue();

    AffineMap newMap =
        AffineMap::get(map.getNumDims(), map.getNumSymbols(), addExpr.getLHS());
    Value newApply = rewriter.create<affine::AffineApplyOp>(
        apply.getLoc(), newMap, apply.getOperands());
    Value offset =
        rewriter.create<arith::ConstantIndexOp>(apply.getLoc(), constantOffset);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        apply, newApply, offset, arith::IntegerOverflowFlags::nsw);
    return success();
  }
};

/// Converts sequences of the form:
///   %add = arith.addi %in, C overflow<nsw>
///   %x = affine.apply affine_map<expr(d0)>(%add)
/// to
///   %x = affine.apply affine_map<expr(d0 + C)>(%in)
struct FoldApplySymbolOrDimSum final : OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineApplyOp apply,
                                PatternRewriter &rewriter) const override {
    AffineMap map = apply.getMap();
    SmallVector<AffineExpr> replacements;
    replacements.reserve(map.getNumInputs());
    int64_t numDims = map.getNumDims();
    auto getCurrExpr = [&](int64_t i) -> AffineExpr {
      if (i >= numDims)
        return rewriter.getAffineSymbolExpr(i - numDims);
      return rewriter.getAffineDimExpr(i);
    };
    bool didReplace = false;
    for (int64_t i = 0, e = map.getNumInputs(); i < e; ++i) {
      AffineExpr currExpr = getCurrExpr(i);
      OpOperand &operand = apply->getOpOperand(i);
      std::optional<int64_t> maybeOffset =
          getValueConstantRhs<arith::AddIOp>(operand.get(), /*nsw=*/true);
      if (!maybeOffset) {
        replacements.push_back(currExpr);
        continue;
      }
      if (!didReplace) {
        rewriter.startOpModification(apply);
      }
      // Replace the dim/symbol with the lhs of the producing add.
      operand.assign(operand.get().getDefiningOp<arith::AddIOp>().getLhs());
      replacements.push_back(currExpr +
                             rewriter.getAffineConstantExpr(*maybeOffset));
      didReplace = true;
    }

    if (!didReplace) {
      return rewriter.notifyMatchFailure(
          apply, "no constant addition operands to replace");
    }

    apply.setMap(map.replaceDimsAndSymbols(
        ArrayRef<AffineExpr>(replacements).take_front(numDims),
        ArrayRef<AffineExpr>(replacements).drop_front(numDims), numDims,
        map.getNumSymbols()));

    rewriter.finalizeOpModification(apply);
    return success();
  }
};

/// Converts constant index operands into constants added after the linearize
/// if the linearization basis is static.
struct PropagateConstantAddsThroughLinearize final
    : OpRewritePattern<affine::AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    int64_t indexCount = op.getMultiIndex().size();
    int64_t runningElementCount = 1;
    int64_t runningOffset = 0;
    Value zero = nullptr;
    auto getZero = [&]() {
      if (zero)
        return zero;
      zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      return zero;
    };
    bool didReplace = false;

    auto matchConstantAndReplaceOperand = [&](OpOperand &operand) {
      APInt constant;
      Value replacement = nullptr;
      if (matchPattern(operand.get(), m_ConstantInt(&constant)) &&
          !constant.isZero()) {
        // If an operand is constant and non-zero, add its total contribution
        // to the running offset and replace the multi_index operand with 0.
        runningOffset += constant.getSExtValue() * runningElementCount;
        replacement = getZero();
      } else if (std::optional<int64_t> offset =
                     getValueConstantRhs<arith::AddIOp>(operand.get(),
                                                        /*nsw=*/true)) {
        runningOffset += offset.value() * runningElementCount;
        replacement = operand.get().getDefiningOp<arith::AddIOp>().getLhs();
      }
      if (replacement) {
        if (!didReplace) {
          rewriter.startOpModification(op);
        }
        operand.assign(replacement);
        didReplace = true;
      }
    };

    // Iterate the static basis in reverse and accumulate the constant offset
    // to add.
    for (auto [i, size] : llvm::enumerate(llvm::reverse(op.getStaticBasis()))) {
      OpOperand &operand = op.getMultiIndexMutable()[indexCount - i - 1];
      matchConstantAndReplaceOperand(operand);

      // Update the running linearization count, or break if the next size is
      // dynamic.
      if (ShapedType::isDynamic(size)) {
        runningElementCount = ShapedType::kDynamic;
        break;
      }
      runningElementCount *= size;
    }

    // Repeat one more time if the basis has one fewer entry than number of
    // indices.
    if (indexCount != op.getStaticBasis().size() &&
        ShapedType::isStatic(runningElementCount)) {
      OpOperand &operand = op.getMultiIndexMutable()[0];
      matchConstantAndReplaceOperand(operand);
    }

    if (!didReplace) {
      return rewriter.notifyMatchFailure(
          op, "no constant add multi_index operands");
    }

    rewriter.finalizeOpModification(op);

    rewriter.setInsertionPointAfter(op);
    Value offset =
        rewriter.create<arith::ConstantIndexOp>(op.getLoc(), runningOffset);
    auto addOp = rewriter.create<arith::AddIOp>(
        op.getLoc(), op.getResult(), offset, arith::IntegerOverflowFlags::nsw);
    rewriter.replaceAllUsesExcept(op, addOp, addOp);
    return success();
  }
};

/// Converts operands multiplied by a constant product into the input to the
/// mul if divisible by the basis element. For example,
///
///   %x4 = arith.muli %x, %c4 overflow<nsw> : index
///   %linearize = affine.linearize_index [..., %x4, ...] (..., 16, ...)
///
/// into
///
///   %linearize = affine.linearize_index [..., %x, %c0, ...] (..., 4, 4, ...)
struct FoldDivisibleConstantMulsIntoLinearize final
    : OpRewritePattern<affine::AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getDynamicBasis().empty()) {
      return rewriter.notifyMatchFailure(op, "unimplemented: dynamic basis");
    }
    int64_t indexCount = op.getMultiIndex().size();
    SmallVector<Value> newMultiIndex;
    SmallVector<int64_t> newStaticBasis;
    Value zero = nullptr;
    auto getZero = [&]() {
      if (zero)
        return zero;
      zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
      return zero;
    };

    // Iterate the static basis in reverse and accumulate the constant offset
    // to add.
    for (auto [i, size] : llvm::enumerate(llvm::reverse(op.getStaticBasis()))) {
      assert(ShapedType::isStatic(size) && "unexpected dynamic basis element");
      Value operand = op.getMultiIndex()[indexCount - i - 1];
      std::optional<int64_t> coefficient =
          getValueConstantRhs<arith::MulIOp>(operand, /*nsw=*/true);
      // If the basis size is indivisible by the coefficient, splitting up the
      // size by dividing by the coefficient isn't possible.
      if (coefficient && size % coefficient.value() == 0) {
        newMultiIndex.push_back(getZero());
        newMultiIndex.push_back(
            operand.getDefiningOp<arith::MulIOp>().getLhs());
        newStaticBasis.push_back(coefficient.value());
        newStaticBasis.push_back(size / coefficient.value());
      } else {
        newStaticBasis.push_back(size);
        newMultiIndex.push_back(operand);
      }
    }

    // Handle the last entry if no bound given.
    if (indexCount != op.getStaticBasis().size()) {
      Value operand = op.getMultiIndex()[0];
      std::optional<int64_t> coefficient =
          getValueConstantRhs<arith::MulIOp>(operand, /*nsw=*/true);
      // Since there is no outermost size, no need to verify divisibility.
      if (coefficient) {
        newMultiIndex.push_back(getZero());
        newMultiIndex.push_back(
            operand.getDefiningOp<arith::MulIOp>().getLhs());
        newStaticBasis.push_back(coefficient.value());
      } else {
        newMultiIndex.push_back(operand);
      }
    }

    if (newMultiIndex.size() == op.getMultiIndex().size()) {
      return rewriter.notifyMatchFailure(
          op, "no foldable constant mul multi_index operands");
    }

    SmallVector<Value> reversedIndex(llvm::reverse(newMultiIndex));
    SmallVector<int64_t> reversedBasis(llvm::reverse(newStaticBasis));

    // Disjoint status is unaffected by this transform.
    rewriter.replaceOpWithNewOp<affine::AffineLinearizeIndexOp>(
        op, reversedIndex, reversedBasis, op.getDisjoint());
    return success();
  }
};

} // namespace

void PropagateConstantOffsetsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns
      .add<PropagateConstantAddsThroughLinearize, ExtractConstantApplyOffset,
           FoldApplySymbolOrDimSum, FoldDivisibleConstantMulsIntoLinearize>(
          context);
  // Apply canonicalization to apply new composition opportunities.
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  // Add canonicalization to compose adds.
  arith::AddIOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler
