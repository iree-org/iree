// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- SetEncoding.cpp -------------------------------------===//
// Sets the encoding for compute operations to allow execution of the
// operations in tiled layouts.
//===---------------------------------------------------------------------===//

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

// Returns the element type of `t` if it is a `ShapedType`, else return
// `t` itself.
static Type getElementTypeOrType(Type t) {
  if (auto shapedType = t.dyn_cast<ShapedType>()) {
    return shapedType.getElementType();
  }
  return t;
}

/// Returns a constant 0 of type `elementType`.
static FailureOr<Value> getZero(OpBuilder &builder, Location loc,
                                Type elementType) {
  Attribute zeroVal =
      TypeSwitch<Type, Attribute>(elementType)
          .Case<FloatType>([&](FloatType floatType) -> Attribute {
            return builder.getFloatAttr(floatType, 0);
          })
          .Case<IntegerType>([&](IntegerType intType) -> Attribute {
            return builder.getIntegerAttr(intType, 0);
          })
          .Default([](Type type) { return nullptr; });
  if (!zeroVal) return failure();
  return builder.create<arith::ConstantOp>(loc, zeroVal, elementType)
      .getResult();
}

/// Pads `value` to `padding` if needed. If no padding is specified,
/// return `value` itself.
static FailureOr<Value> padIfNeeded(OpBuilder &builder, Location loc,
                                    Value value,
                                    Optional<int64_t> padding = llvm::None) {
  if (!padding) return value;

  OpFoldResult paddingOfr = builder.getIndexAttr(padding.value());
  FailureOr<SmallVector<OpFoldResult>> shape =
      LinalgExt::getDims(builder, loc, value);
  if (failed(shape)) {
    return failure();
  }

  OpFoldResult zero = builder.getIndexAttr(0);
  SmallVector<OpFoldResult> lowPad(shape->size(), zero);
  SmallVector<OpFoldResult> highPad(shape->size(), zero);

  // The low padding is always zero. The high padding is
  // shape.ceildDiv(padding) - shape.
  AffineExpr paddingExpr, shapeExpr;
  bindSymbols(builder.getContext(), paddingExpr, shapeExpr);
  AffineExpr highPadExpr =
      shapeExpr.ceilDiv(paddingExpr) * paddingExpr - shapeExpr;
  for (auto shape : llvm::enumerate(shape.value())) {
    highPad[shape.index()] = makeComposedFoldedAffineApply(
        builder, loc, highPadExpr, {paddingOfr, shape.value()});
  }

  // If all high padding evaluate to 0, then nothing to do.
  if (llvm::all_of(highPad, [](OpFoldResult ofr) {
        return isConstantIntValue(ofr, 0);
      })) {
    return value;
  }

  FailureOr<Value> zeroVal =
      getZero(builder, loc, getElementTypeOrSelf(value.getType()));
  if (failed(zeroVal)) {
    return failure();
  }
  auto padOp = builder.create<tensor::PadOp>(loc, /*resultType=*/nullptr, value,
                                             lowPad, highPad, zeroVal.value());
  return padOp.getResult();
}

namespace {

/// Rewrites the matmul op to work on tensors with encoding. Optionally
/// also pads the operands.
struct SetMatmulEncoding : public OpRewritePattern<linalg::MatmulOp> {
  SetMatmulEncoding(MLIRContext *context, int64_t padding,
                    PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        padding(padding) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasTensorSemantics()) return failure();
    auto inputs = matmulOp.getDpsInputOperands();
    auto outputs = matmulOp.getDpsInitOperands();
    auto hasEncoding = [](OpOperand *operand) -> bool {
      auto type = operand->get().getType().dyn_cast<RankedTensorType>();
      return type && type.getEncoding();
    };
    if (llvm::any_of(inputs, hasEncoding) ||
        llvm::any_of(outputs, hasEncoding)) {
      return failure();
    }
    Location loc = matmulOp.getLoc();

    // Set encoding for LHS (pad if necessary)
    FailureOr<Value> lhs =
        padIfNeeded(rewriter, loc, inputs[0]->get(), padding);
    if (failed(lhs)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad lhs");
    }
    Value lhsEncoding = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, lhs.value(), IREE::LinalgExt::TensorEncoding::GEMM_LHS);

    // Set encoding for RHS (pad if necessary)
    FailureOr<Value> rhs =
        padIfNeeded(rewriter, loc, inputs[1]->get(), padding);
    if (failed(rhs)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad rhs");
    }
    Value rhsEncoding = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, rhs.value(), IREE::LinalgExt::TensorEncoding::GEMM_RHS_TRANSPOSE);

    // Set encoding for OUTS (pad if necessary)
    FailureOr<Value> output =
        padIfNeeded(rewriter, loc, outputs[0]->get(), padding);
    if (failed(output)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to pad output");
    }
    Value outsEncoding = rewriter.create<IREE::LinalgExt::SetEncodingOp>(
        loc, output.value(), IREE::LinalgExt::TensorEncoding::GEMM_RESULT);

    auto matmulTiled = rewriter.create<linalg::MatmulOp>(
        loc, outsEncoding.getType(), ValueRange{lhsEncoding, rhsEncoding},
        outsEncoding);
    auto unsetEncoding = rewriter.create<IREE::LinalgExt::UnsetEncodingOp>(
        loc, matmulTiled.getResult(0));

    Value replacement = unsetEncoding.getResult();
    // If the output was padded, extract the actual output.
    if (output.value() != outputs[0]->get()) {
      auto replacementRank =
          replacement.getType().cast<RankedTensorType>().getRank();
      // Offsets are all 0.
      OpFoldResult zero = rewriter.getIndexAttr(0);
      SmallVector<OpFoldResult> offsets(replacementRank, zero);
      // Strides are all 1.
      OpFoldResult one = rewriter.getIndexAttr(1);
      SmallVector<OpFoldResult> strides(replacementRank, one);

      // Sizes are computed by original output size.
      FailureOr<SmallVector<OpFoldResult>> sizes =
          LinalgExt::getDims(rewriter, loc, outputs[0]->get());
      if (failed(sizes)) {
        return rewriter.notifyMatchFailure(matmulOp,
                                           "failed to get shape of result");
      }
      replacement = rewriter.create<tensor::ExtractSliceOp>(
          loc, replacement, offsets, sizes.value(), strides);
    }

    rewriter.replaceOp(matmulOp, replacement);
    return success();
  }

 private:
  int64_t padding;
};

struct SetEncodingPass : public SetEncodingBase<SetEncodingPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void SetEncodingPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<SetMatmulEncoding>(context, defaultPadding);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createSetEncodingPass() {
  return std::make_unique<SetEncodingPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
