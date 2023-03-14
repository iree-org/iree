// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Patterns that have a direct lowering from TF to Linalg and IREE. For these,
// we use a high benefit and lower them directly. Some of these are temporary
// while additional work lands upstream.

#include "iree_tf_compiler/TF/Passes.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/ChloOps.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace TFOps = mlir::TF;

namespace mlir {
namespace iree_integrations {
namespace TF {

namespace {

static PatternBenefit OVERRIDE_BENEFIT = 1000;

struct ConvertExplicitSqueezePattern
    : public OpRewritePattern<TFOps::SqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFOps::SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        op.getInput().getType().dyn_cast<RankedTensorType>();
    RankedTensorType resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      // This will happen if shape inference could not determine a rank,
      // which we do not support.
      return rewriter.notifyMatchFailure(op, "not ranked result");
    }

    auto reassociationIndices =
        mlir::getReassociationIndicesForReshape(inputType, resultType);
    if (!reassociationIndices) {
      return rewriter.notifyMatchFailure(
          op, "could not compute reassociation indices");
    }

    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        op, resultType, op.getInput(), *reassociationIndices);
    return success();
  }
};

// Converts a tf.expand_dims op with a constant dim directly to a linalg
// expanding reshape.
struct ConvertConstExpandDimsPattern
    : public OpRewritePattern<TFOps::ExpandDimsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFOps::ExpandDimsOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        op.getInput().getType().dyn_cast<RankedTensorType>();
    RankedTensorType resultType = op.getType().dyn_cast<RankedTensorType>();
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "not ranked");
    }
    DenseIntElementsAttr dimAttr;
    if (!matchPattern(op.getDim(), m_Constant(&dimAttr))) {
      return rewriter.notifyMatchFailure(op, "not constant dim");
    }
    int expandDim = (*dimAttr.value_begin<APInt>()).getSExtValue();
    auto dims = llvm::to_vector<6>(resultType.getShape());
    if (expandDim < 0) {
      expandDim += dims.size();
      if (expandDim < 0) {
        return rewriter.notifyMatchFailure(op, "illegal insertion dim");
      }
    }
    if (expandDim >= dims.size()) {
      return rewriter.notifyMatchFailure(op, "illegal insertion dim");
    }
    dims[expandDim] = 1;

    RankedTensorType expandedType =
        RankedTensorType::get(dims, resultType.getElementType());

    if (expandedType != resultType) {
      return rewriter.notifyMatchFailure(
          op, "inferred expanded type not equal to result type");
    }

    auto reassociationIndices =
        mlir::getReassociationIndicesForReshape(inputType, expandedType);
    if (!reassociationIndices) {
      return rewriter.notifyMatchFailure(
          op, "could not compute reassociation indices");
    }

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, expandedType, op.getInput(), *reassociationIndices);
    return success();
  }
};

}  // namespace

void populateDirectLoweringPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns) {
  patterns.insert<ConvertConstExpandDimsPattern>(context, OVERRIDE_BENEFIT);
  patterns.insert<ConvertExplicitSqueezePattern>(context, OVERRIDE_BENEFIT);
}

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
