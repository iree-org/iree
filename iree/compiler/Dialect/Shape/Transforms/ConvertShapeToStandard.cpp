// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

namespace {

class ConvertFromExtent : public OpConversionPattern<FromExtentTensorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      FromExtentTensorOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto input = op.extent_tensor();
    ShapedType inputTy = input.getType().cast<ShapedType>();
    if (!inputTy.hasRank() || inputTy.getRank() != 1) {
      return failure();
    }

    llvm::SmallVector<Value, 4> extracted_elements;
    auto valueCount = inputTy.getDimSize(0);
    extracted_elements.reserve(valueCount);
    for (int i = 0; i < valueCount; i++) {
      auto index = rewriter.create<ConstantIndexOp>(op.getLoc(), i);
      Value dim = rewriter.create<tensor::ExtractOp>(
          op.getLoc(), inputTy.getElementType(), input, index.getResult());
      if (!dim.getType().isIndex()) {
        dim = rewriter.create<IndexCastOp>(op.getLoc(), rewriter.getIndexType(),
                                           dim);
      }
      extracted_elements.push_back(dim);
    }

    SmallVector<int64_t, 4> dims;
    dims.resize(valueCount, -1);
    rewriter.replaceOpWithNewOp<Shape::MakeRankedShapeOp>(
        op, Shape::RankedShapeType::get(dims, op.getContext()),
        extracted_elements);

    return success();
  }
};

}  // namespace

// Populates patterns that will convert shape calculations into standard ops.
void populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ConvertFromExtent>(context);
}

// Sets up legality for shape calculation materialization conversions.
void setupShapeToStandardLegality(ConversionTarget &target) {
  target.addIllegalOp<FromExtentTensorOp>();
  target.addLegalOp<Shape::MakeRankedShapeOp>();
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
