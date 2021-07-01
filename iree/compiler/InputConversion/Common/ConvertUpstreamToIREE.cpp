// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/InputConversion/Common/PassDetail.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct ConvertTensorCastPattern : public OpConversionPattern<tensor::CastOp> {
  using OpConversionPattern<tensor::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tensor::CastOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = operands.front();
    ShapedType inputType = input.getType().dyn_cast<ShapedType>();
    ShapedType resultType =
        typeConverter->convertType(op.getType()).dyn_cast_or_null<ShapedType>();
    if (!inputType || !resultType || !inputType.hasRank() ||
        !resultType.hasRank()) {
      return rewriter.notifyMatchFailure(op, "not ranked shaped types");
    }
    // This should not happen, except in the context of type conversion.
    if (inputType.getRank() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(op, "mismatched rank");
    }

    // Resolve dims to the most specific value.
    int rank = inputType.getRank();
    SmallVector<Value> dimSizes(rank);
    auto resolveDimSize = [&](int position) -> Value {
      if (!dimSizes[position]) {
        // Find the most specific.
        if (!inputType.isDynamicDim(position) ||
            !resultType.isDynamicDim(position)) {
          // Static dim.
          int64_t dimSize = !inputType.isDynamicDim(position)
                                ? inputType.getDimSize(position)
                                : resultType.getDimSize(position);
          dimSizes[position] = rewriter.create<ConstantIndexOp>(loc, dimSize);
        } else {
          // Dynamic dim.
          dimSizes[position] =
              rewriter.create<tensor::DimOp>(loc, input, position);
        }
      }

      return dimSizes[position];
    };

    SmallVector<Value> sourceDynamicDims;
    SmallVector<Value> targetDynamicDims;
    for (int i = 0; i < rank; i++) {
      if (inputType.isDynamicDim(i)) {
        sourceDynamicDims.push_back(resolveDimSize(i));
      }
      if (resultType.isDynamicDim(i)) {
        targetDynamicDims.push_back(resolveDimSize(i));
      }
    }

    // TODO: Decide if this needs to be replaced with a flow.tensor.cast
    // See https://github.com/google/iree/issues/6418
    rewriter.replaceOpWithNewOp<IREE::Flow::TensorReshapeOp>(
        op, resultType, input, sourceDynamicDims, targetDynamicDims);

    return success();
  }
};

}  // namespace

void populateConvertUpstreamToIREEPatterns(MLIRContext *context,
                                           TypeConverter &typeConverter,
                                           OwningRewritePatternList &patterns) {
  patterns.add<ConvertTensorCastPattern>(typeConverter, context);
}

namespace {

struct ConvertUpstreamToIREEPass
    : public ConvertUpstreamToIREEBase<ConvertUpstreamToIREEPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, tensor::TensorDialect,
                    IREE::Flow::FlowDialect>();
  }

  void runOnOperation() override;
};

}  // namespace

void ConvertUpstreamToIREEPass::runOnOperation() {
  OwningRewritePatternList patterns(&getContext());
  MLIRContext *context = &getContext();
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type t) { return t; });
  populateConvertUpstreamToIREEPatterns(&getContext(), typeConverter, patterns);

  ConversionTarget target(*context);
  target.addIllegalOp<tensor::CastOp>();

  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<IREE::Flow::FlowDialect>();

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>> createConvertUpstreamToIREE() {
  return std::make_unique<ConvertUpstreamToIREEPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
