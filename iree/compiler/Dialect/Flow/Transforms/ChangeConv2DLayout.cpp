// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Shuffles elements in a FHWC constant tensor to follow HWCF layout.
DenseElementsAttr convertFHWCTensorToHWCF(DenseElementsAttr inputTensor) {
  if (inputTensor.isSplat()) return inputTensor;
  auto inputType = inputTensor.getType().cast<RankedTensorType>();

  ArrayRef<int64_t> inputShape = inputType.getShape();
  assert(inputShape.size() == 4);
  int64_t oc = inputShape[0], h = inputShape[1];
  int64_t w = inputShape[2], ic = inputShape[3];

  SmallVector<Attribute, 4> values;
  values.resize(inputType.getNumElements());
  for (unsigned i = 0; i < oc; ++i) {
    for (unsigned j = 0; j < h; ++j) {
      for (unsigned k = 0; k < w; ++k) {
        for (unsigned l = 0; l < ic; ++l) {
          unsigned outputIndex = j * w * ic * oc + k * ic * oc + l * oc + i;
          values[outputIndex] = inputTensor.getValue({i, j, k, l});
        }
      }
    }
  }

  SmallVector<int64_t, 4> outputShape = {h, w, ic, oc};
  auto outputType = RankedTensorType::get(
      outputShape, inputType.getElementType(), inputType.getEncoding());
  return DenseElementsAttr::get(outputType, values);
}

/// Converts linalg.conv_2d_input_nhwc_filter_ohwi_poly with constant filters
/// into linalg.conv_2d_input_nhwc_filter_hwcf.
struct ChangeConv2DFilterFromFHWCToHWCF
    : public OpRewritePattern<linalg::Conv2DInputNhwcFilterOhwiPolyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DInputNhwcFilterOhwiPolyOp convOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = cast<linalg::LinalgOp>(convOp.getOperation());
    if (linalgOp.hasBufferSemantics()) return failure();

    DenseElementsAttr fhwcTensor;
    Value input = convOp.getOperand(0), fhwcFilter = convOp.getOperand(1);
    if (!matchPattern(fhwcFilter, m_Constant(&fhwcTensor))) return failure();

    auto hwcfTensor = convertFHWCTensorToHWCF(fhwcTensor);
    auto hwcfFilter =
        rewriter.create<ConstantOp>(fhwcFilter.getLoc(), hwcfTensor);

    rewriter.replaceOpWithNewOp<linalg::ConvInputNHWCFilterHWCFOp>(
        convOp, convOp.getResultTypes(), ValueRange{input, hwcfFilter},
        ValueRange{convOp.getOperand(2)}, convOp.dilationsAttr(),
        convOp.stridesAttr());

    return success();
  }
};

struct ChangeConv2DLayoutPass
    : public ChangeConv2DLayoutBase<ChangeConv2DLayoutPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ChangeConv2DFilterFromFHWCToHWCF>(context);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createChangeConv2DLayoutPass() {
  return std::make_unique<ChangeConv2DLayoutPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
