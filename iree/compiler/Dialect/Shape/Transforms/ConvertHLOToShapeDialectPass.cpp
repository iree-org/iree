// Copyright 2020 Google LLC
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

#include <numeric>

#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {
namespace {

// Returns a 1-d i64 elements attribute populated with numbers from start to
// end, excluding.
static DenseIntElementsAttr getI64ElementsAttrForSeq(int start, int end,
                                                     Builder &builder) {
  int size = end - start;

  SmallVector<int64_t, 4> vals;
  vals.resize(size);
  std::iota(vals.begin(), vals.end(), start);

  TensorType ty = RankedTensorType::get({size}, builder.getIntegerType(64));
  return DenseIntElementsAttr::get(ty, vals);
}

class ConvertDynamicBroadcastInDim
    : public OpConversionPattern<xla_hlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      xla_hlo::DynamicBroadcastInDimOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    xla_hlo::DynamicBroadcastInDimOpOperandAdaptor adapter(operands);
    Value rankedShape = rewriter.create<Shape::FromExtentTensorOp>(
        op.getLoc(), adapter.output_dimensions());
    rewriter.replaceOpWithNewOp<Shape::RankedBroadcastInDimOp>(
        op, op.getType(), adapter.operand(), rankedShape,
        op.broadcast_dimensions());
    return success();
  }
};

class ConvertHLOToShapePass
    : public PassWrapper<ConvertHLOToShapePass, FunctionPass> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;

    conversionTarget.addLegalDialect<ShapeDialect>();
    conversionTarget.addLegalDialect<StandardOpsDialect>();
    conversionTarget.addLegalDialect<xla_hlo::XlaHloDialect>();

    conversionTarget.addIllegalOp<xla_hlo::DynamicBroadcastInDimOp>();
    conversionPatterns.insert<ConvertDynamicBroadcastInDim>(&getContext());

    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

// Converts shape-sensitive HLOs to be based on facilities in the shape
// dialect.
std::unique_ptr<OperationPass<FuncOp>> createConvertHLOToShapePass() {
  return std::make_unique<Shape::ConvertHLOToShapePass>();
}

static PassRegistration<Shape::ConvertHLOToShapePass> pass(
    "iree-shape-convert-hlo",
    "Converts dynamic shape dependent HLO ops to shaped variants.");

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
