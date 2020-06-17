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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

// Legalize the type from operand() -> result() for tie_shape op.
// At this level, we preserve any remaining tie_shapes since they may still
// provide information in some contexts.
class LegalizeTieShapePattern : public OpConversionPattern<Shape::TieShapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      Shape::TieShapeOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<Shape::TieShapeOp>(op, operands[0],
                                                   operands[1]);
    return success();
  }
};

// Lowers dim operations against values that were originally tensors but have
// been converted to HAL buffer types.
class BackingBufferBufferViewDimPattern : public OpConversionPattern<DimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DimOp dimOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    DimOp::Adaptor operands(rawOperands);
    if (!dimOp.memrefOrTensor().getType().isa<TensorType>() ||
        !IREE::HAL::TensorRewriteAdaptor::isValidNewType(
            operands.memrefOrTensor().getType())) {
      return failure();
    }
    auto adaptor = IREE::HAL::TensorRewriteAdaptor::get(
        dimOp.getLoc(), dimOp.memrefOrTensor(), operands.memrefOrTensor(),
        rewriter);

    Optional<int64_t> index = dimOp.getConstantIndex();
    assert(index.hasValue() && "expect constant index in `std.dim` operation");

    auto dimIndex = rewriter.getI32IntegerAttr(index.getValue());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewDimOp>(
        dimOp, dimOp.getResult().getType(), adaptor.getBufferView(), dimIndex);
    return success();
  }
};

// Lowers rank operations against values that were originally tensors but have
// been converted to HAL buffer types.
class BackingBufferBufferViewRankPattern : public OpConversionPattern<RankOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RankOp rankOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    if (!IREE::HAL::TensorRewriteAdaptor::isValidNewType(
            rawOperands[0].getType())) {
      return failure();
    }
    auto adaptor = IREE::HAL::TensorRewriteAdaptor::get(
        rankOp.getLoc(), rankOp.getOperand(), rawOperands[0], rewriter);

    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewRankOp>(
        rankOp, rankOp.getResult().getType(), adaptor.getBufferView());
    return success();
  }
};

}  // namespace

void populateHalBufferViewShapePatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns,
                                        TypeConverter &converter) {
  patterns.insert<BackingBufferBufferViewDimPattern,
                  BackingBufferBufferViewRankPattern, LegalizeTieShapePattern>(
      context);
}

}  // namespace iree_compiler
}  // namespace mlir
