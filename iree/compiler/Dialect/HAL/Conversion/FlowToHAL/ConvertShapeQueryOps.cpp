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
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

static Value getBackingView(Value v) {
  if (v.getType().isa<IREE::HAL::BufferViewType>()) {
    return v;
  } else if (auto bufferViewBufferOp =
                 llvm::dyn_cast_or_null<IREE::HAL::BufferViewBufferOp>(
                     v.getDefiningOp())) {
    return bufferViewBufferOp.buffer_view();
  }
  return {};
}

// Matches:
//   %0 = ... buffer_view
//   %1 = dim %0, ...
// or
//   %0 = ... buffer_view ...
//   %1 = hal.buffer_view_buffer %0
//   %2 = dim %1, ...
// Note that such patterns are not legal but are created in various custom op
// conversions.
class BackingBufferBufferViewDimPattern : public OpConversionPattern<DimOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      DimOp dimOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    DimOpOperandAdaptor operands(rawOperands);
    auto view = getBackingView(operands.memrefOrTensor());
    if (!view) return failure();

    auto dimIndex = rewriter.getI32IntegerAttr(dimOp.getIndex());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewDimOp>(
        dimOp, dimOp.getResult().getType(), view, dimIndex);
    return success();
  }
};

// Matches:
//   %0 = ... buffer_view
//   %1 = rank %0, ...
// or
//   %0 = ... buffer_view ...
//   %1 = hal.buffer_view_buffer %0
//   %2 = rank %1, ...
// Note that such a pattern is not legal but is created in various custom op
// conversions.
class BackingBufferBufferViewRankPattern : public OpConversionPattern<RankOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RankOp rankOp, llvm::ArrayRef<Value> rawOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto view = getBackingView(rawOperands[0]);
    if (!view) return failure();

    rewriter.replaceOpWithNewOp<IREE::HAL::BufferViewRankOp>(
        rankOp, rankOp.getResult().getType(), view);
    return success();
  }
};

}  // namespace

void populateHalBufferViewShapePatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns,
                                        TypeConverter &converter) {
  patterns.insert<BackingBufferBufferViewDimPattern,
                  BackingBufferBufferViewRankPattern>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
