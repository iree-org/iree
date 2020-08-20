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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
       Value dim = rewriter.create<ExtractElementOp>(op.getLoc(),
         inputTy.getElementType(), input, index.getResult());
       if (!dim.getType().isIndex()) {
         dim = rewriter.create<IndexCastOp>(op.getLoc(), rewriter.getIndexType(), dim);
       }
       extracted_elements.push_back(dim);
     }

     SmallVector<int64_t, 4> dims;
     dims.resize(valueCount, -1);
     rewriter.replaceOpWithNewOp<Shape::MakeRankedShapeOp>(
       op, Shape::RankedShapeType::get(dims, op.getContext()), extracted_elements);

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
}  // naemspace iree_compiler
}  // namespace mlir