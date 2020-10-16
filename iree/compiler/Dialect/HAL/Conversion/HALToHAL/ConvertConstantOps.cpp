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
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {

class ConstantSubspanConversion
    : public OpConversionPattern<IREE::HAL::ConstantSubspanOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::ConstantSubspanOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto bufferValue = rewriter.createOrFold<IREE::HAL::VariableLoadOp>(
        op.getLoc(), IREE::HAL::BufferType::get(rewriter.getContext()),
        op.runtime_buffer().getLeafReference());
    auto offsetValue = rewriter.createOrFold<mlir::ConstantOp>(
        op.getLoc(), op.runtime_range().offsetAttr());
    auto lengthValue = rewriter.createOrFold<mlir::ConstantOp>(
        op.getLoc(), op.runtime_range().lengthAttr());
    rewriter.replaceOpWithNewOp<IREE::HAL::BufferSubspanOp>(
        op, IREE::HAL::BufferType::get(rewriter.getContext()), bufferValue,
        offsetValue, lengthValue);
    return success();
  }
};

}  // namespace

void populateHALConstantToHALPatterns(MLIRContext *context,
                                      OwningRewritePatternList &patterns,
                                      TypeConverter &typeConverter) {
  patterns.insert<ConstantSubspanConversion>(typeConverter, context);
}

}  // namespace iree_compiler
}  // namespace mlir
