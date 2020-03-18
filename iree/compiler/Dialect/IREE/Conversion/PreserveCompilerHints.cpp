// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/IREE/Conversion/PreserveCompilerHints.h"

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {
class PreserveDoNotOptimize
    : public OpConversionPattern<IREE::DoNotOptimizeOp> {
 public:
  using OpConversionPattern<IREE::DoNotOptimizeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      IREE::DoNotOptimizeOp op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<IREE::DoNotOptimizeOp>(op, operands,
                                                       op.getAttrs());
    return success();
  }
};
}  // namespace

void setupCompilerHintsLegality(MLIRContext *context, ConversionTarget &target,
                                TypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<IREE::DoNotOptimizeOp>(
      [&](IREE::DoNotOptimizeOp op) {
        return llvm::all_of(op.getResultTypes(), [&typeConverter](Type t) {
          return typeConverter.isLegal(t);
        });
      });
}

void populatePreserveCompilerHintsPatterns(MLIRContext *context,
                                           OwningRewritePatternList &patterns) {
  patterns.insert<PreserveDoNotOptimize>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
