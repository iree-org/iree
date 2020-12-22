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

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_integrations {

template <typename SRC, typename DST>
class OpConversion : public OpConversionPattern<SRC> {
 public:
  OpConversion(MLIRContext *context) : OpConversionPattern<SRC>(context) {}

  LogicalResult matchAndRewrite(
      SRC srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto operation = srcOp.getOperation();
    rewriter.replaceOpWithNewOp<DST>(srcOp, operation->getResultTypes(),
                                     operands, operation->getAttrs());
    return success();
  }
};

template <typename T, typename Converter>
class ConversionPass : public PassWrapper<T, OperationPass<ModuleOp>> {
 public:
  virtual void Setup(ConversionTarget &target,
                     OwningRewritePatternList &pattern) = 0;

  void runOnOperation() override {
    if (failed(run())) {
      this->signalPassFailure();
    }
  }

  LogicalResult run() {
    auto module = this->getOperation();
    OwningRewritePatternList patterns;
    Converter typeConverter;

    // Lower to the standard string operations.
    ConversionTarget target(this->getContext());
    Setup(target, patterns);

    // Add Dynamic legal ops for calls, returns, and functions.
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      Converter typeConverter;
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
      Converter typeConverter;
      auto func = [&](Type type) { return typeConverter.isLegal(type); };
      return llvm::all_of(op.getOperandTypes(), func);
    });

    target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
      Converter typeConverter;
      auto func = [&](Type type) { return typeConverter.isLegal(type); };
      return llvm::all_of(op.getOperandTypes(), func) &&
             llvm::all_of(op.getResultTypes(), func);
    });

    populateFuncOpTypeConversionPattern(patterns, &this->getContext(),
                                        typeConverter);
    populateCallOpTypeConversionPattern(patterns, &this->getContext(),
                                        typeConverter);

    auto result = applyPartialConversion(module.getOperation(), target,
                                         std::move(patterns));

    // Partial conversion doesn't include return types. Update in a separate
    // walk.
    module.walk([&](Operation *op) {
      for (auto result : op->getResults()) {
        auto result_type = result.getType();
        auto new_type = typeConverter.convertType(result_type);
        if (new_type) {
          result.setType(typeConverter.convertType(result_type));
        }
      }
    });

    return result;
  }
};

}  // namespace iree_integrations
}  // namespace mlir
