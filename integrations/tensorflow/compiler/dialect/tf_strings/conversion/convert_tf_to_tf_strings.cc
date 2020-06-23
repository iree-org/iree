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

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.h"

#include <cstddef>

#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/dialect.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/ops.h"
#include "integrations/tensorflow/compiler/dialect/tf_strings/ir/types.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_compiler {
namespace tf_strings {

namespace {

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.inc"

class StringTypeConverter : public TypeConverter {
 public:
  StringTypeConverter() {
    // Required to covert any unknown or already converted types.
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> Type {
      if (type.getElementType().isa<TF::StringType>()) {
        auto elementType = tf_strings::StringType::get(type.getContext());
        // TODO(suderman): Find a better way to identify tensor<!tf.string> and
        // !tf.string.
        // Tensorflow only operates on tensors, so "scalar" strings are actually
        // rank-0 tensors of strings. For now separate operating on tensors of
        // strings and scalar strings by forcing all rank-0 tensors of strings
        // to strings.
        if (type.getRank() == 0) {
          return tf_strings::StringType::get(type.getContext());
        }
        return RankedTensorType::get(type.getShape(), elementType);
      }

      return type;
    });
    addConversion([](TF::StringType type) {
      return tf_strings::StringType::get(type.getContext());
    });
  }
};

struct StringFormatOpLowering : public OpRewritePattern<TF::StringFormatOp> {
  using OpRewritePattern<TF::StringFormatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::StringFormatOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.inputs();
    // TODO(suderman): Implement a variadic version. For now assume one input.
    if (inputs.size() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "Variadic StringFormat unsupported.");

    auto input = inputs[0];

    rewriter.replaceOpWithNewOp<tf_strings::StringTensorToStringOp>(op, input);
    return success();
  }
};

class LowerTensorflowToStringsPass
    : public PassWrapper<LowerTensorflowToStringsPass,
                         OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override {
    if (failed(run())) {
      signalPassFailure();
    }
  }
  LogicalResult run() {
    auto module = getOperation();
    OpBuilder builder(module.getContext());
    OwningRewritePatternList patterns;
    StringTypeConverter typeConverter;

    // Lower to the standard string operations.
    ConversionTarget target(getContext());
    target.addIllegalOp<TF::AsStringOp>();
    target.addIllegalOp<TF::PrintV2Op>();
    target.addLegalDialect<tf_strings::TFStringsDialect>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      StringTypeConverter typeConverter;
      return typeConverter.isSignatureLegal(op.getType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    target.addDynamicallyLegalOp<ReturnOp>([](ReturnOp op) {
      StringTypeConverter typeConverter;
      auto func = [&](Type type) { return typeConverter.isLegal(type); };
      return llvm::all_of(op.getOperandTypes(), func);
    });

    target.addDynamicallyLegalOp<CallOp>([](CallOp op) {
      StringTypeConverter typeConverter;
      auto func = [&](Type type) { return typeConverter.isLegal(type); };
      return llvm::all_of(op.getOperandTypes(), func) &&
             llvm::all_of(op.getResultTypes(), func);
    });

    populateFuncOpTypeConversionPattern(patterns, &getContext(), typeConverter);
    populateCallOpTypeConversionPattern(patterns, &getContext(), typeConverter);
    populateTFToTFStringsPatterns(&getContext(), patterns);

    auto result =
        applyPartialConversion(module.getOperation(), target, patterns);

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

}  // namespace

void populateTFToTFStringsPatterns(MLIRContext *ctx,
                                   OwningRewritePatternList &patterns) {
  populateWithGenerated(ctx, &patterns);
  patterns.insert<StringFormatOpLowering>(ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTfToTfStrings() {
  return std::make_unique<LowerTensorflowToStringsPass>();
}

static PassRegistration<LowerTensorflowToStringsPass> pass(
    "convert-tensorflow-to-tf-strings", "Lower tensorflow to tf-strings.");

}  // namespace tf_strings
}  // namespace iree_compiler
}  // namespace mlir
