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
#include "mlir/Conversion/StandardToStandard/StandardToStandard.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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

namespace {

#include "integrations/tensorflow/compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.inc"

class StringTypeConverter : public TypeConverter {
 public:
  StringTypeConverter() {
    addConversion([](RankedTensorType type) -> Type {
      if (type.getElementType().isa<TF::StringType>()) {
        auto elementType = TFStrings::StringType::get(type.getContext());
        return RankedTensorType::get(type.getShape(), elementType);
      }

      return type;
    });
    addConversion([](TF::StringType type) {
      return TFStrings::StringType::get(type.getContext());
    });
  }

  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value> inputs,
                                   Location loc) override {
    llvm_unreachable("unhandled materialization");
    return nullptr;
  }
};

class LowerTensorflowToStringsPass
    : public ModulePass<LowerTensorflowToStringsPass> {
 public:
  void runOnModule() override {
    if (failed(run())) {
      signalPassFailure();
    }
  }
  LogicalResult run() {
    auto module = getModule();
    OpBuilder builder(module.getContext());
    OwningRewritePatternList patterns;
    StringTypeConverter typeConverter;

    // Lower to the standard string operations.
    ConversionTarget target(getContext());
    target.addIllegalOp<TF::AsStringOp>();
    target.addIllegalOp<TF::PrintV2Op>();
    target.addLegalDialect<TFStrings::TFStringsDialect>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      StringTypeConverter typeConverter;
      return typeConverter.isSignatureLegal(op.getType());
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

    auto result = applyPartialConversion(module.getOperation(), target,
                                         patterns, &typeConverter);

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
}

std::unique_ptr<OpPassBase<ModuleOp>> createLowerTensorflowToStringsPass() {
  return std::make_unique<LowerTensorflowToStringsPass>();
}

static PassRegistration<LowerTensorflowToStringsPass> pass(
    "convert-tensorflow-to-tf-strings", "Lower tensorflow to tf-strings.");

}  // namespace iree_compiler
}  // namespace mlir
