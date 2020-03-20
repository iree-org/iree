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

struct AsStringOpLowering : public OpRewritePattern<TF::AsStringOp> {
  using OpRewritePattern<TF::AsStringOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::AsStringOp op,
                                PatternRewriter &rewriter) const override {
    Value replacement = op.input();
    replacement =
        rewriter.create<TFStrings::ToStringOp>(op.getLoc(), replacement);
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct PrintOpLowering : public OpRewritePattern<TF::PrintV2Op> {
  using OpRewritePattern<TF::PrintV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::PrintV2Op op,
                                PatternRewriter &rewriter) const override {
    auto input = op.input();
    rewriter.create<TFStrings::PrintOp>(op.getLoc(), input);
    rewriter.eraseOp(op);
    return success();
  }
};

class StringTypeConverter : public TypeConverter {
 public:
  StringTypeConverter() {
    addConversion([](ShapedType type) {
      auto elementType = TFStrings::StringType::get(type.getContext());
      return RankedTensorType::get(type.getShape(), elementType);
    });
    addConversion([](TF::StringType type) {
      return TFStrings::StringType::get(type.getContext());
    });
    addConversion([](Type type) -> Optional<Type> {
      if (!getElementTypeOrSelf(type).isa<TF::StringType>()) {
        return type;
      }
      return llvm::None;
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

    target.addLegalDialect<TFStrings::TFStringsDialect>();
    target.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
      StringTypeConverter typeConverter;
      return typeConverter.isSignatureLegal(op.getType());
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
    return result;
  }
};

}  // namespace

void populateTFToTFStringsPatterns(MLIRContext *ctx,
                                   OwningRewritePatternList &patterns) {
  patterns.insert<AsStringOpLowering, PrintOpLowering>(ctx);
}

std::unique_ptr<OpPassBase<ModuleOp>> createLowerTensorflowToStringsPass() {
  return std::make_unique<LowerTensorflowToStringsPass>();
}

static PassRegistration<LowerTensorflowToStringsPass> pass(
    "convert-tensorflow-to-tf-strings", "Lower tensorflow to tf-strings.");

}  // namespace iree_compiler
}  // namespace mlir
