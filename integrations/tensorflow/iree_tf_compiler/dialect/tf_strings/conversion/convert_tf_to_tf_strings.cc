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

#include "iree_tf_compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.h"

#include <cstddef>

#include "iree_tf_compiler/dialect/tf_strings/ir/dialect.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/ops.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/types.h"
#include "iree_tf_compiler/dialect/utils/conversion_utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

namespace {

#include "iree_tf_compiler/dialect/tf_strings/conversion/convert_tf_to_tf_strings.inc"

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

struct GatherV2OpLowering : public OpRewritePattern<TF::GatherV2Op> {
  using OpRewritePattern<TF::GatherV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::GatherV2Op op,
                                PatternRewriter &rewriter) const override {
    auto tensor = op.params();
    auto tensorTy = tensor.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy || !tensorTy.getElementType().isa<TF::StringType>()) {
      return failure();
    }

    DenseIntElementsAttr axis;
    if (!matchPattern(op.axis(), m_Constant(&axis))) {
      return failure();
    }

    if (axis.getType().cast<ShapedType>().getRank() != 0) {
      return failure();
    }

    auto axisValue = axis.getValue<IntegerAttr>({});
    auto axisInt = axisValue.getValue().getZExtValue();

    if (axisInt != tensorTy.getRank() - 1) {
      return failure();
    }

    auto resultTy = op.getType().cast<ShapedType>();
    rewriter.replaceOpWithNewOp<tf_strings::GatherOp>(
        op,
        RankedTensorType::get(resultTy.getShape(),
                              tf_strings::StringType::get(op.getContext())),
        tensor, op.indices());

    return success();
  }
};

class ConvertTFToTFStringsPass
    : public ConversionPass<ConvertTFToTFStringsPass, StringTypeConverter> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect, TFStringsDialect,
                    StandardOpsDialect>();
  }

  void Setup(ConversionTarget &target,
             OwningRewritePatternList &patterns) override {
    target.addIllegalOp<TF::AsStringOp>();
    target.addIllegalOp<TF::PrintV2Op>();
    target.addLegalDialect<tf_strings::TFStringsDialect>();

    populateTFToTFStringsPatterns(&this->getContext(), patterns);
  }
};

}  // namespace

void populateTFToTFStringsPatterns(MLIRContext *ctx,
                                   OwningRewritePatternList &patterns) {
  populateWithGenerated(ctx, patterns);
  patterns.insert<GatherV2OpLowering>(ctx);
  patterns.insert<StringFormatOpLowering>(ctx);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFToTFStringsPass() {
  return std::make_unique<ConvertTFToTFStringsPass>();
}

static PassRegistration<ConvertTFToTFStringsPass> pass(
    "iree-tf-convert-to-tf-strings",
    "Converts TF string ops to the IREE tf_strings dialect");

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir
