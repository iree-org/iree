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

#include "iree_tf_compiler/dialect/tf_strings/conversion/convert_tf_strings_to_strings.h"

#include "iree_tf_compiler/dialect/tf_strings/ir/dialect.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/ops.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/types.h"
#include "iree_tf_compiler/dialect/utils/conversion_utils.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Ops.h"
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
namespace tf_strings {

namespace {

class StringTypeConverter : public TypeConverter {
 public:
  StringTypeConverter() {
    // Required to covert any unknown or already converted types.
    addConversion([](Type type) { return type; });
    addConversion([](tf_strings::StringType type) {
      return iree_compiler::IREE::Strings::StringType::get(type.getContext());
    });
    addConversion([](TensorType type) -> Type {
      if (type.getElementType().isa<tf_strings::StringType>()) {
        return iree_compiler::IREE::Strings::StringTensorType::get(
            type.getContext());
      }
      return type;
    });
  }
};

class ConvertTFStringsToStringsPass
    : public ConversionPass<ConvertTFStringsToStringsPass,
                            StringTypeConverter> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<TFStringsDialect, iree_compiler::IREE::Strings::StringsDialect,
                StandardOpsDialect>();
  }

  void Setup(ConversionTarget &target,
             OwningRewritePatternList &patterns) override {
    target.addIllegalDialect<tf_strings::TFStringsDialect>();
    target.addLegalDialect<iree_compiler::IREE::Strings::StringsDialect>();
    populateTFStringsToStringsPatterns(&this->getContext(), patterns);
  }
};
}  // namespace

void populateTFStringsToStringsPatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns) {
  patterns.insert<
      OpConversion<tf_strings::PrintOp, iree_compiler::IREE::Strings::PrintOp>>(
      context);
  patterns.insert<OpConversion<tf_strings::ToStringTensorOp,
                               iree_compiler::IREE::Strings::ToStringTensorOp>>(
      context);
  patterns.insert<OpConversion<tf_strings::GatherOp,
                               iree_compiler::IREE::Strings::GatherOp>>(
      context);
  patterns.insert<
      OpConversion<tf_strings::StringTensorToStringOp,
                   iree_compiler::IREE::Strings::StringTensorToStringOp>>(
      context);
  patterns.insert<OpConversion<tf_strings::ToStringOp,
                               iree_compiler::IREE::Strings::I32ToStringOp>>(
      context);
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFStringsToStringsPass() {
  return std::make_unique<ConvertTFStringsToStringsPass>();
}

static PassRegistration<ConvertTFStringsToStringsPass> pass(
    "iree-tf-strings-convert-to-strings",
    "Converts TF string ops to the IREE tf_strings dialect");

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir
