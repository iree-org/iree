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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_tf_tensorlist_to_tensorlist.h"

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "integrations/tensorflow/compiler/dialect/utils/conversion_utils.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace tf_tensorlist {

namespace {

class TensorListTypeConverter : public TypeConverter {
 public:
  TensorListTypeConverter() {
    // Required to covert any unknown or already converted types.
    addConversion([](Type type) { return type; });
    addConversion([](tf_tensorlist::TensorListType type) {
      return IREE::TensorList::TensorListType::get(type.getContext());
    });
  }
};

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter);

void populateTFTensorListToTensorListPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<
      OpConversion<tf_tensorlist::Reserve, IREE::TensorList::ReserveTensor>>(
      context);
  patterns
      .insert<OpConversion<tf_tensorlist::GetItem, IREE::TensorList::GetItem>>(
          context);
  patterns
      .insert<OpConversion<tf_tensorlist::SetItem, IREE::TensorList::SetItem>>(
          context);
  patterns.insert<
      OpConversion<tf_tensorlist::FromTensor, IREE::TensorList::FromTensor>>(
      context);
  patterns.insert<
      OpConversion<tf_tensorlist::Concat, IREE::TensorList::ConcatTensor>>(
      context);
  patterns.insert<
      OpConversion<tf_tensorlist::Stack, IREE::TensorList::StackTensor>>(
      context);
}

class ConvertTFTensorlistToTensorlistPass
    : public ConversionPass<ConvertTFTensorlistToTensorlistPass,
                            TensorListTypeConverter> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tf_tensorlist::TFTensorListDialect,
                    IREE::TensorList::TensorListDialect, StandardOpsDialect>();
  }

  void Setup(ConversionTarget &target,
             OwningRewritePatternList &patterns) override {
    target.addIllegalDialect<tf_tensorlist::TFTensorListDialect>();
    target.addLegalDialect<IREE::TensorList::TensorListDialect>();
    populateTFTensorListToTensorListPatterns(&this->getContext(), patterns);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFTensorListToTensorListPass() {
  return std::make_unique<ConvertTFTensorlistToTensorlistPass>();
}

static PassRegistration<ConvertTFTensorlistToTensorlistPass> pass(
    "iree-tf-tensorlist-convert-to-tensorlist",
    "Converts TF string ops to the IREE tf_strings dialect");

}  // namespace tf_tensorlist
}  // namespace iree_compiler
}  // namespace mlir
