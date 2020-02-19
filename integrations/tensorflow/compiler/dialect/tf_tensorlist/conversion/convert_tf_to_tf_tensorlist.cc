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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace tf_tensorlist {

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_tf_to_tf_tensorlist.inc"

class ConvertTfToTfTensorList
    : public OperationPass<ConvertTfToTfTensorList, FuncOp> {
 public:
  void runOnOperation() override;
};

void ConvertTfToTfTensorList::runOnOperation() {
  auto func = getOperation();

  OwningRewritePatternList patterns;
  populateWithGenerated(&getContext(), &patterns);
  ConversionTarget target(getContext());
  target.addLegalDialect<TfTensorListDialect>();
  target.addIllegalOp<TF::TensorListReserveOp>();
  target.addIllegalOp<TF::TensorListGetItemOp>();
  target.addIllegalOp<TF::TensorListSetItemOp>();

  if (failed(applyPartialConversion(func, target, patterns))) {
    func.emitError() << "unable to lower to tf_tensorlist dialect";
    return signalPassFailure();
  }
}

static PassRegistration<ConvertTfToTfTensorList> pass(
    "convert-tf-to-tf_tensorlist", "Convert to more precise types");

std::unique_ptr<OpPassBase<FuncOp>> createConvertTfToTfTensorList() {
  return std::make_unique<ConvertTfToTfTensorList>();
}

}  // namespace tf_tensorlist
}  // namespace mlir
