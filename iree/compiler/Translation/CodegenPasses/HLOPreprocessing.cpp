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
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct HLOToLinalgPreprocessing
    : public FunctionPass<HLOToLinalgPreprocessing> {
  void runOnFunction() override {
    ConversionTarget conversionTarget(getContext());
    OwningRewritePatternList conversionPatterns;

    conversionTarget
        .addLegalDialect<xla_hlo::XlaHloDialect, StandardOpsDialect>();
    conversionTarget.addIllegalOp<xla_hlo::BatchNormInferenceOp>();

    xla_hlo::PopulateUnfuseBatchNormPatterns(&getContext(),
                                             &conversionPatterns);
    if (failed(applyPartialConversion(getFunction(), conversionTarget,
                                      conversionPatterns))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createHLOPreprocessingPass() {
  return std::make_unique<HLOToLinalgPreprocessing>();
}

static PassRegistration<HLOToLinalgPreprocessing> legalize_pass(
    "iree-hlo-to-linalg-preprocessing",
    "Apply hlo to hlo transformations for some hlo ops");

}  // namespace iree_compiler
}  // namespace mlir
